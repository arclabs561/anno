//! Comprehensive Task-Dataset-Backend Evaluation System
//!
//! This module provides a unified evaluation framework that:
//! - Maps tasks to suitable datasets
//! - Maps datasets to compatible backends
//! - Runs evaluations across all valid combinations
//! - Generates comprehensive reports
//!
//! # Design Philosophy
//!
//! - **Trait-based**: Backend capabilities detected via trait implementations
//! - **Many-to-many**: Each task can use multiple datasets, each dataset can evaluate multiple tasks
//! - **Comprehensive**: Evaluates all valid task-dataset-backend combinations
//! - **Extensible**: Easy to add new tasks, datasets, or backends

use crate::eval::backend_factory::BackendFactory;
use crate::eval::loader::{DatasetId, DatasetLoader, LoadedDataset};
#[cfg(feature = "eval-profiling")]
use crate::eval::profiling;
use crate::eval::task_mapping::{
    dataset_tasks, get_task_backends, get_task_datasets, Task, TaskMapping,
};
use crate::{Entity, Model, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Configuration for task evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskEvalConfig {
    /// Which tasks to evaluate
    pub tasks: Vec<Task>,
    /// Which datasets to use (if empty, uses all suitable datasets for each task)
    pub datasets: Vec<DatasetId>,
    /// Which backends to test (if empty, uses all compatible backends)
    pub backends: Vec<String>,
    /// Maximum number of examples per dataset (for quick testing)
    pub max_examples: Option<usize>,
    /// Random seed for sampling (for reproducibility and varied testing)
    pub seed: Option<u64>,
    /// Whether to skip datasets that aren't cached
    pub require_cached: bool,
}

impl Default for TaskEvalConfig {
    fn default() -> Self {
        Self {
            tasks: Task::all().to_vec(),
            datasets: vec![],
            backends: vec![],
            max_examples: None,
            seed: Some(42),
            require_cached: false,
        }
    }
}

/// Results from evaluating a task-dataset-backend combination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskEvalResult {
    /// Task being evaluated
    pub task: Task,
    /// Dataset used
    pub dataset: DatasetId,
    /// Backend name
    pub backend: String,
    /// Whether evaluation succeeded
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Metrics (task-specific, stored as JSON-serializable map)
    pub metrics: HashMap<String, f64>,
    /// Number of examples evaluated
    pub num_examples: usize,
    /// Time taken in milliseconds (if available)
    pub duration_ms: Option<f64>,
}

impl TaskEvalResult {
    /// Check if this is a "skipped" result (feature not available) vs actual failure
    pub fn is_skipped(&self) -> bool {
        if self.success {
            return false;
        }
        if let Some(ref err) = self.error {
            err.contains("Feature not available") || err.contains("requires '")
        } else {
            false
        }
    }

    /// Get primary F1 metric for ranking
    pub fn primary_f1(&self) -> Option<f64> {
        self.metrics
            .get("f1")
            .or_else(|| self.metrics.get("conll_f1"))
            .or_else(|| self.metrics.get("strict_f1"))
            .copied()
    }
}

/// Comprehensive evaluation results across all combinations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveEvalResults {
    /// Individual evaluation results
    pub results: Vec<TaskEvalResult>,
    /// Summary statistics
    pub summary: EvalSummary,
}

/// Summary statistics for comprehensive evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalSummary {
    /// Total combinations evaluated
    pub total_combinations: usize,
    /// Successful evaluations
    pub successful: usize,
    /// Failed evaluations (actual errors, not skipped)
    pub failed: usize,
    /// Skipped evaluations (feature not available, etc.)
    pub skipped: usize,
    /// Tasks evaluated
    pub tasks: Vec<Task>,
    /// Datasets used
    pub datasets: Vec<DatasetId>,
    /// Backends tested
    pub backends: Vec<String>,
}

/// Evaluator for task-dataset-backend combinations.
pub struct TaskEvaluator {
    loader: DatasetLoader,
    #[allow(dead_code)] // Reserved for future use
    mapping: TaskMapping,
}

impl TaskEvaluator {
    /// Create a new task evaluator.
    pub fn new() -> Result<Self> {
        Ok(Self {
            loader: DatasetLoader::new()?,
            mapping: TaskMapping::build(),
        })
    }

    /// Run comprehensive evaluation across all valid combinations.
    pub fn evaluate_all(&self, config: TaskEvalConfig) -> Result<ComprehensiveEvalResults> {
        let mut results = Vec::new();
        let mut tasks_evaluated = Vec::new();
        let mut datasets_used = Vec::new();
        let mut backends_tested = Vec::new();

        // Determine which tasks to evaluate
        let tasks = if config.tasks.is_empty() {
            Task::all().to_vec()
        } else {
            config.tasks.clone()
        };

        for task in &tasks {
            tasks_evaluated.push(*task);

            // Get suitable datasets for this task
            let datasets = if config.datasets.is_empty() {
                get_task_datasets(*task)
            } else {
                // Filter to datasets that support this task
                config
                    .datasets
                    .iter()
                    .filter(|d| dataset_tasks(**d).contains(task))
                    .copied()
                    .collect()
            };

            for dataset in &datasets {
                if !datasets_used.contains(dataset) {
                    datasets_used.push(*dataset);
                }

                // Check if dataset is cached (if required)
                if config.require_cached && !self.loader.is_cached(*dataset) {
                    continue;
                }

                // Get compatible backends for this task
                let backends = if config.backends.is_empty() {
                    get_task_backends(*task)
                        .iter()
                        .map(|s| s.to_string())
                        .collect()
                } else {
                    config.backends.clone()
                };

                for backend_name in &backends {
                    if !backends_tested.contains(backend_name) {
                        backends_tested.push(backend_name.clone());
                    }

                    // Evaluate this combination
                    let result =
                        self.evaluate_combination(*task, *dataset, backend_name, &config)?;
                    results.push(result);
                }
            }
        }

        let skipped = results.iter().filter(|r| r.is_skipped()).count();
        let failed = results.iter().filter(|r| !r.success && !r.is_skipped()).count();
        let summary = EvalSummary {
            total_combinations: results.len(),
            successful: results.iter().filter(|r| r.success).count(),
            failed,
            skipped,
            tasks: tasks_evaluated,
            datasets: datasets_used,
            backends: backends_tested,
        };

        #[cfg(feature = "eval-profiling")]
        profiling::print_summary();

        Ok(ComprehensiveEvalResults { results, summary })
    }

    /// Evaluate a single task-dataset-backend combination.
    fn evaluate_combination(
        &self,
        task: Task,
        dataset: DatasetId,
        backend_name: &str,
        config: &TaskEvalConfig,
    ) -> Result<TaskEvalResult> {
        // Load dataset
        let dataset_data = {
            #[cfg(feature = "eval-advanced")]
            {
                match self.loader.load_or_download(dataset) {
                    Ok(data) => data,
                    Err(e) => {
                        return Ok(TaskEvalResult {
                            task,
                            dataset,
                            backend: backend_name.to_string(),
                            success: false,
                            error: Some(format!("Failed to load dataset: {}", e)),
                            metrics: HashMap::new(),
                            num_examples: 0,
                            duration_ms: None,
                        });
                    }
                }
            }
            #[cfg(not(feature = "eval-advanced"))]
            {
                // Without eval-advanced, only load from cache
                match self.loader.load(dataset) {
                    Ok(data) => data,
                    Err(e) => {
                        return Ok(TaskEvalResult {
                            task,
                            dataset,
                            backend: backend_name.to_string(),
                            success: false,
                            error: Some(format!("Failed to load dataset (not cached, eval-advanced feature required for download): {}", e)),
                            metrics: HashMap::new(),
                            num_examples: 0,
                            duration_ms: None,
                        });
                    }
                }
            }
        };

        // Sample sentences if configured (with seed for reproducibility)
        let total = dataset_data.sentences.len();
        let (sampled_data, sentences_to_use) = if let Some(max) = config.max_examples {
            if max >= total {
                (dataset_data, total)
            } else {
                // Simple deterministic shuffle based on seed (works for all features)
                let seed = config.seed.unwrap_or(42);
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut indices: Vec<(usize, u64)> = (0..total)
                    .map(|i| {
                        let mut hasher = DefaultHasher::new();
                        seed.hash(&mut hasher);
                        i.hash(&mut hasher);
                        (i, hasher.finish())
                    })
                    .collect();
                indices.sort_by_key(|(_, hash)| *hash);
                let selected_indices: Vec<usize> = indices.iter()
                    .take(max)
                    .map(|(i, _)| *i)
                    .collect();
                let sampled_sentences: Vec<_> = selected_indices.iter()
                    .filter_map(|&i| dataset_data.sentences.get(i).cloned())
                    .collect();
                use crate::eval::loader::LoadedDataset;
                let sampled_dataset = LoadedDataset {
                    id: dataset_data.id,
                    sentences: sampled_sentences,
                    loaded_at: dataset_data.loaded_at.clone(),
                    source_url: dataset_data.source_url.clone(),
                };
                (sampled_dataset, max)
            }
        } else {
            (dataset_data, total)
        };

        // Try to create backend (this is a placeholder - actual implementation
        // would need backend factory)
        let start = Instant::now();
        let result = match self.try_evaluate_backend(task, dataset, backend_name, &sampled_data) {
            Ok(metrics) => {
                let duration = start.elapsed().as_secs_f64() * 1000.0;
                TaskEvalResult {
                    task,
                    dataset,
                    backend: backend_name.to_string(),
                    success: true,
                    error: None,
                    metrics,
                    num_examples: sentences_to_use,
                    duration_ms: Some(duration),
                }
            }
            Err(e) => {
                let duration = start.elapsed().as_secs_f64() * 1000.0;
                TaskEvalResult {
                    task,
                    dataset,
                    backend: backend_name.to_string(),
                    success: false,
                    error: Some(format!("{}", e)),
                    metrics: HashMap::new(),
                    num_examples: sentences_to_use,
                    duration_ms: Some(duration),
                }
            }
        };

        Ok(result)
    }

    /// Evaluate a backend on a task with actual inference and metrics.
    ///
    /// This implementation:
    /// 1. Creates backend instance via `BackendFactory`
    /// 2. Runs inference on dataset examples
    /// 3. Computes task-specific metrics (P/R/F1 for NER, MUC/B³/CEAF for coref, etc.)
    /// 4. Returns metrics as a map
    fn try_evaluate_backend(
        &self,
        task: Task,
        dataset: DatasetId,
        backend_name: &str,
        dataset_data: &LoadedDataset,
    ) -> Result<HashMap<String, f64>> {
        // Validate task-dataset compatibility
        let dataset_tasks = dataset_tasks(dataset);
        if !dataset_tasks.contains(&task) {
            return Err(crate::Error::InvalidInput(format!(
                "Dataset {:?} does not support task {:?}",
                dataset, task
            )));
        }

        // Validate task-backend compatibility
        let backend_tasks: Vec<String> = get_task_backends(task)
            .iter()
            .map(|s| s.to_string())
            .collect();
        if !backend_tasks.contains(&backend_name.to_string()) {
            return Err(crate::Error::InvalidInput(format!(
                "Backend '{}' does not support task {:?}",
                backend_name, task
            )));
        }

        // Create backend instance
        let backend = BackendFactory::create(backend_name)?;

        // Run task-specific evaluation
        match task {
            Task::NER | Task::DiscontinuousNER => {
                self.evaluate_ner_task(backend_name, &*backend, dataset, dataset_data)
            }
            Task::IntraDocCoref | Task::AbstractAnaphora => {
                self.evaluate_coref_task(backend_name, dataset_data)
            }
            Task::RelationExtraction => self.evaluate_relation_task(backend_name, dataset_data),
            _ => {
                // Placeholder for other tasks
                let mut metrics = HashMap::new();
                metrics.insert("validation_passed".to_string(), 1.0);
                metrics.insert(
                    "num_examples".to_string(),
                    dataset_data.sentences.len() as f64,
                );
                Ok(metrics)
            }
        }
    }

    /// Evaluate NER task with actual inference.
    fn evaluate_ner_task(
        &self,
        backend_name: &str,
        backend: &dyn Model,
        dataset: DatasetId,
        dataset_data: &LoadedDataset,
    ) -> Result<HashMap<String, f64>> {
        use crate::eval::ner_metrics::evaluate_entities;

        #[cfg(feature = "eval-profiling")]
        profiling::start("evaluate_ner_task");

        // Pre-allocate vectors with estimated capacity to reduce reallocations
        let estimated_entities = dataset_data.sentences.len() * 3; // Rough estimate: ~3 entities per sentence
        let mut all_gold = Vec::with_capacity(estimated_entities);
        let mut all_predicted = Vec::with_capacity(estimated_entities);
        let mut total_chars = 0;
        let start_time = Instant::now();

        // Extract dataset entity types and map to model-compatible labels
        let dataset_labels = dataset.entity_types();
        let mapped_labels = Self::map_dataset_labels_to_model(dataset_labels, backend_name);

        // Check if this is a zero-shot backend that needs custom labels
        let is_zero_shot = matches!(
            backend_name.to_lowercase().as_str(),
            "nuner" | "gliner_onnx" | "gliner_candle" | "gliner2"
        );

        // Process sentences (parallel if rayon is available, sequential otherwise)
        let total_sentences = dataset_data.sentences.len();

        #[cfg(feature = "eval-parallel")]
        {
            use rayon::prelude::*;
            use std::cell::RefCell;
            use std::sync::atomic::{AtomicUsize, Ordering};
            use std::sync::Arc;
            use std::sync::Mutex;

            // For parallel processing, use thread-local storage to cache backends per thread
            // This avoids the need to share state across threads while still caching per thread
            thread_local! {
                static THREAD_CACHED_BACKEND: RefCell<Option<(String, Box<dyn std::any::Any>)>> = RefCell::new(None);
            }

            let backend_name_arc = Arc::new(backend_name.to_string());
            let mapped_labels_arc = Arc::new(mapped_labels.clone());
            let is_zero_shot_flag = is_zero_shot;

            let progress_counter = AtomicUsize::new(0);
            let last_progress_percent = Arc::new(Mutex::new(0));
            let start_time_arc = Arc::new(Mutex::new(start_time));

            let all_results: Vec<_> = dataset_data.sentences
                .par_iter()
                .enumerate()
                .map(|(_idx, sentence)| {
                    let text = sentence.text();
                    let chars_count = text.chars().count();

                    // Extract gold entities
                    let gold_entities: Vec<Entity> = sentence.entities().iter().map(|g| {
                        let mut entity = Entity::new(
                            g.text.clone(),
                            g.entity_type.clone(),
                            g.start,
                            g.end,
                            1.0,
                        );
                        entity.provenance = Some(crate::Provenance::ml("gold", 1.0));
                        entity
                    }).collect();

                    // Run inference - use thread-local cached backend for zero-shot models
                    let entities_result = if is_zero_shot_flag && !mapped_labels_arc.is_empty() {
                        THREAD_CACHED_BACKEND.with(|cache| {
                            let mut cached = cache.borrow_mut();
                            // Check if we have a cached backend for this backend_name
                            if let Some((ref cached_name, ref backend)) = *cached {
                                if cached_name == backend_name_arc.as_str() {
                                    // Use cached backend
                                    return Self::extract_with_cached_backend(
                                        backend_name_arc.as_str(),
                                        backend.as_ref(),
                                        &text,
                                        &mapped_labels_arc
                                    );
                                }
                            }
                            // Create and cache new backend for this thread
                            match Self::create_zero_shot_backend(backend_name_arc.as_str()) {
                                Ok(new_backend) => {
                                    let result = Self::extract_with_cached_backend(
                                        backend_name_arc.as_str(),
                                        new_backend.as_ref(),
                                        &text,
                                        &mapped_labels_arc
                                    );
                                    *cached = Some((backend_name_arc.to_string(), new_backend));
                                    result
                                }
                                Err(e) => Err(e),
                            }
                        })
                    } else {
                        backend.extract_entities(&text, None)
                    };

                    // Update progress with time estimates
                    let processed = progress_counter.fetch_add(1, Ordering::Relaxed) + 1;
                    let current_percent = (processed * 100) / total_sentences;
                    let mut last_percent = last_progress_percent.lock().unwrap();
                    if current_percent >= *last_percent + 10 || processed % 10 == 0 {
                        let elapsed = start_time_arc.lock().unwrap().elapsed();
                        let elapsed_secs = elapsed.as_secs_f64();
                        let rate = if elapsed_secs > 0.0 {
                            processed as f64 / elapsed_secs
                        } else {
                            0.0
                        };
                        let remaining = if rate > 0.0 {
                            ((total_sentences - processed) as f64 / rate) as u64
                        } else {
                            0
                        };
                        let remaining_str = if remaining > 0 {
                            format!(" (~{}s remaining)", remaining)
                        } else {
                            String::new()
                        };
                        eprint!("\rProcessing: {}/{} sentences ({:.0}%) for backend '{}' on dataset '{}'{}\x1b[K",
                            processed, total_sentences, current_percent, backend_name, dataset.to_string(), remaining_str);
                        *last_percent = current_percent;
                    }

                    (chars_count, gold_entities, entities_result)
                })
                .collect();

            // Final progress update with timing
            let total_elapsed = start_time.elapsed();
            let total_secs = total_elapsed.as_secs_f64();
            let rate = if total_secs > 0.0 {
                total_sentences as f64 / total_secs
            } else {
                0.0
            };
            eprint!("\rProcessing: {}/{} sentences (100.0%) for backend '{}' on dataset '{}' (completed in {:.1}s, {:.1} sentences/s)\x1b[K",
                total_sentences, total_sentences, backend_name, dataset.to_string(), total_secs, rate);
            eprintln!(); // Newline after progress

            // Aggregate results
            for (chars_count, gold_entities, entities_result) in all_results {
                total_chars += chars_count;
                all_gold.extend(gold_entities);

                match entities_result {
                    Ok(entities) => {
                        all_predicted.extend(entities);
                    }
                    Err(e) => {
                        eprintln!("\nWarning: Backend inference failed: {}", e);
                    }
                }
            }
        }

        #[cfg(not(feature = "eval-parallel"))]
        {
            // For zero-shot backends, create a cached instance once to avoid recreating for each sentence
            let zero_shot_backend: Option<Box<dyn std::any::Any>> =
                if is_zero_shot && !mapped_labels.is_empty() {
                    Some(Self::create_zero_shot_backend(backend_name)?)
                } else {
                    None
                };

            // Sequential processing (fallback when rayon not available)
            for (idx, sentence) in dataset_data.sentences.iter().enumerate() {
                // Progress reporting every 10% or every 10 sentences, whichever is more frequent
                if idx % 10 == 0 || idx == total_sentences - 1 {
                    let progress = ((idx + 1) as f64 / total_sentences as f64) * 100.0;
                    let elapsed = start_time.elapsed();
                    let elapsed_secs = elapsed.as_secs_f64();
                    let rate = if elapsed_secs > 0.0 {
                        (idx + 1) as f64 / elapsed_secs
                    } else {
                        0.0
                    };
                    let remaining = if rate > 0.0 {
                        ((total_sentences - idx - 1) as f64 / rate) as u64
                    } else {
                        0
                    };
                    let remaining_str = if remaining > 0 {
                        format!(" (~{}s remaining)", remaining)
                    } else {
                        String::new()
                    };
                    eprint!("\rProcessing: {}/{} sentences ({:.1}%) for backend '{}' on dataset '{}'{}\x1b[K",
                        idx + 1, total_sentences, progress, backend_name, dataset.to_string(), remaining_str);
                }

                let text = sentence.text();
                total_chars += text.chars().count();

                #[cfg(feature = "eval-profiling")]
                profiling::start("extract_gold_entities");
                // Extract gold entities from sentence
                let gold_entities = sentence.entities();
                all_gold.extend(gold_entities.iter().map(|g| {
                    let mut entity =
                        Entity::new(g.text.clone(), g.entity_type.clone(), g.start, g.end, 1.0);
                    entity.provenance = Some(crate::Provenance::ml("gold", 1.0));
                    entity
                }));
                #[cfg(feature = "eval-profiling")]
                profiling::stop("extract_gold_entities");

                #[cfg(feature = "eval-profiling")]
                profiling::start("backend_inference");
                // Run inference - use extract() for zero-shot models, extract_entities() for others
                let entities = if let Some(ref cached) = zero_shot_backend {
                    Self::extract_with_cached_backend(backend_name, cached, &text, &mapped_labels)
                } else {
                    backend.extract_entities(&text, None)
                };
                #[cfg(feature = "eval-profiling")]
                profiling::stop("backend_inference");

                match entities {
                    Ok(entities) => {
                        all_predicted.extend(entities);
                    }
                    Err(e) => {
                        // Log error but continue with other sentences
                        eprintln!(
                            "\nWarning: Backend inference failed for sentence {}: {}",
                            idx + 1,
                            e
                        );
                    }
                }
            }

            // Final progress update with timing
            let total_elapsed = start_time.elapsed();
            let total_secs = total_elapsed.as_secs_f64();
            let rate = if total_secs > 0.0 {
                total_sentences as f64 / total_secs
            } else {
                0.0
            };
            eprint!("\rProcessing: {}/{} sentences (100.0%) for backend '{}' on dataset '{}' (completed in {:.1}s, {:.1} sentences/s)\x1b[K",
                total_sentences, total_sentences, backend_name, dataset.to_string(), total_secs, rate);
            eprintln!(); // Newline after progress
        }

        #[cfg(feature = "eval-profiling")]
        profiling::stop("evaluate_ner_task");

        #[cfg(feature = "eval-profiling")]
        profiling::start("compute_metrics");

        let elapsed = start_time.elapsed();
        let chars_per_second = if elapsed.as_secs_f64() > 0.0 {
            total_chars as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        // Compute metrics
        let eval_results = evaluate_entities(&all_gold, &all_predicted);

        #[cfg(feature = "eval-profiling")]
        profiling::stop("compute_metrics");
        let summary = eval_results.summary();

        // Build metrics map
        let mut metrics = HashMap::new();
        metrics.insert("precision".to_string(), summary.strict_precision);
        metrics.insert("recall".to_string(), summary.strict_recall);
        metrics.insert("f1".to_string(), summary.strict_f1);
        metrics.insert("exact_precision".to_string(), summary.exact_precision);
        metrics.insert("exact_recall".to_string(), summary.exact_recall);
        metrics.insert("exact_f1".to_string(), summary.exact_f1);
        metrics.insert("partial_precision".to_string(), summary.partial_precision);
        metrics.insert("partial_recall".to_string(), summary.partial_recall);
        metrics.insert("partial_f1".to_string(), summary.partial_f1);
        metrics.insert("type_precision".to_string(), summary.type_precision);
        metrics.insert("type_recall".to_string(), summary.type_recall);
        metrics.insert("type_f1".to_string(), summary.type_f1);
        metrics.insert("chars_per_second".to_string(), chars_per_second);
        metrics.insert("num_gold".to_string(), all_gold.len() as f64);
        metrics.insert("num_predicted".to_string(), all_predicted.len() as f64);

        Ok(metrics)
    }

    /// Map dataset entity type labels to model-compatible labels.
    ///
    /// Handles common label variations (e.g., "PER" → "person", "PERSON" → "person").
    /// Public for testing purposes.
    pub(crate) fn map_dataset_labels_to_model(
        dataset_labels: &[&str],
        backend_name: &str,
    ) -> Vec<String> {
        dataset_labels
            .iter()
            .map(|label| {
                // Normalize label to lowercase for matching
                let normalized = label.to_lowercase();
                match normalized.as_str() {
                    // Person variations
                    "per" | "person" => "person".to_string(),
                    // Organization variations
                    "org" | "organization" | "organisation" | "corporation" | "company" => {
                        "organization".to_string()
                    }
                    // Location variations
                    "loc" | "location" | "place" | "gpe" => "location".to_string(),
                    // Other common types
                    "misc" | "miscellaneous" => "misc".to_string(),
                    "date" => "date".to_string(),
                    "time" => "time".to_string(),
                    "money" | "currency" => "money".to_string(),
                    "percent" | "percentage" => "percent".to_string(),
                    "product" => "product".to_string(),
                    "event" => "event".to_string(),
                    "facility" | "fac" => "facility".to_string(),
                    "work_of_art" | "workofart" => "work_of_art".to_string(),
                    "law" => "law".to_string(),
                    "language" => "language".to_string(),
                    "norp" => "norp".to_string(),
                    // For NuNER, keep original if not mapped (it's zero-shot)
                    _ if backend_name == "nuner" => label.to_lowercase(),
                    // For other backends, try to map or use original
                    _ => label.to_lowercase(),
                }
            })
            .collect()
    }

    /// Create a zero-shot backend instance and cache it (wrapped in Any for type erasure).
    ///
    /// This avoids recreating the model for every sentence, which causes ONNX errors.
    fn create_zero_shot_backend(backend_name: &str) -> Result<Box<dyn std::any::Any>> {
        match backend_name.to_lowercase().as_str() {
            "nuner" => {
                #[cfg(feature = "onnx")]
                {
                    use crate::backends::nuner::NuNER;
                    use crate::DEFAULT_NUNER_MODEL;
                    let nuner = NuNER::from_pretrained(DEFAULT_NUNER_MODEL)?;
                    Ok(Box::new(nuner))
                }
                #[cfg(not(feature = "onnx"))]
                {
                    Err(crate::Error::FeatureNotAvailable(
                        "NuNER requires the 'onnx' feature".to_string(),
                    ))
                }
            }
            "gliner_onnx" | "gliner" => {
                #[cfg(feature = "onnx")]
                {
                    use crate::backends::gliner_onnx::GLiNEROnnx;
                    use crate::DEFAULT_GLINER_MODEL;
                    let gliner = GLiNEROnnx::new(DEFAULT_GLINER_MODEL)?;
                    Ok(Box::new(gliner))
                }
                #[cfg(not(feature = "onnx"))]
                {
                    Err(crate::Error::FeatureNotAvailable(
                        "GLiNER requires the 'onnx' feature".to_string(),
                    ))
                }
            }
            "gliner2" => {
                #[cfg(feature = "onnx")]
                {
                    use crate::backends::gliner2::GLiNER2Onnx;
                    use crate::DEFAULT_GLINER2_MODEL;
                    let gliner2 = GLiNER2Onnx::from_pretrained(DEFAULT_GLINER2_MODEL)?;
                    Ok(Box::new(gliner2))
                }
                #[cfg(not(feature = "onnx"))]
                {
                    Err(crate::Error::FeatureNotAvailable(
                        "GLiNER2 requires the 'onnx' feature".to_string(),
                    ))
                }
            }
            "gliner_candle" => {
                #[cfg(feature = "candle")]
                {
                    use crate::backends::gliner_candle::GLiNERCandle;
                    use crate::DEFAULT_GLINER_MODEL;
                    let gliner = GLiNERCandle::from_pretrained(DEFAULT_GLINER_MODEL)?;
                    Ok(Box::new(gliner))
                }
                #[cfg(not(feature = "candle"))]
                {
                    Err(crate::Error::FeatureNotAvailable(
                        "GLiNER Candle requires the 'candle' feature".to_string(),
                    ))
                }
            }
            _ => Err(crate::Error::InvalidInput(format!(
                "Unknown zero-shot backend: {}",
                backend_name
            ))),
        }
    }

    /// Extract entities using cached zero-shot backend instance.
    #[allow(unused_variables)] // False positives - variables are used in feature-gated code
    fn extract_with_cached_backend(
        backend_name: &str,
        cached: &dyn std::any::Any,
        text: &str,
        labels: &[String],
    ) -> Result<Vec<Entity>> {
        // Convert labels to &str slice
        let label_strs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

        match backend_name.to_lowercase().as_str() {
            "nuner" => {
                #[cfg(feature = "onnx")]
                {
                    if let Some(nuner) = cached.downcast_ref::<crate::backends::nuner::NuNER>() {
                        nuner.extract(text, &label_strs, 0.5)
                    } else {
                        Err(crate::Error::InvalidInput(
                            "Failed to downcast cached NuNER backend".to_string(),
                        ))
                    }
                }
                #[cfg(not(feature = "onnx"))]
                {
                    Err(crate::Error::FeatureNotAvailable(
                        "NuNER requires the 'onnx' feature".to_string(),
                    ))
                }
            }
            "gliner_onnx" | "gliner" => {
                #[cfg(feature = "onnx")]
                {
                    if let Some(gliner) =
                        cached.downcast_ref::<crate::backends::gliner_onnx::GLiNEROnnx>()
                    {
                        gliner.extract(text, &label_strs, 0.5)
                    } else {
                        Err(crate::Error::InvalidInput(
                            "Failed to downcast cached GLiNER backend".to_string(),
                        ))
                    }
                }
                #[cfg(not(feature = "onnx"))]
                {
                    Err(crate::Error::FeatureNotAvailable(
                        "GLiNER requires the 'onnx' feature".to_string(),
                    ))
                }
            }
            "gliner2" => {
                #[cfg(feature = "onnx")]
                {
                    use crate::backends::gliner2::TaskSchema;
                    if let Some(gliner2) =
                        cached.downcast_ref::<crate::backends::gliner2::GLiNER2Onnx>()
                    {
                        let schema = TaskSchema::new().with_entities(&label_strs);
                        let result = gliner2.extract(text, &schema)?;
                        Ok(result.entities)
                    } else {
                        Err(crate::Error::InvalidInput(
                            "Failed to downcast cached GLiNER2 backend".to_string(),
                        ))
                    }
                }
                #[cfg(not(feature = "onnx"))]
                {
                    Err(crate::Error::FeatureNotAvailable(
                        "GLiNER2 requires the 'onnx' feature".to_string(),
                    ))
                }
            }
            "gliner_candle" => {
                #[cfg(feature = "candle")]
                {
                    if let Some(gliner) =
                        cached.downcast_ref::<crate::backends::gliner_candle::GLiNERCandle>()
                    {
                        gliner.extract(text, &label_strs, 0.5)
                    } else {
                        Err(crate::Error::InvalidInput(
                            "Failed to downcast cached GLiNER Candle backend".to_string(),
                        ))
                    }
                }
                #[cfg(not(feature = "candle"))]
                {
                    Err(crate::Error::FeatureNotAvailable(
                        "GLiNER Candle requires the 'candle' feature".to_string(),
                    ))
                }
            }
            _ => Err(crate::Error::InvalidInput(format!(
                "Unknown zero-shot backend: {}",
                backend_name
            ))),
        }
    }

    /// Evaluate coreference task.
    fn evaluate_coref_task(
        &self,
        backend_name: &str,
        dataset_data: &LoadedDataset,
    ) -> Result<HashMap<String, f64>> {
        use crate::eval::coref::{entities_to_chains, CorefDocument};
        use crate::eval::coref_metrics::CorefEvaluation;
        use crate::eval::coref_resolver::SimpleCorefResolver;

        // Try to load coreference documents if dataset supports it
        let gold_docs = if dataset_data.id.is_coreference() {
            match self.loader.load_coref(dataset_data.id) {
                Ok(docs) => docs,
                Err(_) => {
                    // Fallback: convert entities to chains if they have canonical_id
                    let mut all_gold_entities = Vec::new();
                    for sentence in &dataset_data.sentences {
                        for entity in sentence.entities() {
                            let e = crate::Entity::new(
                                entity.text.clone(),
                                entity.entity_type.clone(),
                                entity.start,
                                entity.end,
                                1.0,
                            );
                            // If entity has a coreference ID, use it as canonical_id
                            // (This is a placeholder - actual implementation would need
                            // to extract coref IDs from the dataset format)
                            all_gold_entities.push(e);
                        }
                    }
                    // Group entities by sentence for now (simplified)
                    vec![CorefDocument::new(
                        "",
                        entities_to_chains(&all_gold_entities),
                    )]
                }
            }
        } else {
            // Not a coreference dataset - return placeholder
            let mut metrics = HashMap::new();
            metrics.insert(
                "num_sentences".to_string(),
                dataset_data.sentences.len() as f64,
            );
            metrics.insert("error".to_string(), 1.0);
            return Ok(metrics);
        };

        // Run coreference resolver on each document
        let resolver = SimpleCorefResolver::default();
        let mut all_predicted_chains = Vec::new();
        let mut all_gold_chains = Vec::new();

        for doc in &gold_docs {
            // Collect gold chains from the document
            all_gold_chains.extend(doc.chains.clone());

            // Extract entities from the document text using the backend
            let backend = BackendFactory::create(backend_name)?;
            match backend.extract_entities(&doc.text, None) {
                Ok(entities) => {
                    // Resolve coreference on predicted entities
                    let resolved_entities = resolver.resolve(&entities);
                    // Convert resolved entities to chains
                    let predicted_chains = entities_to_chains(&resolved_entities);
                    all_predicted_chains.extend(predicted_chains);
                }
                Err(e) => {
                    // Log error but continue with other documents
                    eprintln!("Warning: Backend inference failed for document: {}", e);
                }
            }
        }

        // Compute coreference metrics
        let eval = CorefEvaluation::compute(&all_predicted_chains, &all_gold_chains);

        let mut metrics = HashMap::new();
        metrics.insert("muc_precision".to_string(), eval.muc.precision);
        metrics.insert("muc_recall".to_string(), eval.muc.recall);
        metrics.insert("muc_f1".to_string(), eval.muc.f1);
        metrics.insert("b3_precision".to_string(), eval.b_cubed.precision);
        metrics.insert("b3_recall".to_string(), eval.b_cubed.recall);
        metrics.insert("b3_f1".to_string(), eval.b_cubed.f1);
        metrics.insert("ceaf_e_precision".to_string(), eval.ceaf_e.precision);
        metrics.insert("ceaf_e_recall".to_string(), eval.ceaf_e.recall);
        metrics.insert("ceaf_e_f1".to_string(), eval.ceaf_e.f1);
        metrics.insert("ceaf_m_precision".to_string(), eval.ceaf_m.precision);
        metrics.insert("ceaf_m_recall".to_string(), eval.ceaf_m.recall);
        metrics.insert("ceaf_m_f1".to_string(), eval.ceaf_m.f1);
        metrics.insert("lea_precision".to_string(), eval.lea.precision);
        metrics.insert("lea_recall".to_string(), eval.lea.recall);
        metrics.insert("lea_f1".to_string(), eval.lea.f1);
        metrics.insert("blanc_precision".to_string(), eval.blanc.precision);
        metrics.insert("blanc_recall".to_string(), eval.blanc.recall);
        metrics.insert("blanc_f1".to_string(), eval.blanc.f1);
        metrics.insert("conll_f1".to_string(), eval.conll_f1);
        metrics.insert("num_documents".to_string(), gold_docs.len() as f64);
        metrics.insert("num_gold_chains".to_string(), all_gold_chains.len() as f64);
        metrics.insert(
            "num_predicted_chains".to_string(),
            all_predicted_chains.len() as f64,
        );

        Ok(metrics)
    }

    /// Evaluate relation extraction task.
    ///
    /// # Limitations
    ///
    /// Currently, relation extraction evaluation is not fully implemented because:
    /// 1. Most datasets in `LoadedDataset` format don't include relation annotations
    /// 2. Relation extraction requires specialized dataset formats (DocRED JSONL, TACRED, etc.)
    /// 3. Backends need to implement relation extraction methods (not just entity extraction)
    ///
    /// This function returns metrics computed from empty relation lists, which will
    /// show 0.0 precision/recall/F1. To properly evaluate relations:
    /// - Add relation annotation parsing to `DatasetLoader`
    /// - Implement relation extraction in backends (e.g., GLiNER2 supports this)
    /// - Load relation-specific datasets (DocRED, TACRED, Re-TACRED)
    fn evaluate_relation_task(
        &self,
        backend_name: &str,
        dataset_data: &LoadedDataset,
    ) -> Result<HashMap<String, f64>> {
        use crate::eval::relation::{evaluate_relations, RelationEvalConfig};

        // Extract gold relations from dataset
        // NOTE: Current dataset format doesn't include relation annotations.
        // This requires dataset-specific parsing (DocRED JSONL, TACRED, etc.)
        let all_gold_relations = Vec::new();
        let all_predicted_relations = Vec::new();

        // Extract entities for relation extraction (even if we can't extract relations yet)
        let backend = BackendFactory::create(backend_name)?;
        for sentence in &dataset_data.sentences {
            let text = sentence.text();
            match backend.extract_entities(&text, None) {
                Ok(_predicted_entities) => {
                    // Future: Extract relations from backend output
                    // This requires backends to implement relation extraction methods
                    // (e.g., GLiNER2 has extract_relations() but it's not in the Model trait)
                }
                Err(e) => {
                    eprintln!("Warning: Backend inference failed: {}", e);
                }
            }
        }

        // Evaluate relations
        let config = RelationEvalConfig::default();
        let metrics_result =
            evaluate_relations(&all_gold_relations, &all_predicted_relations, &config);

        let mut metrics = HashMap::new();
        metrics.insert(
            "boundary_precision".to_string(),
            metrics_result.boundary_precision,
        );
        metrics.insert(
            "boundary_recall".to_string(),
            metrics_result.boundary_recall,
        );
        metrics.insert("boundary_f1".to_string(), metrics_result.boundary_f1);
        metrics.insert(
            "strict_precision".to_string(),
            metrics_result.strict_precision,
        );
        metrics.insert("strict_recall".to_string(), metrics_result.strict_recall);
        metrics.insert("strict_f1".to_string(), metrics_result.strict_f1);
        metrics.insert(
            "num_gold_relations".to_string(),
            all_gold_relations.len() as f64,
        );
        metrics.insert(
            "num_predicted_relations".to_string(),
            all_predicted_relations.len() as f64,
        );
        metrics.insert(
            "num_sentences".to_string(),
            dataset_data.sentences.len() as f64,
        );

        Ok(metrics)
    }
}

impl Default for TaskEvaluator {
    fn default() -> Self {
        Self::new().expect("Failed to create TaskEvaluator")
    }
}

/// Generate a markdown report from evaluation results.
impl ComprehensiveEvalResults {
    /// Convert evaluation results to a markdown-formatted report.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str("# Eval Report\n\n");
        
        // Dense summary line
        let avg_examples: f64 = self.results.iter()
            .filter(|r| r.success)
            .map(|r| r.num_examples as f64)
            .sum::<f64>() / self.summary.successful.max(1) as f64;
        let avg_time: f64 = self.results.iter()
            .filter_map(|r| r.duration_ms)
            .sum::<f64>() / self.results.iter().filter(|r| r.duration_ms.is_some()).count().max(1) as f64;
        
        md.push_str(&format!(
            "Total: {} | ✓: {} | ⊘: {} | ✗: {} | Avg examples: {:.0} | Avg time: {:.0}ms\n\n",
            self.summary.total_combinations,
            self.summary.successful,
            self.summary.skipped,
            self.summary.failed,
            avg_examples,
            avg_time
        ));
        
        // Failures first (most important for debugging)
        let failures: Vec<_> = self.results.iter()
            .filter(|r| !r.success && !r.is_skipped())
            .collect();
        
        if !failures.is_empty() {
            md.push_str("## Failures\n\n");
            md.push_str("| Task | Dataset | Backend | Error |\n");
            md.push_str("|------|---------|---------|-------|\n");
            for result in &failures {
                let error = result.error.as_ref()
                    .map(|e| e.replace('|', "\\|").replace('\n', " "))
                    .unwrap_or_else(|| "N/A".to_string());
                md.push_str(&format!(
                    "| {} | {:?} | {} | {} |\n",
                    result.task.name(), result.dataset, result.backend, error
                ));
            }
            md.push_str("\n");
        }
        
        // Error patterns
        let mut error_patterns: HashMap<String, usize> = HashMap::new();
        for result in failures.iter() {
            if let Some(ref err) = result.error {
                // Extract error pattern (first 50 chars or key phrase)
                let pattern = if err.len() > 50 {
                    err.chars().take(50).collect::<String>() + "..."
                } else {
                    err.clone()
                };
                *error_patterns.entry(pattern).or_insert(0) += 1;
            }
        }
        
        if !error_patterns.is_empty() {
            md.push_str("## Error Patterns\n\n");
            let mut patterns: Vec<_> = error_patterns.iter().collect();
            patterns.sort_by(|a, b| b.1.cmp(a.1));
            for (pattern, count) in patterns {
                md.push_str(&format!("- [{}x] {}\n", count, pattern));
            }
            md.push_str("\n");
        }

        md.push_str("## Results\n\n");
        // Group results by task
        let mut by_task: HashMap<Task, Vec<&TaskEvalResult>> = HashMap::new();
        for result in &self.results {
            by_task.entry(result.task).or_default().push(result);
        }

        for (task, mut results) in by_task {
            md.push_str(&format!("### {}\n\n", task.name()));
            
            // Sort results: successful first (by F1 descending), then skipped, then failed
            results.sort_by(|a, b| {
                match (a.success, b.success) {
                    (true, true) => {
                        let a_f1 = a.primary_f1().unwrap_or(0.0);
                        let b_f1 = b.primary_f1().unwrap_or(0.0);
                        b_f1.partial_cmp(&a_f1).unwrap_or(std::cmp::Ordering::Equal)
                    }
                    (true, false) => std::cmp::Ordering::Less,
                    (false, true) => std::cmp::Ordering::Greater,
                    (false, false) => {
                        match (a.is_skipped(), b.is_skipped()) {
                            (true, false) => std::cmp::Ordering::Less,
                            (false, true) => std::cmp::Ordering::Greater,
                            _ => std::cmp::Ordering::Equal,
                        }
                    }
                }
            });

            // Compact table headers
            let show_metrics = match task {
                Task::NER | Task::DiscontinuousNER => {
                    md.push_str("| Dataset | Backend | F1 | P | R | N | ms |\n");
                    md.push_str("|---------|---------|----|----|----|---|----|\n");
                    true
                }
                Task::IntraDocCoref | Task::AbstractAnaphora => {
                    md.push_str("| Dataset | Backend | CoNLL | MUC | B³ | N | ms |\n");
                    md.push_str("|---------|---------|-------|-----|----|---|----|\n");
                    true
                }
                Task::RelationExtraction => {
                    md.push_str("| Dataset | Backend | Strict | Boundary | N | ms |\n");
                    md.push_str("|---------|---------|--------|----------|---|----|\n");
                    true
                }
                _ => {
                    md.push_str("| Dataset | Backend | N | ms |\n");
                    md.push_str("|---------|---------|---|----|\n");
                    false
                }
            };

            for result in results {
                let time_str = result.duration_ms
                    .map(|d| format!("{:.0}", d))
                    .unwrap_or_else(|| "-".to_string());
                
                if show_metrics && result.success {
                    match task {
                        Task::NER | Task::DiscontinuousNER => {
                            let f1 = result.metrics.get("f1").map(|v| *v * 100.0).unwrap_or(0.0);
                            let p = result.metrics.get("precision").map(|v| *v * 100.0).unwrap_or(0.0);
                            let r = result.metrics.get("recall").map(|v| *v * 100.0).unwrap_or(0.0);
                            md.push_str(&format!(
                                "| {:?} | {} | {:.1} | {:.1} | {:.1} | {} | {} |\n",
                                result.dataset, result.backend, f1, p, r, result.num_examples, time_str
                            ));
                        }
                        Task::IntraDocCoref | Task::AbstractAnaphora => {
                            let conll = result.metrics.get("conll_f1").map(|v| *v * 100.0).unwrap_or(0.0);
                            let muc = result.metrics.get("muc_f1").map(|v| *v * 100.0).unwrap_or(0.0);
                            let b3 = result.metrics.get("b3_f1").map(|v| *v * 100.0).unwrap_or(0.0);
                            md.push_str(&format!(
                                "| {:?} | {} | {:.1} | {:.1} | {:.1} | {} | {} |\n",
                                result.dataset, result.backend, conll, muc, b3, result.num_examples, time_str
                            ));
                        }
                        Task::RelationExtraction => {
                            let strict = result.metrics.get("strict_f1").map(|v| *v * 100.0).unwrap_or(0.0);
                            let boundary = result.metrics.get("boundary_f1").map(|v| *v * 100.0).unwrap_or(0.0);
                            md.push_str(&format!(
                                "| {:?} | {} | {:.1} | {:.1} | {} | {} |\n",
                                result.dataset, result.backend, strict, boundary, result.num_examples, time_str
                            ));
                        }
                        _ => {
                            md.push_str(&format!(
                                "| {:?} | {} | {} | {} |\n",
                                result.dataset, result.backend, result.num_examples, time_str
                            ));
                        }
                    }
                } else {
                    // Failed or skipped - show error
                    let status = if result.is_skipped() { "⊘" } else { "✗" };
                    let error_msg = if result.is_skipped() {
                        "no-feature".to_string()
                    } else {
                        result.error.as_ref()
                            .map(|e| {
                                // Extract key error info
                                if e.contains("Unknown backend") {
                                    "unknown-backend".to_string()
                                } else if e.contains("Failed to load") {
                                    "load-failed".to_string()
                                } else if e.len() > 30 {
                                    e.chars().take(30).collect::<String>() + "..."
                                } else {
                                    e.clone()
                                }
                            })
                            .unwrap_or_else(|| "error".to_string())
                    };
                    md.push_str(&format!(
                        "| {:?} | {} | {} | {} | {} |\n",
                        result.dataset, result.backend, status, error_msg, time_str
                    ));
                }
            }
            md.push('\n');
        }
        
        // Backend summary (compact)
        let mut backend_stats: HashMap<String, (usize, usize, usize, f64)> = HashMap::new();
        for result in &self.results {
            let entry = backend_stats.entry(result.backend.clone()).or_insert((0, 0, 0, 0.0));
            if result.success {
                entry.0 += 1;
                if let Some(f1) = result.primary_f1() {
                    entry.3 += f1;
                }
            } else if result.is_skipped() {
                entry.1 += 1;
            } else {
                entry.2 += 1;
            }
        }
        
        if !backend_stats.is_empty() {
            md.push_str("## Backend Summary\n\n");
            md.push_str("| Backend | ✓ | ⊘ | ✗ | Avg F1 |\n");
            md.push_str("|---------|---|---|---|--------|\n");
            let mut backends: Vec<_> = backend_stats.iter().collect();
            backends.sort_by_key(|(_, (success, _, _, _))| *success);
            backends.reverse();
            for (backend, (success, skipped, failed, total_f1)) in backends {
                let avg_f1 = if *success > 0 { total_f1 / *success as f64 * 100.0 } else { 0.0 };
                md.push_str(&format!(
                    "| {} | {} | {} | {} | {:.1} |\n",
                    backend, success, skipped, failed, avg_f1
                ));
            }
            md.push_str("\n");
        }

        md
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_mapping_build() {
        let mapping = TaskMapping::build();
        assert!(!mapping.task_to_datasets.is_empty());
        assert!(!mapping.dataset_to_tasks.is_empty());
        assert!(!mapping.backend_to_tasks.is_empty());
        assert!(!mapping.task_to_backends.is_empty());
    }

    #[test]
    fn test_task_evaluator_creation() {
        let evaluator = TaskEvaluator::new();
        assert!(evaluator.is_ok());
    }

    #[test]
    fn test_gliner2_capabilities() {
        let tasks = crate::eval::task_mapping::backend_tasks("gliner2");
        assert!(tasks.contains(&Task::NER));
        assert!(tasks.contains(&Task::RelationExtraction));
        assert!(tasks.contains(&Task::TextClassification));
    }
}
