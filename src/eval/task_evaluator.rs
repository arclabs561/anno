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

use crate::backends::inference::ZeroShotNER;
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

// Constants for evaluation
/// 95% confidence interval z-score (normal distribution)
const DEFAULT_Z_SCORE_95: f64 = 1.96;
/// Placeholder standard deviation when actual variance cannot be computed
const DEFAULT_PLACEHOLDER_STD_DEV: f64 = 0.05;
/// Maximum sample size for confidence interval computation (to avoid expensive recomputation)
const MAX_CI_SAMPLE_SIZE: usize = 100;
/// Minimum sample size for confidence interval computation
const MIN_CI_SAMPLE_SIZE: usize = 1;
/// Maximum number of examples for robustness testing (performance limit)
const ROBUSTNESS_TEST_LIMIT: usize = 50;

/// Stratified metrics across multiple dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratifiedMetrics {
    /// Metrics by entity type
    pub by_entity_type: HashMap<String, MetricWithCI>,
    /// Metrics by temporal stratum (if available)
    pub by_temporal_stratum: Option<HashMap<String, MetricWithCI>>,
    /// Metrics by surface form type (proper noun, common noun, pronoun)
    pub by_surface_form: Option<HashMap<String, MetricWithCI>>,
    /// Metrics by mention characteristics (capitalized, partial name, etc.)
    pub by_mention_char: Option<HashMap<String, MetricWithCI>>,
}

/// Metrics with confidence intervals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricWithCI {
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// 95% confidence interval (lower, upper)
    pub ci_95: (f64, f64),
    /// Sample size
    pub n: usize,
}

/// Confidence intervals for key metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceIntervals {
    /// F1 score CI
    pub f1_ci: (f64, f64),
    /// Precision CI
    pub precision_ci: (f64, f64),
    /// Recall CI
    pub recall_ci: (f64, f64),
}

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
    /// Confidence threshold for relation extraction (default: 0.5)
    pub relation_threshold: f32,
    /// Whether to run robustness testing (perturbations)
    pub robustness: bool,
    /// Whether to compute familiarity scores for zero-shot evaluations
    pub compute_familiarity: bool,
    /// Whether to compute temporal stratification (if dataset supports it)
    pub temporal_stratification: bool,
    /// Whether to compute confidence intervals for metrics
    pub confidence_intervals: bool,
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
            relation_threshold: 0.5,
            robustness: false,
            compute_familiarity: true, // Default to true for zero-shot awareness
            temporal_stratification: false,
            confidence_intervals: true, // Default to true for better reporting
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
    /// Label shift/familiarity metrics (if computed for zero-shot)
    pub label_shift: Option<super::types::LabelShift>,
    /// Robustness scores (if robustness testing was enabled)
    #[cfg(feature = "eval-advanced")]
    pub robustness: Option<super::robustness::RobustnessResults>,
    #[cfg(not(feature = "eval-advanced"))]
    /// Robustness testing results (only available with `eval-advanced` feature).
    #[cfg(not(feature = "eval-advanced"))]
    pub robustness: Option<()>, // Placeholder when feature not enabled
    /// Stratified metrics by various dimensions
    pub stratified: Option<StratifiedMetrics>,
    /// Confidence intervals for key metrics (if computed)
    pub confidence_intervals: Option<ConfidenceIntervals>,
    /// KB version used (if available from dataset metadata)
    pub kb_version: Option<String>,
}

impl TaskEvalResult {
    /// Check if this is a "skipped" result (feature not available or incompatible) vs actual failure
    pub fn is_skipped(&self) -> bool {
        if self.success {
            return false;
        }
        if let Some(ref err) = self.error {
            err.contains("Feature not available")
                || err.contains("requires '")
                || err.contains("Incompatible entity types")
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
    // Temporary storage for per-example scores (used during evaluation)
    // Cloned when needed to avoid borrow checker issues
    #[allow(dead_code)] // Used internally
    per_example_scores_cache:
        std::sync::Mutex<Option<Vec<(Vec<crate::Entity>, Vec<crate::Entity>, String)>>>,
}

impl TaskEvaluator {
    /// Create a new task evaluator.
    pub fn new() -> Result<Self> {
        Ok(Self {
            loader: DatasetLoader::new()?,
            mapping: TaskMapping::build(),
            per_example_scores_cache: std::sync::Mutex::new(None),
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
        let failed = results
            .iter()
            .filter(|r| !r.success && !r.is_skipped())
            .count();
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

    /// Check if backend is compatible with dataset entity types.
    ///
    /// - `stacked`: Compatible with most types (combines pattern+heuristic)
    /// - ML backends: Always compatible (zero-shot or trained)
    /// - `pattern`: Only structured entities (not named entities)
    /// - `heuristic`: Only Person, Organization, Location
    fn is_backend_compatible(backend_name: &str, dataset: DatasetId) -> bool {
        let entity_types = dataset.entity_types();
        let normalized_types: Vec<String> = entity_types.iter().map(|t| t.to_lowercase()).collect();

        match backend_name {
            // Stacked combines pattern+heuristic, so it's compatible with most types
            "stacked" => true,
            // ML backends are zero-shot or trained, so compatible
            "bert_onnx" | "candle_ner" | "nuner" | "gliner_onnx" | "gliner_candle" | "gliner2"
            | "w2ner" | "gliner_poly" | "deberta_v3" | "albert" | "universal_ner" | "tplinker" => {
                true
            }
            // Pattern only does structured entities (not named entities)
            "pattern" => {
                // PatternNER only extracts: Date, Time, Money, Percent, Email, URL, Phone
                // Not compatible with named entity datasets
                false
            }
            // Heuristic only does Person, Organization, Location
            "heuristic" => {
                let supported = [
                    "person",
                    "per",
                    "organization",
                    "org",
                    "location",
                    "loc",
                    "misc",
                ];
                normalized_types
                    .iter()
                    .all(|t| supported.iter().any(|s| t == s || t.starts_with(s)))
            }
            _ => true, // Unknown backends - assume compatible
        }
    }

    /// Evaluate a single task-dataset-backend combination.
    fn evaluate_combination(
        &self,
        task: Task,
        dataset: DatasetId,
        backend_name: &str,
        config: &TaskEvalConfig,
    ) -> Result<TaskEvalResult> {
        // Validate inputs
        if backend_name.is_empty() {
            return Err(crate::Error::InvalidInput(
                "Backend name cannot be empty".to_string(),
            ));
        }

        // Check compatibility before loading dataset
        if !Self::is_backend_compatible(backend_name, dataset) {
            return Ok(TaskEvalResult {
                task,
                dataset,
                backend: backend_name.to_string(),
                success: false,
                error: Some(format!(
                    "Incompatible entity types: backend '{}' doesn't support dataset entity types: {:?}",
                    backend_name,
                    dataset.entity_types()
                )),
                metrics: HashMap::new(),
                num_examples: 0,
                duration_ms: None,
                label_shift: None,
                #[cfg(feature = "eval-advanced")]
                robustness: None,
                #[cfg(not(feature = "eval-advanced"))]
                robustness: None,
                stratified: None,
                confidence_intervals: None,
                kb_version: None,
            });
        }
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
                            label_shift: None,
                            robustness: None,
                            stratified: None,
                            confidence_intervals: None,
                            kb_version: None,
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
                            label_shift: None,
                            #[cfg(feature = "eval-advanced")]
                            robustness: None,
                            #[cfg(not(feature = "eval-advanced"))]
                            robustness: None,
                            stratified: None,
                            confidence_intervals: None,
                            kb_version: None,
                        });
                    }
                }
            }
        };

        // Validate dataset is not empty
        if dataset_data.sentences.is_empty() {
            return Ok(TaskEvalResult {
                task,
                dataset,
                backend: backend_name.to_string(),
                success: false,
                error: Some(format!(
                    "Dataset '{}' is empty (no sentences found)",
                    dataset.name()
                )),
                metrics: HashMap::new(),
                num_examples: 0,
                duration_ms: None,
                label_shift: None,
                #[cfg(feature = "eval-advanced")]
                robustness: None,
                #[cfg(not(feature = "eval-advanced"))]
                robustness: None,
                stratified: None,
                confidence_intervals: None,
                kb_version: None,
            });
        }

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
                let selected_indices: Vec<usize> =
                    indices.iter().take(max).map(|(i, _)| *i).collect();
                let sampled_sentences: Vec<_> = selected_indices
                    .iter()
                    .filter_map(|&i| dataset_data.sentences.get(i).cloned())
                    .collect();
                use crate::eval::loader::LoadedDataset;
                let sampled_dataset = LoadedDataset {
                    id: dataset_data.id,
                    sentences: sampled_sentences,
                    loaded_at: dataset_data.loaded_at.clone(),
                    source_url: dataset_data.source_url.clone(),
                    temporal_metadata: dataset_data.temporal_metadata.clone(),
                };
                (sampled_dataset, max)
            }
        } else {
            (dataset_data, total)
        };

        // Try to evaluate backend (handles backend creation internally)
        let start = Instant::now();
        let result =
            match self.try_evaluate_backend(task, dataset, backend_name, &sampled_data, config) {
                Ok(metrics) => {
                    let duration = start.elapsed().as_secs_f64() * 1000.0;

                    // Compute familiarity for zero-shot backends
                    let label_shift = if config.compute_familiarity {
                        self.compute_familiarity_if_zero_shot(backend_name, &sampled_data)
                    } else {
                        None
                    };

                    // Run robustness testing if enabled
                    #[cfg(feature = "eval-advanced")]
                    let robustness_result: Option<
                        super::robustness::RobustnessResults,
                    > = if config.robustness && matches!(task, Task::NER | Task::DiscontinuousNER) {
                        self.compute_robustness(backend_name, &sampled_data, config)
                    } else {
                        None
                    };

                    // Compute stratified metrics (use per-example scores if available)
                    let stratified = if matches!(task, Task::NER | Task::DiscontinuousNER) {
                        let per_example_opt = {
                            self.per_example_scores_cache
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .clone()
                        };
                        if let Some(per_example) = per_example_opt.as_ref() {
                            self.compute_stratified_metrics_from_scores(
                                &sampled_data,
                                &metrics,
                                Some(per_example),
                            )
                        } else {
                            self.compute_stratified_metrics(&sampled_data, &metrics)
                        }
                    } else {
                        None
                    };

                    // Compute confidence intervals if requested (use per-example scores if available)
                    let confidence_intervals = if config.confidence_intervals {
                        let per_example_opt = {
                            self.per_example_scores_cache
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .clone()
                        };
                        if let Some(per_example) = per_example_opt.as_ref() {
                            self.compute_confidence_intervals_from_scores(per_example)
                        } else {
                            self.compute_confidence_intervals(
                                &sampled_data,
                                task,
                                backend_name,
                                &metrics,
                                config,
                            )
                        }
                    } else {
                        None
                    };

                    // Clear cache after use (handle mutex poisoning gracefully)
                    let mut cache = self
                        .per_example_scores_cache
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    *cache = None;

                    // Extract KB version if available
                    let kb_version = Self::extract_kb_version(&sampled_data);

                    TaskEvalResult {
                        task,
                        dataset,
                        backend: backend_name.to_string(),
                        success: true,
                        error: None,
                        metrics,
                        num_examples: sentences_to_use,
                        duration_ms: Some(duration),
                        label_shift,
                        #[cfg(feature = "eval-advanced")]
                        robustness: robustness_result,
                        #[cfg(not(feature = "eval-advanced"))]
                        robustness: None,
                        stratified,
                        confidence_intervals,
                        kb_version,
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
                        label_shift: None,
                        #[cfg(feature = "eval-advanced")]
                        robustness: None,
                        #[cfg(not(feature = "eval-advanced"))]
                        robustness: None,
                        stratified: None,
                        confidence_intervals: None,
                        kb_version: None,
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
        config: &TaskEvalConfig,
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

        // Run task-specific evaluation
        // Note: Coref tasks don't use BackendFactory (they use create_coref_resolver)
        match task {
            Task::NER | Task::DiscontinuousNER => {
                let backend = BackendFactory::create(backend_name)?;
                // Check availability before evaluation
                if !backend.is_available() {
                    return Err(crate::Error::FeatureNotAvailable(format!(
                        "Backend '{}' is not available (feature not enabled or model not loaded)",
                        backend_name
                    )));
                }
                self.evaluate_ner_task(backend_name, &*backend, dataset, dataset_data, config)
            }
            Task::IntraDocCoref | Task::AbstractAnaphora => {
                // Coref tasks use create_coref_resolver, not BackendFactory
                // Skip BackendFactory::create() to avoid "Unknown backend" error
                self.evaluate_coref_task(backend_name, dataset_data, config)
            }
            Task::RelationExtraction => {
                // Relation extraction requires a Model backend
                let backend = BackendFactory::create(backend_name)?;
                // Check availability before evaluation
                if !backend.is_available() {
                    return Err(crate::Error::FeatureNotAvailable(format!(
                        "Backend '{}' is not available (feature not enabled or model not loaded)",
                        backend_name
                    )));
                }
                self.evaluate_relation_task(backend_name, &*backend, dataset_data, config)
            }
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
        config: &TaskEvalConfig,
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

        // Track per-example scores for stratified metrics and confidence intervals
        let track_per_example = config.confidence_intervals || config.temporal_stratification;
        let mut per_example_scores: Vec<(Vec<Entity>, Vec<Entity>, String)> = Vec::new();

        // Extract dataset entity types and map to model-compatible labels
        let dataset_labels = dataset.entity_types();
        let mapped_labels = Self::map_dataset_labels_to_model(dataset_labels, backend_name);

        // Check if this is a zero-shot backend that needs custom labels
        let is_zero_shot = matches!(
            backend_name.to_lowercase().as_str(),
            "nuner" | "gliner_onnx" | "gliner_candle" | "gliner2" | "gliner_poly" | "universal_ner"
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

            // Normalize backend name to lowercase for consistent caching
            let backend_name_normalized = backend_name.to_lowercase();
            let backend_name_arc = Arc::new(backend_name_normalized);
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
                            // Check if we have a cached backend for this backend_name (case-insensitive)
                            if let Some((ref cached_name, ref backend)) = *cached {
                                if cached_name.to_lowercase() == backend_name_arc.as_str() {
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
                                    // Store normalized (lowercase) name for consistent matching
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
                    // Use unwrap_or_else to handle mutex poisoning gracefully
                    let mut last_percent = last_progress_percent.lock().unwrap_or_else(|e| e.into_inner());
                    if current_percent >= *last_percent + 10 || processed % 10 == 0 {
                        let elapsed = start_time_arc.lock().unwrap_or_else(|e| e.into_inner()).elapsed();
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

                    let text = sentence.text();
                    (chars_count, gold_entities, entities_result, text.to_string())
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

            // Aggregate results and track per-example scores if needed
            for (chars_count, gold_entities, entities_result, text) in all_results {
                total_chars += chars_count;
                all_gold.extend(gold_entities.clone());

                match entities_result {
                    Ok(entities) => {
                        all_predicted.extend(entities.clone());
                        if track_per_example {
                            per_example_scores.push((gold_entities, entities, text));
                        }
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
                        idx + 1, total_sentences, progress, backend_name, dataset, remaining_str);
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
                        all_predicted.extend(entities.clone());
                        if track_per_example {
                            let gold: Vec<Entity> = gold_entities
                                .iter()
                                .map(|g| {
                                    let mut entity = Entity::new(
                                        g.text.clone(),
                                        g.entity_type.clone(),
                                        g.start,
                                        g.end,
                                        1.0,
                                    );
                                    entity.provenance = Some(crate::Provenance::ml("gold", 1.0));
                                    entity
                                })
                                .collect();
                            per_example_scores.push((gold, entities, text.to_string()));
                        }
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
                total_sentences, total_sentences, backend_name, dataset, total_secs, rate);
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

        // Store per-example scores for later use in stratified metrics and confidence intervals
        if !per_example_scores.is_empty() {
            *self.per_example_scores_cache.lock().unwrap() = Some(per_example_scores);
        } else {
            *self.per_example_scores_cache.lock().unwrap() = None;
        }

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
                    // For zero-shot backends, preserve original labels (they can handle any type)
                    _ if matches!(
                        backend_name.to_lowercase().as_str(),
                        "nuner"
                            | "gliner_onnx"
                            | "gliner_candle"
                            | "gliner2"
                            | "gliner_poly"
                            | "universal_ner"
                    ) =>
                    {
                        label.to_lowercase()
                    }
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
            "gliner_poly" => {
                #[cfg(feature = "onnx")]
                {
                    use crate::backends::gliner_poly::GLiNERPoly;
                    use crate::DEFAULT_GLINER_MODEL;
                    let gliner_poly = GLiNERPoly::new(DEFAULT_GLINER_MODEL)?;
                    Ok(Box::new(gliner_poly))
                }
                #[cfg(not(feature = "onnx"))]
                {
                    Err(crate::Error::FeatureNotAvailable(
                        "GLiNER Poly requires the 'onnx' feature".to_string(),
                    ))
                }
            }
            "universal_ner" => {
                use crate::backends::universal_ner::UniversalNER;
                let universal_ner = UniversalNER::new()?;
                Ok(Box::new(universal_ner))
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
            "gliner_poly" => {
                #[cfg(feature = "onnx")]
                {
                    if let Some(gliner_poly) =
                        cached.downcast_ref::<crate::backends::gliner_poly::GLiNERPoly>()
                    {
                        gliner_poly.extract_with_types(text, &label_strs, 0.5)
                    } else {
                        Err(crate::Error::InvalidInput(
                            "Failed to downcast cached GLiNER Poly backend".to_string(),
                        ))
                    }
                }
                #[cfg(not(feature = "onnx"))]
                {
                    Err(crate::Error::FeatureNotAvailable(
                        "GLiNER Poly requires the 'onnx' feature".to_string(),
                    ))
                }
            }
            "universal_ner" => {
                if let Some(universal_ner) =
                    cached.downcast_ref::<crate::backends::universal_ner::UniversalNER>()
                {
                    universal_ner.extract_with_types(text, &label_strs, 0.5)
                } else {
                    Err(crate::Error::InvalidInput(
                        "Failed to downcast cached UniversalNER backend".to_string(),
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
        _config: &TaskEvalConfig,
    ) -> Result<HashMap<String, f64>> {
        use crate::eval::backend_factory::create_coref_resolver;
        use crate::eval::coref::entities_to_chains;
        use crate::eval::coref_metrics::CorefEvaluation;

        // Try to load coreference documents if dataset supports it
        let gold_docs = if dataset_data.id.is_coreference() {
            match self.loader.load_coref(dataset_data.id) {
                Ok(docs) => {
                    if docs.is_empty() {
                        // If load_coref returns empty, try downloading first
                        #[cfg(feature = "eval-advanced")]
                        {
                            if let Err(e) = self.loader.load_or_download_coref(dataset_data.id) {
                                return Err(crate::Error::InvalidInput(format!(
                                    "Failed to load coreference dataset {:?}: {}",
                                    dataset_data.id, e
                                )));
                            }
                            // Retry after download
                            self.loader.load_coref(dataset_data.id)?
                        }
                        #[cfg(not(feature = "eval-advanced"))]
                        {
                            return Err(crate::Error::InvalidInput(format!(
                                "Coreference dataset {:?} not cached. Enable eval-advanced feature to auto-download.",
                                dataset_data.id
                            )));
                        }
                    } else {
                        docs
                    }
                }
                Err(e) => {
                    // Try downloading if not cached
                    #[cfg(feature = "eval-advanced")]
                    {
                        if let Err(dl_err) = self.loader.load_or_download_coref(dataset_data.id) {
                            return Err(crate::Error::InvalidInput(format!(
                                "Failed to load/download coreference dataset {:?}: {} (original: {})",
                                dataset_data.id, dl_err, e
                            )));
                        }
                        // Retry after download
                        self.loader.load_coref(dataset_data.id)?
                    }
                    #[cfg(not(feature = "eval-advanced"))]
                    {
                        return Err(crate::Error::InvalidInput(format!(
                            "Coreference dataset {:?} not cached: {}. Enable eval-advanced feature to auto-download.",
                            dataset_data.id, e
                        )));
                    }
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

        // Create coreference resolver (not a Model backend)
        let resolver = create_coref_resolver(backend_name)?;

        // Use a NER backend to extract entities first (heuristic or stacked as default)
        let ner_backend_name = if backend_name == "coref_resolver" {
            "stacked" // Default NER backend for coref evaluation
        } else {
            backend_name // If a specific NER backend was requested
        };

        let ner_backend = BackendFactory::create(ner_backend_name)?;
        let mut all_predicted_chains = Vec::new();
        let mut all_gold_chains = Vec::new();

        for doc in &gold_docs {
            // Collect gold chains from the document
            all_gold_chains.extend(doc.chains.clone());

            // Extract entities from the document text using NER backend
            match ner_backend.extract_entities(&doc.text, None) {
                Ok(entities) => {
                    // Resolve coreference on predicted entities
                    let resolved_entities = resolver.resolve(&entities);
                    // Convert resolved entities to chains
                    let predicted_chains = entities_to_chains(&resolved_entities);
                    all_predicted_chains.extend(predicted_chains);
                }
                Err(e) => {
                    // Log error but continue with other documents
                    eprintln!("Warning: NER backend inference failed for document: {}", e);
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

        // Add chain-length stratification metrics
        if let Some(ref chain_stats) = eval.chain_stats {
            metrics.insert(
                "chain_long_count".to_string(),
                chain_stats.long_chain_count as f64,
            );
            metrics.insert(
                "chain_short_count".to_string(),
                chain_stats.short_chain_count as f64,
            );
            metrics.insert(
                "chain_singleton_count".to_string(),
                chain_stats.singleton_count as f64,
            );
            metrics.insert("chain_long_f1".to_string(), chain_stats.long_chain_f1);
            metrics.insert("chain_short_f1".to_string(), chain_stats.short_chain_f1);
            metrics.insert("chain_singleton_f1".to_string(), chain_stats.singleton_f1);
        }
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
    fn evaluate_relation_task(
        &self,
        backend_name: &str,
        backend: &dyn Model,
        dataset_data: &LoadedDataset,
        config: &TaskEvalConfig,
    ) -> Result<HashMap<String, f64>> {
        use crate::eval::relation::{
            evaluate_relations, RelationEvalConfig, RelationGold, RelationPrediction,
        };

        // Load gold relations from dataset
        let relation_docs = match self.loader.load_relation(dataset_data.id) {
            Ok(docs) => docs,
            Err(e) => {
                // If relation loading fails, return empty metrics
                eprintln!(
                    "Warning: Failed to load relations for {:?}: {}",
                    dataset_data.id, e
                );
                let mut metrics = HashMap::new();
                metrics.insert("boundary_f1".to_string(), 0.0);
                metrics.insert("strict_f1".to_string(), 0.0);
                metrics.insert("num_gold_relations".to_string(), 0.0);
                metrics.insert("num_predicted_relations".to_string(), 0.0);
                metrics.insert(
                    "num_sentences".to_string(),
                    dataset_data.sentences.len() as f64,
                );
                return Ok(metrics);
            }
        };

        // Collect all gold relations
        let mut all_gold_relations: Vec<RelationGold> = Vec::new();
        for doc in &relation_docs {
            all_gold_relations.extend(doc.relations.iter().cloned());
        }

        // Extract predicted relations from backend
        let mut all_predicted_relations: Vec<RelationPrediction> = Vec::new();

        // Extract relations using RelationExtractor if backend supports it
        // GLiNER2 backends implement RelationExtractor
        use crate::backends::inference::RelationExtractor;

        // Try to create RelationExtractor instance for relation extraction backends
        let relation_extractor: Option<Box<dyn RelationExtractor>> = match backend_name {
            #[cfg(feature = "onnx")]
            "gliner2" | "gliner2onnx" => {
                use crate::backends::gliner2::GLiNER2Onnx;
                use crate::DEFAULT_GLINER2_MODEL;
                match GLiNER2Onnx::from_pretrained(DEFAULT_GLINER2_MODEL) {
                    Ok(extractor) => Some(Box::new(extractor) as Box<dyn RelationExtractor>),
                    Err(e) => {
                        eprintln!(
                            "Warning: Failed to create GLiNER2Onnx for relation extraction: {}",
                            e
                        );
                        None
                    }
                }
            }
            #[cfg(all(feature = "candle", feature = "onnx"))]
            "gliner2_candle" | "gliner2candle" => {
                use crate::backends::gliner2::GLiNER2Candle;
                use crate::DEFAULT_GLINER2_MODEL;
                match GLiNER2Candle::from_pretrained(DEFAULT_GLINER2_MODEL) {
                    Ok(extractor) => Some(Box::new(extractor) as Box<dyn RelationExtractor>),
                    Err(e) => {
                        eprintln!(
                            "Warning: Failed to create GLiNER2Candle for relation extraction: {}",
                            e
                        );
                        None
                    }
                }
            }
            "tplinker" | "tplink" => {
                use crate::backends::tplinker::TPLinker;
                // TPLinker::new() returns Result, but for placeholder it always succeeds
                match TPLinker::new() {
                    Ok(extractor) => Some(Box::new(extractor) as Box<dyn RelationExtractor>),
                    Err(_) => None, // Should not happen for placeholder
                }
            }
            _ => None,
        };

        // Extract relations from each document
        for doc in &relation_docs {
            let text = &doc.text;

            if let Some(ref rel_extractor) = relation_extractor {
                // Use RelationExtractor to extract relations
                // Get entity types and relation types from gold relations
                let entity_types: Vec<&str> = doc
                    .relations
                    .iter()
                    .flat_map(|r| vec![r.head_type.as_str(), r.tail_type.as_str()])
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect();

                let relation_types: Vec<&str> = doc
                    .relations
                    .iter()
                    .map(|r| r.relation_type.as_str())
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect();

                // Use configurable threshold from TaskEvalConfig
                match rel_extractor.extract_with_relations(
                    text,
                    &entity_types,
                    &relation_types,
                    config.relation_threshold,
                ) {
                    Ok(extraction) => {
                        // Convert ExtractionWithRelations to RelationPrediction
                        for triple in &extraction.relations {
                            if let (Some(head), Some(tail)) = (
                                extraction.entities.get(triple.head_idx),
                                extraction.entities.get(triple.tail_idx),
                            ) {
                                all_predicted_relations.push(RelationPrediction {
                                    head_span: (head.start, head.end),
                                    head_type: head.entity_type.as_label().to_string(),
                                    tail_span: (tail.start, tail.end),
                                    tail_type: tail.entity_type.as_label().to_string(),
                                    relation_type: triple.relation_type.clone(),
                                    confidence: triple.confidence,
                                });
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Warning: Relation extraction failed: {}", e);
                    }
                }
            } else {
                // Fallback: Extract entities and create placeholder relations
                let entities = match backend.extract_entities(text, None) {
                    Ok(ents) => ents,
                    Err(e) => {
                        eprintln!("Warning: Entity extraction failed: {}", e);
                        continue;
                    }
                };

                // Create placeholder relations for nearby entity pairs
                if entities.len() >= 2 {
                    for i in 0..entities.len() {
                        for j in (i + 1)..entities.len().min(i + 3) {
                            let head = &entities[i];
                            let tail = &entities[j];

                            all_predicted_relations.push(RelationPrediction {
                                head_span: (head.start, head.end),
                                head_type: head.entity_type.as_label().to_string(),
                                tail_span: (tail.start, tail.end),
                                tail_type: tail.entity_type.as_label().to_string(),
                                relation_type: "RELATED".to_string(), // Placeholder
                                confidence: 0.5,
                            });
                        }
                    }
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
        let avg_examples: f64 = self
            .results
            .iter()
            .filter(|r| r.success)
            .map(|r| r.num_examples as f64)
            .sum::<f64>()
            / self.summary.successful.max(1) as f64;
        let avg_time: f64 = self
            .results
            .iter()
            .filter_map(|r| r.duration_ms)
            .sum::<f64>()
            / self
                .results
                .iter()
                .filter(|r| r.duration_ms.is_some())
                .count()
                .max(1) as f64;

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
        let failures: Vec<_> = self
            .results
            .iter()
            .filter(|r| !r.success && !r.is_skipped())
            .collect();

        if !failures.is_empty() {
            md.push_str("## Failures\n\n");
            md.push_str("| Task | Dataset | Backend | Error |\n");
            md.push_str("|------|---------|---------|-------|\n");
            for result in &failures {
                let error = result
                    .error
                    .as_ref()
                    .map(|e| e.replace('|', "\\|").replace('\n', " "))
                    .unwrap_or_else(|| "N/A".to_string());
                md.push_str(&format!(
                    "| {} | {:?} | {} | {} |\n",
                    result.task.name(),
                    result.dataset,
                    result.backend,
                    error
                ));
            }
            md.push('\n');
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
            md.push('\n');
        }

        md.push_str("## Results\n\n");

        // Filter out skipped entries for cleaner report (show summary instead)
        let skipped_count = self.results.iter().filter(|r| r.is_skipped()).count();
        if skipped_count > 0 {
            md.push_str(&format!(
                "**Note**: {} combinations skipped (features not enabled or incompatible). Showing successful and failed results only.\n\n",
                skipped_count
            ));
        }

        // Add compatibility notes
        md.push_str("**Compatibility Notes**:\n");
        md.push_str("- `stacked`: Combines pattern+heuristic, supports structured entities (date/time/money/etc) and named entities (PER/ORG/LOC), but not biomedical types\n");
        md.push_str("- `pattern`: Only structured entities (date, time, money, percent, email, URL, phone)\n");
        md.push_str("- `heuristic`: Only named entities (Person, Organization, Location)\n");
        md.push_str("- `0.0 F1` with N>0: Backend doesn't support dataset entity types\n");
        md.push_str("- `N=0` or `N=1`: Dataset parsing issue or insufficient data\n\n");

        // Group results by task, filtering out skipped
        let mut by_task: HashMap<Task, Vec<&TaskEvalResult>> = HashMap::new();
        for result in &self.results {
            if !result.is_skipped() {
                by_task.entry(result.task).or_default().push(result);
            }
        }

        for (task, mut results) in by_task {
            md.push_str(&format!("### {}\n\n", task.name()));

            // Sort results: successful first (by F1 descending), then skipped, then failed
            results.sort_by(|a, b| match (a.success, b.success) {
                (true, true) => {
                    let a_f1 = a.primary_f1().unwrap_or(0.0);
                    let b_f1 = b.primary_f1().unwrap_or(0.0);
                    b_f1.partial_cmp(&a_f1).unwrap_or(std::cmp::Ordering::Equal)
                }
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                (false, false) => match (a.is_skipped(), b.is_skipped()) {
                    (true, false) => std::cmp::Ordering::Less,
                    (false, true) => std::cmp::Ordering::Greater,
                    _ => std::cmp::Ordering::Equal,
                },
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
                let time_str = result
                    .duration_ms
                    .map(|d| format!("{:.0}", d))
                    .unwrap_or_else(|| "-".to_string());

                if show_metrics && result.success {
                    match task {
                        Task::NER | Task::DiscontinuousNER => {
                            let f1 = result.metrics.get("f1").map(|v| *v * 100.0).unwrap_or(0.0);
                            let p = result
                                .metrics
                                .get("precision")
                                .map(|v| *v * 100.0)
                                .unwrap_or(0.0);
                            let r = result
                                .metrics
                                .get("recall")
                                .map(|v| *v * 100.0)
                                .unwrap_or(0.0);

                            // Add familiarity note for zero-shot backends
                            let mut note_parts = Vec::new();
                            if let Some(ref label_shift) = result.label_shift {
                                if label_shift.is_inflated() {
                                    note_parts.push(format!(
                                        "⚠ familiarity={:.0}%",
                                        label_shift.familiarity * 100.0
                                    ));
                                }
                            }

                            // Add note for 0.0 F1 scores
                            let note = if f1 < 0.1 && result.num_examples > 0 {
                                // Check if it's an incompatible entity type issue
                                let dataset_entity_types = result.dataset.entity_types();
                                let backend_name = &result.backend;
                                if backend_name == "stacked"
                                    || backend_name == "heuristic"
                                    || backend_name == "pattern"
                                {
                                    // Stacked/heuristic/pattern have limited entity type support
                                    let normalized_types: Vec<String> = dataset_entity_types
                                        .iter()
                                        .map(|t| t.to_lowercase())
                                        .collect();
                                    let supports_structured = normalized_types.iter().any(|t| {
                                        t.contains("date")
                                            || t.contains("time")
                                            || t.contains("money")
                                            || t.contains("percent")
                                            || t.contains("email")
                                            || t.contains("url")
                                            || t.contains("phone")
                                    });
                                    let supports_named = normalized_types.iter().any(|t| {
                                        t.contains("person")
                                            || t.contains("organization")
                                            || t.contains("location")
                                    });
                                    let supports_biomedical = normalized_types.iter().any(|t| {
                                        t.contains("disease")
                                            || t.contains("chemical")
                                            || t.contains("gene")
                                            || t.contains("protein")
                                            || t.contains("anatomy")
                                    });

                                    if backend_name == "pattern" && !supports_structured {
                                        " (pattern: no structured entities)"
                                    } else if backend_name == "heuristic" && !supports_named {
                                        " (heuristic: no PER/ORG/LOC)"
                                    } else if backend_name == "stacked"
                                        && !supports_structured
                                        && !supports_named
                                    {
                                        if supports_biomedical {
                                            " (stacked: biomedical not supported)"
                                        } else {
                                            " (stacked: incompatible types)"
                                        }
                                    } else {
                                        ""
                                    }
                                } else if result.num_examples == 0 {
                                    " (N=0: no data)"
                                } else {
                                    ""
                                }
                            } else {
                                ""
                            };

                            md.push_str(&format!(
                                "| {:?} | {} | {:.1} | {:.1} | {:.1} | {} | {} |{}\n",
                                result.dataset,
                                result.backend,
                                f1,
                                p,
                                r,
                                result.num_examples,
                                time_str,
                                note
                            ));

                            // Add stratified metrics section if available
                            if let Some(ref stratified) = result.stratified {
                                if !stratified.by_entity_type.is_empty() {
                                    md.push_str("\n#### Stratified by Entity Type\n\n");
                                    md.push_str("| Type | F1 | CI 95% | N |\n");
                                    md.push_str("|------|----|--------|---|\n");
                                    let mut types: Vec<_> =
                                        stratified.by_entity_type.iter().collect();
                                    types.sort_by_key(|(k, _)| *k);
                                    for (type_str, metric_ci) in types {
                                        let ci_str = format!(
                                            "[{:.2}, {:.2}]",
                                            metric_ci.ci_95.0, metric_ci.ci_95.1
                                        );
                                        md.push_str(&format!(
                                            "| {} | {:.2} | {} | {} |\n",
                                            type_str, metric_ci.mean, ci_str, metric_ci.n
                                        ));
                                    }
                                    md.push('\n');
                                }
                            }

                            // Add temporal stratification if available
                            if let Some(ref stratified) = result.stratified {
                                if let Some(ref temporal) = stratified.by_temporal_stratum {
                                    if !temporal.is_empty() {
                                        md.push_str("\n#### Temporal Stratification\n\n");
                                        md.push_str("| Stratum | F1 | CI 95% | N |\n");
                                        md.push_str("|---------|----|--------|---|\n");
                                        for (stratum, metric) in temporal {
                                            md.push_str(&format!(
                                                "| {} | {:.2} | [{:.2}, {:.2}] | {} |\n",
                                                stratum,
                                                metric.mean,
                                                metric.ci_95.0,
                                                metric.ci_95.1,
                                                metric.n
                                            ));
                                        }
                                        md.push('\n');
                                    }
                                }
                            }

                            // Add confidence intervals if available
                            if let Some(ref ci) = result.confidence_intervals {
                                md.push_str(&format!(
                                    "\n**Confidence Intervals (95%)**: F1: [{:.2}, {:.2}], P: [{:.2}, {:.2}], R: [{:.2}, {:.2}]\n\n",
                                    ci.f1_ci.0, ci.f1_ci.1,
                                    ci.precision_ci.0, ci.precision_ci.1,
                                    ci.recall_ci.0, ci.recall_ci.1
                                ));
                            }
                        }
                        Task::IntraDocCoref | Task::AbstractAnaphora => {
                            let conll = result
                                .metrics
                                .get("conll_f1")
                                .map(|v| *v * 100.0)
                                .unwrap_or(0.0);
                            let muc = result
                                .metrics
                                .get("muc_f1")
                                .map(|v| *v * 100.0)
                                .unwrap_or(0.0);
                            let b3 = result
                                .metrics
                                .get("b3_f1")
                                .map(|v| *v * 100.0)
                                .unwrap_or(0.0);

                            // Add note for 0.0 scores with low N
                            let note = if conll < 0.1 && result.num_examples <= 1 {
                                " (N≤1: insufficient data or parsing issue)"
                            } else {
                                ""
                            };

                            md.push_str(&format!(
                                "| {:?} | {} | {:.1} | {:.1} | {:.1} | {} | {} |{}\n",
                                result.dataset,
                                result.backend,
                                conll,
                                muc,
                                b3,
                                result.num_examples,
                                time_str,
                                note
                            ));

                            // Add chain-length stratification if available in metrics
                            if let Some(long_f1) = result.metrics.get("chain_long_f1") {
                                md.push_str("\n#### Chain-Length Stratification\n\n");
                                md.push_str("| Chain Type | Count | F1 |\n");
                                md.push_str("|------------|-------|----|\n");
                                if let Some(long_count) = result.metrics.get("chain_long_count") {
                                    md.push_str(&format!(
                                        "| Long (>10) | {:.0} | {:.2} |\n",
                                        long_count,
                                        long_f1 * 100.0
                                    ));
                                }
                                if let Some(short_f1) = result.metrics.get("chain_short_f1") {
                                    if let Some(short_count) =
                                        result.metrics.get("chain_short_count")
                                    {
                                        md.push_str(&format!(
                                            "| Short (2-10) | {:.0} | {:.2} |\n",
                                            short_count,
                                            short_f1 * 100.0
                                        ));
                                    }
                                }
                                if let Some(singleton_f1) = result.metrics.get("chain_singleton_f1")
                                {
                                    if let Some(singleton_count) =
                                        result.metrics.get("chain_singleton_count")
                                    {
                                        md.push_str(&format!(
                                            "| Singleton (1) | {:.0} | {:.2} |\n",
                                            singleton_count,
                                            singleton_f1 * 100.0
                                        ));
                                    }
                                }
                                md.push('\n');
                            }
                        }
                        Task::RelationExtraction => {
                            let strict = result
                                .metrics
                                .get("strict_f1")
                                .map(|v| *v * 100.0)
                                .unwrap_or(0.0);
                            let boundary = result
                                .metrics
                                .get("boundary_f1")
                                .map(|v| *v * 100.0)
                                .unwrap_or(0.0);
                            md.push_str(&format!(
                                "| {:?} | {} | {:.1} | {:.1} | {} | {} |\n",
                                result.dataset,
                                result.backend,
                                strict,
                                boundary,
                                result.num_examples,
                                time_str
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
                        result
                            .error
                            .as_ref()
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
            let entry = backend_stats
                .entry(result.backend.clone())
                .or_insert((0, 0, 0, 0.0));
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
                let avg_f1 = if *success > 0 {
                    total_f1 / *success as f64 * 100.0
                } else {
                    0.0
                };
                md.push_str(&format!(
                    "| {} | {} | {} | {} | {:.1} |\n",
                    backend, success, skipped, failed, avg_f1
                ));
            }
            md.push('\n');
        }

        md
    }
}

// =============================================================================
// Helper Functions for Advanced Evaluation Features
// =============================================================================

impl TaskEvaluator {
    /// Extract KB version from dataset metadata if available.
    ///
    /// Returns KB version string if temporal metadata contains it.
    fn extract_kb_version(dataset_data: &super::loader::LoadedDataset) -> Option<String> {
        dataset_data.temporal_metadata.as_ref()?.kb_version.clone()
    }

    /// Compute familiarity for zero-shot backends.
    ///
    /// Returns None if backend is not zero-shot or if familiarity cannot be computed.
    fn compute_familiarity_if_zero_shot(
        &self,
        backend_name: &str,
        dataset_data: &LoadedDataset,
    ) -> Option<super::types::LabelShift> {
        // Check if this is a zero-shot backend
        let is_zero_shot = matches!(
            backend_name.to_lowercase().as_str(),
            "nuner" | "gliner_onnx" | "gliner_candle" | "gliner2" | "gliner_poly" | "universal_ner"
        );

        if !is_zero_shot {
            return None;
        }

        // Extract dataset entity types
        let eval_types: Vec<String> = dataset_data
            .sentences
            .iter()
            .flat_map(|s| s.entities())
            .map(|e| e.entity_type.as_label().to_string())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        // For zero-shot backends, we don't have training types, so we use a heuristic:
        // Common entity types that zero-shot models are typically trained on
        let common_train_types = vec![
            "person".to_string(),
            "organization".to_string(),
            "location".to_string(),
            "PER".to_string(),
            "ORG".to_string(),
            "LOC".to_string(),
            "PERSON".to_string(),
            "ORGANIZATION".to_string(),
        ];

        Some(super::types::LabelShift::from_type_sets(
            &common_train_types,
            &eval_types,
        ))
    }

    /// Compute confidence intervals for key metrics.
    ///
    /// Uses normal approximation: CI = mean ± 1.96 * std_dev / sqrt(n)
    ///
    /// Note: This is a simplified version. For proper CI computation, we'd need
    /// per-example scores to compute variance. This placeholder uses a fixed std_dev.
    /// Compute confidence intervals from aggregate metrics (fallback method).
    ///
    /// This is a simplified version that uses aggregate metrics with placeholder
    /// standard deviation. Prefer `compute_confidence_intervals_from_scores` when
    /// per-example scores are available.
    fn compute_confidence_intervals_from_aggregate(
        &self,
        metrics: &HashMap<String, f64>,
    ) -> Option<ConfidenceIntervals> {
        // For now, we compute CI from the metrics themselves
        // In a full implementation, we'd need per-example scores to compute proper CIs
        // This is a placeholder that uses the metric values as if they were means

        let f1 = metrics.get("f1")?;
        let precision = metrics.get("precision")?;
        let recall = metrics.get("recall")?;

        // Placeholder: assume std_dev = DEFAULT_PLACEHOLDER_STD_DEV (would need actual variance computation)
        // In practice, this should be computed from per-example scores
        let std_dev = DEFAULT_PLACEHOLDER_STD_DEV;
        let z = DEFAULT_Z_SCORE_95; // 95% CI
        let margin = z * std_dev;

        Some(ConfidenceIntervals {
            f1_ci: ((f1 - margin).max(0.0), (f1 + margin).min(1.0)),
            precision_ci: ((precision - margin).max(0.0), (precision + margin).min(1.0)),
            recall_ci: ((recall - margin).max(0.0), (recall + margin).min(1.0)),
        })
    }

    /// Compute confidence intervals from per-example scores (improved version).
    ///
    /// Computes variance from per-example F1, precision, recall scores.
    ///
    /// # Performance Note
    ///
    /// This function creates a new backend instance and re-runs inference on a sample
    /// of the dataset to compute per-example scores. This is intentional - proper CI
    /// computation requires per-example variance, which isn't available from aggregate
    /// metrics alone.
    ///
    /// # Limitations
    ///
    /// - Samples up to `MAX_CI_SAMPLE_SIZE` examples for performance
    /// - Creates a new backend instance (doesn't reuse from main evaluation)
    /// - For zero-shot backends, creates and uses zero-shot backend instance
    /// Compute confidence intervals from per-example scores or aggregate metrics.
    ///
    /// This is the primary method for computing confidence intervals.
    /// For NER tasks, it samples sentences and re-runs inference to get per-example scores.
    /// For other tasks, it falls back to aggregate metrics with placeholder variance.
    fn compute_confidence_intervals(
        &self,
        dataset_data: &LoadedDataset,
        task: Task,
        backend_name: &str,
        aggregate_metrics: &HashMap<String, f64>,
        _config: &TaskEvalConfig,
    ) -> Option<ConfidenceIntervals> {
        // For NER tasks, compute per-example scores
        if !matches!(task, Task::NER | Task::DiscontinuousNER) {
            return self.compute_confidence_intervals_from_aggregate(aggregate_metrics);
        }

        // Sample a subset for CI computation (to avoid expensive recomputation)
        // Ensure sample_size is at least MIN_CI_SAMPLE_SIZE and doesn't exceed dataset size
        let dataset_len = dataset_data.sentences.len();
        if dataset_len == 0 {
            return self.compute_confidence_intervals_from_aggregate(aggregate_metrics);
        }
        let sample_size = dataset_len.min(MAX_CI_SAMPLE_SIZE).max(MIN_CI_SAMPLE_SIZE);
        let sample: Vec<_> = dataset_data.sentences.iter().take(sample_size).collect();

        // Compute per-example F1, precision, recall
        let mut f1_scores = Vec::new();
        let mut precision_scores = Vec::new();
        let mut recall_scores = Vec::new();

        // Try to create backend for per-example evaluation
        let backend = match BackendFactory::create(backend_name) {
            Ok(b) => b,
            Err(_) => return self.compute_confidence_intervals_from_aggregate(aggregate_metrics),
        };

        if !backend.is_available() {
            return self.compute_confidence_intervals_from_aggregate(aggregate_metrics);
        }

        let dataset_labels = dataset_data.id.entity_types();
        let mapped_labels = Self::map_dataset_labels_to_model(&dataset_labels, backend_name);
        let is_zero_shot = matches!(
            backend_name.to_lowercase().as_str(),
            "nuner" | "gliner_onnx" | "gliner_candle" | "gliner2" | "gliner_poly" | "universal_ner"
        );

        for sentence in sample {
            let text = sentence.text();
            let gold: Vec<Entity> = sentence
                .entities()
                .iter()
                .map(|g| {
                    let mut entity =
                        Entity::new(g.text.clone(), g.entity_type.clone(), g.start, g.end, 1.0);
                    entity.provenance = Some(crate::Provenance::ml("gold", 1.0));
                    entity
                })
                .collect();

            let predicted = if is_zero_shot && !mapped_labels.is_empty() {
                // For zero-shot backends, use extract_with_types
                // Create zero-shot backend instance (reuse thread-local cache if available)
                match Self::create_zero_shot_backend(backend_name) {
                    Ok(zero_shot_backend) => {
                        match Self::extract_with_cached_backend(
                            backend_name,
                            zero_shot_backend.as_ref(),
                            &text,
                            &mapped_labels,
                        ) {
                            Ok(entities) => entities,
                            Err(_) => continue,
                        }
                    }
                    Err(_) => continue,
                }
            } else {
                match backend.extract_entities(&text, None) {
                    Ok(e) => e,
                    Err(_) => continue,
                }
            };

            // Compute per-example metrics
            use crate::eval::ner_metrics::evaluate_entities;
            let result = evaluate_entities(&gold, &predicted);
            let summary = result.summary();
            f1_scores.push(summary.strict_f1);
            precision_scores.push(summary.strict_precision);
            recall_scores.push(summary.strict_recall);
        }

        if f1_scores.is_empty() {
            return self.compute_confidence_intervals_from_aggregate(aggregate_metrics);
        }

        // Compute mean and std_dev
        let n = f1_scores.len() as f64;
        let f1_mean = f1_scores.iter().sum::<f64>() / n;
        let precision_mean = precision_scores.iter().sum::<f64>() / n;
        let recall_mean = recall_scores.iter().sum::<f64>() / n;

        // Use sample variance (Bessel's correction: n-1) for unbiased estimate
        let f1_variance = if n > 1.0 {
            f1_scores
                .iter()
                .map(|&x| (x - f1_mean).powi(2))
                .sum::<f64>()
                / (n - 1.0)
        } else {
            0.0
        };
        let precision_variance = if n > 1.0 {
            precision_scores
                .iter()
                .map(|&x| (x - precision_mean).powi(2))
                .sum::<f64>()
                / (n - 1.0)
        } else {
            0.0
        };
        let recall_variance = if n > 1.0 {
            recall_scores
                .iter()
                .map(|&x| (x - recall_mean).powi(2))
                .sum::<f64>()
                / (n - 1.0)
        } else {
            0.0
        };

        let f1_std_dev = f1_variance.sqrt();
        let precision_std_dev = precision_variance.sqrt();
        let recall_std_dev = recall_variance.sqrt();

        // 95% CI: mean ± DEFAULT_Z_SCORE_95 * std_dev / sqrt(n)
        let z = DEFAULT_Z_SCORE_95;
        let f1_margin = z * f1_std_dev / n.sqrt();
        let precision_margin = z * precision_std_dev / n.sqrt();
        let recall_margin = z * recall_std_dev / n.sqrt();

        Some(ConfidenceIntervals {
            f1_ci: (
                (f1_mean - f1_margin).max(0.0),
                (f1_mean + f1_margin).min(1.0),
            ),
            precision_ci: (
                (precision_mean - precision_margin).max(0.0),
                (precision_mean + precision_margin).min(1.0),
            ),
            recall_ci: (
                (recall_mean - recall_margin).max(0.0),
                (recall_mean + recall_margin).min(1.0),
            ),
        })
    }

    /// Compute robustness testing results.
    ///
    /// # Performance Note
    ///
    /// This function creates a new backend instance and runs robustness tests on up to
    /// `ROBUSTNESS_TEST_LIMIT` examples. This is intentional - robustness testing requires
    /// running perturbations that may affect backend state.
    ///
    /// # Limitations
    ///
    /// - Limited to `ROBUSTNESS_TEST_LIMIT` examples for performance
    /// - Creates a new backend instance (doesn't reuse from main evaluation)
    #[cfg(feature = "eval-advanced")]
    pub(crate) fn compute_robustness(
        &self,
        backend_name: &str,
        dataset_data: &LoadedDataset,
        config: &TaskEvalConfig,
    ) -> Option<super::robustness::RobustnessResults> {
        use super::robustness::RobustnessEvaluator;
        use crate::Entity;

        // Create backend for robustness testing
        // NOTE: We create a new backend instance here rather than reusing from main evaluation
        // because robustness testing may modify backend state through perturbations
        let backend = match BackendFactory::create(backend_name) {
            Ok(b) => b,
            Err(_) => return None,
        };

        if !backend.is_available() {
            return None;
        }

        // Prepare test cases (limit to ROBUSTNESS_TEST_LIMIT for performance)
        let test_cases: Vec<(String, Vec<Entity>)> = dataset_data
            .sentences
            .iter()
            .take(ROBUSTNESS_TEST_LIMIT)
            .map(|s| {
                let gold: Vec<Entity> = s
                    .entities()
                    .iter()
                    .map(|g| {
                        let mut entity =
                            Entity::new(g.text.clone(), g.entity_type.clone(), g.start, g.end, 1.0);
                        entity.provenance = Some(crate::Provenance::ml("gold", 1.0));
                        entity
                    })
                    .collect();
                (s.text().to_string(), gold)
            })
            .collect();

        if test_cases.is_empty() {
            return None;
        }

        // Create robustness evaluator
        let evaluator = RobustnessEvaluator {
            seed: config.seed.unwrap_or(42),
            ..Default::default()
        };

        // Run robustness evaluation
        Some(evaluator.evaluate(backend.as_ref(), &test_cases))
    }

    /// Compute stratified metrics from per-example scores.
    ///
    /// Uses actual per-example F1/precision/recall to compute per-type metrics.
    /// This is the primary method when per-example scores are available.
    fn compute_stratified_metrics_from_scores(
        &self,
        dataset_data: &LoadedDataset,
        aggregate_metrics: &HashMap<String, f64>,
        per_example_scores: Option<&Vec<(Vec<Entity>, Vec<Entity>, String)>>,
    ) -> Option<StratifiedMetrics> {
        use crate::eval::ner_metrics::evaluate_entities;

        // If we have per-example scores, use them for proper stratification
        if let Some(per_example) = per_example_scores {
            // Compute per-type metrics from per-example scores
            let mut by_type_scores: HashMap<String, Vec<(f64, f64, f64)>> = HashMap::new(); // (f1, precision, recall)

            for (gold, predicted, _text) in per_example {
                // Compute per-example metrics
                let result = evaluate_entities(gold, predicted);
                let summary = result.summary();

                // Group by entity type
                for entity in gold {
                    let type_str = entity.entity_type.as_label().to_string();
                    by_type_scores.entry(type_str).or_default().push((
                        summary.strict_f1,
                        summary.strict_precision,
                        summary.strict_recall,
                    ));
                }
            }

            // Compute mean and CI for each type
            let mut by_entity_type = HashMap::new();
            for (type_str, scores) in by_type_scores {
                if scores.is_empty() {
                    continue;
                }

                let n = scores.len() as f64;
                let f1_mean = scores.iter().map(|(f1, _, _)| f1).sum::<f64>() / n;
                // Note: precision_mean and recall_mean computed but not used in CI (using F1 only for now)
                let _precision_mean = scores.iter().map(|(_, p, _)| p).sum::<f64>() / n;
                let _recall_mean = scores.iter().map(|(_, _, r)| r).sum::<f64>() / n;

                // Use sample variance (Bessel's correction: n-1) for unbiased estimate
                let f1_variance = if n > 1.0 {
                    scores
                        .iter()
                        .map(|(f1, _, _)| (f1 - f1_mean).powi(2))
                        .sum::<f64>()
                        / (n - 1.0)
                } else {
                    0.0
                };
                let f1_std_dev = f1_variance.sqrt();

                let z = DEFAULT_Z_SCORE_95;
                let margin = z * f1_std_dev / n.sqrt();

                by_entity_type.insert(
                    type_str,
                    MetricWithCI {
                        mean: f1_mean,
                        std_dev: f1_std_dev,
                        ci_95: ((f1_mean - margin).max(0.0), (f1_mean + margin).min(1.0)),
                        n: scores.len(),
                    },
                );
            }

            // Compute temporal stratification if metadata available
            let by_temporal_stratum = if let Some(ref temporal) = dataset_data.temporal_metadata {
                self.compute_temporal_stratification(per_example, temporal)
            } else {
                None
            };

            return Some(StratifiedMetrics {
                by_entity_type,
                by_temporal_stratum,
                by_surface_form: None, // Would need proper noun detection
                by_mention_char: None, // Would need mention analysis
            });
        }

        // Fallback to simplified version using aggregate metrics
        self.compute_stratified_metrics(dataset_data, aggregate_metrics)
    }

    /// Compute temporal stratification from per-example scores and temporal metadata.
    fn compute_temporal_stratification(
        &self,
        per_example_scores: &[(Vec<Entity>, Vec<Entity>, String)],
        temporal_metadata: &super::loader::TemporalMetadata,
    ) -> Option<HashMap<String, MetricWithCI>> {
        use crate::eval::ner_metrics::evaluate_entities;

        // If no temporal cutoff, can't stratify
        let cutoff = temporal_metadata.temporal_cutoff.as_ref()?;

        // Parse cutoff date (ISO 8601 format: YYYY-MM-DD)
        // For now, we use a simple heuristic: all examples are pre-cutoff
        // Future: would need entity creation dates or document timestamps to properly stratify
        let _cutoff_date = cutoff.split('T').next()?; // Remove time if present
                                                      // Note: cutoff date parsing removed - not used in current heuristic implementation

        // Group examples by temporal stratum
        let mut pre_cutoff_scores = Vec::new();
        let mut post_cutoff_scores = Vec::new();

        // Heuristic: Split examples in half based on order
        // First half treated as pre-cutoff, second half as post-cutoff
        // This approximates temporal drift when entity creation dates are unavailable
        let total = per_example_scores.len();
        let cutoff_index = total / 2;

        for (idx, (gold, predicted, _text)) in per_example_scores.iter().enumerate() {
            // Split data in half: first half = pre-cutoff, second half = post-cutoff
            // This is a heuristic approximation - proper temporal stratification would
            // require entity creation dates from entity linking or document timestamps
            let is_post_cutoff = idx >= cutoff_index;

            // Compute per-example metrics
            let result = evaluate_entities(gold, predicted);
            let summary = result.summary();

            if is_post_cutoff {
                post_cutoff_scores.push(summary.strict_f1);
            } else {
                pre_cutoff_scores.push(summary.strict_f1);
            }
        }

        // Compute metrics for each stratum
        let mut by_temporal = HashMap::new();

        if !pre_cutoff_scores.is_empty() {
            let n = pre_cutoff_scores.len() as f64;
            let mean = pre_cutoff_scores.iter().sum::<f64>() / n;
            // Use sample variance (Bessel's correction: n-1) for unbiased estimate
            let variance = if n > 1.0 {
                pre_cutoff_scores
                    .iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>()
                    / (n - 1.0)
            } else {
                0.0
            };
            let std_dev = variance.sqrt();
            let z = DEFAULT_Z_SCORE_95;
            let margin = z * std_dev / n.sqrt();

            by_temporal.insert(
                "pre_cutoff".to_string(),
                MetricWithCI {
                    mean,
                    std_dev,
                    ci_95: ((mean - margin).max(0.0), (mean + margin).min(1.0)),
                    n: pre_cutoff_scores.len(),
                },
            );
        }

        if !post_cutoff_scores.is_empty() {
            let n = post_cutoff_scores.len() as f64;
            let mean = post_cutoff_scores.iter().sum::<f64>() / n;
            // Use sample variance (Bessel's correction: n-1) for unbiased estimate
            let variance = if n > 1.0 {
                post_cutoff_scores
                    .iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>()
                    / (n - 1.0)
            } else {
                0.0
            };
            let std_dev = variance.sqrt();
            let z = DEFAULT_Z_SCORE_95;
            let margin = z * std_dev / n.sqrt();

            by_temporal.insert(
                "post_cutoff".to_string(),
                MetricWithCI {
                    mean,
                    std_dev,
                    ci_95: ((mean - margin).max(0.0), (mean + margin).min(1.0)),
                    n: post_cutoff_scores.len(),
                },
            );
        }

        if by_temporal.is_empty() {
            None
        } else {
            Some(by_temporal)
        }
    }

    /// Compute confidence intervals from per-example scores.
    fn compute_confidence_intervals_from_scores(
        &self,
        per_example_scores: &[(Vec<Entity>, Vec<Entity>, String)],
    ) -> Option<ConfidenceIntervals> {
        use crate::eval::ner_metrics::evaluate_entities;

        if per_example_scores.is_empty() {
            return None;
        }

        let mut f1_scores = Vec::new();
        let mut precision_scores = Vec::new();
        let mut recall_scores = Vec::new();

        for (gold, predicted, _text) in per_example_scores {
            let result = evaluate_entities(gold, predicted);
            let summary = result.summary();
            f1_scores.push(summary.strict_f1);
            precision_scores.push(summary.strict_precision);
            recall_scores.push(summary.strict_recall);
        }

        // Compute mean and std_dev
        let n = f1_scores.len() as f64;
        let f1_mean = f1_scores.iter().sum::<f64>() / n;
        let precision_mean = precision_scores.iter().sum::<f64>() / n;
        let recall_mean = recall_scores.iter().sum::<f64>() / n;

        // Use sample variance (Bessel's correction: n-1) for unbiased estimate
        let f1_variance = if n > 1.0 {
            f1_scores
                .iter()
                .map(|&x| (x - f1_mean).powi(2))
                .sum::<f64>()
                / (n - 1.0)
        } else {
            0.0
        };
        let precision_variance = if n > 1.0 {
            precision_scores
                .iter()
                .map(|&x| (x - precision_mean).powi(2))
                .sum::<f64>()
                / (n - 1.0)
        } else {
            0.0
        };
        let recall_variance = if n > 1.0 {
            recall_scores
                .iter()
                .map(|&x| (x - recall_mean).powi(2))
                .sum::<f64>()
                / (n - 1.0)
        } else {
            0.0
        };

        let f1_std_dev = f1_variance.sqrt();
        let precision_std_dev = precision_variance.sqrt();
        let recall_std_dev = recall_variance.sqrt();

        // 95% CI: mean ± 1.96 * std_dev / sqrt(n)
        let z = DEFAULT_Z_SCORE_95;
        let f1_margin = z * f1_std_dev / n.sqrt();
        let precision_margin = z * precision_std_dev / n.sqrt();
        let recall_margin = z * recall_std_dev / n.sqrt();

        Some(ConfidenceIntervals {
            f1_ci: (
                (f1_mean - f1_margin).max(0.0),
                (f1_mean + f1_margin).min(1.0),
            ),
            precision_ci: (
                (precision_mean - precision_margin).max(0.0),
                (precision_mean + precision_margin).min(1.0),
            ),
            recall_ci: (
                (recall_mean - recall_margin).max(0.0),
                (recall_mean + recall_margin).min(1.0),
            ),
        })
    }

    /// Compute stratified metrics across multiple dimensions.
    ///
    /// # Limitations
    ///
    /// **Current Implementation**: This is a simplified version that uses aggregate metrics
    /// as placeholders for all entity types. A full implementation would require:
    /// - Per-example predictions for each entity type
    /// - Per-type F1, precision, recall computation
    /// - Proper confidence intervals from per-type variance
    ///
    /// **Why Simplified**: Computing per-type metrics requires re-running inference with
    /// per-example tracking, which is expensive. The current implementation provides
    /// entity type distribution but uses aggregate metrics as placeholders.
    ///
    /// # Future Improvements
    ///
    /// To implement proper stratification:
    /// 1. Collect per-example predictions during main evaluation
    /// 2. Group predictions by entity type
    /// 3. Compute per-type metrics from grouped predictions
    /// 4. Calculate proper confidence intervals from per-type variance
    pub(crate) fn compute_stratified_metrics(
        &self,
        dataset_data: &LoadedDataset,
        metrics: &HashMap<String, f64>,
    ) -> Option<StratifiedMetrics> {
        // Extract entity types from dataset (single pass)
        let mut type_counts: HashMap<String, usize> = HashMap::new();
        for sentence in &dataset_data.sentences {
            for entity in sentence.entities() {
                let type_str = entity.entity_type.as_label().to_string();
                *type_counts.entry(type_str).or_insert(0) += 1;
            }
        }

        if type_counts.is_empty() {
            return None;
        }

        // Build per-type metrics (simplified - uses aggregate metrics as placeholder)
        // NOTE: This is a placeholder implementation. All entity types show the same
        // aggregate metrics. For proper stratification, we'd need per-example predictions
        // grouped by entity type.
        let mut by_entity_type = HashMap::new();
        let aggregate_f1 = metrics.get("f1").copied().unwrap_or(0.0);
        for (type_str, count) in type_counts {
            // Use aggregate metrics as placeholder (proper implementation needs per-type scores)
            // TODO: Compute actual per-type metrics from per-example predictions
            let mean = aggregate_f1; // All types use aggregate F1 (placeholder)
            let std_dev = DEFAULT_PLACEHOLDER_STD_DEV; // Placeholder - should compute from per-type scores
            let z = DEFAULT_Z_SCORE_95;
            let margin = z * std_dev;
            by_entity_type.insert(
                type_str,
                MetricWithCI {
                    mean,
                    std_dev,
                    ci_95: ((mean - margin).max(0.0), (mean + margin).min(1.0)),
                    n: count, // Use actual count from dataset
                },
            );
        }

        Some(StratifiedMetrics {
            by_entity_type,
            by_temporal_stratum: None, // Would need temporal metadata
            by_surface_form: None,     // Would need proper noun detection
            by_mention_char: None,     // Would need mention analysis
        })
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
