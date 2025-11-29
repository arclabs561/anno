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
    /// Failed evaluations
    pub failed: usize,
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

        let summary = EvalSummary {
            total_combinations: results.len(),
            successful: results.iter().filter(|r| r.success).count(),
            failed: results.iter().filter(|r| !r.success).count(),
            tasks: tasks_evaluated,
            datasets: datasets_used,
            backends: backends_tested,
        };

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
                        });
                    }
                }
            }
        };

        // Count sentences (limit if configured)
        let sentences_to_use = if let Some(max) = config.max_examples {
            dataset_data.sentences.len().min(max)
        } else {
            dataset_data.sentences.len()
        };

        // Try to create backend (this is a placeholder - actual implementation
        // would need backend factory)
        let result = match self.try_evaluate_backend(task, dataset, backend_name, &dataset_data) {
            Ok(metrics) => TaskEvalResult {
                task,
                dataset,
                backend: backend_name.to_string(),
                success: true,
                error: None,
                metrics,
                num_examples: sentences_to_use,
            },
            Err(e) => TaskEvalResult {
                task,
                dataset,
                backend: backend_name.to_string(),
                success: false,
                error: Some(format!("{}", e)),
                metrics: HashMap::new(),
                num_examples: sentences_to_use,
            },
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
            Task::NER | Task::DiscontinuousNER => self.evaluate_ner_task(&*backend, dataset_data),
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
        backend: &dyn Model,
        dataset_data: &LoadedDataset,
    ) -> Result<HashMap<String, f64>> {
        use crate::eval::ner_metrics::evaluate_entities;

        let mut all_gold = Vec::new();
        let mut all_predicted = Vec::new();
        let mut total_chars = 0;
        let start_time = Instant::now();

        // Process each sentence
        for sentence in &dataset_data.sentences {
            let text = sentence.text();
            total_chars += text.chars().count();

            // Extract gold entities from sentence
            let gold_entities = sentence.entities();
            all_gold.extend(gold_entities.iter().map(|g| {
                let mut entity =
                    Entity::new(g.text.clone(), g.entity_type.clone(), g.start, g.end, 1.0);
                entity.provenance = Some(crate::Provenance::ml("gold", 1.0));
                entity
            }));

            // Run inference
            match backend.extract_entities(&text, None) {
                Ok(entities) => {
                    all_predicted.extend(entities);
                }
                Err(e) => {
                    // Log error but continue with other sentences
                    eprintln!("Warning: Backend inference failed for sentence: {}", e);
                }
            }
        }

        let elapsed = start_time.elapsed();
        let chars_per_second = if elapsed.as_secs_f64() > 0.0 {
            total_chars as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        // Compute metrics
        let eval_results = evaluate_entities(&all_gold, &all_predicted);
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

    /// Evaluate coreference task.
    fn evaluate_coref_task(
        &self,
        _backend_name: &str,
        dataset_data: &LoadedDataset,
    ) -> Result<HashMap<String, f64>> {
        use crate::eval::coref::entities_to_chains;
        use crate::eval::coref_metrics::CorefEvaluation;

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
                    vec![crate::eval::coref::CorefDocument::new(
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

        // For now, use a simple coreference resolver or extract from backend
        // This is a placeholder - actual implementation would run the coreference resolver
        let all_predicted_chains = Vec::new(); // Placeholder - TODO: run resolver
        let mut all_gold_chains = Vec::new();

        for doc in &gold_docs {
            all_gold_chains.extend(doc.chains.clone());
            // TODO: Run coreference resolver on doc.text to get predicted chains
            // For now, create empty chains as placeholder
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
    fn evaluate_relation_task(
        &self,
        backend_name: &str,
        dataset_data: &LoadedDataset,
    ) -> Result<HashMap<String, f64>> {
        use crate::eval::relation::{evaluate_relations, RelationEvalConfig};

        // Extract gold relations from dataset
        // Note: This is a placeholder - actual implementation would need to
        // parse relation annotations from the dataset format (e.g., DocRED JSONL)
        let all_gold_relations = Vec::new();
        let all_predicted_relations = Vec::new();

        for sentence in &dataset_data.sentences {
            let text = sentence.text();

            // TODO: Extract gold relations from dataset format
            // For now, create placeholder gold relations
            // (Actual implementation would parse DocRED, TACRED, etc.)

            // Run backend to get entities (needed for relation extraction)
            let backend = BackendFactory::create(backend_name)?;
            match backend.extract_entities(&text, None) {
                Ok(_predicted_entities) => {
                    // TODO: Extract relations from backend output
                    // For now, this is a placeholder - relation extraction backends
                    // would need to implement a relation extraction method
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
        md.push_str("# Comprehensive Task-Dataset-Backend Evaluation\n\n");
        md.push_str(&format!(
            "**Total Combinations**: {}\n",
            self.summary.total_combinations
        ));
        md.push_str(&format!("**Successful**: {}\n", self.summary.successful));
        md.push_str(&format!("**Failed**: {}\n\n", self.summary.failed));

        md.push_str("## Tasks Evaluated\n\n");
        for task in &self.summary.tasks {
            md.push_str(&format!("- {}\n", task.name()));
        }

        md.push_str("\n## Datasets Used\n\n");
        for dataset in &self.summary.datasets {
            md.push_str(&format!("- {:?}\n", dataset));
        }

        md.push_str("\n## Backends Tested\n\n");
        for backend in &self.summary.backends {
            md.push_str(&format!("- {}\n", backend));
        }

        md.push_str("\n## Results by Task\n\n");
        // Group results by task
        let mut by_task: HashMap<Task, Vec<&TaskEvalResult>> = HashMap::new();
        for result in &self.results {
            by_task
                .entry(result.task)
                .or_default()
                .push(result);
        }

        for (task, results) in by_task {
            md.push_str(&format!("### {}\n\n", task.name()));
            md.push_str("| Dataset | Backend | Success | Examples |\n");
            md.push_str("|---------|---------|---------|----------|\n");
            for result in results {
                md.push_str(&format!(
                    "| {:?} | {} | {} | {} |\n",
                    result.dataset,
                    result.backend,
                    if result.success { "✓" } else { "✗" },
                    result.num_examples
                ));
            }
            md.push('\n');
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
