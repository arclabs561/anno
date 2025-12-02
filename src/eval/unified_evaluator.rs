//! Unified Evaluation System
//!
//! Provides a single, consistent API for all evaluation types, replacing
//! the multiple entry points (TaskEvaluator, EvalHarness, direct functions, etc.)
//!
//! # Design Goals
//!
//! - **Single Entry Point**: One API for all evaluation needs
//! - **Unified Results**: Consistent result types across all evaluations
//! - **Composable**: Easy to combine standard + bias + calibration
//! - **Type-Safe**: Compile-time checking for configurations
//!
//! # Example
//!
//! ```rust
//! use anno::eval::unified_evaluator::EvalSystem;
//! use anno::eval::task_mapping::Task;
//!
//! let results = EvalSystem::new()
//!     .with_tasks(vec![Task::NER])
//!     .with_datasets(vec![])  // All suitable datasets
//!     .with_backends(vec!["gliner2".to_string()])
//!     .with_bias_analysis(true)
//!     .with_calibration(true)
//!     .run()?;
//!
//! println!("Standard F1: {:.1}%", results.standard.f1 * 100.0);
//! if let Some(bias) = &results.bias {
//!     println!("Bias parity gap: {:.1}%", bias.demographic.as_ref()
//!         .map(|d| d.max_gap * 100.0).unwrap_or(0.0));
//! }
//! ```

use crate::{Model, Result};
use serde::{Deserialize, Serialize};
#[cfg(feature = "eval-advanced")]
use std::collections::HashMap;

#[cfg(feature = "eval-advanced")]
use crate::eval::loader::DatasetId;
#[cfg(feature = "eval-advanced")]
use crate::eval::task_evaluator::{TaskEvalConfig, TaskEvaluator};
#[cfg(feature = "eval-advanced")]
use crate::eval::task_mapping::Task;

#[cfg(feature = "eval-bias")]
use crate::eval::bias_config::BiasDatasetConfig;
#[cfg(feature = "eval-bias")]
use crate::eval::coref_resolver::SimpleCorefResolver;
#[cfg(feature = "eval-bias")]
use crate::eval::demographic_bias::{create_diverse_name_dataset, DemographicBiasEvaluator};
#[cfg(feature = "eval-bias")]
use crate::eval::gender_bias::{create_winobias_templates, GenderBiasEvaluator};
#[cfg(feature = "eval-bias")]
use crate::eval::length_bias::{create_length_varied_dataset, EntityLengthEvaluator};
#[cfg(feature = "eval-bias")]
use crate::eval::temporal_bias::{create_temporal_name_dataset, TemporalBiasEvaluator};

#[cfg(feature = "eval-advanced")]
use crate::eval::backend_name::BackendName;

// =============================================================================
// Unified Results
// =============================================================================

/// Unified evaluation results combining all evaluation types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedEvalResults {
    /// Standard task evaluation results (NER, Coref, etc.)
    #[cfg(feature = "eval-advanced")]
    pub standard: Option<StandardEvalResults>,

    /// Bias evaluation results
    #[cfg(feature = "eval-bias")]
    pub bias: Option<BiasEvalResults>,

    /// Calibration results (if enabled)
    #[cfg(feature = "eval-advanced")]
    pub calibration: Option<CalibrationEvalResults>,

    /// Data quality results (if enabled)
    #[cfg(feature = "eval-advanced")]
    pub data_quality: Option<DataQualityEvalResults>,

    /// Warnings and notes
    pub warnings: Vec<String>,

    /// Evaluation metadata
    pub metadata: EvalMetadata,
}

/// Standard task evaluation results.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg(feature = "eval-advanced")]
pub struct StandardEvalResults {
    /// Overall F1 score
    pub f1: f64,
    /// Precision
    pub precision: f64,
    /// Recall
    pub recall: f64,
    /// Per-task results
    pub per_task: HashMap<String, TaskResults>,
    /// Per-dataset results
    pub per_dataset: HashMap<String, DatasetResults>,
    /// Per-backend results
    pub per_backend: HashMap<String, BackendResults>,
}

/// Task-specific results.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg(feature = "eval-advanced")]
pub struct TaskResults {
    pub task: String,
    pub f1: f64,
    pub precision: f64,
    pub recall: f64,
    pub num_examples: usize,
}

/// Dataset-specific results.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg(feature = "eval-advanced")]
pub struct DatasetResults {
    pub dataset: String,
    pub f1: f64,
    pub precision: f64,
    pub recall: f64,
    pub num_examples: usize,
}

/// Backend-specific results.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg(feature = "eval-advanced")]
pub struct BackendResults {
    pub backend: String,
    pub f1: f64,
    pub precision: f64,
    pub recall: f64,
    pub num_examples: usize,
}

/// Bias evaluation results.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg(feature = "eval-bias")]
pub struct BiasEvalResults {
    /// Gender bias results
    pub gender: Option<GenderBiasSummary>,
    /// Demographic bias results
    pub demographic: Option<DemographicBiasSummary>,
    /// Temporal bias results
    pub temporal: Option<TemporalBiasSummary>,
    /// Length bias results
    pub length: Option<LengthBiasSummary>,
}

/// Gender bias summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg(feature = "eval-bias")]
pub struct GenderBiasSummary {
    pub bias_gap: f64,
    pub pro_stereotype_accuracy: f64,
    pub anti_stereotype_accuracy: f64,
}

/// Demographic bias summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg(feature = "eval-bias")]
pub struct DemographicBiasSummary {
    pub ethnicity_parity_gap: f64,
    pub script_bias_gap: f64,
    pub overall_recognition_rate: f64,
}

/// Temporal bias summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg(feature = "eval-bias")]
pub struct TemporalBiasSummary {
    pub historical_modern_gap: f64,
    pub historical_rate: f64,
    pub modern_rate: f64,
}

/// Length bias summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg(feature = "eval-bias")]
pub struct LengthBiasSummary {
    pub short_vs_long_gap: f64,
    pub short_entity_f1: f64,
    pub long_entity_f1: f64,
}

/// Calibration results.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg(feature = "eval-advanced")]
pub struct CalibrationEvalResults {
    pub ece: f64,
    pub mce: f64,
    pub brier_score: f64,
}

/// Data quality results.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg(feature = "eval-advanced")]
pub struct DataQualityEvalResults {
    pub leakage_detected: bool,
    pub redundancy_rate: f64,
    pub ambiguous_count: usize,
}

/// Evaluation metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalMetadata {
    pub timestamp: String,
    pub model_name: Option<String>,
    pub total_duration_ms: Option<f64>,
    pub num_examples: usize,
}

// =============================================================================
// Unified Evaluator
// =============================================================================

/// Unified evaluation system - single entry point for all evaluations.
pub struct EvalSystem {
    #[cfg(feature = "eval-advanced")]
    tasks: Vec<Task>,
    #[cfg(feature = "eval-advanced")]
    datasets: Vec<DatasetId>,
    #[cfg(feature = "eval-advanced")]
    backends: Vec<String>,
    #[cfg(feature = "eval-advanced")]
    max_examples: Option<usize>,
    #[cfg(feature = "eval-advanced")]
    seed: Option<u64>,

    #[cfg(feature = "eval-bias")]
    include_bias: bool,
    #[cfg(feature = "eval-bias")]
    bias_config: Option<BiasDatasetConfig>,

    #[cfg(feature = "eval-advanced")]
    include_calibration: bool,
    #[cfg(feature = "eval-advanced")]
    include_data_quality: bool,

    model: Option<Box<dyn Model>>,
    model_name: Option<String>,

    /// Coreference resolver for coreference evaluation tasks
    /// Uses Arc to allow sharing across multiple evaluation calls
    coref_resolver: Option<std::sync::Arc<dyn crate::eval::coref_resolver::CoreferenceResolver>>,
}

impl EvalSystem {
    /// Create a new unified evaluation system.
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "eval-advanced")]
            tasks: vec![],
            #[cfg(feature = "eval-advanced")]
            datasets: vec![],
            #[cfg(feature = "eval-advanced")]
            backends: vec![],
            #[cfg(feature = "eval-advanced")]
            max_examples: None,
            #[cfg(feature = "eval-advanced")]
            seed: Some(42),

            #[cfg(feature = "eval-bias")]
            include_bias: false,
            #[cfg(feature = "eval-bias")]
            bias_config: None,

            #[cfg(feature = "eval-advanced")]
            include_calibration: false,
            #[cfg(feature = "eval-advanced")]
            include_data_quality: false,

            model: None,
            model_name: None,
            coref_resolver: None,
        }
    }

    /// Set tasks to evaluate.
    #[cfg(feature = "eval-advanced")]
    pub fn with_tasks(mut self, tasks: Vec<Task>) -> Self {
        self.tasks = tasks;
        self
    }

    /// Set datasets to use.
    #[cfg(feature = "eval-advanced")]
    pub fn with_datasets(mut self, datasets: Vec<DatasetId>) -> Self {
        self.datasets = datasets;
        self
    }

    /// Set backends to test.
    #[cfg(feature = "eval-advanced")]
    pub fn with_backends(mut self, backends: Vec<String>) -> Self {
        self.backends = backends;
        self
    }

    /// Set backends using type-safe BackendName enum.
    #[cfg(feature = "eval-advanced")]
    pub fn with_backend_names(mut self, backends: Vec<BackendName>) -> Self {
        self.backends = backends
            .into_iter()
            .map(|b| b.as_str().to_string())
            .collect();
        self
    }

    /// Set maximum examples per dataset.
    ///
    /// Pass `None` to remove limit (evaluate all examples).
    #[cfg(feature = "eval-advanced")]
    pub fn with_max_examples(mut self, max: Option<usize>) -> Self {
        self.max_examples = max;
        self
    }

    /// Add a task to evaluate.
    #[cfg(feature = "eval-advanced")]
    pub fn add_task(mut self, task: Task) -> Self {
        if !self.tasks.contains(&task) {
            self.tasks.push(task);
        }
        self
    }

    /// Add a dataset to use.
    #[cfg(feature = "eval-advanced")]
    pub fn add_dataset(mut self, dataset: DatasetId) -> Self {
        if !self.datasets.contains(&dataset) {
            self.datasets.push(dataset);
        }
        self
    }

    /// Add a backend to test.
    #[cfg(feature = "eval-advanced")]
    pub fn add_backend(mut self, backend: String) -> Self {
        if !self.backends.contains(&backend) {
            self.backends.push(backend);
        }
        self
    }

    /// Add a backend using type-safe BackendName enum.
    #[cfg(feature = "eval-advanced")]
    pub fn add_backend_name(mut self, backend: BackendName) -> Self {
        let backend_str = backend.as_str().to_string();
        if !self.backends.contains(&backend_str) {
            self.backends.push(backend_str);
        }
        self
    }

    /// Set random seed.
    #[cfg(feature = "eval-advanced")]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Enable bias analysis.
    #[cfg(feature = "eval-bias")]
    pub fn with_bias_analysis(mut self, enable: bool) -> Self {
        self.include_bias = enable;
        if enable && self.bias_config.is_none() {
            self.bias_config = Some(
                BiasDatasetConfig::default()
                    .with_frequency_weighting()
                    .with_validation(),
            );
        }
        self
    }

    /// Set bias evaluation configuration.
    #[cfg(feature = "eval-bias")]
    pub fn with_bias_config(mut self, config: BiasDatasetConfig) -> Self {
        self.bias_config = Some(config);
        self.include_bias = true;
        self
    }

    /// Enable calibration analysis.
    #[cfg(feature = "eval-advanced")]
    pub fn with_calibration(mut self, enable: bool) -> Self {
        self.include_calibration = enable;
        self
    }

    /// Enable data quality checks.
    #[cfg(feature = "eval-advanced")]
    pub fn with_data_quality(mut self, enable: bool) -> Self {
        self.include_data_quality = enable;
        self
    }

    /// Set model to evaluate (for bias/calibration that need model instance).
    pub fn with_model(mut self, model: Box<dyn Model>, name: Option<String>) -> Self {
        self.model = Some(model);
        self.model_name = name;
        self
    }

    /// Set coreference resolver to evaluate.
    ///
    /// This allows evaluating coreference resolvers (e.g., `TrainedBoxCorefResolver` from matryoshka-box)
    /// using anno's evaluation infrastructure.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use matryoshka_box_inference::trained_resolver::TrainedBoxCorefResolver;
    /// use anno::eval::unified_evaluator::EvalSystem;
    /// use anno::eval::task_mapping::Task;
    ///
    /// let resolver = TrainedBoxCorefResolver::new(trained_boxes, config);
    /// let results = EvalSystem::new()
    ///     .with_coref_resolver(Box::new(resolver))
    ///     .with_tasks(vec![Task::Coreference { metrics: vec![] }])
    ///     .run()?;
    /// ```
    pub fn with_coref_resolver(
        mut self,
        resolver: Box<dyn crate::eval::coref_resolver::CoreferenceResolver>,
    ) -> Self {
        self.coref_resolver = Some(std::sync::Arc::from(resolver));
        self
    }

    /// Run all enabled evaluations.
    pub fn run(self) -> Result<UnifiedEvalResults> {
        use std::time::Instant;

        let start = Instant::now();
        let mut warnings = Vec::new();

        // Run standard evaluation
        #[cfg(feature = "eval-advanced")]
        let standard_result = self.run_standard_evaluation(&mut warnings)?;

        // Run bias evaluation
        #[cfg(feature = "eval-bias")]
        let bias = if self.include_bias {
            match self.run_bias_evaluation(&mut warnings) {
                Ok(results) => Some(results),
                Err(e) => {
                    warnings.push(format!("Bias evaluation failed: {}", e));
                    None
                }
            }
        } else {
            None
        };

        // Run calibration (if model provided)
        #[cfg(feature = "eval-advanced")]
        let calibration = if self.include_calibration && self.model.is_some() {
            match self.run_calibration(&mut warnings) {
                Ok(results) => Some(results),
                Err(e) => {
                    warnings.push(format!("Calibration evaluation failed: {}", e));
                    None
                }
            }
        } else {
            None
        };

        // Run data quality checks
        #[cfg(feature = "eval-advanced")]
        let data_quality = if self.include_data_quality {
            match self.run_data_quality(&mut warnings) {
                Ok(results) => Some(results),
                Err(e) => {
                    warnings.push(format!("Data quality checks failed: {}", e));
                    None
                }
            }
        } else {
            None
        };

        let duration = start.elapsed();

        #[cfg(feature = "eval-advanced")]
        let num_examples = standard_result
            .as_ref()
            .map(|s| s.per_task.values().map(|t| t.num_examples).sum::<usize>())
            .unwrap_or(0);
        #[cfg(not(feature = "eval-advanced"))]
        let num_examples = 0;

        Ok(UnifiedEvalResults {
            #[cfg(feature = "eval-advanced")]
            standard: standard_result,
            #[cfg(feature = "eval-bias")]
            bias,
            #[cfg(feature = "eval-advanced")]
            calibration,
            #[cfg(feature = "eval-advanced")]
            data_quality,
            warnings,
            metadata: EvalMetadata {
                timestamp: chrono::Utc::now().to_rfc3339(),
                model_name: self.model_name.clone(),
                total_duration_ms: Some(duration.as_secs_f64() * 1000.0),
                num_examples,
            },
        })
    }

    /// Run standard task evaluation.
    ///
    /// **Empty vector semantics**:
    /// - Empty `tasks` → uses all available tasks
    /// - Empty `datasets` → uses all suitable datasets for each task
    /// - Empty `backends` → uses all compatible backends for each task
    #[cfg(feature = "eval-advanced")]
    fn run_standard_evaluation(
        &self,
        _warnings: &mut Vec<String>,
    ) -> Result<Option<StandardEvalResults>> {
        // Note: get_task_datasets and get_task_backends are available but not used here
        // as we rely on TaskEvaluator's internal logic for dataset/backend selection

        // Empty tasks = use all tasks
        let tasks = if self.tasks.is_empty() {
            Task::all().to_vec()
        } else {
            self.tasks.clone()
        };

        if tasks.is_empty() {
            return Ok(None);
        }

        let evaluator = TaskEvaluator::new().map_err(|e| {
            crate::Error::InvalidInput(format!("Failed to create TaskEvaluator: {}", e))
        })?;

        let config = TaskEvalConfig {
            tasks: tasks.clone(),
            datasets: self.datasets.clone(),
            backends: self.backends.clone(),
            max_examples: self.max_examples,
            seed: self.seed,
            require_cached: false,
            relation_threshold: 0.5,
            robustness: false,
            compute_familiarity: true,
            temporal_stratification: false,
            confidence_intervals: true,
            custom_coref_resolver: self.coref_resolver.clone(),
        };

        let comprehensive_results = evaluator.evaluate_all(config)?;

        // Aggregate results
        let mut per_task: HashMap<String, TaskResults> = HashMap::new();
        let mut per_dataset: HashMap<String, DatasetResults> = HashMap::new();
        let mut per_backend: HashMap<String, BackendResults> = HashMap::new();

        let mut total_f1_weighted = 0.0;
        let mut total_precision_weighted = 0.0;
        let mut total_recall_weighted = 0.0;
        let mut total_examples = 0;

        for result in &comprehensive_results.results {
            if !result.success {
                continue;
            }

            let f1 = result.metrics.get("f1").copied().unwrap_or(0.0);
            let precision = result.metrics.get("precision").copied().unwrap_or(0.0);
            let recall = result.metrics.get("recall").copied().unwrap_or(0.0);
            let examples = result.num_examples;

            // Weight by number of examples for overall average
            total_f1_weighted += f1 * examples as f64;
            total_precision_weighted += precision * examples as f64;
            total_recall_weighted += recall * examples as f64;
            total_examples += examples;

            // Per-task aggregation (weighted by number of examples)
            let task_key = format!("{:?}", result.task);
            per_task
                .entry(task_key.clone())
                .and_modify(|t| {
                    // Weighted average: (old_f1 * old_count + new_f1 * new_count) / total_count
                    let old_count = t.num_examples as f64;
                    let new_count = result.num_examples as f64;
                    let total_count = old_count + new_count;

                    if total_count > 0.0 {
                        t.f1 = (t.f1 * old_count + f1 * new_count) / total_count;
                        t.precision =
                            (t.precision * old_count + precision * new_count) / total_count;
                        t.recall = (t.recall * old_count + recall * new_count) / total_count;
                    }
                    t.num_examples += result.num_examples;
                })
                .or_insert_with(|| TaskResults {
                    task: task_key,
                    f1,
                    precision,
                    recall,
                    num_examples: result.num_examples,
                });

            // Per-dataset aggregation (weighted by number of examples)
            let dataset_key = format!("{:?}", result.dataset);
            per_dataset
                .entry(dataset_key.clone())
                .and_modify(|d| {
                    let old_count = d.num_examples as f64;
                    let new_count = result.num_examples as f64;
                    let total_count = old_count + new_count;

                    if total_count > 0.0 {
                        d.f1 = (d.f1 * old_count + f1 * new_count) / total_count;
                        d.precision =
                            (d.precision * old_count + precision * new_count) / total_count;
                        d.recall = (d.recall * old_count + recall * new_count) / total_count;
                    }
                    d.num_examples += result.num_examples;
                })
                .or_insert_with(|| DatasetResults {
                    dataset: dataset_key,
                    f1,
                    precision,
                    recall,
                    num_examples: result.num_examples,
                });

            // Per-backend aggregation (weighted by number of examples)
            per_backend
                .entry(result.backend.clone())
                .and_modify(|b| {
                    let old_count = b.num_examples as f64;
                    let new_count = result.num_examples as f64;
                    let total_count = old_count + new_count;

                    if total_count > 0.0 {
                        b.f1 = (b.f1 * old_count + f1 * new_count) / total_count;
                        b.precision =
                            (b.precision * old_count + precision * new_count) / total_count;
                        b.recall = (b.recall * old_count + recall * new_count) / total_count;
                    }
                    b.num_examples += result.num_examples;
                })
                .or_insert_with(|| BackendResults {
                    backend: result.backend.clone(),
                    f1,
                    precision,
                    recall,
                    num_examples: result.num_examples,
                });
        }

        // Weighted average across all results
        let avg_f1 = if total_examples > 0 {
            total_f1_weighted / total_examples as f64
        } else {
            0.0
        };
        let avg_precision = if total_examples > 0 {
            total_precision_weighted / total_examples as f64
        } else {
            0.0
        };
        let avg_recall = if total_examples > 0 {
            total_recall_weighted / total_examples as f64
        } else {
            0.0
        };

        Ok(Some(StandardEvalResults {
            f1: avg_f1,
            precision: avg_precision,
            recall: avg_recall,
            per_task,
            per_dataset,
            per_backend,
        }))
    }

    /// Run bias evaluation.
    #[cfg(feature = "eval-bias")]
    fn run_bias_evaluation(&self, warnings: &mut Vec<String>) -> Result<BiasEvalResults> {
        let model = self.model.as_deref().ok_or_else(|| {
            crate::Error::InvalidInput(
                "Bias evaluation requires a model instance. Use with_model()".to_string(),
            )
        })?;

        let config = self.bias_config.clone().unwrap_or_else(|| {
            BiasDatasetConfig::default()
                .with_frequency_weighting()
                .with_validation()
        });

        let mut gender: Option<GenderBiasSummary> = None;
        let mut demographic: Option<DemographicBiasSummary> = None;
        let mut temporal: Option<TemporalBiasSummary> = None;
        let mut length: Option<LengthBiasSummary> = None;

        // Gender bias (coreference)
        // Note: Gender bias requires CoreferenceResolver, not Model.
        // If the provided model implements CoreferenceResolver, we could use it,
        // but for now we use a default resolver. This is a known limitation.
        let resolver = SimpleCorefResolver::default();
        let templates = create_winobias_templates();
        let evaluator = GenderBiasEvaluator::new(true);
        let gender_results = evaluator.evaluate_resolver(&resolver, &templates);
        gender = Some(GenderBiasSummary {
            bias_gap: gender_results.bias_gap,
            pro_stereotype_accuracy: gender_results.pro_stereotype_accuracy,
            anti_stereotype_accuracy: gender_results.anti_stereotype_accuracy,
        });

        // Demographic bias
        let names = create_diverse_name_dataset();
        let demo_evaluator = DemographicBiasEvaluator::with_config(true, config.clone());
        let demo_results = demo_evaluator.evaluate_ner(model, &names);
        demographic = Some(DemographicBiasSummary {
            ethnicity_parity_gap: demo_results.ethnicity_parity_gap,
            script_bias_gap: demo_results.script_bias_gap,
            overall_recognition_rate: demo_results.overall_recognition_rate,
        });

        // Temporal bias
        let temporal_names = create_temporal_name_dataset();
        let temporal_evaluator = TemporalBiasEvaluator::new(true);
        let temporal_results = temporal_evaluator.evaluate(model, &temporal_names);
        temporal = Some(TemporalBiasSummary {
            historical_modern_gap: temporal_results.historical_modern_gap,
            historical_rate: temporal_results.historical_rate,
            modern_rate: temporal_results.modern_rate,
        });

        // Length bias
        let length_examples = create_length_varied_dataset();
        let length_evaluator = EntityLengthEvaluator::new(true);
        let length_results = length_evaluator.evaluate(model, &length_examples);
        length = Some(LengthBiasSummary {
            short_vs_long_gap: length_results.short_vs_long_gap,
            short_entity_f1: length_results
                .by_word_bucket
                .get("SingleWord")
                .copied()
                .unwrap_or(0.0),
            long_entity_f1: length_results
                .by_word_bucket
                .get("FourPlusWords")
                .copied()
                .unwrap_or(0.0),
        });

        Ok(BiasEvalResults {
            gender,
            demographic,
            temporal,
            length,
        })
    }

    /// Run calibration analysis.
    #[cfg(feature = "eval-advanced")]
    fn run_calibration(&self, _warnings: &mut Vec<String>) -> Result<CalibrationEvalResults> {
        // TODO: Implement calibration evaluation
        // For now, return placeholder
        Ok(CalibrationEvalResults {
            ece: 0.0,
            mce: 0.0,
            brier_score: 0.0,
        })
    }

    /// Run data quality checks.
    #[cfg(feature = "eval-advanced")]
    fn run_data_quality(&self, _warnings: &mut Vec<String>) -> Result<DataQualityEvalResults> {
        // TODO: Implement data quality evaluation
        // For now, return placeholder
        Ok(DataQualityEvalResults {
            leakage_detected: false,
            redundancy_rate: 0.0,
            ambiguous_count: 0,
        })
    }
}

impl Default for EvalSystem {
    fn default() -> Self {
        Self::new()
    }
}
