//! Configuration Builder Pattern
//!
//! Provides builder pattern for evaluation configurations to reduce complexity
//! and improve usability.

#[cfg(feature = "eval-bias")]
use crate::eval::bias_config::BiasDatasetConfig;
#[cfg(feature = "eval-advanced")]
use crate::eval::loader::DatasetId;
#[cfg(feature = "eval-advanced")]
use crate::eval::task_mapping::Task;

/// Builder for TaskEvalConfig.
#[derive(Debug, Clone)]
#[cfg(feature = "eval-advanced")]
pub struct TaskEvalConfigBuilder {
    tasks: Vec<Task>,
    datasets: Vec<DatasetId>,
    backends: Vec<String>,
    max_examples: Option<usize>,
    seed: Option<u64>,
    require_cached: bool,
    relation_threshold: f32,
    robustness: bool,
    compute_familiarity: bool,
    temporal_stratification: bool,
    confidence_intervals: bool,
}

#[cfg(feature = "eval-advanced")]
impl TaskEvalConfigBuilder {
    /// Create a new builder with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set tasks to evaluate.
    pub fn with_tasks(mut self, tasks: Vec<Task>) -> Self {
        self.tasks = tasks;
        self
    }

    /// Add a task.
    pub fn add_task(mut self, task: Task) -> Self {
        self.tasks.push(task);
        self
    }

    /// Set datasets to use.
    pub fn with_datasets(mut self, datasets: Vec<DatasetId>) -> Self {
        self.datasets = datasets;
        self
    }

    /// Add a dataset.
    pub fn add_dataset(mut self, dataset: DatasetId) -> Self {
        self.datasets.push(dataset);
        self
    }

    /// Set backends to test.
    pub fn with_backends(mut self, backends: Vec<String>) -> Self {
        self.backends = backends;
        self
    }

    /// Add a backend.
    pub fn add_backend(mut self, backend: String) -> Self {
        self.backends.push(backend);
        self
    }

    /// Set maximum examples per dataset.
    ///
    /// If `max` is 0, this is treated as unlimited (None).
    /// Otherwise, limits evaluation to the specified number of examples per dataset.
    pub fn with_max_examples(mut self, max: usize) -> Self {
        if max > 0 {
            self.max_examples = Some(max);
        } else {
            self.max_examples = None; // 0 means unlimited
        }
        self
    }

    /// Set random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Require datasets to be cached (skip downloads).
    pub fn require_cached(mut self, require: bool) -> Self {
        self.require_cached = require;
        self
    }

    /// Set relation extraction threshold.
    pub fn with_relation_threshold(mut self, threshold: f32) -> Self {
        self.relation_threshold = threshold;
        self
    }

    /// Enable robustness testing.
    pub fn with_robustness(mut self, enable: bool) -> Self {
        self.robustness = enable;
        self
    }

    /// Enable familiarity computation.
    pub fn with_familiarity(mut self, enable: bool) -> Self {
        self.compute_familiarity = enable;
        self
    }

    /// Enable temporal stratification.
    pub fn with_temporal_stratification(mut self, enable: bool) -> Self {
        self.temporal_stratification = enable;
        self
    }

    /// Enable confidence intervals.
    pub fn with_confidence_intervals(mut self, enable: bool) -> Self {
        self.confidence_intervals = enable;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> crate::eval::task_evaluator::TaskEvalConfig {
        crate::eval::task_evaluator::TaskEvalConfig {
            tasks: self.tasks,
            datasets: self.datasets,
            backends: self.backends,
            max_examples: self.max_examples,
            seed: self.seed,
            require_cached: self.require_cached,
            relation_threshold: self.relation_threshold,
            robustness: self.robustness,
            compute_familiarity: self.compute_familiarity,
            temporal_stratification: self.temporal_stratification,
            confidence_intervals: self.confidence_intervals,
            custom_coref_resolver: None,
        }
    }
}

#[cfg(feature = "eval-advanced")]
impl Default for TaskEvalConfigBuilder {
    fn default() -> Self {
        Self {
            tasks: vec![],
            datasets: vec![],
            backends: vec![],
            max_examples: None,
            seed: Some(42),
            require_cached: false,
            relation_threshold: 0.5f32,
            robustness: false,
            compute_familiarity: true,
            temporal_stratification: false,
            confidence_intervals: true,
        }
    }
}

/// Builder for BiasDatasetConfig.
#[derive(Debug, Clone)]
#[cfg(feature = "eval-bias")]
pub struct BiasDatasetConfigBuilder {
    frequency_weighted: bool,
    validate_distributions: bool,
    min_samples_per_category: usize,
    evaluation_seeds: Vec<u64>,
    confidence_level: f64,
    detailed: bool,
}

#[cfg(feature = "eval-bias")]
impl BiasDatasetConfigBuilder {
    /// Create a new builder with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable frequency-weighted evaluation.
    pub fn with_frequency_weighting(mut self, enable: bool) -> Self {
        self.frequency_weighted = enable;
        self
    }

    /// Enable distribution validation.
    pub fn with_validation(mut self, enable: bool) -> Self {
        self.validate_distributions = enable;
        self
    }

    /// Set minimum samples per category.
    pub fn with_min_samples(mut self, min: usize) -> Self {
        self.min_samples_per_category = min;
        self
    }

    /// Set evaluation seeds.
    pub fn with_seeds(mut self, seeds: Vec<u64>) -> Self {
        self.evaluation_seeds = seeds;
        self
    }

    /// Add a seed.
    pub fn add_seed(mut self, seed: u64) -> Self {
        self.evaluation_seeds.push(seed);
        self
    }

    /// Set confidence level (e.g., 0.95 for 95% CI).
    pub fn with_confidence_level(mut self, level: f64) -> Self {
        self.confidence_level = level;
        self
    }

    /// Enable detailed results.
    pub fn with_detailed(mut self, detailed: bool) -> Self {
        self.detailed = detailed;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> BiasDatasetConfig {
        BiasDatasetConfig {
            frequency_weighted: self.frequency_weighted,
            validate_distributions: self.validate_distributions,
            min_samples_per_category: self.min_samples_per_category,
            evaluation_seeds: self.evaluation_seeds,
            confidence_level: self.confidence_level,
            detailed: self.detailed,
        }
    }
}

#[cfg(feature = "eval-bias")]
impl Default for BiasDatasetConfigBuilder {
    fn default() -> Self {
        Self {
            frequency_weighted: false,
            validate_distributions: false,
            min_samples_per_category: 10,
            evaluation_seeds: vec![42],
            confidence_level: 0.95,
            detailed: false,
        }
    }
}
