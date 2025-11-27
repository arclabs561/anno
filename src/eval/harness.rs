//! Evaluation harness for comprehensive NER benchmarking.
//!
//! Integrates datasets, metrics, and backends into a unified evaluation framework.
//!
//! # Philosophy
//!
//! Following burntsushi's approach: real-world evaluation with clear, honest metrics.
//! No cherry-picking success cases - measure what actually matters.
//!
//! # Usage
//!
//! ```rust,ignore
//! use anno::eval::harness::{EvalHarness, EvalConfig, BackendRegistry};
//! use anno::{PatternNER, StatisticalNER, StackedNER};
//!
//! // Create harness with default config
//! let mut harness = EvalHarness::new(EvalConfig::default())?;
//!
//! // Register backends
//! harness.register("pattern", Box::new(PatternNER::new()));
//! harness.register("statistical", Box::new(StatisticalNER::new()));
//! harness.register("stacked", Box::new(StackedNER::new()));
//!
//! // Run on synthetic data
//! let results = harness.run_synthetic()?;
//!
//! // Or run on real datasets (requires network feature)
//! #[cfg(feature = "network")]
//! let results = harness.run_real_datasets(&[DatasetId::WikiGold, DatasetId::Wnut17])?;
//!
//! // Generate HTML report
//! let html = harness.to_html_report(&results)?;
//! std::fs::write("eval_report.html", html)?;
//! ```

use crate::eval::datasets::GoldEntity;
use crate::eval::loader::{DatasetId, DatasetLoader};
use crate::eval::synthetic::{
    all_datasets, datasets_by_difficulty, datasets_by_domain, AnnotatedExample, Difficulty, Domain,
};
use crate::eval::types::MetricWithVariance;
use crate::eval::{evaluate_ner_model, TypeMetrics};
use crate::{Error, Model, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for evaluation runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalConfig {
    /// Maximum examples per dataset (0 = no limit)
    pub max_examples_per_dataset: usize,
    /// Include per-difficulty breakdown
    pub breakdown_by_difficulty: bool,
    /// Include per-domain breakdown
    pub breakdown_by_domain: bool,
    /// Include per-entity-type metrics
    pub breakdown_by_type: bool,
    /// Run warmup iteration before timing
    pub warmup: bool,
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Minimum confidence threshold (filter predictions below this)
    pub min_confidence: Option<f64>,
    /// Cache directory for downloaded datasets
    pub cache_dir: Option<String>,
    /// Use type mapping to normalize domain-specific entity types to standard NER types.
    ///
    /// When enabled:
    /// - MIT Movie "Actor" -> "Person"
    /// - MIT Restaurant "Restaurant_Name" -> "Organization"
    /// - Biomedical "Disease" -> "Other" (or custom)
    pub normalize_types: bool,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            max_examples_per_dataset: 0, // No limit
            breakdown_by_difficulty: true,
            breakdown_by_domain: true,
            breakdown_by_type: true,
            warmup: true,
            warmup_iterations: 1,
            min_confidence: None,
            cache_dir: None,
            normalize_types: false, // Preserve original types by default
        }
    }
}

impl EvalConfig {
    /// Quick evaluation (limited examples, no breakdowns).
    pub fn quick() -> Self {
        Self {
            max_examples_per_dataset: 100,
            breakdown_by_difficulty: false,
            breakdown_by_domain: false,
            breakdown_by_type: true,
            warmup: false,
            warmup_iterations: 0,
            min_confidence: None,
            cache_dir: None,
            normalize_types: false,
        }
    }

    /// Full evaluation (all examples, all breakdowns).
    pub fn full() -> Self {
        Self {
            max_examples_per_dataset: 0,
            breakdown_by_difficulty: true,
            breakdown_by_domain: true,
            breakdown_by_type: true,
            warmup: true,
            warmup_iterations: 2,
            min_confidence: None,
            cache_dir: None,
            normalize_types: true, // Full eval normalizes types
        }
    }
    
    /// With type normalization enabled.
    ///
    /// When enabled, domain-specific entity types are mapped to standard NER types:
    /// - MIT Movie "Actor" -> "Person"
    /// - MIT Restaurant "Restaurant_Name" -> "Organization"
    /// - Biomedical "Disease" -> "Other"
    pub fn with_type_normalization(mut self) -> Self {
        self.normalize_types = true;
        self
    }
}

// =============================================================================
// Backend Registry
// =============================================================================

/// Registry for NER backends to evaluate.
pub struct BackendRegistry {
    backends: Vec<(String, String, Box<dyn Model>)>, // (name, description, model)
}

impl BackendRegistry {
    /// Create empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
        }
    }

    /// Register a backend for evaluation.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        model: Box<dyn Model>,
    ) {
        self.backends
            .push((name.into(), description.into(), model));
    }

    /// Get number of registered backends.
    pub fn len(&self) -> usize {
        self.backends.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.backends.is_empty()
    }

    /// Iterate over backends.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str, &dyn Model)> {
        self.backends
            .iter()
            .map(|(name, desc, model)| (name.as_str(), desc.as_str(), model.as_ref()))
    }

    /// Register default zero-dependency backends.
    pub fn register_defaults(&mut self) {
        use crate::{HybridNER, PatternNER, StackedNER, StatisticalNER};

        self.register(
            "PatternNER",
            "Regex patterns (DATE/MONEY/EMAIL/etc.)",
            Box::new(PatternNER::new()),
        );
        self.register(
            "StatisticalNER",
            "Heuristics (PER/ORG/LOC)",
            Box::new(StatisticalNER::new()),
        );
        self.register(
            "StackedNER",
            "Pattern + Statistical combined",
            Box::new(StackedNER::new()),
        );
        self.register(
            "HybridNER",
            "Pattern-only hybrid mode",
            Box::new(HybridNER::pattern_only()),
        );
    }

    /// Register ONNX backends if feature enabled.
    #[cfg(feature = "onnx")]
    pub fn register_onnx(&mut self) {
        use crate::{BertNEROnnx, DEFAULT_BERT_ONNX_MODEL};

        match BertNEROnnx::new(DEFAULT_BERT_ONNX_MODEL) {
            Ok(bert) => {
                self.register("BertNEROnnx", "BERT NER via ONNX (~86% F1)", Box::new(bert));
            }
            Err(e) => {
                log::warn!("Failed to load BERT ONNX: {}", e);
            }
        }
    }

    /// Register Candle backends if feature enabled.
    #[cfg(feature = "candle")]
    pub fn register_candle(&mut self) {
        use crate::CandleNER;

        match CandleNER::new(crate::DEFAULT_CANDLE_MODEL) {
            Ok(candle) => {
                self.register(
                    "CandleNER",
                    "Pure Rust BERT NER via Candle",
                    Box::new(candle),
                );
            }
            Err(e) => {
                log::warn!("Failed to load Candle NER: {}", e);
            }
        }
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        let mut registry = Self::new();
        registry.register_defaults();

        #[cfg(feature = "onnx")]
        registry.register_onnx();

        #[cfg(feature = "candle")]
        registry.register_candle();

        registry
    }
}

// =============================================================================
// Results Structures
// =============================================================================

/// Results for a single backend on a single dataset/split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendDatasetResult {
    /// Backend name
    pub backend_name: String,
    /// Dataset/split name
    pub dataset_name: String,
    /// Number of examples evaluated
    pub num_examples: usize,
    /// Number of gold entities
    pub num_gold_entities: usize,
    /// Precision
    pub precision: f64,
    /// Recall
    pub recall: f64,
    /// F1 score
    pub f1: f64,
    /// Macro F1 (average across entity types)
    pub macro_f1: Option<f64>,
    /// Entities found by model
    pub found: usize,
    /// Entities expected (gold)
    pub expected: usize,
    /// Per-entity-type metrics
    pub per_type: HashMap<String, TypeMetrics>,
    /// Evaluation duration
    pub duration_ms: f64,
    /// Tokens per second
    pub tokens_per_second: f64,
}

/// Aggregate results across multiple datasets for a backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendAggregateResult {
    /// Backend name
    pub backend_name: String,
    /// Backend description
    pub description: String,
    /// F1 with variance across datasets
    pub f1: MetricWithVariance,
    /// Precision with variance
    pub precision: MetricWithVariance,
    /// Recall with variance
    pub recall: MetricWithVariance,
    /// Total examples evaluated
    pub total_examples: usize,
    /// Total entities found
    pub total_found: usize,
    /// Total entities expected
    pub total_expected: usize,
    /// Total duration
    pub total_duration_ms: f64,
    /// Per-dataset results
    pub per_dataset: Vec<BackendDatasetResult>,
}

/// Full evaluation results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResults {
    /// Timestamp
    pub timestamp: String,
    /// Configuration used
    pub config: EvalConfig,
    /// Results per backend
    pub backends: Vec<BackendAggregateResult>,
    /// Breakdown by difficulty (if enabled)
    pub by_difficulty: Option<HashMap<String, Vec<BackendDatasetResult>>>,
    /// Breakdown by domain (if enabled)
    pub by_domain: Option<HashMap<String, Vec<BackendDatasetResult>>>,
    /// Dataset statistics
    pub dataset_stats: DatasetStatsSummary,
}

/// Summary statistics about the evaluated datasets.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DatasetStatsSummary {
    /// Total examples
    pub total_examples: usize,
    /// Total gold entities
    pub total_entities: usize,
    /// Entity type distribution
    pub entity_type_distribution: HashMap<String, usize>,
    /// Domain distribution (for synthetic)
    pub domain_distribution: HashMap<String, usize>,
    /// Difficulty distribution (for synthetic)
    pub difficulty_distribution: HashMap<String, usize>,
}

// =============================================================================
// Evaluation Harness
// =============================================================================

/// Main evaluation harness.
pub struct EvalHarness {
    config: EvalConfig,
    registry: BackendRegistry,
    loader: Option<DatasetLoader>,
}

impl EvalHarness {
    /// Create a new harness with given config.
    pub fn new(config: EvalConfig) -> Result<Self> {
        let loader = if let Some(ref dir) = config.cache_dir {
            Some(DatasetLoader::with_cache_dir(dir)?)
        } else {
            DatasetLoader::new().ok()
        };

        Ok(Self {
            config,
            registry: BackendRegistry::new(),
            loader,
        })
    }

    /// Create with default config and default backends.
    pub fn with_defaults() -> Result<Self> {
        let mut harness = Self::new(EvalConfig::default())?;
        harness.registry = BackendRegistry::default();
        Ok(harness)
    }

    /// Create with custom config and default backends.
    pub fn with_config(config: EvalConfig) -> Result<Self> {
        let mut harness = Self::new(config)?;
        harness.registry = BackendRegistry::default();
        Ok(harness)
    }

    /// Register a backend.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        model: Box<dyn Model>,
    ) {
        self.registry.register(name, description, model);
    }

    /// Register all default backends.
    pub fn register_defaults(&mut self) {
        self.registry.register_defaults();
    }

    /// Get read-only reference to registry.
    pub fn registry(&self) -> &BackendRegistry {
        &self.registry
    }

    /// Get mutable reference to registry.
    pub fn registry_mut(&mut self) -> &mut BackendRegistry {
        &mut self.registry
    }

    /// Get number of registered backends.
    pub fn backend_count(&self) -> usize {
        self.registry.len()
    }

    /// Run evaluation on synthetic datasets.
    pub fn run_synthetic(&self) -> Result<EvalResults> {
        if self.registry.is_empty() {
            return Err(Error::InvalidInput(
                "No backends registered for evaluation".to_string(),
            ));
        }

        let all_examples = all_datasets();
        let test_cases: Vec<_> = all_examples
            .iter()
            .filter(|ex| !ex.text.is_empty())
            .take(if self.config.max_examples_per_dataset > 0 {
                self.config.max_examples_per_dataset
            } else {
                usize::MAX
            })
            .map(|ex| (ex.text.clone(), ex.entities.clone()))
            .collect();

        let dataset_stats = compute_dataset_stats(&all_examples);

        let mut backends_results = Vec::new();

        for (name, desc, model) in self.registry.iter() {
            let result = self.evaluate_model_on_cases(model, name, "synthetic", &test_cases)?;
            backends_results.push((name.to_string(), desc.to_string(), vec![result]));
        }

        // Aggregate results
        let backends = backends_results
            .into_iter()
            .map(|(name, desc, results)| aggregate_backend_results(&name, &desc, results))
            .collect();

        // Breakdowns
        let by_difficulty = if self.config.breakdown_by_difficulty {
            Some(self.compute_difficulty_breakdown()?)
        } else {
            None
        };

        let by_domain = if self.config.breakdown_by_domain {
            Some(self.compute_domain_breakdown()?)
        } else {
            None
        };

        Ok(EvalResults {
            timestamp: chrono::Utc::now().to_rfc3339(),
            config: self.config.clone(),
            backends,
            by_difficulty,
            by_domain,
            dataset_stats,
        })
    }

    /// Run evaluation on real datasets (requires network feature for downloading).
    #[cfg(feature = "network")]
    pub fn run_real_datasets(&self, datasets: &[DatasetId]) -> Result<EvalResults> {
        if self.registry.is_empty() {
            return Err(Error::InvalidInput(
                "No backends registered for evaluation".to_string(),
            ));
        }

        let loader = self.loader.as_ref().ok_or_else(|| {
            Error::InvalidInput("Dataset loader not initialized".to_string())
        })?;

        let mut all_test_cases: Vec<(String, Vec<GoldEntity>)> = Vec::new();
        let mut dataset_results: HashMap<String, Vec<(String, Vec<GoldEntity>)>> = HashMap::new();

        for dataset_id in datasets {
            match loader.load_or_download(*dataset_id) {
                Ok(loaded) => {
                    let cases = loaded.to_test_cases();
                    let limited: Vec<_> = if self.config.max_examples_per_dataset > 0 {
                        cases
                            .into_iter()
                            .take(self.config.max_examples_per_dataset)
                            .collect()
                    } else {
                        cases
                    };
                    dataset_results.insert(dataset_id.name().to_string(), limited.clone());
                    all_test_cases.extend(limited);
                }
                Err(e) => {
                    log::warn!("Failed to load {}: {}", dataset_id.name(), e);
                }
            }
        }

        if all_test_cases.is_empty() {
            return Err(Error::InvalidInput("No datasets loaded".to_string()));
        }

        let mut backends_results = Vec::new();

        for (name, desc, model) in self.registry.iter() {
            let mut per_dataset_results = Vec::new();

            for (dataset_name, cases) in &dataset_results {
                let result =
                    self.evaluate_model_on_cases(model, name, dataset_name, cases)?;
                per_dataset_results.push(result);
            }

            backends_results.push((name.to_string(), desc.to_string(), per_dataset_results));
        }

        let backends = backends_results
            .into_iter()
            .map(|(name, desc, results)| aggregate_backend_results(&name, &desc, results))
            .collect();

        Ok(EvalResults {
            timestamp: chrono::Utc::now().to_rfc3339(),
            config: self.config.clone(),
            backends,
            by_difficulty: None,
            by_domain: None,
            dataset_stats: DatasetStatsSummary::default(),
        })
    }

    /// Load cached dataset without downloading.
    pub fn run_cached_datasets(&self, datasets: &[DatasetId]) -> Result<EvalResults> {
        if self.registry.is_empty() {
            return Err(Error::InvalidInput(
                "No backends registered for evaluation".to_string(),
            ));
        }

        let loader = self.loader.as_ref().ok_or_else(|| {
            Error::InvalidInput("Dataset loader not initialized".to_string())
        })?;

        let mut all_test_cases: Vec<(String, Vec<GoldEntity>)> = Vec::new();
        let mut dataset_results: HashMap<String, Vec<(String, Vec<GoldEntity>)>> = HashMap::new();

        for dataset_id in datasets {
            if loader.is_cached(*dataset_id) {
                match loader.load(*dataset_id) {
                    Ok(loaded) => {
                        let cases = loaded.to_test_cases();
                        let limited: Vec<_> = if self.config.max_examples_per_dataset > 0 {
                            cases
                                .into_iter()
                                .take(self.config.max_examples_per_dataset)
                                .collect()
                        } else {
                            cases
                        };
                        dataset_results.insert(dataset_id.name().to_string(), limited.clone());
                        all_test_cases.extend(limited);
                    }
                    Err(e) => {
                        log::warn!("Failed to load cached {}: {}", dataset_id.name(), e);
                    }
                }
            }
        }

        if all_test_cases.is_empty() {
            return Err(Error::InvalidInput(
                "No cached datasets available".to_string(),
            ));
        }

        let mut backends_results = Vec::new();

        for (name, desc, model) in self.registry.iter() {
            let mut per_dataset_results = Vec::new();

            for (dataset_name, cases) in &dataset_results {
                let result =
                    self.evaluate_model_on_cases(model, name, dataset_name, cases)?;
                per_dataset_results.push(result);
            }

            backends_results.push((name.to_string(), desc.to_string(), per_dataset_results));
        }

        let backends = backends_results
            .into_iter()
            .map(|(name, desc, results)| aggregate_backend_results(&name, &desc, results))
            .collect();

        Ok(EvalResults {
            timestamp: chrono::Utc::now().to_rfc3339(),
            config: self.config.clone(),
            backends,
            by_difficulty: None,
            by_domain: None,
            dataset_stats: DatasetStatsSummary::default(),
        })
    }

    /// Evaluate a single backend on test cases.
    fn evaluate_model_on_cases(
        &self,
        model: &dyn Model,
        backend_name: &str,
        dataset_name: &str,
        test_cases: &[(String, Vec<GoldEntity>)],
    ) -> Result<BackendDatasetResult> {
        // Warmup
        if self.config.warmup && !test_cases.is_empty() {
            for _ in 0..self.config.warmup_iterations {
                let _ = model.extract_entities(&test_cases[0].0, None);
            }
        }

        let start = Instant::now();
        let results = evaluate_ner_model(model, test_cases)?;
        let duration = start.elapsed();

        let total_gold: usize = test_cases.iter().map(|(_, gold)| gold.len()).sum();

        Ok(BackendDatasetResult {
            backend_name: backend_name.to_string(),
            dataset_name: dataset_name.to_string(),
            num_examples: test_cases.len(),
            num_gold_entities: total_gold,
            precision: results.precision,
            recall: results.recall,
            f1: results.f1,
            macro_f1: results.macro_f1,
            found: results.found,
            expected: results.expected,
            per_type: results.per_type,
            duration_ms: duration.as_secs_f64() * 1000.0,
            tokens_per_second: results.tokens_per_second,
        })
    }

    /// Compute breakdown by difficulty.
    fn compute_difficulty_breakdown(&self) -> Result<HashMap<String, Vec<BackendDatasetResult>>> {
        let difficulties = [
            Difficulty::Easy,
            Difficulty::Medium,
            Difficulty::Hard,
            Difficulty::Adversarial,
        ];

        let mut breakdown = HashMap::new();

        for difficulty in difficulties {
            let subset: Vec<_> = datasets_by_difficulty(difficulty)
                .into_iter()
                .filter(|ex| !ex.text.is_empty())
                .map(|ex| (ex.text, ex.entities))
                .collect();

            if subset.is_empty() {
                continue;
            }

            let difficulty_name = format!("{:?}", difficulty);
            let mut difficulty_results = Vec::new();

            for (name, _desc, model) in self.registry.iter() {
                let result =
                    self.evaluate_model_on_cases(model, name, &difficulty_name, &subset)?;
                difficulty_results.push(result);
            }

            breakdown.insert(difficulty_name, difficulty_results);
        }

        Ok(breakdown)
    }

    /// Compute breakdown by domain.
    fn compute_domain_breakdown(&self) -> Result<HashMap<String, Vec<BackendDatasetResult>>> {
        let domains = [
            Domain::News,
            Domain::Financial,
            Domain::Technical,
            Domain::Sports,
            Domain::Entertainment,
            Domain::Politics,
            Domain::Ecommerce,
            Domain::Travel,
            Domain::Weather,
            Domain::Academic,
            Domain::Historical,
            Domain::Food,
            Domain::RealEstate,
            Domain::Conversational,
            Domain::SocialMedia,
            Domain::Biomedical,
            Domain::Legal,
            Domain::Scientific,
        ];

        let mut breakdown = HashMap::new();

        for domain in domains {
            let subset: Vec<_> = datasets_by_domain(domain)
                .into_iter()
                .filter(|ex| !ex.text.is_empty())
                .map(|ex| (ex.text, ex.entities))
                .collect();

            if subset.is_empty() {
                continue;
            }

            let domain_name = format!("{:?}", domain);
            let mut domain_results = Vec::new();

            for (name, _desc, model) in self.registry.iter() {
                let result = self.evaluate_model_on_cases(model, name, &domain_name, &subset)?;
                domain_results.push(result);
            }

            breakdown.insert(domain_name, domain_results);
        }

        Ok(breakdown)
    }
}

// =============================================================================
// HTML Report Generation
// =============================================================================

impl EvalResults {
    /// Generate HTML report.
    pub fn to_html(&self) -> String {
        let mut html = String::new();

        html.push_str(HTML_HEAD);
        html.push_str("<body>\n");
        html.push_str("<div class=\"container\">\n");

        // Header
        html.push_str("<h1>NER Evaluation Report</h1>\n");
        html.push_str(&format!(
            "<p class=\"timestamp\">Generated: {}</p>\n",
            self.timestamp
        ));

        // Dataset stats
        html.push_str("<h2>Dataset Summary</h2>\n");
        html.push_str("<div class=\"stats-grid\">\n");
        html.push_str(&format!(
            "<div class=\"stat-box\"><span class=\"stat-value\">{}</span><span class=\"stat-label\">Examples</span></div>\n",
            self.dataset_stats.total_examples
        ));
        html.push_str(&format!(
            "<div class=\"stat-box\"><span class=\"stat-value\">{}</span><span class=\"stat-label\">Entities</span></div>\n",
            self.dataset_stats.total_entities
        ));
        html.push_str(&format!(
            "<div class=\"stat-box\"><span class=\"stat-value\">{}</span><span class=\"stat-label\">Backends</span></div>\n",
            self.backends.len()
        ));
        html.push_str("</div>\n");

        // Overall results table
        html.push_str("<h2>Overall Results</h2>\n");
        html.push_str("<table>\n");
        html.push_str("<thead><tr><th>Backend</th><th>F1</th><th>Precision</th><th>Recall</th><th>Found/Expected</th><th>Time</th></tr></thead>\n");
        html.push_str("<tbody>\n");

        for backend in &self.backends {
            let f1_class = if backend.f1.mean > 0.8 {
                "good"
            } else if backend.f1.mean > 0.5 {
                "ok"
            } else {
                "poor"
            };

            html.push_str(&format!(
                "<tr><td><strong>{}</strong><br><small>{}</small></td>\
                 <td class=\"{}\"><strong>{:.1}%</strong><br><small>{}</small></td>\
                 <td>{:.1}%</td>\
                 <td>{:.1}%</td>\
                 <td>{} / {}</td>\
                 <td>{:.1}ms</td></tr>\n",
                backend.backend_name,
                backend.description,
                f1_class,
                backend.f1.mean * 100.0,
                backend.f1.format_with_ci(),
                backend.precision.mean * 100.0,
                backend.recall.mean * 100.0,
                backend.total_found,
                backend.total_expected,
                backend.total_duration_ms,
            ));
        }
        html.push_str("</tbody></table>\n");

        // Difficulty breakdown
        if let Some(ref by_diff) = self.by_difficulty {
            html.push_str("<h2>Results by Difficulty</h2>\n");
            html.push_str(&self.render_breakdown_table(by_diff));
        }

        // Domain breakdown
        if let Some(ref by_dom) = self.by_domain {
            html.push_str("<h2>Results by Domain</h2>\n");
            html.push_str(&self.render_breakdown_table(by_dom));
        }

        // Per-type metrics for best backend
        if let Some(best) = self.backends.iter().max_by(|a, b| {
            a.f1.mean
                .partial_cmp(&b.f1.mean)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            if !best.per_dataset.is_empty() {
                html.push_str(&format!(
                    "<h2>Per-Type Metrics ({})</h2>\n",
                    best.backend_name
                ));

                // Aggregate per-type from all datasets
                let mut type_metrics: HashMap<String, Vec<&TypeMetrics>> = HashMap::new();
                for ds_result in &best.per_dataset {
                    for (type_name, metrics) in &ds_result.per_type {
                        type_metrics
                            .entry(type_name.clone())
                            .or_default()
                            .push(metrics);
                    }
                }

                html.push_str("<table>\n");
                html.push_str("<thead><tr><th>Type</th><th>F1</th><th>Precision</th><th>Recall</th><th>Correct/Expected</th></tr></thead>\n");
                html.push_str("<tbody>\n");

                let mut sorted_types: Vec<_> = type_metrics.iter().collect();
                sorted_types.sort_by(|a, b| {
                    let avg_f1_a = a.1.iter().map(|m| m.f1).sum::<f64>() / a.1.len() as f64;
                    let avg_f1_b = b.1.iter().map(|m| m.f1).sum::<f64>() / b.1.len() as f64;
                    avg_f1_b
                        .partial_cmp(&avg_f1_a)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                for (type_name, metrics_list) in sorted_types {
                    let avg_f1 = metrics_list.iter().map(|m| m.f1).sum::<f64>() / metrics_list.len() as f64;
                    let avg_p = metrics_list.iter().map(|m| m.precision).sum::<f64>() / metrics_list.len() as f64;
                    let avg_r = metrics_list.iter().map(|m| m.recall).sum::<f64>() / metrics_list.len() as f64;
                    let total_correct: usize = metrics_list.iter().map(|m| m.correct).sum();
                    let total_expected: usize = metrics_list.iter().map(|m| m.expected).sum();

                    let f1_class = if avg_f1 > 0.8 {
                        "good"
                    } else if avg_f1 > 0.5 {
                        "ok"
                    } else {
                        "poor"
                    };

                    html.push_str(&format!(
                        "<tr><td>{}</td><td class=\"{}\">{:.1}%</td><td>{:.1}%</td><td>{:.1}%</td><td>{}/{}</td></tr>\n",
                        type_name,
                        f1_class,
                        avg_f1 * 100.0,
                        avg_p * 100.0,
                        avg_r * 100.0,
                        total_correct,
                        total_expected,
                    ));
                }
                html.push_str("</tbody></table>\n");
            }
        }

        // Entity type distribution
        if !self.dataset_stats.entity_type_distribution.is_empty() {
            html.push_str("<h2>Entity Type Distribution</h2>\n");
            html.push_str("<table>\n");
            html.push_str("<thead><tr><th>Type</th><th>Count</th><th>Percent</th></tr></thead>\n");
            html.push_str("<tbody>\n");

            let total: usize = self.dataset_stats.entity_type_distribution.values().sum();
            let mut sorted: Vec<_> = self.dataset_stats.entity_type_distribution.iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(a.1));

            for (type_name, count) in sorted {
                let pct = (*count as f64 / total as f64) * 100.0;
                html.push_str(&format!(
                    "<tr><td>{}</td><td>{}</td><td>{:.1}%</td></tr>\n",
                    type_name, count, pct
                ));
            }
            html.push_str("</tbody></table>\n");
        }

        html.push_str("</div>\n</body></html>\n");
        html
    }

    /// Render a breakdown table.
    fn render_breakdown_table(
        &self,
        breakdown: &HashMap<String, Vec<BackendDatasetResult>>,
    ) -> String {
        let mut html = String::new();
        html.push_str("<table>\n");

        // Get backend names from first entry
        let backend_names: Vec<_> = breakdown
            .values()
            .next()
            .map(|results| results.iter().map(|r| r.backend_name.as_str()).collect())
            .unwrap_or_default();

        // Header
        html.push_str("<thead><tr><th>Category</th>");
        for name in &backend_names {
            html.push_str(&format!("<th>{}</th>", name));
        }
        html.push_str("</tr></thead>\n");

        // Body
        html.push_str("<tbody>\n");
        let mut sorted_keys: Vec<_> = breakdown.keys().collect();
        sorted_keys.sort();

        for key in sorted_keys {
            let results = &breakdown[key];
            html.push_str(&format!("<tr><td>{}</td>", key));

            for backend_name in &backend_names {
                if let Some(result) = results.iter().find(|r| r.backend_name == *backend_name) {
                    let f1_class = if result.f1 > 0.8 {
                        "good"
                    } else if result.f1 > 0.5 {
                        "ok"
                    } else {
                        "poor"
                    };
                    html.push_str(&format!(
                        "<td class=\"{}\">{:.1}%</td>",
                        f1_class,
                        result.f1 * 100.0
                    ));
                } else {
                    html.push_str("<td>-</td>");
                }
            }
            html.push_str("</tr>\n");
        }
        html.push_str("</tbody></table>\n");

        html
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Compute dataset statistics from examples.
fn compute_dataset_stats(examples: &[AnnotatedExample]) -> DatasetStatsSummary {
    let mut entity_type_dist: HashMap<String, usize> = HashMap::new();
    let mut domain_dist: HashMap<String, usize> = HashMap::new();
    let mut difficulty_dist: HashMap<String, usize> = HashMap::new();

    let mut total_entities = 0;

    for ex in examples {
        *domain_dist.entry(format!("{:?}", ex.domain)).or_insert(0) += 1;
        *difficulty_dist
            .entry(format!("{:?}", ex.difficulty))
            .or_insert(0) += 1;

        for entity in &ex.entities {
            let type_name = format!("{:?}", entity.entity_type);
            *entity_type_dist.entry(type_name).or_insert(0) += 1;
            total_entities += 1;
        }
    }

    DatasetStatsSummary {
        total_examples: examples.len(),
        total_entities,
        entity_type_distribution: entity_type_dist,
        domain_distribution: domain_dist,
        difficulty_distribution: difficulty_dist,
    }
}

/// Aggregate results from multiple datasets into a single backend result.
fn aggregate_backend_results(
    name: &str,
    desc: &str,
    results: Vec<BackendDatasetResult>,
) -> BackendAggregateResult {
    if results.is_empty() {
        return BackendAggregateResult {
            backend_name: name.to_string(),
            description: desc.to_string(),
            f1: MetricWithVariance::default(),
            precision: MetricWithVariance::default(),
            recall: MetricWithVariance::default(),
            total_examples: 0,
            total_found: 0,
            total_expected: 0,
            total_duration_ms: 0.0,
            per_dataset: vec![],
        };
    }

    let f1s: Vec<f64> = results.iter().map(|r| r.f1).collect();
    let precisions: Vec<f64> = results.iter().map(|r| r.precision).collect();
    let recalls: Vec<f64> = results.iter().map(|r| r.recall).collect();

    BackendAggregateResult {
        backend_name: name.to_string(),
        description: desc.to_string(),
        f1: MetricWithVariance::from_samples(&f1s),
        precision: MetricWithVariance::from_samples(&precisions),
        recall: MetricWithVariance::from_samples(&recalls),
        total_examples: results.iter().map(|r| r.num_examples).sum(),
        total_found: results.iter().map(|r| r.found).sum(),
        total_expected: results.iter().map(|r| r.expected).sum(),
        total_duration_ms: results.iter().map(|r| r.duration_ms).sum(),
        per_dataset: results,
    }
}

// =============================================================================
// HTML Template
// =============================================================================

const HTML_HEAD: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NER Evaluation Report</title>
<style>
:root {
    --bg: #0d1117;
    --fg: #c9d1d9;
    --accent: #58a6ff;
    --good: #3fb950;
    --ok: #d29922;
    --poor: #f85149;
    --border: #30363d;
    --surface: #161b22;
}
* { box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
    background: var(--bg);
    color: var(--fg);
    margin: 0;
    padding: 0;
    line-height: 1.6;
}
.container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
h1 { color: var(--accent); border-bottom: 2px solid var(--border); padding-bottom: 0.5rem; }
h2 { color: var(--fg); margin-top: 2rem; }
.timestamp { color: #8b949e; font-size: 0.9rem; }
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin: 1rem 0; }
.stat-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.stat-value { display: block; font-size: 2rem; font-weight: bold; color: var(--accent); }
.stat-label { display: block; font-size: 0.8rem; color: #8b949e; text-transform: uppercase; }
table {
    width: 100%;
    border-collapse: collapse;
    background: var(--surface);
    border-radius: 8px;
    overflow: hidden;
    margin: 1rem 0;
}
th, td { padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid var(--border); }
th { background: #21262d; color: var(--fg); font-weight: 600; }
tr:hover { background: #21262d; }
.good { color: var(--good); }
.ok { color: var(--ok); }
.poor { color: var(--poor); }
small { color: #8b949e; }
</style>
</head>
"#;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_config_default() {
        let config = EvalConfig::default();
        assert!(config.breakdown_by_difficulty);
        assert!(config.breakdown_by_domain);
    }

    #[test]
    fn test_backend_registry() {
        let mut registry = BackendRegistry::new();
        assert!(registry.is_empty());

        registry.register_defaults();
        assert!(!registry.is_empty());
        assert!(registry.len() >= 4); // At least 4 default backends
    }

    #[test]
    fn test_harness_creation() {
        let harness = EvalHarness::with_defaults();
        assert!(harness.is_ok());
    }

    #[test]
    fn test_synthetic_eval() {
        let harness = EvalHarness::with_defaults().unwrap();
        let results = harness.run_synthetic();
        assert!(results.is_ok());

        let results = results.unwrap();
        assert!(!results.backends.is_empty());
    }

    #[test]
    fn test_html_generation() {
        let harness = EvalHarness::with_defaults().unwrap();
        let results = harness.run_synthetic().unwrap();
        let html = results.to_html();

        assert!(html.contains("<html"));
        assert!(html.contains("NER Evaluation Report"));
        assert!(html.contains("PatternNER"));
    }
}

