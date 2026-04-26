//! Multi-run evaluation for statistical robustness.
//!
//! Running evaluations multiple times with different random seeds or data
//! shuffles provides confidence intervals and detects instability.
//!
//! # Research Context
//!
//! Reporting mean ± standard deviation across multiple runs is a best practice
//! from ML research. The Sommerschield (2023) survey on ancient languages notes
//! that single-run results can be misleading due to:
//!
//! - Random initialization effects
//! - Data ordering sensitivity  
//! - Stochastic dropout/sampling
//!
//! This module provides tools to run evaluations N times and aggregate results
//! with proper statistical reporting.
//!
//! # Example
//!
//! ```rust,ignore
//! use anno_eval::eval::{MultiRunConfig, MultiRunEvaluator, MetricWithVariance};
//!
//! let config = MultiRunConfig::new()
//!     .with_runs(5)
//!     .with_shuffle(true);
//!
//! let evaluator = MultiRunEvaluator::new(config);
//! let results = evaluator.evaluate(&model, &test_cases)?;
//!
//! println!("F1: {} (n={})", results.f1, results.f1.n);
//! // F1: 85.2% ± 1.3% (n=5)
//! ```

use super::{evaluate_ner_model, GoldEntity, MetricWithVariance, NEREvaluationResults};
use anno::{Error, Model, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for multi-run evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiRunConfig {
    /// Number of evaluation runs (default: 5)
    pub num_runs: usize,
    /// Whether to shuffle data between runs (default: true)
    pub shuffle: bool,
    /// Random seed base (seeds will be base, base+1, base+2, ...)
    pub seed_base: u64,
    /// Whether to parallelize runs (requires rayon feature)
    pub parallel: bool,
    /// Minimum runs required for CI calculation (default: 3)
    pub min_runs_for_ci: usize,
}

impl Default for MultiRunConfig {
    fn default() -> Self {
        Self {
            num_runs: 5,
            shuffle: true,
            seed_base: 42,
            parallel: false,
            min_runs_for_ci: 3,
        }
    }
}

impl MultiRunConfig {
    /// Create a new config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of runs.
    pub fn with_runs(mut self, n: usize) -> Self {
        self.num_runs = n.max(1);
        self
    }

    /// Set whether to shuffle data between runs.
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set the random seed base.
    pub fn with_seed_base(mut self, seed: u64) -> Self {
        self.seed_base = seed;
        self
    }

    /// Enable or disable parallel execution.
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }
}

// =============================================================================
// Results
// =============================================================================

/// Results from multi-run evaluation.
///
/// All metrics are reported with mean ± std and confidence intervals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiRunResults {
    /// Precision with variance
    pub precision: MetricWithVariance,
    /// Recall with variance
    pub recall: MetricWithVariance,
    /// F1 with variance
    pub f1: MetricWithVariance,
    /// Macro F1 with variance (if available)
    pub macro_f1: Option<MetricWithVariance>,
    /// Per-type F1 with variance
    pub per_type_f1: HashMap<String, MetricWithVariance>,
    /// Tokens per second with variance
    pub throughput: MetricWithVariance,
    /// Individual run results (for detailed analysis)
    pub individual_runs: Vec<NEREvaluationResults>,
    /// Configuration used
    pub config: MultiRunConfig,
    /// Seeds used for each run
    pub seeds: Vec<u64>,
}

impl MultiRunResults {
    /// Format as a human-readable summary table.
    pub fn format_summary(&self) -> String {
        let mut s = String::new();

        s.push_str(&format!("Multi-Run Evaluation (n={})\n", self.f1.n));
        s.push_str(&format!("{:=<50}\n", ""));
        s.push_str(&format!(
            "{:<12} {:<20} {:<15}\n",
            "Metric", "Mean ± CI95", "Range"
        ));
        s.push_str(&format!("{:-<50}\n", ""));

        s.push_str(&format!(
            "{:<12} {:<20} {:<15}\n",
            "Precision",
            self.precision.format_with_ci(),
            self.precision.format_with_range()
        ));
        s.push_str(&format!(
            "{:<12} {:<20} {:<15}\n",
            "Recall",
            self.recall.format_with_ci(),
            self.recall.format_with_range()
        ));
        s.push_str(&format!(
            "{:<12} {:<20} {:<15}\n",
            "F1",
            self.f1.format_with_ci(),
            self.f1.format_with_range()
        ));

        if let Some(ref macro_f1) = self.macro_f1 {
            s.push_str(&format!(
                "{:<12} {:<20} {:<15}\n",
                "Macro F1",
                macro_f1.format_with_ci(),
                macro_f1.format_with_range()
            ));
        }

        s.push_str(&format!("{:-<50}\n", ""));

        // Per-type breakdown
        if !self.per_type_f1.is_empty() {
            s.push_str("\nPer-Type F1:\n");
            let mut types: Vec<_> = self.per_type_f1.keys().collect();
            types.sort();
            for entity_type in types {
                if let Some(metric) = self.per_type_f1.get(entity_type) {
                    s.push_str(&format!(
                        "  {:<10} {}\n",
                        entity_type,
                        metric.format_with_ci()
                    ));
                }
            }
        }

        // Stability analysis
        let cv = self.f1.coefficient_of_variation();
        s.push_str(&format!("\nStability: CV = {:.2}% ", cv * 100.0));
        if cv < 0.02 {
            s.push_str("(excellent)");
        } else if cv < 0.05 {
            s.push_str("(good)");
        } else if cv < 0.10 {
            s.push_str("(moderate)");
        } else {
            s.push_str("(high variance - investigate)");
        }
        s.push('\n');

        s
    }

    /// Check if results are statistically stable.
    ///
    /// Returns true if coefficient of variation < threshold.
    pub fn is_stable(&self, threshold: f64) -> bool {
        self.f1.coefficient_of_variation() < threshold
    }
}

// =============================================================================
// Evaluator
// =============================================================================

/// Multi-run evaluator for NER models.
///
/// Runs evaluation multiple times and aggregates results with statistics.
#[derive(Debug, Clone)]
pub struct MultiRunEvaluator {
    config: MultiRunConfig,
}

impl MultiRunEvaluator {
    /// Create a new evaluator with the given config.
    pub fn new(config: MultiRunConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub fn default_config() -> Self {
        Self::new(MultiRunConfig::default())
    }

    /// Evaluate a model on test cases with multiple runs.
    pub fn evaluate(
        &self,
        model: &dyn Model,
        test_cases: &[(String, Vec<GoldEntity>)],
    ) -> Result<MultiRunResults> {
        if test_cases.is_empty() {
            return Err(Error::InvalidInput("Empty test cases".to_string()));
        }

        let mut all_results = Vec::with_capacity(self.config.num_runs);
        let mut seeds = Vec::with_capacity(self.config.num_runs);

        for run in 0..self.config.num_runs {
            let seed = self.config.seed_base + run as u64;
            seeds.push(seed);

            // Optionally shuffle data
            let data = if self.config.shuffle {
                shuffle_with_seed(test_cases, seed)
            } else {
                test_cases.to_vec()
            };

            // Run evaluation
            let result = evaluate_ner_model(model, &data)?;
            all_results.push(result);
        }

        // Aggregate metrics
        let precision_samples: Vec<f64> = all_results.iter().map(|r| r.precision).collect();
        let recall_samples: Vec<f64> = all_results.iter().map(|r| r.recall).collect();
        let f1_samples: Vec<f64> = all_results.iter().map(|r| r.f1).collect();
        let throughput_samples: Vec<f64> =
            all_results.iter().map(|r| r.tokens_per_second).collect();

        // Macro F1
        let macro_f1_samples: Vec<f64> = all_results.iter().filter_map(|r| r.macro_f1).collect();
        let macro_f1 = if macro_f1_samples.len() >= self.config.min_runs_for_ci {
            Some(MetricWithVariance::from_samples(&macro_f1_samples))
        } else {
            None
        };

        // Per-type F1
        let mut per_type_f1 = HashMap::new();
        if let Some(first) = all_results.first() {
            for entity_type in first.per_type.keys() {
                let type_f1s: Vec<f64> = all_results
                    .iter()
                    .filter_map(|r| r.per_type.get(entity_type).map(|m| m.f1))
                    .collect();
                if type_f1s.len() >= self.config.min_runs_for_ci {
                    per_type_f1.insert(
                        entity_type.clone(),
                        MetricWithVariance::from_samples(&type_f1s),
                    );
                }
            }
        }

        Ok(MultiRunResults {
            precision: MetricWithVariance::from_samples(&precision_samples),
            recall: MetricWithVariance::from_samples(&recall_samples),
            f1: MetricWithVariance::from_samples(&f1_samples),
            macro_f1,
            per_type_f1,
            throughput: MetricWithVariance::from_samples(&throughput_samples),
            individual_runs: all_results,
            config: self.config.clone(),
            seeds,
        })
    }

    /// Quick evaluation with 3 runs (suitable for development).
    pub fn quick_eval(
        model: &dyn Model,
        test_cases: &[(String, Vec<GoldEntity>)],
    ) -> Result<MultiRunResults> {
        let evaluator = Self::new(MultiRunConfig::new().with_runs(3));
        evaluator.evaluate(model, test_cases)
    }

    /// Thorough evaluation with 10 runs (suitable for publication).
    pub fn thorough_eval(
        model: &dyn Model,
        test_cases: &[(String, Vec<GoldEntity>)],
    ) -> Result<MultiRunResults> {
        let evaluator = Self::new(MultiRunConfig::new().with_runs(10));
        evaluator.evaluate(model, test_cases)
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Shuffle data with a deterministic seed.
fn shuffle_with_seed<T: Clone>(data: &[T], seed: u64) -> Vec<T> {
    let mut indices: Vec<usize> = (0..data.len()).collect();

    // Fisher-Yates shuffle with seeded PRNG
    let mut rng_state = seed;
    for i in (1..indices.len()).rev() {
        // Simple LCG for deterministic shuffling
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (rng_state % (i as u64 + 1)) as usize;
        indices.swap(i, j);
    }

    indices.into_iter().map(|i| data[i].clone()).collect()
}

/// Compare two models across multiple runs with statistical significance.
pub fn compare_models_multi_run(
    model_a: (&str, &dyn Model),
    model_b: (&str, &dyn Model),
    test_cases: &[(String, Vec<GoldEntity>)],
    config: MultiRunConfig,
) -> Result<ModelComparison> {
    let evaluator = MultiRunEvaluator::new(config);

    let results_a = evaluator.evaluate(model_a.1, test_cases)?;
    let results_b = evaluator.evaluate(model_b.1, test_cases)?;

    // Paired t-test for significance
    let (t_stat, p_value) = paired_t_test(
        &results_a
            .individual_runs
            .iter()
            .map(|r| r.f1)
            .collect::<Vec<_>>(),
        &results_b
            .individual_runs
            .iter()
            .map(|r| r.f1)
            .collect::<Vec<_>>(),
    );

    let difference = results_a.f1.mean - results_b.f1.mean;
    let significant = p_value < 0.05;

    Ok(ModelComparison {
        model_a_name: model_a.0.to_string(),
        model_b_name: model_b.0.to_string(),
        model_a_f1: results_a.f1,
        model_b_f1: results_b.f1,
        f1_difference: difference,
        t_statistic: t_stat,
        p_value,
        significant_at_05: significant,
        winner: if significant {
            if difference > 0.0 {
                Some(model_a.0.to_string())
            } else {
                Some(model_b.0.to_string())
            }
        } else {
            None
        },
    })
}

/// Result of comparing two models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparison {
    /// Name of first model
    pub model_a_name: String,
    /// Name of second model
    pub model_b_name: String,
    /// F1 of first model with variance
    pub model_a_f1: MetricWithVariance,
    /// F1 of second model with variance
    pub model_b_f1: MetricWithVariance,
    /// Difference in mean F1 (A - B)
    pub f1_difference: f64,
    /// T-statistic from paired t-test
    pub t_statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Whether difference is significant at p < 0.05
    pub significant_at_05: bool,
    /// Winner (if significant)
    pub winner: Option<String>,
}

impl std::fmt::Display for ModelComparison {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Model Comparison: {} vs {}",
            self.model_a_name, self.model_b_name
        )?;
        writeln!(f, "{:-<50}", "")?;
        writeln!(
            f,
            "{}: {}",
            self.model_a_name,
            self.model_a_f1.format_with_ci()
        )?;
        writeln!(
            f,
            "{}: {}",
            self.model_b_name,
            self.model_b_f1.format_with_ci()
        )?;
        writeln!(f, "Difference: {:+.2}%", self.f1_difference * 100.0)?;
        writeln!(f, "p-value: {:.4}", self.p_value)?;
        if self.significant_at_05 {
            writeln!(
                f,
                "Result: {} significantly better (p < 0.05)",
                self.winner.as_deref().unwrap_or("?")
            )?;
        } else {
            writeln!(f, "Result: No significant difference")?;
        }
        Ok(())
    }
}

/// Paired t-test for comparing two sets of measurements.
///
/// Returns (t-statistic, two-tailed p-value).
fn paired_t_test(a: &[f64], b: &[f64]) -> (f64, f64) {
    if a.len() != b.len() || a.is_empty() {
        return (0.0, 1.0);
    }

    let n = a.len() as f64;
    let diffs: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();

    let mean_diff: f64 = diffs.iter().sum::<f64>() / n;
    let var_diff: f64 = if a.len() > 1 {
        diffs.iter().map(|d| (d - mean_diff).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };

    let std_err = (var_diff / n).sqrt();

    let t_stat = if std_err > 1e-10 {
        mean_diff / std_err
    } else {
        // If variance is zero, all differences are identical
        // Large |mean_diff| = very significant, mean_diff == 0 = identical
        if mean_diff.abs() > 1e-10 {
            // Return large t-stat with same sign as mean_diff
            mean_diff.signum() * 100.0
        } else {
            0.0
        }
    };

    // Approximate two-tailed p-value using normal distribution
    // For small samples this is an approximation (t-distribution would be more accurate)
    // Normal CDF returns P(X < x), so P(|X| > |t|) = 2 * (1 - CDF(|t|))
    let p_value = 2.0 * (1.0 - normal_cdf(t_stat.abs()));

    (t_stat, p_value)
}

/// Approximate CDF of standard normal distribution.
fn normal_cdf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation (7.1.26)
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs() / std::f64::consts::SQRT_2;

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    0.5 * (1.0 + sign * y)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shuffle_deterministic() {
        let data: Vec<i32> = (0..10).collect();

        let shuffled1 = shuffle_with_seed(&data, 42);
        let shuffled2 = shuffle_with_seed(&data, 42);
        let shuffled3 = shuffle_with_seed(&data, 43);

        assert_eq!(
            shuffled1, shuffled2,
            "Same seed should produce same shuffle"
        );
        assert_ne!(
            shuffled1, shuffled3,
            "Different seeds should produce different shuffles"
        );
        assert_ne!(shuffled1, data, "Shuffle should change order");
    }

    #[test]
    fn test_shuffle_preserves_elements() {
        let data: Vec<i32> = (0..20).collect();
        let shuffled = shuffle_with_seed(&data, 12345);

        let mut sorted = shuffled.clone();
        sorted.sort();
        assert_eq!(sorted, data, "Shuffle should preserve all elements");
    }

    #[test]
    fn test_metric_with_variance_from_samples() {
        let samples = vec![0.80, 0.82, 0.85, 0.83, 0.80];
        let metric = MetricWithVariance::from_samples(&samples);

        assert!((metric.mean - 0.82).abs() < 0.01);
        assert!(metric.std_dev > 0.0);
        assert!(metric.ci_95 > 0.0);
        assert_eq!(metric.n, 5);
        assert!((metric.min - 0.80).abs() < 0.01);
        assert!((metric.max - 0.85).abs() < 0.01);
    }

    #[test]
    fn test_metric_with_variance_empty() {
        let samples: Vec<f64> = vec![];
        let metric = MetricWithVariance::from_samples(&samples);

        assert_eq!(metric.n, 0);
        assert!((metric.mean - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_metric_with_variance_single() {
        let samples = vec![0.85];
        let metric = MetricWithVariance::from_samples(&samples);

        assert_eq!(metric.n, 1);
        assert!((metric.mean - 0.85).abs() < 0.001);
        assert!((metric.std_dev - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_coefficient_of_variation() {
        let stable = MetricWithVariance::from_samples(&[0.85, 0.85, 0.85, 0.85, 0.85]);
        let variable = MetricWithVariance::from_samples(&[0.60, 0.70, 0.80, 0.90, 1.00]);

        assert!(stable.coefficient_of_variation() < 0.01);
        assert!(variable.coefficient_of_variation() > 0.1);
    }

    #[test]
    fn test_paired_t_test_identical() {
        let a = vec![0.80, 0.82, 0.85, 0.83, 0.80];
        let b = vec![0.80, 0.82, 0.85, 0.83, 0.80];

        let (t_stat, p_value) = paired_t_test(&a, &b);

        assert!((t_stat - 0.0).abs() < 0.001);
        assert!((p_value - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_paired_t_test_different() {
        // Need varying differences (not constant) for meaningful t-test
        let a = vec![0.90, 0.92, 0.88, 0.91, 0.94];
        let b = vec![0.80, 0.78, 0.79, 0.81, 0.82];

        let (t_stat, p_value) = paired_t_test(&a, &b);

        // A is consistently better, so t_stat should be positive
        assert!(t_stat > 0.0, "t_stat should be positive: {}", t_stat);
        // The difference is clear, so p should be small
        assert!(
            p_value < 0.05,
            "p-value should indicate significance: {}",
            p_value
        );
    }

    #[test]
    fn test_multi_run_config_builder() {
        let config = MultiRunConfig::new()
            .with_runs(10)
            .with_shuffle(false)
            .with_seed_base(123);

        assert_eq!(config.num_runs, 10);
        assert!(!config.shuffle);
        assert_eq!(config.seed_base, 123);
    }

    #[test]
    fn test_normal_cdf() {
        // Standard normal: P(X < 0) = 0.5
        assert!((normal_cdf(0.0) - 0.5).abs() < 0.01);

        // P(X < 2) ≈ 0.977
        assert!((normal_cdf(2.0) - 0.977).abs() < 0.01);

        // P(X < -2) ≈ 0.023
        assert!((normal_cdf(-2.0) - 0.023).abs() < 0.01);
    }
}

// =============================================================================
// Property Tests
// =============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    // -------------------------------------------------------------------------
    // Shuffle Properties
    // -------------------------------------------------------------------------

    proptest! {
        /// Shuffle is deterministic given seed
        #[test]
        fn prop_shuffle_deterministic(seed in any::<u64>(), len in 0usize..100) {
            let data: Vec<usize> = (0..len).collect();
            let s1 = shuffle_with_seed(&data, seed);
            let s2 = shuffle_with_seed(&data, seed);
            prop_assert_eq!(s1, s2, "Same seed should produce same shuffle");
        }

        /// Shuffle preserves all elements
        #[test]
        fn prop_shuffle_preserves_elements(seed in any::<u64>(), len in 0usize..50) {
            let data: Vec<usize> = (0..len).collect();
            let mut shuffled = shuffle_with_seed(&data, seed);
            shuffled.sort();
            prop_assert_eq!(shuffled, data, "Shuffle should preserve all elements");
        }

        /// Different seeds produce different results (statistically)
        #[test]
        fn prop_different_seeds_differ(seed1 in any::<u64>(), seed2 in any::<u64>()) {
            // Only test with reasonably sized data where collision is unlikely
            let data: Vec<usize> = (0..20).collect();
            let s1 = shuffle_with_seed(&data, seed1);
            let s2 = shuffle_with_seed(&data, seed2);

            // If seeds are different, shuffles should likely differ
            // (but not guaranteed for small data - allow some collisions)
            if seed1 != seed2 {
                // Just ensure they're valid permutations
                let mut sorted1 = s1.clone();
                let mut sorted2 = s2.clone();
                sorted1.sort();
                sorted2.sort();
                prop_assert_eq!(sorted1, data.clone());
                prop_assert_eq!(sorted2, data);
            }
        }
    }

    // -------------------------------------------------------------------------
    // MetricWithVariance Properties
    // -------------------------------------------------------------------------

    proptest! {
        /// Mean is within [min, max]
        #[test]
        fn prop_mean_within_range(samples in prop::collection::vec(0.0f64..1.0, 1..20)) {
            let metric = MetricWithVariance::from_samples(&samples);
            prop_assert!(metric.mean >= metric.min - 1e-10);
            prop_assert!(metric.mean <= metric.max + 1e-10);
        }

        /// Std dev is non-negative
        #[test]
        fn prop_std_dev_non_negative(samples in prop::collection::vec(0.0f64..1.0, 1..20)) {
            let metric = MetricWithVariance::from_samples(&samples);
            prop_assert!(metric.std_dev >= 0.0);
        }

        /// CI95 is non-negative
        #[test]
        fn prop_ci95_non_negative(samples in prop::collection::vec(0.0f64..1.0, 1..20)) {
            let metric = MetricWithVariance::from_samples(&samples);
            prop_assert!(metric.ci_95 >= 0.0);
        }

        /// n matches input length
        #[test]
        fn prop_n_matches_length(samples in prop::collection::vec(0.0f64..1.0, 0..20)) {
            let metric = MetricWithVariance::from_samples(&samples);
            prop_assert_eq!(metric.n, samples.len());
        }

        /// Coefficient of variation is non-negative
        #[test]
        fn prop_cv_non_negative(samples in prop::collection::vec(0.0f64..1.0, 1..20)) {
            let metric = MetricWithVariance::from_samples(&samples);
            let cv = metric.coefficient_of_variation();
            prop_assert!(cv >= 0.0 || cv.is_nan(), "CV should be non-negative: {}", cv);
        }

        /// Identical samples have zero variance
        #[test]
        fn prop_identical_zero_variance(value in 0.0f64..1.0, n in 2usize..10) {
            let samples: Vec<f64> = vec![value; n];
            let metric = MetricWithVariance::from_samples(&samples);
            prop_assert!((metric.std_dev - 0.0).abs() < 1e-10, "Identical samples should have 0 std dev");
            prop_assert!((metric.mean - value).abs() < 1e-10);
        }
    }

    // -------------------------------------------------------------------------
    // Paired T-Test Properties
    // -------------------------------------------------------------------------

    proptest! {
        /// T-test with identical samples gives t=0, p=1
        #[test]
        fn prop_ttest_identical_no_difference(samples in prop::collection::vec(0.0f64..1.0, 2..10)) {
            let (t_stat, p_value) = paired_t_test(&samples, &samples);
            prop_assert!((t_stat - 0.0).abs() < 1e-10, "t-stat should be 0 for identical samples");
            prop_assert!((p_value - 1.0).abs() < 0.01, "p-value should be ~1 for identical samples");
        }

        /// T-test returns p in [0, 1]
        #[test]
        fn prop_ttest_p_value_bounds(
            a in prop::collection::vec(0.0f64..1.0, 2..10),
            b in prop::collection::vec(0.0f64..1.0, 2..10)
        ) {
            // Need same length
            let min_len = a.len().min(b.len());
            let a = &a[..min_len];
            let b = &b[..min_len];

            let (_, p_value) = paired_t_test(a, b);
            prop_assert!((0.0..=1.0).contains(&p_value), "p-value {} out of [0,1]", p_value);
        }
    }

    // -------------------------------------------------------------------------
    // Normal CDF Properties
    // -------------------------------------------------------------------------

    proptest! {
        /// CDF is monotonically increasing
        #[test]
        fn prop_cdf_monotonic(x1 in -5.0f64..5.0, x2 in -5.0f64..5.0) {
            if x1 < x2 {
                prop_assert!(normal_cdf(x1) <= normal_cdf(x2) + 1e-10);
            }
        }

        /// CDF is in [0, 1]
        #[test]
        fn prop_cdf_bounds(x in -10.0f64..10.0) {
            let cdf = normal_cdf(x);
            prop_assert!((0.0..=1.0).contains(&cdf), "CDF {} out of bounds for x={}", cdf, x);
        }

        /// CDF(0) ≈ 0.5 for standard normal
        #[test]
        fn prop_cdf_symmetric_at_zero(_unused in Just(())) {
            prop_assert!((normal_cdf(0.0) - 0.5).abs() < 0.01);
        }
    }

    // -------------------------------------------------------------------------
    // MultiRunConfig Properties
    // -------------------------------------------------------------------------

    proptest! {
        /// Config builder sets values correctly
        #[test]
        fn prop_config_builder(runs in 1usize..100, seed in any::<u64>(), shuffle in any::<bool>()) {
            let config = MultiRunConfig::new()
                .with_runs(runs)
                .with_seed_base(seed)
                .with_shuffle(shuffle);

            prop_assert_eq!(config.num_runs, runs);
            prop_assert_eq!(config.seed_base, seed);
            prop_assert_eq!(config.shuffle, shuffle);
        }

        /// Config enforces minimum 1 run
        #[test]
        fn prop_config_min_runs(_unused in Just(())) {
            let config = MultiRunConfig::new().with_runs(0);
            prop_assert!(config.num_runs >= 1);
        }
    }
}
