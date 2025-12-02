//! Configuration and statistical utilities for bias evaluation.
//!
//! Provides configuration structures and statistical reporting for bias evaluation
//! datasets, including confidence intervals, effect sizes, and frequency weighting.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Bias Dataset Configuration
// =============================================================================

/// Configuration for bias evaluation datasets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiasDatasetConfig {
    /// Minimum samples per category for statistical validity
    pub min_samples_per_category: usize,
    /// Use frequency-weighted sampling from real distributions
    pub frequency_weighted: bool,
    /// Validate against reference distributions
    pub validate_distributions: bool,
    /// Multiple seeds for variance estimation
    pub evaluation_seeds: Vec<u64>,
    /// Confidence level for intervals (default: 0.95)
    pub confidence_level: f64,
    /// Include detailed per-category metrics
    pub detailed: bool,
}

impl Default for BiasDatasetConfig {
    fn default() -> Self {
        Self {
            min_samples_per_category: 30,
            frequency_weighted: false,
            validate_distributions: false,
            evaluation_seeds: vec![42, 123, 456, 789, 999],
            confidence_level: 0.95,
            detailed: false,
        }
    }
}

impl BiasDatasetConfig {
    /// Create a new configuration with recommended settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create configuration with frequency weighting enabled.
    pub fn with_frequency_weighting(mut self) -> Self {
        self.frequency_weighted = true;
        self
    }

    /// Create configuration with validation enabled.
    pub fn with_validation(mut self) -> Self {
        self.validate_distributions = true;
        self
    }

    /// Set minimum samples per category.
    pub fn with_min_samples(mut self, min: usize) -> Self {
        self.min_samples_per_category = min;
        self
    }

    /// Set evaluation seeds for variance estimation.
    pub fn with_seeds(mut self, seeds: Vec<u64>) -> Self {
        self.evaluation_seeds = seeds;
        self
    }

    /// Enable detailed reporting.
    pub fn with_detailed(mut self, detailed: bool) -> Self {
        self.detailed = detailed;
        self
    }
}

// =============================================================================
// Statistical Results
// =============================================================================

/// Statistical results with confidence intervals and effect sizes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalBiasResults {
    /// Mean bias gap or recognition rate
    pub mean: f64,
    /// Standard deviation across seeds/runs
    pub std_dev: f64,
    /// 95% confidence interval (lower, upper)
    pub ci_95: (f64, f64),
    /// Minimum value observed
    pub min: f64,
    /// Maximum value observed
    pub max: f64,
    /// Effect size (Cohen's d) if comparing two groups
    pub effect_size: Option<f64>,
    /// Number of samples
    pub n: usize,
    /// Standard error
    pub std_error: f64,
}

impl StatisticalBiasResults {
    /// Create from a vector of values (e.g., across multiple seeds).
    pub fn from_values(values: &[f64], confidence_level: f64) -> Self {
        if values.is_empty() {
            return Self {
                mean: 0.0,
                std_dev: 0.0,
                ci_95: (0.0, 0.0),
                min: 0.0,
                max: 0.0,
                effect_size: None,
                n: 0,
                std_error: 0.0,
            };
        }

        let n = values.len();
        let mean = values.iter().sum::<f64>() / n as f64;
        let variance = if n > 1 {
            values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };
        let std_dev = variance.sqrt();
        let std_error = std_dev / (n as f64).sqrt();

        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        // Approximate 95% CI using t-distribution (simplified: using z-score for n>=30)
        let z_score = if confidence_level == 0.95 {
            1.96
        } else if confidence_level == 0.99 {
            2.576
        } else {
            // Approximate for other levels
            1.96 * (confidence_level / 0.95)
        };
        let margin = z_score * std_error;
        let ci_95 = (mean - margin, mean + margin);

        Self {
            mean,
            std_dev,
            ci_95,
            min,
            max,
            effect_size: None,
            n,
            std_error,
        }
    }

    /// Compute effect size (Cohen's d) between two groups.
    pub fn compute_effect_size(group1: &[f64], group2: &[f64]) -> f64 {
        if group1.is_empty() || group2.is_empty() {
            return 0.0;
        }

        let mean1 = group1.iter().sum::<f64>() / group1.len() as f64;
        let mean2 = group2.iter().sum::<f64>() / group2.len() as f64;

        let var1 = if group1.len() > 1 {
            group1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (group1.len() - 1) as f64
        } else {
            0.0
        };

        let var2 = if group2.len() > 1 {
            group2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (group2.len() - 1) as f64
        } else {
            0.0
        };

        let pooled_std = ((var1 + var2) / 2.0).sqrt();
        if pooled_std == 0.0 {
            return 0.0;
        }

        (mean1 - mean2) / pooled_std
    }

    /// Format as string with confidence interval.
    pub fn format_with_ci(&self) -> String {
        format!(
            "{:.3} (95% CI: {:.3} - {:.3}, n={}, SD={:.3})",
            self.mean, self.ci_95.0, self.ci_95.1, self.n, self.std_dev
        )
    }
}

// =============================================================================
// Frequency-Weighted Results
// =============================================================================

/// Results with both unweighted and frequency-weighted metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyWeightedResults {
    /// Unweighted recognition rate
    pub unweighted_rate: f64,
    /// Frequency-weighted recognition rate
    pub weighted_rate: f64,
    /// Frequency distribution used (name -> frequency)
    pub frequency_distribution: HashMap<String, f64>,
    /// Number of samples
    pub n: usize,
}

impl FrequencyWeightedResults {
    /// Create from recognition results and frequencies.
    pub fn new(recognized: &[bool], frequencies: &HashMap<String, f64>, names: &[String]) -> Self {
        if recognized.is_empty() {
            return Self {
                unweighted_rate: 0.0,
                weighted_rate: 0.0,
                frequency_distribution: frequencies.clone(),
                n: 0,
            };
        }

        let unweighted_rate =
            recognized.iter().filter(|&&r| r).count() as f64 / recognized.len() as f64;

        // Weighted rate: sum(recognized[i] * frequency[i]) / sum(frequency[i])
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for (i, &rec) in recognized.iter().enumerate() {
            if i < names.len() {
                let freq = frequencies
                    .get(&names[i])
                    .copied()
                    .unwrap_or(1.0 / names.len() as f64);
                if rec {
                    weighted_sum += freq;
                }
                total_weight += freq;
            }
        }

        let weighted_rate = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            unweighted_rate
        };

        Self {
            unweighted_rate,
            weighted_rate,
            frequency_distribution: frequencies.clone(),
            n: recognized.len(),
        }
    }
}

// =============================================================================
// Distribution Validation
// =============================================================================

/// Validation results comparing dataset distribution to reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionValidation {
    /// Whether distribution matches reference (within tolerance)
    pub is_valid: bool,
    /// Maximum deviation from reference
    pub max_deviation: f64,
    /// Per-category deviations
    pub category_deviations: HashMap<String, f64>,
    /// Tolerance used for validation
    pub tolerance: f64,
}

impl DistributionValidation {
    /// Validate distribution against reference.
    pub fn validate(
        observed: &HashMap<String, f64>,
        reference: &HashMap<String, f64>,
        tolerance: f64,
    ) -> Self {
        let mut max_deviation: f64 = 0.0;
        let mut category_deviations = HashMap::new();

        for (category, &ref_value) in reference {
            let obs_value = observed.get(category).copied().unwrap_or(0.0);
            let deviation = (obs_value - ref_value).abs();
            category_deviations.insert(category.clone(), deviation);
            max_deviation = max_deviation.max(deviation);
        }

        // Check for categories in observed but not in reference
        for category in observed.keys() {
            if !reference.contains_key(category) {
                let deviation = observed[category];
                category_deviations.insert(category.clone(), deviation);
                max_deviation = max_deviation.max(deviation);
            }
        }

        let is_valid = max_deviation <= tolerance;

        Self {
            is_valid,
            max_deviation,
            category_deviations,
            tolerance,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistical_results() {
        let values = vec![0.8, 0.82, 0.79, 0.81, 0.83];
        let results = StatisticalBiasResults::from_values(&values, 0.95);

        assert!((results.mean - 0.81).abs() < 0.01);
        assert!(results.n == 5);
        assert!(results.ci_95.0 < results.mean);
        assert!(results.ci_95.1 > results.mean);
    }

    #[test]
    fn test_effect_size() {
        let group1 = vec![0.9, 0.91, 0.89, 0.92, 0.88];
        let group2 = vec![0.7, 0.71, 0.69, 0.72, 0.68];

        let d = StatisticalBiasResults::compute_effect_size(&group1, &group2);
        assert!(d > 0.0); // Should be positive (group1 > group2)
                          // Effect size should be large (groups are 0.2 apart with small variance)
                          // Cohen's d = (0.9 - 0.7) / pooled_std, which can be > 10 for very small std
        assert!(d < 100.0); // Should be reasonable (allowing for small variance case)
    }

    #[test]
    fn test_frequency_weighted() {
        let recognized = vec![true, false, true, true, false];
        let mut frequencies = HashMap::new();
        frequencies.insert("Name1".to_string(), 0.5);
        frequencies.insert("Name2".to_string(), 0.3);
        frequencies.insert("Name3".to_string(), 0.2);
        let names = vec![
            "Name1".to_string(),
            "Name2".to_string(),
            "Name3".to_string(),
            "Name1".to_string(),
            "Name2".to_string(),
        ];

        let results = FrequencyWeightedResults::new(&recognized, &frequencies, &names);
        assert!(results.unweighted_rate > 0.0);
        assert!(results.weighted_rate > 0.0);
    }

    #[test]
    fn test_distribution_validation() {
        let mut observed = HashMap::new();
        observed.insert("A".to_string(), 0.5);
        observed.insert("B".to_string(), 0.5);

        let mut reference = HashMap::new();
        reference.insert("A".to_string(), 0.48);
        reference.insert("B".to_string(), 0.52);

        let validation = DistributionValidation::validate(&observed, &reference, 0.1);
        assert!(validation.is_valid); // Within 10% tolerance
        assert!(validation.max_deviation < 0.1);
    }
}
