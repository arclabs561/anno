//! Statistical metric types with variance and confidence intervals.
//!
//! This module provides [`MetricStats`], the canonical type for representing
//! evaluation metrics with statistical information. It unifies the previously
//! separate `MetricWithVariance` and `MetricWithCI` types.

use serde::{Deserialize, Serialize};

/// Statistical metrics with variance and confidence intervals.
///
/// This is the canonical type for evaluation metrics that need to track
/// statistical properties across multiple runs or samples.
///
/// # Features
///
/// - Mean, standard deviation, and min/max range
/// - 95% confidence interval (both ± and lower/upper bounds)
/// - Sample count for determining statistical significance
/// - Formatting helpers for display
///
/// # Examples
///
/// ```rust
/// use anno::core::MetricStats;
///
/// let stats = MetricStats::from_samples(&[0.85, 0.87, 0.82, 0.88, 0.84]);
/// println!("F1: {}", stats.format_ci());
/// println!("Range: {}", stats.format_range());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MetricStats {
    /// Mean value of the metric
    pub mean: f64,
    /// Standard deviation (sample, Bessel-corrected)
    pub std_dev: f64,
    /// 95% confidence interval half-width (±)
    pub ci_half_width: f64,
    /// 95% CI lower bound
    pub ci_lower: f64,
    /// 95% CI upper bound
    pub ci_upper: f64,
    /// Minimum observed value
    pub min: f64,
    /// Maximum observed value
    pub max: f64,
    /// Number of samples
    pub n: usize,
}

impl Default for MetricStats {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std_dev: 0.0,
            ci_half_width: 0.0,
            ci_lower: 0.0,
            ci_upper: 0.0,
            min: 0.0,
            max: 0.0,
            n: 0,
        }
    }
}

impl MetricStats {
    /// Create from a slice of sample values.
    ///
    /// Uses sample standard deviation (Bessel's correction) and
    /// t-distribution approximation for 95% CI.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use anno::core::MetricStats;
    ///
    /// let stats = MetricStats::from_samples(&[0.80, 0.82, 0.85, 0.83, 0.80]);
    /// assert!((stats.mean - 0.82).abs() < 0.01);
    /// assert_eq!(stats.n, 5);
    /// ```
    #[must_use]
    pub fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }

        let n = samples.len();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let std_dev = if n > 1 {
            let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        // 95% CI using t-distribution approximation
        // For n >= 30, use z = 1.96; otherwise approximate with t
        let t_value = if n >= 30 {
            1.96
        } else if n >= 10 {
            2.0 + 0.1 / (n as f64).sqrt()
        } else {
            // More conservative for very small samples
            2.5
        };

        let ci_half_width = if n > 1 {
            t_value * std_dev / (n as f64).sqrt()
        } else {
            0.0
        };

        let ci_lower = mean - ci_half_width;
        let ci_upper = mean + ci_half_width;

        Self {
            mean,
            std_dev,
            ci_half_width,
            ci_lower,
            ci_upper,
            min,
            max,
            n,
        }
    }

    /// Create from a single value (no variance).
    #[must_use]
    pub fn from_single(value: f64) -> Self {
        Self {
            mean: value,
            std_dev: 0.0,
            ci_half_width: 0.0,
            ci_lower: value,
            ci_upper: value,
            min: value,
            max: value,
            n: 1,
        }
    }

    /// Format as "mean ± ci" string (percentages).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use anno::core::MetricStats;
    ///
    /// let stats = MetricStats::from_samples(&[0.85, 0.87, 0.83]);
    /// println!("{}", stats.format_ci());
    /// ```
    #[must_use]
    pub fn format_ci(&self) -> String {
        if self.n == 0 {
            return "N/A".to_string();
        }
        format!(
            "{:.1}% ± {:.1}%",
            self.mean * 100.0,
            self.ci_half_width * 100.0
        )
    }

    /// Format as "mean (min-max)" string (percentages).
    #[must_use]
    pub fn format_range(&self) -> String {
        if self.n == 0 {
            return "N/A".to_string();
        }
        format!(
            "{:.1}% ({:.1}%-{:.1}%)",
            self.mean * 100.0,
            self.min * 100.0,
            self.max * 100.0
        )
    }

    /// Format as "mean [ci_lower, ci_upper]" string (percentages).
    #[must_use]
    pub fn format_ci_bounds(&self) -> String {
        if self.n == 0 {
            return "N/A".to_string();
        }
        format!(
            "{:.1}% [{:.1}%, {:.1}%]",
            self.mean * 100.0,
            self.ci_lower * 100.0,
            self.ci_upper * 100.0
        )
    }

    /// Get coefficient of variation (CV = std_dev / mean).
    ///
    /// Lower CV indicates more stable/consistent results.
    /// - CV < 0.05: Excellent stability
    /// - CV < 0.10: Good stability
    /// - CV < 0.20: Moderate stability
    /// - CV >= 0.20: High variance, investigate
    #[must_use]
    pub fn coefficient_of_variation(&self) -> f64 {
        if self.mean.abs() < 1e-10 {
            0.0
        } else {
            self.std_dev / self.mean
        }
    }

    /// Check if results are statistically stable.
    ///
    /// Returns true if coefficient of variation is below threshold.
    #[must_use]
    pub fn is_stable(&self, cv_threshold: f64) -> bool {
        self.coefficient_of_variation() < cv_threshold
    }

    /// Get 95% CI as a tuple (lower, upper).
    ///
    /// For compatibility with code expecting tuple representation.
    #[must_use]
    pub fn ci_95_tuple(&self) -> (f64, f64) {
        (self.ci_lower, self.ci_upper)
    }

    /// Merge with another MetricStats (combine samples).
    ///
    /// Uses Welford's online algorithm for numerically stable combination.
    #[must_use]
    pub fn merge(&self, other: &MetricStats) -> MetricStats {
        if self.n == 0 {
            return *other;
        }
        if other.n == 0 {
            return *self;
        }

        let n = self.n + other.n;
        let mean = (self.mean * self.n as f64 + other.mean * other.n as f64) / n as f64;

        // Combined variance using parallel algorithm
        let delta = other.mean - self.mean;
        let m2_self = self.std_dev.powi(2) * (self.n - 1) as f64;
        let m2_other = other.std_dev.powi(2) * (other.n - 1) as f64;
        let m2 = m2_self + m2_other + delta.powi(2) * self.n as f64 * other.n as f64 / n as f64;
        let std_dev = if n > 1 {
            (m2 / (n - 1) as f64).sqrt()
        } else {
            0.0
        };

        let t_value = if n >= 30 {
            1.96
        } else {
            2.0 + 0.1 / (n as f64).sqrt()
        };
        let ci_half_width = if n > 1 {
            t_value * std_dev / (n as f64).sqrt()
        } else {
            0.0
        };

        MetricStats {
            mean,
            std_dev,
            ci_half_width,
            ci_lower: mean - ci_half_width,
            ci_upper: mean + ci_half_width,
            min: self.min.min(other.min),
            max: self.max.max(other.max),
            n,
        }
    }
}

impl std::fmt::Display for MetricStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format_ci())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_samples() {
        let samples = vec![0.80, 0.82, 0.85, 0.83, 0.80];
        let stats = MetricStats::from_samples(&samples);

        assert!((stats.mean - 0.82).abs() < 0.01);
        assert!(stats.std_dev > 0.0);
        assert!(stats.ci_half_width > 0.0);
        assert_eq!(stats.n, 5);
        assert!((stats.min - 0.80).abs() < 0.001);
        assert!((stats.max - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_from_single() {
        let stats = MetricStats::from_single(0.85);

        assert!((stats.mean - 0.85).abs() < 0.001);
        assert!((stats.std_dev - 0.0).abs() < 0.001);
        assert_eq!(stats.n, 1);
    }

    #[test]
    fn test_empty_samples() {
        let stats = MetricStats::from_samples(&[]);

        assert_eq!(stats.n, 0);
        assert!((stats.mean - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_coefficient_of_variation() {
        let stable = MetricStats::from_samples(&[0.85, 0.85, 0.85, 0.85, 0.85]);
        let variable = MetricStats::from_samples(&[0.60, 0.70, 0.80, 0.90, 1.00]);

        assert!(stable.coefficient_of_variation() < 0.01);
        assert!(variable.coefficient_of_variation() > 0.1);
    }

    #[test]
    fn test_ci_bounds() {
        let stats = MetricStats::from_samples(&[0.80, 0.82, 0.85, 0.83, 0.80]);

        assert!(stats.ci_lower <= stats.mean);
        assert!(stats.ci_upper >= stats.mean);
        assert!((stats.ci_upper - stats.ci_lower - 2.0 * stats.ci_half_width).abs() < 0.001);
    }

    #[test]
    fn test_merge() {
        let a = MetricStats::from_samples(&[0.80, 0.82, 0.84]);
        let b = MetricStats::from_samples(&[0.86, 0.88, 0.90]);
        let merged = a.merge(&b);

        assert_eq!(merged.n, 6);
        assert!((merged.mean - 0.85).abs() < 0.01);
        assert!((merged.min - 0.80).abs() < 0.001);
        assert!((merged.max - 0.90).abs() < 0.001);
    }

    #[test]
    fn test_serde_roundtrip() {
        let stats = MetricStats::from_samples(&[0.85, 0.87, 0.83]);
        let json = serde_json::to_string(&stats).expect("serialize MetricStats");
        let recovered: MetricStats = serde_json::from_str(&json).expect("deserialize MetricStats");

        assert!((stats.mean - recovered.mean).abs() < 0.001);
        assert_eq!(stats.n, recovered.n);
    }

    #[test]
    fn test_format() {
        let stats = MetricStats::from_samples(&[0.85, 0.87, 0.83]);

        let ci = stats.format_ci();
        assert!(ci.contains('%'));

        let range = stats.format_range();
        assert!(range.contains('-'));

        let bounds = stats.format_ci_bounds();
        assert!(bounds.contains('['));
    }

    #[test]
    fn test_format_empty_and_single() {
        let empty = MetricStats::from_samples(&[]);
        assert_eq!(empty.format_ci(), "N/A");
        assert_eq!(empty.format_range(), "N/A");
        assert_eq!(empty.format_ci_bounds(), "N/A");

        let single = MetricStats::from_single(0.90);
        let ci = single.format_ci();
        assert!(ci.contains("90.0%"), "single format_ci: {}", ci);
        assert!(ci.contains("0.0%"), "single should have zero CI: {}", ci);
    }

    #[test]
    fn test_is_stable() {
        let stable = MetricStats::from_samples(&[0.85, 0.85, 0.85, 0.85]);
        assert!(stable.is_stable(0.01));

        let variable = MetricStats::from_samples(&[0.50, 0.70, 0.90, 1.00]);
        assert!(!variable.is_stable(0.05));
    }

    #[test]
    fn test_merge_with_empty() {
        let stats = MetricStats::from_samples(&[0.80, 0.85]);
        let empty = MetricStats::default();

        let merged_left = empty.merge(&stats);
        assert_eq!(merged_left.n, stats.n);
        assert!((merged_left.mean - stats.mean).abs() < 0.001);

        let merged_right = stats.merge(&empty);
        assert_eq!(merged_right.n, stats.n);
        assert!((merged_right.mean - stats.mean).abs() < 0.001);
    }

    #[test]
    fn test_ci_95_tuple() {
        let stats = MetricStats::from_samples(&[0.80, 0.82, 0.84, 0.86, 0.88]);
        let (lower, upper) = stats.ci_95_tuple();
        assert!(lower < stats.mean);
        assert!(upper > stats.mean);
        assert!((upper - lower - 2.0 * stats.ci_half_width).abs() < 0.001);
    }

    #[test]
    fn test_display_uses_format_ci() {
        let stats = MetricStats::from_samples(&[0.85, 0.87, 0.83]);
        let display = format!("{}", stats);
        assert_eq!(display, stats.format_ci());
    }
}
