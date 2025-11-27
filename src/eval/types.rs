//! Evaluation types: MetricValue, GoalCheckResult, etc.
//!
//! These are shared primitives for evaluation that can be reused
//! across NER evaluation and other evaluation tasks.

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A type-safe metric value bounded to [0.0, 1.0].
///
/// Ensures metrics like precision, recall, and F1 are always valid.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, PartialOrd)]
#[serde(transparent)]
pub struct MetricValue(f64);

/// A metric with variance and confidence interval.
///
/// Tracks the mean, standard deviation, and 95% confidence interval
/// for a metric computed across multiple samples/runs/datasets.
///
/// # Example
///
/// ```rust
/// use anno::eval::MetricWithVariance;
///
/// let metric = MetricWithVariance::from_samples(&[0.85, 0.87, 0.82, 0.88, 0.84]);
/// println!("F1: {:.1}% ± {:.1}% (95% CI)", metric.mean * 100.0, metric.ci_95 * 100.0);
/// // F1: 85.2% ± 2.1% (95% CI)
/// ```
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct MetricWithVariance {
    /// Mean value of the metric
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// 95% confidence interval (±)
    pub ci_95: f64,
    /// Minimum observed value
    pub min: f64,
    /// Maximum observed value
    pub max: f64,
    /// Number of samples
    pub n: usize,
}

impl MetricWithVariance {
    /// Create from a slice of sample values.
    ///
    /// Uses sample standard deviation (Bessel's correction) and
    /// t-distribution approximation for 95% CI.
    pub fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self {
                mean: 0.0,
                std_dev: 0.0,
                ci_95: 0.0,
                min: 0.0,
                max: 0.0,
                n: 0,
            };
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
        } else {
            // Conservative t-value approximation for smaller samples
            2.0 + 0.1 / (n as f64).sqrt()
        };
        let ci_95 = if n > 1 {
            t_value * std_dev / (n as f64).sqrt()
        } else {
            0.0
        };

        Self {
            mean,
            std_dev,
            ci_95,
            min,
            max,
            n,
        }
    }

    /// Format as "mean ± ci95" string.
    pub fn format_with_ci(&self) -> String {
        if self.n == 0 {
            return "N/A".to_string();
        }
        format!("{:.1}% ± {:.1}%", self.mean * 100.0, self.ci_95 * 100.0)
    }

    /// Format as "mean (min-max)" string.
    pub fn format_with_range(&self) -> String {
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

    /// Get coefficient of variation (CV = std_dev / mean).
    pub fn coefficient_of_variation(&self) -> f64 {
        if self.mean.abs() < 1e-10 {
            0.0
        } else {
            self.std_dev / self.mean
        }
    }
}

impl Default for MetricWithVariance {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std_dev: 0.0,
            ci_95: 0.0,
            min: 0.0,
            max: 0.0,
            n: 0,
        }
    }
}

impl std::fmt::Display for MetricWithVariance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format_with_ci())
    }
}

impl MetricValue {
    /// Create a new MetricValue, clamping to [0.0, 1.0].
    ///
    /// # Example
    /// ```
    /// use anno::eval::MetricValue;
    /// let v = MetricValue::new(0.95);
    /// assert!((v.get() - 0.95).abs() < 1e-6);
    /// ```
    pub fn new(value: f64) -> Self {
        MetricValue(value.clamp(0.0, 1.0))
    }

    /// Try to create a MetricValue, returning error if out of bounds.
    pub fn try_new(value: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&value) {
            return Err(Error::InvalidInput(format!(
                "MetricValue must be in [0.0, 1.0], got {}",
                value
            )));
        }
        Ok(MetricValue(value))
    }

    /// Get the underlying value.
    #[inline]
    pub fn get(&self) -> f64 {
        self.0
    }
}

impl Default for MetricValue {
    fn default() -> Self {
        MetricValue(0.0)
    }
}

impl std::fmt::Display for MetricValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.4}", self.0)
    }
}

impl From<f64> for MetricValue {
    fn from(value: f64) -> Self {
        MetricValue::new(value)
    }
}

/// Result of checking evaluation goals.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GoalCheckResult {
    /// Whether all goals were met.
    pub passed: bool,
    /// Individual goal check results.
    pub checks: HashMap<String, GoalCheck>,
    /// Summary message.
    pub summary: Option<String>,
}

impl GoalCheckResult {
    /// Create a new GoalCheckResult (defaults to passed = true).
    pub fn new() -> Self {
        Self {
            passed: true,
            checks: HashMap::new(),
            summary: None,
        }
    }

    /// Add a goal check result.
    pub fn add_check(&mut self, name: impl Into<String>, check: GoalCheck) {
        if !check.passed {
            self.passed = false;
        }
        self.checks.insert(name.into(), check);
    }

    /// Add a failure (convenience method for add_check with fail).
    pub fn add_failure(&mut self, name: impl Into<String>, actual: f64, threshold: f64) {
        self.add_check(name, GoalCheck::fail(threshold, actual));
    }

    /// Add a success (convenience method for add_check with pass).
    pub fn add_success(&mut self, name: impl Into<String>, actual: f64, threshold: f64) {
        self.add_check(name, GoalCheck::pass(threshold, actual));
    }

    /// Get number of passed checks.
    pub fn passed_count(&self) -> usize {
        self.checks.values().filter(|c| c.passed).count()
    }

    /// Get number of failed checks.
    pub fn failed_count(&self) -> usize {
        self.checks.values().filter(|c| !c.passed).count()
    }
}

/// Individual goal check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalCheck {
    /// Whether this goal was met.
    pub passed: bool,
    /// Expected threshold.
    pub threshold: f64,
    /// Actual value achieved.
    pub actual: f64,
    /// Optional message.
    pub message: Option<String>,
}

impl GoalCheck {
    /// Create a new goal check.
    pub fn new(passed: bool, threshold: f64, actual: f64) -> Self {
        Self {
            passed,
            threshold,
            actual,
            message: None,
        }
    }

    /// Create a passing check.
    pub fn pass(threshold: f64, actual: f64) -> Self {
        Self::new(true, threshold, actual)
    }

    /// Create a failing check.
    pub fn fail(threshold: f64, actual: f64) -> Self {
        Self::new(false, threshold, actual)
    }

    /// Add a message to the check.
    pub fn with_message(mut self, msg: impl Into<String>) -> Self {
        self.message = Some(msg.into());
        self
    }
}

// =============================================================================
// Label Shift Quantification (Familiarity-inspired)
// =============================================================================

/// Label shift between training and evaluation entity types.
///
/// # Why This Matters
///
/// Imagine you trained a model on `{PER, ORG, LOC}` and then evaluate on
/// `{PERSON, COMPANY, CITY}`. Is that zero-shot? Technically yes (new labels).
/// Practically no (same concepts).
///
/// ```text
/// ┌─────────────────────────────────────────────────────────────────────────┐
/// │                    THE LABEL SHIFT PROBLEM                              │
/// ├─────────────────────────────────────────────────────────────────────────┤
/// │                                                                         │
/// │  TRAINING LABELS           EVAL LABELS         ARE THEY THE SAME?       │
/// │  ───────────────           ───────────         ──────────────────       │
/// │                                                                         │
/// │  PER ───────────────────── PERSON             ✓ Obviously (renamed)     │
/// │  ORG ───────────────────── COMPANY            ✓ Subset relationship     │
/// │  LOC ───────────────────── CITY               ✓ Subset relationship     │
/// │                                                                         │
/// │  ??? ←─────────────────── DISEASE            ✗ TRUE ZERO-SHOT!         │
/// │  ??? ←─────────────────── DRUG               ✗ TRUE ZERO-SHOT!         │
/// │                                                                         │
/// │  If 80% of eval types have training equivalents, your F1 is inflated.   │
/// └─────────────────────────────────────────────────────────────────────────┘
/// ```
///
/// # Embedding Space View
///
/// Labels that seem different can be close in embedding space:
///
/// ```text
///                    EMBEDDING SPACE (2D projection)
///                    ───────────────────────────────
///
///            PER ●───────────────● PERSON
///                      │
///                 very close in
///                embedding space
///
///            ORG ●─────● COMPANY
///
///            LOC ●─────────● CITY
///
///
///                                        ● DISEASE    ← Far from all
///                                                       training types!
///                                        ● DRUG       ← This is TRUE
///                                                       zero-shot.
///
/// F1 on {PERSON, COMPANY, CITY}:  85%  (but model "knew" these)
/// F1 on {DISEASE, DRUG}:          45%  (honest zero-shot)
/// ```
///
/// # Research Context (arXiv:2412.10121 "Familiarity")
///
/// Key findings from Golde et al. (2024):
/// - 80%+ label overlap in NuNER/PileNER → inflated F1 scores
/// - True zero-shot: evaluate only on types NOT in training
/// - Familiarity = semantic similarity × frequency weighting
///
/// # Example
///
/// ```rust
/// use anno::eval::LabelShift;
///
/// let shift = LabelShift {
///     overlap_ratio: 0.85,    // 85% of eval types in train
///     familiarity: 0.72,      // Semantic similarity score
///     true_zero_shot_types: vec!["DISEASE".into(), "DRUG".into()],
///     transfer_difficulty: "low".into(),
/// };
///
/// // High overlap = easy transfer, but NOT true zero-shot
/// assert!(shift.is_inflated());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelShift {
    /// Fraction of eval types found in training data (exact string match).
    pub overlap_ratio: f64,
    
    /// Familiarity score: semantic similarity weighted by frequency.
    /// Range: [0, 1]. Higher = more similar training/eval types.
    pub familiarity: f64,
    
    /// Entity types in eval NOT present in training (true zero-shot).
    pub true_zero_shot_types: Vec<String>,
    
    /// Qualitative difficulty: "low", "medium", "high".
    pub transfer_difficulty: String,
}

impl LabelShift {
    /// Check if F1 scores are likely inflated due to high label overlap.
    ///
    /// Threshold from Familiarity paper: >0.8 overlap is concerning.
    #[must_use]
    pub fn is_inflated(&self) -> bool {
        self.overlap_ratio > 0.8 || self.familiarity > 0.85
    }

    /// Get count of true zero-shot types.
    #[must_use]
    pub fn true_zero_shot_count(&self) -> usize {
        self.true_zero_shot_types.len()
    }

    /// Compute label shift from training and eval type sets.
    ///
    /// # Arguments
    /// * `train_types` - Entity types seen during training
    /// * `eval_types` - Entity types in evaluation benchmark
    ///
    /// # Note
    ///
    /// This is a simple string-match overlap. For semantic similarity
    /// (true Familiarity), use embeddings. See arXiv:2412.10121.
    #[must_use]
    pub fn from_type_sets(train_types: &[String], eval_types: &[String]) -> Self {
        let train_set: std::collections::HashSet<_> = train_types.iter().collect();
        let eval_set: std::collections::HashSet<_> = eval_types.iter().collect();

        // Exact match overlap
        let overlap_count = eval_set.intersection(&train_set).count();
        let overlap_ratio = if eval_types.is_empty() {
            0.0
        } else {
            overlap_count as f64 / eval_types.len() as f64
        };

        // True zero-shot = eval types NOT in training
        let true_zero_shot_types: Vec<String> = eval_set
            .difference(&train_set)
            .map(|s| (*s).clone())
            .collect();

        // Simple familiarity heuristic (proper version needs embeddings)
        let familiarity = overlap_ratio; // Placeholder

        let transfer_difficulty = if overlap_ratio > 0.8 {
            "low"
        } else if overlap_ratio > 0.4 {
            "medium"
        } else {
            "high"
        }
        .to_string();

        Self {
            overlap_ratio,
            familiarity,
            true_zero_shot_types,
            transfer_difficulty,
        }
    }
}

impl std::fmt::Display for LabelShift {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LabelShift(overlap={:.0}%, familiarity={:.2}, zero-shot={}, difficulty={})",
            self.overlap_ratio * 100.0,
            self.familiarity,
            self.true_zero_shot_types.len(),
            self.transfer_difficulty
        )
    }
}

// =============================================================================
// Coreference Chain Statistics (arXiv:2401.00238 inspired)
// =============================================================================

/// Statistics for stratified coreference evaluation.
///
/// # Why Chain Length Matters: A Narrative
///
/// Imagine analyzing "Pride and Prejudice":
///
/// ```text
/// ┌─────────────────────────────────────────────────────────────────────────┐
/// │                    COREFERENCE IN A NOVEL                               │
/// ├─────────────────────────────────────────────────────────────────────────┤
/// │                                                                         │
/// │  LONG CHAINS (>10 mentions) - THE PROTAGONISTS                          │
/// │  ─────────────────────────────────────────────                          │
/// │                                                                         │
/// │  "Elizabeth" ─── "she" ─── "Lizzy" ─── "her" ─── "Miss Bennet" ───...  │
/// │       │            │          │          │            │                 │
/// │       └────────────┴──────────┴──────────┴────────────┘                 │
/// │                         800+ mentions                                   │
/// │                                                                         │
/// │  Getting these right = understanding the PLOT.                          │
/// │  Who did what to whom? What's Elizabeth's arc?                          │
/// │                                                                         │
/// │  SHORT CHAINS (2-10 mentions) - SECONDARY CHARACTERS                    │
/// │  ───────────────────────────────────────────────────                    │
/// │                                                                         │
/// │  "Mr. Collins" ─── "he" ─── "the clergyman"                             │
/// │       │              │             │                                    │
/// │       └──────────────┴─────────────┘                                    │
/// │                  15 mentions                                            │
/// │                                                                         │
/// │  Important for context, but errors here are less catastrophic.          │
/// │                                                                         │
/// │  SINGLETONS (1 mention) - BACKGROUND                                    │
/// │  ───────────────────────────────────────                                │
/// │                                                                         │
/// │  "a tall man" ─── (no other mentions)                                   │
/// │  "the servant" ─── (no other mentions)                                  │
/// │                                                                         │
/// │  These aren't really coreference—they're just entity detection.         │
/// │  Including them in CoNLL F1 INFLATES your score.                        │
/// └─────────────────────────────────────────────────────────────────────────┘
/// ```
///
/// # The Problem with Averaged Metrics
///
/// ```text
/// Model Performance:
///
///   Long chains (protagonists):  92% F1  ← Model understands plot!
///   Short chains (secondary):    71% F1  ← Decent
///   Singletons (background):     45% F1  ← Poor, but who cares?
///
/// CoNLL F1 (averaged):           65% F1  ← Misleadingly low!
///
/// The average HIDES that the model is excellent at what matters most.
///
/// ALWAYS report stratified metrics:
///   • "Protagonist F1: 92%"
///   • "Secondary F1: 71%"
///   • "Singleton F1: 45% (excluded from final score)"
/// ```
///
/// # Research Context (arXiv:2401.00238)
///
/// "How to Evaluate Coreference in Literary Texts?"
/// - A single CoNLL F1 score is "uninformative, or even misleading."
/// - Stratify by chain length for interpretable results.
///
/// # Example
///
/// ```rust
/// use anno::eval::CorefChainStats;
///
/// let stats = CorefChainStats {
///     long_chain_count: 3,      // Main characters
///     short_chain_count: 15,    // Secondary
///     singleton_count: 42,      // Isolated
///     long_chain_f1: 0.92,      // Good on main characters
///     short_chain_f1: 0.71,     // Weaker on secondary
///     singleton_f1: 0.45,       // Poor on singletons
/// };
///
/// // Report metrics separately, not averaged
/// println!("Main characters: {:.1}% F1", stats.long_chain_f1 * 100.0);
/// ```
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct CorefChainStats {
    /// Number of long chains (>10 mentions).
    pub long_chain_count: usize,
    /// Number of short chains (2-10 mentions).
    pub short_chain_count: usize,
    /// Number of singletons (1 mention).
    pub singleton_count: usize,
    /// F1 score on long chains only.
    pub long_chain_f1: f64,
    /// F1 score on short chains only.
    pub short_chain_f1: f64,
    /// F1 score on singletons (if evaluated).
    pub singleton_f1: f64,
}

impl CorefChainStats {
    /// Total chain count.
    #[must_use]
    pub fn total_chains(&self) -> usize {
        self.long_chain_count + self.short_chain_count + self.singleton_count
    }

    /// Weighted F1 (by chain count).
    ///
    /// Note: This is NOT the same as CoNLL F1 (which averages MUC, B³, CEAF-e).
    #[must_use]
    pub fn weighted_f1(&self) -> f64 {
        let total = self.total_chains();
        if total == 0 {
            return 0.0;
        }

        let weighted_sum = self.long_chain_f1 * self.long_chain_count as f64
            + self.short_chain_f1 * self.short_chain_count as f64
            + self.singleton_f1 * self.singleton_count as f64;

        weighted_sum / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_value_clamping() {
        assert_eq!(MetricValue::new(0.5).get(), 0.5);
        assert_eq!(MetricValue::new(-0.5).get(), 0.0);
        assert_eq!(MetricValue::new(1.5).get(), 1.0);
    }

    #[test]
    fn test_metric_value_try_new() {
        assert!(MetricValue::try_new(0.5).is_ok());
        assert!(MetricValue::try_new(-0.1).is_err());
        assert!(MetricValue::try_new(1.1).is_err());
    }

    #[test]
    fn test_goal_check_result() {
        let mut result = GoalCheckResult::new();
        assert!(result.passed);

        result.add_check("precision", GoalCheck::pass(0.8, 0.85));
        assert!(result.passed);

        result.add_check("recall", GoalCheck::fail(0.9, 0.75));
        assert!(!result.passed);

        assert_eq!(result.passed_count(), 1);
        assert_eq!(result.failed_count(), 1);
    }

    #[test]
    fn test_metric_with_variance_from_samples() {
        let samples = vec![0.85, 0.87, 0.82, 0.88, 0.84];
        let m = MetricWithVariance::from_samples(&samples);

        // Mean should be 0.852
        assert!((m.mean - 0.852).abs() < 0.001);
        assert_eq!(m.n, 5);
        assert!((m.min - 0.82).abs() < 0.001);
        assert!((m.max - 0.88).abs() < 0.001);
        assert!(m.std_dev > 0.0);
        assert!(m.ci_95 > 0.0);
    }

    #[test]
    fn test_metric_with_variance_empty() {
        let m = MetricWithVariance::from_samples(&[]);
        assert_eq!(m.n, 0);
        assert_eq!(m.mean, 0.0);
        assert_eq!(m.format_with_ci(), "N/A");
    }

    #[test]
    fn test_metric_with_variance_single() {
        let m = MetricWithVariance::from_samples(&[0.9]);
        assert!((m.mean - 0.9).abs() < 0.001);
        assert_eq!(m.std_dev, 0.0);
        assert_eq!(m.ci_95, 0.0);
        assert_eq!(m.n, 1);
    }

    #[test]
    fn test_metric_with_variance_format() {
        let samples = vec![0.85, 0.87, 0.82, 0.88, 0.84];
        let m = MetricWithVariance::from_samples(&samples);
        
        // Should format nicely
        let formatted = m.format_with_ci();
        assert!(formatted.contains("%"));
        assert!(formatted.contains("±"));
        
        let range = m.format_with_range();
        assert!(range.contains("82.0%"));
        assert!(range.contains("88.0%"));
    }
}
