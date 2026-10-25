//! Confidence calibration metrics for NER evaluation.
//!
//! Measures whether a model's confidence scores align with actual correctness.
//! A well-calibrated model should have high confidence for correct predictions
//! and low confidence for incorrect ones.
//!
//! # ⚠️ Important: Confidence Score Semantics
//!
//! Calibration metrics are only meaningful for **probabilistically calibrated**
//! confidence scores (i.e., scores that approximate P(correct|prediction)).
//!
//! | Backend | `ExtractionMethod` | Calibrated? | Notes |
//! |---------|-------------------|-------------|-------|
//! | BertNEROnnx, GLiNEROnnx | `Neural` | ✓ Yes | Softmax probabilities |
//! | RegexNER | `Pattern` | ✗ No | Hardcoded values (e.g., 0.95) |
//! | HeuristicNER | `Heuristic` | ✗ No | Rule-based scores |
//! | StackedNER | Mixed | Partial | Depends on entity type |
//!
//! **Running calibration analysis on RegexNER or HeuristicNER produces
//! meaningless results.** Use `ExtractionMethod::is_calibrated()` to check.
//!
//! # Research Background
//!
//! Calibration is critical for production NER systems where:
//! - Downstream systems need reliable confidence thresholds
//! - Human review should focus on low-confidence predictions
//! - False confidence is worse than admitted uncertainty
//!
//! See: Guo et al. (2017) "On Calibration of Modern Neural Networks" (arXiv:1706.04599)
//!
//! # Key Metrics
//!
//! - **Expected Calibration Error (ECE)**: Weighted average of per-bin calibration error
//! - **Maximum Calibration Error (MCE)**: Worst-case calibration in any bin
//! - **Brier Score**: Mean squared error of probabilistic predictions
//! - **Confidence Gap**: Difference between avg confidence on correct vs incorrect
//!
//! # Example
//!
//! ```rust
//! use anno_eval::eval::calibration::{CalibrationEvaluator, CalibrationResults};
//!
//! // Only use with probabilistic confidence scores (e.g., from neural models)
//! let predictions = vec![
//!     (0.95, true),   // High confidence, correct
//!     (0.80, true),   // Medium confidence, correct
//!     (0.60, false),  // Low confidence, incorrect (good!)
//!     (0.90, false),  // High confidence, incorrect (bad!)
//! ];
//!
//! let results = CalibrationEvaluator::compute(&predictions);
//! println!("ECE: {:.3}", results.ece);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Calibration Results
// =============================================================================

/// Results of calibration evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResults {
    /// Expected Calibration Error (lower is better, 0 = perfect)
    pub ece: f64,
    /// Maximum Calibration Error (lower is better)
    pub mce: f64,
    /// Brier Score (lower is better, 0 = perfect)
    pub brier_score: f64,
    /// Average confidence of correct predictions
    pub avg_confidence_correct: f64,
    /// Average confidence of incorrect predictions
    pub avg_confidence_incorrect: f64,
    /// Confidence gap (correct - incorrect, higher is better)
    pub confidence_gap: f64,
    /// Reliability diagram data (bin_midpoint -> (avg_confidence, accuracy, count))
    pub reliability_bins: Vec<ReliabilityBin>,
    /// Total predictions evaluated
    pub total_predictions: usize,
    /// Accuracy at different confidence thresholds
    pub threshold_accuracy: HashMap<String, ThresholdMetrics>,
}

/// A single bin in the reliability diagram.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityBin {
    /// Bin range (e.g., 0.0-0.1)
    pub range: (f64, f64),
    /// Average confidence in this bin
    pub avg_confidence: f64,
    /// Accuracy (fraction correct) in this bin
    pub accuracy: f64,
    /// Number of predictions in this bin
    pub count: usize,
    /// Calibration error for this bin: |accuracy - avg_confidence|
    pub calibration_error: f64,
}

/// Metrics at a specific confidence threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdMetrics {
    /// Accuracy of predictions above this threshold
    pub accuracy: f64,
    /// Coverage (fraction of predictions above threshold)
    pub coverage: f64,
    /// Count of predictions above threshold
    pub count: usize,
}

// =============================================================================
// Calibration Evaluator
// =============================================================================

/// Evaluator for confidence calibration.
#[derive(Debug, Clone)]
pub struct CalibrationEvaluator {
    /// Number of bins for reliability diagram
    pub num_bins: usize,
    /// Confidence thresholds to evaluate
    pub thresholds: Vec<f64>,
}

impl Default for CalibrationEvaluator {
    fn default() -> Self {
        Self {
            num_bins: 10,
            thresholds: vec![0.5, 0.7, 0.8, 0.9, 0.95],
        }
    }
}

impl CalibrationEvaluator {
    /// Create a new evaluator with custom bins.
    pub fn new(num_bins: usize) -> Self {
        Self {
            num_bins,
            ..Default::default()
        }
    }

    /// Compute calibration metrics from (confidence, correct) pairs.
    pub fn compute(predictions: &[(f64, bool)]) -> CalibrationResults {
        Self::default().evaluate(predictions)
    }

    /// Evaluate calibration on predictions.
    ///
    /// Each prediction is a tuple of (confidence_score, is_correct).
    ///
    /// Computes:
    /// - **ECE**: `Σ(n_i / N) × |acc_i - conf_i|` where bins partition [0, 1]
    /// - **Brier Score**: `(1/N) × Σ(conf_i - target_i)²` where target is 1 if correct, 0 otherwise
    ///
    /// Reference: Guo et al. (2017) "On Calibration of Modern Neural Networks" (arXiv:1706.04599)
    pub fn evaluate(&self, predictions: &[(f64, bool)]) -> CalibrationResults {
        // `statskit` asserts that probabilities are finite and within [0, 1].
        // The previous implementation silently accepted invalid confidences, so
        // filter them out to preserve the permissive (non-panicking) behaviour.
        let valid: Vec<(f64, bool)> = predictions
            .iter()
            .copied()
            .filter(|(conf, _)| conf.is_finite() && (0.0..=1.0).contains(conf))
            .collect();

        if valid.is_empty() {
            return CalibrationResults {
                ece: 0.0,
                mce: 0.0,
                brier_score: 0.0,
                avg_confidence_correct: 0.0,
                avg_confidence_incorrect: 0.0,
                confidence_gap: 0.0,
                reliability_bins: Vec::new(),
                total_predictions: 0,
                threshold_accuracy: HashMap::new(),
            };
        }

        let y_true: Vec<usize> = valid.iter().map(|&(_, c)| usize::from(c)).collect();
        let y_prob: Vec<f64> = valid.iter().map(|&(c, _)| c).collect();

        let ece =
            statskit::calibration::expected_calibration_error(&y_true, &y_prob, self.num_bins);
        let mce = statskit::calibration::maximum_calibration_error(&y_true, &y_prob, self.num_bins);
        let brier_score = statskit::calibration::brier_score(&y_true, &y_prob);
        let bins = statskit::calibration::reliability_diagram(&y_true, &y_prob, self.num_bins);

        let bin_width = 1.0 / self.num_bins as f64;
        let reliability_bins = bins
            .into_iter()
            .map(|bin| {
                let idx =
                    ((bin.mean_predicted * self.num_bins as f64) as usize).min(self.num_bins - 1);
                ReliabilityBin {
                    range: (idx as f64 * bin_width, (idx + 1) as f64 * bin_width),
                    avg_confidence: bin.mean_predicted,
                    accuracy: bin.fraction_positive,
                    count: bin.count,
                    calibration_error: (bin.fraction_positive - bin.mean_predicted).abs(),
                }
            })
            .collect();

        // Compute confidence statistics
        let correct_confs: Vec<f64> = valid
            .iter()
            .filter(|(_, c)| *c)
            .map(|(conf, _)| *conf)
            .collect();
        let incorrect_confs: Vec<f64> = valid
            .iter()
            .filter(|(_, c)| !*c)
            .map(|(conf, _)| *conf)
            .collect();

        let avg_confidence_correct = if correct_confs.is_empty() {
            0.0
        } else {
            correct_confs.iter().sum::<f64>() / correct_confs.len() as f64
        };

        let avg_confidence_incorrect = if incorrect_confs.is_empty() {
            0.0
        } else {
            incorrect_confs.iter().sum::<f64>() / incorrect_confs.len() as f64
        };

        // Compute threshold metrics
        let mut threshold_accuracy = HashMap::new();
        for &threshold in &self.thresholds {
            let above: Vec<_> = valid.iter().filter(|(c, _)| *c >= threshold).collect();

            if above.is_empty() {
                threshold_accuracy.insert(
                    format!("{:.2}", threshold),
                    ThresholdMetrics {
                        accuracy: 0.0,
                        coverage: 0.0,
                        count: 0,
                    },
                );
            } else {
                let acc = above.iter().filter(|(_, correct)| *correct).count() as f64
                    / above.len() as f64;
                let cov = above.len() as f64 / valid.len() as f64;
                threshold_accuracy.insert(
                    format!("{:.2}", threshold),
                    ThresholdMetrics {
                        accuracy: acc,
                        coverage: cov,
                        count: above.len(),
                    },
                );
            }
        }

        CalibrationResults {
            ece,
            mce,
            brier_score,
            avg_confidence_correct,
            avg_confidence_incorrect,
            confidence_gap: avg_confidence_correct - avg_confidence_incorrect,
            reliability_bins,
            total_predictions: valid.len(),
            threshold_accuracy,
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Check if a model is well-calibrated.
///
/// Rules of thumb:
/// - ECE < 0.05: Well calibrated
/// - ECE 0.05-0.10: Moderately calibrated
/// - ECE > 0.10: Poorly calibrated
pub fn calibration_grade(ece: f64) -> &'static str {
    if ece < 0.05 {
        "Well calibrated"
    } else if ece < 0.10 {
        "Moderately calibrated"
    } else if ece < 0.15 {
        "Poorly calibrated"
    } else {
        "Very poorly calibrated"
    }
}

/// Check if confidence gap is healthy.
///
/// A healthy model should have higher confidence for correct predictions.
/// Gap > 0.2 suggests good confidence discrimination.
pub fn confidence_gap_grade(gap: f64) -> &'static str {
    if gap > 0.3 {
        "Excellent discrimination"
    } else if gap > 0.2 {
        "Good discrimination"
    } else if gap > 0.1 {
        "Moderate discrimination"
    } else if gap > 0.0 {
        "Weak discrimination"
    } else {
        "No discrimination (or reversed)"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_calibration() {
        // Perfect calibration: confidence equals accuracy
        let predictions = vec![
            (0.9, true),
            (0.9, true),
            (0.9, true),
            (0.9, true),
            (0.9, true),
            (0.9, true),
            (0.9, true),
            (0.9, true),
            (0.9, true),
            (0.9, false), // 9/10 correct at confidence 0.9
        ];

        let results = CalibrationEvaluator::compute(&predictions);

        // ECE should be very low for perfect calibration
        assert!(
            results.ece < 0.1,
            "ECE should be low for well-calibrated predictions"
        );
    }

    #[test]
    fn test_overconfident_model() {
        // Overconfident: high confidence but low accuracy
        let predictions = vec![
            (0.95, false),
            (0.95, false),
            (0.95, false),
            (0.95, true),
            (0.95, false), // 1/5 correct at confidence 0.95
        ];

        let results = CalibrationEvaluator::compute(&predictions);

        // ECE should be high for overconfident model
        assert!(
            results.ece > 0.5,
            "ECE should be high for overconfident predictions"
        );
    }

    #[test]
    fn test_confidence_gap() {
        let predictions = vec![
            (0.95, true),
            (0.90, true),
            (0.85, true),
            (0.30, false),
            (0.25, false),
            (0.20, false),
        ];

        let results = CalibrationEvaluator::compute(&predictions);

        assert!(
            results.avg_confidence_correct > 0.8,
            "Correct predictions should have high confidence"
        );
        assert!(
            results.avg_confidence_incorrect < 0.4,
            "Incorrect predictions should have low confidence"
        );
        assert!(
            results.confidence_gap > 0.4,
            "Should have large confidence gap"
        );
    }

    #[test]
    fn test_threshold_metrics() {
        let predictions = vec![
            (0.95, true),
            (0.85, true),
            (0.75, false),
            (0.65, true),
            (0.55, false),
        ];

        let results = CalibrationEvaluator::compute(&predictions);

        // At 0.80 threshold, only 2 predictions (0.95, 0.85), both correct
        let t80 = results.threshold_accuracy.get("0.80").unwrap();
        assert!((t80.accuracy - 1.0).abs() < 0.01, "Should be 100% at 0.80");
        assert!((t80.coverage - 0.4).abs() < 0.01, "Coverage should be 40%");
    }

    #[test]
    fn test_empty_predictions() {
        let results = CalibrationEvaluator::compute(&[]);
        assert_eq!(results.total_predictions, 0);
        assert_eq!(results.ece, 0.0);
    }

    #[test]
    fn test_calibration_grades() {
        assert_eq!(calibration_grade(0.03), "Well calibrated");
        assert_eq!(calibration_grade(0.07), "Moderately calibrated");
        assert_eq!(calibration_grade(0.12), "Poorly calibrated");
        assert_eq!(calibration_grade(0.25), "Very poorly calibrated");
    }

    #[test]
    fn test_entropy_single_source() {
        // Single source = zero entropy
        let scores = vec![0.9];
        let entropy = confidence_entropy(&scores);
        assert!(
            (entropy - 0.0).abs() < 0.001,
            "Single source should have 0 entropy"
        );
    }

    #[test]
    fn test_entropy_agreement() {
        // Sources agree = low entropy
        let scores = vec![0.9, 0.88, 0.92];
        let entropy = confidence_entropy(&scores);
        assert!(
            entropy < 0.5,
            "Agreeing sources should have low entropy: {}",
            entropy
        );
    }

    #[test]
    fn test_entropy_conflict() {
        // Sources disagree = high entropy
        let scores = vec![0.95, 0.05, 0.5, 0.8, 0.2];
        let entropy = confidence_entropy(&scores);
        assert!(
            entropy > 0.5,
            "Conflicting sources should have high entropy: {}",
            entropy
        );
    }

    #[test]
    fn test_entropy_filter() {
        let candidates = [
            ("Apple Inc.", vec![0.9, 0.88, 0.92]), // Agreement
            ("Apple", vec![0.95, 0.05, 0.5]),      // Conflict
            ("Microsoft", vec![0.85, 0.87]),       // Agreement
        ];

        let filter = EntropyFilter::new(0.6);
        let filtered: Vec<_> = candidates
            .iter()
            .filter(|(_, scores)| filter.should_keep(scores))
            .map(|(name, _)| *name)
            .collect();

        assert!(filtered.contains(&"Apple Inc."));
        assert!(filtered.contains(&"Microsoft"));
        assert!(
            !filtered.contains(&"Apple"),
            "Conflicting 'Apple' should be filtered"
        );
    }

    #[test]
    fn test_bit_identical_to_previous_implementation() {
        // Fixed input set whose previous manual implementation produced these
        // exact numbers. Pinned so future statskit changes are intentional.
        let predictions = vec![
            (0.95, true),
            (0.90, true),
            (0.85, false),
            (0.80, true),
            (0.75, false),
            (0.60, true),
            (0.55, false),
            (0.40, false),
            (0.25, true),
            (0.10, false),
        ];
        let results = CalibrationEvaluator::compute(&predictions);

        assert!((results.ece - 0.375).abs() < 1e-12);
        assert!((results.mce - 0.75).abs() < 1e-12);
        assert!((results.brier_score - 0.25325).abs() < 1e-12);
        assert!((results.avg_confidence_correct - 0.70).abs() < 1e-12);
        assert!((results.avg_confidence_incorrect - 0.53).abs() < 1e-12);
        assert!((results.confidence_gap - 0.17).abs() < 1e-12);
        assert_eq!(results.total_predictions, 10);

        // Reliability bins, in ascending bin-index order.
        assert_eq!(results.reliability_bins.len(), 8);
        let expected_bins = [
            ((0.1, 0.2), 0.10, 0.0, 1, 0.10),
            ((0.2, 0.3), 0.25, 1.0, 1, 0.75),
            ((0.4, 0.5), 0.40, 0.0, 1, 0.40),
            ((0.5, 0.6), 0.55, 0.0, 1, 0.55),
            ((0.6, 0.7), 0.60, 1.0, 1, 0.40),
            ((0.7, 0.8), 0.75, 0.0, 1, 0.75),
            ((0.8, 0.9), 0.825, 0.5, 2, 0.325),
            ((0.9, 1.0), 0.925, 1.0, 2, 0.075),
        ];
        for (bin, expected) in results.reliability_bins.iter().zip(expected_bins.iter()) {
            assert!((bin.range.0 - expected.0 .0).abs() < 1e-12);
            assert!((bin.range.1 - expected.0 .1).abs() < 1e-12);
            assert!((bin.avg_confidence - expected.1).abs() < 1e-12);
            assert!((bin.accuracy - expected.2).abs() < 1e-12);
            assert_eq!(bin.count, expected.3);
            assert!((bin.calibration_error - expected.4).abs() < 1e-12);
        }

        // Threshold metrics.
        let t50 = results.threshold_accuracy.get("0.50").unwrap();
        assert!((t50.accuracy - 4.0 / 7.0).abs() < 1e-12);
        assert!((t50.coverage - 0.7).abs() < 1e-12);
        assert_eq!(t50.count, 7);

        let t70 = results.threshold_accuracy.get("0.70").unwrap();
        assert!((t70.accuracy - 0.6).abs() < 1e-12);
        assert!((t70.coverage - 0.5).abs() < 1e-12);
        assert_eq!(t70.count, 5);

        let t80 = results.threshold_accuracy.get("0.80").unwrap();
        assert!((t80.accuracy - 0.75).abs() < 1e-12);
        assert!((t80.coverage - 0.4).abs() < 1e-12);
        assert_eq!(t80.count, 4);

        let t90 = results.threshold_accuracy.get("0.90").unwrap();
        assert!((t90.accuracy - 1.0).abs() < 1e-12);
        assert!((t90.coverage - 0.2).abs() < 1e-12);
        assert_eq!(t90.count, 2);

        let t95 = results.threshold_accuracy.get("0.95").unwrap();
        assert!((t95.accuracy - 1.0).abs() < 1e-12);
        assert!((t95.coverage - 0.1).abs() < 1e-12);
        assert_eq!(t95.count, 1);
    }

    #[test]
    fn test_non_finite_input_is_filtered() {
        let predictions = vec![
            (0.9, true),
            (f64::NAN, false),
            (0.8, true),
            (f64::INFINITY, false),
            (0.7, false),
        ];
        let results = CalibrationEvaluator::compute(&predictions);
        let expected = CalibrationEvaluator::compute(&[(0.9, true), (0.8, true), (0.7, false)]);

        assert_eq!(results.total_predictions, 3);
        assert!((results.ece - expected.ece).abs() < 1e-12);
        assert!((results.mce - expected.mce).abs() < 1e-12);
        assert!((results.brier_score - expected.brier_score).abs() < 1e-12);
        assert!((results.avg_confidence_correct - expected.avg_confidence_correct).abs() < 1e-12);
        assert!(
            (results.avg_confidence_incorrect - expected.avg_confidence_incorrect).abs() < 1e-12
        );
        assert!((results.confidence_gap - expected.confidence_gap).abs() < 1e-12);
        assert_eq!(
            results.reliability_bins.len(),
            expected.reliability_bins.len()
        );
        assert_eq!(
            results.threshold_accuracy.len(),
            expected.threshold_accuracy.len()
        );
    }
}

// =============================================================================
// Entropy-Based Conflict Detection (TruthfulRAG-style)
// =============================================================================

/// Compute disagreement metric for confidence scores from multiple sources.
///
/// # TruthfulRAG Research Background
///
/// When multiple sources provide confidence scores for the same entity/fact,
/// high disagreement indicates conflict. TruthfulRAG (EMNLP 2024) uses this
/// to identify facts that need verification:
///
/// - **Low disagreement**: Sources agree → likely reliable
/// - **High disagreement**: Sources disagree → needs human review or rejection
///
/// # Formula
///
/// Uses normalized standard deviation of scores:
/// ```text
/// disagreement = std_dev(scores) / 0.5
/// ```
/// where 0.5 is the maximum possible std dev for scores in \[0,1\].
/// This maps to \[0, 1\] where 0 = perfect agreement, 1 = maximum disagreement.
///
/// # Example
///
/// ```rust
/// use anno_eval::eval::calibration::confidence_entropy;
///
/// // Sources agree (low disagreement)
/// let scores = vec![0.9, 0.88, 0.92];
/// assert!(confidence_entropy(&scores) < 0.3);
///
/// // Sources disagree (high disagreement)
/// let scores = vec![0.95, 0.05, 0.5];
/// assert!(confidence_entropy(&scores) > 0.5);
/// ```
#[must_use]
pub fn confidence_entropy(scores: &[f64]) -> f64 {
    if scores.len() <= 1 {
        return 0.0; // Single source = no disagreement
    }

    // Compute standard deviation
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    // Use sample variance (Bessel's correction: n-1) for unbiased estimate
    let n = scores.len() as f64;
    let variance = if n > 1.0 {
        scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };
    let std_dev = variance.sqrt();

    // Normalize by maximum possible std dev for [0,1] scores
    // Max std dev is 0.5 (when half are 0 and half are 1)
    (std_dev / 0.5).min(1.0)
}

/// Compute variance of confidence scores (simpler alternative to entropy).
///
/// High variance indicates disagreement between sources.
#[must_use]
pub fn confidence_variance(scores: &[f64]) -> f64 {
    if scores.len() <= 1 {
        return 0.0;
    }

    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64
}

/// Filter for rejecting high-entropy (conflicting) entity extractions.
///
/// # Usage in RAG Systems
///
/// When multiple retrieval passes or models extract the same entity with
/// different confidences, use this filter to:
///
/// 1. Accept entities where sources agree (low entropy)
/// 2. Flag/reject entities where sources disagree (high entropy)
///
/// # Example
///
/// ```rust
/// use anno_eval::eval::calibration::EntropyFilter;
///
/// let filter = EntropyFilter::new(0.6);  // Reject if entropy > 0.6
///
/// // Multiple models extracted "Apple" with these confidences:
/// let apple_scores = vec![0.95, 0.05, 0.5];  // Disagreement
/// assert!(!filter.should_keep(&apple_scores), "Should reject conflicting extractions");
///
/// let microsoft_scores = vec![0.9, 0.88, 0.92];  // Agreement
/// assert!(filter.should_keep(&microsoft_scores), "Should keep agreeing extractions");
/// ```
#[derive(Debug, Clone)]
pub struct EntropyFilter {
    /// Maximum allowed entropy (0.0-1.0)
    pub max_entropy: f64,
    /// Minimum number of sources required
    pub min_sources: usize,
    /// Use variance instead of entropy (faster, simpler)
    pub use_variance: bool,
    /// Maximum variance threshold (if use_variance=true)
    pub max_variance: f64,
}

impl Default for EntropyFilter {
    fn default() -> Self {
        Self {
            max_entropy: 0.7, // Moderate threshold
            min_sources: 2,   // Need at least 2 sources
            use_variance: false,
            max_variance: 0.1, // ~0.3 std dev
        }
    }
}

impl EntropyFilter {
    /// Create with specific entropy threshold.
    #[must_use]
    pub fn new(max_entropy: f64) -> Self {
        Self {
            max_entropy,
            ..Default::default()
        }
    }

    /// Create a strict filter (low threshold = high agreement required).
    #[must_use]
    pub fn strict() -> Self {
        Self {
            max_entropy: 0.4,
            min_sources: 3,
            ..Default::default()
        }
    }

    /// Create a permissive filter (high threshold = accepts more disagreement).
    #[must_use]
    pub fn permissive() -> Self {
        Self {
            max_entropy: 0.85,
            min_sources: 2,
            ..Default::default()
        }
    }

    /// Check if scores indicate sufficient agreement to keep the extraction.
    #[must_use]
    pub fn should_keep(&self, scores: &[f64]) -> bool {
        if scores.len() < self.min_sources {
            return true; // Not enough sources to judge
        }

        if self.use_variance {
            confidence_variance(scores) <= self.max_variance
        } else {
            confidence_entropy(scores) <= self.max_entropy
        }
    }

    /// Compute the entropy/variance for logging/debugging.
    #[must_use]
    pub fn compute_score(&self, scores: &[f64]) -> f64 {
        if self.use_variance {
            confidence_variance(scores)
        } else {
            confidence_entropy(scores)
        }
    }

    /// Grade the level of agreement.
    #[must_use]
    pub fn agreement_grade(&self, scores: &[f64]) -> &'static str {
        let score = self.compute_score(scores);
        if self.use_variance {
            if score < 0.02 {
                "Strong agreement"
            } else if score < 0.05 {
                "Good agreement"
            } else if score < 0.1 {
                "Moderate agreement"
            } else {
                "Disagreement"
            }
        } else if score < 0.3 {
            "Strong agreement"
        } else if score < 0.5 {
            "Good agreement"
        } else if score < 0.7 {
            "Moderate agreement"
        } else {
            "Disagreement"
        }
    }
}
