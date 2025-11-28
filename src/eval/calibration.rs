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
//! | PatternNER | `Pattern` | ✗ No | Hardcoded values (e.g., 0.95) |
//! | StatisticalNER | `Heuristic` | ✗ No | Rule-based scores |
//! | StackedNER | Mixed | Partial | Depends on entity type |
//!
//! **Running calibration analysis on PatternNER or StatisticalNER produces
//! meaningless results.** Use `ExtractionMethod::is_calibrated()` to check.
//!
//! # Research Background
//!
//! Calibration is critical for production NER systems where:
//! - Downstream systems need reliable confidence thresholds
//! - Human review should focus on low-confidence predictions
//! - False confidence is worse than admitted uncertainty
//!
//! See: Guo et al. (2017) "On Calibration of Modern Neural Networks"
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
//! use anno::eval::calibration::{CalibrationEvaluator, CalibrationResults};
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
    pub fn evaluate(&self, predictions: &[(f64, bool)]) -> CalibrationResults {
        if predictions.is_empty() {
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

        // Build reliability bins
        let bin_width = 1.0 / self.num_bins as f64;
        let mut bins: Vec<Vec<(f64, bool)>> = vec![Vec::new(); self.num_bins];

        for &(conf, correct) in predictions {
            let bin_idx = ((conf * self.num_bins as f64) as usize).min(self.num_bins - 1);
            bins[bin_idx].push((conf, correct));
        }

        // Compute per-bin metrics
        let mut reliability_bins = Vec::new();
        let mut ece_sum = 0.0;
        let mut mce: f64 = 0.0;

        for (i, bin) in bins.iter().enumerate() {
            if bin.is_empty() {
                continue;
            }

            let range_start = i as f64 * bin_width;
            let range_end = (i + 1) as f64 * bin_width;

            let avg_confidence = bin.iter().map(|(c, _)| c).sum::<f64>() / bin.len() as f64;
            let accuracy =
                bin.iter().filter(|(_, correct)| *correct).count() as f64 / bin.len() as f64;
            let calibration_error = (accuracy - avg_confidence).abs();

            // Weighted contribution to ECE
            let weight = bin.len() as f64 / predictions.len() as f64;
            ece_sum += weight * calibration_error;
            mce = mce.max(calibration_error);

            reliability_bins.push(ReliabilityBin {
                range: (range_start, range_end),
                avg_confidence,
                accuracy,
                count: bin.len(),
                calibration_error,
            });
        }

        // Compute Brier score
        let brier_score = predictions
            .iter()
            .map(|(conf, correct)| {
                let target = if *correct { 1.0 } else { 0.0 };
                (conf - target).powi(2)
            })
            .sum::<f64>()
            / predictions.len() as f64;

        // Compute confidence statistics
        let correct_confs: Vec<f64> = predictions
            .iter()
            .filter(|(_, c)| *c)
            .map(|(conf, _)| *conf)
            .collect();
        let incorrect_confs: Vec<f64> = predictions
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
            let above: Vec<_> = predictions.iter().filter(|(c, _)| *c >= threshold).collect();

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
                let acc =
                    above.iter().filter(|(_, correct)| *correct).count() as f64 / above.len() as f64;
                let cov = above.len() as f64 / predictions.len() as f64;
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
            ece: ece_sum,
            mce,
            brier_score,
            avg_confidence_correct,
            avg_confidence_incorrect,
            confidence_gap: avg_confidence_correct - avg_confidence_incorrect,
            reliability_bins,
            total_predictions: predictions.len(),
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
            (0.9, false), // 90% accuracy at 90% confidence
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
            (0.95, false), // Only 20% accuracy at 95% confidence
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
}

