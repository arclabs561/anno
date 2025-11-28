//! Confidence threshold analysis for NER systems.
//!
//! Analyzes how precision, recall, and F1 change at different confidence thresholds.
//! Useful for:
//! - Finding optimal operating points
//! - Understanding precision-recall tradeoffs
//! - Setting production thresholds
//!
//! # Example
//!
//! ```rust
//! use anno::eval::threshold_analysis::{ThresholdAnalyzer, PredictionWithConfidence};
//!
//! let predictions = vec![
//!     PredictionWithConfidence::new("John", "PER", 0.95, true),
//!     PredictionWithConfidence::new("maybe", "PER", 0.45, false),  // Wrong
//!     PredictionWithConfidence::new("Google", "ORG", 0.88, true),
//! ];
//!
//! let analyzer = ThresholdAnalyzer::default();
//! let curve = analyzer.analyze(&predictions);
//!
//! println!("Optimal threshold: {:.2} (F1: {:.1}%)",
//!     curve.optimal_threshold, curve.optimal_f1 * 100.0);
//! ```

use serde::{Deserialize, Serialize};

// =============================================================================
// Data Structures
// =============================================================================

/// A prediction with confidence and correctness label.
#[derive(Debug, Clone)]
pub struct PredictionWithConfidence {
    /// Entity text
    pub text: String,
    /// Entity type
    pub entity_type: String,
    /// Model confidence (0.0 to 1.0)
    pub confidence: f64,
    /// Whether this prediction is correct
    pub is_correct: bool,
}

impl PredictionWithConfidence {
    /// Create a new prediction.
    pub fn new(
        text: impl Into<String>,
        entity_type: impl Into<String>,
        confidence: f64,
        is_correct: bool,
    ) -> Self {
        Self {
            text: text.into(),
            entity_type: entity_type.into(),
            confidence,
            is_correct,
        }
    }
}

/// Metrics at a specific threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdPoint {
    /// Confidence threshold
    pub threshold: f64,
    /// Precision at this threshold
    pub precision: f64,
    /// Recall at this threshold (relative to total correct predictions at threshold 0)
    pub recall: f64,
    /// F1 at this threshold
    pub f1: f64,
    /// Number of predictions retained at this threshold
    pub num_predictions: usize,
    /// Number of correct predictions at this threshold
    pub num_correct: usize,
}

/// Full threshold analysis results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdCurve {
    /// Points along the threshold curve
    pub points: Vec<ThresholdPoint>,
    /// Optimal threshold (maximizes F1)
    pub optimal_threshold: f64,
    /// F1 at optimal threshold
    pub optimal_f1: f64,
    /// Precision at optimal threshold
    pub optimal_precision: f64,
    /// Recall at optimal threshold
    pub optimal_recall: f64,
    /// Area under precision-recall curve (approximation)
    pub auc_pr: f64,
    /// Total predictions analyzed
    pub total_predictions: usize,
    /// Total correct predictions (at threshold 0)
    pub total_correct: usize,
    /// High-precision threshold (precision >= 0.95)
    pub high_precision_threshold: Option<f64>,
    /// High-recall threshold (recall >= 0.95)
    pub high_recall_threshold: Option<f64>,
}

// =============================================================================
// Threshold Analyzer
// =============================================================================

/// Analyzer for confidence threshold effects.
#[derive(Debug, Clone)]
pub struct ThresholdAnalyzer {
    /// Number of threshold points to compute
    pub num_points: usize,
}

impl Default for ThresholdAnalyzer {
    fn default() -> Self {
        Self { num_points: 20 }
    }
}

impl ThresholdAnalyzer {
    /// Create analyzer with custom number of points.
    pub fn new(num_points: usize) -> Self {
        Self { num_points: num_points.max(5) }
    }

    /// Analyze threshold effects on predictions.
    pub fn analyze(&self, predictions: &[PredictionWithConfidence]) -> ThresholdCurve {
        if predictions.is_empty() {
            return ThresholdCurve {
                points: Vec::new(),
                optimal_threshold: 0.5,
                optimal_f1: 0.0,
                optimal_precision: 0.0,
                optimal_recall: 0.0,
                auc_pr: 0.0,
                total_predictions: 0,
                total_correct: 0,
                high_precision_threshold: None,
                high_recall_threshold: None,
            };
        }

        let total_correct = predictions.iter().filter(|p| p.is_correct).count();

        // Compute metrics at each threshold
        let mut points = Vec::new();
        let step = 1.0 / self.num_points as f64;

        for i in 0..=self.num_points {
            let threshold = i as f64 * step;
            let point = self.compute_point(predictions, threshold, total_correct);
            points.push(point);
        }

        // Find optimal threshold
        let (_optimal_idx, optimal_point) = points
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.f1.partial_cmp(&b.1.f1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, p)| (i, p.clone()))
            .unwrap_or((0, points[0].clone()));

        // Compute AUC-PR (trapezoidal approximation)
        let auc_pr = self.compute_auc_pr(&points);

        // Find high-precision threshold
        let high_precision_threshold = points
            .iter()
            .filter(|p| p.precision >= 0.95 && p.num_predictions > 0)
            .map(|p| p.threshold)
            .next();

        // Find high-recall threshold (lowest threshold with recall >= 0.95)
        let high_recall_threshold = points
            .iter()
            .rev()
            .filter(|p| p.recall >= 0.95)
            .map(|p| p.threshold)
            .next();

        ThresholdCurve {
            points,
            optimal_threshold: optimal_point.threshold,
            optimal_f1: optimal_point.f1,
            optimal_precision: optimal_point.precision,
            optimal_recall: optimal_point.recall,
            auc_pr,
            total_predictions: predictions.len(),
            total_correct,
            high_precision_threshold,
            high_recall_threshold,
        }
    }

    fn compute_point(
        &self,
        predictions: &[PredictionWithConfidence],
        threshold: f64,
        total_correct: usize,
    ) -> ThresholdPoint {
        let retained: Vec<_> = predictions
            .iter()
            .filter(|p| p.confidence >= threshold)
            .collect();

        let num_predictions = retained.len();
        let num_correct = retained.iter().filter(|p| p.is_correct).count();

        let precision = if num_predictions == 0 {
            1.0 // No predictions = no false positives
        } else {
            num_correct as f64 / num_predictions as f64
        };

        let recall = if total_correct == 0 {
            1.0
        } else {
            num_correct as f64 / total_correct as f64
        };

        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };

        ThresholdPoint {
            threshold,
            precision,
            recall,
            f1,
            num_predictions,
            num_correct,
        }
    }

    fn compute_auc_pr(&self, points: &[ThresholdPoint]) -> f64 {
        if points.len() < 2 {
            return 0.0;
        }

        // Sort by recall (descending) for proper AUC computation
        let mut sorted: Vec<_> = points.iter().collect();
        sorted.sort_by(|a, b| b.recall.partial_cmp(&a.recall).unwrap_or(std::cmp::Ordering::Equal));

        let mut auc = 0.0;
        for i in 1..sorted.len() {
            let recall_diff = sorted[i - 1].recall - sorted[i].recall;
            let avg_precision = (sorted[i - 1].precision + sorted[i].precision) / 2.0;
            auc += recall_diff * avg_precision;
        }

        auc
    }
}

// =============================================================================
// Display Helpers
// =============================================================================

/// Format threshold curve as ASCII table.
pub fn format_threshold_table(curve: &ThresholdCurve) -> String {
    let mut output = String::new();

    output.push_str("Threshold   Precision   Recall      F1    Predictions\n");
    output.push_str("--------------------------------------------------------\n");

    for point in &curve.points {
        output.push_str(&format!(
            "   {:.2}       {:5.1}%    {:5.1}%    {:5.1}%      {:4}\n",
            point.threshold,
            point.precision * 100.0,
            point.recall * 100.0,
            point.f1 * 100.0,
            point.num_predictions,
        ));
    }

    output.push_str("--------------------------------------------------------\n");
    output.push_str(&format!(
        "Optimal: threshold={:.2}, F1={:.1}%, P={:.1}%, R={:.1}%\n",
        curve.optimal_threshold,
        curve.optimal_f1 * 100.0,
        curve.optimal_precision * 100.0,
        curve.optimal_recall * 100.0,
    ));
    output.push_str(&format!("AUC-PR: {:.3}\n", curve.auc_pr));

    if let Some(t) = curve.high_precision_threshold {
        output.push_str(&format!("High-precision (>=95%) threshold: {:.2}\n", t));
    }
    if let Some(t) = curve.high_recall_threshold {
        output.push_str(&format!("High-recall (>=95%) threshold: {:.2}\n", t));
    }

    output
}

/// Interpret threshold curve quality.
pub fn interpret_curve(curve: &ThresholdCurve) -> Vec<String> {
    let mut insights = Vec::new();

    // AUC-PR interpretation
    if curve.auc_pr >= 0.9 {
        insights.push("Excellent calibration (AUC-PR >= 0.9)".into());
    } else if curve.auc_pr >= 0.7 {
        insights.push("Good calibration (AUC-PR >= 0.7)".into());
    } else if curve.auc_pr >= 0.5 {
        insights.push("Moderate calibration (AUC-PR >= 0.5)".into());
    } else {
        insights.push("Poor calibration (AUC-PR < 0.5) - confidence scores unreliable".into());
    }

    // Optimal threshold interpretation
    if curve.optimal_threshold < 0.3 {
        insights.push("Low optimal threshold suggests model is underconfident".into());
    } else if curve.optimal_threshold > 0.7 {
        insights.push("High optimal threshold suggests model tends to overpredict".into());
    }

    // Precision-recall tradeoff
    if curve.optimal_precision > 0.9 && curve.optimal_recall < 0.7 {
        insights.push("High precision but low recall - consider lowering threshold".into());
    } else if curve.optimal_recall > 0.9 && curve.optimal_precision < 0.7 {
        insights.push("High recall but low precision - consider raising threshold".into());
    }

    // High-precision availability
    if curve.high_precision_threshold.is_some() {
        insights.push("Can achieve 95%+ precision with threshold tuning".into());
    } else {
        insights.push("Cannot achieve 95% precision at any threshold".into());
    }

    insights
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_predictions() {
        let predictions = vec![
            PredictionWithConfidence::new("A", "T", 0.9, true),
            PredictionWithConfidence::new("B", "T", 0.8, true),
            PredictionWithConfidence::new("C", "T", 0.7, true),
        ];

        let analyzer = ThresholdAnalyzer::new(10);
        let curve = analyzer.analyze(&predictions);

        // All correct = perfect precision at all thresholds
        for point in &curve.points {
            if point.num_predictions > 0 {
                assert!((point.precision - 1.0).abs() < 0.01);
            }
        }
    }

    #[test]
    fn test_confidence_ordering() {
        let predictions = vec![
            PredictionWithConfidence::new("High", "T", 0.95, true),
            PredictionWithConfidence::new("Med", "T", 0.50, false),
            PredictionWithConfidence::new("Low", "T", 0.20, false),
        ];

        let analyzer = ThresholdAnalyzer::new(10);
        let curve = analyzer.analyze(&predictions);

        // High threshold should have better precision
        let high_point = curve.points.iter().find(|p| p.threshold >= 0.9).unwrap();
        let low_point = curve.points.iter().find(|p| p.threshold <= 0.1).unwrap();

        assert!(high_point.precision >= low_point.precision);
    }

    #[test]
    fn test_empty_predictions() {
        let predictions: Vec<PredictionWithConfidence> = vec![];
        let analyzer = ThresholdAnalyzer::default();
        let curve = analyzer.analyze(&predictions);

        assert_eq!(curve.total_predictions, 0);
        assert!(curve.points.is_empty());
    }

    #[test]
    fn test_optimal_threshold_found() {
        let predictions = vec![
            PredictionWithConfidence::new("A", "T", 0.9, true),
            PredictionWithConfidence::new("B", "T", 0.8, true),
            PredictionWithConfidence::new("C", "T", 0.3, false),
            PredictionWithConfidence::new("D", "T", 0.2, false),
        ];

        let analyzer = ThresholdAnalyzer::new(10);
        let curve = analyzer.analyze(&predictions);

        // Optimal should be around 0.5 to filter out the low-confidence wrong predictions
        assert!(curve.optimal_threshold >= 0.3);
        assert!(curve.optimal_threshold <= 0.9);
    }

    #[test]
    fn test_auc_pr_bounds() {
        let predictions = vec![
            PredictionWithConfidence::new("A", "T", 0.9, true),
            PredictionWithConfidence::new("B", "T", 0.5, false),
        ];

        let analyzer = ThresholdAnalyzer::default();
        let curve = analyzer.analyze(&predictions);

        assert!(curve.auc_pr >= 0.0);
        assert!(curve.auc_pr <= 1.0);
    }
}

