//! Advanced evaluation metrics for NER.
//!
//! Provides additional metrics beyond basic Precision/Recall/F1:
//! - Partial match metrics (overlap-based)
//! - Confidence threshold analysis
//! - Per-language metrics (for multilingual datasets)
//! - Cross-dataset comparison utilities

use super::datasets::GoldEntity;
use crate::Entity;

/// Partial match metrics (overlap-based).
///
/// Measures how well predicted entities overlap with ground truth,
/// even if boundaries don't match exactly.
#[derive(Debug, Clone)]
pub struct PartialMatchMetrics {
    /// Overlap threshold for considering a match (0.0-1.0)
    pub overlap_threshold: f64,
    /// Precision at this overlap threshold
    pub precision: f64,
    /// Recall at this overlap threshold
    pub recall: f64,
    /// F1 at this overlap threshold
    pub f1: f64,
    /// Number of partial matches found
    pub partial_matches: usize,
}

/// Calculate overlap between two entity spans.
///
/// Returns overlap ratio (0.0-1.0) based on intersection over union.
#[must_use]
pub fn calculate_overlap(
    pred_start: usize,
    pred_end: usize,
    gt_start: usize,
    gt_end: usize,
) -> f64 {
    let intersection_start = pred_start.max(gt_start);
    let intersection_end = pred_end.min(gt_end);

    if intersection_start >= intersection_end {
        return 0.0;
    }

    let intersection = (intersection_end - intersection_start) as f64;
    let union = ((pred_end - pred_start) + (gt_end - gt_start)
        - (intersection_end - intersection_start)) as f64;

    if union == 0.0 {
        return 1.0; // Both spans are empty
    }

    intersection / union
}

/// Calculate partial match metrics.
///
/// # Arguments
/// * `predicted` - Predicted entities
/// * `ground_truth` - Ground truth entities
/// * `overlap_threshold` - Minimum overlap ratio to consider a match (default: 0.5)
///
/// # Returns
/// Partial match metrics
pub fn calculate_partial_match_metrics(
    predicted: &[Entity],
    ground_truth: &[GoldEntity],
    overlap_threshold: f64,
) -> PartialMatchMetrics {
    let mut true_positives = 0;
    let mut _false_positives = 0;

    // Track which ground truth entities have been matched
    let mut gt_matched = vec![false; ground_truth.len()];

    // For each predicted entity, find best matching ground truth
    for pred in predicted {
        let mut best_match: Option<(usize, f64)> = None;

        for (gt_idx, gt) in ground_truth.iter().enumerate() {
            if gt_matched[gt_idx] {
                continue; // Already matched
            }

            // Check entity type matches
            if !crate::eval::entity_type_matches(&pred.entity_type, &gt.entity_type) {
                continue;
            }

            // Calculate overlap
            let overlap = calculate_overlap(pred.start, pred.end, gt.start, gt.end);

            if overlap >= overlap_threshold && best_match.as_ref().map_or(true, |m| m.1 < overlap) {
                best_match = Some((gt_idx, overlap));
            }
        }

        if let Some((gt_idx, _)) = best_match {
            true_positives += 1;
            gt_matched[gt_idx] = true;
        } else {
            _false_positives += 1;
        }
    }

    // Count unmatched ground truth as false negatives (for potential future use)
    let _false_negatives = gt_matched.iter().filter(|&&matched| !matched).count();

    let precision = if !predicted.is_empty() {
        true_positives as f64 / predicted.len() as f64
    } else {
        0.0
    };

    let recall = if !ground_truth.is_empty() {
        true_positives as f64 / ground_truth.len() as f64
    } else {
        0.0
    };

    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    PartialMatchMetrics {
        overlap_threshold,
        precision,
        recall,
        f1,
        partial_matches: true_positives,
    }
}

/// Confidence threshold analysis.
///
/// Analyzes model performance at different confidence thresholds.
#[derive(Debug, Clone)]
pub struct ConfidenceThresholdAnalysis {
    /// Threshold values tested
    pub thresholds: Vec<f64>,
    /// Metrics at each threshold
    pub metrics_at_threshold: Vec<(f64, PartialMatchMetrics)>,
    /// Optimal threshold (highest F1)
    pub optimal_threshold: Option<f64>,
}

/// Analyze performance at different confidence thresholds.
pub fn analyze_confidence_thresholds(
    predicted: &[Entity],
    ground_truth: &[GoldEntity],
    overlap_threshold: f64,
) -> ConfidenceThresholdAnalysis {
    let thresholds: Vec<f64> = (0..=10).map(|i| i as f64 / 10.0).collect();

    let mut metrics_at_threshold = Vec::new();
    let mut best_f1 = 0.0;
    let mut optimal_threshold = None;

    for threshold in &thresholds {
        // Filter predictions by confidence
        let filtered: Vec<&Entity> = predicted
            .iter()
            .filter(|e| e.confidence >= *threshold)
            .collect();

        // Convert to owned for metrics calculation
        let filtered_owned: Vec<Entity> = filtered.iter().map(|e| (*e).clone()).collect();

        let metrics =
            calculate_partial_match_metrics(&filtered_owned, ground_truth, overlap_threshold);

        if metrics.f1 > best_f1 {
            best_f1 = metrics.f1;
            optimal_threshold = Some(*threshold);
        }

        metrics_at_threshold.push((*threshold, metrics));
    }

    ConfidenceThresholdAnalysis {
        thresholds,
        metrics_at_threshold,
        optimal_threshold,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EntityType;

    #[test]
    fn test_calculate_overlap() {
        // Exact match
        assert!((calculate_overlap(0, 10, 0, 10) - 1.0).abs() < 0.001);

        // Partial overlap
        let overlap = calculate_overlap(0, 10, 5, 15);
        assert!(overlap > 0.0 && overlap < 1.0);

        // No overlap
        assert!((calculate_overlap(0, 10, 20, 30) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_calculate_partial_match_metrics() {
        let predicted = vec![Entity::new("John Smith", EntityType::Person, 0, 10, 0.9)];

        let ground_truth = vec![GoldEntity {
            text: "John Smith".to_string(),
            entity_type: EntityType::Person,
            original_label: "PER".to_string(),
            start: 0,
            end: 10,
        }];

        let metrics = calculate_partial_match_metrics(&predicted, &ground_truth, 0.5);
        assert!((metrics.precision - 1.0).abs() < 0.001);
        assert!((metrics.recall - 1.0).abs() < 0.001);
    }
}
