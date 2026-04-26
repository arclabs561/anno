//! Advanced evaluation metrics for NER.
//!
//! Provides additional metrics beyond basic Precision/Recall/F1:
//! - Partial match metrics (overlap-based)
//! - Confidence threshold analysis
//! - Per-language metrics (for multilingual datasets)
//! - Cross-dataset comparison utilities

use super::datasets::GoldEntity;
use anno::Entity;
use std::collections::HashSet;

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
            let overlap = calculate_overlap(pred.start(), pred.end(), gt.start, gt.end);

            let should_update = match best_match {
                None => true,
                Some((_, best_overlap)) => best_overlap < overlap,
            };
            if overlap >= overlap_threshold && should_update {
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

// =============================================================================
// CORE-KG-inspired Extraction Diagnostics (Duplication / Noise)
// =============================================================================

/// Lightweight extraction-quality diagnostics inspired by CORE-KG.
///
/// These are **heuristics**. They are useful for monitoring regressions and surfacing
/// obvious quality problems (e.g., repeated identical nodes, garbage spans) but they
/// are not “ground truth” correctness metrics.
#[derive(Debug, Clone, Default)]
pub struct ExtractionQualityMetrics {
    /// Total number of extracted entities.
    pub total: usize,
    /// Count of entities considered duplicates under conservative normalization.
    pub duplicates: usize,
    /// Duplicate rate (duplicates / total).
    pub duplication_rate: f64,
    /// Count of entities considered “noisy” spans (punctuation-only, numeric-only, etc.).
    pub noisy: usize,
    /// Noise rate (noisy / total).
    pub noise_rate: f64,
}

fn normalize_for_duplication(text: &str) -> String {
    // Conservative normalization:
    // - Unicode-aware lowercasing
    // - Keep only alphanumeric codepoints
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

fn is_noisy_span(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() {
        return true;
    }

    // Punctuation/whitespace-only
    if t.chars()
        .all(|c| c.is_whitespace() || c.is_ascii_punctuation())
    {
        return true;
    }

    // Numeric-only (common “legal noise” / artifacts)
    if t.chars().all(|c| c.is_ascii_digit() || c.is_whitespace()) {
        return true;
    }

    // If normalization removes everything, it's effectively garbage.
    normalize_for_duplication(t).is_empty()
}

/// Compute duplication/noise metrics for a set of extracted entities.
#[must_use]
pub fn compute_extraction_quality_metrics(entities: &[Entity]) -> ExtractionQualityMetrics {
    let total = entities.len();
    if total == 0 {
        return ExtractionQualityMetrics::default();
    }

    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut duplicates = 0usize;
    let mut noisy = 0usize;

    for e in entities {
        if is_noisy_span(&e.text) {
            noisy += 1;
        }

        let key = (
            format!("{:?}", e.entity_type),
            normalize_for_duplication(&e.text),
        );
        if seen.contains(&key) {
            duplicates += 1;
        } else {
            seen.insert(key);
        }
    }

    ExtractionQualityMetrics {
        total,
        duplicates,
        duplication_rate: duplicates as f64 / total as f64,
        noisy,
        noise_rate: noisy as f64 / total as f64,
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

// ============================================================================
// Text Classification Metrics
// ============================================================================

/// Metrics for text classification tasks.
///
/// Supports multi-class classification with macro/micro/weighted averaging.
#[derive(Debug, Clone, Default)]
pub struct ClassificationMetrics {
    /// Total number of examples
    pub total: usize,
    /// Number of correct predictions
    pub correct: usize,
    /// Per-class true positives
    pub class_tp: std::collections::HashMap<String, usize>,
    /// Per-class false positives
    pub class_fp: std::collections::HashMap<String, usize>,
    /// Per-class false negatives
    pub class_fn: std::collections::HashMap<String, usize>,
    /// Per-class support (total examples per class)
    pub class_support: std::collections::HashMap<String, usize>,
}

impl ClassificationMetrics {
    /// Create new empty metrics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a prediction to the metrics.
    pub fn add(&mut self, predicted: &str, actual: &str) {
        self.total += 1;

        // Update support
        *self.class_support.entry(actual.to_string()).or_insert(0) += 1;

        if predicted == actual {
            self.correct += 1;
            *self.class_tp.entry(actual.to_string()).or_insert(0) += 1;
        } else {
            // False positive for predicted class
            *self.class_fp.entry(predicted.to_string()).or_insert(0) += 1;
            // False negative for actual class
            *self.class_fn.entry(actual.to_string()).or_insert(0) += 1;
        }
    }

    /// Overall accuracy.
    #[must_use]
    pub fn accuracy(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.correct as f64 / self.total as f64
    }

    /// Macro-averaged precision (unweighted average across classes).
    #[must_use]
    pub fn macro_precision(&self) -> f64 {
        let classes: std::collections::HashSet<_> = self
            .class_support
            .keys()
            .chain(self.class_fp.keys())
            .collect();

        if classes.is_empty() {
            return 0.0;
        }

        let sum: f64 = classes
            .iter()
            .map(|class| self.class_precision(class))
            .sum();

        sum / classes.len() as f64
    }

    /// Macro-averaged recall (unweighted average across classes).
    #[must_use]
    pub fn macro_recall(&self) -> f64 {
        if self.class_support.is_empty() {
            return 0.0;
        }

        let sum: f64 = self
            .class_support
            .keys()
            .map(|class| self.class_recall(class))
            .sum();

        sum / self.class_support.len() as f64
    }

    /// Macro-averaged F1 score.
    #[must_use]
    pub fn macro_f1(&self) -> f64 {
        let p = self.macro_precision();
        let r = self.macro_recall();
        if p + r == 0.0 {
            return 0.0;
        }
        2.0 * p * r / (p + r)
    }

    /// Micro-averaged precision (aggregate TP/FP across classes).
    #[must_use]
    pub fn micro_precision(&self) -> f64 {
        let tp: usize = self.class_tp.values().sum();
        let fp: usize = self.class_fp.values().sum();
        if tp + fp == 0 {
            return 0.0;
        }
        tp as f64 / (tp + fp) as f64
    }

    /// Micro-averaged recall (aggregate TP/FN across classes).
    #[must_use]
    pub fn micro_recall(&self) -> f64 {
        let tp: usize = self.class_tp.values().sum();
        let fn_sum: usize = self.class_fn.values().sum();
        if tp + fn_sum == 0 {
            return 0.0;
        }
        tp as f64 / (tp + fn_sum) as f64
    }

    /// Micro-averaged F1 score.
    #[must_use]
    pub fn micro_f1(&self) -> f64 {
        let p = self.micro_precision();
        let r = self.micro_recall();
        if p + r == 0.0 {
            return 0.0;
        }
        2.0 * p * r / (p + r)
    }

    /// Weighted F1 score (weighted by class support).
    #[must_use]
    pub fn weighted_f1(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }

        let sum: f64 = self
            .class_support
            .iter()
            .map(|(class, &support)| {
                let f1 = self.class_f1(class);
                f1 * support as f64
            })
            .sum();

        sum / self.total as f64
    }

    /// Precision for a specific class.
    #[must_use]
    pub fn class_precision(&self, class: &str) -> f64 {
        let tp = *self.class_tp.get(class).unwrap_or(&0);
        let fp = *self.class_fp.get(class).unwrap_or(&0);
        if tp + fp == 0 {
            return 0.0;
        }
        tp as f64 / (tp + fp) as f64
    }

    /// Recall for a specific class.
    #[must_use]
    pub fn class_recall(&self, class: &str) -> f64 {
        let tp = *self.class_tp.get(class).unwrap_or(&0);
        let fn_count = *self.class_fn.get(class).unwrap_or(&0);
        if tp + fn_count == 0 {
            return 0.0;
        }
        tp as f64 / (tp + fn_count) as f64
    }

    /// F1 for a specific class.
    #[must_use]
    pub fn class_f1(&self, class: &str) -> f64 {
        let p = self.class_precision(class);
        let r = self.class_recall(class);
        if p + r == 0.0 {
            return 0.0;
        }
        2.0 * p * r / (p + r)
    }

    /// Get all class labels.
    #[must_use]
    pub fn classes(&self) -> Vec<&String> {
        let mut classes: Vec<_> = self.class_support.keys().collect();
        classes.sort();
        classes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anno::EntityType;

    #[test]
    fn test_classification_metrics() {
        let mut metrics = ClassificationMetrics::new();

        // Add some predictions
        metrics.add("sports", "sports");
        metrics.add("sports", "sports");
        metrics.add("business", "business");
        metrics.add("sports", "business"); // Misclassification

        assert_eq!(metrics.total, 4);
        assert_eq!(metrics.correct, 3);
        assert!((metrics.accuracy() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_classification_macro_f1() {
        let mut metrics = ClassificationMetrics::new();

        // Perfect classification
        metrics.add("a", "a");
        metrics.add("b", "b");

        assert!((metrics.macro_f1() - 1.0).abs() < 0.001);
    }

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
