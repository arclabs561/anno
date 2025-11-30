//! Discontinuous NER evaluation metrics.
//!
//! # Overview
//!
//! Discontinuous named entities span non-contiguous text regions, common in:
//! - Coordination structures: "New York and Los Angeles airports"
//! - Biomedical text: "left and right ventricle"
//! - Legal documents: "paragraphs 2(a), 3(b), and 4(c)"
//!
//! # Evaluation Metrics
//!
//! This module implements metrics from the W2NER paper (arXiv:2112.10070):
//!
//! | Metric | Description |
//! |--------|-------------|
//! | Exact F1 | All spans must match exactly |
//! | Entity Boundary F1 (EBF) | Head and tail tokens correct |
//! | Partial Span F1 | Overlap-based matching for each segment |
//!
//! # Research Alignment
//!
//! From W2NER paper:
//! > "The Entity Boundary F1 score (EBF) employs a more lenient strategy that
//! > allows for prediction of more potential entities, considering whether
//! > the head and tail tokens of predicted entities are correctly identified."
//!
//! Benchmark datasets for discontinuous NER:
//! - **CADEC**: Clinical Adverse Drug Events (F1: ~70-80%)
//! - **ShARe13**: Clinical entity recognition 2013 (F1: ~80-85%)
//! - **ShARe14**: Clinical entity recognition 2014 (F1: ~85%)

use crate::DiscontinuousEntity;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Ground truth discontinuous entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscontinuousGold {
    /// Non-contiguous spans (start, end) pairs
    pub spans: Vec<(usize, usize)>,
    /// Entity type label
    pub entity_type: String,
    /// Original text (concatenated from spans)
    pub text: String,
}

impl DiscontinuousGold {
    /// Create a new discontinuous gold entity.
    pub fn new(
        spans: Vec<(usize, usize)>,
        entity_type: impl Into<String>,
        text: impl Into<String>,
    ) -> Self {
        Self {
            spans,
            entity_type: entity_type.into(),
            text: text.into(),
        }
    }

    /// Create a contiguous gold entity (single span).
    pub fn contiguous(
        start: usize,
        end: usize,
        entity_type: impl Into<String>,
        text: impl Into<String>,
    ) -> Self {
        Self {
            spans: vec![(start, end)],
            entity_type: entity_type.into(),
            text: text.into(),
        }
    }

    /// Check if this entity is contiguous (single span).
    pub fn is_contiguous(&self) -> bool {
        self.spans.len() == 1
    }

    /// Get the bounding range (min start, max end).
    pub fn bounding_range(&self) -> Option<(usize, usize)> {
        let min_start = self.spans.iter().map(|(s, _)| *s).min()?;
        let max_end = self.spans.iter().map(|(_, e)| *e).max()?;
        Some((min_start, max_end))
    }

    /// Total character length across all spans.
    pub fn total_length(&self) -> usize {
        self.spans.iter().map(|(s, e)| e - s).sum()
    }
}

/// Metrics for discontinuous NER evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscontinuousNERMetrics {
    /// Exact match F1: all spans must match exactly
    pub exact_f1: f64,
    /// Exact match precision
    pub exact_precision: f64,
    /// Exact match recall
    pub exact_recall: f64,
    /// Entity Boundary F1: head/tail tokens correct
    pub entity_boundary_f1: f64,
    /// Entity Boundary precision
    pub entity_boundary_precision: f64,
    /// Entity Boundary recall
    pub entity_boundary_recall: f64,
    /// Partial span F1: overlap-based matching
    pub partial_span_f1: f64,
    /// Partial span precision
    pub partial_span_precision: f64,
    /// Partial span recall
    pub partial_span_recall: f64,
    /// Total predicted entities
    pub num_predicted: usize,
    /// Total gold entities
    pub num_gold: usize,
    /// Exact matches
    pub exact_matches: usize,
    /// Boundary matches
    pub boundary_matches: usize,
    /// Per-type breakdown
    pub per_type: HashMap<String, TypeMetrics>,
}

/// Per-type metrics for discontinuous NER.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TypeMetrics {
    /// Exact F1 for this type
    pub exact_f1: f64,
    /// Entity boundary F1 for this type
    pub boundary_f1: f64,
    /// Number of gold entities of this type
    pub gold_count: usize,
    /// Number of predicted entities of this type
    pub pred_count: usize,
    /// Number of exact matches
    pub exact_matches: usize,
}

/// Configuration for discontinuous NER evaluation.
#[derive(Debug, Clone)]
pub struct DiscontinuousEvalConfig {
    /// Overlap threshold for partial matching (default: 0.5)
    pub overlap_threshold: f64,
    /// Whether to require type match (default: true)
    pub require_type_match: bool,
}

impl Default for DiscontinuousEvalConfig {
    fn default() -> Self {
        Self {
            overlap_threshold: 0.5,
            require_type_match: true,
        }
    }
}

/// Evaluate discontinuous NER predictions against gold standard.
///
/// # Arguments
/// * `gold` - Ground truth discontinuous entities
/// * `pred` - Predicted discontinuous entities
/// * `config` - Evaluation configuration
///
/// # Returns
/// Comprehensive metrics for discontinuous NER
///
/// # Example
///
/// ```rust
/// use anno::eval::discontinuous::{evaluate_discontinuous_ner, DiscontinuousGold, DiscontinuousEvalConfig};
/// use anno::DiscontinuousEntity;
///
/// let gold = vec![
///     DiscontinuousGold::new(
///         vec![(0, 8), (25, 33)],
///         "location",
///         "New York airports"
///     ),
/// ];
///
/// let pred = vec![
///     DiscontinuousEntity {
///         spans: vec![(0, 8), (25, 33)],
///         text: "New York airports".to_string(),
///         entity_type: "location".to_string(),
///         confidence: 0.9,
///     },
/// ];
///
/// let metrics = evaluate_discontinuous_ner(&gold, &pred, &DiscontinuousEvalConfig::default());
/// assert!((metrics.exact_f1 - 1.0).abs() < 0.001);
/// ```
pub fn evaluate_discontinuous_ner(
    gold: &[DiscontinuousGold],
    pred: &[DiscontinuousEntity],
    config: &DiscontinuousEvalConfig,
) -> DiscontinuousNERMetrics {
    if gold.is_empty() && pred.is_empty() {
        return DiscontinuousNERMetrics {
            exact_f1: 1.0,
            exact_precision: 1.0,
            exact_recall: 1.0,
            entity_boundary_f1: 1.0,
            entity_boundary_precision: 1.0,
            entity_boundary_recall: 1.0,
            partial_span_f1: 1.0,
            partial_span_precision: 1.0,
            partial_span_recall: 1.0,
            num_predicted: 0,
            num_gold: 0,
            exact_matches: 0,
            boundary_matches: 0,
            per_type: HashMap::new(),
        };
    }

    // Track matches
    let mut gold_matched_exact = vec![false; gold.len()];
    let mut gold_matched_boundary = vec![false; gold.len()];
    let mut pred_matched_exact = vec![false; pred.len()];
    let mut pred_matched_boundary = vec![false; pred.len()];

    // Per-type tracking
    let mut type_stats: HashMap<String, (usize, usize, usize, usize)> = HashMap::new(); // (gold, pred, exact, boundary)

    // Count gold per type
    for g in gold {
        let entry = type_stats.entry(g.entity_type.clone()).or_default();
        entry.0 += 1;
    }

    // Count pred per type
    for p in pred {
        let entry = type_stats.entry(p.entity_type.clone()).or_default();
        entry.1 += 1;
    }

    // Exact matching: spans must match exactly
    for (pi, p) in pred.iter().enumerate() {
        for (gi, g) in gold.iter().enumerate() {
            if gold_matched_exact[gi] {
                continue;
            }

            // Type match check
            if config.require_type_match && p.entity_type != g.entity_type {
                continue;
            }

            // Exact span match
            if spans_match_exactly(&p.spans, &g.spans) {
                gold_matched_exact[gi] = true;
                pred_matched_exact[pi] = true;

                let entry = type_stats.entry(g.entity_type.clone()).or_default();
                entry.2 += 1;
                break;
            }
        }
    }

    // Boundary matching: head and tail tokens correct
    for (pi, p) in pred.iter().enumerate() {
        for (gi, g) in gold.iter().enumerate() {
            if gold_matched_boundary[gi] {
                continue;
            }

            if config.require_type_match && p.entity_type != g.entity_type {
                continue;
            }

            // Boundary match: first span start and last span end
            if boundaries_match(&p.spans, &g.spans) {
                gold_matched_boundary[gi] = true;
                pred_matched_boundary[pi] = true;

                let entry = type_stats.entry(g.entity_type.clone()).or_default();
                entry.3 += 1;
                break;
            }
        }
    }

    // Calculate partial span overlap scores
    let mut partial_precision_sum = 0.0;
    let mut partial_recall_sum = 0.0;

    for p in pred {
        let best_overlap = gold
            .iter()
            .filter(|g| !config.require_type_match || p.entity_type == g.entity_type)
            .map(|g| calculate_multi_span_overlap(&p.spans, &g.spans))
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        partial_precision_sum += best_overlap;
    }

    for g in gold {
        let best_overlap = pred
            .iter()
            .filter(|p| !config.require_type_match || p.entity_type == g.entity_type)
            .map(|p| calculate_multi_span_overlap(&p.spans, &g.spans))
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        partial_recall_sum += best_overlap;
    }

    // Calculate metrics
    let exact_matches = pred_matched_exact.iter().filter(|&&m| m).count();
    let boundary_matches = pred_matched_boundary.iter().filter(|&&m| m).count();

    let exact_precision = if !pred.is_empty() {
        exact_matches as f64 / pred.len() as f64
    } else {
        0.0
    };
    let exact_recall = if !gold.is_empty() {
        exact_matches as f64 / gold.len() as f64
    } else {
        0.0
    };
    let exact_f1 = f1_score(exact_precision, exact_recall);

    let boundary_precision = if !pred.is_empty() {
        boundary_matches as f64 / pred.len() as f64
    } else {
        0.0
    };
    let boundary_recall = if !gold.is_empty() {
        boundary_matches as f64 / gold.len() as f64
    } else {
        0.0
    };
    let entity_boundary_f1 = f1_score(boundary_precision, boundary_recall);

    let partial_span_precision = if !pred.is_empty() {
        partial_precision_sum / pred.len() as f64
    } else {
        0.0
    };
    let partial_span_recall = if !gold.is_empty() {
        partial_recall_sum / gold.len() as f64
    } else {
        0.0
    };
    let partial_span_f1 = f1_score(partial_span_precision, partial_span_recall);

    // Build per-type metrics
    let per_type: HashMap<String, TypeMetrics> = type_stats
        .into_iter()
        .map(|(t, (gold_count, pred_count, exact, boundary))| {
            let exact_p = if pred_count > 0 {
                exact as f64 / pred_count as f64
            } else {
                0.0
            };
            let exact_r = if gold_count > 0 {
                exact as f64 / gold_count as f64
            } else {
                0.0
            };
            let boundary_p = if pred_count > 0 {
                boundary as f64 / pred_count as f64
            } else {
                0.0
            };
            let boundary_r = if gold_count > 0 {
                boundary as f64 / gold_count as f64
            } else {
                0.0
            };

            (
                t,
                TypeMetrics {
                    exact_f1: f1_score(exact_p, exact_r),
                    boundary_f1: f1_score(boundary_p, boundary_r),
                    gold_count,
                    pred_count,
                    exact_matches: exact,
                },
            )
        })
        .collect();

    DiscontinuousNERMetrics {
        exact_f1,
        exact_precision,
        exact_recall,
        entity_boundary_f1,
        entity_boundary_precision: boundary_precision,
        entity_boundary_recall: boundary_recall,
        partial_span_f1,
        partial_span_precision,
        partial_span_recall,
        num_predicted: pred.len(),
        num_gold: gold.len(),
        exact_matches,
        boundary_matches,
        per_type,
    }
}

/// Check if two span sets match exactly.
fn spans_match_exactly(a: &[(usize, usize)], b: &[(usize, usize)]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    let mut a_sorted: Vec<_> = a.to_vec();
    let mut b_sorted: Vec<_> = b.to_vec();
    a_sorted.sort();
    b_sorted.sort();

    a_sorted == b_sorted
}

/// Check if boundaries (first start, last end) match.
fn boundaries_match(a: &[(usize, usize)], b: &[(usize, usize)]) -> bool {
    match (a.is_empty(), b.is_empty()) {
        (true, true) => true,
        (true, false) | (false, true) => false,
        (false, false) => {
            // Safe: we've verified both are non-empty
            let (Some(a_min), Some(a_max)) = (
                a.iter().map(|(s, _)| *s).min(),
                a.iter().map(|(_, e)| *e).max(),
            ) else {
                return false;
            };
            let (Some(b_min), Some(b_max)) = (
                b.iter().map(|(s, _)| *s).min(),
                b.iter().map(|(_, e)| *e).max(),
            ) else {
                return false;
            };
            a_min == b_min && a_max == b_max
        }
    }
}

/// Calculate overlap between two sets of spans.
///
/// Uses Intersection over Union (IoU) across all spans.
fn calculate_multi_span_overlap(a: &[(usize, usize)], b: &[(usize, usize)]) -> f64 {
    let a_chars: std::collections::HashSet<usize> = a.iter().flat_map(|(s, e)| *s..*e).collect();
    let b_chars: std::collections::HashSet<usize> = b.iter().flat_map(|(s, e)| *s..*e).collect();

    let intersection = a_chars.intersection(&b_chars).count();
    let union = a_chars.union(&b_chars).count();

    if union == 0 {
        return 1.0; // Both empty
    }

    intersection as f64 / union as f64
}

/// Calculate F1 score from precision and recall.
fn f1_score(precision: f64, recall: f64) -> f64 {
    if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match() {
        let gold = vec![DiscontinuousGold::new(
            vec![(0, 5), (10, 15)],
            "LOC",
            "test",
        )];
        let pred = vec![DiscontinuousEntity {
            spans: vec![(0, 5), (10, 15)],
            text: "test".to_string(),
            entity_type: "LOC".to_string(),
            confidence: 0.9,
        }];

        let metrics = evaluate_discontinuous_ner(&gold, &pred, &DiscontinuousEvalConfig::default());
        assert!((metrics.exact_f1 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_boundary_match() {
        let gold = vec![DiscontinuousGold::new(
            vec![(0, 5), (10, 15)],
            "LOC",
            "test",
        )];
        // Same boundaries but different internal span structure
        let pred = vec![DiscontinuousEntity {
            spans: vec![(0, 3), (3, 5), (10, 15)],
            text: "test".to_string(),
            entity_type: "LOC".to_string(),
            confidence: 0.9,
        }];

        let metrics = evaluate_discontinuous_ner(&gold, &pred, &DiscontinuousEvalConfig::default());
        // Not exact match
        assert!(metrics.exact_f1 < 1.0);
        // But boundary match
        assert!((metrics.entity_boundary_f1 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_contiguous_entity() {
        let gold = DiscontinuousGold::contiguous(0, 10, "PER", "John Smith");
        assert!(gold.is_contiguous());
        assert_eq!(gold.total_length(), 10);
    }

    #[test]
    fn test_bounding_range() {
        let gold = DiscontinuousGold::new(vec![(0, 5), (20, 30)], "LOC", "test");
        assert_eq!(gold.bounding_range(), Some((0, 30)));
    }

    #[test]
    fn test_empty_inputs() {
        let metrics = evaluate_discontinuous_ner(&[], &[], &DiscontinuousEvalConfig::default());
        assert!((metrics.exact_f1 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_type_mismatch() {
        let gold = vec![DiscontinuousGold::new(vec![(0, 5)], "PER", "John")];
        let pred = vec![DiscontinuousEntity {
            spans: vec![(0, 5)],
            text: "John".to_string(),
            entity_type: "ORG".to_string(), // Wrong type
            confidence: 0.9,
        }];

        let config = DiscontinuousEvalConfig {
            require_type_match: true,
            ..Default::default()
        };
        let metrics = evaluate_discontinuous_ner(&gold, &pred, &config);
        assert!(metrics.exact_f1 < 0.001);
    }

    #[test]
    fn test_partial_overlap() {
        let gold = vec![DiscontinuousGold::new(vec![(0, 10)], "LOC", "test")];
        let pred = vec![DiscontinuousEntity {
            spans: vec![(5, 15)], // 50% overlap
            text: "test".to_string(),
            entity_type: "LOC".to_string(),
            confidence: 0.9,
        }];

        let metrics = evaluate_discontinuous_ner(&gold, &pred, &DiscontinuousEvalConfig::default());
        // Should have partial overlap score
        assert!(metrics.partial_span_f1 > 0.0);
        assert!(metrics.partial_span_f1 < 1.0);
    }

    #[test]
    fn test_multi_span_overlap() {
        let a = vec![(0, 10), (20, 30)];
        let b = vec![(5, 25)];

        let overlap = calculate_multi_span_overlap(&a, &b);
        // a covers: 0-10, 20-30 = 20 chars
        // b covers: 5-25 = 20 chars
        // intersection: 5-10, 20-25 = 10 chars
        // union: 0-30 = 30 chars (but with gap, so actually 25 chars)
        assert!(overlap > 0.0);
        assert!(overlap < 1.0);
    }
}
