//! Relation extraction evaluation metrics.
//!
//! # Overview
//!
//! Relation extraction identifies semantic relationships between entity pairs,
//! producing (head, relation, tail) triples. This module provides metrics for:
//!
//! - **Boundary evaluation (Rel)**: Relation correct if entity boundaries match
//! - **Strict evaluation (Rel+)**: Relation correct if entities exactly match
//! - **Per-relation breakdown**: F1 for each relation type
//!
//! # Evaluation Protocols
//!
//! From joint entity-relation extraction research (arXiv:2502.09247):
//!
//! | Mode | Entity Match | Relation Match | Use Case |
//! |------|--------------|----------------|----------|
//! | Boundary (Rel) | Span overlap | Type match | Lenient |
//! | Strict (Rel+) | Exact span + type | Type match | Strict |
//!
//! # Research Alignment
//!
//! Standard benchmarks:
//! - **DocRED**: Document-level RE (F1: ~60-65%)
//! - **TACRED**: Sentence-level RE (F1: ~70-75%)
//! - **SciERC**: Scientific domain (F1: ~45-50%)
//! - **NYT**: News text (F1: ~50-60%)

use crate::RelationTriple;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Ground truth relation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationGold {
    /// Head entity span (start, end)
    pub head_span: (usize, usize),
    /// Head entity type
    pub head_type: String,
    /// Head entity text
    pub head_text: String,
    /// Tail entity span (start, end)
    pub tail_span: (usize, usize),
    /// Tail entity type
    pub tail_type: String,
    /// Tail entity text
    pub tail_text: String,
    /// Relation type
    pub relation_type: String,
}

impl RelationGold {
    /// Create a new relation gold standard.
    pub fn new(
        head_span: (usize, usize),
        head_type: impl Into<String>,
        head_text: impl Into<String>,
        tail_span: (usize, usize),
        tail_type: impl Into<String>,
        tail_text: impl Into<String>,
        relation_type: impl Into<String>,
    ) -> Self {
        Self {
            head_span,
            head_type: head_type.into(),
            head_text: head_text.into(),
            tail_span,
            tail_type: tail_type.into(),
            tail_text: tail_text.into(),
            relation_type: relation_type.into(),
        }
    }
}

/// Predicted relation with entity information.
#[derive(Debug, Clone)]
pub struct RelationPrediction {
    /// Head entity span (start, end)
    pub head_span: (usize, usize),
    /// Head entity type
    pub head_type: String,
    /// Tail entity span (start, end)
    pub tail_span: (usize, usize),
    /// Tail entity type
    pub tail_type: String,
    /// Relation type
    pub relation_type: String,
    /// Confidence score
    pub confidence: f32,
}

impl RelationPrediction {
    /// Create from a RelationTriple and entity list.
    pub fn from_triple_with_entities(
        triple: &RelationTriple,
        entities: &[crate::Entity],
    ) -> Option<Self> {
        let head = entities.get(triple.head_idx)?;
        let tail = entities.get(triple.tail_idx)?;

        Some(Self {
            head_span: (head.start, head.end),
            head_type: head.entity_type.as_label().to_string(),
            tail_span: (tail.start, tail.end),
            tail_type: tail.entity_type.as_label().to_string(),
            relation_type: triple.relation_type.clone(),
            confidence: triple.confidence,
        })
    }
}

/// Metrics for relation extraction evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationMetrics {
    /// Boundary evaluation F1 (Rel): entity boundaries match
    pub boundary_f1: f64,
    /// Boundary precision
    pub boundary_precision: f64,
    /// Boundary recall
    pub boundary_recall: f64,
    /// Strict evaluation F1 (Rel+): exact entity match
    pub strict_f1: f64,
    /// Strict precision
    pub strict_precision: f64,
    /// Strict recall
    pub strict_recall: f64,
    /// Total predicted relations
    pub num_predicted: usize,
    /// Total gold relations
    pub num_gold: usize,
    /// Boundary matches
    pub boundary_matches: usize,
    /// Strict matches
    pub strict_matches: usize,
    /// Per-relation-type breakdown
    pub per_relation: HashMap<String, RelationTypeMetrics>,
}

/// Per-relation-type metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RelationTypeMetrics {
    /// F1 score for this relation type (boundary mode)
    pub boundary_f1: f64,
    /// F1 score for this relation type (strict mode)
    pub strict_f1: f64,
    /// Number of gold relations of this type
    pub gold_count: usize,
    /// Number of predicted relations of this type
    pub pred_count: usize,
    /// Boundary matches
    pub boundary_matches: usize,
    /// Strict matches
    pub strict_matches: usize,
}

/// Configuration for relation evaluation.
#[derive(Debug, Clone)]
pub struct RelationEvalConfig {
    /// Overlap threshold for boundary matching (default: 0.5)
    pub overlap_threshold: f64,
    /// Whether to require entity type match (default: true)
    pub require_entity_type_match: bool,
    /// Whether relation direction matters (default: true)
    pub directed_relations: bool,
}

impl Default for RelationEvalConfig {
    fn default() -> Self {
        Self {
            overlap_threshold: 0.5,
            require_entity_type_match: true,
            directed_relations: true,
        }
    }
}

/// Evaluate relation extraction predictions against gold standard.
///
/// # Arguments
/// * `gold` - Ground truth relations
/// * `pred` - Predicted relations
/// * `config` - Evaluation configuration
///
/// # Returns
/// Comprehensive relation extraction metrics
///
/// # Example
///
/// ```rust
/// use anno::eval::relation::{evaluate_relations, RelationGold, RelationPrediction, RelationEvalConfig};
///
/// let gold = vec![
///     RelationGold::new(
///         (0, 10), "PER", "Steve Jobs",
///         (20, 25), "ORG", "Apple",
///         "FOUNDED"
///     ),
/// ];
///
/// let pred = vec![
///     RelationPrediction {
///         head_span: (0, 10),
///         head_type: "PER".to_string(),
///         tail_span: (20, 25),
///         tail_type: "ORG".to_string(),
///         relation_type: "FOUNDED".to_string(),
///         confidence: 0.9,
///     },
/// ];
///
/// let metrics = evaluate_relations(&gold, &pred, &RelationEvalConfig::default());
/// assert!((metrics.strict_f1 - 1.0).abs() < 0.001);
/// ```
pub fn evaluate_relations(
    gold: &[RelationGold],
    pred: &[RelationPrediction],
    config: &RelationEvalConfig,
) -> RelationMetrics {
    if gold.is_empty() && pred.is_empty() {
        return RelationMetrics {
            boundary_f1: 1.0,
            boundary_precision: 1.0,
            boundary_recall: 1.0,
            strict_f1: 1.0,
            strict_precision: 1.0,
            strict_recall: 1.0,
            num_predicted: 0,
            num_gold: 0,
            boundary_matches: 0,
            strict_matches: 0,
            per_relation: HashMap::new(),
        };
    }

    // Track matches
    let mut gold_matched_boundary = vec![false; gold.len()];
    let mut gold_matched_strict = vec![false; gold.len()];
    let mut pred_matched_boundary = vec![false; pred.len()];
    let mut pred_matched_strict = vec![false; pred.len()];

    // Per-relation tracking: (gold_count, pred_count, boundary_matches, strict_matches)
    let mut rel_stats: HashMap<String, (usize, usize, usize, usize)> = HashMap::new();

    // Count gold per relation type
    for g in gold {
        let entry = rel_stats.entry(g.relation_type.clone()).or_default();
        entry.0 += 1;
    }

    // Count pred per relation type
    for p in pred {
        let entry = rel_stats.entry(p.relation_type.clone()).or_default();
        entry.1 += 1;
    }

    // Strict matching: exact entity spans + relation type
    for (pi, p) in pred.iter().enumerate() {
        // Skip predictions that are already matched
        if pred_matched_strict[pi] {
            continue;
        }
        for (gi, g) in gold.iter().enumerate() {
            if gold_matched_strict[gi] {
                continue;
            }

            // Relation type must match (case-insensitive)
            if p.relation_type.to_lowercase() != g.relation_type.to_lowercase() {
                continue;
            }

            // Check entity type match if required
            if config.require_entity_type_match
                && (p.head_type != g.head_type || p.tail_type != g.tail_type)
            {
                continue;
            }

            // Exact span match (or reversed if undirected)
            let forward_match = p.head_span == g.head_span && p.tail_span == g.tail_span;
            let reverse_match = !config.directed_relations
                && p.head_span == g.tail_span
                && p.tail_span == g.head_span;

            if forward_match || reverse_match {
                gold_matched_strict[gi] = true;
                pred_matched_strict[pi] = true;

                let entry = rel_stats.entry(g.relation_type.clone()).or_default();
                entry.3 += 1;
                break;
            }
        }
    }

    // Boundary matching: overlapping entity spans + relation type
    for (pi, p) in pred.iter().enumerate() {
        // Skip predictions that are already matched
        if pred_matched_boundary[pi] {
            continue;
        }
        for (gi, g) in gold.iter().enumerate() {
            if gold_matched_boundary[gi] {
                continue;
            }

            // Relation type must match (case-insensitive)
            if p.relation_type.to_lowercase() != g.relation_type.to_lowercase() {
                continue;
            }

            if config.require_entity_type_match
                && (p.head_type != g.head_type || p.tail_type != g.tail_type)
            {
                continue;
            }

            // Boundary match: overlapping spans
            let head_overlap = calculate_span_overlap(p.head_span, g.head_span);
            let tail_overlap = calculate_span_overlap(p.tail_span, g.tail_span);

            let forward_match = head_overlap >= config.overlap_threshold
                && tail_overlap >= config.overlap_threshold;

            let reverse_match = if !config.directed_relations {
                let rev_head_overlap = calculate_span_overlap(p.head_span, g.tail_span);
                let rev_tail_overlap = calculate_span_overlap(p.tail_span, g.head_span);
                rev_head_overlap >= config.overlap_threshold
                    && rev_tail_overlap >= config.overlap_threshold
            } else {
                false
            };

            if forward_match || reverse_match {
                gold_matched_boundary[gi] = true;
                pred_matched_boundary[pi] = true;

                let entry = rel_stats.entry(g.relation_type.clone()).or_default();
                entry.2 += 1;
                break;
            }
        }
    }

    // Calculate metrics
    let boundary_matches = pred_matched_boundary.iter().filter(|&&m| m).count();
    let strict_matches = pred_matched_strict.iter().filter(|&&m| m).count();

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
    let boundary_f1 = f1_score(boundary_precision, boundary_recall);

    let strict_precision = if !pred.is_empty() {
        strict_matches as f64 / pred.len() as f64
    } else {
        0.0
    };
    let strict_recall = if !gold.is_empty() {
        strict_matches as f64 / gold.len() as f64
    } else {
        0.0
    };
    let strict_f1 = f1_score(strict_precision, strict_recall);

    // Build per-relation metrics
    let per_relation: HashMap<String, RelationTypeMetrics> = rel_stats
        .into_iter()
        .map(|(rel, (gold_count, pred_count, boundary, strict))| {
            let b_p = if pred_count > 0 {
                boundary as f64 / pred_count as f64
            } else {
                0.0
            };
            let b_r = if gold_count > 0 {
                boundary as f64 / gold_count as f64
            } else {
                0.0
            };
            let s_p = if pred_count > 0 {
                strict as f64 / pred_count as f64
            } else {
                0.0
            };
            let s_r = if gold_count > 0 {
                strict as f64 / gold_count as f64
            } else {
                0.0
            };

            (
                rel,
                RelationTypeMetrics {
                    boundary_f1: f1_score(b_p, b_r),
                    strict_f1: f1_score(s_p, s_r),
                    gold_count,
                    pred_count,
                    boundary_matches: boundary,
                    strict_matches: strict,
                },
            )
        })
        .collect();

    RelationMetrics {
        boundary_f1,
        boundary_precision,
        boundary_recall,
        strict_f1,
        strict_precision,
        strict_recall,
        num_predicted: pred.len(),
        num_gold: gold.len(),
        boundary_matches,
        strict_matches,
        per_relation,
    }
}

/// Render relation evaluation as HTML report.
pub fn render_relation_eval_html(metrics: &RelationMetrics) -> String {
    let mut html = String::new();
    html.push_str("<!DOCTYPE html>\n<html><head><title>Relation Extraction Evaluation</title>");
    html.push_str("<style>body{font-family:monospace;margin:20px;}table{border-collapse:collapse;}th,td{padding:8px;border:1px solid #ddd;}</style>");
    html.push_str("</head><body>");
    html.push_str("<h1>Relation Extraction Evaluation</h1>");
    html.push_str("<h2>Overall Metrics</h2>");
    html.push_str("<table>");
    html.push_str("<tr><th>Metric</th><th>Boundary (Rel)</th><th>Strict (Rel+)</th></tr>");
    html.push_str(&format!(
        "<tr><td>Precision</td><td>{:.3}</td><td>{:.3}</td></tr>",
        metrics.boundary_precision, metrics.strict_precision
    ));
    html.push_str(&format!(
        "<tr><td>Recall</td><td>{:.3}</td><td>{:.3}</td></tr>",
        metrics.boundary_recall, metrics.strict_recall
    ));
    html.push_str(&format!(
        "<tr><td>F1</td><td>{:.3}</td><td>{:.3}</td></tr>",
        metrics.boundary_f1, metrics.strict_f1
    ));
    html.push_str("</table>");
    html.push_str(&format!(
        "<p>Gold: {}  Predicted: {}  Boundary matches: {}  Strict matches: {}</p>",
        metrics.num_gold, metrics.num_predicted, metrics.boundary_matches, metrics.strict_matches
    ));

    if !metrics.per_relation.is_empty() {
        html.push_str("<h2>Per-Relation Breakdown</h2>");
        html.push_str("<table>");
        html.push_str("<tr><th>Relation Type</th><th>Boundary F1</th><th>Strict F1</th><th>Gold</th><th>Pred</th><th>Boundary Matches</th><th>Strict Matches</th></tr>");
        let mut rels: Vec<_> = metrics.per_relation.iter().collect();
        rels.sort_by(|a, b| b.1.gold_count.cmp(&a.1.gold_count));
        for (rel_type, rel_metrics) in rels {
            html.push_str(&format!(
                "<tr><td>{}</td><td>{:.3}</td><td>{:.3}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                rel_type, rel_metrics.boundary_f1, rel_metrics.strict_f1,
                rel_metrics.gold_count, rel_metrics.pred_count,
                rel_metrics.boundary_matches, rel_metrics.strict_matches
            ));
        }
        html.push_str("</table>");
    }

    html.push_str("</body></html>");
    html
}

impl RelationMetrics {
    /// Generate human-readable string representation.
    pub fn to_string_human(&self, verbose: bool) -> String {
        let mut out = String::new();

        out.push_str("Relation Extraction Evaluation\n");
        out.push_str(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n",
        );
        out.push_str(&format!(
            "Boundary (Rel):  P={:.1}%  R={:.1}%  F1={:.1}%\n",
            self.boundary_precision * 100.0,
            self.boundary_recall * 100.0,
            self.boundary_f1 * 100.0
        ));
        out.push_str(&format!(
            "Strict (Rel+):   P={:.1}%  R={:.1}%  F1={:.1}%\n",
            self.strict_precision * 100.0,
            self.strict_recall * 100.0,
            self.strict_f1 * 100.0
        ));
        out.push_str(&format!(
            "Gold: {}  Predicted: {}  Boundary matches: {}  Strict matches: {}\n",
            self.num_gold, self.num_predicted, self.boundary_matches, self.strict_matches
        ));

        if verbose && !self.per_relation.is_empty() {
            out.push_str("\nPer-Relation Breakdown:\n");
            let mut rels: Vec<_> = self.per_relation.iter().collect();
            rels.sort_by(|a, b| b.1.gold_count.cmp(&a.1.gold_count));

            for (rel_type, metrics) in rels {
                if metrics.gold_count > 0 || metrics.pred_count > 0 {
                    // Use stored precision/recall from RelationTypeMetrics (already computed)
                    // Calculate from stored matches to avoid duplication
                    let boundary_p = if metrics.pred_count > 0 {
                        metrics.boundary_matches as f64 / metrics.pred_count as f64
                    } else {
                        0.0
                    };
                    let boundary_r = if metrics.gold_count > 0 {
                        metrics.boundary_matches as f64 / metrics.gold_count as f64
                    } else {
                        0.0
                    };
                    let strict_p = if metrics.pred_count > 0 {
                        metrics.strict_matches as f64 / metrics.pred_count as f64
                    } else {
                        0.0
                    };
                    let strict_r = if metrics.gold_count > 0 {
                        metrics.strict_matches as f64 / metrics.gold_count as f64
                    } else {
                        0.0
                    };
                    // Use stored F1 scores (already computed correctly)
                    out.push_str(&format!(
                        "  {:20} Boundary: F1={:.1}% (P={:.1}% R={:.1}%)  Strict: F1={:.1}% (P={:.1}% R={:.1}%)  [gold={} pred={} matches={}/{}]\n",
                        rel_type,
                        metrics.boundary_f1 * 100.0,
                        boundary_p * 100.0,
                        boundary_r * 100.0,
                        metrics.strict_f1 * 100.0,
                        strict_p * 100.0,
                        strict_r * 100.0,
                        metrics.gold_count,
                        metrics.pred_count,
                        metrics.boundary_matches,
                        metrics.strict_matches
                    ));
                }
            }
        }

        out
    }
}

impl std::fmt::Display for RelationMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string_human(false))
    }
}

/// Calculate overlap (IoU) between two spans.
fn calculate_span_overlap(a: (usize, usize), b: (usize, usize)) -> f64 {
    let intersection_start = a.0.max(b.0);
    let intersection_end = a.1.min(b.1);

    if intersection_start >= intersection_end {
        return 0.0;
    }

    let intersection = (intersection_end - intersection_start) as f64;
    let union = ((a.1 - a.0) + (b.1 - b.0) - (intersection_end - intersection_start)) as f64;

    if union == 0.0 {
        return 1.0;
    }

    intersection / union
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
    fn test_exact_relation_match() {
        let gold = vec![RelationGold::new(
            (0, 10),
            "PER",
            "Steve Jobs",
            (20, 25),
            "ORG",
            "Apple",
            "FOUNDED",
        )];
        let pred = vec![RelationPrediction {
            head_span: (0, 10),
            head_type: "PER".to_string(),
            tail_span: (20, 25),
            tail_type: "ORG".to_string(),
            relation_type: "FOUNDED".to_string(),
            confidence: 0.9,
        }];

        let metrics = evaluate_relations(&gold, &pred, &RelationEvalConfig::default());
        assert!((metrics.strict_f1 - 1.0).abs() < 0.001);
        assert!((metrics.boundary_f1 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_boundary_match_not_strict() {
        let gold = vec![RelationGold::new(
            (0, 10),
            "PER",
            "Steve Jobs",
            (20, 30),
            "ORG",
            "Apple Inc",
            "FOUNDED",
        )];
        // Overlapping but not exact
        let pred = vec![RelationPrediction {
            head_span: (0, 10),
            head_type: "PER".to_string(),
            tail_span: (20, 25), // Different end
            tail_type: "ORG".to_string(),
            relation_type: "FOUNDED".to_string(),
            confidence: 0.9,
        }];

        let metrics = evaluate_relations(&gold, &pred, &RelationEvalConfig::default());
        // Strict should fail
        assert!(metrics.strict_f1 < 1.0);
        // Boundary should succeed (50% overlap > 0.5 threshold)
        assert!(metrics.boundary_f1 > 0.0);
    }

    #[test]
    fn test_wrong_relation_type() {
        let gold = vec![RelationGold::new(
            (0, 10),
            "PER",
            "Steve Jobs",
            (20, 25),
            "ORG",
            "Apple",
            "FOUNDED",
        )];
        let pred = vec![RelationPrediction {
            head_span: (0, 10),
            head_type: "PER".to_string(),
            tail_span: (20, 25),
            tail_type: "ORG".to_string(),
            relation_type: "WORKS_FOR".to_string(), // Wrong relation
            confidence: 0.9,
        }];

        let metrics = evaluate_relations(&gold, &pred, &RelationEvalConfig::default());
        assert!(metrics.strict_f1 < 0.001);
    }

    #[test]
    fn test_undirected_relations() {
        let gold = vec![RelationGold::new(
            (0, 10),
            "PER",
            "Alice",
            (20, 25),
            "PER",
            "Bob",
            "SIBLING",
        )];
        // Reversed head/tail
        let pred = vec![RelationPrediction {
            head_span: (20, 25),
            head_type: "PER".to_string(),
            tail_span: (0, 10),
            tail_type: "PER".to_string(),
            relation_type: "SIBLING".to_string(),
            confidence: 0.9,
        }];

        // Directed: should fail
        let config_directed = RelationEvalConfig {
            directed_relations: true,
            ..Default::default()
        };
        let metrics = evaluate_relations(&gold, &pred, &config_directed);
        assert!(metrics.strict_f1 < 0.001);

        // Undirected: should succeed
        let config_undirected = RelationEvalConfig {
            directed_relations: false,
            ..Default::default()
        };
        let metrics = evaluate_relations(&gold, &pred, &config_undirected);
        assert!((metrics.strict_f1 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_empty_inputs() {
        let metrics = evaluate_relations(&[], &[], &RelationEvalConfig::default());
        assert!((metrics.strict_f1 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_per_relation_breakdown() {
        let gold = vec![
            RelationGold::new((0, 5), "PER", "A", (10, 15), "ORG", "B", "FOUNDED"),
            RelationGold::new((20, 25), "PER", "C", (30, 35), "ORG", "D", "WORKS_FOR"),
        ];
        let pred = vec![RelationPrediction {
            head_span: (0, 5),
            head_type: "PER".to_string(),
            tail_span: (10, 15),
            tail_type: "ORG".to_string(),
            relation_type: "FOUNDED".to_string(),
            confidence: 0.9,
        }];

        let metrics = evaluate_relations(&gold, &pred, &RelationEvalConfig::default());

        // Check per-relation breakdown
        assert!(metrics.per_relation.contains_key("FOUNDED"));
        assert!(metrics.per_relation.contains_key("WORKS_FOR"));

        let founded = metrics.per_relation.get("FOUNDED").unwrap();
        assert!((founded.strict_f1 - 1.0).abs() < 0.001); // Perfect for FOUNDED

        let works_for = metrics.per_relation.get("WORKS_FOR").unwrap();
        assert!(works_for.strict_f1 < 0.001); // No predictions for WORKS_FOR
    }

    #[test]
    fn test_span_overlap() {
        // Exact match
        assert!((calculate_span_overlap((0, 10), (0, 10)) - 1.0).abs() < 0.001);

        // No overlap
        assert!(calculate_span_overlap((0, 5), (10, 15)) < 0.001);

        // 50% overlap
        let overlap = calculate_span_overlap((0, 10), (5, 15));
        assert!(overlap > 0.3 && overlap < 0.4); // IoU for this case
    }
}
