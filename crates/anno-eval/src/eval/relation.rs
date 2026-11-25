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
//! Standard benchmarks include DocRED, TACRED, SciERC, and NYT-style corpora.

use anno::RelationTriple;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Ordered label inventory supplied to every document in a relation evaluation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RelationSchema {
    /// Entity labels in prompt order.
    pub entity_types: Vec<String>,
    /// Directed relation labels in prompt order.
    pub relation_types: Vec<String>,
}

impl RelationSchema {
    /// Reject unusable or duplicate labels before any inference.
    pub fn validate(&self) -> crate::Result<()> {
        for (name, labels) in [
            ("entity_types", &self.entity_types),
            ("relation_types", &self.relation_types),
        ] {
            let mut seen = std::collections::HashSet::new();
            if labels.is_empty() {
                return Err(crate::Error::InvalidInput(format!(
                    "relation schema {name} must be nonempty"
                )));
            }
            for label in labels {
                if label.trim().is_empty() || label.trim() != label || !seen.insert(label) {
                    return Err(crate::Error::InvalidInput(format!(
                        "relation schema {name} has an empty, padded or duplicate label: {label:?}"
                    )));
                }
            }
        }
        Ok(())
    }
}

/// How the relation evaluator selected its candidate label vocabulary.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum RelationPromptPolicy {
    /// Only labels present in this document's reference relations were offered.
    DocumentGold,
    /// Every document received this caller-supplied vocabulary in the recorded order.
    Fixed(RelationSchema),
}

impl RelationPromptPolicy {
    pub(crate) fn labels<'a>(&'a self, gold: &'a [RelationGold]) -> (Vec<&'a str>, Vec<&'a str>) {
        match self {
            Self::Fixed(schema) => (
                schema.entity_types.iter().map(String::as_str).collect(),
                schema.relation_types.iter().map(String::as_str).collect(),
            ),
            Self::DocumentGold => (
                gold.iter()
                    .flat_map(|r| [r.head_type.as_str(), r.tail_type.as_str()])
                    .collect::<std::collections::BTreeSet<_>>()
                    .into_iter()
                    .collect(),
                gold.iter()
                    .map(|r| r.relation_type.as_str())
                    .collect::<std::collections::BTreeSet<_>>()
                    .into_iter()
                    .collect(),
            ),
        }
    }
}

/// Create relation predictions from entity pairs using a conservative heuristic baseline.
///
/// This is intended as an evaluation fallback when a backend does not implement
/// joint relation extraction. It is deliberately simple and low-dependency.
///
/// Notes:
/// - `relation_types` is accepted for call-site compatibility, but is currently not used
///   for filtering (the heuristic emits a small, fixed label set + `UNKNOWN`).
/// - Offsets are treated as **character** offsets.
#[must_use]
pub fn create_entity_pair_relations(
    entities: &[anno::Entity],
    text: &str,
    relation_types: &[&str],
) -> Vec<RelationPrediction> {
    let _ = relation_types;

    let text_char_len = text.chars().count();
    let max_distance = 200; // catch some cross-sentence relations

    let mut pred_relations = Vec::new();

    // Validate entities first to avoid panics.
    let valid_entities: Vec<&anno::Entity> = entities
        .iter()
        .filter(|e| e.start() < e.end() && e.end() <= text_char_len && e.start() < text_char_len)
        .collect();

    // Limit to avoid O(n²) explosion with many entities.
    let max_entities = 50.min(valid_entities.len());

    for i in 0..max_entities {
        for j in (i + 1)..max_entities {
            let head = valid_entities[i];
            let tail = valid_entities[j];

            // Calculate distance using character offsets.
            let distance = if tail.start() >= head.end() {
                tail.start() - head.end()
            } else if head.start() >= tail.end() {
                head.start() - tail.end()
            } else {
                // Overlapping entities - skip.
                continue;
            };

            if distance > max_distance {
                continue;
            }

            // Extract text between entities using character offsets (not byte offsets).
            let between_text = if head.end() <= tail.start() {
                text.chars()
                    .skip(head.end())
                    .take(tail.start() - head.end())
                    .collect::<String>()
            } else {
                text.chars()
                    .skip(tail.end())
                    .take(head.start() - tail.end())
                    .collect::<String>()
            };

            let between_lower = between_text.to_lowercase();
            let rel_type = if between_lower.contains("founded") || between_lower.contains("founder")
            {
                "FOUNDED"
            } else if between_lower.contains("works for")
                || between_lower.contains("employee")
                || between_lower.contains("employed")
            {
                "WORKS_FOR"
            } else if between_lower.contains("located in")
                || between_lower.contains("based in")
                || between_lower.contains("headquartered")
            {
                "LOCATED_IN"
            } else if between_lower.contains("born in") || between_lower.contains("native of") {
                "BORN_IN"
            } else if between_lower.contains("ceo of")
                || between_lower.contains("president of")
                || between_lower.contains("leads")
            {
                "HEAD_OF"
            } else if between_lower.contains("acquired")
                || between_lower.contains("bought")
                || between_lower.contains("merged")
            {
                "ACQUIRED"
            } else {
                // Explicit "UNKNOWN" so it cannot accidentally match a gold label.
                "UNKNOWN"
            };

            pred_relations.push(RelationPrediction {
                head_span: (head.start(), head.end()),
                head_type: head.entity_type.as_label().to_string(),
                tail_span: (tail.start(), tail.end()),
                tail_type: tail.entity_type.as_label().to_string(),
                relation_type: rel_type.to_string(),
                confidence: 0.5,
            });
        }
    }

    pred_relations
}

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
        entities: &[anno::Entity],
    ) -> Option<Self> {
        let head = entities.get(triple.head_idx)?;
        let tail = entities.get(triple.tail_idx)?;

        Some(Self {
            head_span: (head.start(), head.end()),
            head_type: head.entity_type.as_label().to_string(),
            tail_span: (tail.start(), tail.end()),
            tail_type: tail.entity_type.as_label().to_string(),
            relation_type: triple.relation_type.clone(),
            confidence: triple.confidence.value() as f32,
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

/// Evaluate relation extraction predictions within one document.
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
/// use anno_eval::eval::relation::{evaluate_relations, RelationGold, RelationPrediction, RelationEvalConfig};
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
    evaluate_relation_documents([(gold, pred)], config)
}

/// Match gold and predicted relations within each document, then compute micro scores
/// from corpus counts. Spans in different documents occupy separate coordinate spaces.
pub fn evaluate_relation_documents<'a>(
    documents: impl IntoIterator<Item = (&'a [RelationGold], &'a [RelationPrediction])>,
    config: &RelationEvalConfig,
) -> RelationMetrics {
    let mut by_label = HashMap::new();
    for (gold, pred) in documents {
        count_document(gold, pred, config, &mut by_label);
    }

    let mut num_gold = 0;
    let mut num_predicted = 0;
    let mut boundary_matches = 0;
    let mut strict_matches = 0;
    let mut per_relation = HashMap::with_capacity(by_label.len());
    for (display, mut counts) in by_label.into_values() {
        num_gold += counts.gold_count;
        num_predicted += counts.pred_count;
        boundary_matches += counts.boundary_matches;
        strict_matches += counts.strict_matches;
        counts.boundary_f1 = f1_score(
            ratio(counts.boundary_matches, counts.pred_count),
            ratio(counts.boundary_matches, counts.gold_count),
        );
        counts.strict_f1 = f1_score(
            ratio(counts.strict_matches, counts.pred_count),
            ratio(counts.strict_matches, counts.gold_count),
        );
        per_relation.insert(display, counts);
    }

    // A corpus with no relations on either side is a perfect empty prediction.
    let (boundary_precision, boundary_recall, strict_precision, strict_recall) =
        if num_gold == 0 && num_predicted == 0 {
            (1.0, 1.0, 1.0, 1.0)
        } else {
            (
                ratio(boundary_matches, num_predicted),
                ratio(boundary_matches, num_gold),
                ratio(strict_matches, num_predicted),
                ratio(strict_matches, num_gold),
            )
        };
    RelationMetrics {
        boundary_f1: f1_score(boundary_precision, boundary_recall),
        boundary_precision,
        boundary_recall,
        strict_f1: f1_score(strict_precision, strict_recall),
        strict_precision,
        strict_recall,
        num_predicted,
        num_gold,
        boundary_matches,
        strict_matches,
        per_relation,
    }
}

fn ratio(matches: usize, items: usize) -> f64 {
    if items == 0 {
        0.0
    } else {
        matches as f64 / items as f64
    }
}

fn count_document(
    gold: &[RelationGold],
    pred: &[RelationPrediction],
    config: &RelationEvalConfig,
    by_label: &mut HashMap<String, (String, RelationTypeMetrics)>,
) {
    let mut gold_matched_boundary = vec![false; gold.len()];
    let mut gold_matched_strict = vec![false; gold.len()];
    // Count gold per relation type
    for g in gold {
        let (display, counts) = by_label.entry(g.relation_type.to_lowercase()).or_default();
        // Prefer the gold spelling. Resolve case variants deterministically across documents.
        if counts.gold_count == 0 || g.relation_type < *display {
            display.clone_from(&g.relation_type);
        }
        counts.gold_count += 1;
    }

    // Count pred per relation type
    for p in pred {
        let (display, counts) = by_label.entry(p.relation_type.to_lowercase()).or_default();
        if counts.gold_count == 0 && (counts.pred_count == 0 || p.relation_type < *display) {
            display.clone_from(&p.relation_type);
        }
        counts.pred_count += 1;
    }

    // Strict matching: exact entity spans + relation type
    for p in pred {
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

                let (_, counts) = by_label
                    .get_mut(&g.relation_type.to_lowercase())
                    .expect("counted gold label");
                counts.strict_matches += 1;
                break;
            }
        }
    }

    // Boundary matching: overlapping entity spans + relation type
    for p in pred {
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

                let (_, counts) = by_label
                    .get_mut(&g.relation_type.to_lowercase())
                    .expect("counted gold label");
                counts.boundary_matches += 1;
                break;
            }
        }
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
        rels.sort_by_key(|b| std::cmp::Reverse(b.1.gold_count));
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
            rels.sort_by_key(|b| std::cmp::Reverse(b.1.gold_count));

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
    fn corpus_scores_use_relation_counts_and_ignore_empty_documents() {
        let matched = RelationGold::new((0, 4), "ORG", "Acme", (8, 12), "ORG", "Beta", "OWNS");
        let missed = RelationGold::new((1, 2), "ORG", "X", (5, 6), "ORG", "Y", "OWNS");
        let predicted = RelationPrediction {
            head_span: (0, 4),
            head_type: "ORG".into(),
            tail_span: (8, 12),
            tail_type: "ORG".into(),
            relation_type: "owns".into(), // Matching and the breakdown both ignore case.
            confidence: 1.0,
        };
        let gold = [matched.clone(), missed];
        let predictions = [predicted];
        // First document: one correct and one missed. Second: miss the same offsets.
        let docs: [(&[RelationGold], &[RelationPrediction]); 3] = [
            (&gold, &predictions),
            (std::slice::from_ref(&matched), &[]),
            (&[], &[]),
        ];
        let actual = evaluate_relation_documents(docs, &RelationEvalConfig::default());
        assert_eq!(
            (
                actual.num_gold,
                actual.num_predicted,
                actual.strict_matches,
                actual.boundary_matches
            ),
            (3, 1, 1, 1)
        );
        assert_eq!((actual.strict_f1, actual.boundary_f1), (0.5, 0.5));
        assert_eq!(actual.per_relation.len(), 1);
        let by_type = &actual.per_relation["OWNS"];
        assert_eq!(
            (
                by_type.gold_count,
                by_type.pred_count,
                by_type.strict_f1,
                by_type.boundary_f1
            ),
            (3, 1, 0.5, 0.5)
        );
    }

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

    #[test]
    fn test_relation_type_case_insensitive() {
        // Relation types should match case-insensitively
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
            relation_type: "founded".to_string(), // lowercase
            confidence: 0.9,
        }];

        let metrics = evaluate_relations(&gold, &pred, &RelationEvalConfig::default());
        assert!(
            (metrics.strict_f1 - 1.0).abs() < 0.001,
            "Relation type matching should be case-insensitive"
        );
    }

    #[test]
    fn test_entity_type_match_disabled() {
        // When require_entity_type_match is false, entity types don't need to match
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
            head_type: "PERSON".to_string(), // Different from PER
            tail_span: (20, 25),
            tail_type: "COMPANY".to_string(), // Different from ORG
            relation_type: "FOUNDED".to_string(),
            confidence: 0.9,
        }];

        // With type match required: should fail
        let config_strict = RelationEvalConfig {
            require_entity_type_match: true,
            ..Default::default()
        };
        let metrics = evaluate_relations(&gold, &pred, &config_strict);
        assert!(metrics.strict_f1 < 0.001, "Type mismatch should fail");

        // Without type match required: should succeed
        let config_lenient = RelationEvalConfig {
            require_entity_type_match: false,
            ..Default::default()
        };
        let metrics = evaluate_relations(&gold, &pred, &config_lenient);
        assert!(
            (metrics.strict_f1 - 1.0).abs() < 0.001,
            "Without type matching, should succeed"
        );
    }

    #[test]
    fn test_no_gold_all_pred() {
        // No gold annotations, but predictions exist
        let gold: Vec<RelationGold> = vec![];
        let pred = vec![RelationPrediction {
            head_span: (0, 10),
            head_type: "PER".to_string(),
            tail_span: (20, 25),
            tail_type: "ORG".to_string(),
            relation_type: "FOUNDED".to_string(),
            confidence: 0.9,
        }];

        let metrics = evaluate_relations(&gold, &pred, &RelationEvalConfig::default());
        // Precision = 0 (no correct), Recall = undefined (no gold) -> 0
        assert!(metrics.strict_precision < 0.001);
        assert!(metrics.strict_f1 < 0.001);
    }

    #[test]
    fn test_all_gold_no_pred() {
        // Gold annotations exist, but no predictions
        let gold = vec![RelationGold::new(
            (0, 10),
            "PER",
            "Steve Jobs",
            (20, 25),
            "ORG",
            "Apple",
            "FOUNDED",
        )];
        let pred: Vec<RelationPrediction> = vec![];

        let metrics = evaluate_relations(&gold, &pred, &RelationEvalConfig::default());
        // Precision = undefined (no pred) -> 0, Recall = 0 (no correct)
        assert!(metrics.strict_recall < 0.001);
        assert!(metrics.strict_f1 < 0.001);
    }

    #[test]
    fn test_boundary_vs_strict_matching() {
        // Test that boundary matching is more lenient than strict
        let gold = vec![RelationGold::new(
            (0, 15),
            "PER",
            "Steve Jobs Jr.",
            (25, 35),
            "ORG",
            "Apple Inc.",
            "FOUNDED",
        )];
        // Predictions have overlapping but not exact spans
        let pred = vec![RelationPrediction {
            head_span: (0, 10), // Partial overlap (10/15 = 66%)
            head_type: "PER".to_string(),
            tail_span: (25, 30), // Partial overlap (5/10 = 50%)
            tail_type: "ORG".to_string(),
            relation_type: "FOUNDED".to_string(),
            confidence: 0.9,
        }];

        let metrics = evaluate_relations(&gold, &pred, &RelationEvalConfig::default());

        // Strict should fail (spans don't match exactly)
        assert!(
            metrics.strict_f1 < 0.001,
            "Strict should fail for partial overlap"
        );
        // Boundary should succeed (>= 0.5 overlap threshold)
        assert!(
            metrics.boundary_f1 > 0.5,
            "Boundary should succeed for sufficient overlap"
        );
    }
}
