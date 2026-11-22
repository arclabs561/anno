//! Legacy NER Evaluation Metrics (MUC/SemEval-2013 standards).
//!
//! # When to Use This Module
//!
//! Use this for **backwards compatibility** with published benchmarks and papers
//! that report results using MUC/SemEval-2013 evaluation methodology.
//!
//! For **modern evaluation** (2024+), prefer:
//! - Dataset quality metrics (unseen entity ratio, entity ambiguity, cross-corpus evaluation)
//! - Fine-grained error taxonomy analysis
//! - Semantic similarity metrics for LLM-based systems
//!
//! # Known Limitations (2023-2024 Research)
//!
//! **Benchmark noise**: public NER benchmarks can contain annotation noise; this can
//! distort “headline” scores and error analyses. See CleanCoNLL (Rücker & Akbik, 2023)
//! for one concrete audit/correction pass on CoNLL-03.
//!
//! **Measurement conflation**: Strict F1 conflates:
//! - Boundary errors ("John" vs "John Smith")
//! - Type errors (Person vs Organization)
//! - Dataset artifacts (inconsistent annotation guidelines)
//!
//! **Generalization blindness**: Same-corpus F1 doesn't predict cross-corpus performance.
//! Models memorize training entities; unseen entities cause most real errors.
//!
//! # Evaluation Schemas (SemEval-2013 Task 9.1)
//!
//! - **Strict**: Exact boundary AND exact type match required
//! - **Exact**: Exact boundary match only (type ignored)
//! - **Partial**: Partial boundary overlap (type ignored)
//! - **Type**: Some overlap required, type must match
//!
//! Each schema tracks MUC-style counts:
//! - **Correct (COR)**: System output matches gold annotation
//! - **Incorrect (INC)**: System and gold don't match
//! - **Partial (PAR)**: Partial match (boundaries overlap but not exact)
//! - **Missing (MIS)**: Gold annotation not captured by system
//! - **Spurious (SPU)**: System produces entity not in gold
//!
//! # References
//!
//! - CleanCoNLL (2023): <https://arxiv.org/abs/2310.16225>
//! - OntoNotes Errors (2024): <https://arxiv.org/abs/2406.19172>
//! - TMR - Tough Mentions Recall (2021): <https://arxiv.org/abs/2103.12312>

use anno::{Entity, EntityType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::Range;

/// A gold or predicted entity span for evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EvalSpan {
    /// Entity type label
    pub entity_type: EntityType,
    /// Start offset (character or token)
    pub start: usize,
    /// End offset (exclusive)
    pub end: usize,
    /// Surface text
    pub text: String,
}

impl EvalSpan {
    /// Create a new evaluation span.
    #[must_use]
    pub fn new(entity_type: EntityType, start: usize, end: usize, text: impl Into<String>) -> Self {
        Self {
            entity_type,
            start,
            end,
            text: text.into(),
        }
    }

    /// Check if two spans have exact boundary match.
    #[must_use]
    pub fn exact_boundary_match(&self, other: &Self) -> bool {
        self.start == other.start && self.end == other.end
    }

    /// Check if two spans have any overlap.
    #[must_use]
    pub fn has_overlap(&self, other: &Self) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// Check if types match.
    #[must_use]
    pub fn type_match(&self, other: &Self) -> bool {
        super::entity_type_matches(&self.entity_type, &other.entity_type)
    }
}

/// Projects an entity onto its bounding interval. Use [`evaluate_entities`] for
/// entities that may contain discontinuous mentions.
impl From<&Entity> for EvalSpan {
    fn from(entity: &Entity) -> Self {
        Self {
            entity_type: entity.entity_type.clone(),
            start: entity.start(),
            end: entity.end(),
            text: entity.text.clone(),
        }
    }
}

// Preserve the contiguous public EvalSpan API while borrowing canonical segments
// directly from an Entity. The common single-range case needs no heap allocation.
enum Coverage<'a> {
    Contiguous(Range<usize>),
    Segmented(&'a [Range<usize>]),
}

struct ScoringSpan<'a> {
    entity_type: &'a EntityType,
    coverage: Coverage<'a>,
}

impl ScoringSpan<'_> {
    fn ranges(&self) -> &[Range<usize>] {
        match &self.coverage {
            Coverage::Contiguous(range) => std::slice::from_ref(range),
            Coverage::Segmented(ranges) => ranges,
        }
    }

    fn type_key(&self) -> String {
        format!("{:?}", EntityType::from_label(self.entity_type.as_label()))
    }
}

/// MUC-style evaluation counts.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MucCounts {
    /// Number of correct matches
    pub correct: usize,
    /// Number of incorrect matches
    pub incorrect: usize,
    /// Number of partial matches
    pub partial: usize,
    /// Number of missed gold entities
    pub missed: usize,
    /// Number of spurious system entities
    pub spurious: usize,
}

impl MucCounts {
    /// Total possible entities (gold standard count).
    #[must_use]
    pub fn possible(&self) -> usize {
        self.correct + self.incorrect + self.partial + self.missed
    }

    /// Total actual entities (system output count).
    #[must_use]
    pub fn actual(&self) -> usize {
        self.correct + self.incorrect + self.partial + self.spurious
    }

    /// Precision for exact match schema.
    #[must_use]
    pub fn precision_exact(&self) -> f64 {
        let actual = self.actual();
        if actual == 0 {
            return 0.0;
        }
        self.correct as f64 / actual as f64
    }

    /// Recall for exact match schema.
    #[must_use]
    pub fn recall_exact(&self) -> f64 {
        let possible = self.possible();
        if possible == 0 {
            return 0.0;
        }
        self.correct as f64 / possible as f64
    }

    /// Precision for partial match schema (partial counts as 0.5).
    #[must_use]
    pub fn precision_partial(&self) -> f64 {
        let actual = self.actual();
        if actual == 0 {
            return 0.0;
        }
        (self.correct as f64 + 0.5 * self.partial as f64) / actual as f64
    }

    /// Recall for partial match schema (partial counts as 0.5).
    #[must_use]
    pub fn recall_partial(&self) -> f64 {
        let possible = self.possible();
        if possible == 0 {
            return 0.0;
        }
        (self.correct as f64 + 0.5 * self.partial as f64) / possible as f64
    }

    /// F1 score for exact match.
    #[must_use]
    pub fn f1_exact(&self) -> f64 {
        let p = self.precision_exact();
        let r = self.recall_exact();
        if p + r == 0.0 {
            return 0.0;
        }
        2.0 * p * r / (p + r)
    }

    /// F1 score for partial match.
    #[must_use]
    pub fn f1_partial(&self) -> f64 {
        let p = self.precision_partial();
        let r = self.recall_partial();
        if p + r == 0.0 {
            return 0.0;
        }
        2.0 * p * r / (p + r)
    }

    /// Merge counts from another set.
    pub fn merge(&mut self, other: &MucCounts) {
        self.correct += other.correct;
        self.incorrect += other.incorrect;
        self.partial += other.partial;
        self.missed += other.missed;
        self.spurious += other.spurious;
    }
}

/// Complete NER evaluation results across all schemas.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NerEvalResults {
    /// Strict evaluation: exact boundary AND exact type
    pub strict: MucCounts,
    /// Exact evaluation: exact boundary only
    pub exact: MucCounts,
    /// Partial evaluation: partial boundary overlap
    pub partial: MucCounts,
    /// Type evaluation: some overlap + type match
    pub ent_type: MucCounts,
    /// Per-entity-type breakdown (strict mode only)
    pub by_type: HashMap<String, MucCounts>,
}

impl NerEvalResults {
    /// Create new empty results.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get summary statistics.
    #[must_use]
    pub fn summary(&self) -> NerEvalSummary {
        NerEvalSummary {
            strict_precision: self.strict.precision_exact(),
            strict_recall: self.strict.recall_exact(),
            strict_f1: self.strict.f1_exact(),
            exact_precision: self.exact.precision_exact(),
            exact_recall: self.exact.recall_exact(),
            exact_f1: self.exact.f1_exact(),
            partial_precision: self.partial.precision_partial(),
            partial_recall: self.partial.recall_partial(),
            partial_f1: self.partial.f1_partial(),
            type_precision: self.ent_type.precision_exact(),
            type_recall: self.ent_type.recall_exact(),
            type_f1: self.ent_type.f1_exact(),
        }
    }

    /// Merge results from another evaluation.
    pub fn merge(&mut self, other: &NerEvalResults) {
        self.strict.merge(&other.strict);
        self.exact.merge(&other.exact);
        self.partial.merge(&other.partial);
        self.ent_type.merge(&other.ent_type);

        for (entity_type, counts) in &other.by_type {
            self.by_type
                .entry(entity_type.clone())
                .or_default()
                .merge(counts);
        }
    }

    /// Format as markdown table.
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let summary = self.summary();
        format!(
            "| Schema | Precision | Recall | F1 |\n\
             |--------|-----------|--------|----|\n\
             | Strict | {:.1}% | {:.1}% | {:.1}% |\n\
             | Exact  | {:.1}% | {:.1}% | {:.1}% |\n\
             | Partial| {:.1}% | {:.1}% | {:.1}% |\n\
             | Type   | {:.1}% | {:.1}% | {:.1}% |",
            summary.strict_precision * 100.0,
            summary.strict_recall * 100.0,
            summary.strict_f1 * 100.0,
            summary.exact_precision * 100.0,
            summary.exact_recall * 100.0,
            summary.exact_f1 * 100.0,
            summary.partial_precision * 100.0,
            summary.partial_recall * 100.0,
            summary.partial_f1 * 100.0,
            summary.type_precision * 100.0,
            summary.type_recall * 100.0,
            summary.type_f1 * 100.0,
        )
    }
}

/// Summary metrics across all evaluation schemas.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NerEvalSummary {
    /// Strict schema precision
    pub strict_precision: f64,
    /// Strict schema recall
    pub strict_recall: f64,
    /// Strict schema F1
    pub strict_f1: f64,
    /// Exact schema precision
    pub exact_precision: f64,
    /// Exact schema recall
    pub exact_recall: f64,
    /// Exact schema F1
    pub exact_f1: f64,
    /// Partial schema precision
    pub partial_precision: f64,
    /// Partial schema recall
    pub partial_recall: f64,
    /// Partial schema F1
    pub partial_f1: f64,
    /// Type schema precision
    pub type_precision: f64,
    /// Type schema recall
    pub type_recall: f64,
    /// Type schema F1
    pub type_f1: f64,
}

/// Evaluate NER predictions against gold annotations.
///
/// Implements SemEval-2013 Task 9.1 evaluation methodology.
///
/// Exact matches take precedence over partial overlaps. Within each match
/// priority we choose a maximum-cardinality assignment in canonical span order.
/// Each gold span and prediction contributes to at most one pair.
///
////// # Example
///
/// ```
/// use anno_eval::eval::ner_metrics::{evaluate_ner, EvalSpan};
/// use anno::EntityType;
///
/// let gold = vec![
///     EvalSpan::new(EntityType::Person, 0, 8, "John Doe"),
///     EvalSpan::new(EntityType::Location, 15, 23, "New York"),
/// ];
///
/// let predicted = vec![
///     EvalSpan::new(EntityType::Person, 0, 8, "John Doe"),  // Correct
///     EvalSpan::new(EntityType::Organization, 15, 23, "New York"),  // Wrong type
/// ];
///
/// let results = evaluate_ner(&gold, &predicted);
/// assert_eq!(results.strict.correct, 1);
/// assert_eq!(results.strict.incorrect, 1);
/// ```
#[must_use]
pub fn evaluate_ner(gold: &[EvalSpan], predicted: &[EvalSpan]) -> NerEvalResults {
    fn views(spans: &[EvalSpan]) -> Vec<ScoringSpan<'_>> {
        spans
            .iter()
            .map(|s| ScoringSpan {
                entity_type: &s.entity_type,
                coverage: Coverage::Contiguous(s.start..s.end),
            })
            .collect()
    }
    score_spans(&views(gold), &views(predicted))
}

fn score_spans(gold: &[ScoringSpan<'_>], predicted: &[ScoringSpan<'_>]) -> NerEvalResults {
    let mut results = NerEvalResults::new();

    // Track which predictions have been matched
    let mut matched_preds: Vec<bool> = vec![false; predicted.len()];

    // Stable span order also fixes which equally good assignment is frozen
    // before the next priority is considered.
    fn canonical_order(spans: &[ScoringSpan<'_>]) -> Vec<usize> {
        let mut indices: Vec<_> = (0..spans.len()).collect();
        let keys: Vec<_> = spans
            .iter()
            .map(|span| {
                let ranges = span.ranges();
                (
                    ranges.first().map(|r| r.start),
                    ranges.last().map(|r| r.end),
                    span.type_key(),
                )
            })
            .collect();
        indices.sort_by(|&i, &j| {
            keys[i].cmp(&keys[j]).then_with(|| {
                let boundaries = |r: &Range<usize>| (r.start, r.end);
                spans[i]
                    .ranges()
                    .iter()
                    .map(boundaries)
                    .cmp(spans[j].ranges().iter().map(boundaries))
            })
        });
        indices
    }
    let gold_order = canonical_order(gold);
    let pred_order = canonical_order(predicted);
    let mut gold_match = vec![None; gold.len()];
    for tier in [
        MatchType::ExactBoth,
        MatchType::ExactBoundaryWrongType,
        MatchType::PartialBoundaryCorrectType,
        MatchType::PartialBoundaryWrongType,
    ] {
        let gs: Vec<_> = gold_order
            .iter()
            .copied()
            .filter(|&g| gold_match[g].is_none())
            .collect();
        let ps: Vec<_> = pred_order
            .iter()
            .copied()
            .filter(|&p| !matched_preds[p])
            .collect();
        let pairs = super::matching::maximum_cardinality_matching(ps.len(), gs.len(), |p, g| {
            classify_match(&gold[gs[g]], &predicted[ps[p]]) == tier
        });
        for (p, g) in pairs.into_iter().enumerate() {
            if let Some(g) = g {
                gold_match[gs[g]] = Some((ps[p], tier));
                matched_preds[ps[p]] = true;
            }
        }
    }

    for (gold_span, best_match) in gold.iter().zip(gold_match) {
        let type_counts = results.by_type.entry(gold_span.type_key()).or_default();

        match best_match {
            Some((pred_idx, match_type)) => {
                let pred_span = &predicted[pred_idx];

                // A wrong type consumes one gold item and one prediction in different classes.
                if match_type == MatchType::ExactBoth {
                    type_counts.correct += 1;
                } else if match_type == MatchType::PartialBoundaryCorrectType {
                    type_counts.incorrect += 1;
                } else {
                    type_counts.missed += 1;
                    results
                        .by_type
                        .entry(pred_span.type_key())
                        .or_default()
                        .spurious += 1;
                }
                update_counts(&mut results, match_type);
            }
            None => {
                // Missing: gold entity not found
                results.strict.missed += 1;
                results.exact.missed += 1;
                results.partial.missed += 1;
                results.ent_type.missed += 1;

                type_counts.missed += 1;
            }
        }
    }

    // Count spurious predictions (not matched to any gold)
    for (pred_idx, matched) in matched_preds.iter().enumerate() {
        if !matched {
            results.strict.spurious += 1;
            results.exact.spurious += 1;
            results.partial.spurious += 1;
            results.ent_type.spurious += 1;

            results
                .by_type
                .entry(predicted[pred_idx].type_key())
                .or_default()
                .spurious += 1;
        }
    }

    results
}

/// Classification of match between gold and predicted spans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatchType {
    /// No match
    None,
    /// Partial boundary overlap, type mismatch
    PartialBoundaryWrongType,
    /// Partial boundary overlap, type match
    PartialBoundaryCorrectType,
    /// Exact boundary, type mismatch
    ExactBoundaryWrongType,
    /// Exact boundary AND type match
    ExactBoth,
}

/// Classify the match type between gold and predicted spans.
fn classify_match(gold: &ScoringSpan<'_>, pred: &ScoringSpan<'_>) -> MatchType {
    let (gold_ranges, pred_ranges) = (gold.ranges(), pred.ranges());
    if gold_ranges.is_empty() || pred_ranges.is_empty() {
        return MatchType::None;
    }
    let same_type = super::entity_type_matches(gold.entity_type, pred.entity_type);
    if gold_ranges == pred_ranges {
        return if same_type {
            MatchType::ExactBoth
        } else {
            MatchType::ExactBoundaryWrongType
        };
    }
    let (mut g, mut p) = (0, 0);
    while g < gold_ranges.len() && p < pred_ranges.len() {
        let (left, right) = (&gold_ranges[g], &pred_ranges[p]);
        if left.start.max(right.start) < left.end.min(right.end) {
            return if same_type {
                MatchType::PartialBoundaryCorrectType
            } else {
                MatchType::PartialBoundaryWrongType
            };
        }
        if left.end <= right.end {
            g += 1;
        } else {
            p += 1;
        }
    }
    MatchType::None
}

/// Update counts based on match type.
fn update_counts(results: &mut NerEvalResults, match_type: MatchType) {
    let exact_boundary = matches!(
        match_type,
        MatchType::ExactBoth | MatchType::ExactBoundaryWrongType
    );
    let type_match = matches!(
        match_type,
        MatchType::ExactBoth | MatchType::PartialBoundaryCorrectType
    );

    // Strict: both boundary and type must be exact
    if exact_boundary && type_match {
        results.strict.correct += 1;
    } else {
        results.strict.incorrect += 1;
    }

    // Exact: boundary must be exact, type ignored
    if exact_boundary {
        results.exact.correct += 1;
    } else {
        results.exact.incorrect += 1;
    }

    // Partial: any overlap counts, boundary can be partial
    match match_type {
        MatchType::ExactBoth | MatchType::ExactBoundaryWrongType => {
            results.partial.correct += 1;
        }
        MatchType::PartialBoundaryCorrectType | MatchType::PartialBoundaryWrongType => {
            results.partial.partial += 1;
        }
        MatchType::None => {
            results.partial.incorrect += 1;
        }
    }

    // Type: overlap required + type must match
    if type_match {
        results.ent_type.correct += 1;
    } else {
        results.ent_type.incorrect += 1;
    }
}

/// Evaluate entity mentions directly, retaining every covered segment.
/// Exact matching requires identical covered ranges. Partial and type matching
/// require at least one covered position in common.
#[must_use]
pub fn evaluate_entities(gold: &[Entity], predicted: &[Entity]) -> NerEvalResults {
    fn views(entities: &[Entity]) -> Vec<ScoringSpan<'_>> {
        entities
            .iter()
            .map(|e| ScoringSpan {
                entity_type: &e.entity_type,
                coverage: match &e.discontinuous_span {
                    Some(span) => Coverage::Segmented(span.segments()),
                    None => Coverage::Contiguous(e.start()..e.end()),
                },
            })
            .collect()
    }
    score_spans(&views(gold), &views(predicted))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_character_support_agrees_with_a_bitset_oracle_in_all_schemas() {
        let text = "Aé🦊B";
        let mention = |mask: u8, label: EntityType, store_segments: bool| -> Vec<Entity> {
            if mask == 0 {
                return vec![];
            }
            let covered: Vec<_> = (0..4).filter(|i| mask & (1 << i) != 0).collect();
            let start = covered[0];
            let end = covered.last().unwrap() + 1;
            let surface: String = text.chars().skip(start).take(end - start).collect();
            let mut entity = Entity::new(surface, label, start, end, 1.0);
            if store_segments || end - start != covered.len() {
                let ranges = covered.into_iter().map(|i| i..i + 1).collect();
                entity.set_discontinuous_span(anno::DiscontinuousSpan::new(ranges));
            }
            vec![entity]
        };
        for g in 0..16 {
            for p in 0..16 {
                for same_type in [false, true] {
                    let gold = mention(g, EntityType::Person, false);
                    let label = if same_type {
                        EntityType::Person
                    } else {
                        EntityType::Organization
                    };
                    let predicted = mention(p, label, true);
                    let metrics = evaluate_entities(&gold, &predicted);
                    let exact = g != 0 && g == p;
                    let overlap = g & p != 0;
                    let expected_partial = if exact {
                        1.0
                    } else if overlap {
                        0.5
                    } else {
                        0.0
                    };
                    let scores = metrics.summary();
                    assert_eq!(
                        (
                            scores.strict_f1,
                            scores.exact_f1,
                            scores.partial_f1,
                            scores.type_f1
                        ),
                        (
                            f64::from(exact && same_type),
                            f64::from(exact),
                            expected_partial,
                            f64::from(overlap && same_type)
                        ),
                        "gold={g:04b}, prediction={p:04b}, same_type={same_type}"
                    );
                    if g != 0 {
                        let person = &metrics.by_type["Person"];
                        assert_eq!(person.correct, usize::from(exact && same_type));
                        assert_eq!(person.missed, usize::from(!overlap || !same_type));
                    }
                }
            }
        }
    }

    /// nervaluate matches a prediction to an exact gold span before any
    /// overlapping one, so an outer nested gold span cannot take the inner
    /// span's exact prediction. nervaluate: strict correct=1, missed=1.
    #[test]
    fn evaluate_ner_prefers_exact_match_across_gold_spans() {
        let gold = vec![
            EvalSpan::new(EntityType::Person, 0, 9, "John Smith"),
            EvalSpan::new(EntityType::Person, 0, 3, "John"),
        ];
        let pred = vec![EvalSpan::new(EntityType::Person, 0, 3, "John")];
        let r = evaluate_ner(&gold, &pred);
        assert_eq!(r.strict.correct, 1, "strict={:?}", r.strict);
        assert_eq!(r.strict.missed, 1);
        assert_eq!(r.strict.incorrect, 0);
    }

    fn span(t: EntityType, start: usize, end: usize) -> EvalSpan {
        EvalSpan::new(t, start, end, "test")
    }

    #[test]
    fn test_exact_match() {
        let gold = vec![span(EntityType::Person, 0, 5)];
        let pred = vec![span(EntityType::Person, 0, 5)];

        let results = evaluate_ner(&gold, &pred);
        assert_eq!(results.strict.correct, 1);
        assert_eq!(results.exact.correct, 1);
        assert_eq!(results.partial.correct, 1);
        assert_eq!(results.ent_type.correct, 1);
    }

    #[test]
    fn test_wrong_type() {
        let gold = vec![span(EntityType::Person, 0, 5)];
        let pred = vec![span(EntityType::Organization, 0, 5)];

        let results = evaluate_ner(&gold, &pred);
        // Strict: incorrect (type mismatch)
        assert_eq!(results.strict.incorrect, 1);
        // Exact: correct (boundary matches)
        assert_eq!(results.exact.correct, 1);
        // Partial: correct (exact boundary)
        assert_eq!(results.partial.correct, 1);
        // Type: incorrect (type mismatch)
        assert_eq!(results.ent_type.incorrect, 1);
    }

    #[test]
    fn test_custom_type_labels_match_case_insensitively() {
        let gold = vec![span(
            EntityType::custom("MISC", anno::EntityCategory::Misc),
            0,
            5,
        )];
        let pred = vec![
            span(EntityType::custom("misc", anno::EntityCategory::Misc), 0, 5),
            span(
                EntityType::custom("misc", anno::EntityCategory::Misc),
                10,
                15,
            ),
        ];

        let results = evaluate_ner(&gold, &pred);
        assert_eq!(results.strict.correct, 1);
        assert_eq!(results.ent_type.correct, 1);
        assert_eq!(results.by_type.len(), 1);
        let counts = results.by_type.values().next().unwrap();
        assert_eq!(counts.correct, 1);
        assert_eq!(counts.spurious, 1);
    }

    #[test]
    fn wnut_gold_and_decoded_labels_share_type_counts() {
        use crate::eval::loader::{DatasetId, DatasetLoader};
        use anno::schema::{map_to_canonical, DatasetSchema};

        let dir = tempfile::tempdir().unwrap();
        let loader = DatasetLoader::with_cache_dir(dir.path()).unwrap();
        for label in DatasetId::Wnut17.entity_types() {
            let data = loader
                .parse_content_str(&format!("example\tB-{label}\n"), DatasetId::Wnut17)
                .unwrap();
            let gold_type = data.sentences[0].entities().pop().unwrap().entity_type;
            let type_key = format!("{gold_type:?}");
            let gold = [span(gold_type, 0, 7)];
            let pred = [
                span(map_to_canonical(label, None), 0, 7),
                span(map_to_canonical(label, Some(DatasetSchema::WNUT17)), 8, 15),
            ];

            let results = evaluate_ner(&gold, &pred);
            assert_eq!(results.strict.correct, 1, "{label}");
            assert_eq!(results.by_type.len(), 1, "{label}");
            let counts = &results.by_type[&type_key];
            assert_eq!((counts.correct, counts.spurious), (1, 1), "{label}");
        }
    }

    #[test]
    fn test_partial_boundary() {
        let gold = vec![span(EntityType::Person, 0, 10)];
        let pred = vec![span(EntityType::Person, 0, 8)]; // Partial overlap

        let results = evaluate_ner(&gold, &pred);
        // Strict: incorrect (boundary mismatch)
        assert_eq!(results.strict.incorrect, 1);
        assert_eq!(results.by_type["Person"].incorrect, 1);
        // Exact: incorrect (boundary mismatch)
        assert_eq!(results.exact.incorrect, 1);
        // Partial: partial match
        assert_eq!(results.partial.partial, 1);
        // Type: correct (overlap + type match)
        assert_eq!(results.ent_type.correct, 1);
    }

    #[test]
    fn test_missing_entity() {
        let gold = vec![span(EntityType::Person, 0, 5)];
        let pred: Vec<EvalSpan> = vec![];

        let results = evaluate_ner(&gold, &pred);
        assert_eq!(results.strict.missed, 1);
        assert_eq!(results.exact.missed, 1);
        assert_eq!(results.partial.missed, 1);
        assert_eq!(results.ent_type.missed, 1);
    }

    #[test]
    fn test_spurious_entity() {
        let gold: Vec<EvalSpan> = vec![];
        let pred = vec![span(EntityType::Person, 0, 5)];

        let results = evaluate_ner(&gold, &pred);
        assert_eq!(results.strict.spurious, 1);
        assert_eq!(results.exact.spurious, 1);
        assert_eq!(results.partial.spurious, 1);
        assert_eq!(results.ent_type.spurious, 1);
    }

    #[test]
    fn test_precision_recall_f1() {
        let gold = vec![
            span(EntityType::Person, 0, 5),
            span(EntityType::Location, 10, 15),
        ];
        let pred = vec![
            span(EntityType::Person, 0, 5),         // Correct
            span(EntityType::Organization, 20, 25), // Spurious
        ];

        let results = evaluate_ner(&gold, &pred);

        // 1 correct, 1 spurious, 1 missed
        assert_eq!(results.strict.correct, 1);
        assert_eq!(results.strict.spurious, 1);
        assert_eq!(results.strict.missed, 1);

        // Precision: 1/2 = 0.5
        assert!((results.strict.precision_exact() - 0.5).abs() < 0.01);
        // Recall: 1/2 = 0.5
        assert!((results.strict.recall_exact() - 0.5).abs() < 0.01);
        // F1: 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
        assert!((results.strict.f1_exact() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_markdown_output() {
        let gold = vec![span(EntityType::Person, 0, 5)];
        let pred = vec![span(EntityType::Person, 0, 5)];

        let results = evaluate_ner(&gold, &pred);
        let md = results.to_markdown();

        assert!(md.contains("Strict"));
        assert!(md.contains("100.0%"));
    }

    #[test]
    fn test_per_type_breakdown() {
        let gold = vec![
            span(EntityType::Person, 0, 5),
            span(EntityType::Person, 10, 15),
            span(EntityType::Location, 20, 25),
        ];
        let pred = vec![
            span(EntityType::Person, 0, 5),         // Correct
            span(EntityType::Organization, 10, 15), // Wrong type
            span(EntityType::Location, 20, 25),     // Correct
        ];

        let results = evaluate_ner(&gold, &pred);

        let person = &results.by_type["Person"];
        assert_eq!((person.correct, person.missed, person.actual()), (1, 1, 1));
        let org = &results.by_type["Organization"];
        assert_eq!((org.spurious, org.possible(), org.actual()), (1, 0, 1));
        assert_eq!(results.by_type["Location"].correct, 1);
        assert_eq!(results.strict.incorrect, 1);
    }
}

#[cfg(test)]
#[test]
fn overlapping_entities_score_the_same_in_every_order() {
    let person = |a, b| EvalSpan::new(EntityType::Person, a, b, "x");
    let mut gold = vec![person(0, 4), person(5, 9)];
    let mut pred = vec![person(2, 7), person(0, 2)];
    for _ in 0..2 {
        for _ in 0..2 {
            let scores = evaluate_ner(&gold, &pred);
            assert_eq!(scores.partial.partial, 2);
            assert_eq!(scores.ent_type.correct, 2);
            pred.reverse();
        }
        gold.reverse();
    }
}
