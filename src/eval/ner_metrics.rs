//! Legacy NER Evaluation Metrics (MUC/SemEval-2013 standards).
//!
//! # When to Use This Module
//!
//! Use this for **backwards compatibility** with published benchmarks and papers
//! that report results using MUC/SemEval-2013 evaluation methodology.
//!
//! For **modern evaluation** (2024+), prefer:
//! - [`crate::eval::dataset_quality`]: Dataset quality metrics (unseen entity ratio,
//!   entity ambiguity, cross-corpus evaluation)
//! - [`crate::eval::error_analysis`]: Fine-grained error taxonomy
//! - Semantic similarity metrics for LLM-based systems
//!
//! # Known Limitations (2023-2024 Research)
//!
//! **Benchmark noise**: 7% of CoNLL-03 labels are incorrect. On the original benchmark,
//! **47% of "errors" were actually correct predictions** penalized by annotation mistakes.
//! After correction (CleanCoNLL), SOTA F1 jumped from 94% to 97.1%.
//! (Rücker & Akbik, EMNLP 2023)
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

use crate::{Entity, EntityType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
        self.entity_type == other.entity_type
    }
}

impl From<&Entity> for EvalSpan {
    fn from(entity: &Entity) -> Self {
        Self {
            entity_type: entity.entity_type.clone(),
            start: entity.start,
            end: entity.end,
            text: entity.text.clone(),
        }
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
/// # Example
///
/// ```
/// use anno::eval::ner_metrics::{evaluate_ner, EvalSpan};
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
    let mut results = NerEvalResults::new();

    // Track which predictions have been matched
    let mut matched_preds: Vec<bool> = vec![false; predicted.len()];

    for gold_span in gold {
        let entity_type_str = format!("{:?}", gold_span.entity_type);

        // Find best matching prediction
        let mut best_match: Option<(usize, MatchType)> = None;

        for (pred_idx, pred_span) in predicted.iter().enumerate() {
            if matched_preds[pred_idx] {
                continue;
            }

            let match_type = classify_match(gold_span, pred_span);
            if match_type != MatchType::None {
                // Prefer exact matches over partial
                if best_match.is_none()
                    || match_type.priority() > best_match.as_ref().map_or(0, |(_, m)| m.priority())
                {
                    best_match = Some((pred_idx, match_type));
                }
            }
        }

        match best_match {
            Some((pred_idx, match_type)) => {
                matched_preds[pred_idx] = true;
                let pred_span = &predicted[pred_idx];

                // Update counts based on match type
                update_counts(&mut results, gold_span, pred_span, match_type);

                // Per-type tracking (strict mode)
                let type_counts = results.by_type.entry(entity_type_str).or_default();
                if match_type == MatchType::ExactBoth {
                    type_counts.correct += 1;
                } else {
                    type_counts.incorrect += 1;
                }
            }
            None => {
                // Missing: gold entity not found
                results.strict.missed += 1;
                results.exact.missed += 1;
                results.partial.missed += 1;
                results.ent_type.missed += 1;

                results.by_type.entry(entity_type_str).or_default().missed += 1;
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

            let entity_type_str = format!("{:?}", predicted[pred_idx].entity_type);
            results.by_type.entry(entity_type_str).or_default().spurious += 1;
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

impl MatchType {
    /// Priority for selecting best match (higher = better).
    fn priority(self) -> u8 {
        match self {
            MatchType::None => 0,
            MatchType::PartialBoundaryWrongType => 1,
            MatchType::PartialBoundaryCorrectType => 2,
            MatchType::ExactBoundaryWrongType => 3,
            MatchType::ExactBoth => 4,
        }
    }
}

/// Classify the match type between gold and predicted spans.
fn classify_match(gold: &EvalSpan, pred: &EvalSpan) -> MatchType {
    let exact_boundary = gold.exact_boundary_match(pred);
    let has_overlap = gold.has_overlap(pred);
    let type_match = gold.type_match(pred);

    if exact_boundary && type_match {
        MatchType::ExactBoth
    } else if exact_boundary && !type_match {
        MatchType::ExactBoundaryWrongType
    } else if has_overlap && type_match {
        MatchType::PartialBoundaryCorrectType
    } else if has_overlap && !type_match {
        MatchType::PartialBoundaryWrongType
    } else {
        MatchType::None
    }
}

/// Update counts based on match type.
fn update_counts(
    results: &mut NerEvalResults,
    gold: &EvalSpan,
    pred: &EvalSpan,
    match_type: MatchType,
) {
    let exact_boundary = gold.exact_boundary_match(pred);
    let type_match = gold.type_match(pred);

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
    if type_match && (exact_boundary || gold.has_overlap(pred)) {
        results.ent_type.correct += 1;
    } else {
        results.ent_type.incorrect += 1;
    }
}

/// Convenience function to evaluate Entity slices directly.
#[must_use]
pub fn evaluate_entities(gold: &[Entity], predicted: &[Entity]) -> NerEvalResults {
    let gold_spans: Vec<EvalSpan> = gold.iter().map(EvalSpan::from).collect();
    let pred_spans: Vec<EvalSpan> = predicted.iter().map(EvalSpan::from).collect();
    evaluate_ner(&gold_spans, &pred_spans)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_partial_boundary() {
        let gold = vec![span(EntityType::Person, 0, 10)];
        let pred = vec![span(EntityType::Person, 0, 8)]; // Partial overlap

        let results = evaluate_ner(&gold, &pred);
        // Strict: incorrect (boundary mismatch)
        assert_eq!(results.strict.incorrect, 1);
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

        // Check per-type breakdown
        let person_counts = results.by_type.get("Person").unwrap();
        assert_eq!(person_counts.correct, 1);
        assert_eq!(person_counts.incorrect, 1);

        let loc_counts = results.by_type.get("Location").unwrap();
        assert_eq!(loc_counts.correct, 1);
    }
}
