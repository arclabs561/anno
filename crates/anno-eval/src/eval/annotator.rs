//! Multi-annotator disagreement modeling.
//!
//! # The Ground Truth Problem
//!
//! NER annotation is inherently subjective. Different annotators disagree about:
//! - **Entity boundaries**: "New York City" vs "New York"
//! - **Entity types**: Is "Apple" a company or a product?
//! - **What counts as an entity**: Generic nouns? Metonymic references?
//!
//! Traditional evaluation assumes a single gold standard, hiding this uncertainty.
//! This module models annotator disagreement explicitly.
//!
//! # Disagreement Metrics
//!
//! | Metric | Description |
//! |--------|-------------|
//! | **Agreement Rate** | % of tokens/spans all annotators agree on |
//! | **Fleiss' Kappa** | Chance-corrected agreement on type, over spans that two or more annotators marked |
//! | **Soft F1** | F1 weighted by annotator agreement |
//!
//! # Example
//!
//! ```rust
//! use anno_eval::eval::annotator::{MultiAnnotatorCorpus, AnnotatorAnalyzer};
//!
//! let mut corpus = MultiAnnotatorCorpus::new();
//!
//! // Add annotations from multiple annotators
//! corpus.add_annotation("doc1", "annotator_A", vec![
//!     ("Barack Obama", 0, 12, "PER"),
//!     ("Hawaii", 25, 31, "LOC"),
//! ]);
//! corpus.add_annotation("doc1", "annotator_B", vec![
//!     ("Barack Obama", 0, 12, "PER"),
//!     ("Hawaii", 25, 31, "GPE"),  // Different type!
//! ]);
//!
//! let analyzer = AnnotatorAnalyzer::new(&corpus);
//! let stats = analyzer.compute_agreement();
//! println!("Span agreement: {:.2}%", stats.span_agreement * 100.0);
//! println!("Type agreement: {:.2}%", stats.type_agreement * 100.0);
//! ```
//!
//! # Research Background
//!
//! - **Hirschman et al. (1998)**: "Automating Coreference" [cmp-lg/9803001]
//!   - Found only 16% of interannotator disagreements were genuine coreference disagreement
//!   - 84% were systematic errors (missed pronouns, overlooked chains, zone issues)
//!   - Two-stage annotation (markables first, linking second) can improve agreement
//! - Plank et al. (2014): "Learning part-of-speech taggers with inter-annotator agreement loss"
//! - Pavlick & Kwiatkowski (2019): "Inherent Disagreements in Human Textual Inferences"
//! - Uma et al. (2021): "Learning from Disagreement: A Survey"

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// =============================================================================
// Core Types
// =============================================================================

/// A single annotation from one annotator.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Annotation {
    /// Entity text
    pub text: String,
    /// Start character offset
    pub start: usize,
    /// End character offset
    pub end: usize,
    /// Entity type
    pub entity_type: String,
    /// Annotator ID
    pub annotator: String,
}

impl Annotation {
    /// Create a new annotation.
    pub fn new(text: &str, start: usize, end: usize, entity_type: &str, annotator: &str) -> Self {
        Self {
            text: text.to_string(),
            start,
            end,
            entity_type: entity_type.to_string(),
            annotator: annotator.to_string(),
        }
    }

    /// Check if this annotation overlaps with another.
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// Check if spans are identical (ignoring type).
    pub fn same_span(&self, other: &Self) -> bool {
        self.start == other.start && self.end == other.end
    }
}

/// Document with annotations from multiple annotators.
#[derive(Debug, Clone, Default)]
pub struct AnnotatedDocument {
    /// Document ID
    pub doc_id: String,
    /// Document text
    pub text: String,
    /// Annotations grouped by annotator
    pub annotations: HashMap<String, Vec<Annotation>>,
}

impl AnnotatedDocument {
    /// Create a new annotated document.
    pub fn new(doc_id: &str, text: &str) -> Self {
        Self {
            doc_id: doc_id.to_string(),
            text: text.to_string(),
            annotations: HashMap::new(),
        }
    }

    /// Add annotations from an annotator.
    pub fn add_annotator(&mut self, annotator: &str, annotations: Vec<Annotation>) {
        self.annotations.insert(annotator.to_string(), annotations);
    }

    /// Get all unique annotators.
    pub fn annotators(&self) -> Vec<&str> {
        self.annotations.keys().map(|s| s.as_str()).collect()
    }

    /// Get all unique spans across all annotators.
    pub fn unique_spans(&self) -> HashSet<(usize, usize)> {
        self.annotations
            .values()
            .flat_map(|anns| anns.iter().map(|a| (a.start, a.end)))
            .collect()
    }
}

// The document's grouping key identifies the rater, including for imported annotations.
struct SpanRatings<'a> {
    text: &'a str,
    labels: HashMap<&'a str, HashSet<&'a str>>,
}

impl<'a> SpanRatings<'a> {
    fn type_counts(&self) -> HashMap<&'a str, usize> {
        let mut counts = HashMap::new();
        for &label in self
            .labels
            .values()
            .filter(|labels| labels.len() == 1)
            .flatten()
        {
            *counts.entry(label).or_default() += 1;
        }
        counts
    }
}

impl AnnotatedDocument {
    fn span_ratings(&self) -> HashMap<(usize, usize), SpanRatings<'_>> {
        let mut spans = HashMap::new();
        for (rater, annotations) in &self.annotations {
            for ann in annotations {
                let ratings = spans
                    .entry((ann.start, ann.end))
                    .or_insert_with(|| SpanRatings {
                        text: &ann.text,
                        labels: HashMap::new(),
                    });
                ratings
                    .labels
                    .entry(rater.as_str())
                    .or_default()
                    .insert(&ann.entity_type);
            }
        }
        spans
    }
}

/// Corpus with multi-annotator annotations.
#[derive(Debug, Clone, Default)]
pub struct MultiAnnotatorCorpus {
    /// Documents with annotations
    pub documents: HashMap<String, AnnotatedDocument>,
    /// All annotator IDs
    pub annotators: HashSet<String>,
}

impl MultiAnnotatorCorpus {
    /// Create a new corpus.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an annotation.
    pub fn add_annotation(
        &mut self,
        doc_id: &str,
        annotator: &str,
        annotations: Vec<(&str, usize, usize, &str)>,
    ) {
        self.annotators.insert(annotator.to_string());

        let doc = self
            .documents
            .entry(doc_id.to_string())
            .or_insert_with(|| AnnotatedDocument::new(doc_id, ""));

        let anns: Vec<Annotation> = annotations
            .into_iter()
            .map(|(text, start, end, etype)| Annotation::new(text, start, end, etype, annotator))
            .collect();

        doc.add_annotator(annotator, anns);
    }

    /// Get number of documents.
    pub fn num_documents(&self) -> usize {
        self.documents.len()
    }

    /// Get number of annotators.
    pub fn num_annotators(&self) -> usize {
        self.annotators.len()
    }
}

// =============================================================================
// Agreement Metrics
// =============================================================================

/// Agreement statistics across annotators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgreementStats {
    /// Proportion of spans where all annotators agree on existence
    pub span_agreement: f64,
    /// Proportion of spans where all annotators agree on type
    pub type_agreement: f64,
    /// Chance-corrected type agreement on spans rated by at least two annotators.
    pub fleiss_kappa: f64,
    /// Per-type agreement rates
    pub type_specific_agreement: HashMap<String, f64>,
    /// Most disagreed spans (for error analysis)
    pub contentious_spans: Vec<ContentiousSpan>,
    /// Number of annotators
    pub num_annotators: usize,
    /// Number of documents
    pub num_documents: usize,
}

/// A span with significant annotator disagreement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentiousSpan {
    /// Document ID
    pub doc_id: String,
    /// Start offset
    pub start: usize,
    /// End offset
    pub end: usize,
    /// Text
    pub text: String,
    /// All reported types, with each annotator listed once for each distinct label.
    pub types_assigned: HashMap<String, Vec<String>>, // type -> [annotators]
    /// Disagreement score (0 = full agreement, 1 = no agreement)
    pub disagreement: f64,
}

/// Analyzer for multi-annotator agreement.
pub struct AnnotatorAnalyzer<'a> {
    corpus: &'a MultiAnnotatorCorpus,
}

impl<'a> AnnotatorAnalyzer<'a> {
    /// Create a new analyzer.
    pub fn new(corpus: &'a MultiAnnotatorCorpus) -> Self {
        Self { corpus }
    }

    /// Compute agreement statistics.
    pub fn compute_agreement(&self) -> AgreementStats {
        let mut span_agree_count = 0;
        let mut span_total = 0;
        let mut type_agree_count = 0;
        let mut type_total = 0;
        let mut type_counts: HashMap<String, (usize, usize)> = HashMap::new(); // (agree, total)
        let mut contentious: Vec<ContentiousSpan> = Vec::new();

        for (doc_id, doc) in &self.corpus.documents {
            let n_annotators = doc.annotations.len();
            for ((start, end), ratings) in doc.span_ratings() {
                span_total += 1;
                if ratings.labels.len() == n_annotators {
                    span_agree_count += 1;
                }

                let votes = ratings.type_counts();
                let best_count = votes.values().max().copied().unwrap_or(0);
                let unanimous = best_count == n_annotators;
                type_total += 1;
                if unanimous {
                    type_agree_count += 1;
                }

                // Each reported type has one denominator entry for this span, even
                // if it was reported only by a rater with conflicting labels.
                let mut types_for_span: HashMap<String, Vec<String>> = HashMap::new();
                for (rater, labels) in &ratings.labels {
                    for &label in labels {
                        types_for_span
                            .entry(label.to_owned())
                            .or_default()
                            .push((*rater).to_owned());
                    }
                }
                for label in types_for_span.keys() {
                    let entry = type_counts.entry(label.clone()).or_default();
                    entry.1 += 1;
                    if unanimous {
                        entry.0 += 1;
                    }
                }

                if !unanimous {
                    let disagreement = 1.0 - best_count as f64 / n_annotators as f64;
                    contentious.push(ContentiousSpan {
                        doc_id: doc_id.clone(),
                        start,
                        end,
                        text: ratings.text.to_owned(),
                        types_assigned: types_for_span,
                        disagreement,
                    });
                }
            }
        }

        // Sort contentious by disagreement
        contentious.sort_by(|a, b| b.disagreement.total_cmp(&a.disagreement));

        // Compute Fleiss' kappa
        let fleiss_kappa = self.compute_fleiss_kappa();

        // Per-type agreement
        let type_specific_agreement: HashMap<String, f64> = type_counts
            .into_iter()
            .map(|(t, (agree, total))| {
                (
                    t,
                    if total > 0 {
                        agree as f64 / total as f64
                    } else {
                        0.0
                    },
                )
            })
            .collect();

        AgreementStats {
            span_agreement: if span_total > 0 {
                span_agree_count as f64 / span_total as f64
            } else {
                0.0
            },
            type_agreement: if type_total > 0 {
                type_agree_count as f64 / type_total as f64
            } else {
                0.0
            },
            fleiss_kappa,
            type_specific_agreement,
            contentious_spans: contentious.into_iter().take(20).collect(), // Top 20
            num_annotators: self.corpus.num_annotators(),
            num_documents: self.corpus.num_documents(),
        }
    }

    /// Compute Fleiss' kappa over the spans that two or more annotators marked.
    ///
    /// An item is a span; its votes are the types the annotators gave it.
    /// Chance agreement is the sum of squared category proportions pooled
    /// over those items, which is Fleiss' formula when every item has the
    /// same number of votes. With unequal counts each span contributes equally
    /// to observed agreement, while each vote contributes equally to chance agreement.
    /// Each annotator supplies at most one type for a span. Conflicting labels
    /// from one annotator leave that vote unspecified.
    /// Spans marked by one annotator carry no agreement information and are
    /// left out, so this measures agreement on type, not on span boundaries.
    fn compute_fleiss_kappa(&self) -> f64 {
        let mut total_agreement = 0.0;
        let mut count = 0;
        let mut category_votes: HashMap<&str, usize> = HashMap::new();
        let mut all_votes = 0usize;

        for doc in self.corpus.documents.values() {
            for ratings in doc.span_ratings().values() {
                let counts = ratings.type_counts();
                let n: usize = counts.values().sum();
                if n < 2 {
                    continue;
                }
                for (&label, &votes) in &counts {
                    *category_votes.entry(label).or_insert(0) += votes;
                }
                let agreeing_pairs: f64 = counts.values().map(|&v| v as f64 * (v - 1) as f64).sum();
                total_agreement += agreeing_pairs / (n as f64 * (n - 1) as f64);
                count += 1;
                all_votes += n;
            }
        }

        if count == 0 {
            return 0.0;
        }

        let p_bar = total_agreement / count as f64;
        let p_e: f64 = category_votes
            .values()
            .map(|&v| {
                let p = v as f64 / all_votes as f64;
                p * p
            })
            .sum();

        if category_votes.len() == 1 {
            // Every vote is in one category: agreement is complete, and
            // kappa is undefined; report it as full agreement.
            return 1.0;
        }

        (p_bar - p_e) / (1.0 - p_e)
    }

    /// Create a "soft" gold standard by aggregating annotations.
    ///
    /// Confidence is the number of distinct raters voting for the selected type
    /// divided by the number of raters on the document. A rater who gives a span
    /// conflicting types supplies no type vote. A span with no type votes is omitted.
    /// Equal vote counts select the lexicographically first label. Results are in span order.
    pub fn aggregate_gold(&self, doc_id: &str) -> Vec<(Annotation, f64)> {
        let doc = match self.corpus.documents.get(doc_id) {
            Some(d) => d,
            None => return Vec::new(),
        };

        let n_annotators = doc.annotations.len();
        let mut result = Vec::new();
        for ((start, end), ratings) in doc.span_ratings() {
            let votes = ratings.type_counts();
            if let Some((best_type, best_count)) = votes
                .into_iter()
                .max_by_key(|&(label, n)| (n, std::cmp::Reverse(label)))
            {
                result.push((
                    Annotation::new(ratings.text, start, end, best_type, "aggregated"),
                    best_count as f64 / n_annotators as f64,
                ));
            }
        }

        result.sort_unstable_by_key(|(ann, _)| (ann.start, ann.end));
        result
    }
}

// =============================================================================
// Soft Evaluation
// =============================================================================

/// Compute soft F1 that accounts for annotator disagreement.
///
/// Traditional F1 treats all gold annotations as equally certain.
/// Soft F1 weights matches by annotator agreement.
#[derive(Debug, Clone, Default)]
pub struct SoftEvaluator {
    /// Minimum agreement threshold to count as gold
    pub min_agreement: f64,
}

impl SoftEvaluator {
    /// Create a new soft evaluator.
    pub fn new(min_agreement: f64) -> Self {
        Self { min_agreement }
    }

    /// Compute soft precision/recall/F1.
    pub fn evaluate(
        &self,
        predictions: &[Annotation],
        gold_with_confidence: &[(Annotation, f64)],
    ) -> SoftMetrics {
        let mut weighted_tp = 0.0;
        let mut weighted_fp = 0.0;
        let mut weighted_fn = 0.0;

        // Filter gold by minimum agreement
        let gold: Vec<_> = gold_with_confidence
            .iter()
            .filter(|(_, conf)| *conf >= self.min_agreement)
            .collect();

        // For each prediction, find best matching gold
        let mut matched_gold: HashSet<usize> = HashSet::new();

        for pred in predictions {
            let mut best_match: Option<(usize, f64)> = None;

            for (i, (g, conf)) in gold.iter().enumerate() {
                let should_update = match best_match {
                    None => true,
                    Some((_, best_conf)) => *conf > best_conf,
                };
                if pred.same_span(g)
                    && pred.entity_type == g.entity_type
                    && !matched_gold.contains(&i)
                    && should_update
                {
                    best_match = Some((i, *conf));
                }
            }

            if let Some((idx, conf)) = best_match {
                weighted_tp += conf;
                matched_gold.insert(idx);
            } else {
                weighted_fp += 1.0;
            }
        }

        // Unmatched gold are false negatives
        for (i, (_, conf)) in gold.iter().enumerate() {
            if !matched_gold.contains(&i) {
                weighted_fn += *conf;
            }
        }

        let precision = if weighted_tp + weighted_fp > 0.0 {
            weighted_tp / (weighted_tp + weighted_fp)
        } else {
            0.0
        };

        let recall = if weighted_tp + weighted_fn > 0.0 {
            weighted_tp / (weighted_tp + weighted_fn)
        } else {
            0.0
        };

        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        SoftMetrics {
            precision,
            recall,
            f1,
            weighted_tp,
            weighted_fp,
            weighted_fn,
        }
    }
}

/// Soft evaluation metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftMetrics {
    /// Weighted precision
    pub precision: f64,
    /// Weighted recall
    pub recall: f64,
    /// Weighted F1
    pub f1: f64,
    /// Weighted true positives
    pub weighted_tp: f64,
    /// Weighted false positives
    pub weighted_fp: f64,
    /// Weighted false negatives
    pub weighted_fn: f64,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_annotation_overlap() {
        let a1 = Annotation::new("Barack Obama", 0, 12, "PER", "A");
        let a2 = Annotation::new("Obama", 7, 12, "PER", "B");
        let a3 = Annotation::new("Hawaii", 20, 26, "LOC", "A");

        assert!(a1.overlaps(&a2));
        assert!(!a1.overlaps(&a3));
    }

    #[test]
    fn test_multi_annotator_corpus() {
        let mut corpus = MultiAnnotatorCorpus::new();

        corpus.add_annotation(
            "doc1",
            "annotator_A",
            vec![("Barack Obama", 0, 12, "PER"), ("Hawaii", 25, 31, "LOC")],
        );
        corpus.add_annotation(
            "doc1",
            "annotator_B",
            vec![
                ("Barack Obama", 0, 12, "PER"),
                ("Hawaii", 25, 31, "GPE"), // Different type
            ],
        );

        assert_eq!(corpus.num_annotators(), 2);
        assert_eq!(corpus.num_documents(), 1);
    }

    #[test]
    fn test_agreement_computation() {
        let mut corpus = MultiAnnotatorCorpus::new();

        corpus.add_annotation(
            "doc1",
            "A",
            vec![("Obama", 0, 5, "PER"), ("Hawaii", 10, 16, "LOC")],
        );
        corpus.add_annotation(
            "doc1",
            "B",
            vec![("Obama", 0, 5, "PER"), ("Hawaii", 10, 16, "LOC")],
        );

        let analyzer = AnnotatorAnalyzer::new(&corpus);
        let stats = analyzer.compute_agreement();

        // Full agreement
        assert_eq!(stats.span_agreement, 1.0);
        assert_eq!(stats.type_agreement, 1.0);
    }

    #[test]
    fn test_contentious_spans() {
        let mut corpus = MultiAnnotatorCorpus::new();

        corpus.add_annotation("doc1", "A", vec![("Apple", 0, 5, "ORG")]);
        corpus.add_annotation("doc1", "B", vec![("Apple", 0, 5, "PRODUCT")]);
        corpus.add_annotation("doc1", "C", vec![("Apple", 0, 5, "ORG")]);

        let analyzer = AnnotatorAnalyzer::new(&corpus);
        let stats = analyzer.compute_agreement();

        // Should have contentious span for Apple
        assert!(!stats.contentious_spans.is_empty());
        assert_eq!(stats.contentious_spans[0].text, "Apple");
    }

    #[test]
    fn test_aggregate_gold() {
        let mut corpus = MultiAnnotatorCorpus::new();

        corpus.add_annotation("doc1", "A", vec![("Apple", 0, 5, "ORG")]);
        corpus.add_annotation("doc1", "B", vec![("Apple", 0, 5, "ORG")]);
        corpus.add_annotation("doc1", "C", vec![("Apple", 0, 5, "PRODUCT")]);

        let analyzer = AnnotatorAnalyzer::new(&corpus);
        let gold = analyzer.aggregate_gold("doc1");

        assert_eq!(gold.len(), 1);
        assert_eq!(gold[0].0.entity_type, "ORG"); // Majority vote
        assert!(gold[0].1 > 0.5); // > 50% agreement
    }

    #[test]
    fn tied_votes_have_a_stable_label_and_span_order() {
        for labels in [["PER", "ORG"], ["ORG", "PER"]] {
            let mut corpus = MultiAnnotatorCorpus::new();
            for (rater, label) in ["A", "B"].into_iter().zip(labels) {
                corpus.add_annotation("doc", rater, vec![("b", 2, 3, label), ("a", 0, 1, label)]);
            }
            let analyzer = AnnotatorAnalyzer::new(&corpus);
            for _ in 0..8 {
                let gold = analyzer.aggregate_gold("doc");
                let summary: Vec<_> = gold
                    .iter()
                    .map(|(a, conf)| (a.start, a.entity_type.as_str(), *conf))
                    .collect();
                assert_eq!(summary, [(0, "ORG", 0.5), (2, "ORG", 0.5)]);
            }
        }
    }

    #[test]
    fn repeated_labels_do_not_replace_a_missing_rater() {
        let mut corpus = MultiAnnotatorCorpus::new();
        corpus.add_annotation("doc", "B", vec![]);
        for copies in [1, 2, 3] {
            corpus.add_annotation("doc", "A", vec![("x", 0, 1, "PER"); copies]);
            let analyzer = AnnotatorAnalyzer::new(&corpus);
            let stats = analyzer.compute_agreement();
            assert_eq!((stats.span_agreement, stats.type_agreement), (0.0, 0.0));
            assert_eq!(stats.type_specific_agreement["PER"], 0.0);
            let span = &stats.contentious_spans[0];
            assert_eq!(span.disagreement, 0.5);
            assert_eq!(span.types_assigned["PER"], ["A"]);
            let gold = analyzer.aggregate_gold("doc");
            assert_eq!(gold.len(), 1);
            assert_eq!(gold[0].1, 0.5);
        }
        corpus.add_annotation("doc", "B", vec![("x", 0, 1, "PER"); 2]);
        let analyzer = AnnotatorAnalyzer::new(&corpus);
        let stats = analyzer.compute_agreement();
        assert_eq!((stats.span_agreement, stats.type_agreement), (1.0, 1.0));
        assert!(stats.contentious_spans.is_empty());
        assert_eq!(stats.type_specific_agreement["PER"], 1.0);
        assert_eq!(analyzer.aggregate_gold("doc")[0].1, 1.0);
    }

    #[test]
    fn conflicting_labels_mark_presence_and_remain_in_diagnostics() {
        let mut corpus = MultiAnnotatorCorpus::new();
        corpus.add_annotation("doc", "B", vec![("x", 0, 1, "LOC"); 2]);
        for labels in [["PER", "LOC", "PER"], ["LOC", "PER", "PER"]] {
            corpus.add_annotation(
                "doc",
                "A",
                labels.iter().map(|label| ("x", 0, 1, *label)).collect(),
            );
            let analyzer = AnnotatorAnalyzer::new(&corpus);
            let stats = analyzer.compute_agreement();
            assert_eq!((stats.span_agreement, stats.type_agreement), (1.0, 0.0));
            assert_eq!(stats.type_specific_agreement["PER"], 0.0);
            assert_eq!(stats.type_specific_agreement["LOC"], 0.0);
            let span = &stats.contentious_spans[0];
            assert_eq!(span.disagreement, 0.5);
            assert_eq!(span.types_assigned["PER"], ["A"]);
            let mut loc = span.types_assigned["LOC"].clone();
            loc.sort();
            assert_eq!(loc, ["A", "B"]);
            let gold = analyzer.aggregate_gold("doc");
            assert_eq!(gold.len(), 1);
            assert_eq!(gold[0].0.entity_type, "LOC");
            assert_eq!(gold[0].1, 0.5);
        }
        corpus.add_annotation("doc", "B", vec![("x", 0, 1, "LOC"), ("x", 0, 1, "ORG")]);
        let analyzer = AnnotatorAnalyzer::new(&corpus);
        assert!(analyzer.aggregate_gold("doc").is_empty());
        assert_eq!(
            analyzer.compute_agreement().contentious_spans[0].disagreement,
            1.0
        );
    }

    #[test]
    fn test_soft_evaluation() {
        let predictions = vec![Annotation::new("Obama", 0, 5, "PER", "model")];
        let gold_with_conf = vec![(Annotation::new("Obama", 0, 5, "PER", "gold"), 0.9)];

        let evaluator = SoftEvaluator::new(0.5);
        let metrics = evaluator.evaluate(&predictions, &gold_with_conf);

        assert!(metrics.f1 > 0.8); // Should be close to 1.0 weighted by confidence
    }

    fn corpus_of(a: &[&'static str], b: &[&'static str]) -> MultiAnnotatorCorpus {
        let mut corpus = MultiAnnotatorCorpus::new();
        let spans = |types: &[&'static str]| -> Vec<(&'static str, usize, usize, &'static str)> {
            types
                .iter()
                .enumerate()
                .map(|(k, t)| ("x", 10 * k, 10 * k + 1, *t))
                .collect()
        };
        corpus.add_annotation("doc", "A", spans(a));
        corpus.add_annotation("doc", "B", spans(b));
        corpus
    }

    #[test]
    fn fleiss_kappa_uses_category_frequencies_for_chance_agreement() {
        // Observed agreement 0.75; PER has 5 of 8 votes and LOC 3, so chance
        // agreement is 34/64 and kappa is (0.75 - 0.53125) / 0.46875.
        let corpus = corpus_of(&["PER", "PER", "PER", "LOC"], &["PER", "PER", "LOC", "LOC"]);
        let kappa = AnnotatorAnalyzer::new(&corpus)
            .compute_agreement()
            .fleiss_kappa;
        assert!((kappa - 0.466_666_7).abs() < 1e-6, "kappa = {kappa}");
    }

    #[test]
    fn fleiss_kappa_is_near_zero_when_one_category_dominates() {
        // Nine spans agreed as PER and one split: 90% observed agreement, but
        // 90.5% is expected by chance. A fixed chance of 0.5 reported 0.8.
        let mut a = vec!["PER"; 10];
        let b = vec!["PER"; 10];
        a[9] = "LOC";
        let corpus = corpus_of(&a, &b);
        let kappa = AnnotatorAnalyzer::new(&corpus)
            .compute_agreement()
            .fleiss_kappa;
        assert!(
            (kappa - (0.9 - 0.905) / 0.095).abs() < 1e-9,
            "kappa = {kappa}"
        );
    }
    #[test]
    fn fleiss_kappa_counts_each_annotator_once_per_span() {
        let mut corpus = corpus_of(&["PER", "LOC"], &["PER", "PER"]);
        let expected = AnnotatorAnalyzer::new(&corpus).compute_fleiss_kappa();
        let a = corpus
            .documents
            .get_mut("doc")
            .unwrap()
            .annotations
            .get_mut("A")
            .unwrap();
        a.push(a[1].clone());
        // A span seen by A alone supplies no pair, even when A marks it twice.
        a.extend(vec![Annotation::new("y", 30, 31, "ORG", "A"); 2]);
        let actual = AnnotatorAnalyzer::new(&corpus).compute_fleiss_kappa();
        assert!((expected - actual).abs() < 1e-10, "{actual} vs {expected}");
    }

    #[test]
    fn fleiss_kappa_omits_a_raters_conflicting_labels_regardless_of_order() {
        let mut corpus = corpus_of(&["PER", "LOC"], &["PER", "PER"]);
        let expected = AnnotatorAnalyzer::new(&corpus).compute_fleiss_kappa();
        for labels in [["PER", "LOC"], ["LOC", "PER"]] {
            corpus.add_annotation(
                "doc",
                "C",
                vec![("x", 0, 1, labels[0]), ("x", 0, 1, labels[1])],
            );
            let actual = AnnotatorAnalyzer::new(&corpus).compute_fleiss_kappa();
            assert!((expected - actual).abs() < 1e-10, "{actual} vs {expected}");
        }
    }
}
