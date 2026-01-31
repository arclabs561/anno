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
//! | **Fleiss' Kappa** | Chance-corrected agreement for multiple annotators |
//! | **Krippendorff's Alpha** | Agreement metric that handles missing data |
//! | **Soft F1** | F1 weighted by annotator agreement |
//!
//! # Example
//!
//! ```rust
//! use anno::eval::annotator::{MultiAnnotatorCorpus, AnnotatorAnalyzer};
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
    /// Fleiss' kappa for span identification
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
    /// Types assigned by different annotators
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
            let spans = doc.unique_spans();
            let annotators: Vec<_> = doc.annotators();

            for (start, end) in spans {
                span_total += 1;

                // Collect annotations for this span
                let mut types_for_span: HashMap<String, Vec<String>> = HashMap::new();
                let mut annotators_with_span = 0;

                for annotator in &annotators {
                    if let Some(anns) = doc.annotations.get(*annotator) {
                        for ann in anns {
                            if ann.start == start && ann.end == end {
                                types_for_span
                                    .entry(ann.entity_type.clone())
                                    .or_default()
                                    .push((*annotator).to_string());
                                annotators_with_span += 1;
                            }
                        }
                    }
                }

                // Check span agreement (all annotators marked this span)
                if annotators_with_span == annotators.len() {
                    span_agree_count += 1;
                }

                // Check type agreement
                type_total += 1;
                let all_same_type = types_for_span.len() == 1;
                if all_same_type && annotators_with_span == annotators.len() {
                    type_agree_count += 1;
                }

                // Track per-type agreement
                for (etype, ann_list) in &types_for_span {
                    let entry = type_counts.entry(etype.clone()).or_insert((0, 0));
                    entry.1 += 1;
                    if ann_list.len() == annotators.len() {
                        entry.0 += 1;
                    }
                }

                // Track contentious spans
                if types_for_span.len() > 1 || annotators_with_span < annotators.len() {
                    let disagreement = 1.0
                        - (types_for_span.values().map(|v| v.len()).max().unwrap_or(0) as f64
                            / annotators.len() as f64);
                    let text = doc
                        .annotations
                        .values()
                        .flat_map(|anns| anns.iter())
                        .find(|a| a.start == start && a.end == end)
                        .map(|a| a.text.clone())
                        .unwrap_or_default();

                    contentious.push(ContentiousSpan {
                        doc_id: doc_id.clone(),
                        start,
                        end,
                        text,
                        types_assigned: types_for_span.clone(),
                        disagreement,
                    });
                }
            }
        }

        // Sort contentious by disagreement
        contentious.sort_by(|a, b| {
            b.disagreement
                .partial_cmp(&a.disagreement)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

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

    /// Compute Fleiss' kappa for inter-annotator agreement.
    fn compute_fleiss_kappa(&self) -> f64 {
        // Simplified Fleiss' kappa computation
        // Full implementation would need token-level annotation matrix

        let mut total_agreement = 0.0;
        let mut count = 0;

        for doc in self.corpus.documents.values() {
            let spans = doc.unique_spans();
            let n_annotators = doc.annotators().len();
            if n_annotators < 2 {
                continue;
            }

            for (start, end) in spans {
                let mut votes: HashMap<String, usize> = HashMap::new();
                let mut n_votes = 0;

                for anns in doc.annotations.values() {
                    for ann in anns {
                        if ann.start == start && ann.end == end {
                            *votes.entry(ann.entity_type.clone()).or_insert(0) += 1;
                            n_votes += 1;
                        }
                    }
                }

                if n_votes >= 2 {
                    // Agreement for this item
                    let sum_sq: usize = votes.values().map(|v| v * v).sum();
                    let p_i = (sum_sq - n_votes) as f64 / (n_votes * (n_votes - 1)) as f64;
                    total_agreement += p_i;
                    count += 1;
                }
            }
        }

        if count == 0 {
            return 0.0;
        }

        let p_bar = total_agreement / count as f64;
        // Simplified: assume uniform category distribution for P_e
        let p_e = 0.5; // This should be computed from category frequencies

        if (1.0_f64 - p_e).abs() < 1e-7 {
            return 1.0;
        }

        (p_bar - p_e) / (1.0 - p_e)
    }

    /// Create a "soft" gold standard by aggregating annotations.
    ///
    /// Returns annotations with confidence based on annotator agreement.
    pub fn aggregate_gold(&self, doc_id: &str) -> Vec<(Annotation, f64)> {
        let doc = match self.corpus.documents.get(doc_id) {
            Some(d) => d,
            None => return Vec::new(),
        };

        let spans = doc.unique_spans();
        let n_annotators = doc.annotators().len();
        let mut result = Vec::new();

        for (start, end) in spans {
            let mut type_votes: HashMap<String, usize> = HashMap::new();
            let mut text = String::new();
            let mut total_votes = 0;

            for anns in doc.annotations.values() {
                for ann in anns {
                    if ann.start == start && ann.end == end {
                        *type_votes.entry(ann.entity_type.clone()).or_insert(0) += 1;
                        text = ann.text.clone();
                        total_votes += 1;
                    }
                }
            }

            if total_votes > 0 {
                // Majority vote for type
                let (best_type, best_count) = type_votes
                    .iter()
                    .max_by_key(|(_, c)| *c)
                    .map(|(t, c)| (t.clone(), *c))
                    .expect("type_votes should not be empty");

                // Confidence based on agreement
                let span_confidence = total_votes as f64 / n_annotators as f64;
                let type_confidence = best_count as f64 / total_votes as f64;
                let confidence = span_confidence * type_confidence;

                result.push((
                    Annotation::new(&text, start, end, &best_type, "aggregated"),
                    confidence,
                ));
            }
        }

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
    fn test_soft_evaluation() {
        let predictions = vec![Annotation::new("Obama", 0, 5, "PER", "model")];
        let gold_with_conf = vec![(Annotation::new("Obama", 0, 5, "PER", "gold"), 0.9)];

        let evaluator = SoftEvaluator::new(0.5);
        let metrics = evaluator.evaluate(&predictions, &gold_with_conf);

        assert!(metrics.f1 > 0.8); // Should be close to 1.0 weighted by confidence
    }
}
