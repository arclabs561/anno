//! Advanced evaluation metrics for NER.
//!
//! Provides additional metrics beyond basic Precision/Recall/F1:
//! - Partial match metrics (overlap-based)
//! - Confidence threshold analysis
//! - Per-language metrics (for multilingual datasets)
//! - Cross-dataset comparison utilities

use super::datasets::GoldEntity;
use anno::Entity;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

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

/// Compute duplication/noise metrics with duplicate detection scoped to each document.
///
/// Repeated entity surface forms in separate documents are legitimate mentions, so they
/// are not counted as duplicates. A duplicate is the same normalized surface, type, and
/// exact span repeated within one document.
#[must_use]
pub(crate) fn compute_document_extraction_quality_metrics<'a>(
    documents: impl IntoIterator<Item = &'a [Entity]>,
) -> ExtractionQualityMetrics {
    let mut metrics = ExtractionQualityMetrics::default();

    for entities in documents {
        let mut seen: HashSet<(anno::EntityType, usize, usize, String)> = HashSet::new();
        for entity in entities {
            metrics.total += 1;
            if is_noisy_span(&entity.text) {
                metrics.noisy += 1;
            }

            let key = (
                entity.entity_type.clone(),
                entity.start(),
                entity.end(),
                normalize_for_duplication(&entity.text),
            );
            if !seen.insert(key) {
                metrics.duplicates += 1;
            }
        }
    }

    if metrics.total > 0 {
        metrics.duplication_rate = metrics.duplicates as f64 / metrics.total as f64;
        metrics.noise_rate = metrics.noisy as f64 / metrics.total as f64;
    }

    metrics
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

// =============================================================================
// Character-Level PII Masking Metrics
// =============================================================================

/// Character-level counts for PII masking evaluation.
///
/// Every field counts Unicode scalar values (characters), not bytes. `anno::Entity`
/// spans are character offsets, so no byte conversion is needed.
///
/// The counts are additive across documents. Metrics derived from them (precision,
/// recall, F1, F2, FPR) are pooled micro metrics, not averages of per-document scores.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CharMaskingCounts {
    /// Characters masked by the prediction that are also gold PII (`TP`).
    pub true_positive_chars: usize,
    /// All characters covered by predicted spans (`TP + FP`).
    pub predicted_chars: usize,
    /// All characters covered by gold PII spans (`TP + FN`).
    pub gold_chars: usize,
    /// Characters in the source text that are not covered by any gold PII span.
    pub non_pii_chars: usize,
    /// Total characters in the source text.
    pub total_chars: usize,
}

impl CharMaskingCounts {
    /// Character-level precision: `TP / predicted`.
    ///
    /// Returns `0.0` when nothing was masked.
    #[must_use]
    pub fn precision(&self) -> f64 {
        ratio(self.true_positive_chars, self.predicted_chars)
    }

    /// Character-level recall: `TP / gold`.
    ///
    /// Returns `0.0` when the document contains no gold PII.
    #[must_use]
    pub fn recall(&self) -> f64 {
        ratio(self.true_positive_chars, self.gold_chars)
    }

    /// Character-level F1 (harmonic mean of precision and recall).
    #[must_use]
    pub fn f1(&self) -> f64 {
        f_beta(self.precision(), self.recall(), 1.0)
    }

    /// Character-level F2 (recall-weighted; `beta = 2`).
    ///
    /// `F2 = 5PR / (4P + R)`. Missing PII characters lowers F2 more than it lowers
    /// F1, which is the intended bias for redaction.
    #[must_use]
    pub fn f2(&self) -> f64 {
        f_beta(self.precision(), self.recall(), 2.0)
    }

    /// False-positive rate: over-masked non-PII characters divided by all non-PII characters.
    ///
    /// `FPR = (predicted - TP) / non_pii`. Returns `0.0` when the document has no
    /// non-PII characters (the rate is undefined, but zero is the safe reporting value).
    #[must_use]
    pub fn false_positive_rate(&self) -> f64 {
        ratio(
            self.predicted_chars
                .saturating_sub(self.true_positive_chars),
            self.non_pii_chars,
        )
    }

    /// Merge counts from another document or aggregate.
    pub fn merge(&mut self, other: &Self) {
        self.true_positive_chars += other.true_positive_chars;
        self.predicted_chars += other.predicted_chars;
        self.gold_chars += other.gold_chars;
        self.non_pii_chars += other.non_pii_chars;
        self.total_chars += other.total_chars;
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn f_beta(precision: f64, recall: f64, beta: f64) -> f64 {
    let beta_squared = beta * beta;
    let denominator = beta_squared * precision + recall;
    if denominator == 0.0 {
        0.0
    } else {
        (1.0 + beta_squared) * precision * recall / denominator
    }
}

/// Character-level metrics for one entity type.
///
/// Per-type rows attribute each predicted span to the gold type with the largest
/// character overlap (ties broken by the type label). Predicted characters that
/// overlap no gold span are unattributed and appear only in the overall FPR.
/// Rows are diagnostic and not additive: one predicted span can overlap several types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharMaskingTypeMetrics {
    /// Entity type label (for example `PER` or `DATE`).
    pub entity_type: String,
    /// Correctly masked characters for this type.
    pub true_positive_chars: usize,
    /// Predicted characters attributed to this type.
    pub predicted_chars: usize,
    /// Gold characters of this type.
    pub gold_chars: usize,
    /// Character-level precision for this type.
    pub precision: f64,
    /// Character-level recall for this type.
    pub recall: f64,
    /// Character-level F1 for this type.
    pub f1: f64,
    /// Character-level F2 for this type.
    pub f2: f64,
    /// Over-masked characters attributed to this type divided by all non-PII characters.
    ///
    /// The denominator is the document-level non-PII character count, so this is a
    /// diagnostic slice rather than a type-local rate.
    pub false_positive_rate: f64,
}

/// Optional bootstrap confidence intervals for character masking metrics.
///
/// Each tuple is the equal-tailed 95% percentile interval `(lower, upper)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharMaskingCi95 {
    /// Precision interval.
    pub precision: (f64, f64),
    /// Recall interval.
    pub recall: (f64, f64),
    /// F1 interval.
    pub f1: (f64, f64),
    /// F2 interval.
    pub f2: (f64, f64),
    /// False-positive-rate interval.
    pub false_positive_rate: (f64, f64),
}

/// Character-level evaluation of PII masking.
///
/// # Overlap resolution
///
/// Predicted spans are converted to the **union of masked characters**: overlapping
/// and adjacent spans are merged before counting, so duplicate or nested predictions
/// never inflate precision or recall. Gold spans are unioned the same way.
///
/// # Why character-level
///
/// Entity/span F1 treats a one-character boundary error the same as a completely missed
/// identifier. For redaction the two are not equivalent: an unmasked character is a
/// privacy failure, while a small over-mask is usually tolerable. This metric therefore
/// reports character precision, recall, F1 and a recall-weighted F2, plus a
/// false-positive rate over the document's non-PII characters.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CharMaskingMetrics {
    /// Pooled character counts.
    pub counts: CharMaskingCounts,
    /// Character-level precision.
    pub precision: f64,
    /// Character-level recall.
    pub recall: f64,
    /// Character-level F1.
    pub f1: f64,
    /// Character-level F2 (recall-weighted).
    pub f2: f64,
    /// False-positive rate over non-PII characters.
    pub false_positive_rate: f64,
    /// Exact-boundary span F1 (SemEval `exact` schema; entity type is ignored).
    ///
    /// Reported alongside the character metrics so a reader can see how a span-level
    /// scorer would judge the same predictions.
    pub strict_span_f1: f64,
    /// Number of raw gold spans before overlap resolution.
    pub gold_span_count: usize,
    /// Number of raw predicted spans before overlap resolution.
    pub predicted_span_count: usize,
    /// Per-entity-type diagnostic breakdown.
    pub by_entity_type: HashMap<String, CharMaskingTypeMetrics>,
    /// Optional deterministic bootstrap confidence intervals.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ci_95: Option<CharMaskingCi95>,
}

impl CharMaskingMetrics {
    fn from_counts(counts: CharMaskingCounts) -> Self {
        Self {
            precision: counts.precision(),
            recall: counts.recall(),
            f1: counts.f1(),
            f2: counts.f2(),
            false_positive_rate: counts.false_positive_rate(),
            counts,
            ..Self::default()
        }
    }
}

/// One document for character masking evaluation.
///
/// `predicted` holds the spans the redactor would mask. It can come from any backend;
/// the metric only reads character offsets and entity types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharMaskingDocument {
    /// Source text. Entity offsets index this text's characters.
    pub text: String,
    /// Gold PII spans.
    pub gold: Vec<Entity>,
    /// Predicted mask spans.
    pub predicted: Vec<Entity>,
}

/// Evaluate character-level PII masking for one document.
///
/// # Arguments
///
/// * `text` - Source text; offsets are character offsets into this string.
/// * `gold` - Gold PII entities.
/// * `predicted` - Predicted mask spans (any entity types).
///
/// # Overlap resolution
///
/// Both gold and predicted spans are unioned into character sets. Overlapping
/// predictions therefore mask the union of their characters and are not double-counted.
#[must_use]
pub fn evaluate_char_masking(
    text: &str,
    gold: &[Entity],
    predicted: &[Entity],
) -> CharMaskingMetrics {
    let char_count = text.chars().count();
    let gold_ranges = union_char_ranges(
        gold.iter()
            .filter_map(|entity| entity_char_range(entity, char_count)),
        char_count,
    );
    let predicted_ranges = union_char_ranges(
        predicted
            .iter()
            .filter_map(|entity| entity_char_range(entity, char_count)),
        char_count,
    );

    let gold_chars = covered_char_count(&gold_ranges);
    let predicted_chars = covered_char_count(&predicted_ranges);
    let true_positive_chars = intersection_char_count(&gold_ranges, &predicted_ranges);
    let counts = CharMaskingCounts {
        true_positive_chars,
        predicted_chars,
        gold_chars,
        non_pii_chars: char_count.saturating_sub(gold_chars),
        total_chars: char_count,
    };

    let mut metrics = CharMaskingMetrics::from_counts(counts);
    metrics.strict_span_f1 = super::ner_metrics::evaluate_entities(gold, predicted)
        .exact
        .f1_exact();
    metrics.gold_span_count = gold.len();
    metrics.predicted_span_count = predicted.len();
    metrics.by_entity_type =
        per_type_char_masking(char_count, gold, predicted, metrics.counts.non_pii_chars);
    metrics
}

/// Evaluate character-level PII masking over a corpus.
///
/// Counts are pooled across documents before metrics are computed. When `bootstrap`
/// is true, deterministic document-level bootstrap confidence intervals are added;
/// the helper reuses the crate's existing seeded bootstrap index and percentile
/// interval functions.
#[must_use]
pub fn evaluate_char_masking_documents(
    documents: &[CharMaskingDocument],
    bootstrap: bool,
) -> CharMaskingMetrics {
    let mut totals = CharMaskingCounts::default();
    let mut strict_counts = super::ner_metrics::NerEvalResults::new();
    let mut per_type: HashMap<String, (usize, usize, usize)> = HashMap::new();
    let mut document_counts = Vec::with_capacity(documents.len());

    for document in documents {
        let metrics = evaluate_char_masking(&document.text, &document.gold, &document.predicted);
        totals.merge(&metrics.counts);
        strict_counts.merge(&super::ner_metrics::evaluate_entities(
            &document.gold,
            &document.predicted,
        ));
        for (label, type_metrics) in &metrics.by_entity_type {
            let entry = per_type.entry(label.clone()).or_default();
            entry.0 += type_metrics.true_positive_chars;
            entry.1 += type_metrics.predicted_chars;
            entry.2 += type_metrics.gold_chars;
        }
        document_counts.push(metrics.counts);
    }

    let mut metrics = CharMaskingMetrics::from_counts(totals);
    metrics.strict_span_f1 = strict_counts.exact.f1_exact();
    metrics.gold_span_count = documents.iter().map(|doc| doc.gold.len()).sum();
    metrics.predicted_span_count = documents.iter().map(|doc| doc.predicted.len()).sum();
    metrics.by_entity_type = per_type
        .into_iter()
        .map(
            |(label, (true_positive_chars, predicted_chars, gold_chars))| {
                let type_metrics = CharMaskingTypeMetrics {
                    entity_type: label.clone(),
                    true_positive_chars,
                    predicted_chars,
                    gold_chars,
                    precision: ratio(true_positive_chars, predicted_chars),
                    recall: ratio(true_positive_chars, gold_chars),
                    f1: f_beta(
                        ratio(true_positive_chars, predicted_chars),
                        ratio(true_positive_chars, gold_chars),
                        1.0,
                    ),
                    f2: f_beta(
                        ratio(true_positive_chars, predicted_chars),
                        ratio(true_positive_chars, gold_chars),
                        2.0,
                    ),
                    false_positive_rate: ratio(
                        predicted_chars.saturating_sub(true_positive_chars),
                        metrics.counts.non_pii_chars,
                    ),
                };
                (label, type_metrics)
            },
        )
        .collect();

    if bootstrap {
        metrics.ci_95 = bootstrap_char_masking_ci(&document_counts);
    }
    metrics
}

fn entity_char_range(entity: &Entity, char_count: usize) -> Option<(usize, usize)> {
    let start = entity.start();
    let end = entity.end();
    if start >= end || start >= char_count {
        return None;
    }
    let clipped_end = end.min(char_count);
    (start < clipped_end).then_some((start, clipped_end))
}

fn union_char_ranges<I>(ranges: I, char_count: usize) -> Vec<(usize, usize)>
where
    I: IntoIterator<Item = (usize, usize)>,
{
    let mut ranges: Vec<(usize, usize)> = ranges
        .into_iter()
        .filter_map(|(start, end)| {
            if start >= end || start >= char_count {
                return None;
            }
            let clipped_end = end.min(char_count);
            (start < clipped_end).then_some((start, clipped_end))
        })
        .collect();
    ranges.sort_unstable();

    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (start, end) in ranges {
        if let Some(last) = merged.last_mut() {
            if start <= last.1 {
                last.1 = last.1.max(end);
                continue;
            }
        }
        merged.push((start, end));
    }
    merged
}

fn covered_char_count(ranges: &[(usize, usize)]) -> usize {
    ranges.iter().map(|(start, end)| end - start).sum()
}

fn intersection_char_count(left: &[(usize, usize)], right: &[(usize, usize)]) -> usize {
    let (mut left_index, mut right_index, mut total) = (0usize, 0usize, 0usize);
    while left_index < left.len() && right_index < right.len() {
        let start = left[left_index].0.max(right[right_index].0);
        let end = left[left_index].1.min(right[right_index].1);
        if start < end {
            total += end - start;
        }
        if left[left_index].1 < right[right_index].1 {
            left_index += 1;
        } else {
            right_index += 1;
        }
    }
    total
}

fn per_type_char_masking(
    char_count: usize,
    gold: &[Entity],
    predicted: &[Entity],
    non_pii_chars: usize,
) -> HashMap<String, CharMaskingTypeMetrics> {
    let mut gold_by_type: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
    for entity in gold {
        if let Some(range) = entity_char_range(entity, char_count) {
            gold_by_type
                .entry(entity.entity_type.as_label().to_string())
                .or_default()
                .push(range);
        }
    }
    for ranges in gold_by_type.values_mut() {
        *ranges = union_char_ranges(ranges.iter().copied(), char_count);
    }

    // Attribute each raw predicted span to the gold type it overlaps most. This keeps
    // overlapping predictions in one bucket and avoids inflating every type's count.
    let mut predicted_by_type: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
    for entity in predicted {
        let Some(predicted_range) = entity_char_range(entity, char_count) else {
            continue;
        };
        let mut best: Option<(usize, String)> = None;
        for (label, ranges) in &gold_by_type {
            let overlap = intersection_char_count(&[predicted_range], ranges);
            if overlap == 0 {
                continue;
            }
            let replace = match &best {
                None => true,
                Some((best_overlap, best_label)) => {
                    overlap > *best_overlap || (overlap == *best_overlap && label < best_label)
                }
            };
            if replace {
                best = Some((overlap, label.clone()));
            }
        }
        if let Some((_, label)) = best {
            predicted_by_type
                .entry(label)
                .or_default()
                .push(predicted_range);
        }
    }

    gold_by_type
        .into_iter()
        .map(|(label, gold_ranges)| {
            let predicted_ranges = union_char_ranges(
                predicted_by_type.remove(&label).unwrap_or_default(),
                char_count,
            );
            let gold_chars = covered_char_count(&gold_ranges);
            let predicted_chars = covered_char_count(&predicted_ranges);
            let true_positive_chars = intersection_char_count(&gold_ranges, &predicted_ranges);
            let precision = ratio(true_positive_chars, predicted_chars);
            let recall = ratio(true_positive_chars, gold_chars);
            let metrics = CharMaskingTypeMetrics {
                entity_type: label.clone(),
                true_positive_chars,
                predicted_chars,
                gold_chars,
                precision,
                recall,
                f1: f_beta(precision, recall, 1.0),
                f2: f_beta(precision, recall, 2.0),
                false_positive_rate: ratio(
                    predicted_chars.saturating_sub(true_positive_chars),
                    non_pii_chars,
                ),
            };
            (label, metrics)
        })
        .collect()
}

fn bootstrap_char_masking_ci(document_counts: &[CharMaskingCounts]) -> Option<CharMaskingCi95> {
    if document_counts.is_empty() {
        return None;
    }

    use super::task_evaluator::{
        bootstrap_ci_95, bootstrap_index, CI_BOOTSTRAP_REPLICATES, CI_BOOTSTRAP_SEED,
    };

    let sample_count = document_counts.len();
    let mut state = CI_BOOTSTRAP_SEED;
    let mut precision_samples = Vec::with_capacity(CI_BOOTSTRAP_REPLICATES);
    let mut recall_samples = Vec::with_capacity(CI_BOOTSTRAP_REPLICATES);
    let mut f1_samples = Vec::with_capacity(CI_BOOTSTRAP_REPLICATES);
    let mut f2_samples = Vec::with_capacity(CI_BOOTSTRAP_REPLICATES);
    let mut fpr_samples = Vec::with_capacity(CI_BOOTSTRAP_REPLICATES);

    for _ in 0..CI_BOOTSTRAP_REPLICATES {
        let mut pooled = CharMaskingCounts::default();
        for _ in 0..sample_count {
            let index = bootstrap_index(&mut state, sample_count);
            pooled.merge(&document_counts[index]);
        }
        precision_samples.push(pooled.precision());
        recall_samples.push(pooled.recall());
        f1_samples.push(pooled.f1());
        f2_samples.push(pooled.f2());
        fpr_samples.push(pooled.false_positive_rate());
    }

    Some(CharMaskingCi95 {
        precision: bootstrap_ci_95(&mut precision_samples),
        recall: bootstrap_ci_95(&mut recall_samples),
        f1: bootstrap_ci_95(&mut f1_samples),
        f2: bootstrap_ci_95(&mut f2_samples),
        false_positive_rate: bootstrap_ci_95(&mut fpr_samples),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use anno::{EntityCategory, EntityType};

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

    #[test]
    fn document_scoped_duplicate_metric_ignores_same_name_in_separate_sentences() {
        let first = vec![Entity::new("Paris", EntityType::Location, 0, 5, 0.9)];
        let second = vec![Entity::new("Paris", EntityType::Location, 0, 5, 0.9)];

        let metrics =
            compute_document_extraction_quality_metrics([first.as_slice(), second.as_slice()]);

        assert_eq!(metrics.total, 2);
        assert_eq!(metrics.duplicates, 0);
        assert_eq!(metrics.duplication_rate, 0.0);
    }

    #[test]
    fn document_scoped_duplicate_metric_allows_distinct_mentions_in_one_sentence() {
        let document = vec![
            Entity::new("Paris", EntityType::Location, 0, 5, 0.9),
            Entity::new("Paris", EntityType::Location, 12, 17, 0.9),
        ];

        let metrics = compute_document_extraction_quality_metrics([document.as_slice()]);

        assert_eq!(metrics.total, 2);
        assert_eq!(metrics.duplicates, 0);
        assert_eq!(metrics.duplication_rate, 0.0);
    }

    #[test]
    fn document_scoped_duplicate_metric_counts_repeated_exact_span() {
        let document = vec![
            Entity::new("Paris", EntityType::Location, 0, 5, 0.9),
            Entity::new("Paris", EntityType::Location, 0, 5, 0.8),
        ];

        let metrics = compute_document_extraction_quality_metrics([document.as_slice()]);

        assert_eq!(metrics.total, 2);
        assert_eq!(metrics.duplicates, 1);
        assert_eq!(metrics.duplication_rate, 0.5);
    }

    fn char_entity(text: &str, entity_type: EntityType, start: usize, end: usize) -> Entity {
        Entity::new(text, entity_type, start, end, 1.0)
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-9,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn char_masking_exact_match_scores_one() {
        // 20-character identifier, fully masked.
        let text = "ABCDEFGHIJKLMNOPQRST";
        let gold = vec![char_entity(
            text,
            EntityType::custom("ID_NUMBER", EntityCategory::Misc),
            0,
            20,
        )];
        let predicted = gold.clone();

        let metrics = evaluate_char_masking(text, &gold, &predicted);

        assert_close(metrics.precision, 1.0);
        assert_close(metrics.recall, 1.0);
        assert_close(metrics.f1, 1.0);
        assert_close(metrics.f2, 1.0);
        assert_close(metrics.false_positive_rate, 0.0);
        assert_close(metrics.strict_span_f1, 1.0);
        assert_eq!(metrics.counts.true_positive_chars, 20);
        assert_eq!(metrics.counts.predicted_chars, 20);
        assert_eq!(metrics.counts.gold_chars, 20);
        assert_eq!(metrics.counts.non_pii_chars, 0);
        assert_eq!(metrics.counts.total_chars, 20);
    }

    #[test]
    fn char_masking_one_missed_character_lowers_recall_to_095() {
        // One character of a 20-character identifier is left unmasked.
        let text = "ABCDEFGHIJKLMNOPQRST";
        let gold = vec![char_entity(
            text,
            EntityType::custom("ID_NUMBER", EntityCategory::Misc),
            0,
            20,
        )];
        let predicted = vec![char_entity(
            text,
            EntityType::custom("ID_NUMBER", EntityCategory::Misc),
            0,
            19,
        )];

        let metrics = evaluate_char_masking(text, &gold, &predicted);

        assert_close(metrics.precision, 1.0);
        assert_close(metrics.recall, 0.95);
        // 2 * 1 * 0.95 / (1 + 0.95).
        assert_close(metrics.f1, 0.974_358_974_358_974_4);
        // 5 * 1 * 0.95 / (4 * 1 + 0.95).
        assert_close(metrics.f2, 0.959_595_959_595_959_6);
        assert_close(metrics.false_positive_rate, 0.0);
        assert_eq!(metrics.counts.true_positive_chars, 19);
        assert_eq!(metrics.counts.gold_chars, 20);
        assert_eq!(metrics.counts.non_pii_chars, 0);

        // Standard F-beta with beta > 1 is bounded by precision and recall. With perfect
        // precision and recall 0.95, F2 is *below* F1 because it sits closer to the
        // imperfect recall. Recall weighting shows up as a larger drop from the perfect
        // score, not as F2 > F1 in this case.
        assert!(metrics.f2 < metrics.f1);
        assert!((1.0 - metrics.f2) > (1.0 - metrics.f1));

        // Exact-boundary span F1 sees the boundary error as a complete miss.
        assert_close(metrics.strict_span_f1, 0.0);
    }

    #[test]
    fn char_masking_one_over_masked_word_keeps_recall_and_raises_fpr() {
        // Gold is only "Alice"; the prediction also masks "met".
        let text = "Alice met Bob";
        let gold = vec![char_entity("Alice", EntityType::Person, 0, 5)];
        let predicted = vec![
            char_entity("Alice", EntityType::Person, 0, 5),
            char_entity(
                "met",
                EntityType::custom("MISC", EntityCategory::Misc),
                6,
                9,
            ),
        ];

        let metrics = evaluate_char_masking(text, &gold, &predicted);

        assert_close(metrics.recall, 1.0);
        // 5 correctly masked / 8 predicted characters.
        assert_close(metrics.precision, 0.625);
        assert_close(metrics.f1, 0.769_230_769_230_769_2);
        assert_close(metrics.f2, 0.892_857_142_857_142_9);
        // 3 over-masked characters / 8 non-PII characters.
        assert_close(metrics.false_positive_rate, 0.375);
        assert_eq!(metrics.counts.true_positive_chars, 5);
        assert_eq!(metrics.counts.predicted_chars, 8);
        assert_eq!(metrics.counts.gold_chars, 5);
        assert_eq!(metrics.counts.non_pii_chars, 8);

        // With perfect recall and imperfect precision, recall-weighted F2 is higher.
        assert!(metrics.f2 > metrics.f1);
        assert_close(metrics.strict_span_f1, 2.0 * 0.5 * 1.0 / (0.5 + 1.0));
    }

    #[test]
    fn char_masking_overlapping_predictions_use_union_of_masked_characters() {
        // Two overlapping predictions together cover the full 10-character gold span.
        let text = "ABCDEFGHIJKLMNOPQRST";
        let gold = vec![char_entity(text, EntityType::Person, 0, 10)];
        let predicted = vec![
            char_entity(text, EntityType::Person, 0, 6),
            char_entity(text, EntityType::Person, 4, 10),
        ];

        let metrics = evaluate_char_masking(text, &gold, &predicted);

        assert_close(metrics.precision, 1.0);
        assert_close(metrics.recall, 1.0);
        assert_close(metrics.f1, 1.0);
        assert_close(metrics.f2, 1.0);
        assert_close(metrics.false_positive_rate, 0.0);
        assert_eq!(metrics.counts.predicted_chars, 10);
        assert_eq!(metrics.predicted_span_count, 2);
        // The span scorer still sees two boundary errors, which is exactly the
        // divergence the character metric is meant to expose.
        assert_close(metrics.strict_span_f1, 0.0);
    }

    #[test]
    fn char_masking_counts_characters_not_bytes_for_multibyte_text() {
        // 6 characters, 14 UTF-8 bytes. Gold "李雷" occupies characters 4..6.
        let text = "名前: 李雷";
        let gold = vec![char_entity("李雷", EntityType::Person, 4, 6)];
        let predicted = vec![char_entity(" 李雷", EntityType::Person, 3, 6)];

        let metrics = evaluate_char_masking(text, &gold, &predicted);

        assert_eq!(metrics.counts.total_chars, 6);
        assert_eq!(metrics.counts.gold_chars, 2);
        assert_eq!(metrics.counts.predicted_chars, 3);
        assert_eq!(metrics.counts.true_positive_chars, 2);
        assert_eq!(metrics.counts.non_pii_chars, 4);
        assert_close(metrics.precision, 2.0 / 3.0);
        assert_close(metrics.recall, 1.0);
        assert_close(metrics.f1, 0.8);
        assert_close(metrics.f2, 10.0 / 11.0);
        // One over-masked (non-PII) character out of four.
        assert_close(metrics.false_positive_rate, 0.25);

        let exact = evaluate_char_masking(text, &gold, &gold);
        assert_eq!(exact.counts.total_chars, 6);
        assert_eq!(exact.counts.gold_chars, 2);
        assert_close(exact.precision, 1.0);
        assert_close(exact.recall, 1.0);
    }

    #[test]
    fn char_masking_per_type_breakdown_attributes_spans() {
        let text = "Alice 123-45-6789";
        let gold = vec![
            char_entity("Alice", EntityType::Person, 0, 5),
            char_entity(
                "123-45-6789",
                EntityType::custom("ID_NUMBER", EntityCategory::Misc),
                6,
                17,
            ),
        ];
        let predicted = gold.clone();

        let metrics = evaluate_char_masking(text, &gold, &predicted);

        assert_eq!(metrics.by_entity_type.len(), 2);
        let person = metrics.by_entity_type.get("PER").unwrap();
        assert_close(person.precision, 1.0);
        assert_close(person.recall, 1.0);
        assert_eq!(person.gold_chars, 5);
        let id = metrics.by_entity_type.get("ID_NUMBER").unwrap();
        assert_close(id.precision, 1.0);
        assert_close(id.recall, 1.0);
        assert_eq!(id.gold_chars, 11);
    }
}
