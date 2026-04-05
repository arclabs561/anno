//! Coreference resolution evaluation metrics.
//!
//! This module implements the standard coreference scoring metrics used in
//! shared tasks (CoNLL-2011/2012, CRAC). Each metric captures a different
//! aspect of clustering quality:
//!
//! | Metric  | Unit of evaluation | Reference |
//! |---------|--------------------|-----------|
//! | MUC     | Links between mentions within an entity | Vilain et al., 1995 |
//! | B3      | Per-mention precision/recall | Bagga & Baldwin, 1998 |
//! | CEAF-e  | Entity-level optimal alignment | Luo, 2005 |
//! | CEAF-m  | Mention-level optimal alignment | Luo, 2005 |
//! | LEA     | Link-based, entity-importance weighted | Moosavi & Strube, 2016 |
//! | BLANC   | Rand-index over mention pairs | Recasens & Hovy, 2010 |
//! | CoNLL   | Avg of MUC, B3, CEAF-e F1 scores | Pradhan et al., 2012 |
//!
//! This module is shared by both `anno` (analysis features) and `anno-eval` (evaluation harness).
//! It is intentionally dependency-light: it relies only on `anno-core`, `serde`, and `std`.

use crate::coref::CorefChain;
use crate::types::CorefChainStats;
use anno_core::MentionType;
use std::collections::{HashMap, HashSet};

type SpanId = (usize, usize);

/// Span mode for mention matching: full span or head span.
///
/// Head-match mode is used in CRAC shared tasks where two mentions match
/// if their syntactic heads overlap, even when full spans differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpanMode {
    /// Use the full mention span `(start, end)`.
    #[default]
    Full,
    /// Use the head span `(head_start, head_end)`, falling back to full span
    /// when head annotations are absent.
    Head,
}

fn span_for(mention: &crate::coref::Mention, mode: SpanMode) -> SpanId {
    match mode {
        SpanMode::Full => mention.span_id(),
        SpanMode::Head => mention.span_id_head(),
    }
}

fn build_mention_index(chains: &[CorefChain]) -> HashMap<SpanId, usize> {
    build_mention_index_mode(chains, SpanMode::Full)
}

fn build_mention_index_mode(chains: &[CorefChain], mode: SpanMode) -> HashMap<SpanId, usize> {
    let mut index = HashMap::new();
    for (chain_idx, chain) in chains.iter().enumerate() {
        for mention in &chain.mentions {
            index.insert(span_for(mention, mode), chain_idx);
        }
    }
    index
}

fn all_mention_spans_mode(chains: &[CorefChain], mode: SpanMode) -> HashSet<SpanId> {
    chains
        .iter()
        .flat_map(|c| c.mentions.iter().map(move |m| span_for(m, mode)))
        .collect()
}

fn common_mentions(pred: &[CorefChain], gold: &[CorefChain]) -> HashSet<SpanId> {
    common_mentions_mode(pred, gold, SpanMode::Full)
}

fn common_mentions_mode(
    pred: &[CorefChain],
    gold: &[CorefChain],
    mode: SpanMode,
) -> HashSet<SpanId> {
    let pred_spans = all_mention_spans_mode(pred, mode);
    let gold_spans = all_mention_spans_mode(gold, mode);
    pred_spans.intersection(&gold_spans).copied().collect()
}

/// Filter out singleton chains (chains with exactly one mention).
///
/// CRAC 2022-2025 shared tasks compute primary scores (B3, CEAF, LEA)
/// without singletons. This function strips them from both predicted
/// and gold chain sets before scoring.
#[must_use]
pub fn filter_singletons(chains: &[CorefChain]) -> Vec<CorefChain> {
    chains.iter().filter(|c| c.len() > 1).cloned().collect()
}

/// Compute F1 from precision and recall.
#[inline]
fn prf1(precision: f64, recall: f64) -> f64 {
    if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    }
}

/// Coreference evaluation scores (precision, recall, F1).
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct CorefScores {
    /// Precision.
    pub precision: f64,
    /// Recall.
    pub recall: f64,
    /// F1.
    pub f1: f64,
}

impl CorefScores {
    /// Create a new score triple (F1 computed automatically).
    ///
    /// ```
    /// use anno_metrics::coref_metrics::CorefScores;
    ///
    /// let s = CorefScores::new(0.8, 0.6);
    /// assert!((s.f1 - 2.0 * 0.8 * 0.6 / (0.8 + 0.6)).abs() < 1e-9);
    /// ```
    #[must_use]
    pub fn new(precision: f64, recall: f64) -> Self {
        let f1 = prf1(precision, recall);
        Self {
            precision,
            recall,
            f1,
        }
    }

    /// Create from an existing `(precision, recall, f1)` triple.
    #[must_use]
    pub fn from_tuple((p, r, f1): (f64, f64, f64)) -> Self {
        Self {
            precision: p,
            recall: r,
            f1,
        }
    }
}

/// Complete coreference evaluation results.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct CorefEvaluation {
    /// MUC metric.
    pub muc: CorefScores,
    /// B³ metric.
    pub b_cubed: CorefScores,
    /// CEAF entity-based (phi4).
    pub ceaf_e: CorefScores,
    /// CEAF mention-based (phi3).
    pub ceaf_m: CorefScores,
    /// LEA metric.
    pub lea: CorefScores,
    /// BLANC metric.
    pub blanc: CorefScores,
    /// CoNLL F1 (average of MUC, B³, CEAFe).
    pub conll_f1: f64,
    /// Chain-length stratified diagnostics.
    pub chain_stats: Option<CorefChainStats>,
    /// Zero-anaphor (empty-node) evaluation, if zeros are present.
    pub zero_anaphor: Option<ZeroAnaphorEvaluation>,
}

impl CorefEvaluation {
    /// Compute the full metric bundle.
    ///
    /// ```
    /// use anno_core::core::coref::{CorefChain, Mention};
    /// use anno_metrics::coref_metrics::CorefEvaluation;
    ///
    /// let gold = vec![CorefChain::new(vec![
    ///     Mention::new("John", 0, 4),
    ///     Mention::new("he", 10, 12),
    /// ])];
    /// let eval = CorefEvaluation::compute(&gold, &gold);
    /// assert!((eval.conll_f1 - 1.0).abs() < 1e-9);
    /// assert!((eval.muc.f1 - 1.0).abs() < 1e-9);
    /// ```
    #[must_use]
    pub fn compute(predicted: &[CorefChain], gold: &[CorefChain]) -> Self {
        let muc = CorefScores::from_tuple(muc_score(predicted, gold));
        let b_cubed = CorefScores::from_tuple(b_cubed_score(predicted, gold));
        let ceaf_e = CorefScores::from_tuple(ceaf_e_score(predicted, gold));
        let ceaf_m = CorefScores::from_tuple(ceaf_m_score(predicted, gold));
        let lea = CorefScores::from_tuple(lea_score(predicted, gold));
        let blanc = CorefScores::from_tuple(blanc_score(predicted, gold));
        let conll = conll_f1(predicted, gold);
        let chain_stats = compute_chain_length_stratified(predicted, gold);
        let zero_anaphor = ZeroAnaphorEvaluation::compute(predicted, gold);

        Self {
            muc,
            b_cubed,
            ceaf_e,
            ceaf_m,
            lea,
            blanc,
            conll_f1: conll,
            chain_stats: Some(chain_stats),
            zero_anaphor,
        }
    }

    /// Compute the full metric bundle with singletons excluded.
    ///
    /// CRAC 2022-2025 shared tasks use this as the primary scoring mode:
    /// chains with only one mention are removed from both predicted and
    /// gold before computing B3, CEAF, LEA, and CoNLL scores.
    ///
    /// Scores typically differ from the singleton-included variant because
    /// singletons inflate B3 precision/recall when correctly matched but
    /// contribute no signal to MUC (which already ignores them).
    ///
    /// ```
    /// use anno_core::core::coref::{CorefChain, Mention};
    /// use anno_metrics::coref_metrics::CorefEvaluation;
    ///
    /// let gold = vec![
    ///     CorefChain::new(vec![
    ///         Mention::new("John", 0, 4),
    ///         Mention::new("he", 10, 12),
    ///     ]),
    ///     CorefChain::singleton(Mention::new("Paris", 20, 25)),
    /// ];
    /// let with = CorefEvaluation::compute(&gold, &gold);
    /// let without = CorefEvaluation::compute_without_singletons(&gold, &gold);
    /// // B3 F1 is 1.0 either way for identical input, but the chain_stats differ.
    /// assert!(with.chain_stats.as_ref().unwrap().singleton_count == 1);
    /// assert!(without.chain_stats.as_ref().unwrap().singleton_count == 0);
    /// ```
    #[must_use]
    pub fn compute_without_singletons(predicted: &[CorefChain], gold: &[CorefChain]) -> Self {
        let pred_filtered = filter_singletons(predicted);
        let gold_filtered = filter_singletons(gold);
        Self::compute(&pred_filtered, &gold_filtered)
    }

    /// Extract per-metric F1 scores.
    #[must_use]
    pub fn all_f1_scores(&self) -> Vec<f64> {
        vec![
            self.muc.f1,
            self.b_cubed.f1,
            self.ceaf_e.f1,
            self.ceaf_m.f1,
            self.lea.f1,
            self.blanc.f1,
        ]
    }

    /// Average F1 across all included metrics (diagnostic).
    #[must_use]
    pub fn average_f1(&self) -> f64 {
        let scores = self.all_f1_scores();
        scores.iter().sum::<f64>() / scores.len().max(1) as f64
    }

    /// Standard deviation of F1 scores across metrics.
    #[must_use]
    pub fn f1_std_dev(&self) -> f64 {
        let scores = self.all_f1_scores();
        let mean = self.average_f1();
        let variance: f64 =
            scores.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / scores.len().max(1) as f64;
        variance.sqrt()
    }
}

impl std::fmt::Display for CorefEvaluation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Coreference Evaluation Results:")?;
        writeln!(
            f,
            "  MUC:     P={:.1}%  R={:.1}%  F1={:.1}%",
            self.muc.precision * 100.0,
            self.muc.recall * 100.0,
            self.muc.f1 * 100.0
        )?;
        writeln!(
            f,
            "  B³:      P={:.1}%  R={:.1}%  F1={:.1}%",
            self.b_cubed.precision * 100.0,
            self.b_cubed.recall * 100.0,
            self.b_cubed.f1 * 100.0
        )?;
        writeln!(
            f,
            "  CEAFe:   P={:.1}%  R={:.1}%  F1={:.1}%",
            self.ceaf_e.precision * 100.0,
            self.ceaf_e.recall * 100.0,
            self.ceaf_e.f1 * 100.0
        )?;
        writeln!(
            f,
            "  LEA:     P={:.1}%  R={:.1}%  F1={:.1}%",
            self.lea.precision * 100.0,
            self.lea.recall * 100.0,
            self.lea.f1 * 100.0
        )?;
        writeln!(
            f,
            "  BLANC:   P={:.1}%  R={:.1}%  F1={:.1}%",
            self.blanc.precision * 100.0,
            self.blanc.recall * 100.0,
            self.blanc.f1 * 100.0
        )?;
        writeln!(f, "  CoNLL:   F1={:.1}%", self.conll_f1 * 100.0)?;

        if let Some(z) = self.zero_anaphor {
            writeln!(
                f,
                "  Zero-Anaphor: P={:.1}%  R={:.1}%  F1={:.1}% (tp={} wl={} fp={} fn={})",
                z.precision * 100.0,
                z.recall * 100.0,
                z.f1 * 100.0,
                z.tp,
                z.wl,
                z.fp,
                z.fn_,
            )?;
        }

        Ok(())
    }
}

// =============================================================================
// MUC (Vilain et al., 1995)
// =============================================================================

/// MUC link-based metric (Vilain et al., 1995).
///
/// Counts the minimum number of links needed to reconstruct each gold entity
/// from the predicted partition. For a gold entity K with mentions partitioned
/// into p(K) predicted clusters:
///
/// ```text
///   Recall    = Sigma(|K_i| - |p(K_i)|) / Sigma(|K_i| - 1)
///   Precision = Sigma(|R_j| - |p(R_j)|) / Sigma(|R_j| - 1)
/// ```
///
/// where K_i are gold entities, R_j are predicted entities, and p(X) is the
/// set of partitions induced by the other clustering. Singletons contribute
/// nothing (denominator term |X| - 1 = 0 when |X| = 1).
///
/// Returns `(precision, recall, f1)`. Perfect prediction yields `(1.0, 1.0, 1.0)`.
///
/// ```
/// use anno_core::core::coref::{CorefChain, Mention};
/// use anno_metrics::coref_metrics::muc_score;
///
/// let gold = vec![CorefChain::new(vec![
///     Mention::new("John", 0, 4),
///     Mention::new("he", 10, 12),
/// ])];
/// let (p, r, f1) = muc_score(&gold, &gold);
/// assert!((f1 - 1.0).abs() < 1e-9);
/// ```
#[must_use]
pub fn muc_score(predicted: &[CorefChain], gold: &[CorefChain]) -> (f64, f64, f64) {
    let common = common_mentions(predicted, gold);
    if common.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let pred_index = build_mention_index(predicted);

    let (mut recall_num, mut recall_den) = (0.0, 0.0);
    for gold_chain in gold {
        let gold_mentions: Vec<SpanId> = gold_chain
            .mentions
            .iter()
            .map(|m| m.span_id())
            .filter(|s| common.contains(s))
            .collect();
        if gold_mentions.len() <= 1 {
            continue;
        }

        let mut pred_partitions: HashSet<usize> = HashSet::new();
        for span in &gold_mentions {
            if let Some(&chain_idx) = pred_index.get(span) {
                pred_partitions.insert(chain_idx);
            }
        }

        recall_num += (gold_mentions.len() - pred_partitions.len().max(1)) as f64;
        recall_den += (gold_mentions.len() - 1) as f64;
    }

    let gold_index = build_mention_index(gold);
    let (mut prec_num, mut prec_den) = (0.0, 0.0);
    for pred_chain in predicted {
        let pred_mentions: Vec<SpanId> = pred_chain
            .mentions
            .iter()
            .map(|m| m.span_id())
            .filter(|s| common.contains(s))
            .collect();
        if pred_mentions.len() <= 1 {
            continue;
        }

        let mut gold_partitions: HashSet<usize> = HashSet::new();
        for span in &pred_mentions {
            if let Some(&chain_idx) = gold_index.get(span) {
                gold_partitions.insert(chain_idx);
            }
        }

        prec_num += (pred_mentions.len() - gold_partitions.len().max(1)) as f64;
        prec_den += (pred_mentions.len() - 1) as f64;
    }

    let precision = if prec_den > 0.0 {
        prec_num / prec_den
    } else {
        0.0
    };
    let recall = if recall_den > 0.0 {
        recall_num / recall_den
    } else {
        0.0
    };
    let f1 = prf1(precision, recall);
    (precision, recall, f1)
}

// =============================================================================
// B³ (Bagga & Baldwin, 1998)
// =============================================================================

/// B-cubed mention-based metric (Bagga & Baldwin, 1998).
///
/// Computes precision and recall per mention, then averages over all mentions.
/// For each mention m, let K(m) be its gold entity and R(m) its predicted entity:
///
/// ```text
///   Precision_m = |K(m) cap R(m)| / |R(m)|
///   Recall_m    = |K(m) cap R(m)| / |K(m)|
///   P = (1/N) Sigma Precision_m
///   R = (1/N) Sigma Recall_m
/// ```
///
/// where N is the number of mentions and `cap` denotes set intersection.
/// B3 is sensitive to singleton entities, unlike MUC.
///
/// Returns `(precision, recall, f1)`.
///
/// ```
/// use anno_core::core::coref::{CorefChain, Mention};
/// use anno_metrics::coref_metrics::b_cubed_score;
///
/// let gold = vec![CorefChain::new(vec![
///     Mention::new("John", 0, 4),
///     Mention::new("he", 10, 12),
/// ])];
/// let (p, r, f1) = b_cubed_score(&gold, &gold);
/// assert!((f1 - 1.0).abs() < 1e-9);
/// ```
#[must_use]
pub fn b_cubed_score(predicted: &[CorefChain], gold: &[CorefChain]) -> (f64, f64, f64) {
    let common = common_mentions(predicted, gold);
    if common.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let pred_index = build_mention_index(predicted);
    let gold_index = build_mention_index(gold);

    let mut precision_sum = 0.0;
    let mut recall_sum = 0.0;
    let mut pred_count = 0usize;
    let mut gold_count = 0usize;

    for gold_chain in gold {
        for mention in &gold_chain.mentions {
            let span = mention.span_id();
            if !common.contains(&span) {
                continue;
            }
            gold_count += 1;

            if let Some(&pred_chain_idx) = pred_index.get(&span) {
                let pred_chain = &predicted[pred_chain_idx];
                let pred_spans: HashSet<SpanId> =
                    pred_chain.mentions.iter().map(|m| m.span_id()).collect();
                let gold_spans: HashSet<SpanId> =
                    gold_chain.mentions.iter().map(|m| m.span_id()).collect();
                let overlap = pred_spans.intersection(&gold_spans).count();
                recall_sum += overlap as f64 / gold_chain.mentions.len().max(1) as f64;
            }
        }
    }

    for pred_chain in predicted {
        for mention in &pred_chain.mentions {
            let span = mention.span_id();
            if !common.contains(&span) {
                continue;
            }
            pred_count += 1;

            if let Some(&gold_chain_idx) = gold_index.get(&span) {
                let gold_chain = &gold[gold_chain_idx];
                let pred_spans: HashSet<SpanId> =
                    pred_chain.mentions.iter().map(|m| m.span_id()).collect();
                let gold_spans: HashSet<SpanId> =
                    gold_chain.mentions.iter().map(|m| m.span_id()).collect();
                let overlap = pred_spans.intersection(&gold_spans).count();
                precision_sum += overlap as f64 / pred_chain.mentions.len().max(1) as f64;
            }
        }
    }

    let precision = if pred_count > 0 {
        precision_sum / pred_count as f64
    } else {
        0.0
    };
    let recall = if gold_count > 0 {
        recall_sum / gold_count as f64
    } else {
        0.0
    };
    let f1 = prf1(precision, recall);

    (precision, recall, f1)
}

/// B-cubed score using head-match mode.
///
/// Identical to [`b_cubed_score`] but uses `span_id_head()` instead of
/// `span_id()` for mention identity. Two mentions with different full spans
/// but identical head spans will be treated as the same mention.
///
/// This is the matching mode used in CRAC shared tasks when head annotations
/// are available.
///
/// ```
/// use anno_core::core::coref::{CorefChain, Mention};
/// use anno_metrics::coref_metrics::b_cubed_score_head;
///
/// // Two mentions with different full spans but the same head.
/// let gold = vec![CorefChain::new(vec![
///     Mention::with_head("the president", 0, 13, 4, 13),
///     Mention::with_head("he", 20, 22, 20, 22),
/// ])];
/// let pred = vec![CorefChain::new(vec![
///     Mention::with_head("the former president", 0, 20, 11, 20),
///     Mention::with_head("he", 20, 22, 20, 22),
/// ])];
/// // Under head-match: "president" (4,13) != (11,20), so these are different spans.
/// // But if heads matched, the score would reflect that.
/// ```
#[must_use]
pub fn b_cubed_score_head(predicted: &[CorefChain], gold: &[CorefChain]) -> (f64, f64, f64) {
    let mode = SpanMode::Head;
    let common = common_mentions_mode(predicted, gold, mode);
    if common.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let pred_index = build_mention_index_mode(predicted, mode);
    let gold_index = build_mention_index_mode(gold, mode);

    let mut precision_sum = 0.0;
    let mut recall_sum = 0.0;
    let mut pred_count = 0usize;
    let mut gold_count = 0usize;

    for gold_chain in gold {
        for mention in &gold_chain.mentions {
            let span = span_for(mention, mode);
            if !common.contains(&span) {
                continue;
            }
            gold_count += 1;

            if let Some(&pred_chain_idx) = pred_index.get(&span) {
                let pred_chain = &predicted[pred_chain_idx];
                let pred_spans: HashSet<SpanId> = pred_chain
                    .mentions
                    .iter()
                    .map(|m| span_for(m, mode))
                    .collect();
                let gold_spans: HashSet<SpanId> = gold_chain
                    .mentions
                    .iter()
                    .map(|m| span_for(m, mode))
                    .collect();
                let overlap = pred_spans.intersection(&gold_spans).count();
                recall_sum += overlap as f64 / gold_chain.mentions.len().max(1) as f64;
            }
        }
    }

    for pred_chain in predicted {
        for mention in &pred_chain.mentions {
            let span = span_for(mention, mode);
            if !common.contains(&span) {
                continue;
            }
            pred_count += 1;

            if let Some(&gold_chain_idx) = gold_index.get(&span) {
                let gold_chain = &gold[gold_chain_idx];
                let pred_spans: HashSet<SpanId> = pred_chain
                    .mentions
                    .iter()
                    .map(|m| span_for(m, mode))
                    .collect();
                let gold_spans: HashSet<SpanId> = gold_chain
                    .mentions
                    .iter()
                    .map(|m| span_for(m, mode))
                    .collect();
                let overlap = pred_spans.intersection(&gold_spans).count();
                precision_sum += overlap as f64 / pred_chain.mentions.len().max(1) as f64;
            }
        }
    }

    let precision = if pred_count > 0 {
        precision_sum / pred_count as f64
    } else {
        0.0
    };
    let recall = if gold_count > 0 {
        recall_sum / gold_count as f64
    } else {
        0.0
    };
    let f1 = prf1(precision, recall);

    (precision, recall, f1)
}

// =============================================================================
// CEAF (Luo, 2005)
// =============================================================================

/// Luo's phi3: raw mention intersection count `|K cap R|`. Used for CEAF-m.
fn ceaf_phi3(pred_chain: &CorefChain, gold_chain: &CorefChain) -> f64 {
    let pred_spans: HashSet<SpanId> = pred_chain.mentions.iter().map(|m| m.span_id()).collect();
    let gold_spans: HashSet<SpanId> = gold_chain.mentions.iter().map(|m| m.span_id()).collect();
    pred_spans.intersection(&gold_spans).count() as f64
}

/// Luo's phi4: Dice coefficient `2|K cap R| / (|K| + |R|)`. Used for CEAF-e.
fn ceaf_phi4(pred_chain: &CorefChain, gold_chain: &CorefChain) -> f64 {
    let pred_spans: HashSet<SpanId> = pred_chain.mentions.iter().map(|m| m.span_id()).collect();
    let gold_spans: HashSet<SpanId> = gold_chain.mentions.iter().map(|m| m.span_id()).collect();
    let overlap = pred_spans.intersection(&gold_spans).count();
    if overlap > 0 {
        (2 * overlap) as f64 / (pred_chain.len() + gold_chain.len()) as f64
    } else {
        0.0
    }
}

fn greedy_assignment(
    pred: &[CorefChain],
    gold: &[CorefChain],
    sim_fn: fn(&CorefChain, &CorefChain) -> f64,
) -> f64 {
    if pred.is_empty() || gold.is_empty() {
        return 0.0;
    }

    let mut similarities: Vec<(usize, usize, f64)> = Vec::new();
    for (i, p) in pred.iter().enumerate() {
        for (j, g) in gold.iter().enumerate() {
            let sim = sim_fn(p, g);
            if sim > 0.0 {
                similarities.push((i, j, sim));
            }
        }
    }

    similarities.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut used_pred: HashSet<usize> = HashSet::new();
    let mut used_gold: HashSet<usize> = HashSet::new();
    let mut total_sim = 0.0;

    for (pred_idx, gold_idx, sim) in similarities {
        if !used_pred.contains(&pred_idx) && !used_gold.contains(&gold_idx) {
            total_sim += sim;
            used_pred.insert(pred_idx);
            used_gold.insert(gold_idx);
        }
    }

    total_sim
}

/// CEAF entity-based metric (Luo, 2005), using the phi4 (Dice) similarity.
///
/// Finds a one-to-one alignment between predicted and gold entities
/// using a greedy approximation (not the Kuhn-Munkres / Hungarian algorithm).
/// The entity similarity function phi4 is the Dice coefficient:
///
/// ```text
///   phi4(K, R) = 2|K cap R| / (|K| + |R|)
///   Precision  = Sigma phi4(K*_i, R*_i) / |R|   (number of predicted entities)
///   Recall     = Sigma phi4(K*_i, R*_i) / |K|   (number of gold entities)
/// ```
///
/// where (K\*\_i, R\*\_i) are the aligned entity pairs and the denominators
/// count the number of entities (not mentions).
///
/// Returns `(precision, recall, f1)`.
///
/// ```
/// use anno_core::core::coref::{CorefChain, Mention};
/// use anno_metrics::coref_metrics::ceaf_e_score;
///
/// let gold = vec![CorefChain::new(vec![
///     Mention::new("John", 0, 4),
///     Mention::new("he", 10, 12),
/// ])];
/// let (_, _, f1) = ceaf_e_score(&gold, &gold);
/// assert!((f1 - 1.0).abs() < 1e-9);
/// ```
#[must_use]
pub fn ceaf_e_score(predicted: &[CorefChain], gold: &[CorefChain]) -> (f64, f64, f64) {
    let similarity = greedy_assignment(predicted, gold, ceaf_phi4);
    let precision = if !predicted.is_empty() {
        similarity / predicted.len() as f64
    } else {
        0.0
    };
    let recall = if !gold.is_empty() {
        similarity / gold.len() as f64
    } else {
        0.0
    };
    let f1 = prf1(precision, recall);
    (precision, recall, f1)
}

/// CEAF mention-based metric (Luo, 2005), using the phi3 similarity function.
///
/// Uses raw mention intersection count, normalized by total mentions:
///
/// ```text
///   phi3(K, R) = |K cap R|
///   Precision  = Sigma phi3(K*_i, R*_i) / Sigma |R_j|
///   Recall     = Sigma phi3(K*_i, R*_i) / Sigma |K_j|
/// ```
///
/// Returns `(precision, recall, f1)`.
///
/// ```
/// use anno_core::core::coref::{CorefChain, Mention};
/// use anno_metrics::coref_metrics::ceaf_m_score;
///
/// let gold = vec![CorefChain::new(vec![
///     Mention::new("John", 0, 4),
///     Mention::new("he", 10, 12),
/// ])];
/// let (_, _, f1) = ceaf_m_score(&gold, &gold);
/// assert!((f1 - 1.0).abs() < 1e-9);
/// ```
#[must_use]
pub fn ceaf_m_score(predicted: &[CorefChain], gold: &[CorefChain]) -> (f64, f64, f64) {
    let similarity = greedy_assignment(predicted, gold, ceaf_phi3);
    let pred_mentions: usize = predicted.iter().map(|c| c.len()).sum();
    let gold_mentions: usize = gold.iter().map(|c| c.len()).sum();

    let precision = if pred_mentions > 0 {
        similarity / pred_mentions as f64
    } else {
        0.0
    };
    let recall = if gold_mentions > 0 {
        similarity / gold_mentions as f64
    } else {
        0.0
    };
    let f1 = prf1(precision, recall);
    (precision, recall, f1)
}

// =============================================================================
// LEA (Moosavi & Strube, 2016)
// =============================================================================

/// LEA link-based entity-aware metric (Moosavi & Strube, 2016).
///
/// Weights each entity by its size (importance) and measures the fraction
/// of correctly resolved within-entity links:
///
/// ```text
///   link(K_i) = |correct coreference links in K_i| / (|K_i| choose 2)
///   Recall    = Sigma(|K_i| * link(K_i)) / Sigma |K_i|
///   Precision = Sigma(|R_j| * link(R_j)) / Sigma |R_j|
/// ```
///
/// where K_i are gold entities, R_j are predicted entities, and a "correct
/// coreference link" is a mention pair (m_a, m_b) that appears in the same
/// entity in both the gold and predicted clusterings.
///
/// LEA is designed to give larger entities proportionally more weight,
/// addressing a limitation of B3 where small entities dominate the score.
///
/// Returns `(precision, recall, f1)`.
///
/// ```
/// use anno_core::core::coref::{CorefChain, Mention};
/// use anno_metrics::coref_metrics::lea_score;
///
/// let gold = vec![CorefChain::new(vec![
///     Mention::new("John", 0, 4),
///     Mention::new("he", 10, 12),
/// ])];
/// let (_, _, f1) = lea_score(&gold, &gold);
/// assert!((f1 - 1.0).abs() < 1e-9);
/// ```
#[must_use]
pub fn lea_score(predicted: &[CorefChain], gold: &[CorefChain]) -> (f64, f64, f64) {
    let common = common_mentions(predicted, gold);
    if common.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let pred_index = build_mention_index(predicted);
    let gold_index = build_mention_index(gold);

    let (mut recall_num, mut recall_den) = (0.0, 0.0);
    for gold_chain in gold {
        let gold_mentions: Vec<SpanId> = gold_chain
            .mentions
            .iter()
            .map(|m| m.span_id())
            .filter(|s| common.contains(s))
            .collect();
        if gold_mentions.is_empty() {
            continue;
        }
        let importance = gold_mentions.len() as f64;
        recall_den += importance;

        if gold_mentions.len() == 1 {
            let span = gold_mentions[0];
            if let Some(&pred_chain_idx) = pred_index.get(&span) {
                let pred_chain = &predicted[pred_chain_idx];
                let pred_in_common: Vec<SpanId> = pred_chain
                    .mentions
                    .iter()
                    .map(|m| m.span_id())
                    .filter(|s| common.contains(s))
                    .collect();
                if pred_in_common.len() == 1 {
                    recall_num += importance;
                }
            }
        } else {
            let mut correct_links = 0usize;
            let total_links = gold_mentions.len() * (gold_mentions.len() - 1) / 2;
            for i in 0..gold_mentions.len() {
                for j in (i + 1)..gold_mentions.len() {
                    let span_i = gold_mentions[i];
                    let span_j = gold_mentions[j];
                    if let (Some(&pred_i), Some(&pred_j)) =
                        (pred_index.get(&span_i), pred_index.get(&span_j))
                    {
                        if pred_i == pred_j {
                            correct_links += 1;
                        }
                    }
                }
            }
            let resolution = if total_links > 0 {
                correct_links as f64 / total_links as f64
            } else {
                0.0
            };
            recall_num += importance * resolution;
        }
    }

    let (mut prec_num, mut prec_den) = (0.0, 0.0);
    for pred_chain in predicted {
        let pred_mentions: Vec<SpanId> = pred_chain
            .mentions
            .iter()
            .map(|m| m.span_id())
            .filter(|s| common.contains(s))
            .collect();
        if pred_mentions.is_empty() {
            continue;
        }
        let importance = pred_mentions.len() as f64;
        prec_den += importance;

        if pred_mentions.len() == 1 {
            let span = pred_mentions[0];
            if let Some(&gold_chain_idx) = gold_index.get(&span) {
                let gold_chain = &gold[gold_chain_idx];
                let gold_in_common: Vec<SpanId> = gold_chain
                    .mentions
                    .iter()
                    .map(|m| m.span_id())
                    .filter(|s| common.contains(s))
                    .collect();
                if gold_in_common.len() == 1 {
                    prec_num += importance;
                }
            }
        } else {
            let mut correct_links = 0usize;
            let total_links = pred_mentions.len() * (pred_mentions.len() - 1) / 2;
            for i in 0..pred_mentions.len() {
                for j in (i + 1)..pred_mentions.len() {
                    let span_i = pred_mentions[i];
                    let span_j = pred_mentions[j];
                    if let (Some(&gold_i), Some(&gold_j)) =
                        (gold_index.get(&span_i), gold_index.get(&span_j))
                    {
                        if gold_i == gold_j {
                            correct_links += 1;
                        }
                    }
                }
            }
            let resolution = if total_links > 0 {
                correct_links as f64 / total_links as f64
            } else {
                0.0
            };
            prec_num += importance * resolution;
        }
    }

    let precision = if prec_den > 0.0 {
        prec_num / prec_den
    } else {
        0.0
    };
    let recall = if recall_den > 0.0 {
        recall_num / recall_den
    } else {
        0.0
    };
    let f1 = prf1(precision, recall);

    (precision, recall, f1)
}

// =============================================================================
// BLANC (Recasens & Hovy, 2010)
// =============================================================================

/// BLANC Rand-index-style metric (Recasens & Hovy, 2010).
///
/// Evaluates coreference by classifying all mention pairs as either
/// coreferent (C) or non-coreferent (N), then computing P/R/F1 for
/// each class separately:
///
/// ```text
///   P_c = tp_c / (tp_c + fp_c)      R_c = tp_c / (tp_c + fn_c)
///   P_n = tp_n / (tp_n + fp_n)      R_n = tp_n / (tp_n + fn_n)
///   P   = (P_c + P_n) / 2           R   = (R_c + R_n) / 2
///   F1  = (F1_c + F1_n) / 2
/// ```
///
/// where tp/fp/fn are true-positive, false-positive, and false-negative
/// counts for coreferent (c) and non-coreferent (n) pair decisions.
/// A false positive coreferent pair is simultaneously a false negative
/// non-coreferent pair, and vice versa.
///
/// Returns `(precision, recall, f1)`.
///
/// ```
/// use anno_core::core::coref::{CorefChain, Mention};
/// use anno_metrics::coref_metrics::blanc_score;
///
/// // Two chains: coreferent and non-coreferent pairs both exist.
/// let gold = vec![
///     CorefChain::new(vec![
///         Mention::new("John", 0, 4),
///         Mention::new("he", 10, 12),
///     ]),
///     CorefChain::new(vec![
///         Mention::new("Mary", 20, 24),
///     ]),
/// ];
/// let (_, _, f1) = blanc_score(&gold, &gold);
/// assert!((f1 - 1.0).abs() < 1e-9);
/// ```
#[must_use]
pub fn blanc_score(predicted: &[CorefChain], gold: &[CorefChain]) -> (f64, f64, f64) {
    let common: Vec<SpanId> = common_mentions(predicted, gold).into_iter().collect();
    if common.len() < 2 {
        return (1.0, 1.0, 1.0);
    }

    let pred_index = build_mention_index(predicted);
    let gold_index = build_mention_index(gold);

    let mut coref_tp = 0usize;
    let mut coref_fp = 0usize;
    let mut coref_fn = 0usize;
    let mut non_coref_tp = 0usize;
    let mut non_coref_fp = 0usize;
    let mut non_coref_fn = 0usize;

    for i in 0..common.len() {
        for j in (i + 1)..common.len() {
            let span_i = common[i];
            let span_j = common[j];

            let pred_same = match (pred_index.get(&span_i), pred_index.get(&span_j)) {
                (Some(&pi), Some(&pj)) => pi == pj,
                _ => false,
            };
            let gold_same = match (gold_index.get(&span_i), gold_index.get(&span_j)) {
                (Some(&gi), Some(&gj)) => gi == gj,
                _ => false,
            };

            match (pred_same, gold_same) {
                (true, true) => coref_tp += 1,
                (true, false) => {
                    coref_fp += 1;
                    non_coref_fn += 1;
                }
                (false, true) => {
                    coref_fn += 1;
                    non_coref_fp += 1;
                }
                (false, false) => non_coref_tp += 1,
            }
        }
    }

    let coref_precision = if coref_tp + coref_fp > 0 {
        coref_tp as f64 / (coref_tp + coref_fp) as f64
    } else {
        0.0
    };
    let coref_recall = if coref_tp + coref_fn > 0 {
        coref_tp as f64 / (coref_tp + coref_fn) as f64
    } else {
        0.0
    };
    let coref_f1 = if coref_precision + coref_recall > 0.0 {
        2.0 * coref_precision * coref_recall / (coref_precision + coref_recall)
    } else {
        0.0
    };

    let non_coref_precision = if non_coref_tp + non_coref_fp > 0 {
        non_coref_tp as f64 / (non_coref_tp + non_coref_fp) as f64
    } else {
        0.0
    };
    let non_coref_recall = if non_coref_tp + non_coref_fn > 0 {
        non_coref_tp as f64 / (non_coref_tp + non_coref_fn) as f64
    } else {
        0.0
    };
    let non_coref_f1 = if non_coref_precision + non_coref_recall > 0.0 {
        2.0 * non_coref_precision * non_coref_recall / (non_coref_precision + non_coref_recall)
    } else {
        0.0
    };

    let precision = (coref_precision + non_coref_precision) / 2.0;
    let recall = (coref_recall + non_coref_recall) / 2.0;
    let f1 = (coref_f1 + non_coref_f1) / 2.0;
    (precision, recall, f1)
}

// =============================================================================
// CoNLL F1
// =============================================================================

/// CoNLL F1: the official shared-task aggregate (Pradhan et al., 2012).
///
/// ```text
///   CoNLL_F1 = (F1_MUC + F1_B3 + F1_CEAFe) / 3
/// ```
///
/// This is the primary metric used in CoNLL-2011/2012 shared tasks on
/// coreference resolution. It balances the link-based (MUC), mention-based
/// (B3), and entity-based (CEAF-e) perspectives.
///
/// ```
/// use anno_core::core::coref::{CorefChain, Mention};
/// use anno_metrics::coref_metrics::conll_f1;
///
/// let gold = vec![CorefChain::new(vec![
///     Mention::new("John", 0, 4),
///     Mention::new("he", 10, 12),
/// ])];
/// let f1 = conll_f1(&gold, &gold);
/// assert!((f1 - 1.0).abs() < 1e-9);
/// ```
#[must_use]
pub fn conll_f1(predicted: &[CorefChain], gold: &[CorefChain]) -> f64 {
    let (_, _, muc_f1) = muc_score(predicted, gold);
    let (_, _, b3_f1) = b_cubed_score(predicted, gold);
    let (_, _, ceaf_f1) = ceaf_e_score(predicted, gold);
    (muc_f1 + b3_f1 + ceaf_f1) / 3.0
}

// =============================================================================
// Chain-length stratification
// =============================================================================

/// Compute chain-length stratified diagnostics using LEA within each stratum.
#[must_use]
pub fn compute_chain_length_stratified(
    predicted: &[CorefChain],
    gold: &[CorefChain],
) -> CorefChainStats {
    let mut long_pred: Vec<&CorefChain> = Vec::new();
    let mut short_pred: Vec<&CorefChain> = Vec::new();
    let mut singleton_pred: Vec<&CorefChain> = Vec::new();

    let mut long_gold: Vec<&CorefChain> = Vec::new();
    let mut short_gold: Vec<&CorefChain> = Vec::new();
    let mut singleton_gold: Vec<&CorefChain> = Vec::new();

    for chain in predicted {
        if chain.len() > 10 {
            long_pred.push(chain);
        } else if chain.len() > 1 {
            short_pred.push(chain);
        } else {
            singleton_pred.push(chain);
        }
    }

    for chain in gold {
        if chain.len() > 10 {
            long_gold.push(chain);
        } else if chain.len() > 1 {
            short_gold.push(chain);
        } else {
            singleton_gold.push(chain);
        }
    }

    let long_chain_f1 = if !long_pred.is_empty() || !long_gold.is_empty() {
        let (_, _, f1) = lea_score(
            &long_pred.iter().map(|c| (*c).clone()).collect::<Vec<_>>(),
            &long_gold.iter().map(|c| (*c).clone()).collect::<Vec<_>>(),
        );
        f1
    } else {
        0.0
    };

    let short_chain_f1 = if !short_pred.is_empty() || !short_gold.is_empty() {
        let (_, _, f1) = lea_score(
            &short_pred.iter().map(|c| (*c).clone()).collect::<Vec<_>>(),
            &short_gold.iter().map(|c| (*c).clone()).collect::<Vec<_>>(),
        );
        f1
    } else {
        0.0
    };

    let singleton_f1 = if !singleton_pred.is_empty() || !singleton_gold.is_empty() {
        let (_, _, f1) = lea_score(
            &singleton_pred
                .iter()
                .map(|c| (*c).clone())
                .collect::<Vec<_>>(),
            &singleton_gold
                .iter()
                .map(|c| (*c).clone())
                .collect::<Vec<_>>(),
        );
        f1
    } else {
        0.0
    };

    CorefChainStats {
        long_chain_count: long_gold.len(),
        short_chain_count: short_gold.len(),
        singleton_count: singleton_gold.len(),
        long_chain_f1,
        short_chain_f1,
        singleton_f1,
    }
}

// =============================================================================
// Zero-anaphor evaluation (CorefUD-style)
// =============================================================================

/// CorefUD-style anaphor-decomposable evaluation for zero/empty mentions.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct ZeroAnaphorEvaluation {
    /// Precision.
    pub precision: f64,
    /// Recall.
    pub recall: f64,
    /// F1.
    pub f1: f64,
    /// True positives.
    pub tp: usize,
    /// Wrong linkages.
    pub wl: usize,
    /// False positives.
    pub fp: usize,
    /// False negatives.
    pub fn_: usize,
    /// Gold anaphor count.
    pub gold_anaphors: usize,
    /// Predicted anaphor count.
    pub pred_anaphors: usize,
}

impl ZeroAnaphorEvaluation {
    /// Compute CorefUD-style scoring for zero mentions.
    ///
    /// Returns `None` when neither gold nor predicted contains any zero/empty mentions.
    #[must_use]
    pub fn compute(predicted: &[CorefChain], gold: &[CorefChain]) -> Option<Self> {
        fn zero_spans(chains: &[CorefChain]) -> HashSet<SpanId> {
            chains
                .iter()
                .flat_map(|c| c.mentions.iter())
                .filter(|m| m.mention_type == Some(MentionType::Zero) || m.start == m.end)
                .map(|m| m.span_id())
                .collect()
        }

        fn preceding_spans(chain: &CorefChain, anchor_start: usize) -> HashSet<SpanId> {
            chain
                .mentions
                .iter()
                .filter(|m| m.end <= anchor_start)
                .map(|m| m.span_id())
                .collect()
        }

        let gold_zero = zero_spans(gold);
        let pred_zero = zero_spans(predicted);
        let all_zero: HashSet<SpanId> = gold_zero.union(&pred_zero).copied().collect();
        if all_zero.is_empty() {
            return None;
        }

        let gold_index = build_mention_index(gold);
        let pred_index = build_mention_index(predicted);

        let mut tp = 0usize;
        let mut wl = 0usize;
        let mut fp = 0usize;
        let mut fn_ = 0usize;
        let mut gold_anaphors = 0usize;
        let mut pred_anaphors = 0usize;

        for (z_start, z_end) in all_zero {
            let gold_chain = gold_index
                .get(&(z_start, z_end))
                .and_then(|&idx| gold.get(idx));
            let mut gold_pre = gold_chain
                .map(|c| preceding_spans(c, z_start))
                .unwrap_or_default();
            gold_pre.remove(&(z_start, z_end));
            let gold_anaphoric = !gold_pre.is_empty();

            let pred_chain = pred_index
                .get(&(z_start, z_end))
                .and_then(|&idx| predicted.get(idx));
            let mut pred_pre = pred_chain
                .map(|c| preceding_spans(c, z_start))
                .unwrap_or_default();
            pred_pre.remove(&(z_start, z_end));
            let pred_anaphoric = !pred_pre.is_empty();

            if gold_anaphoric {
                gold_anaphors += 1;
                if !pred_anaphoric {
                    fn_ += 1;
                    continue;
                }
                pred_anaphors += 1;
                if gold_pre.intersection(&pred_pre).next().is_some() {
                    tp += 1;
                } else {
                    wl += 1;
                }
            } else if pred_anaphoric {
                pred_anaphors += 1;
                fp += 1;
            }
        }

        let precision = if tp + wl + fp > 0 {
            tp as f64 / (tp + wl + fp) as f64
        } else {
            0.0
        };
        let recall = if tp + wl + fn_ > 0 {
            tp as f64 / (tp + wl + fn_) as f64
        } else {
            0.0
        };
        let f1 = prf1(precision, recall);

        Some(Self {
            precision,
            recall,
            f1,
            tp,
            wl,
            fp,
            fn_,
            gold_anaphors,
            pred_anaphors,
        })
    }
}

// =============================================================================
// Multi-document aggregation + significance testing
// =============================================================================

/// Aggregate evaluation results across multiple documents.
#[derive(Debug, Clone, Default)]
pub struct AggregateCorefEvaluation {
    /// Per-document evaluations.
    pub per_document: Vec<CorefEvaluation>,
    /// Mean scores across documents.
    pub mean: CorefEvaluation,
    /// Standard deviation of scores.
    pub std_dev: CorefScoreStdDev,
    /// Number of documents.
    pub num_documents: usize,
}

/// Standard deviation for each metric.
#[derive(Debug, Clone, Default)]
pub struct CorefScoreStdDev {
    /// MUC F1 standard deviation.
    pub muc: f64,
    /// B³ F1 standard deviation.
    pub b_cubed: f64,
    /// CEAFe F1 standard deviation.
    pub ceaf_e: f64,
    /// CEAFm F1 standard deviation.
    pub ceaf_m: f64,
    /// LEA F1 standard deviation.
    pub lea: f64,
    /// BLANC F1 standard deviation.
    pub blanc: f64,
    /// CoNLL F1 standard deviation.
    pub conll: f64,
}

fn std_dev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    variance.sqrt()
}

impl AggregateCorefEvaluation {
    /// Compute aggregate metrics over multiple document pairs.
    ///
    /// Each pair is `(predicted_chains, gold_chains)`.
    #[must_use]
    pub fn compute(document_pairs: &[(&[CorefChain], &[CorefChain])]) -> Self {
        if document_pairs.is_empty() {
            return Self::default();
        }

        let evaluations: Vec<CorefEvaluation> = document_pairs
            .iter()
            .map(|(pred, gold)| CorefEvaluation::compute(pred, gold))
            .collect();

        let n = evaluations.len() as f64;

        let mean_muc_p = evaluations.iter().map(|e| e.muc.precision).sum::<f64>() / n;
        let mean_muc_r = evaluations.iter().map(|e| e.muc.recall).sum::<f64>() / n;
        let mean_b3_p = evaluations.iter().map(|e| e.b_cubed.precision).sum::<f64>() / n;
        let mean_b3_r = evaluations.iter().map(|e| e.b_cubed.recall).sum::<f64>() / n;
        let mean_ceafe_p = evaluations.iter().map(|e| e.ceaf_e.precision).sum::<f64>() / n;
        let mean_ceafe_r = evaluations.iter().map(|e| e.ceaf_e.recall).sum::<f64>() / n;
        let mean_ceafm_p = evaluations.iter().map(|e| e.ceaf_m.precision).sum::<f64>() / n;
        let mean_ceafm_r = evaluations.iter().map(|e| e.ceaf_m.recall).sum::<f64>() / n;
        let mean_lea_p = evaluations.iter().map(|e| e.lea.precision).sum::<f64>() / n;
        let mean_lea_r = evaluations.iter().map(|e| e.lea.recall).sum::<f64>() / n;
        let mean_blanc_p = evaluations.iter().map(|e| e.blanc.precision).sum::<f64>() / n;
        let mean_blanc_r = evaluations.iter().map(|e| e.blanc.recall).sum::<f64>() / n;

        let mean = CorefEvaluation {
            muc: CorefScores::new(mean_muc_p, mean_muc_r),
            b_cubed: CorefScores::new(mean_b3_p, mean_b3_r),
            ceaf_e: CorefScores::new(mean_ceafe_p, mean_ceafe_r),
            ceaf_m: CorefScores::new(mean_ceafm_p, mean_ceafm_r),
            lea: CorefScores::new(mean_lea_p, mean_lea_r),
            blanc: CorefScores::new(mean_blanc_p, mean_blanc_r),
            conll_f1: evaluations.iter().map(|e| e.conll_f1).sum::<f64>() / n,
            chain_stats: None,
            zero_anaphor: None,
        };

        let std_muc = std_dev(&evaluations.iter().map(|e| e.muc.f1).collect::<Vec<_>>());
        let std_b3 = std_dev(&evaluations.iter().map(|e| e.b_cubed.f1).collect::<Vec<_>>());
        let std_ceafe = std_dev(&evaluations.iter().map(|e| e.ceaf_e.f1).collect::<Vec<_>>());
        let std_ceafm = std_dev(&evaluations.iter().map(|e| e.ceaf_m.f1).collect::<Vec<_>>());
        let std_lea = std_dev(&evaluations.iter().map(|e| e.lea.f1).collect::<Vec<_>>());
        let std_blanc = std_dev(&evaluations.iter().map(|e| e.blanc.f1).collect::<Vec<_>>());
        let std_conll = std_dev(&evaluations.iter().map(|e| e.conll_f1).collect::<Vec<_>>());

        Self {
            per_document: evaluations,
            mean,
            std_dev: CorefScoreStdDev {
                muc: std_muc,
                b_cubed: std_b3,
                ceaf_e: std_ceafe,
                ceaf_m: std_ceafm,
                lea: std_lea,
                blanc: std_blanc,
                conll: std_conll,
            },
            num_documents: document_pairs.len(),
        }
    }
}

/// Result of a paired significance test between two systems (very rough p-value approximation).
#[derive(Debug, Clone)]
pub struct SignificanceTest {
    /// System A mean score.
    pub mean_a: f64,
    /// System B mean score.
    pub mean_b: f64,
    /// Difference (A - B).
    pub difference: f64,
    /// Standard error of the difference.
    pub std_error: f64,
    /// t-statistic.
    pub t_statistic: f64,
    /// p-value (two-tailed, approximate).
    pub p_value: f64,
    /// Number of samples.
    pub n: usize,
    /// Significant at p < 0.05.
    pub significant_05: bool,
    /// Significant at p < 0.01.
    pub significant_01: bool,
}

impl SignificanceTest {
    /// Perform a paired t-test on CoNLL F1 scores (approximate).
    #[must_use]
    pub fn paired_t_test(scores_a: &[f64], scores_b: &[f64]) -> Self {
        assert_eq!(
            scores_a.len(),
            scores_b.len(),
            "Scores must have same length"
        );
        let n = scores_a.len();
        if n < 2 {
            return Self {
                mean_a: scores_a.first().copied().unwrap_or(0.0),
                mean_b: scores_b.first().copied().unwrap_or(0.0),
                difference: 0.0,
                std_error: 0.0,
                t_statistic: 0.0,
                p_value: 1.0,
                n,
                significant_05: false,
                significant_01: false,
            };
        }

        let differences: Vec<f64> = scores_a
            .iter()
            .zip(scores_b.iter())
            .map(|(a, b)| a - b)
            .collect();
        let mean_diff = differences.iter().sum::<f64>() / n as f64;
        let mean_a = scores_a.iter().sum::<f64>() / n as f64;
        let mean_b = scores_b.iter().sum::<f64>() / n as f64;

        let variance: f64 = differences
            .iter()
            .map(|&d| (d - mean_diff).powi(2))
            .sum::<f64>()
            / (n - 1) as f64;
        let std_diff = variance.sqrt();
        let std_error = std_diff / (n as f64).sqrt();
        let t_stat = if std_error > 0.0 {
            mean_diff / std_error
        } else {
            0.0
        };

        // Rough p-value approximation: use normal critical values for df>=30, else conservative.
        let abs_t = t_stat.abs();
        let p_value = if n >= 30 {
            if abs_t >= 2.576 {
                0.01
            } else if abs_t >= 1.96 {
                0.05
            } else {
                0.10
            }
        } else if abs_t >= 2.75 {
            0.01
        } else if abs_t >= 2.04 {
            0.05
        } else {
            0.10
        };

        Self {
            mean_a,
            mean_b,
            difference: mean_diff,
            std_error,
            t_statistic: t_stat,
            p_value,
            n,
            significant_05: p_value < 0.05,
            significant_01: p_value < 0.01,
        }
    }

    /// True if system A is significantly better than system B.
    #[must_use]
    pub fn a_better_than_b(&self) -> bool {
        self.significant_05 && self.difference > 0.0
    }

    /// True if system B is significantly better than system A.
    #[must_use]
    pub fn b_better_than_a(&self) -> bool {
        self.significant_05 && self.difference < 0.0
    }
}

/// Compare two systems using a paired t-test on CoNLL F1.
#[must_use]
pub fn compare_systems(
    system_a: &[CorefEvaluation],
    system_b: &[CorefEvaluation],
) -> SignificanceTest {
    let scores_a: Vec<f64> = system_a.iter().map(|e| e.conll_f1).collect();
    let scores_b: Vec<f64> = system_b.iter().map(|e| e.conll_f1).collect();
    SignificanceTest::paired_t_test(&scores_a, &scores_b)
}

// =============================================================================
// Window fragmentation diagnostic
// =============================================================================

/// Diagnostics for long-document / windowed coreference: fragmentation across windows.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct WindowFragmentationStats {
    /// Window size (characters).
    pub window_size: usize,
    /// Window overlap (characters).
    pub window_overlap: usize,
    /// Number of gold chains spanning 2+ windows.
    pub multiwindow_gold_chains: usize,
    /// Number of multiwindow gold chains fragmented in predictions.
    pub fragmented_gold_chains: usize,
    /// Number of adjacent-window boundary checks performed.
    pub boundary_checks: usize,
    /// Number of boundary checks where mentions fall into different predicted clusters.
    pub boundary_splits: usize,
    /// Number of gold mentions in multiwindow chains missing from predictions.
    pub missing_mentions_in_multiwindow_chains: usize,
}

impl WindowFragmentationStats {
    /// Compute fragmentation stats for one document.
    #[must_use]
    pub fn compute(
        predicted: &[CorefChain],
        gold: &[CorefChain],
        window_size: usize,
        window_overlap: usize,
    ) -> Option<Self> {
        if window_size == 0 {
            return None;
        }
        let step = window_size.saturating_sub(window_overlap).max(1);

        fn window_idx_for(start: usize, step: usize) -> usize {
            start / step
        }

        let pred_index = build_mention_index(predicted);
        let mut stats = Self {
            window_size,
            window_overlap,
            ..Default::default()
        };

        for gold_chain in gold {
            if gold_chain.mentions.len() <= 1 {
                continue;
            }

            let mut windows: HashSet<usize> = HashSet::new();
            let mut pred_clusters: HashSet<Option<usize>> = HashSet::new();

            for m in &gold_chain.mentions {
                windows.insert(window_idx_for(m.start, step));
                let span = m.span_id();
                let pred = pred_index.get(&span).copied();
                if pred.is_none() {
                    stats.missing_mentions_in_multiwindow_chains += 1;
                }
                pred_clusters.insert(pred);
            }

            if windows.len() <= 1 {
                continue;
            }
            stats.multiwindow_gold_chains += 1;

            let fragmented = pred_clusters.len() > 1 || pred_clusters.contains(&None);
            if fragmented {
                stats.fragmented_gold_chains += 1;
            }

            let mut sorted_windows: Vec<usize> = windows.into_iter().collect();
            sorted_windows.sort_unstable();
            for pair in sorted_windows.windows(2) {
                let w0 = pair[0];
                let w1 = pair[1];
                stats.boundary_checks += 1;

                let mut pred_in_w0: HashSet<usize> = HashSet::new();
                let mut pred_in_w1: HashSet<usize> = HashSet::new();

                for m in &gold_chain.mentions {
                    let w = window_idx_for(m.start, step);
                    let Some(&pidx) = pred_index.get(&m.span_id()) else {
                        continue;
                    };
                    if w == w0 {
                        pred_in_w0.insert(pidx);
                    } else if w == w1 {
                        pred_in_w1.insert(pidx);
                    }
                }

                let shared = pred_in_w0.intersection(&pred_in_w1).next().is_some();
                if !shared {
                    stats.boundary_splits += 1;
                }
            }
        }

        if stats.multiwindow_gold_chains == 0 {
            None
        } else {
            Some(stats)
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coref::{CorefChain, Mention};

    // ---- helpers ----

    /// Shorthand: create a mention at given offsets.
    fn m(text: &str, start: usize, end: usize) -> Mention {
        Mention::new(text, start, end)
    }

    /// Build chains from a vec of vecs of (text, start, end).
    fn chains(specs: Vec<Vec<(&str, usize, usize)>>) -> Vec<CorefChain> {
        specs
            .into_iter()
            .map(|mentions| {
                CorefChain::new(mentions.into_iter().map(|(t, s, e)| m(t, s, e)).collect())
            })
            .collect()
    }

    const EPS: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    fn approx_eq_loose(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    // =========================================================================
    // 1. prf1 helper
    // =========================================================================

    #[test]
    fn prf1_perfect() {
        assert!(approx_eq(prf1(1.0, 1.0), 1.0));
    }

    #[test]
    fn prf1_balanced_half() {
        assert!(approx_eq(prf1(0.5, 0.5), 0.5));
    }

    #[test]
    fn prf1_zero_recall() {
        assert!(approx_eq(prf1(1.0, 0.0), 0.0));
    }

    #[test]
    fn prf1_both_zero() {
        assert!(approx_eq(prf1(0.0, 0.0), 0.0));
    }

    #[test]
    fn prf1_zero_precision() {
        assert!(approx_eq(prf1(0.0, 1.0), 0.0));
    }

    // =========================================================================
    // 2. CorefScores::new
    // =========================================================================

    #[test]
    fn coref_scores_new_computes_harmonic_mean() {
        let s = CorefScores::new(0.8, 0.6);
        let expected_f1 = 2.0 * 0.8 * 0.6 / (0.8 + 0.6);
        assert!(approx_eq(s.f1, expected_f1));
        assert!(approx_eq(s.precision, 0.8));
        assert!(approx_eq(s.recall, 0.6));
    }

    #[test]
    fn coref_scores_new_perfect() {
        let s = CorefScores::new(1.0, 1.0);
        assert!(approx_eq(s.f1, 1.0));
    }

    #[test]
    fn coref_scores_new_zero() {
        let s = CorefScores::new(0.0, 0.0);
        assert!(approx_eq(s.f1, 0.0));
    }

    // =========================================================================
    // 3. muc_score
    // =========================================================================

    #[test]
    fn muc_perfect_prediction() {
        // Gold: {A(0,1), B(2,3), C(4,5)}
        // Pred: identical
        let gold = chains(vec![vec![("A", 0, 1), ("B", 2, 3), ("C", 4, 5)]]);
        let pred = gold.clone();
        let (p, r, f1) = muc_score(&pred, &gold);
        assert!(approx_eq(p, 1.0), "p={p}");
        assert!(approx_eq(r, 1.0), "r={r}");
        assert!(approx_eq(f1, 1.0), "f1={f1}");
    }

    #[test]
    fn muc_all_singletons_vs_one_chain() {
        // Gold: one chain {A, B, C}
        // Pred: three singletons {A}, {B}, {C}
        let gold = chains(vec![vec![("A", 0, 1), ("B", 2, 3), ("C", 4, 5)]]);
        let pred = chains(vec![
            vec![("A", 0, 1)],
            vec![("B", 2, 3)],
            vec![("C", 4, 5)],
        ]);
        let (p, r, f1) = muc_score(&pred, &gold);
        // Recall: gold chain has 3 mentions, split into 3 predicted partitions -> (3-3)/(3-1) = 0
        assert!(approx_eq(r, 0.0), "r={r}");
        // Precision: each pred chain has 1 mention, so pred chains with <=1 common mention are skipped
        assert!(approx_eq(p, 0.0), "p={p}");
        assert!(approx_eq(f1, 0.0), "f1={f1}");
    }

    #[test]
    fn muc_empty_inputs() {
        let empty: Vec<CorefChain> = vec![];
        let (p, r, f1) = muc_score(&empty, &empty);
        assert!(approx_eq(p, 0.0));
        assert!(approx_eq(r, 0.0));
        assert!(approx_eq(f1, 0.0));
    }

    #[test]
    fn muc_singleton_only() {
        // Both gold and pred have a single singleton -- MUC skips chains with <=1 common mention
        let gold = chains(vec![vec![("A", 0, 1)]]);
        let pred = gold.clone();
        let (p, r, f1) = muc_score(&pred, &gold);
        // MUC is undefined for singletons; implementation returns 0
        assert!(approx_eq(p, 0.0));
        assert!(approx_eq(r, 0.0));
        assert!(approx_eq(f1, 0.0));
    }

    #[test]
    fn muc_partial_overlap() {
        // Gold: {A, B, C}, {D, E}
        // Pred: {A, B}, {C, D, E}
        // C moved from chain 0 to chain 1, D and E correctly together
        let gold = chains(vec![
            vec![("A", 0, 1), ("B", 2, 3), ("C", 4, 5)],
            vec![("D", 6, 7), ("E", 8, 9)],
        ]);
        let pred = chains(vec![
            vec![("A", 0, 1), ("B", 2, 3)],
            vec![("C", 4, 5), ("D", 6, 7), ("E", 8, 9)],
        ]);
        let (p, r, f1) = muc_score(&pred, &gold);
        assert!(p > 0.0 && p < 1.0, "p={p} should be partial");
        assert!(r > 0.0 && r < 1.0, "r={r} should be partial");
        assert!(f1 > 0.0 && f1 < 1.0, "f1={f1} should be partial");
    }

    // =========================================================================
    // 4. b_cubed_score
    // =========================================================================

    #[test]
    fn b_cubed_perfect_prediction() {
        let gold = chains(vec![
            vec![("A", 0, 1), ("B", 2, 3)],
            vec![("C", 4, 5), ("D", 6, 7)],
        ]);
        let pred = gold.clone();
        let (p, r, f1) = b_cubed_score(&pred, &gold);
        assert!(approx_eq(p, 1.0), "p={p}");
        assert!(approx_eq(r, 1.0), "r={r}");
        assert!(approx_eq(f1, 1.0), "f1={f1}");
    }

    #[test]
    fn b_cubed_empty_inputs() {
        let empty: Vec<CorefChain> = vec![];
        let (p, r, f1) = b_cubed_score(&empty, &empty);
        assert!(approx_eq(p, 0.0));
        assert!(approx_eq(r, 0.0));
        assert!(approx_eq(f1, 0.0));
    }

    #[test]
    fn b_cubed_over_clustering() {
        // Gold: one chain {A, B, C, D}
        // Pred: two chains {A, B}, {C, D} -- over-split
        let gold = chains(vec![vec![
            ("A", 0, 1),
            ("B", 2, 3),
            ("C", 4, 5),
            ("D", 6, 7),
        ]]);
        let pred = chains(vec![
            vec![("A", 0, 1), ("B", 2, 3)],
            vec![("C", 4, 5), ("D", 6, 7)],
        ]);
        let (p, r, f1) = b_cubed_score(&pred, &gold);
        // Precision = 1.0 (each pred cluster is a subset of gold)
        assert!(approx_eq(p, 1.0), "p={p}");
        // Recall < 1.0 (gold cluster split)
        assert!(r < 1.0, "r={r} should be < 1.0");
        assert!(r > 0.0, "r={r} should be > 0.0");
        assert!(f1 < 1.0 && f1 > 0.0, "f1={f1}");
    }

    #[test]
    fn b_cubed_under_clustering() {
        // Gold: {A, B}, {C, D}
        // Pred: one chain {A, B, C, D} -- merged
        let gold = chains(vec![
            vec![("A", 0, 1), ("B", 2, 3)],
            vec![("C", 4, 5), ("D", 6, 7)],
        ]);
        let pred = chains(vec![vec![
            ("A", 0, 1),
            ("B", 2, 3),
            ("C", 4, 5),
            ("D", 6, 7),
        ]]);
        let (p, r, f1) = b_cubed_score(&pred, &gold);
        // Recall = 1.0 (every gold pair is in same pred cluster)
        assert!(approx_eq(r, 1.0), "r={r}");
        // Precision < 1.0 (pred cluster contains non-coreferent mentions)
        assert!(p < 1.0, "p={p} should be < 1.0");
        assert!(p > 0.0, "p={p} should be > 0.0");
        assert!(f1 < 1.0 && f1 > 0.0, "f1={f1}");
    }

    // =========================================================================
    // 5. ceaf_e_score
    // =========================================================================

    #[test]
    fn ceaf_e_perfect_prediction() {
        let gold = chains(vec![vec![("A", 0, 1), ("B", 2, 3)], vec![("C", 4, 5)]]);
        let pred = gold.clone();
        let (p, r, f1) = ceaf_e_score(&pred, &gold);
        assert!(approx_eq(p, 1.0), "p={p}");
        assert!(approx_eq(r, 1.0), "r={r}");
        assert!(approx_eq(f1, 1.0), "f1={f1}");
    }

    #[test]
    fn ceaf_e_empty_inputs() {
        let empty: Vec<CorefChain> = vec![];
        let (p, r, f1) = ceaf_e_score(&empty, &empty);
        assert!(approx_eq(p, 0.0));
        assert!(approx_eq(r, 0.0));
        assert!(approx_eq(f1, 0.0));
    }

    #[test]
    fn ceaf_e_no_overlap() {
        // Gold and pred have completely different mentions
        let gold = chains(vec![vec![("A", 0, 1), ("B", 2, 3)]]);
        let pred = chains(vec![vec![("X", 10, 11), ("Y", 12, 13)]]);
        let (p, r, f1) = ceaf_e_score(&pred, &gold);
        assert!(approx_eq(p, 0.0), "p={p}");
        assert!(approx_eq(r, 0.0), "r={r}");
        assert!(approx_eq(f1, 0.0), "f1={f1}");
    }

    // =========================================================================
    // 6. lea_score
    // =========================================================================

    #[test]
    fn lea_perfect_prediction() {
        let gold = chains(vec![
            vec![("A", 0, 1), ("B", 2, 3), ("C", 4, 5)],
            vec![("D", 6, 7), ("E", 8, 9)],
        ]);
        let pred = gold.clone();
        let (p, r, f1) = lea_score(&pred, &gold);
        assert!(approx_eq(p, 1.0), "p={p}");
        assert!(approx_eq(r, 1.0), "r={r}");
        assert!(approx_eq(f1, 1.0), "f1={f1}");
    }

    #[test]
    fn lea_empty_inputs() {
        let empty: Vec<CorefChain> = vec![];
        let (p, r, f1) = lea_score(&empty, &empty);
        assert!(approx_eq(p, 0.0));
        assert!(approx_eq(r, 0.0));
        assert!(approx_eq(f1, 0.0));
    }

    #[test]
    fn lea_partial_overlap() {
        let gold = chains(vec![vec![("A", 0, 1), ("B", 2, 3), ("C", 4, 5)]]);
        let pred = chains(vec![vec![("A", 0, 1), ("B", 2, 3)], vec![("C", 4, 5)]]);
        let (_p, r, f1) = lea_score(&pred, &gold);
        assert!(r < 1.0 && r > 0.0, "r={r}");
        assert!(f1 < 1.0 && f1 > 0.0, "f1={f1}");
    }

    // =========================================================================
    // 7. conll_f1
    // =========================================================================

    #[test]
    fn conll_f1_perfect() {
        let gold = chains(vec![
            vec![("A", 0, 1), ("B", 2, 3), ("C", 4, 5)],
            vec![("D", 6, 7), ("E", 8, 9)],
        ]);
        let pred = gold.clone();
        let score = conll_f1(&pred, &gold);
        assert!(approx_eq(score, 1.0), "conll_f1={score}");
    }

    #[test]
    fn conll_f1_empty() {
        let empty: Vec<CorefChain> = vec![];
        let score = conll_f1(&empty, &empty);
        assert!(approx_eq(score, 0.0), "conll_f1={score}");
    }

    #[test]
    fn conll_f1_is_average_of_three() {
        let gold = chains(vec![
            vec![("A", 0, 1), ("B", 2, 3), ("C", 4, 5)],
            vec![("D", 6, 7), ("E", 8, 9)],
        ]);
        let pred = chains(vec![
            vec![("A", 0, 1), ("B", 2, 3)],
            vec![("C", 4, 5), ("D", 6, 7), ("E", 8, 9)],
        ]);
        let score = conll_f1(&pred, &gold);
        let (_, _, muc_f) = muc_score(&pred, &gold);
        let (_, _, b3_f) = b_cubed_score(&pred, &gold);
        let (_, _, ceafe_f) = ceaf_e_score(&pred, &gold);
        let expected = (muc_f + b3_f + ceafe_f) / 3.0;
        assert!(
            approx_eq(score, expected),
            "conll_f1={score}, expected={expected}"
        );
    }

    // =========================================================================
    // 8. CorefEvaluation::compute
    // =========================================================================

    #[test]
    fn coref_evaluation_compute_perfect() {
        let gold = chains(vec![
            vec![("A", 0, 1), ("B", 2, 3)],
            vec![("C", 4, 5), ("D", 6, 7)],
        ]);
        let pred = gold.clone();
        let eval = CorefEvaluation::compute(&pred, &gold);
        assert!(approx_eq(eval.muc.f1, 1.0), "muc.f1={}", eval.muc.f1);
        assert!(approx_eq(eval.b_cubed.f1, 1.0), "b3.f1={}", eval.b_cubed.f1);
        assert!(
            approx_eq(eval.ceaf_e.f1, 1.0),
            "ceafe.f1={}",
            eval.ceaf_e.f1
        );
        assert!(approx_eq(eval.lea.f1, 1.0), "lea.f1={}", eval.lea.f1);
        assert!(approx_eq(eval.conll_f1, 1.0), "conll_f1={}", eval.conll_f1);
        assert!(eval.chain_stats.is_some());
    }

    #[test]
    fn coref_evaluation_compute_populates_all_metrics() {
        let gold = chains(vec![vec![("A", 0, 1), ("B", 2, 3), ("C", 4, 5)]]);
        // pred splits the gold chain: A+B in one chain, C alone -- partial match
        let pred = chains(vec![vec![("A", 0, 1), ("B", 2, 3)], vec![("C", 4, 5)]]);
        let eval = CorefEvaluation::compute(&pred, &gold);
        // All metrics should be populated (may not be perfect)
        assert!(eval.muc.f1.is_finite());
        assert!(eval.b_cubed.f1.is_finite());
        assert!(eval.ceaf_e.f1.is_finite());
        assert!(eval.ceaf_m.f1.is_finite());
        assert!(eval.lea.f1.is_finite());
        assert!(eval.blanc.f1.is_finite());
        assert!(eval.conll_f1.is_finite());
        // The partial prediction should produce non-zero recall for B³:
        // A and B are correctly co-referred, so recall > 0.
        assert!(
            eval.b_cubed.recall > 0.0,
            "B³ recall should be > 0 for partial prediction, got {}",
            eval.b_cubed.recall
        );
    }

    // =========================================================================
    // 9. AggregateCorefEvaluation::compute
    // =========================================================================

    #[test]
    fn aggregate_multiple_documents() {
        let gold1 = chains(vec![vec![("A", 0, 1), ("B", 2, 3)]]);
        let pred1 = gold1.clone();
        let gold2 = chains(vec![vec![("X", 0, 1), ("Y", 2, 3), ("Z", 4, 5)]]);
        let pred2 = gold2.clone();

        let pairs: Vec<(&[CorefChain], &[CorefChain])> = vec![(&pred1, &gold1), (&pred2, &gold2)];
        let agg = AggregateCorefEvaluation::compute(&pairs);

        assert_eq!(agg.num_documents, 2);
        assert_eq!(agg.per_document.len(), 2);
        // Both perfect -> mean should be perfect
        assert!(
            approx_eq(agg.mean.conll_f1, 1.0),
            "mean conll={}",
            agg.mean.conll_f1
        );
        // Std dev should be 0 for identical scores
        assert!(
            approx_eq(agg.std_dev.conll, 0.0),
            "std_dev conll={}",
            agg.std_dev.conll
        );
    }

    #[test]
    fn aggregate_empty_document_list() {
        let pairs: Vec<(&[CorefChain], &[CorefChain])> = vec![];
        let agg = AggregateCorefEvaluation::compute(&pairs);
        assert_eq!(agg.num_documents, 0);
        assert!(agg.per_document.is_empty());
    }

    // =========================================================================
    // 10. Edge cases
    // =========================================================================

    #[test]
    fn edge_overlapping_mention_spans() {
        // Mentions with overlapping spans (unusual but possible)
        // "John Smith" (0,10) and "Smith" (5,10) in the same chain
        let gold = chains(vec![vec![
            ("John Smith", 0, 10),
            ("Smith", 5, 10),
            ("he", 15, 17),
        ]]);
        let pred = gold.clone();
        let (p, r, f1) = b_cubed_score(&pred, &gold);
        assert!(approx_eq(p, 1.0), "p={p}");
        assert!(approx_eq(r, 1.0), "r={r}");
        assert!(approx_eq(f1, 1.0), "f1={f1}");
    }

    #[test]
    fn edge_singletons_only() {
        // All chains are singletons in both gold and pred
        let gold = chains(vec![
            vec![("A", 0, 1)],
            vec![("B", 2, 3)],
            vec![("C", 4, 5)],
        ]);
        let pred = gold.clone();
        // MUC is 0 for singletons
        let (_, _, muc_f) = muc_score(&pred, &gold);
        assert!(approx_eq(muc_f, 0.0));
        // B-cubed should be perfect for identical singletons
        let (p, r, f1) = b_cubed_score(&pred, &gold);
        assert!(approx_eq(p, 1.0), "b3 p={p}");
        assert!(approx_eq(r, 1.0), "b3 r={r}");
        assert!(approx_eq(f1, 1.0), "b3 f1={f1}");
    }

    #[test]
    fn edge_no_common_mentions() {
        // Pred and gold share no mention spans at all
        let gold = chains(vec![vec![("A", 0, 1), ("B", 2, 3)]]);
        let pred = chains(vec![vec![("X", 100, 101), ("Y", 102, 103)]]);
        let (p, r, f1) = muc_score(&pred, &gold);
        assert!(approx_eq(p, 0.0));
        assert!(approx_eq(r, 0.0));
        assert!(approx_eq(f1, 0.0));
        let (p, r, f1) = b_cubed_score(&pred, &gold);
        assert!(approx_eq(p, 0.0));
        assert!(approx_eq(r, 0.0));
        assert!(approx_eq(f1, 0.0));
        let (p, r, f1) = lea_score(&pred, &gold);
        assert!(approx_eq(p, 0.0));
        assert!(approx_eq(r, 0.0));
        assert!(approx_eq(f1, 0.0));
    }

    #[test]
    fn edge_pred_empty_gold_nonempty() {
        let gold = chains(vec![vec![("A", 0, 1), ("B", 2, 3)]]);
        let empty: Vec<CorefChain> = vec![];
        let (_p, _r, f1) = muc_score(&empty, &gold);
        assert!(approx_eq(f1, 0.0), "muc f1={f1}");
        let (_p, _r, f1) = b_cubed_score(&empty, &gold);
        assert!(approx_eq(f1, 0.0), "b3 f1={f1}");
        let score = conll_f1(&empty, &gold);
        assert!(approx_eq(score, 0.0));
    }

    #[test]
    fn edge_gold_empty_pred_nonempty() {
        let pred = chains(vec![vec![("A", 0, 1), ("B", 2, 3)]]);
        let empty: Vec<CorefChain> = vec![];
        let (_p, _r, f1) = muc_score(&pred, &empty);
        assert!(approx_eq(f1, 0.0), "muc f1={f1}");
        let (_p, _r, f1) = b_cubed_score(&pred, &empty);
        assert!(approx_eq(f1, 0.0), "b3 f1={f1}");
        let score = conll_f1(&pred, &empty);
        assert!(approx_eq(score, 0.0));
    }

    #[test]
    fn blanc_perfect_prediction() {
        let gold = chains(vec![
            vec![("A", 0, 1), ("B", 2, 3)],
            vec![("C", 4, 5), ("D", 6, 7)],
        ]);
        let pred = gold.clone();
        let (p, r, f1) = blanc_score(&pred, &gold);
        assert!(approx_eq(p, 1.0), "p={p}");
        assert!(approx_eq(r, 1.0), "r={r}");
        assert!(approx_eq(f1, 1.0), "f1={f1}");
    }

    /// Discriminating test: verify CEAF-e uses Dice (phi4) with entity denominators,
    /// and CEAF-m uses raw count (phi3) with mention denominators.
    ///
    /// Gold: {A,B,C,D} {E}  (5 mentions, 2 entities)
    /// Pred: {A,B} {C,D,E}  (5 mentions, 2 entities)
    ///
    /// CEAF-e (phi4 = Dice, denom = #entities):
    ///   greedy aligns gold{A,B,C,D}<->pred{A,B}: phi4 = 2*2/(4+2) = 2/3
    ///   then gold{E}<->pred{C,D,E}: phi4 = 2*1/(1+3) = 1/2
    ///   total = 7/6, P = R = 7/12 ~ 0.583
    ///
    /// CEAF-m (phi3 = raw count, denom = #mentions):
    ///   greedy aligns gold{A,B,C,D}<->pred{A,B}: phi3 = 2
    ///   then gold{E}<->pred{C,D,E}: phi3 = 1
    ///   total = 3, P = R = 3/5 = 0.6
    #[test]
    fn ceaf_e_vs_ceaf_m_differ() {
        let gold = chains(vec![
            vec![("A", 0, 1), ("B", 2, 3), ("C", 4, 5), ("D", 6, 7)],
            vec![("E", 8, 9)],
        ]);
        let pred = chains(vec![
            vec![("A", 0, 1), ("B", 2, 3)],
            vec![("C", 4, 5), ("D", 6, 7), ("E", 8, 9)],
        ]);
        let (pe, re, _) = ceaf_e_score(&pred, &gold);
        let (pm, rm, _) = ceaf_m_score(&pred, &gold);

        // CEAF-e: Dice similarity / #entities = (2/3 + 1/2) / 2 = 7/12
        let expected_e = 7.0 / 12.0;
        assert!(
            approx_eq(pe, expected_e),
            "ceaf_e P={pe} expected {expected_e}"
        );
        assert!(
            approx_eq(re, expected_e),
            "ceaf_e R={re} expected {expected_e}"
        );

        // CEAF-m: raw count / #mentions = 3/5 = 0.6
        assert!(approx_eq(pm, 0.6), "ceaf_m P={pm} expected 0.6");
        assert!(approx_eq(rm, 0.6), "ceaf_m R={rm} expected 0.6");

        // They give different values (the whole point of having two variants)
        assert!(
            (pe - pm).abs() > 0.01,
            "ceaf_e and ceaf_m should differ: e={pe}, m={pm}"
        );
    }

    #[test]
    fn ceaf_m_perfect() {
        let gold = chains(vec![vec![("A", 0, 1), ("B", 2, 3)], vec![("C", 4, 5)]]);
        let pred = gold.clone();
        let (p, r, f1) = ceaf_m_score(&pred, &gold);
        assert!(approx_eq(p, 1.0), "p={p}");
        assert!(approx_eq(r, 1.0), "r={r}");
        assert!(approx_eq(f1, 1.0), "f1={f1}");
    }

    #[test]
    fn all_metrics_symmetric_for_identical_input() {
        // When pred == gold, all metrics should give P=R=F1
        let gold = chains(vec![
            vec![("A", 0, 1), ("B", 2, 3), ("C", 4, 5)],
            vec![("D", 6, 7), ("E", 8, 9)],
            vec![("F", 10, 11)],
        ]);
        let pred = gold.clone();
        let eval = CorefEvaluation::compute(&pred, &gold);
        // For each metric, P should equal R
        assert!(
            approx_eq_loose(eval.muc.precision, eval.muc.recall),
            "muc P={} R={}",
            eval.muc.precision,
            eval.muc.recall
        );
        assert!(
            approx_eq_loose(eval.b_cubed.precision, eval.b_cubed.recall),
            "b3 P={} R={}",
            eval.b_cubed.precision,
            eval.b_cubed.recall
        );
        assert!(
            approx_eq_loose(eval.ceaf_e.precision, eval.ceaf_e.recall),
            "ceafe P={} R={}",
            eval.ceaf_e.precision,
            eval.ceaf_e.recall
        );
        assert!(
            approx_eq_loose(eval.lea.precision, eval.lea.recall),
            "lea P={} R={}",
            eval.lea.precision,
            eval.lea.recall
        );
    }

    #[test]
    fn display_format_smoke() {
        let gold = chains(vec![vec![("A", 0, 1), ("B", 2, 3)]]);
        let eval = CorefEvaluation::compute(&gold, &gold);
        let display = format!("{eval}");
        assert!(display.contains("MUC:"));
        assert!(display.contains("CoNLL:"));
    }

    // =========================================================================
    // 11. Singleton-excluded scoring (CRAC 2022-2025 primary mode)
    // =========================================================================

    #[test]
    fn singleton_excluded_scores_diverge_from_included() {
        // Scenario: 100 correctly-matched singletons + 2 two-mention chains.
        // B3 with singletons: the 100 singletons inflate accuracy.
        // B3 without singletons: only the 2 multi-mention chains matter.
        //
        // Gold: 100 singletons + {A,B} + {C,D}
        // Pred: 100 singletons + {A,C} + {B,D}  (chains are wrong)
        let mut gold_chains: Vec<Vec<(&str, usize, usize)>> = Vec::new();
        let mut pred_chains: Vec<Vec<(&str, usize, usize)>> = Vec::new();

        // 100 singletons (offset by 1000 to avoid collision)
        for i in 0..100 {
            let start = 1000 + i * 2;
            let end = start + 1;
            gold_chains.push(vec![("s", start, end)]);
            pred_chains.push(vec![("s", start, end)]);
        }

        // Two gold chains: {A(0,1), B(2,3)} and {C(4,5), D(6,7)}
        gold_chains.push(vec![("A", 0, 1), ("B", 2, 3)]);
        gold_chains.push(vec![("C", 4, 5), ("D", 6, 7)]);

        // Pred chains are wrong: {A(0,1), C(4,5)} and {B(2,3), D(6,7)}
        pred_chains.push(vec![("A", 0, 1), ("C", 4, 5)]);
        pred_chains.push(vec![("B", 2, 3), ("D", 6, 7)]);

        let gold = chains(gold_chains);
        let pred = chains(pred_chains);

        let with_singletons = CorefEvaluation::compute(&pred, &gold);
        let without_singletons = CorefEvaluation::compute_without_singletons(&pred, &gold);

        // B3 F1 with singletons should be high (singletons dominate).
        // B3 F1 without singletons should be lower (only wrong chains remain).
        assert!(
            with_singletons.b_cubed.f1 > without_singletons.b_cubed.f1,
            "B3 with singletons ({}) should be higher than without ({})",
            with_singletons.b_cubed.f1,
            without_singletons.b_cubed.f1,
        );

        // Verify without_singletons B3 < 1.0 (the multi-mention chains are wrong).
        assert!(
            without_singletons.b_cubed.f1 < 1.0,
            "B3 without singletons should be < 1.0, got {}",
            without_singletons.b_cubed.f1
        );

        // Verify without_singletons has 0 singleton count in chain_stats.
        assert_eq!(
            without_singletons
                .chain_stats
                .as_ref()
                .unwrap()
                .singleton_count,
            0
        );
    }

    // =========================================================================
    // 12. Head-match mode
    // =========================================================================

    #[test]
    fn head_match_matches_identical_heads_different_full_spans() {
        // Gold: chain with two mentions whose heads are (4,13) and (20,22).
        //   "the president" [0,13) head=[4,13)
        //   "he" [20,22) head=[20,22)
        // Pred: chain with two mentions whose heads are (4,13) and (20,22).
        //   "the former president" [0,20) head=[4,13)  <-- different full span!
        //   "he" [20,22) head=[20,22)
        //
        // Under exact-match (full span): mentions at (0,13) vs (0,20) differ,
        //   so B3 should be imperfect.
        // Under head-match: heads (4,13) match, so B3 should be perfect.
        let gold = vec![CorefChain::new(vec![
            Mention::with_head("the president", 0, 13, 4, 13),
            Mention::with_head("he", 20, 22, 20, 22),
        ])];
        let pred = vec![CorefChain::new(vec![
            Mention::with_head("the former president", 0, 20, 4, 13),
            Mention::with_head("he", 20, 22, 20, 22),
        ])];

        // Exact-match: different full spans, so common mentions miss (0,13) vs (0,20).
        let (_, _, exact_f1) = b_cubed_score(&pred, &gold);
        assert!(
            exact_f1 < 1.0,
            "exact-match B3 should be < 1.0, got {exact_f1}"
        );

        // Head-match: heads (4,13) match, so both mentions are common.
        let (_, _, head_f1) = b_cubed_score_head(&pred, &gold);
        assert!(
            approx_eq(head_f1, 1.0),
            "head-match B3 should be 1.0, got {head_f1}"
        );
    }

    // =========================================================================
    // 13. Zero-anaphor robustness: start == end with non-Zero mention_type
    // =========================================================================

    #[test]
    fn zero_span_non_zero_mention_type_handling() {
        // A mention with start == end but mention_type != Zero.
        // This can happen with annotation errors or edge cases.
        // The zero_spans() filter inside ZeroAnaphorEvaluation uses OR logic:
        //   mention_type == Zero || start == end
        // So this mention WILL be treated as a zero anaphor even though
        // its type says otherwise. This test documents that behavior.
        let m_zero_span = {
            let mut m = Mention::new("", 5, 5);
            m.mention_type = Some(MentionType::Pronominal); // NOT Zero
            m
        };
        let m_antecedent = Mention::new("John", 0, 4);

        let gold = vec![CorefChain::new(vec![
            m_antecedent.clone(),
            m_zero_span.clone(),
        ])];
        let pred = vec![CorefChain::new(vec![m_antecedent, m_zero_span])];

        let eval = ZeroAnaphorEvaluation::compute(&pred, &gold);
        // The mention with start == end IS detected as a zero anaphor
        // because of the fallback condition.
        assert!(
            eval.is_some(),
            "start == end mention should trigger zero-anaphor evaluation"
        );
        let z = eval.unwrap();
        assert!(
            z.gold_anaphors > 0,
            "gold_anaphors should be > 0 for start == end mention"
        );
        // Since pred == gold, it should be a true positive.
        assert_eq!(z.tp, 1, "should be 1 TP for correctly linked zero anaphor");
        assert_eq!(z.fp, 0, "should be 0 FP");
        assert_eq!(z.fn_, 0, "should be 0 FN");
    }

    // =========================================================================
    // Property-based tests (proptest)
    // =========================================================================

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        /// Strategy: generate a valid precision or recall value in [0.0, 1.0].
        fn unit_interval() -> impl Strategy<Value = f64> {
            (0u32..=1_000_000u32).prop_map(|n| n as f64 / 1_000_000.0)
        }

        /// Strategy: generate a non-empty vec of CorefChains with unique, non-overlapping spans.
        ///
        /// Produces 1..=max_chains chains, each with 1..=max_mentions mentions.
        /// Spans are assigned sequentially to guarantee uniqueness.
        fn arb_chains(
            max_chains: usize,
            max_mentions: usize,
        ) -> impl Strategy<Value = Vec<CorefChain>> {
            // First decide the shape: how many chains, and how many mentions per chain.
            proptest::collection::vec(1..=max_mentions, 1..=max_chains).prop_map(|sizes| {
                let mut offset = 0usize;
                sizes
                    .into_iter()
                    .map(|n| {
                        let mentions: Vec<Mention> = (0..n)
                            .map(|_| {
                                let start = offset;
                                let end = offset + 1;
                                offset += 2; // gap so spans never collide
                                Mention::new("m", start, end)
                            })
                            .collect();
                        CorefChain::new(mentions)
                    })
                    .collect()
            })
        }

        // 1. F1 range: prf1(p, r) is in [0.0, 1.0] for any p, r in [0.0, 1.0].
        proptest! {
            #[test]
            fn prop_prf1_range(p in unit_interval(), r in unit_interval()) {
                let f1 = prf1(p, r);
                prop_assert!(f1 >= 0.0, "f1={f1} < 0 for p={p}, r={r}");
                prop_assert!(f1 <= 1.0, "f1={f1} > 1 for p={p}, r={r}");
            }
        }

        // 2. Perfect score: identical pred and gold => all metrics F1 == 1.0.
        //    Requires chains with >= 2 mentions each (MUC needs multi-mention chains).
        proptest! {
            #[test]
            fn prop_perfect_score(
                chains in proptest::collection::vec(2..=5usize, 1..=4)
            ) {
                // Build chains with unique spans.
                let mut offset = 0usize;
                let built: Vec<CorefChain> = chains.iter().map(|&n| {
                    let mentions: Vec<Mention> = (0..n).map(|_| {
                        let start = offset;
                        let end = offset + 1;
                        offset += 2;
                        Mention::new("m", start, end)
                    }).collect();
                    CorefChain::new(mentions)
                }).collect();

                let eval = CorefEvaluation::compute(&built, &built);
                let eps = 1e-9;
                prop_assert!((eval.muc.f1 - 1.0).abs() < eps,
                    "MUC F1={} != 1.0", eval.muc.f1);
                prop_assert!((eval.b_cubed.f1 - 1.0).abs() < eps,
                    "B3 F1={} != 1.0", eval.b_cubed.f1);
                prop_assert!((eval.ceaf_e.f1 - 1.0).abs() < eps,
                    "CEAFe F1={} != 1.0", eval.ceaf_e.f1);
                prop_assert!((eval.lea.f1 - 1.0).abs() < eps,
                    "LEA F1={} != 1.0", eval.lea.f1);
            }
        }

        // 3. Symmetry of prf1: prf1(p, r) == prf1(r, p).
        proptest! {
            #[test]
            fn prop_prf1_symmetric(p in unit_interval(), r in unit_interval()) {
                let f1_pr = prf1(p, r);
                let f1_rp = prf1(r, p);
                prop_assert!(
                    (f1_pr - f1_rp).abs() < 1e-15,
                    "prf1({p},{r})={f1_pr} != prf1({r},{p})={f1_rp}"
                );
            }
        }

        // 4. CoNLL F1 range: always in [0.0, 1.0] for any valid input.
        proptest! {
            #[test]
            fn prop_conll_f1_range(
                pred in arb_chains(4, 4),
                gold in arb_chains(4, 4),
            ) {
                let score = conll_f1(&pred, &gold);
                prop_assert!(score >= 0.0, "conll_f1={score} < 0");
                prop_assert!(score <= 1.0 + 1e-9, "conll_f1={score} > 1");
            }
        }

        // 5. Empty clusters: empty input should not panic and returns 0.0.
        proptest! {
            #[test]
            fn prop_empty_clusters_no_panic(
                other in arb_chains(3, 3),
            ) {
                let empty: Vec<CorefChain> = vec![];

                // Empty vs empty
                let eval_ee = CorefEvaluation::compute(&empty, &empty);
                prop_assert!(eval_ee.conll_f1.is_finite());

                // Empty vs non-empty
                let eval_eo = CorefEvaluation::compute(&empty, &other);
                prop_assert!(eval_eo.conll_f1.is_finite());
                prop_assert!(eval_eo.conll_f1 >= 0.0);
                prop_assert!(eval_eo.conll_f1 <= 1.0 + 1e-9);

                // Non-empty vs empty
                let eval_oe = CorefEvaluation::compute(&other, &empty);
                prop_assert!(eval_oe.conll_f1.is_finite());
                prop_assert!(eval_oe.conll_f1 >= 0.0);
                prop_assert!(eval_oe.conll_f1 <= 1.0 + 1e-9);
            }
        }

        // 6. CEAF-e uses entity denominators: similarity / P == #entities (predicted).
        //    For identical pred=gold, similarity from phi4 self-alignment should be
        //    exactly #entities (since phi4(K, K) = 1.0 for each chain).
        //    Therefore precision = similarity / #pred_entities = 1.0,
        //    meaning similarity == #pred_entities.
        proptest! {
            #[test]
            fn prop_ceaf_e_entity_denominators(
                chains_spec in arb_chains(5, 5),
            ) {
                // When pred == gold, each chain aligns to itself with phi4 = 1.0.
                // Total similarity = number of chains.
                // Precision = similarity / #pred_entities, recall = similarity / #gold_entities.
                let (p, r, f1) = ceaf_e_score(&chains_spec, &chains_spec);

                // Both should be 1.0 (meaning denominator = #entities, not #mentions).
                prop_assert!(
                    (p - 1.0).abs() < 1e-9,
                    "CEAF-e precision should be 1.0 for identical clusters, got {p}. \
                     #chains={}, total_mentions={}",
                    chains_spec.len(),
                    chains_spec.iter().map(|c| c.len()).sum::<usize>()
                );
                prop_assert!(
                    (r - 1.0).abs() < 1e-9,
                    "CEAF-e recall should be 1.0 for identical clusters, got {r}"
                );
                prop_assert!(
                    (f1 - 1.0).abs() < 1e-9,
                    "CEAF-e F1 should be 1.0 for identical clusters, got {f1}"
                );
            }
        }

        // 7. All metrics are 1.0 for identical single-mention (singleton) clusters.
        //    This is an edge case: MUC is known to be 0/0 for singletons, so we
        //    only check B3, CEAF-e, CEAF-m, and LEA.
        proptest! {
            #[test]
            fn prop_all_metrics_one_for_identical_singletons(
                n_chains in 1usize..=8,
            ) {
                // Build n singleton chains with unique spans.
                let singletons: Vec<CorefChain> = (0..n_chains)
                    .map(|i| {
                        CorefChain::new(vec![Mention::new("m", i * 2, i * 2 + 1)])
                    })
                    .collect();

                let (_b3_p, _b3_r, b3_f1) = b_cubed_score(&singletons, &singletons);
                prop_assert!(
                    (b3_f1 - 1.0).abs() < 1e-9,
                    "B3 F1 should be 1.0 for identical singletons, got {b3_f1}"
                );

                let (_ce_p, _ce_r, ce_f1) = ceaf_e_score(&singletons, &singletons);
                prop_assert!(
                    (ce_f1 - 1.0).abs() < 1e-9,
                    "CEAF-e F1 should be 1.0 for identical singletons, got {ce_f1}"
                );

                let (_cm_p, _cm_r, cm_f1) = ceaf_m_score(&singletons, &singletons);
                prop_assert!(
                    (cm_f1 - 1.0).abs() < 1e-9,
                    "CEAF-m F1 should be 1.0 for identical singletons, got {cm_f1}"
                );

                // LEA: singletons have 0 choose 2 = 0 links, so link resolution
                // is trivially 0/0. Implementation should handle this gracefully.
                let (_lea_p, _lea_r, lea_f1) = lea_score(&singletons, &singletons);
                prop_assert!(
                    lea_f1.is_finite(),
                    "LEA F1 should be finite for identical singletons, got {lea_f1}"
                );
            }
        }
    }
}
