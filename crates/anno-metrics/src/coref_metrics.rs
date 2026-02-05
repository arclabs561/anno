//! Coreference resolution evaluation metrics.
//!
//! This module is shared by both `anno` (analysis features) and `anno-eval` (evaluation harness).
//! It is intentionally dependency-light: it relies only on `anno-core`, `serde`, and `std`.

use crate::coref::CorefChain;
use crate::types::CorefChainStats;
use anno_core::MentionType;
use std::collections::{HashMap, HashSet};

type SpanId = (usize, usize);

fn build_mention_index(chains: &[CorefChain]) -> HashMap<SpanId, usize> {
    let mut index = HashMap::new();
    for (chain_idx, chain) in chains.iter().enumerate() {
        for mention in &chain.mentions {
            index.insert(mention.span_id(), chain_idx);
        }
    }
    index
}

fn all_mention_spans(chains: &[CorefChain]) -> HashSet<SpanId> {
    chains
        .iter()
        .flat_map(|c| c.mentions.iter().map(|m| m.span_id()))
        .collect()
}

fn common_mentions(pred: &[CorefChain], gold: &[CorefChain]) -> HashSet<SpanId> {
    let pred_spans = all_mention_spans(pred);
    let gold_spans = all_mention_spans(gold);
    pred_spans.intersection(&gold_spans).copied().collect()
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
    /// Create a new score triple.
    #[must_use]
    pub fn new(precision: f64, recall: f64) -> Self {
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
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

/// MUC link-based metric.
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
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    (precision, recall, f1)
}

// =============================================================================
// B³ (Bagga & Baldwin, 1998)
// =============================================================================

/// B³ mention-based metric.
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
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    (precision, recall, f1)
}

// =============================================================================
// CEAF (Luo, 2005)
// =============================================================================

fn ceaf_phi4(pred_chain: &CorefChain, gold_chain: &CorefChain) -> f64 {
    let pred_spans: HashSet<SpanId> = pred_chain.mentions.iter().map(|m| m.span_id()).collect();
    let gold_spans: HashSet<SpanId> = gold_chain.mentions.iter().map(|m| m.span_id()).collect();
    pred_spans.intersection(&gold_spans).count() as f64
}

fn ceaf_phi3(pred_chain: &CorefChain, gold_chain: &CorefChain) -> f64 {
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

/// CEAF entity-based (phi4).
#[must_use]
pub fn ceaf_e_score(predicted: &[CorefChain], gold: &[CorefChain]) -> (f64, f64, f64) {
    let similarity = greedy_assignment(predicted, gold, ceaf_phi4);
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
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    (precision, recall, f1)
}

/// CEAF mention-based (phi3).
#[must_use]
pub fn ceaf_m_score(predicted: &[CorefChain], gold: &[CorefChain]) -> (f64, f64, f64) {
    let similarity = greedy_assignment(predicted, gold, ceaf_phi3);
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
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    (precision, recall, f1)
}

// =============================================================================
// LEA (Moosavi & Strube, 2016)
// =============================================================================

/// LEA link-based entity-aware metric.
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
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    (precision, recall, f1)
}

// =============================================================================
// BLANC (Recasens & Hovy, 2010)
// =============================================================================

/// BLANC Rand-index-style metric.
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

/// CoNLL F1 = avg(F1(MUC), F1(B³), F1(CEAFe)).
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
            &long_pred.iter().copied().cloned().collect::<Vec<_>>(),
            &long_gold.iter().copied().cloned().collect::<Vec<_>>(),
        );
        f1
    } else {
        0.0
    };

    let short_chain_f1 = if !short_pred.is_empty() || !short_gold.is_empty() {
        let (_, _, f1) = lea_score(
            &short_pred.iter().copied().cloned().collect::<Vec<_>>(),
            &short_gold.iter().copied().cloned().collect::<Vec<_>>(),
        );
        f1
    } else {
        0.0
    };

    let singleton_f1 = if !singleton_pred.is_empty() || !singleton_gold.is_empty() {
        let (_, _, f1) = lea_score(
            &singleton_pred.iter().copied().cloned().collect::<Vec<_>>(),
            &singleton_gold.iter().copied().cloned().collect::<Vec<_>>(),
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
        fn build_mention_index(chains: &[CorefChain]) -> HashMap<SpanId, usize> {
            let mut index = HashMap::new();
            for (chain_idx, chain) in chains.iter().enumerate() {
                for mention in &chain.mentions {
                    index.insert(mention.span_id(), chain_idx);
                }
            }
            index
        }

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
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

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
