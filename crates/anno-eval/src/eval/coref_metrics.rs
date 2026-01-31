//! Coreference resolution evaluation metrics.
//!
//! # Critical Research Context (arXiv:2401.00238)
//!
//! A single CoNLL F1 score is **"uninformative, or even misleading"** (Thalken 2024).
//!
//! Why? Because metrics average over chain lengths, hiding:
//! - Long chains (main characters): Models excel here
//! - Short chains (secondary): Models struggle
//! - Singletons (isolated): Often ignored entirely
//!
//! **Recommendation**: Report per-chain-length metrics, not just CoNLL F1.
//! Use `CorefChainStats` for stratified evaluation.
//!
//! # Metric Summary
//!
//! | Metric | Focus | Key Property |
//! |--------|-------|--------------|
//! | **MUC** | Links | Ignores singletons; counts minimum links |
//! | **B³** | Mentions | Per-mention P/R; inflates with singletons |
//! | **CEAF** | Entities | Optimal alignment; entity-based |
//! | **LEA** | Links+Entities | Link-based but entity-aware |
//! | **BLANC** | Rand index | Best discriminative power; rewards non-links |
//! | **CoNLL** | Composite | Average of MUC, B³, CEAFe |
//!
//! # References
//!
//! - MUC: Vilain et al., 1995
//! - B³: Bagga & Baldwin, 1998
//! - CEAF: Luo, 2005
//! - BLANC: Recasens & Hovy, 2010
//! - LEA: Moosavi & Strube, 2016
//! - Stratified Eval: Thalken et al., 2024
//!
//! # Example
//!
//! ```rust
//! use anno::eval::coref::{Mention, CorefChain};
//! use anno::eval::coref_metrics::{muc_score, b_cubed_score, conll_f1};
//!
//! let gold = vec![
//!     CorefChain::new(vec![
//!         Mention::new("John", 0, 4),
//!         Mention::new("he", 20, 22),
//!     ]),
//! ];
//! let pred = vec![
//!     CorefChain::new(vec![
//!         Mention::new("John", 0, 4),
//!         Mention::new("he", 20, 22),
//!     ]),
//! ];
//!
//! let (p, r, f1) = muc_score(&pred, &gold);
//! assert!((f1 - 1.0).abs() < 0.001); // Perfect match
//! ```

use super::coref::CorefChain;
use anno_core::MentionType;
use std::collections::{HashMap, HashSet};

// =============================================================================
// Result Types
// =============================================================================

/// Coreference evaluation scores (precision, recall, F1).
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct CorefScores {
    /// Precision
    pub precision: f64,
    /// Recall
    pub recall: f64,
    /// F1 score
    pub f1: f64,
}

impl CorefScores {
    /// Create new scores.
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

    /// Create from tuple.
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
    /// MUC metric
    pub muc: CorefScores,
    /// B-cubed metric
    pub b_cubed: CorefScores,
    /// CEAF entity-based (phi4)
    pub ceaf_e: CorefScores,
    /// CEAF mention-based (phi3)
    pub ceaf_m: CorefScores,
    /// LEA metric
    pub lea: CorefScores,
    /// BLANC metric
    pub blanc: CorefScores,
    /// CoNLL F1 (average of MUC, B³, CEAFe)
    pub conll_f1: f64,
    /// Chain-length stratified metrics (if computed)
    pub chain_stats: Option<super::types::CorefChainStats>,
    /// Zero-anaphor (empty-node) evaluation (CorefUD-style), if zeros are present.
    pub zero_anaphor: Option<ZeroAnaphorEvaluation>,
}

/// CorefUD-style anaphor-decomposable evaluation for zero mentions.
///
/// This mirrors the CRAC/CorefUD shared task “zeros” reporting:
/// - Evaluate only zero mentions (`MentionType::Zero`) by their anaphoric behavior.
/// - A zero mention is **anaphoric** if it is NOT the first mention in its cluster
///   (i.e., there exists a preceding mention in the same cluster).
/// - True positive (tp): gold is anaphoric, prediction is anaphoric, and there exists at least
///   one **preceding mention** shared between gold and predicted clusters.
/// - Wrong linkage (wl): gold is anaphoric, prediction is anaphoric, but no shared preceding mention.
/// - False negative (fn): gold is anaphoric, but prediction is not anaphoric or missing.
/// - False positive (fp): gold is NOT anaphoric (or missing), but prediction is anaphoric.
///
/// Precision = tp / (tp + wl + fp)
/// Recall    = tp / (tp + wl + fn)
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct ZeroAnaphorEvaluation {
    /// Precision for CorefUD-style zero-anaphor scoring.
    pub precision: f64,
    /// Recall for CorefUD-style zero-anaphor scoring.
    pub recall: f64,
    /// F1 for CorefUD-style zero-anaphor scoring.
    pub f1: f64,
    /// True positives (gold anaphoric, predicted anaphoric, and a shared preceding mention exists).
    pub tp: usize,
    /// Wrong linkage (gold anaphoric and predicted anaphoric, but no shared preceding mention).
    pub wl: usize,
    /// False positives (predicted anaphoric but gold not anaphoric or missing).
    pub fp: usize,
    /// False negatives (gold anaphoric but predicted not anaphoric or missing).
    pub fn_: usize,
    /// Number of gold zero mentions that are anaphoric (have a preceding mention).
    pub gold_anaphors: usize,
    /// Number of predicted zero mentions that are anaphoric (have a preceding mention).
    pub pred_anaphors: usize,
}

impl ZeroAnaphorEvaluation {
    /// Compute CorefUD-style anaphor-decomposable evaluation for zero mentions.
    ///
    /// Returns `None` when neither gold nor predicted contains any zero/empty mentions.
    #[must_use]
    pub fn compute(predicted: &[CorefChain], gold: &[CorefChain]) -> Option<Self> {
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
            // Gold side
            let gold_chain = gold_index
                .get(&(z_start, z_end))
                .and_then(|&idx| gold.get(idx));
            let gold_pre = gold_chain
                .map(|c| preceding_spans(c, z_start))
                .unwrap_or_default();
            // If the zero mention itself has span (start==end), it will be included in preceding_spans.
            // Remove it to avoid trivial overlap.
            let mut gold_pre = gold_pre;
            gold_pre.remove(&(z_start, z_end));
            let gold_anaphoric = !gold_pre.is_empty();

            // Pred side
            let pred_chain = pred_index
                .get(&(z_start, z_end))
                .and_then(|&idx| predicted.get(idx));
            let pred_pre = pred_chain
                .map(|c| preceding_spans(c, z_start))
                .unwrap_or_default();
            let mut pred_pre = pred_pre;
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

/// Diagnostics for long-document / windowed coreference: fragmentation across windows.
///
/// This is a supplemental metric intended to catch a common long-doc failure mode:
/// a gold entity is mentioned across multiple windows, but the system splits it into
/// multiple predicted clusters because it fails to “stitch” across the boundary.
///
/// This diagnostic is intentionally simple and mention-span based (exact span match),
/// aligned with the rest of `coref_metrics.rs`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct WindowFragmentationStats {
    /// Window size (in characters) used for diagnostics.
    pub window_size: usize,
    /// Window overlap (in characters) used for diagnostics.
    pub window_overlap: usize,
    /// Number of gold chains whose mentions span 2+ windows.
    pub multiwindow_gold_chains: usize,
    /// Number of multiwindow gold chains that are fragmented in predictions.
    pub fragmented_gold_chains: usize,
    /// Number of adjacent-window boundary checks performed.
    pub boundary_checks: usize,
    /// Number of boundary checks where mentions fall into different predicted clusters.
    pub boundary_splits: usize,
    /// Number of gold mentions in multiwindow chains that are missing from predictions.
    pub missing_mentions_in_multiwindow_chains: usize,
}

impl WindowFragmentationStats {
    /// Compute fragmentation stats for one document.
    ///
    /// - Mentions are assigned to a window by their `start` offset.
    /// - Only gold chains that touch 2+ windows are counted as “multiwindow”.
    /// - A chain is “fragmented” if its mentions map to 2+ predicted clusters, or if
    ///   any mention is missing from predictions.
    #[must_use]
    pub fn compute(
        predicted: &[CorefChain],
        gold: &[CorefChain],
        window_size: usize,
        window_overlap: usize,
    ) -> Option<Self> {
        type SpanId = (usize, usize);

        if window_size == 0 {
            return None;
        }
        let step = window_size.saturating_sub(window_overlap).max(1);

        fn build_mention_index(chains: &[CorefChain]) -> HashMap<SpanId, usize> {
            let mut index = HashMap::new();
            for (chain_idx, chain) in chains.iter().enumerate() {
                for mention in &chain.mentions {
                    index.insert(mention.span_id(), chain_idx);
                }
            }
            index
        }

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

            // Fragmentation: more than one predicted cluster, or any missing mention.
            let fragmented = pred_clusters.len() > 1 || pred_clusters.contains(&None);
            if fragmented {
                stats.fragmented_gold_chains += 1;
            }

            // Boundary splits: check adjacent windows for shared predicted cluster membership.
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

                // If we have no predicted mentions in one side, consider it a split.
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

impl CorefEvaluation {
    /// Compute all metrics.
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
}

impl CorefEvaluation {
    /// Get all F1 scores as a vector (for variance analysis).
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

    /// Average F1 across all metrics (similar to CoNLL but including LEA, BLANC, CEAFm).
    #[must_use]
    pub fn average_f1(&self) -> f64 {
        let scores = self.all_f1_scores();
        scores.iter().sum::<f64>() / scores.len() as f64
    }

    /// Standard deviation of F1 scores across metrics.
    #[must_use]
    pub fn f1_std_dev(&self) -> f64 {
        let scores = self.all_f1_scores();
        let mean = self.average_f1();
        let variance: f64 =
            scores.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / scores.len() as f64;
        variance.sqrt()
    }

    /// Check if system is over-clustering (MUC recall > precision).
    #[must_use]
    pub fn is_over_clustering(&self) -> bool {
        self.muc.recall > self.muc.precision + 0.05
    }

    /// Check if system is under-clustering (MUC precision > recall).
    #[must_use]
    pub fn is_under_clustering(&self) -> bool {
        self.muc.precision > self.muc.recall + 0.05
    }

    /// Get a summary string for comparison tables.
    #[must_use]
    pub fn summary_line(&self) -> String {
        format!(
            "MUC={:.1}% B³={:.1}% CEAFe={:.1}% LEA={:.1}% BLANC={:.1}% CoNLL={:.1}%",
            self.muc.f1 * 100.0,
            self.b_cubed.f1 * 100.0,
            self.ceaf_e.f1 * 100.0,
            self.lea.f1 * 100.0,
            self.blanc.f1 * 100.0,
            self.conll_f1 * 100.0,
        )
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
            "  CEAFm:   P={:.1}%  R={:.1}%  F1={:.1}%",
            self.ceaf_m.precision * 100.0,
            self.ceaf_m.recall * 100.0,
            self.ceaf_m.f1 * 100.0
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
                z.fn_
            )?;
        }

        // Add chain-length stratification if available
        if let Some(ref stats) = self.chain_stats {
            writeln!(f, "\n  Chain-Length Stratification:")?;
            writeln!(
                f,
                "    Long chains (>10): {} chains, F1={:.1}%",
                stats.long_chain_count,
                stats.long_chain_f1 * 100.0
            )?;
            writeln!(
                f,
                "    Short chains (2-10): {} chains, F1={:.1}%",
                stats.short_chain_count,
                stats.short_chain_f1 * 100.0
            )?;
            writeln!(
                f,
                "    Singletons (1): {} chains, F1={:.1}%",
                stats.singleton_count,
                stats.singleton_f1 * 100.0
            )?;
        }

        Ok(())
    }
}

// =============================================================================
// Helper: Mention Indexing
// =============================================================================

type SpanId = (usize, usize);

/// Build mention -> chain index.
fn build_mention_index(chains: &[CorefChain]) -> HashMap<SpanId, usize> {
    let mut index = HashMap::new();
    for (chain_idx, chain) in chains.iter().enumerate() {
        for mention in &chain.mentions {
            index.insert(mention.span_id(), chain_idx);
        }
    }
    index
}

/// Get all mentions as span IDs.
fn all_mention_spans(chains: &[CorefChain]) -> HashSet<SpanId> {
    chains
        .iter()
        .flat_map(|c| c.mentions.iter().map(|m| m.span_id()))
        .collect()
}

/// Get common mentions between predicted and gold.
fn common_mentions(pred: &[CorefChain], gold: &[CorefChain]) -> HashSet<SpanId> {
    let pred_spans = all_mention_spans(pred);
    let gold_spans = all_mention_spans(gold);
    pred_spans.intersection(&gold_spans).copied().collect()
}

// =============================================================================
// MUC Score (Vilain et al., 1995)
// =============================================================================

/// MUC (Message Understanding Conference) coreference metric.
///
/// Link-based metric that counts the minimum number of links needed to partition
/// mentions into gold clusters.
///
/// Formula: `Precision = |links_predicted ∩ links_gold| / |links_predicted|`
///          `Recall = |links_predicted ∩ links_gold| / |links_gold|`
///
/// Where `links` are the minimum spanning tree edges for each cluster.
///
/// **Pros**: Simple, intuitive
/// **Cons**: Ignores singletons, can be gamed by linking all mentions
///
/// Reference: Vilain et al. (1995) "A model-theoretic coreference scoring scheme"
///
/// # Returns
/// (precision, recall, f1)
#[must_use]
pub fn muc_score(predicted: &[CorefChain], gold: &[CorefChain]) -> (f64, f64, f64) {
    // Filter to common mentions
    let common = common_mentions(predicted, gold);
    if common.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    // MUC Recall: for each gold chain, count links correctly recovered
    let (mut recall_num, mut recall_den) = (0.0, 0.0);
    let pred_index = build_mention_index(predicted);

    for gold_chain in gold {
        // Filter to common mentions
        let gold_mentions: Vec<SpanId> = gold_chain
            .mentions
            .iter()
            .map(|m| m.span_id())
            .filter(|s| common.contains(s))
            .collect();

        if gold_mentions.len() <= 1 {
            continue; // Skip singletons
        }

        // Count partitions in predicted
        let mut pred_partitions: HashSet<usize> = HashSet::new();
        for span in &gold_mentions {
            if let Some(&chain_idx) = pred_index.get(span) {
                pred_partitions.insert(chain_idx);
            }
        }

        // Recall numerator: (|gold_mentions| - |partitions|)
        // Recall denominator: (|gold_mentions| - 1)
        recall_num += (gold_mentions.len() - pred_partitions.len().max(1)) as f64;
        recall_den += (gold_mentions.len() - 1) as f64;
    }

    // MUC Precision: same calculation but swap pred/gold
    let (mut prec_num, mut prec_den) = (0.0, 0.0);
    let gold_index = build_mention_index(gold);

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
// B³ Score (Bagga & Baldwin, 1998)
// =============================================================================

/// B-cubed (B³) coreference metric.
///
/// Per-mention metric that computes precision and recall for each mention,
/// then averages across all mentions.
///
/// Formula: For mention `m` in predicted cluster `C_p` and gold cluster `C_g`:
///          `P(m) = |C_p ∩ C_g| / |C_p|`, `R(m) = |C_p ∩ C_g| / |C_g|`
///          Then average over all mentions.
///
/// **Pros**: Gives credit for partial overlap
/// **Cons**: Inflates scores when singletons present
///
/// Reference: Bagga & Baldwin (1998) "Algorithms for scoring coreference chains"
///
/// # Returns
/// (precision, recall, f1)
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
    let mut pred_count = 0;
    let mut gold_count = 0;

    // For each mention in gold
    for gold_chain in gold {
        for mention in &gold_chain.mentions {
            let span = mention.span_id();
            if !common.contains(&span) {
                continue;
            }

            gold_count += 1;

            // Find predicted chain containing this mention
            if let Some(&pred_chain_idx) = pred_index.get(&span) {
                let pred_chain = &predicted[pred_chain_idx];

                // Count overlap: mentions in both pred and gold chain
                let pred_spans: HashSet<SpanId> =
                    pred_chain.mentions.iter().map(|m| m.span_id()).collect();
                let gold_spans: HashSet<SpanId> =
                    gold_chain.mentions.iter().map(|m| m.span_id()).collect();
                let overlap = pred_spans.intersection(&gold_spans).count();

                // Recall contribution: overlap / |gold_chain|
                recall_sum += overlap as f64 / gold_chain.mentions.len() as f64;
            }
        }
    }

    // For each mention in predicted
    for pred_chain in predicted {
        for mention in &pred_chain.mentions {
            let span = mention.span_id();
            if !common.contains(&span) {
                continue;
            }

            pred_count += 1;

            // Find gold chain containing this mention
            if let Some(&gold_chain_idx) = gold_index.get(&span) {
                let gold_chain = &gold[gold_chain_idx];

                let pred_spans: HashSet<SpanId> =
                    pred_chain.mentions.iter().map(|m| m.span_id()).collect();
                let gold_spans: HashSet<SpanId> =
                    gold_chain.mentions.iter().map(|m| m.span_id()).collect();
                let overlap = pred_spans.intersection(&gold_spans).count();

                // Precision contribution: overlap / |pred_chain|
                precision_sum += overlap as f64 / pred_chain.mentions.len() as f64;
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
// CEAF Score (Luo, 2005)
// =============================================================================

/// CEAF entity-based (phi4) similarity function.
///
/// Counts mentions in both chains.
fn ceaf_phi4(pred_chain: &CorefChain, gold_chain: &CorefChain) -> f64 {
    let pred_spans: HashSet<SpanId> = pred_chain.mentions.iter().map(|m| m.span_id()).collect();
    let gold_spans: HashSet<SpanId> = gold_chain.mentions.iter().map(|m| m.span_id()).collect();
    pred_spans.intersection(&gold_spans).count() as f64
}

/// CEAF mention-based (phi3) similarity function.
///
/// Binary: 1 if chains share any mention, 0 otherwise.
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

/// Solve optimal assignment using Hungarian algorithm approximation.
///
/// For simplicity, uses greedy assignment (exact Hungarian is O(n³)).
fn greedy_assignment(
    pred: &[CorefChain],
    gold: &[CorefChain],
    sim_fn: fn(&CorefChain, &CorefChain) -> f64,
) -> f64 {
    if pred.is_empty() || gold.is_empty() {
        return 0.0;
    }

    // Build similarity matrix
    let mut similarities: Vec<(usize, usize, f64)> = Vec::new();
    for (i, p) in pred.iter().enumerate() {
        for (j, g) in gold.iter().enumerate() {
            let sim = sim_fn(p, g);
            if sim > 0.0 {
                similarities.push((i, j, sim));
            }
        }
    }

    // Sort by similarity descending
    similarities.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // Greedy assignment
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

/// CEAF entity-based (CEAFe/phi4) coreference metric.
///
/// Aligns predicted and gold chains optimally, using number of shared mentions
/// as similarity.
///
/// Formula: `φ₄(C_p, C_g) = |C_p ∩ C_g|` (number of shared mentions)
///          Optimal alignment via Hungarian algorithm, then:
///          `Precision = Σ φ₄(C_p, C_g) / Σ |C_p|`, `Recall = Σ φ₄(C_p, C_g) / Σ |C_g|`
///
/// Reference: Luo (2005) "On coreference resolution performance metrics"
///
/// # Returns
/// (precision, recall, f1)
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

/// CEAF mention-based (CEAFm/phi3) coreference metric.
///
/// # Returns
/// (precision, recall, f1)
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
// LEA Score (Moosavi & Strube, 2016)
// =============================================================================

/// LEA (Link-based Entity-Aware) coreference metric.
///
/// Computes resolution score for each entity based on correctly resolved links,
/// weighted by entity importance.
///
/// Formula: For entity `e` with `n` mentions, importance `w(e) = n(n-1)/2`:
///          `LEA(e) = (correct_links / total_links) × w(e)`
///          Then aggregate: `P = Σ LEA(e) / Σ w(e)` for predicted entities
///
/// Reference: Moosavi & Strube (2016) "Which coreference evaluation metric do you trust?"
///
/// # Returns
/// (precision, recall, f1)
#[must_use]
pub fn lea_score(predicted: &[CorefChain], gold: &[CorefChain]) -> (f64, f64, f64) {
    let common = common_mentions(predicted, gold);
    if common.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let pred_index = build_mention_index(predicted);
    let gold_index = build_mention_index(gold);

    // LEA Recall: for each gold entity, compute resolution score
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
            // Singleton: check if predicted as singleton too
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
            // Multi-mention: compute link resolution score
            let mut correct_links = 0;
            let total_links = gold_mentions.len() * (gold_mentions.len() - 1) / 2;

            for i in 0..gold_mentions.len() {
                for j in (i + 1)..gold_mentions.len() {
                    let span_i = gold_mentions[i];
                    let span_j = gold_mentions[j];

                    // Check if both are in same predicted chain
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

    // LEA Precision: same for predicted entities
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
            let mut correct_links = 0;
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
// BLANC Score (Recasens & Hovy, 2010)
// =============================================================================

/// BLANC coreference metric.
///
/// Implements Rand index for coreference. Unlike other metrics, BLANC:
/// - Rewards correct non-coreference decisions
/// - Does NOT ignore singletons
/// - Has better discriminative power
///
/// Formula: `BLANC = (Coref_F1 + NonCoref_F1) / 2`
///          Where Coref_F1 and NonCoref_F1 are F1 scores for coreference
///          and non-coreference pairs respectively (Rand index components).
///
/// Reference: Recasens & Hovy (2010) "BLANC: Implementing the Rand index for coreference evaluation"
///
/// # Returns
/// (precision, recall, f1)
#[must_use]
pub fn blanc_score(predicted: &[CorefChain], gold: &[CorefChain]) -> (f64, f64, f64) {
    let common: Vec<SpanId> = common_mentions(predicted, gold).into_iter().collect();
    if common.len() < 2 {
        // Need at least 2 mentions to have pairs
        return (1.0, 1.0, 1.0); // Perfect by definition
    }

    let pred_index = build_mention_index(predicted);
    let gold_index = build_mention_index(gold);

    // Count all pairs
    let mut coref_tp = 0; // True positive coreference
    let mut coref_fp = 0; // False positive coreference
    let mut coref_fn = 0; // False negative coreference
    let mut non_coref_tp = 0; // True positive non-coreference
    let mut non_coref_fp = 0; // False positive non-coreference
    let mut non_coref_fn = 0; // False negative non-coreference

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

    // Coreference F1
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

    // Non-coreference F1
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

    // BLANC = average of coref F1 and non-coref F1
    let precision = (coref_precision + non_coref_precision) / 2.0;
    let recall = (coref_recall + non_coref_recall) / 2.0;
    let f1 = (coref_f1 + non_coref_f1) / 2.0;

    (precision, recall, f1)
}

// =============================================================================
// CoNLL F1 (Official shared task metric)
// =============================================================================

/// CoNLL F1 score (official shared task metric).
///
/// Computes the unweighted average of MUC, B³, and CEAFe F1 scores.
/// This is the standard metric used in CoNLL-2011 and CoNLL-2012 shared tasks.
///
/// Formula: `CoNLL_F1 = (MUC_F1 + B³_F1 + CEAFe_F1) / 3`
///
/// **Note**: A single CoNLL F1 score can be "uninformative, or even misleading"
/// (Thalken 2024). Consider reporting per-chain-length metrics via `CorefChainStats`.
///
/// # Returns
/// Average F1 score
#[must_use]
pub fn conll_f1(predicted: &[CorefChain], gold: &[CorefChain]) -> f64 {
    let (_, _, muc_f1) = muc_score(predicted, gold);
    let (_, _, b3_f1) = b_cubed_score(predicted, gold);
    let (_, _, ceaf_f1) = ceaf_e_score(predicted, gold);

    (muc_f1 + b3_f1 + ceaf_f1) / 3.0
}

// =============================================================================
// Multi-Document Evaluation
// =============================================================================

/// Aggregate evaluation results across multiple documents.
#[derive(Debug, Clone, Default)]
pub struct AggregateCorefEvaluation {
    /// Per-document evaluations
    pub per_document: Vec<CorefEvaluation>,
    /// Mean scores across documents
    pub mean: CorefEvaluation,
    /// Standard deviation of scores
    pub std_dev: CorefScoreStdDev,
    /// Number of documents
    pub num_documents: usize,
}

/// Standard deviation for each metric.
#[derive(Debug, Clone, Default)]
pub struct CorefScoreStdDev {
    /// MUC F1 standard deviation
    pub muc: f64,
    /// B-cubed F1 standard deviation
    pub b_cubed: f64,
    /// CEAF-entity F1 standard deviation
    pub ceaf_e: f64,
    /// CEAF-mention F1 standard deviation
    pub ceaf_m: f64,
    /// LEA F1 standard deviation
    pub lea: f64,
    /// BLANC F1 standard deviation
    pub blanc: f64,
    /// CoNLL F1 standard deviation
    pub conll: f64,
}

impl AggregateCorefEvaluation {
    /// Compute aggregate metrics over multiple document pairs.
    ///
    /// Each pair is (predicted_chains, gold_chains).
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

        // Compute means
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
            chain_stats: None, // Aggregate doesn't compute per-document chain stats
            // Aggregate doesn't compute zero-anaphor breakdown; keep None.
            zero_anaphor: None,
        };

        // Compute standard deviations
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

    /// 95% confidence interval (mean ± z*std/sqrt(n)).
    ///
    /// Uses z = 1.96 for the 95% confidence level (two-tailed test).
    #[must_use]
    pub fn confidence_interval_95(&self) -> (f64, f64) {
        /// Z-score for 95% confidence interval (two-tailed).
        /// Corresponds to the 97.5th percentile of the standard normal distribution.
        const Z_SCORE_95: f64 = 1.96;

        let margin = Z_SCORE_95 * self.std_dev.conll / (self.num_documents as f64).sqrt();
        (
            (self.mean.conll_f1 - margin).max(0.0),
            (self.mean.conll_f1 + margin).min(1.0),
        )
    }
}

impl std::fmt::Display for AggregateCorefEvaluation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Aggregate Coreference Evaluation ({} documents):",
            self.num_documents
        )?;
        writeln!(
            f,
            "  MUC:     F1={:.1}% ± {:.1}%",
            self.mean.muc.f1 * 100.0,
            self.std_dev.muc * 100.0
        )?;
        writeln!(
            f,
            "  B³:      F1={:.1}% ± {:.1}%",
            self.mean.b_cubed.f1 * 100.0,
            self.std_dev.b_cubed * 100.0
        )?;
        writeln!(
            f,
            "  CEAFe:   F1={:.1}% ± {:.1}%",
            self.mean.ceaf_e.f1 * 100.0,
            self.std_dev.ceaf_e * 100.0
        )?;
        writeln!(
            f,
            "  LEA:     F1={:.1}% ± {:.1}%",
            self.mean.lea.f1 * 100.0,
            self.std_dev.lea * 100.0
        )?;
        writeln!(
            f,
            "  BLANC:   F1={:.1}% ± {:.1}%",
            self.mean.blanc.f1 * 100.0,
            self.std_dev.blanc * 100.0
        )?;
        let (ci_low, ci_high) = self.confidence_interval_95();
        writeln!(
            f,
            "  CoNLL:   F1={:.1}% ± {:.1}% (95% CI: {:.1}%-{:.1}%)",
            self.mean.conll_f1 * 100.0,
            self.std_dev.conll * 100.0,
            ci_low * 100.0,
            ci_high * 100.0
        )?;
        Ok(())
    }
}

/// Compute standard deviation of a slice.
fn std_dev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    variance.sqrt()
}

// =============================================================================
// Chain-Length Stratified Evaluation (arXiv:2401.00238)
// =============================================================================

/// Compute chain-length stratified metrics for coreference evaluation.
///
/// Stratifies chains by length:
/// - Long chains: >10 mentions (protagonists/main entities)
/// - Short chains: 2-10 mentions (secondary entities)
/// - Singletons: 1 mention (background entities)
///
/// # Research Context
///
/// Thalken et al. (2024) show that single CoNLL F1 hides performance differences:
/// - long chains can behave differently than short chains,
/// - and singletons can dominate counts while contributing little to narrative coherence.
///
/// # Returns
///
/// Stratified statistics with per-chain-length F1 scores.
#[must_use]
pub fn compute_chain_length_stratified(
    predicted: &[CorefChain],
    gold: &[CorefChain],
) -> super::types::CorefChainStats {
    use super::types::CorefChainStats;

    // Separate chains by length
    let mut long_chains_pred: Vec<&CorefChain> = Vec::new();
    let mut short_chains_pred: Vec<&CorefChain> = Vec::new();
    let mut singletons_pred: Vec<&CorefChain> = Vec::new();

    let mut long_chains_gold: Vec<&CorefChain> = Vec::new();
    let mut short_chains_gold: Vec<&CorefChain> = Vec::new();
    let mut singletons_gold: Vec<&CorefChain> = Vec::new();

    for chain in predicted {
        if chain.len() > 10 {
            long_chains_pred.push(chain);
        } else if chain.len() > 1 {
            short_chains_pred.push(chain);
        } else {
            singletons_pred.push(chain);
        }
    }

    for chain in gold {
        if chain.len() > 10 {
            long_chains_gold.push(chain);
        } else if chain.len() > 1 {
            short_chains_gold.push(chain);
        } else {
            singletons_gold.push(chain);
        }
    }

    // Compute F1 for each stratum using LEA (most informative for chain-level evaluation)
    let long_chain_f1 = if !long_chains_pred.is_empty() || !long_chains_gold.is_empty() {
        let (_, _, f1) = lea_score(
            &long_chains_pred
                .iter()
                .copied()
                .cloned()
                .collect::<Vec<_>>(),
            &long_chains_gold
                .iter()
                .copied()
                .cloned()
                .collect::<Vec<_>>(),
        );
        f1
    } else {
        0.0
    };

    let short_chain_f1 = if !short_chains_pred.is_empty() || !short_chains_gold.is_empty() {
        let (_, _, f1) = lea_score(
            &short_chains_pred
                .iter()
                .copied()
                .cloned()
                .collect::<Vec<_>>(),
            &short_chains_gold
                .iter()
                .copied()
                .cloned()
                .collect::<Vec<_>>(),
        );
        f1
    } else {
        0.0
    };

    let singleton_f1 = if !singletons_pred.is_empty() || !singletons_gold.is_empty() {
        let (_, _, f1) = lea_score(
            &singletons_pred.iter().copied().cloned().collect::<Vec<_>>(),
            &singletons_gold.iter().copied().cloned().collect::<Vec<_>>(),
        );
        f1
    } else {
        0.0
    };

    CorefChainStats {
        long_chain_count: long_chains_gold.len(),
        short_chain_count: short_chains_gold.len(),
        singleton_count: singletons_gold.len(),
        long_chain_f1,
        short_chain_f1,
        singleton_f1,
    }
}

// =============================================================================
// Statistical Significance Testing
// =============================================================================

/// Result of a paired significance test between two systems.
#[derive(Debug, Clone)]
pub struct SignificanceTest {
    /// System A mean score
    pub mean_a: f64,
    /// System B mean score
    pub mean_b: f64,
    /// Difference (A - B)
    pub difference: f64,
    /// Standard error of the difference
    pub std_error: f64,
    /// t-statistic
    pub t_statistic: f64,
    /// p-value (two-tailed)
    pub p_value: f64,
    /// Number of samples
    pub n: usize,
    /// Whether the difference is significant at p < 0.05
    pub significant_05: bool,
    /// Whether the difference is significant at p < 0.01
    pub significant_01: bool,
}

impl SignificanceTest {
    /// Perform paired t-test on CoNLL F1 scores.
    ///
    /// Compares system A vs system B on the same set of documents.
    ///
    /// # Arguments
    /// * `scores_a` - CoNLL F1 scores for system A
    /// * `scores_b` - CoNLL F1 scores for system B (same documents)
    ///
    /// # Returns
    /// Significance test result with p-value
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

        // Compute paired differences
        let differences: Vec<f64> = scores_a
            .iter()
            .zip(scores_b.iter())
            .map(|(a, b)| a - b)
            .collect();

        let mean_diff = differences.iter().sum::<f64>() / n as f64;
        let mean_a = scores_a.iter().sum::<f64>() / n as f64;
        let mean_b = scores_b.iter().sum::<f64>() / n as f64;

        // Standard deviation of differences
        let variance: f64 = differences
            .iter()
            .map(|&d| (d - mean_diff).powi(2))
            .sum::<f64>()
            / (n - 1) as f64;
        let std_diff = variance.sqrt();

        // Standard error
        let std_error = std_diff / (n as f64).sqrt();

        // t-statistic
        let t_stat = if std_error > 0.0 {
            mean_diff / std_error
        } else {
            0.0
        };

        // Approximate p-value using normal distribution for large n
        // For small n, this is an approximation (true t-distribution would need a table)
        let p_value = Self::approximate_p_value(t_stat.abs(), n - 1);

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

    /// Approximate two-tailed p-value for t-distribution.
    ///
    /// Uses normal approximation for df > 30, otherwise uses lookup table.
    fn approximate_p_value(t: f64, df: usize) -> f64 {
        // Critical values for common significance levels
        // For df >= 30, t-distribution ≈ normal
        let critical_05 = if df >= 30 {
            1.96
        } else {
            Self::t_critical_05(df)
        };
        let critical_01 = if df >= 30 {
            2.576
        } else {
            Self::t_critical_01(df)
        };
        let critical_001 = if df >= 30 {
            3.29
        } else {
            Self::t_critical_001(df)
        };

        // Rough p-value estimation
        if t < critical_05 {
            // p > 0.05
            0.10 + (critical_05 - t) * 0.10 // Very rough approximation
        } else if t < critical_01 {
            // 0.01 < p < 0.05
            0.05 - (t - critical_05) / (critical_01 - critical_05) * 0.04
        } else if t < critical_001 {
            // 0.001 < p < 0.01
            0.01 - (t - critical_01) / (critical_001 - critical_01) * 0.009
        } else {
            // p < 0.001
            0.001
        }
    }

    /// t critical value for p=0.05 (two-tailed).
    fn t_critical_05(df: usize) -> f64 {
        match df {
            1 => 12.71,
            2 => 4.30,
            3 => 3.18,
            4 => 2.78,
            5 => 2.57,
            6 => 2.45,
            7 => 2.36,
            8 => 2.31,
            9 => 2.26,
            10 => 2.23,
            15 => 2.13,
            20 => 2.09,
            25 => 2.06,
            _ => 2.04,
        }
    }

    /// t critical value for p=0.01 (two-tailed).
    fn t_critical_01(df: usize) -> f64 {
        match df {
            1 => 63.66,
            2 => 9.92,
            3 => 5.84,
            4 => 4.60,
            5 => 4.03,
            6 => 3.71,
            7 => 3.50,
            8 => 3.36,
            9 => 3.25,
            10 => 3.17,
            15 => 2.95,
            20 => 2.85,
            25 => 2.79,
            _ => 2.75,
        }
    }

    /// t critical value for p=0.001 (two-tailed).
    fn t_critical_001(df: usize) -> f64 {
        match df {
            1 => 636.62,
            2 => 31.60,
            3 => 12.92,
            4 => 8.61,
            5 => 6.87,
            6 => 5.96,
            7 => 5.41,
            8 => 5.04,
            9 => 4.78,
            10 => 4.59,
            15 => 4.07,
            20 => 3.85,
            25 => 3.73,
            _ => 3.65,
        }
    }

    /// Check if system A is significantly better than system B.
    #[must_use]
    pub fn a_better_than_b(&self) -> bool {
        self.significant_05 && self.difference > 0.0
    }

    /// Check if system B is significantly better than system A.
    #[must_use]
    pub fn b_better_than_a(&self) -> bool {
        self.significant_05 && self.difference < 0.0
    }
}

impl std::fmt::Display for SignificanceTest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Paired t-test (n={}):", self.n)?;
        writeln!(f, "  System A: {:.1}%", self.mean_a * 100.0)?;
        writeln!(f, "  System B: {:.1}%", self.mean_b * 100.0)?;
        writeln!(f, "  Difference: {:+.1}%", self.difference * 100.0)?;
        writeln!(f, "  t-statistic: {:.3}", self.t_statistic)?;
        writeln!(f, "  p-value: {:.4}", self.p_value)?;

        let sig = if self.significant_01 {
            "** (p < 0.01)"
        } else if self.significant_05 {
            "* (p < 0.05)"
        } else {
            "not significant"
        };
        writeln!(f, "  Significance: {}", sig)?;

        if self.a_better_than_b() {
            writeln!(f, "  Conclusion: System A is significantly better")?;
        } else if self.b_better_than_a() {
            writeln!(f, "  Conclusion: System B is significantly better")?;
        } else {
            writeln!(f, "  Conclusion: No significant difference")?;
        }

        Ok(())
    }
}

/// Compare two systems and return significance test.
///
/// # Arguments
/// * `system_a` - Evaluations for system A (one per document)
/// * `system_b` - Evaluations for system B (same documents)
///
/// # Returns
/// Paired t-test on CoNLL F1 scores
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
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::super::coref::Mention;
    use super::*;

    #[test]
    fn test_zero_anaphor_eval_tp_wl_fn_fp() {
        let john = Mention::new("John", 0, 4);
        let mary = Mention::new("Mary", 10, 14);

        // TP: gold and prediction agree on a preceding antecedent for the zero.
        let zero_tp = Mention::with_type("", 5, 5, MentionType::Zero);
        let gold_tp = vec![CorefChain::with_id(vec![john.clone(), zero_tp.clone()], 1)];
        let pred_tp = vec![CorefChain::with_id(vec![john.clone(), zero_tp.clone()], 10)];
        let tp = ZeroAnaphorEvaluation::compute(&pred_tp, &gold_tp).unwrap();
        assert_eq!(tp.tp, 1);
        assert_eq!(tp.wl, 0);
        assert_eq!(tp.fn_, 0);
        assert_eq!(tp.fp, 0);

        // WL: predicted is anaphoric but links to a different preceding antecedent.
        // Put the zero later so both John and Mary are preceding candidates.
        let zero_wl = Mention::with_type("", 20, 20, MentionType::Zero);
        let gold_wl = vec![CorefChain::with_id(vec![john.clone(), zero_wl.clone()], 1)];
        let pred_wl = vec![CorefChain::with_id(vec![mary.clone(), zero_wl.clone()], 11)];
        let wl = ZeroAnaphorEvaluation::compute(&pred_wl, &gold_wl).unwrap();
        assert_eq!(wl.tp, 0);
        assert_eq!(wl.wl, 1);
        assert_eq!(wl.fn_, 0);
        assert_eq!(wl.fp, 0);

        // FN: predicted contains the zero but treats it as non-anaphoric (singleton).
        let pred_fn = vec![CorefChain::singleton(zero_tp.clone())];
        let fn_eval = ZeroAnaphorEvaluation::compute(&pred_fn, &gold_tp).unwrap();
        assert_eq!(fn_eval.tp, 0);
        assert_eq!(fn_eval.wl, 0);
        assert_eq!(fn_eval.fn_, 1);

        // FP: gold zero is NOT anaphoric (singleton cluster), but prediction treats it as anaphoric.
        let gold_singleton_zero = vec![CorefChain::with_id(vec![zero_tp.clone()], 2)];
        let pred_fp = vec![CorefChain::with_id(vec![john.clone(), zero_tp.clone()], 12)];
        let fp_eval = ZeroAnaphorEvaluation::compute(&pred_fp, &gold_singleton_zero).unwrap();
        assert_eq!(fp_eval.tp, 0);
        assert_eq!(fp_eval.fp, 1);
    }

    #[test]
    fn test_window_fragmentation_stats_basic() {
        // window_size=10, overlap=0 => step=10.
        // Mentions at 0..1 and 20..21 are in windows 0 and 2.
        let m1 = Mention::new("A", 0, 1);
        let m2 = Mention::new("A", 20, 21);
        let gold = vec![CorefChain::with_id(vec![m1.clone(), m2.clone()], 1)];

        // Fragmented prediction: split across two clusters.
        let pred_split = vec![
            CorefChain::with_id(vec![m1.clone()], 10),
            CorefChain::with_id(vec![m2.clone()], 11),
        ];
        let s = WindowFragmentationStats::compute(&pred_split, &gold, 10, 0).unwrap();
        assert_eq!(s.multiwindow_gold_chains, 1);
        assert_eq!(s.fragmented_gold_chains, 1);
        assert_eq!(s.boundary_checks, 1);
        assert_eq!(s.boundary_splits, 1);

        // Not fragmented: same predicted cluster spans both windows.
        let pred_join = vec![CorefChain::with_id(vec![m1.clone(), m2.clone()], 10)];
        let s2 = WindowFragmentationStats::compute(&pred_join, &gold, 10, 0).unwrap();
        assert_eq!(s2.multiwindow_gold_chains, 1);
        assert_eq!(s2.fragmented_gold_chains, 0);
        assert_eq!(s2.boundary_checks, 1);
        assert_eq!(s2.boundary_splits, 0);
    }

    // =========================================================================
    // Property-Based Tests
    // =========================================================================

    /// Property: Perfect match always yields F1 = 1.0 for non-trivial inputs
    #[test]
    fn property_perfect_match_is_one() {
        let test_cases = vec![
            // Two-mention chain
            vec![make_chain(&[("a", 0, 1), ("b", 2, 3)])],
            // Multiple chains
            vec![
                make_chain(&[("a", 0, 1), ("b", 2, 3)]),
                make_chain(&[("c", 4, 5), ("d", 6, 7)]),
            ],
            // Longer chain
            vec![make_chain(&[
                ("a", 0, 1),
                ("b", 2, 3),
                ("c", 4, 5),
                ("d", 6, 7),
            ])],
        ];

        for gold in test_cases {
            let eval = CorefEvaluation::compute(&gold, &gold);
            assert!(
                (eval.conll_f1 - 1.0).abs() < 0.001,
                "Perfect match should have CoNLL F1 = 1.0, got {}",
                eval.conll_f1
            );
        }
    }

    /// Property: Scores are always in [0, 1]
    #[test]
    fn property_scores_bounded() {
        let scenarios = vec![
            // Over-clustering
            (
                vec![make_chain(&[("a", 0, 1), ("b", 2, 3), ("c", 4, 5)])],
                vec![
                    make_chain(&[("a", 0, 1)]),
                    make_chain(&[("b", 2, 3)]),
                    make_chain(&[("c", 4, 5)]),
                ],
            ),
            // Under-clustering
            (
                vec![
                    make_chain(&[("a", 0, 1)]),
                    make_chain(&[("b", 2, 3)]),
                    make_chain(&[("c", 4, 5)]),
                ],
                vec![make_chain(&[("a", 0, 1), ("b", 2, 3), ("c", 4, 5)])],
            ),
            // Partial overlap
            (
                vec![
                    make_chain(&[("a", 0, 1), ("b", 2, 3)]),
                    make_chain(&[("c", 4, 5)]),
                ],
                vec![
                    make_chain(&[("a", 0, 1)]),
                    make_chain(&[("b", 2, 3), ("c", 4, 5)]),
                ],
            ),
        ];

        for (pred, gold) in scenarios {
            let eval = CorefEvaluation::compute(&pred, &gold);

            // Check all scores are in [0, 1]
            for (name, score) in [
                ("MUC P", eval.muc.precision),
                ("MUC R", eval.muc.recall),
                ("MUC F1", eval.muc.f1),
                ("B³ P", eval.b_cubed.precision),
                ("B³ R", eval.b_cubed.recall),
                ("B³ F1", eval.b_cubed.f1),
                ("CEAFe P", eval.ceaf_e.precision),
                ("CEAFe R", eval.ceaf_e.recall),
                ("CEAFe F1", eval.ceaf_e.f1),
                ("LEA P", eval.lea.precision),
                ("LEA R", eval.lea.recall),
                ("LEA F1", eval.lea.f1),
                ("BLANC P", eval.blanc.precision),
                ("BLANC R", eval.blanc.recall),
                ("BLANC F1", eval.blanc.f1),
                ("CoNLL F1", eval.conll_f1),
            ] {
                assert!(
                    (0.0..=1.0).contains(&score),
                    "{} should be in [0, 1], got {}",
                    name,
                    score
                );
            }
        }
    }

    /// Property: F1 is harmonic mean of P and R
    #[test]
    fn property_f1_is_harmonic_mean() {
        let pred = vec![make_chain(&[("a", 0, 1), ("b", 2, 3)])];
        let gold = vec![
            make_chain(&[("a", 0, 1), ("c", 4, 5)]),
            make_chain(&[("b", 2, 3)]),
        ];

        let eval = CorefEvaluation::compute(&pred, &gold);

        // Check F1 = 2PR/(P+R) for each metric
        for (name, scores) in [
            ("MUC", eval.muc),
            ("B³", eval.b_cubed),
            ("CEAFe", eval.ceaf_e),
            ("LEA", eval.lea),
            ("BLANC", eval.blanc),
        ] {
            if scores.precision + scores.recall > 0.0 {
                let expected_f1 =
                    2.0 * scores.precision * scores.recall / (scores.precision + scores.recall);
                assert!(
                    (scores.f1 - expected_f1).abs() < 0.001,
                    "{} F1 should be harmonic mean: expected {:.4}, got {:.4}",
                    name,
                    expected_f1,
                    scores.f1
                );
            }
        }
    }

    /// Property: CoNLL F1 is average of MUC, B³, CEAFe
    #[test]
    fn property_conll_is_average() {
        let pred = vec![
            make_chain(&[("a", 0, 1), ("b", 2, 3)]),
            make_chain(&[("c", 4, 5)]),
        ];
        let gold = vec![
            make_chain(&[("a", 0, 1)]),
            make_chain(&[("b", 2, 3), ("c", 4, 5)]),
        ];

        let eval = CorefEvaluation::compute(&pred, &gold);
        let expected = (eval.muc.f1 + eval.b_cubed.f1 + eval.ceaf_e.f1) / 3.0;

        assert!(
            (eval.conll_f1 - expected).abs() < 0.001,
            "CoNLL F1 should be avg of MUC, B³, CEAFe: expected {:.4}, got {:.4}",
            expected,
            eval.conll_f1
        );
    }

    /// Property: Symmetric scenarios should have equal over/under clustering scores
    #[test]
    fn property_symmetric_clustering_errors() {
        // Over-clustering: 3 singletons merged into 1
        let gold_over = vec![
            make_chain(&[("a", 0, 1)]),
            make_chain(&[("b", 2, 3)]),
            make_chain(&[("c", 4, 5)]),
        ];
        let pred_over = vec![make_chain(&[("a", 0, 1), ("b", 2, 3), ("c", 4, 5)])];

        // Under-clustering: 1 chain split into 3 singletons
        let gold_under = vec![make_chain(&[("a", 0, 1), ("b", 2, 3), ("c", 4, 5)])];
        let pred_under = vec![
            make_chain(&[("a", 0, 1)]),
            make_chain(&[("b", 2, 3)]),
            make_chain(&[("c", 4, 5)]),
        ];

        let eval_over = CorefEvaluation::compute(&pred_over, &gold_over);
        let eval_under = CorefEvaluation::compute(&pred_under, &gold_under);

        // B³ and CEAFe should be symmetric
        assert!(
            (eval_over.b_cubed.f1 - eval_under.b_cubed.f1).abs() < 0.001,
            "B³ should be symmetric: over={:.4}, under={:.4}",
            eval_over.b_cubed.f1,
            eval_under.b_cubed.f1
        );

        assert!(
            (eval_over.ceaf_e.f1 - eval_under.ceaf_e.f1).abs() < 0.001,
            "CEAFe should be symmetric: over={:.4}, under={:.4}",
            eval_over.ceaf_e.f1,
            eval_under.ceaf_e.f1
        );
    }

    // =========================================================================
    // Regular Tests
    // =========================================================================

    fn make_chain(mentions: &[(&str, usize, usize)]) -> CorefChain {
        CorefChain::new(
            mentions
                .iter()
                .map(|(text, start, end)| Mention::new(*text, *start, *end))
                .collect(),
        )
    }

    #[test]
    fn test_perfect_match() {
        let gold = vec![
            make_chain(&[("John", 0, 4), ("he", 20, 22), ("him", 40, 43)]),
            make_chain(&[("Mary", 5, 9), ("she", 25, 28)]),
        ];
        let pred = gold.clone();

        let (_, _, f1) = muc_score(&pred, &gold);
        assert!((f1 - 1.0).abs() < 0.001, "MUC F1 should be 1.0, got {}", f1);

        let (_, _, f1) = b_cubed_score(&pred, &gold);
        assert!((f1 - 1.0).abs() < 0.001, "B³ F1 should be 1.0, got {}", f1);

        let (_, _, f1) = ceaf_e_score(&pred, &gold);
        assert!(
            (f1 - 1.0).abs() < 0.001,
            "CEAFe F1 should be 1.0, got {}",
            f1
        );

        let (_, _, f1) = lea_score(&pred, &gold);
        assert!((f1 - 1.0).abs() < 0.001, "LEA F1 should be 1.0, got {}", f1);

        let (_, _, f1) = blanc_score(&pred, &gold);
        assert!(
            (f1 - 1.0).abs() < 0.001,
            "BLANC F1 should be 1.0, got {}",
            f1
        );

        let conll = conll_f1(&pred, &gold);
        assert!(
            (conll - 1.0).abs() < 0.001,
            "CoNLL F1 should be 1.0, got {}",
            conll
        );
    }

    #[test]
    fn test_no_overlap() {
        let gold = vec![make_chain(&[("John", 0, 4), ("he", 20, 22)])];
        let pred = vec![make_chain(&[("Mary", 5, 9), ("she", 25, 28)])];

        // No common mentions -> all metrics should be 0
        let (_, _, muc_f1) = muc_score(&pred, &gold);
        assert!(muc_f1.abs() < 0.001, "MUC F1 should be 0, got {}", muc_f1);

        let (_, _, b3_f1) = b_cubed_score(&pred, &gold);
        assert!(b3_f1.abs() < 0.001, "B³ F1 should be 0, got {}", b3_f1);
    }

    #[test]
    fn test_partial_match() {
        // Gold: [[John, he, him]]
        // Pred: [[John, he], [him]]  <- split one chain into two
        let gold = vec![make_chain(&[
            ("John", 0, 4),
            ("he", 20, 22),
            ("him", 40, 43),
        ])];
        let pred = vec![
            make_chain(&[("John", 0, 4), ("he", 20, 22)]),
            make_chain(&[("him", 40, 43)]),
        ];

        let (_, _, muc_f1) = muc_score(&pred, &gold);
        // MUC should give partial credit
        assert!(
            muc_f1 > 0.0 && muc_f1 < 1.0,
            "MUC F1 should be partial, got {}",
            muc_f1
        );

        let (_, _, b3_f1) = b_cubed_score(&pred, &gold);
        assert!(
            b3_f1 > 0.0 && b3_f1 < 1.0,
            "B³ F1 should be partial, got {}",
            b3_f1
        );
    }

    #[test]
    fn test_singleton_handling() {
        // MUC ignores singletons
        let gold = vec![
            make_chain(&[("John", 0, 4)]), // Singleton
            make_chain(&[("Mary", 5, 9), ("she", 25, 28)]),
        ];
        let pred = gold.clone();

        // B³ and BLANC should give credit for singletons
        let (_, _, b3_f1) = b_cubed_score(&pred, &gold);
        assert!(
            (b3_f1 - 1.0).abs() < 0.001,
            "B³ should be 1.0 with singletons"
        );

        let (_, _, blanc_f1) = blanc_score(&pred, &gold);
        assert!(
            (blanc_f1 - 1.0).abs() < 0.001,
            "BLANC should be 1.0 with singletons"
        );
    }

    #[test]
    fn test_coref_evaluation_display() {
        let gold = vec![make_chain(&[("John", 0, 4), ("he", 20, 22)])];
        let pred = gold.clone();

        let eval = CorefEvaluation::compute(&pred, &gold);
        let display = format!("{}", eval);

        assert!(display.contains("MUC"));
        assert!(display.contains("B³"));
        assert!(display.contains("CEAFe"));
        assert!(display.contains("CoNLL"));
    }

    #[test]
    fn test_empty_chains() {
        let gold: Vec<CorefChain> = vec![];
        let pred: Vec<CorefChain> = vec![];

        let (_, _, f1) = muc_score(&pred, &gold);
        assert!(f1.abs() < 0.001 || !f1.is_nan());
    }

    #[test]
    fn test_over_clustering() {
        // Gold: [[a], [b], [c]]  - all singletons
        // Pred: [[a, b, c]]      - all in one cluster
        let gold = vec![
            make_chain(&[("a", 0, 1)]),
            make_chain(&[("b", 2, 3)]),
            make_chain(&[("c", 4, 5)]),
        ];
        let pred = vec![make_chain(&[("a", 0, 1), ("b", 2, 3), ("c", 4, 5)])];

        // MUC gives high recall for over-clustering (known flaw)
        let _ = muc_score(&pred, &gold);
        // MUC skips singletons, so this is edge case

        // BLANC should penalize this
        let (_, _, blanc_f1) = blanc_score(&pred, &gold);
        assert!(
            blanc_f1 < 0.5,
            "BLANC should penalize over-clustering, got {}",
            blanc_f1
        );
    }

    #[test]
    fn test_under_clustering() {
        // Gold: [[a, b, c]]      - all in one cluster
        // Pred: [[a], [b], [c]]  - all singletons
        let gold = vec![make_chain(&[("a", 0, 1), ("b", 2, 3), ("c", 4, 5)])];
        let pred = vec![
            make_chain(&[("a", 0, 1)]),
            make_chain(&[("b", 2, 3)]),
            make_chain(&[("c", 4, 5)]),
        ];

        // MUC recall should be 0 (no links recovered)
        let (_, r, _) = muc_score(&pred, &gold);
        assert!(
            r.abs() < 0.001,
            "MUC recall should be 0 for under-clustering, got {}",
            r
        );

        // BLANC should also penalize
        let (_, _, blanc_f1) = blanc_score(&pred, &gold);
        assert!(
            blanc_f1 < 0.5,
            "BLANC should penalize under-clustering, got {}",
            blanc_f1
        );
    }

    // =========================================================================
    // Proptest-Based Property Tests
    // =========================================================================

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        /// Generate a random mention (text, start, end)
        fn arb_mention() -> impl Strategy<Value = Mention> {
            (0usize..100, 1usize..20).prop_map(|(start, len)| {
                let text = format!("m{}_{}", start, len);
                Mention::new(text, start, start + len)
            })
        }

        /// Generate a chain with 1-5 mentions
        fn arb_chain() -> impl Strategy<Value = CorefChain> {
            proptest::collection::vec(arb_mention(), 1..5).prop_map(CorefChain::new)
        }

        /// Generate a clustering with 1-4 chains
        fn arb_clustering() -> impl Strategy<Value = Vec<CorefChain>> {
            proptest::collection::vec(arb_chain(), 1..4)
        }

        /// Generate a clustering with unique mentions (no overlap across chains)
        fn arb_unique_clustering() -> impl Strategy<Value = Vec<CorefChain>> {
            // Generate unique chain of unique mentions
            (1usize..4)
                .prop_flat_map(|num_chains| {
                    proptest::collection::vec(
                        proptest::collection::vec(1usize..20, 1..5),
                        num_chains..=num_chains,
                    )
                })
                .prop_map(|chain_lens| {
                    let mut offset = 0usize;
                    chain_lens
                        .into_iter()
                        .map(|lens| {
                            let mentions: Vec<_> = lens
                                .iter()
                                .map(|&len| {
                                    let m =
                                        Mention::new(format!("m{}", offset), offset, offset + len);
                                    offset += len + 10; // Ensure no overlap
                                    m
                                })
                                .collect();
                            CorefChain::new(mentions)
                        })
                        .collect()
                })
        }

        proptest! {
            /// All metrics should be bounded in [0, 1]
            #[test]
            fn prop_metrics_bounded(pred in arb_clustering(), gold in arb_clustering()) {
                let eval = CorefEvaluation::compute(&pred, &gold);

                // Check all scores are in [0, 1]
                for score in [
                    eval.muc.precision, eval.muc.recall, eval.muc.f1,
                    eval.b_cubed.precision, eval.b_cubed.recall, eval.b_cubed.f1,
                    eval.ceaf_e.precision, eval.ceaf_e.recall, eval.ceaf_e.f1,
                    eval.lea.precision, eval.lea.recall, eval.lea.f1,
                    eval.blanc.precision, eval.blanc.recall, eval.blanc.f1,
                    eval.conll_f1,
                ] {
                    prop_assert!(
                        (0.0..=1.0).contains(&score),
                        "Score {} out of bounds [0, 1]", score
                    );
                }
            }

            /// F1 should be harmonic mean of precision and recall
            #[test]
            fn prop_f1_harmonic_mean(pred in arb_clustering(), gold in arb_clustering()) {
                let eval = CorefEvaluation::compute(&pred, &gold);

                for (name, scores) in [
                    ("MUC", eval.muc),
                    ("B³", eval.b_cubed),
                    ("CEAFe", eval.ceaf_e),
                    ("LEA", eval.lea),
                    ("BLANC", eval.blanc),
                ] {
                    if scores.precision + scores.recall > 1e-10 {
                        let expected = 2.0 * scores.precision * scores.recall
                            / (scores.precision + scores.recall);
                        prop_assert!(
                            (scores.f1 - expected).abs() < 0.001,
                            "{} F1 should be harmonic mean: expected {:.4}, got {:.4}",
                            name, expected, scores.f1
                        );
                    } else {
                        // If P + R = 0, F1 should be 0
                        prop_assert!(
                            scores.f1.abs() < 0.001,
                            "{} F1 should be 0 when P+R=0, got {}", name, scores.f1
                        );
                    }
                }
            }

            /// CoNLL F1 is average of MUC, B³, CEAFe
            #[test]
            fn prop_conll_is_average(pred in arb_clustering(), gold in arb_clustering()) {
                let eval = CorefEvaluation::compute(&pred, &gold);
                let expected = (eval.muc.f1 + eval.b_cubed.f1 + eval.ceaf_e.f1) / 3.0;

                prop_assert!(
                    (eval.conll_f1 - expected).abs() < 0.001,
                    "CoNLL F1 should be (MUC + B³ + CEAFe)/3: expected {:.4}, got {:.4}",
                    expected, eval.conll_f1
                );
            }

            /// Perfect match should always yield F1 = 1.0
            #[test]
            fn prop_perfect_match_one(chains in arb_unique_clustering()) {
                // Filter to non-trivial chains (at least one non-singleton)
                let has_non_singleton = chains.iter().any(|c| c.mentions.len() > 1);

                if has_non_singleton {
                    let eval = CorefEvaluation::compute(&chains, &chains);

                    // All F1 scores should be 1.0
                    prop_assert!(
                        (eval.muc.f1 - 1.0).abs() < 0.001,
                        "MUC F1 for perfect match should be 1.0, got {}", eval.muc.f1
                    );
                    prop_assert!(
                        (eval.b_cubed.f1 - 1.0).abs() < 0.001,
                        "B³ F1 for perfect match should be 1.0, got {}", eval.b_cubed.f1
                    );
                    prop_assert!(
                        (eval.ceaf_e.f1 - 1.0).abs() < 0.001,
                        "CEAFe F1 for perfect match should be 1.0, got {}", eval.ceaf_e.f1
                    );
                    prop_assert!(
                        (eval.conll_f1 - 1.0).abs() < 0.001,
                        "CoNLL F1 for perfect match should be 1.0, got {}", eval.conll_f1
                    );
                }
            }

            /// Empty input handling: no crashes, scores should be 0 or NaN-safe
            #[test]
            fn prop_empty_handling(gold in arb_clustering()) {
                // Empty predictions
                let eval = CorefEvaluation::compute(&[], &gold);
                prop_assert!(eval.conll_f1.is_finite(), "Empty pred should not produce NaN");

                // Empty gold
                let eval = CorefEvaluation::compute(&gold, &[]);
                prop_assert!(eval.conll_f1.is_finite(), "Empty gold should not produce NaN");
            }
        }
    }

    // =========================================================================
    // Edge Case Tests (arXiv:2401.00238 recommendations)
    // =========================================================================

    /// Test: All singletons (single-mention chains)
    /// Per arXiv:2401.00238, models often fail on short chains
    #[test]
    fn edge_case_all_singletons() {
        let gold = vec![
            make_chain(&[("a", 0, 1)]),
            make_chain(&[("b", 2, 3)]),
            make_chain(&[("c", 4, 5)]),
        ];
        let pred = gold.clone();

        let eval = CorefEvaluation::compute(&pred, &gold);

        // MUC ignores singletons, so should be undefined/0
        // B³ should be 1.0 for perfect singleton match
        assert!(
            eval.b_cubed.f1 >= 0.99,
            "B³ should be 1.0 for perfect singleton match, got {}",
            eval.b_cubed.f1
        );
    }

    /// Test: Mixed singleton and multi-mention chains
    #[test]
    fn edge_case_mixed_singletons() {
        let gold = vec![
            make_chain(&[("john", 0, 4), ("he", 10, 12), ("him", 20, 23)]),
            make_chain(&[("mary", 5, 9)]),  // singleton
            make_chain(&[("bob", 30, 33)]), // singleton
        ];

        // Prediction merges all singletons incorrectly
        let pred = vec![
            make_chain(&[("john", 0, 4), ("he", 10, 12), ("him", 20, 23)]),
            make_chain(&[("mary", 5, 9), ("bob", 30, 33)]), // wrong merge
        ];

        let eval = CorefEvaluation::compute(&pred, &gold);

        // Should detect the error
        assert!(
            eval.conll_f1 < 1.0,
            "Merging singletons incorrectly should reduce F1"
        );
    }

    /// Test: Very long chain (main character in novel)
    #[test]
    fn edge_case_long_chain() {
        // Simulate a main character mentioned 20 times
        let mentions: Vec<_> = (0..20)
            .map(|i| (format!("mention_{}", i), i * 10, i * 10 + 5))
            .collect();
        let mention_refs: Vec<_> = mentions
            .iter()
            .map(|(t, s, e)| (t.as_str(), *s, *e))
            .collect();

        let gold = vec![make_chain(&mention_refs)];
        let pred = gold.clone();

        let eval = CorefEvaluation::compute(&pred, &gold);
        assert!(
            (eval.conll_f1 - 1.0).abs() < 0.001,
            "Long chain perfect match should be 1.0"
        );
    }

    /// Test: Splitting a long chain in half
    #[test]
    fn edge_case_split_long_chain() {
        let mentions: Vec<_> = (0..10)
            .map(|i| (format!("m{}", i), i * 10, i * 10 + 5))
            .collect();
        let mention_refs: Vec<_> = mentions
            .iter()
            .map(|(t, s, e)| (t.as_str(), *s, *e))
            .collect();

        let gold = vec![make_chain(&mention_refs)];

        // Split into two chains
        let first_half: Vec<_> = mention_refs[..5].to_vec();
        let second_half: Vec<_> = mention_refs[5..].to_vec();
        let pred = vec![make_chain(&first_half), make_chain(&second_half)];

        let eval = CorefEvaluation::compute(&pred, &gold);

        // Should penalize splitting
        assert!(eval.muc.f1 < 1.0, "Splitting chain should reduce MUC F1");
        assert!(eval.lea.f1 < 1.0, "Splitting chain should reduce LEA F1");
    }

    /// Test: Complete disjoint (no overlap between pred and gold)
    #[test]
    fn edge_case_complete_disjoint() {
        let gold = vec![make_chain(&[("a", 0, 1), ("b", 2, 3)])];
        let pred = vec![make_chain(&[("c", 4, 5), ("d", 6, 7)])];

        let eval = CorefEvaluation::compute(&pred, &gold);

        // No mention overlap means all metrics should reflect this
        // Scores depend on how metrics handle non-overlapping mentions
        assert!(
            eval.conll_f1.is_finite(),
            "Disjoint sets should not produce NaN"
        );
    }

    /// Test: Duplicate mentions in chain (edge case)
    #[test]
    fn edge_case_duplicate_mentions() {
        // Same span mentioned twice (shouldn't happen but should handle gracefully)
        let gold = vec![make_chain(&[("a", 0, 1), ("b", 2, 3)])];
        let pred = vec![make_chain(&[("a", 0, 1), ("a", 0, 1), ("b", 2, 3)])];

        let eval = CorefEvaluation::compute(&pred, &gold);

        // Should not crash
        assert!(
            eval.conll_f1.is_finite(),
            "Duplicate mentions should not produce NaN"
        );
    }

    /// Test: Single two-mention chain (minimal non-trivial case)
    #[test]
    fn edge_case_minimal_chain() {
        let gold = vec![make_chain(&[("john", 0, 4), ("he", 10, 12)])];
        let pred = gold.clone();

        let eval = CorefEvaluation::compute(&pred, &gold);

        assert!(
            (eval.muc.f1 - 1.0).abs() < 0.001,
            "Minimal chain perfect match MUC should be 1.0"
        );
        assert!(
            (eval.b_cubed.f1 - 1.0).abs() < 0.001,
            "Minimal chain perfect match B³ should be 1.0"
        );
    }

    /// Test: Over-clustering (too few chains) - MUC recall > precision
    #[test]
    fn edge_case_over_clustering() {
        // Use multi-mention chains to trigger MUC (MUC ignores singletons)
        let gold = vec![
            make_chain(&[("a1", 0, 2), ("a2", 3, 5)]),
            make_chain(&[("b1", 10, 12), ("b2", 13, 15)]),
            make_chain(&[("c1", 20, 22), ("c2", 23, 25)]),
        ];
        // Merge all into one chain (over-clustering)
        let pred = vec![make_chain(&[
            ("a1", 0, 2),
            ("a2", 3, 5),
            ("b1", 10, 12),
            ("b2", 13, 15),
            ("c1", 20, 22),
            ("c2", 23, 25),
        ])];

        let eval = CorefEvaluation::compute(&pred, &gold);

        // Over-clustering: high recall (found all links) but low precision (spurious links)
        assert!(
            eval.muc.recall > eval.muc.precision,
            "Over-clustering should have MUC recall > precision: R={:.3} P={:.3}",
            eval.muc.recall,
            eval.muc.precision
        );
    }

    /// Test: Under-clustering (too many chains) - MUC precision > recall
    #[test]
    fn edge_case_under_clustering() {
        // One long chain
        let gold = vec![make_chain(&[
            ("a", 0, 1),
            ("b", 2, 3),
            ("c", 4, 5),
            ("d", 6, 7),
            ("e", 8, 9),
            ("f", 10, 11),
        ])];
        // Split into many chains (under-clustering)
        let pred = vec![
            make_chain(&[("a", 0, 1), ("b", 2, 3)]),
            make_chain(&[("c", 4, 5), ("d", 6, 7)]),
            make_chain(&[("e", 8, 9), ("f", 10, 11)]),
        ];

        let eval = CorefEvaluation::compute(&pred, &gold);

        // Under-clustering: high precision (links made are correct) but low recall (missing links)
        assert!(
            eval.muc.precision > eval.muc.recall,
            "Under-clustering should have MUC precision > recall: P={:.3} R={:.3}",
            eval.muc.precision,
            eval.muc.recall
        );
    }

    /// Test: Stratified evaluation by chain length
    #[test]
    fn stratified_by_chain_length() {
        // Gold has chains of different lengths
        let gold = vec![
            make_chain(&[
                ("main", 0, 4),
                ("protagonist", 10, 21),
                ("hero", 30, 34),
                ("she", 40, 43),
            ]),
            make_chain(&[("sidekick", 50, 58), ("friend", 60, 66)]),
            make_chain(&[("villain", 70, 77)]), // singleton
        ];

        // Pred gets long chain right, short chain wrong
        let pred = vec![
            make_chain(&[
                ("main", 0, 4),
                ("protagonist", 10, 21),
                ("hero", 30, 34),
                ("she", 40, 43),
            ]),
            make_chain(&[("sidekick", 50, 58)]), // split
            make_chain(&[("friend", 60, 66)]),   // split
            make_chain(&[("villain", 70, 77)]),
        ];

        let eval = CorefEvaluation::compute(&pred, &gold);

        // Overall F1 hides the problem with short chains
        assert!(
            eval.conll_f1 > 0.5,
            "CoNLL F1 should be reasonable due to long chain success"
        );

        // But if we had per-length stats, short chains would show worse performance
        // This test validates the setup for stratified analysis
    }

    /// Test: Empty chains in input (should be filtered/handled)
    #[test]
    fn edge_case_empty_chains() {
        let gold = vec![
            make_chain(&[("a", 0, 1), ("b", 2, 3)]),
            CorefChain::new(vec![]), // empty chain
        ];
        let pred = vec![make_chain(&[("a", 0, 1), ("b", 2, 3)])];

        let eval = CorefEvaluation::compute(&pred, &gold);

        // Should handle empty chains gracefully
        assert!(
            eval.conll_f1.is_finite(),
            "Empty chains should not produce NaN"
        );
    }

    /// Test: Zero-length mentions (degenerate case)
    #[test]
    fn edge_case_zero_length_mentions() {
        let gold = vec![make_chain(&[("", 0, 0), ("b", 2, 3)])];
        let pred = gold.clone();

        let eval = CorefEvaluation::compute(&pred, &gold);

        // Should handle gracefully
        assert!(
            eval.conll_f1.is_finite(),
            "Zero-length mentions should not produce NaN"
        );
    }

    /// Test: Overlapping mention spans (nested entities)
    #[test]
    fn edge_case_overlapping_spans() {
        // "New York City" contains "York"
        let gold = vec![
            make_chain(&[("new york city", 0, 13), ("nyc", 20, 23)]),
            make_chain(&[("york", 4, 8)]), // overlaps with above
        ];
        let pred = gold.clone();

        let eval = CorefEvaluation::compute(&pred, &gold);

        assert!(
            (eval.conll_f1 - 1.0).abs() < 0.001,
            "Overlapping spans with perfect match should be 1.0"
        );
    }

    /// Test: Unicode mentions
    #[test]
    fn edge_case_unicode_mentions() {
        let gold = vec![
            make_chain(&[("習近平", 0, 9), ("他", 20, 26)]),
            make_chain(&[("Müller", 30, 36), ("er", 40, 42)]),
        ];
        let pred = gold.clone();

        let eval = CorefEvaluation::compute(&pred, &gold);

        assert!(
            (eval.conll_f1 - 1.0).abs() < 0.001,
            "Unicode mentions should work correctly"
        );
    }

    /// Test: Metrics disagree (MUC vs B³ vs CEAF)
    #[test]
    fn metrics_disagreement_scenario() {
        // Scenario where metrics give different signals
        let gold = vec![
            make_chain(&[("a", 0, 1), ("b", 2, 3)]),
            make_chain(&[("c", 4, 5)]),
        ];

        // Pred adds singleton to first chain
        let pred = vec![make_chain(&[("a", 0, 1), ("b", 2, 3), ("c", 4, 5)])];

        let eval = CorefEvaluation::compute(&pred, &gold);

        // MUC focuses on links - adding c doesn't help
        // B³ penalizes the incorrect merge
        // Check that metrics give sensible (possibly different) values
        assert!(eval.muc.f1.is_finite());
        assert!(eval.b_cubed.f1.is_finite());
        assert!(eval.ceaf_e.f1.is_finite());

        // B³ and CEAF should penalize the incorrect merge more than MUC
        // (MUC doesn't care about singletons)
    }
}
