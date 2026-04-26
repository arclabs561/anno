//! Evaluation/analysis primitives for `anno`.
//!
//! This module exists to support analysis-oriented features inside the `anno` crate (e.g.
//! cross-context cluster merging and coreference metric computation) without pulling in the full
//! dataset/harness machinery from `anno-eval`.
//!
//! Notes:
//! - This is intentionally **not** the full evaluation framework. The full harness, datasets, and
//!   reports live in the `anno-eval` crate.
//! - The legacy feature name is `eval`; prefer enabling `analysis` (an alias).

/// Coreference types (re-exported from `anno-core`).
pub mod coref;

/// Small shared analysis structs.
pub mod types {
    use serde::{Deserialize, Serialize};

    /// Chain-length stratified statistics for coreference evaluation.
    #[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
    pub struct CorefChainStats {
        /// Number of long chains (>10 mentions).
        pub long_chain_count: usize,
        /// Number of short chains (2-10 mentions).
        pub short_chain_count: usize,
        /// Number of singletons (1 mention).
        pub singleton_count: usize,
        /// F1 score on long chains only.
        pub long_chain_f1: f64,
        /// F1 score on short chains only.
        pub short_chain_f1: f64,
        /// F1 score on singletons (if evaluated).
        pub singleton_f1: f64,
    }

    impl CorefChainStats {
        /// Total chain count.
        #[must_use]
        pub fn total_chains(&self) -> usize {
            self.long_chain_count + self.short_chain_count + self.singleton_count
        }

        /// Weighted F1 (by chain count).
        ///
        /// Note: this is **not** CoNLL F1; it is a diagnostic aggregation over chain strata.
        #[must_use]
        pub fn weighted_f1(&self) -> f64 {
            let total = self.total_chains();
            if total == 0 {
                return 0.0;
            }

            let weighted_sum = self.long_chain_f1 * self.long_chain_count as f64
                + self.short_chain_f1 * self.short_chain_count as f64
                + self.singleton_f1 * self.singleton_count as f64;
            weighted_sum / total as f64
        }
    }
}

/// Coreference evaluation metrics (MUC, B³, CEAF, LEA, BLANC, CoNLL F1).
pub mod coref_metrics;

/// Cross-context cluster encoding and merge scoring primitives.
pub mod cluster_encoder;

/// Simple coreference resolvers used by analysis/evaluation paths.
pub mod coref_resolver;

pub use cluster_encoder::{
    ClusterEmbedding, ClusterEncoder, ClusterMention, CosineMergeScorer, CrossContextConfig,
    HeuristicClusterEncoder, LocalCluster, MergeScorer, MergedCluster,
};

pub use coref_metrics::{
    b_cubed_score, blanc_score, ceaf_e_score, ceaf_m_score, conll_f1, lea_score, muc_score,
    CorefEvaluation, CorefScores, ZeroAnaphorEvaluation,
};

pub use coref_resolver::{CorefConfig, SimpleCorefResolver};
