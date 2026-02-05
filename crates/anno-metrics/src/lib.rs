//! Shared evaluation/analysis primitives for `anno`.
//!
//! This crate exists to avoid duplicating low-level analysis code across:
//! - `anno` (library backends + analysis features)
//! - `anno-eval` (evaluation harness, datasets, reporting)
//!
//! It depends only on `anno-core` (plus serde for serialization), so it can be used by both
//! without creating dependency cycles.

#![warn(missing_docs)]

/// Coreference types (re-exported from `anno-core`).
pub mod coref {
    pub use anno_core::core::coref::*;
}

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

/// Coreference evaluation metrics.
pub mod coref_metrics;

/// Cluster encoding and merge scoring primitives for cross-context coreference.
pub mod cluster_encoder;
