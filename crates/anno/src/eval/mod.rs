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
    pub use anno_metrics::types::*;
}

/// Coreference evaluation metrics (MUC, B³, CEAF, LEA, BLANC, CoNLL F1).
pub mod coref_metrics {
    pub use anno_metrics::coref_metrics::*;
}

/// Cross-context cluster encoding and merge scoring primitives.
pub mod cluster_encoder {
    pub use anno_metrics::cluster_encoder::*;
}

/// Simple coreference resolvers used by analysis/evaluation paths.
pub mod coref_resolver;

pub use cluster_encoder::{
    ClusterEmbedding, ClusterEncoder, ClusterMention, CrossContextConfig, CosineMergeScorer,
    HeuristicClusterEncoder, LocalCluster, MergeScorer, MergedCluster,
};

pub use coref_metrics::{
    b_cubed_score, blanc_score, ceaf_e_score, ceaf_m_score, conll_f1, lea_score, muc_score,
    CorefEvaluation, CorefScores, ZeroAnaphorEvaluation,
};

pub use coref_resolver::{BoxCorefResolver, CorefConfig, SimpleCorefResolver};

