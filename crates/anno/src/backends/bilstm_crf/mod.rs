//! Backward-compatibility re-exports from [`super::heuristic_crf`].
//!
//! The `bilstm_crf` module was renamed to `heuristic_crf` to honestly reflect
//! that the emission features are heuristic (gazetteers, word shape,
//! capitalization), not a neural BiLSTM encoder. The CRF layer is real.
//!
//! Prefer importing from [`super::heuristic_crf`] directly.

#[allow(deprecated)]
pub use super::heuristic_crf::{HeuristicCrfConfig, HeuristicCrfNER};

/// Backward-compatible alias for [`HeuristicCrfNER`].
#[deprecated(
    since = "0.5.0",
    note = "renamed to HeuristicCrfNER in backends::heuristic_crf"
)]
pub type BiLstmCrfNER = HeuristicCrfNER;

/// Backward-compatible alias for [`HeuristicCrfConfig`].
#[deprecated(
    since = "0.5.0",
    note = "renamed to HeuristicCrfConfig in backends::heuristic_crf"
)]
pub type BiLstmCrfConfig = HeuristicCrfConfig;
