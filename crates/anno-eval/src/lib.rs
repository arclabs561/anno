#![warn(missing_docs)]

//! Evaluation tooling for `anno`.

// `anno-eval` builds on the `anno` API surface and uses the same core types.
//
// Note: these are kept as crate-root exports to avoid massive churn inside the eval modules
// (many files historically referenced `crate::{Error, Result, Model, ...}`).
// They are hidden from docs because `anno-eval`'s public surface should primarily be `eval::*`.
#[doc(hidden)]
pub use anno::{
    backends, DiscontinuousEntity, DiscontinuousNER, Entity, EntityCategory, EntityType, Error,
    HeuristicNER, Model, Provenance, RegexNER, RelationExtractor, RelationTriple, Result,
    StackedNER, DEFAULT_BERT_ONNX_MODEL, DEFAULT_GLINER2_MODEL, DEFAULT_GLINER_MODEL,
    DEFAULT_GLINER_POLY_MODEL, DEFAULT_NUNER_MODEL, DEFAULT_W2NER_MODEL,
};

#[cfg(feature = "onnx")]
#[doc(hidden)]
pub use anno::{BertNEROnnx, GLiNEROnnx};

#[cfg(feature = "candle")]
#[doc(hidden)]
pub use anno::{CandleNER, DEFAULT_CANDLE_MODEL, DEFAULT_GLINER_CANDLE_MODEL};

/// Shared helpers for the muxer-backed matrix sampler harness.
#[cfg(feature = "eval")]
pub mod muxer_harness;

/// Persistent muxer selection history (shared by CI harness and tooling).
#[cfg(feature = "eval")]
pub mod muxer_history;

#[path = "eval/mod.rs"]
pub mod eval;

/// Muxer-backed evaluation harness (shared by tests and tooling).
#[cfg(feature = "eval")]
#[path = "matrix_muxer_ci.rs"]
pub mod muxer_matrix;

/// Aggregate muxer decision/outcome logs.
pub mod muxer_agg_lib;

/// Cross-document coreference / clustering (CDCR).
///
/// This is re-exported at the crate root so downstream code (notably the CLI)
/// doesn't need to import it from the `eval::*` namespace.
pub mod cdcr {
    pub use crate::eval::cdcr::*;
}

// Note: `muxer_matrix` contains its own `#[cfg(test)]` tests.
