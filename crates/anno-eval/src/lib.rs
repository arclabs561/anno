#![warn(missing_docs)]

//! Evaluation tooling for `anno`.

// `anno-eval` builds on the `anno` API surface and uses the same core types.
//
// Note: these are kept as crate-root exports to avoid massive churn inside the eval modules
// (many files historically referenced `crate::{Error, Result, Model, ...}`).
// They are hidden from docs because `anno-eval`'s public surface should primarily be `eval::*`.
#[doc(hidden)]
pub use anno::{
    backends, BertNEROnnx, DiscontinuousEntity, DiscontinuousNER, Entity, EntityType, Error,
    GLiNEROnnx, HeuristicNER, Model, Provenance, RegexNER, RelationExtractor, RelationTriple,
    Result, StackedNER, DEFAULT_BERT_ONNX_MODEL, DEFAULT_GLINER2_MODEL, DEFAULT_GLINER_MODEL,
    DEFAULT_NUNER_MODEL, DEFAULT_W2NER_MODEL,
};

// Some eval modules use `crate::discourse::*` when the `discourse` feature is enabled.
#[cfg(feature = "discourse")]
pub use anno::discourse;

pub mod muxer_harness;

#[path = "eval/mod.rs"]
pub mod eval;

/// Cross-document coreference / clustering (CDCR).
///
/// This is re-exported at the crate root so downstream code (notably the CLI)
/// doesn't need to import it from the `eval::*` namespace.
pub mod cdcr {
    pub use crate::eval::cdcr::*;
}

// CI-friendly matrix harness (tests).
#[cfg(test)]
mod matrix_muxer_ci;
