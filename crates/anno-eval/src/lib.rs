#![warn(missing_docs)]

//! Evaluation tooling for `anno`.

// `anno-eval` builds on the `anno` API surface and uses the same core types.
//
// Note: these are kept as crate-root exports to avoid massive churn inside the eval modules
// (many files historically referenced `crate::{Error, Result, Model, ...}`).
// They are hidden from docs because `anno-eval`'s public surface should primarily be `eval::*`.
#[doc(hidden)]
pub use anno::{
    DiscontinuousEntity, DiscontinuousNER, Entity, EntityType, Error, HeuristicNER, Model,
    Provenance, RegexNER, RelationExtractor, RelationTriple, Result, StackedNER,
};

// Some eval modules use `crate::discourse::*` when the `discourse` feature is enabled.
#[cfg(feature = "discourse")]
pub use anno::discourse;

pub mod muxer_harness;

#[path = "eval/mod.rs"]
pub mod eval;

// CI-friendly matrix harness (tests).
#[cfg(test)]
mod matrix_muxer_ci;
