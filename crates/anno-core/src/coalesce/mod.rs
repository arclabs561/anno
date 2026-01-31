//! # `anno_core::coalesce`
//!
//! Cross-document coalescing groups extracted mentions/tracks into “identities” across documents.
//! This module is implementation-focused: similarity primitives, candidate generation, and
//! clustering utilities used by the `anno` pipeline.
//!
//! ## What is stable
//!
//! - The public surface is intentionally small: prefer the batch `Resolver` or the
//!   streaming `StreamingResolver`, plus the `similarity` building blocks.
//! - Quantitative claims (runtime/quality) belong in generated eval artifacts, not in doc comments.
//!
//! ## Where to look
//!
//! - `resolver`: batch coalescing utilities
//! - `lsh`: candidate generation for scaling beyond all-pairs comparison
//! - `streaming`: incremental coalescing for arriving documents
//! - `similarity`: string/script heuristics and similarity composition

#![warn(missing_docs)]

pub mod lsh;
pub mod resolver;
pub mod script;
pub mod similarity;
pub mod streaming;

// =============================================================================
// Public surface (keep small; the rest is reachable via submodules)
// =============================================================================

pub use resolver::Resolver;
pub use similarity::{
    ChainedSynonyms, NoSynonyms, Script, Similarity, SimilarityConfig, SynonymMatch, SynonymSource,
};
pub use streaming::{EntityCluster, EntityMention, StreamingConfig, StreamingResolver};

// NOTE: additional clustering / evidence aggregation experiments were intentionally
// removed to keep this module paper-grounded and reduce bespoke surface area.
