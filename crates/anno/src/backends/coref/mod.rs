//! Coreference resolution backends.
//!
//! This module groups all within-document coreference resolvers:
//! - [`resolve`] -- Unified `CorefBackend` trait
//! - [`simple`] -- Rule-based resolvers
//! - [`mention_ranking`] -- Mention-ranking coreference (Bourgois & Poibeau 2025)
//! - [`fcoref`] -- Fast neural coreference (ONNX encoder + safetensors scorer)
//! - [`t5`] -- T5-based seq2seq coreference

/// Unified trait for coreference resolution backends.
///
/// Open trait (not sealed) -- external coref backends can implement it.
pub mod resolve;

// Mention-ranking coreference (Bourgois & Poibeau 2025 inspired)
pub mod mention_ranking;

// T5-based coreference resolution
#[cfg(feature = "onnx")]
pub mod t5;

// F-coref: fast neural coreference (ONNX encoder + safetensors scorer heads)
#[cfg(feature = "onnx")]
pub mod fcoref;

// Simple rule-based coreference resolvers.
#[cfg(feature = "analysis")]
pub mod simple;

// Re-exports: keep everything accessible at the coref:: level
pub use resolve::CorefBackend;

// CorefCluster is always available from resolve (not feature-gated).
pub use resolve::CorefCluster;

#[cfg(feature = "onnx")]
pub use t5::{T5Coref, T5CorefConfig};

#[cfg(feature = "onnx")]
pub use fcoref::{FCoref, FCorefConfig};

#[cfg(feature = "analysis")]
pub use simple::{CorefConfig, SimpleCorefResolver};
