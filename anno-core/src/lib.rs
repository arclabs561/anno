//! # anno-core
//!
//! Core types for the anno toolbox: shared data structures used across all crates.
//!
//! This crate provides:
//! - **Entity types**: `Entity`, `EntityType`, `Span`, `Relation`
//! - **Grounded hierarchy**: `Signal`, `Track`, `Identity`, `GroundedDocument`, `Corpus`
//! - **Graph types**: `GraphDocument`, `GraphNode`, `GraphEdge`
//!
//! All other crates in the anno workspace depend on `anno-core` to ensure
//! type compatibility across the toolbox.

pub mod entity;
pub mod error;
pub mod graph;
pub mod grounded;

// Re-exports for convenience
pub use entity::{
    DiscontinuousSpan, Entity, EntityBuilder, EntityCategory, EntityType, EntityViewport,
    ExtractionMethod, HashMapLexicon, HierarchicalConfidence, Lexicon, Provenance, RaggedBatch,
    Relation, Span, SpanCandidate, TypeMapper, ValidationIssue,
};

pub use grounded::{
    Corpus, GroundedDocument, Identity, IdentityId, IdentitySource, Location, Modality, Quantifier,
    Signal, SignalId, SignalRef, Track, TrackId, TrackRef,
};

pub use error::{Error, Result};
pub use graph::{GraphDocument, GraphEdge, GraphExportFormat, GraphNode};
