#![warn(missing_docs)]

//! `anno-core` contains the stable data model and invariants for `anno`.
//!
//! This crate intentionally avoids CLI and evaluation dependencies.

pub mod coalesce;
pub mod core;
pub mod minimal;

pub use crate::core::{
    generate_span_candidates, Animacy, Confidence, CorefChain, CorefDocument, CoreferenceResolver,
    Corpus, DiscontinuousSpan, Entity, EntityBuilder, EntityCategory, EntityType, ExtractionMethod,
    Gender, GroundedDocument, HashMapLexicon, HierarchicalConfidence, Identity, IdentityId,
    IdentitySource, Lexicon, Location, Mention, MentionType, Modality, Number, Person, PhiFeatures,
    Provenance, Quantifier, RaggedBatch, Relation, Signal, SignalId, SignalRef, Span,
    SpanCandidate, Track, TrackId, TrackRef, TrackStats, TypeLabel, TypeMapper, ValidationIssue,
};

pub use crate::core::types::{ByteOffset, CharOffset};

pub use crate::core::types::CanonicalId;

pub use crate::core::grounded::SignalValidationError;
