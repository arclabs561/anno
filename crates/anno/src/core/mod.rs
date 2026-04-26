//! # anno::core
//!
//! Core types for the Anno NLP toolkit: Named Entity Recognition, Coreference
//! Resolution, and Relation Extraction.
//!
//! ## Why This Module Exists
//!
//! NLP pipelines involve many components (tokenizers, NER models, coreference
//! resolvers, entity linkers) that need to share data. Without a common type
//! system, each component defines its own `Entity`, `Span`, `Document` types,
//! requiring tedious conversion code and risking subtle bugs.
//!
//! `anno::core` solves this by providing:
//!
//! 1. **Canonical types** that all components agree on
//! 2. **Rich metadata** beyond basic spans (confidence, provenance, relations)
//! 3. **Grounded hierarchy** for multi-document, multi-modal processing
//! 4. **Coreference** chains and resolver traits
//!
//! ## Core Concepts
//!
//! ### Entities and Spans
//!
//! ```rust,ignore
//! use crate::{Entity, EntityType, Span};
//!
//! let entity = Entity::new("Barack Obama", EntityType::Person)
//!     .with_span(Span::new(0, 12))
//!     .with_confidence(0.95);
//! ```
//!
//! ### Grounded Documents
//!
//! For cross-document coreference, entities are "grounded" to real-world identities:
//!
//! ```rust,ignore
//! use crate::{GroundedDocument, Identity, Signal};
//!
//! // Multiple mentions across documents resolve to one identity
//! let obama_id = Identity::new("Q76"); // Wikidata ID
//! doc1.ground_mention(mention1, obama_id.clone());
//! doc2.ground_mention(mention2, obama_id);
//! ```
//!
//! ## Module Overview
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`entity`] | `Entity`, `Span`, `Relation`, `EntityType` |
//! | [`grounded`] | `GroundedDocument`, `Identity`, `Signal`, `Track` |
//! | [`coref`] | `Mention`, `CorefChain`, `CorefDocument` |
//! | `graph` | Export to Neo4j, GraphML, JSON-LD |
//! | [`provenance`] | Document origin tracking |
//! | [`types`] | `Gender`, `MentionType`, `TypeLabel` |
//!
//! ## Design Philosophy
//!
//! - **Character offsets, not byte offsets**: Unicode-safe from the start
//! - **Immutable where possible**: Entities are built then used, not mutated
//! - **Serde everywhere**: All types serialize for caching and interop
//! - **No ML dependencies**: Pure data types, no torch/candle/onnx
//!
//! ## Minimal surface
//!
//! If you’re downstream and want a small “just the contract” import set, prefer
//! `crate::minimal` (or `anno::core::*` in the `anno` crate) rather than grabbing the entire
//! re-export surface.

pub mod confidence;
pub mod coref;
pub mod entity;
pub mod error;
pub mod grounded;
pub mod provenance;
pub mod types;

// Re-exports for convenience
pub use confidence::Confidence;
pub use entity::{
    generate_span_candidates, DiscontinuousSpan, Entity, EntityBuilder, EntityCategory, EntityType,
    ExtractionMethod, HashMapLexicon, HierarchicalConfidence, Lexicon, Provenance, RaggedBatch,
    Relation, Span, SpanCandidate, TypeMapper, ValidationIssue,
};

pub use grounded::{
    Corpus, GroundedDocument, Identity, IdentityId, IdentitySource, Location, Modality, Quantifier,
    Signal, SignalId, SignalRef, Track, TrackId, TrackRef, TrackStats,
};

pub use error::{Error, Result};

// Coreference types
pub use coref::{entities_to_chains, CorefChain, CorefDocument, CoreferenceResolver, Mention};

// Other modules accessible via crate::module_name
pub use types::{
    Animacy, Gender, MentionType, MetricStats, Number, Person, PhiFeatures, TypeLabel,
};
