//! Lightweight import surface for downstream crates.
//!
//! `anno-core` is ~22k lines. Most downstream crates only need the core data types --
//! entities, spans, mentions, coreference chains, and provenance. This module collects
//! those types into a single `use` target so that consumers avoid pulling in the dataset
//! registry, calibration machinery, graph-export formats, and historical-linguistics types.
//!
//! # When to use `minimal` vs the full re-export
//!
//! | Import path | What you get | What you skip |
//! |---|---|---|
//! | `anno_core::minimal::*` | Data types, builders, coref, provenance | Dataset registry, calibration, historical, graph export |
//! | `anno_core::*` | Everything | Nothing |
//!
//! Prefer `minimal` when your crate only reads/writes annotation data. Use the full
//! surface when you need evaluation infrastructure or dataset loading.
//!
//! # Quick start
//!
//! ```rust
//! use anno_core::minimal::{Entity, EntityType, Span, Mention, CorefChain};
//!
//! // Build an entity at character offsets 0..6.
//! let entity = Entity::new("Berlin", EntityType::Location, 0, 6, 0.95);
//! assert_eq!(entity.text, "Berlin");
//! assert_eq!((entity.start, entity.end), (0, 6));
//!
//! // Build a coreference chain from two mentions.
//! let m1 = Mention::new("John", 0, 4);
//! let m2 = Mention::new("he", 25, 27);
//! let chain = CorefChain::new(vec![m1, m2]);
//! assert_eq!(chain.len(), 2);
//! assert!(!chain.is_singleton());
//! ```
//!
//! # Builder pattern
//!
//! For richer construction, use [`EntityBuilder`]:
//!
//! ```rust
//! use anno_core::minimal::{EntityBuilder, EntityType, Provenance};
//!
//! let entity = EntityBuilder::new("Berlin", EntityType::Location)
//!     .span(0, 6)
//!     .confidence(0.95)
//!     .provenance(Provenance::ml("gliner", 0.95))
//!     .build();
//! assert_eq!(entity.provenance.unwrap().source, "gliner");
//! ```

// -- Entity layer ----------------------------------------------------------
pub use crate::core::entity::{
    DiscontinuousSpan, Entity, EntityBuilder, EntityCategory, EntityType, ExtractionMethod,
    Provenance, Relation, Span,
};

// -- Coreference layer -----------------------------------------------------
pub use crate::core::coref::{CorefChain, Mention};

// -- Scalar/ID types -------------------------------------------------------
pub use crate::core::types::{
    ByteOffset, ByteSpan, CanonicalId, CharOffset, CharSpan, IdentityId, MentionType, SignalId,
    TrackId, TypeLabel,
};
