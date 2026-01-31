//! Shared types for the anno toolbox.
//!
//! This module provides canonical definitions for types that are used across
//! multiple modules to avoid duplication and ensure consistency.
//!
//! # Type-Theoretic Design Philosophy
//!
//! Anno's type system is informed by the **Curry-Howard correspondence**: the
//! principle that types correspond to propositions and programs correspond to
//! proofs. Under this view:
//!
//! - A function `fn resolve(mention: Unresolved) -> Resolved` is a proof that
//!   unresolved mentions can be transformed into resolved ones
//! - Type errors are logical contradictions caught at compile time
//! - Well-typed programs are correct by construction
//!
//! ## Type-Logical Grammar Connection
//!
//! Just as categorial grammar assigns types to words (e.g., `NP`, `NP\S`),
//! we assign types to linguistic objects:
//!
//! | Linguistic Concept | Anno Type | Notes |
//! |-------------------|-----------|-------|
//! | Entity mention | `Signal<Location>` | Level 1 detection |
//! | Coreference chain | `Track` | Level 2 grouping |
//! | KB-linked entity | `Identity` | Level 3 linking |
//! | Phi-features | `PhiFeatures` | Agreement constraints |
//! | Mention category | `MentionType` | Accessibility hierarchy |
//!
//! ## Type Safety Patterns Used
//!
//! 1. **Newtypes**: `SignalId`, `TrackId`, `IdentityId` prevent mixing IDs
//! 2. **Enums with methods**: `Number::is_compatible()` encodes agreement
//! 3. **Option for uncertainty**: `Option<Gender>` = unknown gender
//! 4. **Result for fallibility**: Parsing, validation can fail
//!
//! These patterns catch real bugs at compile time:
//! - Can't pass a `TrackId` where `SignalId` is expected
//! - `is_compatible()` enforces linguistic constraints (person exclusion, dual-plural)
//!
//! # Types
//!
//! ## Identifiers (Type-Safe)
//!
//! - [`SignalId`] - Unique identifier for signals (Level 1 detections)
//! - [`TrackId`] - Unique identifier for tracks (Level 2 coreference chains)
//! - [`IdentityId`] - Unique identifier for identities (Level 3 KB-linked entities)
//! - [`CanonicalId`] - Coreference cluster ID for within-document grouping
//!
//! ## Linguistic Features
//!
//! - [`Gender`] - Gender classification for coreference and bias analysis
//! - [`MentionType`] - Types of referring expressions (proper, nominal, pronominal, zero)
//! - [`PhiFeatures`] - Person/number/gender features for morphological agreement
//!
//! ## Temporal & Statistical
//!
//! - [`MetricStats`] - Statistical metrics with variance and confidence intervals
//! - [`HistoricalDate`] - Dates that can represent BCE/CE with uncertainty
//! - [`TemporalValidity`] - Time range during which an entity is valid
//!
//! ## Discourse & Framing
//!
//! - [`FramingAttitude`] - Supportive/skeptical/neutral framing stance
//! - [`FrecoPair`] - Framing-divergent event coreference pair

pub mod framing;
mod gender;
mod ids;
mod mention_type;
mod metric;
mod phi_features;
mod temporal;
mod type_label;

pub use framing::{
    EventArgument, EventCorefRelation, EventMention, FramingAttitude, FramingDivergenceType,
    FrecoCorpus, FrecoCorpusMetadata, FrecoPair,
};
pub use gender::Gender;
pub use ids::{
    ByteOffset, ByteSpan, CanonicalId, CharOffset, CharSpan, IdentityId, SignalId, TrackId,
};
pub use mention_type::MentionType;
pub use metric::MetricStats;
pub use phi_features::{Number, Person, PhiFeatures};
pub use temporal::{DatePrecision, HistoricalDate, TemporalValidity};
pub use type_label::TypeLabel;
