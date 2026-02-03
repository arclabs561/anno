//! `anno-core` minimal stable surface.
//!
//! This module is the intended “default import set” for downstream crates that want the
//! extraction contract (spans + types + ids) without pulling in evaluation, dataset registry, or
//! graph-export conveniences.
//!
//! Design goal: keep this small and boring. If a type is controversial/experimental, it should
//! not be re-exported here.

pub use crate::core::entity::{
    DiscontinuousSpan, Entity, EntityCategory, EntityType, EntityViewport, Relation, Span,
};
pub use crate::core::types::{
    ByteOffset, ByteSpan, CanonicalId, CharOffset, CharSpan, IdentityId, SignalId, TrackId,
    TypeLabel,
};
