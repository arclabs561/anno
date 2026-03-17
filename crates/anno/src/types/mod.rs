//! Type-level programming patterns for compile-time safety.
//!
//! This module provides witness types and extension traits that encode
//! invariants in the type system. Once data is parsed into these types,
//! downstream code can rely on the invariant without re-checking.
//!
//! # Bounded Value Types
//!
//! | Type | Precision | Domain | When to Use |
//! |------|-----------|--------|-------------|
//! | [`Confidence`] | f64 | [0, 1] | All confidence/probability scores |
//!
//! `Confidence` is re-exported from `anno_core`. It supports construction
//! from raw values (`new`), logits (`from_logit`), and scaled logits
//! (`from_logit_scaled`).
//!
//! # Extension Traits
//!
//! | Trait | Extends | Purpose |
//! |-------|---------|---------|
//! | [`EntitySliceExt`] | `[Entity]` | Filter, sort, group entity collections |
//!
//! # Example
//!
//! ```rust
//! use anno::types::EntitySliceExt;
//! use anno::{Confidence, Entity, EntityType};
//!
//! let conf = Confidence::new(0.95);              // Clamps to [0, 1]
//! let from_logit = Confidence::from_logit(2.5);  // Applies sigmoid
//!
//! let entities = vec![
//!     Entity::new("John", EntityType::Person, 0, 4, conf.value()),
//! ];
//! let high_conf: Vec<_> = entities.above_confidence(0.8).collect();
//! ```

mod ext;
pub use anno_core::Confidence;
pub use ext::EntitySliceExt;

/// Static assertions for struct layouts and invariants.
#[doc(hidden)]
pub mod static_checks {
    // Confidence is zero-cost (same size as f64)
    const _: () =
        assert!(std::mem::size_of::<anno_core::Confidence>() == std::mem::size_of::<f64>());

    // Entity is reasonably sized (fits in a few cache lines)
    const _: () = assert!(std::mem::size_of::<anno_core::Entity>() <= 512);

    // SpanCandidate is small and copyable (used in hot loops)
    const _: () = assert!(std::mem::size_of::<crate::SpanCandidate>() <= 16);

    // HierarchicalConfidence is compact (3 x Confidence/f64 = 24 bytes)
    const _: () = assert!(std::mem::size_of::<crate::HierarchicalConfidence>() <= 24);
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn confidence_clamping_always_valid(value in -10.0f64..10.0) {
            let conf = Confidence::new(value);
            prop_assert!(conf.value() >= 0.0);
            prop_assert!(conf.value() <= 1.0);
        }

        #[test]
        fn confidence_roundtrip_f64(value in 0.0f64..=1.0) {
            let conf = Confidence::new(value);
            let back: f64 = conf.into();
            prop_assert!((back - value).abs() < 1e-15);
        }

        #[test]
        fn confidence_serde_roundtrip(value in 0.0f64..=1.0) {
            let conf = Confidence::new(value);
            let json = serde_json::to_string(&conf).expect("serialization should succeed");
            let restored: Confidence =
                serde_json::from_str(&json).expect("deserialization should succeed");
            prop_assert!((restored.value() - value).abs() < 1e-15);
        }
    }
}
