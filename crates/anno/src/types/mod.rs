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
//! | [`Confidence`] | f64 | [0, 1] | Model/entity confidence scores |
//! | [`Score`] | f32 | [0, 1] | Neural network outputs (GPU-native) |
//!
//! `Confidence` is re-exported from `anno_core`. `Score` converts to
//! `Confidence` via `.to_confidence()`.
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
//! use anno::types::{Score, EntitySliceExt};
//! use anno::{Confidence, Entity, EntityType};
//!
//! let conf = Confidence::new(0.95);         // Clamps to [0, 1]
//! let score = Score::from_logit(2.5);       // Applies sigmoid
//!
//! let conf_from_score: Confidence = score.to_confidence();
//!
//! let entities = vec![
//!     Entity::new("John", EntityType::Person, 0, 4, conf.value()),
//! ];
//! let high_conf: Vec<_> = entities.above_confidence(0.8).collect();
//! ```

mod ext;
mod score;
/// Uncertain predictions and abstention for selective NER.
pub(crate) mod uncertain;

pub use anno_core::Confidence;
pub use ext::EntitySliceExt;
pub use score::Score;

/// Static assertions for struct layouts and invariants.
#[doc(hidden)]
pub mod static_checks {
    // Confidence is zero-cost (same size as f64)
    const _: () =
        assert!(std::mem::size_of::<anno_core::Confidence>() == std::mem::size_of::<f64>());

    // Score is zero-cost (same size as f32)
    const _: () = assert!(std::mem::size_of::<super::Score>() == std::mem::size_of::<f32>());

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

        #[test]
        fn score_saturating_always_valid(value in -10.0f32..10.0) {
            let score = Score::saturating(value);
            prop_assert!(score.get() >= 0.0);
            prop_assert!(score.get() <= 1.0);
        }

        #[test]
        fn score_new_rejects_invalid(value in -10.0f32..10.0) {
            let result = Score::new(value);
            if (0.0..=1.0).contains(&value) && !value.is_nan() {
                prop_assert!(result.is_some());
            } else {
                prop_assert!(result.is_none());
            }
        }

        #[test]
        fn score_from_logit_always_valid(logit in -100.0f32..100.0) {
            let score = Score::from_logit(logit);
            prop_assert!(score.get() >= 0.0);
            prop_assert!(score.get() <= 1.0);
        }

        #[test]
        fn score_from_logit_monotonic(a in -10.0f32..10.0, b in -10.0f32..10.0) {
            let score_a = Score::from_logit(a);
            let score_b = Score::from_logit(b);
            if a < b {
                prop_assert!(score_a.get() <= score_b.get() + 1e-6);
            } else if a > b {
                prop_assert!(score_a.get() >= score_b.get() - 1e-6);
            }
        }

        #[test]
        fn score_to_confidence_preserves_bounds(value in 0.0f32..=1.0) {
            let score = Score::new(value).expect("valid score value");
            let conf = score.to_confidence();
            prop_assert!(conf.value() >= 0.0);
            prop_assert!(conf.value() <= 1.0);
        }

        #[test]
        fn score_confidence_conversion_preserves_ordering(a in 0.0f32..=1.0, b in 0.0f32..=1.0) {
            let score_a = Score::new(a).expect("valid score value");
            let score_b = Score::new(b).expect("valid score value");
            let conf_a = score_a.to_confidence();
            let conf_b = score_b.to_confidence();

            if a < b {
                prop_assert!(conf_a.value() <= conf_b.value() + 1e-6);
            } else if a > b {
                prop_assert!(conf_a.value() >= conf_b.value() - 1e-6);
            }
        }
    }
}
