//! Type-level programming patterns for compile-time safety.
//!
//! This module provides witness types and extension traits that encode
//! invariants in the type system. Once data is parsed into these types,
//! downstream code can rely on the invariant without re-checking.
//!
//! # Design Philosophy: Parse, Don't Validate
//!
//! Instead of repeatedly validating that a confidence score is in [0, 1],
//! parse it once into a `Confidence` type. The type system then guarantees
//! the invariant holds everywhere the value is used.
//!
//! # Bounded Value Types
//!
//! All bounded values in anno share the same pattern:
//!
//! | Type | Precision | Domain | When to Use |
//! |------|-----------|--------|-------------|
//! | [`Confidence`] | f64 | [0, 1] | Model/entity confidence scores |
//! | [`Score`] | f32 | [0, 1] | Neural network outputs (GPU-native) |
//! | [`crate::eval::MetricValue`] | f64 | [0, 1] | Evaluation metrics (P/R/F1) |
//!
//! **Relationship:**
//! - `Score` converts to `Confidence` via `.to_confidence()` (f32 -> f64)
//! - `Confidence` and `MetricValue` both wrap f64 but have different semantics
//! - `Entity.confidence` is raw `f64` for API stability; use `Confidence` internally
//!
//! # Type Aliases
//!
//! | Alias | Points To | Semantic Meaning |
//! |-------|-----------|------------------|
//! | [`Probability`] | `Confidence` | True probability (softmax output) |
//! | [`UnitInterval`] | `Confidence` | Generic `[0, 1]` value (progress, ratio) |
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
//! use anno::types::{Confidence, Score, EntitySliceExt};
//! use anno::{Entity, EntityType};
//!
//! // Parse at boundaries: construct witness type once
//! let conf = Confidence::saturating(0.95);  // Clamps to [0, 1]
//! let score = Score::from_logit(2.5);       // Applies sigmoid
//!
//! // Zero-cost usage: type guarantees bounds, no runtime checks needed
//! assert!(conf.is_high());  // Uses internal value directly
//! let as_f64: f64 = conf.get();
//!
//! // Convert between precisions
//! let conf_from_score: Confidence = score.to_confidence();
//!
//! // Extension traits for collections
//! let entities = vec![
//!     Entity::new("John", EntityType::Person, 0, 4, conf.get()),
//! ];
//! let high_conf: Vec<_> = entities.above_confidence(0.8).collect();
//! ```

mod confidence;
mod ext;
mod score;

pub use confidence::{Confidence, ConfidenceError, Probability, UnitInterval};
pub use ext::EntitySliceExt;
pub use score::Score;

/// Static assertions for struct layouts and invariants.
///
/// These are compile-time checks that ensure critical assumptions hold.
/// If any assertion fails, compilation will fail with an error message.
#[doc(hidden)]
pub mod static_checks {
    use super::*;

    // Confidence is zero-cost (same size as f64)
    const _: () = assert!(std::mem::size_of::<Confidence>() == std::mem::size_of::<f64>());
    const _: () = assert!(std::mem::align_of::<Confidence>() == std::mem::align_of::<f64>());

    // Score is zero-cost (same size as f32)
    const _: () = assert!(std::mem::size_of::<Score>() == std::mem::size_of::<f32>());
    const _: () = assert!(std::mem::align_of::<Score>() == std::mem::align_of::<f32>());

    // Entity is reasonably sized (fits in a few cache lines)
    const _: () = assert!(std::mem::size_of::<crate::Entity>() <= 512);

    // SpanCandidate is small and copyable (used in hot loops)
    const _: () = assert!(std::mem::size_of::<crate::SpanCandidate>() <= 16);

    // HierarchicalConfidence is compact
    const _: () = assert!(std::mem::size_of::<crate::HierarchicalConfidence>() <= 16);
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn confidence_saturating_always_valid(value in -10.0f64..10.0) {
            let conf = Confidence::saturating(value);
            prop_assert!(conf.get() >= 0.0);
            prop_assert!(conf.get() <= 1.0);
        }

        #[test]
        fn confidence_new_rejects_invalid(value in -10.0f64..10.0) {
            let result = Confidence::new(value);
            if (0.0..=1.0).contains(&value) && !value.is_nan() {
                prop_assert!(result.is_some());
            } else {
                prop_assert!(result.is_none());
            }
        }

        #[test]
        fn confidence_roundtrip_f64(value in 0.0f64..=1.0) {
            let conf = Confidence::new(value).unwrap();
            let back: f64 = conf.into();
            prop_assert!((back - value).abs() < 1e-15);
        }

        #[test]
        fn confidence_serde_roundtrip(value in 0.0f64..=1.0) {
            let conf = Confidence::new(value).unwrap();
            let json = serde_json::to_string(&conf).unwrap();
            let restored: Confidence = serde_json::from_str(&json).unwrap();
            prop_assert!((restored.get() - value).abs() < 1e-15);
        }

        #[test]
        fn confidence_combine_bounded(a in 0.0f64..=1.0, b in 0.0f64..=1.0) {
            let ca = Confidence::new(a).unwrap();
            let cb = Confidence::new(b).unwrap();
            let combined = ca.combine(cb);
            prop_assert!(combined.get() >= 0.0);
            prop_assert!(combined.get() <= 1.0);
        }

        #[test]
        fn confidence_lerp_bounded(a in 0.0f64..=1.0, b in 0.0f64..=1.0, t in -1.0f64..2.0) {
            let ca = Confidence::new(a).unwrap();
            let cb = Confidence::new(b).unwrap();
            let result = ca.lerp(cb, t);
            prop_assert!(result.get() >= 0.0);
            prop_assert!(result.get() <= 1.0);
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
            let score = Score::new(value).unwrap();
            let conf = score.to_confidence();
            prop_assert!(conf.get() >= 0.0);
            prop_assert!(conf.get() <= 1.0);
        }

        #[test]
        fn score_confidence_conversion_preserves_ordering(a in 0.0f32..=1.0, b in 0.0f32..=1.0) {
            let score_a = Score::new(a).unwrap();
            let score_b = Score::new(b).unwrap();
            let conf_a = score_a.to_confidence();
            let conf_b = score_b.to_confidence();

            if a < b {
                prop_assert!(conf_a.get() <= conf_b.get() + 1e-6);
            } else if a > b {
                prop_assert!(conf_a.get() >= conf_b.get() - 1e-6);
            }
        }
    }
}
