//! Property-based tests for Entity builder fluent API.
//!
//! These tests verify that the builder pattern works correctly
//! with arbitrary inputs and method chaining.

use anno::{Entity, EntityCategory, EntityType};
use proptest::prelude::*;

proptest! {

    /// Entity builder fluent chaining should work correctly.
    #[test]
    fn entity_builder_fluent_chaining(
        text in ".{1,100}",
        entity_type in prop::sample::select(vec![
            EntityType::Person,
            EntityType::Organization,
            EntityType::Location,
        ]),
        start in 0usize..100,
        end in 0usize..100,
        confidence in 0.0f64..1.0f64,
    ) {
        let char_count = text.chars().count();
        if start < end && end <= char_count {
            let entity = Entity::builder(&text, entity_type.clone())
                .span(start, end)
                .confidence(confidence)
                .build();

            prop_assert_eq!(entity.text, text);
            prop_assert_eq!(entity.start, start);
            prop_assert_eq!(entity.end, end);
            prop_assert!((entity.confidence - confidence.clamp(0.0, 1.0)).abs() < 0.001);
            prop_assert_eq!(entity.entity_type, entity_type);
        }
    }

    /// Entity builder should handle optional fields.
    #[test]
    fn entity_builder_optional_fields(
        text in ".{1,100}",
        start in 0usize..100,
        end in 0usize..100,
        kb_id in prop::option::of("[A-Z0-9]{1,20}"),
    ) {
        let char_count = text.chars().count();
        if start < end && end <= char_count {
            let mut builder = Entity::builder(&text, EntityType::Person)
                .span(start, end)
                .confidence(0.9);

            if let Some(kb) = &kb_id {
                builder = builder.kb_id(kb);
            }

            let entity = builder.build();

            prop_assert_eq!(entity.kb_id, kb_id);
        }
    }

    /// Entity builder should clamp confidence.
    #[test]
    fn entity_builder_confidence_clamping(
        text in ".{1,100}",
        confidence in -1.0f64..2.0f64,
    ) {
        let entity = Entity::builder(&text, EntityType::Person)
            .span(0, text.chars().count().min(100))
            .confidence(confidence)
            .build();

        // Confidence should be clamped to [0.0, 1.0]
        prop_assert!(entity.confidence >= 0.0 && entity.confidence <= 1.0);
    }

    /// Entity builder should handle all entity types.
    #[test]
    fn entity_builder_all_types(
        text in ".{1,50}",
        entity_type in prop::sample::select(vec![
            EntityType::Person,
            EntityType::Organization,
            EntityType::Location,
            EntityType::Date,
            EntityType::Money,
            EntityType::Email,
            EntityType::custom("CUSTOM", EntityCategory::Misc),
        ]),
    ) {
        let char_count = text.chars().count();
        if char_count > 0 {
            let entity = Entity::builder(&text, entity_type.clone())
                .span(0, char_count)
                .build();

            prop_assert_eq!(entity.entity_type, entity_type);
        }
    }

    /// Entity builder should be idempotent for same inputs.
    #[test]
    fn entity_builder_idempotent(
        text in ".{1,100}",
        start in 0usize..100,
        end in 0usize..100,
    ) {
        let char_count = text.chars().count();
        if start < end && end <= char_count {
            let entity1 = Entity::builder(&text, EntityType::Person)
                .span(start, end)
                .confidence(0.9)
                .build();

            let entity2 = Entity::builder(&text, EntityType::Person)
                .span(start, end)
                .confidence(0.9)
                .build();

            prop_assert_eq!(entity1.text, entity2.text);
            prop_assert_eq!(entity1.start, entity2.start);
            prop_assert_eq!(entity1.end, entity2.end);
            prop_assert!((entity1.confidence - entity2.confidence).abs() < 0.001);
        }
    }
}
