//! Invariant Tests for anno
//!
//! These tests verify properties that should ALWAYS hold true,
//! regardless of input. They are designed to catch subtle bugs
//! that unit tests might miss.

use anno::{
    backends::{HeuristicNER, RegexNER, StackedNER},
    Entity, EntityType, Model,
};
use proptest::prelude::*;
use std::collections::HashSet;

// =============================================================================
// Entity Invariants
// =============================================================================

mod entity_invariants {
    use super::*;

    proptest! {
        /// INVARIANT: Entity start must always be <= end
        #[test]
        fn start_lte_end(
            start in 0usize..10000,
            len in 0usize..1000
        ) {
            let end = start.saturating_add(len);
            let entity = Entity::new("test", EntityType::Person, start, end, 0.9);

            prop_assert!(
                entity.start <= entity.end,
                "start {} > end {} violates invariant",
                entity.start, entity.end
            );
        }

        /// INVARIANT: total_len() == end - start for contiguous entities
        #[test]
        fn total_len_equals_span(
            start in 0usize..10000,
            len in 0usize..1000
        ) {
            let end = start.saturating_add(len);
            let entity = Entity::new("test", EntityType::Person, start, end, 0.9);

            prop_assert_eq!(
                entity.total_len(),
                entity.end - entity.start,
                "total_len() != end - start"
            );
        }

        /// INVARIANT: Confidence is always in [0, 1]
        #[test]
        fn confidence_in_range(conf in -10.0f64..10.0f64) {
            let entity = Entity::new("test", EntityType::Person, 0, 4, conf);

            prop_assert!(
                entity.confidence >= 0.0 && entity.confidence <= 1.0,
                "confidence {} not in [0, 1]",
                entity.confidence
            );
        }

        /// INVARIANT: Entity type is preserved
        #[test]
        fn type_preserved(type_idx in 0u8..10) {
            let entity_type = match type_idx {
                0 => EntityType::Person,
                1 => EntityType::Organization,
                2 => EntityType::Location,
                3 => EntityType::Date,
                4 => EntityType::Time,
                5 => EntityType::Money,
                6 => EntityType::Percent,
                7 => EntityType::Email,
                8 => EntityType::Phone,
                _ => EntityType::Other("Custom".to_string()),
            };

            let entity = Entity::new("test", entity_type.clone(), 0, 4, 0.9);

            prop_assert_eq!(entity.entity_type, entity_type);
        }

        /// INVARIANT: Text is preserved exactly
        #[test]
        fn text_preserved(text in "[A-Za-z0-9 ]{1,100}") {
            let entity = Entity::new(&text, EntityType::Person, 0, text.len(), 0.9);

            prop_assert_eq!(&entity.text, &text);
        }
    }
}

// =============================================================================
// Extraction Invariants
// =============================================================================

mod extraction_invariants {
    use super::*;

    proptest! {
        /// INVARIANT: All extracted entities have valid spans within text
        #[test]
        fn entities_within_text_bounds(text in "[A-Za-z0-9 .,@]{1,500}") {
            let char_count = text.chars().count();

            let backends: Vec<Box<dyn Model>> = vec![
                Box::new(RegexNER::new()),
                Box::new(HeuristicNER::new()),
                Box::new(StackedNER::new()),
            ];

            for backend in backends {
                if let Ok(entities) = backend.extract_entities(&text, None) {
                    for entity in &entities {
                        prop_assert!(
                            entity.start <= entity.end,
                            "Entity {} has start > end",
                            entity.text
                        );
                        prop_assert!(
                            entity.end <= char_count,
                            "Entity {} end {} exceeds text length {}",
                            entity.text, entity.end, char_count
                        );
                    }
                }
            }
        }

        /// INVARIANT: Empty text produces no entities
        #[test]
        fn empty_text_no_entities(backend_idx in 0..3u8) {
            let backend: Box<dyn Model> = match backend_idx {
                0 => Box::new(RegexNER::new()),
                1 => Box::new(HeuristicNER::new()),
                _ => Box::new(StackedNER::new()),
            };

            let result = backend.extract_entities("", None);
            prop_assert!(result.is_ok());
            prop_assert!(result.unwrap().is_empty());
        }

        /// INVARIANT: Extraction never panics on valid UTF-8
        #[test]
        fn no_panic_on_utf8(text in "\\PC{0,200}") {
            let backends: Vec<Box<dyn Model>> = vec![
                Box::new(RegexNER::new()),
                Box::new(HeuristicNER::new()),
                Box::new(StackedNER::new()),
            ];

            for backend in backends {
                // Should not panic, even on weird input
                let _ = backend.extract_entities(&text, None);
            }
        }

        /// INVARIANT: All entities have non-empty text (when span > 0)
        #[test]
        fn non_empty_text_for_nonzero_span(text in "[A-Za-z0-9 .,@]{10,200}") {
            let ner = RegexNER::new();

            if let Ok(entities) = ner.extract_entities(&text, None) {
                for entity in &entities {
                    if entity.end > entity.start {
                        prop_assert!(
                            !entity.text.is_empty(),
                            "Entity with span {}..{} has empty text",
                            entity.start, entity.end
                        );
                    }
                }
            }
        }

        /// INVARIANT: Entity confidence is always in [0, 1]
        #[test]
        fn confidence_always_valid(text in "[A-Za-z0-9 .,@$%]{10,200}") {
            let backends: Vec<Box<dyn Model>> = vec![
                Box::new(RegexNER::new()),
                Box::new(HeuristicNER::new()),
                Box::new(StackedNER::new()),
            ];

            for backend in backends {
                if let Ok(entities) = backend.extract_entities(&text, None) {
                    for entity in &entities {
                        prop_assert!(
                            entity.confidence >= 0.0 && entity.confidence <= 1.0,
                            "Entity {} has invalid confidence {}",
                            entity.text, entity.confidence
                        );
                    }
                }
            }
        }
    }
}

// =============================================================================
// Backend Invariants
// =============================================================================

mod backend_invariants {
    use super::*;
    use anno::BatchCapable;

    proptest! {
        /// INVARIANT: Backend name is never empty
        #[test]
        fn name_not_empty(backend_idx in 0..3u8) {
            let backend: Box<dyn Model> = match backend_idx {
                0 => Box::new(RegexNER::new()),
                1 => Box::new(HeuristicNER::new()),
                _ => Box::new(StackedNER::new()),
            };

            prop_assert!(!backend.name().is_empty());
        }

        /// INVARIANT: Available backends return Ok for valid input
        #[test]
        fn available_returns_ok(text in "[A-Za-z ]{1,50}") {
            let backends: Vec<Box<dyn Model>> = vec![
                Box::new(RegexNER::new()),
                Box::new(HeuristicNER::new()),
                Box::new(StackedNER::new()),
            ];

            for backend in backends {
                if backend.is_available() {
                    let result = backend.extract_entities(&text, None);
                    prop_assert!(
                        result.is_ok(),
                        "Available backend failed on valid input: {:?}",
                        result.err()
                    );
                }
            }
        }

        /// INVARIANT: Batch extraction returns same number of results as inputs
        #[test]
        fn batch_count_matches_input(
            texts in prop::collection::vec("[A-Za-z ]{1,30}", 1..10)
        ) {
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

            let ner = RegexNER::new();
            let result = ner.extract_entities_batch(&text_refs, None);

            prop_assert!(result.is_ok());
            prop_assert_eq!(
                result.unwrap().len(),
                texts.len(),
                "Batch result count doesn't match input count"
            );
        }
    }

    #[test]
    fn base_backends_always_available() {
        assert!(RegexNER::new().is_available());
        assert!(HeuristicNER::new().is_available());
        assert!(StackedNER::new().is_available());
    }
}

// =============================================================================
// Consistency Invariants
// =============================================================================

mod consistency_invariants {
    use super::*;

    proptest! {
        /// INVARIANT: Same input produces same output (determinism)
        #[test]
        fn deterministic_extraction(text in "[A-Za-z0-9 .,@]{20,100}") {
            let ner = StackedNER::new();

            let result1 = ner.extract_entities(&text, None).unwrap();
            let result2 = ner.extract_entities(&text, None).unwrap();

            prop_assert_eq!(result1.len(), result2.len());

            for (e1, e2) in result1.iter().zip(result2.iter()) {
                prop_assert_eq!(e1.start, e2.start);
                prop_assert_eq!(e1.end, e2.end);
                prop_assert_eq!(&e1.text, &e2.text);
                prop_assert_eq!(&e1.entity_type, &e2.entity_type);
            }
        }

        /// INVARIANT: StackedNER finds at least what RegexNER finds
        #[test]
        fn stacked_includes_pattern(text in "[A-Za-z0-9 .,@$]{20,100}") {
            let pattern = RegexNER::new();
            let stacked = StackedNER::new();

            let pattern_entities = pattern.extract_entities(&text, None).unwrap();
            let stacked_entities = stacked.extract_entities(&text, None).unwrap();

            // Build set of (start, end, type) tuples from stacked
            let stacked_spans: HashSet<_> = stacked_entities
                .iter()
                .map(|e| (e.start, e.end, e.entity_type.clone()))
                .collect();

            // Every pattern entity should be in stacked (or covered by it)
            for pe in &pattern_entities {
                let pattern_key = (pe.start, pe.end, pe.entity_type.clone());

                // Either exact match or overlapping entity of same type
                let covered = stacked_spans.contains(&pattern_key) ||
                    stacked_entities.iter().any(|se| {
                        se.entity_type == pe.entity_type &&
                        se.start <= pe.start &&
                        se.end >= pe.end
                    });

                prop_assert!(
                    covered,
                    "Pattern entity {:?} not found in stacked results",
                    pe
                );
            }
        }
    }
}

// =============================================================================
// Type Safety Invariants
// =============================================================================

mod type_safety_invariants {
    use super::*;

    #[test]
    fn entity_type_equality_symmetric() {
        let types = [
            EntityType::Person,
            EntityType::Organization,
            EntityType::Location,
            EntityType::Date,
            EntityType::Time,
            EntityType::Money,
            EntityType::Percent,
            EntityType::Email,
            EntityType::Phone,
            EntityType::Url,
            EntityType::Other("Custom".to_string()),
        ];

        for t1 in &types {
            for t2 in &types {
                // Symmetric: t1 == t2 implies t2 == t1
                assert_eq!(t1 == t2, t2 == t1);
            }
        }
    }

    #[test]
    fn entity_type_equality_reflexive() {
        let types = [
            EntityType::Person,
            EntityType::Organization,
            EntityType::Other("Test".to_string()),
        ];

        for t in &types {
            assert_eq!(t, t, "EntityType should be reflexively equal");
        }
    }

    proptest! {
        /// Custom entity types with same string are equal
        #[test]
        fn custom_type_string_equality(name in "[A-Z_]{1,20}") {
            let t1 = EntityType::Other(name.clone());
            let t2 = EntityType::Other(name.clone());

            prop_assert_eq!(t1, t2);
        }

        /// Custom entity types with different strings are not equal
        #[test]
        fn custom_type_string_inequality(
            name1 in "[A-Z_]{1,20}",
            name2 in "[a-z_]{1,20}"
        ) {
            // name1 is uppercase, name2 is lowercase, so they should differ
            if name1.to_lowercase() != name2.to_lowercase() {
                let t1 = EntityType::Other(name1);
                let t2 = EntityType::Other(name2);

                prop_assert_ne!(t1, t2);
            }
        }
    }
}

// =============================================================================
// Memory Safety Invariants (via borrow checker)
// =============================================================================

mod memory_safety_invariants {
    use super::*;

    #[test]
    fn entity_can_be_cloned() {
        let entity = Entity::new("Test", EntityType::Person, 0, 4, 0.9);
        let cloned = entity.clone();

        assert_eq!(entity.text, cloned.text);
        assert_eq!(entity.start, cloned.start);
        assert_eq!(entity.end, cloned.end);
        assert_eq!(entity.entity_type, cloned.entity_type);
    }

    #[test]
    fn entity_type_can_be_cloned() {
        let types = [
            EntityType::Person,
            EntityType::Organization,
            EntityType::Other("Custom".to_string()),
        ];

        for t in &types {
            let cloned = t.clone();
            assert_eq!(t, &cloned);
        }
    }

    #[test]
    fn entities_can_be_moved_to_vec() {
        let ner = RegexNER::new();
        let entities = ner.extract_entities("test@example.com", None).unwrap();

        // Move entities to a new vec
        let moved: Vec<Entity> = entities.into_iter().collect();

        // Should still work
        assert!(!moved.is_empty() || moved.is_empty()); // Always true, but uses moved
    }
}
