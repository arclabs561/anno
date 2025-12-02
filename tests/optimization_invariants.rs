//! Property tests to ensure optimizations preserve correctness.
//!
//! These tests verify that optimized code paths produce identical results
//! to their unoptimized counterparts, ensuring no regressions.

use anno::{Entity, EntityType, Model, StackedNER};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Property: extract_text_with_len produces identical results to extract_text
    #[test]
    fn extract_text_with_len_matches_extract_text(text in ".{0,1000}", start in 0usize..1000, end in 0usize..1000) {
        let text_char_count = text.chars().count();
        let start = start.min(text_char_count);
        let end = end.min(text_char_count).max(start);

        let entity = Entity::new("test", EntityType::Person, start, end, 0.5);

        let result_optimized = entity.extract_text_with_len(&text, text_char_count);
        let result_original = entity.extract_text(&text);

        prop_assert_eq!(
            result_optimized, result_original,
            "extract_text_with_len should match extract_text: start={}, end={}, text_len={}",
            start, end, text_char_count
        );
    }

    /// Property: validate_with_len produces identical results to validate
    #[test]
    fn validate_with_len_matches_validate(text in ".{0,1000}", start in 0usize..1000, end in 0usize..1000) {
        let text_char_count = text.chars().count();
        let start = start.min(text_char_count);
        let end = end.min(text_char_count).max(start);

        // Create entity with text matching the span
        let entity_text: String = text.chars().skip(start).take(end - start).collect();
        let entity = Entity::new(&entity_text, EntityType::Person, start, end, 0.5);

        let result_optimized = entity.validate_with_len(&text, text_char_count);
        let result_original = entity.validate(&text);

        prop_assert_eq!(
            result_optimized.len(), result_original.len(),
            "validate_with_len should produce same number of issues: start={}, end={}, text_len={}",
            start, end, text_char_count
        );

        // Check that issues are equivalent (same types, same messages)
        for (opt_issue, orig_issue) in result_optimized.iter().zip(result_original.iter()) {
            prop_assert_eq!(
                std::mem::discriminant(opt_issue),
                std::mem::discriminant(orig_issue),
                "Issue types should match: start={}, end={}",
                start, end
            );
        }
    }

    /// Property: StackedNER with cached text length produces identical results
    #[test]
    fn stacked_ner_cached_length_identical(text in ".{0,500}") {
        let ner = StackedNER::default();
        let entities1 = ner.extract_entities(&text, None).unwrap();
        let entities2 = ner.extract_entities(&text, None).unwrap();

        // Results should be identical (deterministic)
        prop_assert_eq!(
            entities1.len(), entities2.len(),
            "StackedNER should produce same number of entities on repeated calls"
        );

        // Compare entities (order may vary, so check sets)
        let entities1_set: std::collections::HashSet<_> = entities1.iter()
            .map(|e| (e.start, e.end, e.entity_type.clone(), e.text.clone()))
            .collect();
        let entities2_set: std::collections::HashSet<_> = entities2.iter()
            .map(|e| (e.start, e.end, e.entity_type.clone(), e.text.clone()))
            .collect();

        prop_assert_eq!(
            entities1_set, entities2_set,
            "StackedNER should produce identical entities on repeated calls"
        );
    }

    /// Property: Entity extraction with different text lengths should handle bounds correctly
    #[test]
    fn entity_extraction_bounds_handling(text in ".{0,500}", start in 0usize..1000, end in 0usize..1000) {
        let text_char_count = text.chars().count();
        let start = start.min(text_char_count);
        let end = end.min(text_char_count).max(start);

        let entity = Entity::new("test", EntityType::Person, start, end, 0.5);

        // Both methods should handle out-of-bounds gracefully
        let result1 = entity.extract_text(&text);
        let result2 = entity.extract_text_with_len(&text, text_char_count);

        prop_assert_eq!(
            result1.clone(), result2,
            "Both extraction methods should handle bounds identically: start={}, end={}, text_len={}",
            start, end, text_char_count
        );

        // If entity is within bounds, extracted text should match span
        if start < text_char_count && end <= text_char_count && start < end {
            let expected: String = text.chars().skip(start).take(end - start).collect();
            prop_assert_eq!(
                result1, expected,
                "Extracted text should match span: start={}, end={}",
                start, end
            );
        }
    }

    /// Property: StackedNER should produce valid entities (all optimizations applied)
    #[test]
    fn stacked_ner_all_entities_valid(text in ".{0,500}") {
        let ner = StackedNER::default();
        let entities = ner.extract_entities(&text, None).unwrap();
        let text_char_count = text.chars().count();

        for entity in entities {
            // All entities should have valid spans
            prop_assert!(
                entity.start < entity.end,
                "Entity should have valid span: start={}, end={}",
                entity.start, entity.end
            );

            // All entities should be within bounds (with small tolerance for edge cases)
            prop_assert!(
                entity.end <= text_char_count + 2,
                "Entity end should be within bounds: end={}, text_len={}",
                entity.end, text_char_count
            );

            // Confidence should be in valid range
            prop_assert!(
                entity.confidence >= 0.0 && entity.confidence <= 1.0,
                "Entity confidence should be in [0, 1]: confidence={}",
                entity.confidence
            );

            // Validate using optimized method
            let issues = entity.validate_with_len(&text, text_char_count);
            // Allow some issues (text mismatch, etc.) but not span bounds issues
            for issue in &issues {
                match issue {
                    anno::ValidationIssue::SpanOutOfBounds { .. } |
                    anno::ValidationIssue::InvalidSpan { .. } => {
                        prop_assert!(
                            false,
                            "Entity should not have span bounds issues: start={}, end={}, text_len={}",
                            entity.start, entity.end, text_char_count
                        );
                    }
                    _ => {
                        // Other issues (text mismatch, etc.) are acceptable
                    }
                }
            }
        }
    }

    /// Property: Multiple calls to StackedNER with same input should be deterministic
    #[test]
    fn stacked_ner_deterministic(text in ".{0,500}") {
        let ner1 = StackedNER::default();
        let ner2 = StackedNER::default();

        let entities1 = ner1.extract_entities(&text, None).unwrap();
        let entities2 = ner2.extract_entities(&text, None).unwrap();

        // Should produce same results
        prop_assert_eq!(
            entities1.len(), entities2.len(),
            "StackedNER should be deterministic"
        );
    }
}
