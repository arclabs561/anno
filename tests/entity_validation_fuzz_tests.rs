//! Property-based tests for entity validation.
//!
//! These tests verify that entity validation correctly identifies
//! invalid entities and handles edge cases.

use anno::{Entity, EntityType, ValidationIssue};
use proptest::prelude::*;

proptest! {

    /// Entity validation should catch invalid spans (start >= end).
    #[test]
    fn entity_validation_catches_invalid_spans(
        text in ".{0,100}",
        start in 0usize..200,
        end in 0usize..200,
    ) {
        let char_count = text.chars().count();
        let entity = Entity::new("test", EntityType::Person, start, end, 0.9);
        let issues = entity.validate(&text);

        if start >= end {
            // Invalid span should be caught
            let has_invalid_span = issues.iter().any(|i| matches!(i, ValidationIssue::InvalidSpan {..}));
            prop_assert!(has_invalid_span);
        } else if end > char_count {
            // Out of bounds should be caught
            let has_out_of_bounds = issues.iter().any(|i| matches!(i, ValidationIssue::SpanOutOfBounds {..}));
            prop_assert!(has_out_of_bounds);
        }
        // Valid spans may have text mismatch issues, which is expected
    }

    /// Entity validation should handle confidence bounds correctly.
    #[test]
    fn entity_validation_confidence_bounds(
        confidence in -1.0f64..2.0f64,
    ) {
        let entity = Entity::new("test", EntityType::Person, 0, 4, confidence);
        // Entity::new clamps confidence to [0.0, 1.0], so validation should pass
        let issues = entity.validate("test");
        let has_invalid_confidence = issues.iter().any(|i| matches!(i, ValidationIssue::InvalidConfidence {..}));
        prop_assert!(!has_invalid_confidence);
    }

    /// Entity validation should catch text mismatches.
    #[test]
    fn entity_validation_text_mismatch(
        text in ".{10,100}",
        wrong_text in ".{1,50}",
        start in 0usize..50,
        end in 0usize..50,
    ) {
        let char_count = text.chars().count();
        if start < end && end <= char_count {
            // Get the actual text at the span using character-based slicing
            let actual_text: String = text.chars().skip(start).take(end - start).collect();
            if wrong_text != actual_text {
                let entity = Entity::new(&wrong_text, EntityType::Person, start, end, 0.9);
                let issues = entity.validate(&text);

                // Should detect text mismatch
                let has_text_mismatch = issues.iter().any(|i| matches!(i, ValidationIssue::TextMismatch {..}));
                prop_assert!(has_text_mismatch);
            }
        }
    }

    /// Entity validation should handle empty text correctly.
    #[test]
    fn entity_validation_empty_text(
        start in 0usize..10,
        end in 0usize..10,
    ) {
        let entity = Entity::new("test", EntityType::Person, start, end, 0.9);
        let issues = entity.validate("");

        if start >= end {
            let has_invalid_span = issues.iter().any(|i| matches!(i, ValidationIssue::InvalidSpan {..}));
            prop_assert!(has_invalid_span);
        } else if end > 0 {
            let has_out_of_bounds = issues.iter().any(|i| matches!(i, ValidationIssue::SpanOutOfBounds {..}));
            prop_assert!(has_out_of_bounds);
        }
    }

    /// Entity validation should handle Unicode text correctly.
    #[test]
    fn entity_validation_unicode_text(
        text in ".*",  // Any Unicode
        start in 0usize..200,
        end in 0usize..200,
    ) {
        let char_count = text.chars().count();
        let entity = Entity::new("test", EntityType::Person, start, end, 0.9);
        let issues = entity.validate(&text);

        // Should not panic on Unicode
        // Validation should work correctly with character offsets
        if start < end && end <= char_count {
            // Valid span - might have text mismatch but should not have span issues
            let has_invalid_span = issues.iter().any(|i| matches!(i, ValidationIssue::InvalidSpan {..}));
            let has_out_of_bounds = issues.iter().any(|i| matches!(i, ValidationIssue::SpanOutOfBounds {..}));
            prop_assert!(!has_invalid_span);
            prop_assert!(!has_out_of_bounds);
        }
    }

    /// Entity validation should handle very long text.
    #[test]
    fn entity_validation_long_text(
        text in ".{1000,5000}",  // Very long text
        start in 0usize..1000,
        end in 0usize..1000,
    ) {
        let char_count = text.chars().count();
        let entity = Entity::new("test", EntityType::Person, start, end, 0.9);
        let issues = entity.validate(&text);

        // Should not panic on long text
        if start < end && end <= char_count {
            let has_out_of_bounds = issues.iter().any(|i| matches!(i, ValidationIssue::SpanOutOfBounds {..}));
            prop_assert!(!has_out_of_bounds);
        }
    }

    /// Entity validation should be idempotent.
    #[test]
    fn entity_validation_idempotent(
        text in ".{0,100}",
        start in 0usize..100,
        end in 0usize..100,
    ) {
        let entity = Entity::new("test", EntityType::Person, start, end, 0.9);
        let issues1 = entity.validate(&text);
        let issues2 = entity.validate(&text);

        // Should produce same issues
        prop_assert_eq!(issues1.len(), issues2.len(),
            "Validation not idempotent: {} issues != {} issues", issues1.len(), issues2.len());
    }
}
