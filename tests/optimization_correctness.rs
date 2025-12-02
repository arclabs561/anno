//! Comprehensive correctness tests for optimizations.
//!
//! These tests ensure that optimized code paths produce identical results
//! to unoptimized versions, and that optimizations don't introduce bugs.

use anno::{Entity, EntityType, Model, StackedNER};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Property: extract_text_with_len matches extract_text for all valid inputs
    #[test]
    fn extract_text_optimization_correctness(
        text in ".{0,1000}",
        start in 0usize..2000,
        end in 0usize..2000
    ) {
        let text_char_count = text.chars().count();
        let start = start.min(text_char_count);
        let end = end.min(text_char_count).max(start);

        let entity = Entity::new("test", EntityType::Person, start, end, 0.5);

        let result_optimized = entity.extract_text_with_len(&text, text_char_count);
        let result_original = entity.extract_text(&text);

        prop_assert_eq!(
            result_optimized, result_original,
            "Optimized extract_text_with_len should match original: start={}, end={}, text_len={}",
            start, end, text_char_count
        );
    }

    /// Property: validate_with_len matches validate for all valid inputs
    #[test]
    fn validate_optimization_correctness(
        text in ".{0,1000}",
        start in 0usize..2000,
        end in 0usize..2000
    ) {
        let text_char_count = text.chars().count();
        let start = start.min(text_char_count);
        let end = end.min(text_char_count).max(start);

        // Create entity with text matching the span
        let entity_text: String = text.chars().skip(start).take(end - start).collect();
        let entity = Entity::new(&entity_text, EntityType::Person, start, end, 0.5);

        let result_optimized = entity.validate_with_len(&text, text_char_count);
        let result_original = entity.validate(&text);

        // Should produce same number of issues
        prop_assert_eq!(
            result_optimized.len(), result_original.len(),
            "Optimized validate_with_len should produce same number of issues"
        );

        // All issue types should match
        for (opt_issue, orig_issue) in result_optimized.iter().zip(result_original.iter()) {
            prop_assert_eq!(
                std::mem::discriminant(opt_issue),
                std::mem::discriminant(orig_issue),
                "Issue types should match between optimized and original"
            );
        }
    }

    /// Property: StackedNER with cached length produces identical results
    #[test]
    fn stacked_ner_cached_length_correctness(text in ".{0,500}") {
        let ner = StackedNER::default();

        // Call multiple times - should be deterministic
        let entities1 = ner.extract_entities(&text, None).unwrap();
        let entities2 = ner.extract_entities(&text, None).unwrap();
        let entities3 = ner.extract_entities(&text, None).unwrap();

        // All calls should produce identical results
        prop_assert_eq!(entities1.len(), entities2.len());
        prop_assert_eq!(entities2.len(), entities3.len());

        // Compare entity sets (order may vary)
        // Use integer representation of confidence to avoid f64 comparison issues
        let entities1_set: std::collections::HashSet<_> = entities1.iter()
            .map(|e| (e.start, e.end, e.entity_type.clone(), e.text.clone(), (e.confidence * 1000.0) as u64))
            .collect();
        let entities2_set: std::collections::HashSet<_> = entities2.iter()
            .map(|e| (e.start, e.end, e.entity_type.clone(), e.text.clone(), (e.confidence * 1000.0) as u64))
            .collect();
        let entities3_set: std::collections::HashSet<_> = entities3.iter()
            .map(|e| (e.start, e.end, e.entity_type.clone(), e.text.clone(), (e.confidence * 1000.0) as u64))
            .collect();

        prop_assert_eq!(entities1_set.clone(), entities2_set.clone(), "First two calls should match");
        prop_assert_eq!(entities2_set, entities3_set, "Second and third calls should match");
    }

    /// Property: All entities from StackedNER are valid (optimization doesn't break validation)
    #[test]
    fn stacked_ner_all_entities_valid_after_optimization(text in ".{0,500}") {
        let ner = StackedNER::default();
        let entities = ner.extract_entities(&text, None).unwrap();
        let text_char_count = text.chars().count();

        for entity in entities {
            // Use optimized validation method
            let issues = entity.validate_with_len(&text, text_char_count);

            // Should not have span bounds issues (optimization should handle this)
            for issue in &issues {
                match issue {
                    anno::ValidationIssue::SpanOutOfBounds { .. } |
                    anno::ValidationIssue::InvalidSpan { .. } => {
                        prop_assert!(
                            false,
                            "Optimization should prevent span bounds issues: start={}, end={}, text_len={}",
                            entity.start, entity.end, text_char_count
                        );
                    }
                    _ => {
                        // Other issues are acceptable
                    }
                }
            }

            // Confidence should be valid
            prop_assert!(
                entity.confidence >= 0.0 && entity.confidence <= 1.0,
                "Entity confidence should be valid: {}",
                entity.confidence
            );
        }
    }

    /// Property: Entity extraction handles edge cases correctly with optimizations
    #[test]
    fn entity_extraction_edge_cases_optimized(
        text in ".{0,500}",
        start in 0usize..1000,
        end in 0usize..1000
    ) {
        let text_char_count = text.chars().count();
        let start = start.min(text_char_count);
        let end = end.min(text_char_count).max(start);

        let entity = Entity::new("test", EntityType::Person, start, end, 0.5);

        // Both methods should handle all edge cases identically
        let result1 = entity.extract_text(&text);
        let result2 = entity.extract_text_with_len(&text, text_char_count);

        prop_assert_eq!(
            result1.clone(), result2,
            "Both methods should handle edge cases identically"
        );

        // If out of bounds, both should return empty string
        if start >= text_char_count || end > text_char_count || start >= end {
            prop_assert_eq!(
                result1, "",
                "Out-of-bounds entity should return empty string"
            );
        }
    }
}
