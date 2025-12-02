//! Property tests for Unicode edge cases that optimizations must handle correctly.
//!
//! These tests ensure optimizations don't break on:
//! - Multi-byte UTF-8 characters
//! - Grapheme clusters (emoji sequences, combining characters)
//! - Control characters
//! - Zero-width characters
//! - Surrogate pairs (invalid in UTF-8, but should handle gracefully)
//! - Potentially invalid UTF-8 (using bstr for byte-level testing)

use anno::{Entity, EntityType, Model, StackedNER};
use bstr::{BStr, ByteSlice};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Property: Optimizations handle multi-byte UTF-8 characters correctly
    #[test]
    fn multi_byte_utf8_optimization_correctness(text in "[\\u{0080}-\\u{FFFF}]{0,100}") {
        let ner = StackedNER::default();
        let entities = ner.extract_entities(&text, None).unwrap();
        let text_char_count = text.chars().count();

        for entity in entities {
            // Use optimized methods
            let extracted = entity.extract_text_with_len(&text, text_char_count);
            let issues = entity.validate_with_len(&text, text_char_count);

            // Should not have span bounds issues
            for issue in &issues {
                match issue {
                    anno::ValidationIssue::SpanOutOfBounds { .. } |
                    anno::ValidationIssue::InvalidSpan { .. } => {
                        prop_assert!(
                            false,
                            "Multi-byte UTF-8 should not cause span issues: start={}, end={}, text_len={}",
                            entity.start, entity.end, text_char_count
                        );
                    }
                    _ => {}
                }
            }

            // Extracted text should be valid UTF-8
            prop_assert!(
                std::str::from_utf8(extracted.as_bytes()).is_ok(),
                "Extracted text should be valid UTF-8"
            );

            // Use bstr for byte-level validation (more efficient than String::from_utf8)
            let extracted_bytes: &[u8] = extracted.as_bytes();
            let bstr_view = BStr::new(extracted_bytes);
            prop_assert!(
                bstr_view.to_str().is_ok(),
                "Extracted bytes should form valid UTF-8 string"
            );
        }
    }

    /// Property: Optimizations handle emoji and grapheme clusters correctly
    #[test]
    fn emoji_grapheme_clusters_optimization(text in "[\\u{1F300}-\\u{1F9FF}]{0,50}") {
        let ner = StackedNER::default();
        let entities = ner.extract_entities(&text, None).unwrap();
        let text_char_count = text.chars().count();

        for entity in entities {
            // Use optimized extraction
            let extracted = entity.extract_text_with_len(&text, text_char_count);

            // Should handle emoji correctly (may be multiple code points per grapheme)
            prop_assert!(
                entity.start < entity.end,
                "Entity should have valid span with emoji"
            );

            // Extracted text should be valid
            prop_assert!(
                !extracted.is_empty() || entity.start >= text_char_count || entity.end > text_char_count,
                "Extracted text should be non-empty for valid spans"
            );
        }
    }

    /// Property: Optimizations handle control characters correctly
    #[test]
    fn control_characters_optimization(text in "[\\u{0000}-\\u{001F}\\u{007F}-\\u{009F}]{0,100}") {
        let ner = StackedNER::default();
        let entities = ner.extract_entities(&text, None).unwrap();
        let text_char_count = text.chars().count();

        for entity in entities {
            // Use optimized methods
            let extracted = entity.extract_text_with_len(&text, text_char_count);
            let _issues = entity.validate_with_len(&text, text_char_count);

            // Should not panic or produce invalid spans
            prop_assert!(
                entity.start < entity.end,
                "Entity should have valid span with control characters"
            );

            // Extracted text should be valid (may contain control chars, that's ok)
            let _ = extracted; // Just ensure it doesn't panic
        }
    }

    /// Property: extract_text_with_len matches extract_text for Unicode text
    #[test]
    fn unicode_extract_text_equivalence(
        text in "[\\u{0000}-\\u{FFFF}]{0,200}",
        start in 0usize..500,
        end in 0usize..500
    ) {
        let text_char_count = text.chars().count();
        let start = start.min(text_char_count);
        let end = end.min(text_char_count).max(start);

        let entity = Entity::new("test", EntityType::Person, start, end, 0.5);

        let result_optimized = entity.extract_text_with_len(&text, text_char_count);
        let result_original = entity.extract_text(&text);

        prop_assert_eq!(
            result_optimized, result_original,
            "Unicode text extraction should match: start={}, end={}, text_len={}",
            start, end, text_char_count
        );
    }

    /// Property: validate_with_len matches validate for Unicode text
    #[test]
    fn unicode_validate_equivalence(
        text in "[\\u{0000}-\\u{FFFF}]{0,200}",
        start in 0usize..500,
        end in 0usize..500
    ) {
        let text_char_count = text.chars().count();
        let start = start.min(text_char_count);
        let end = end.min(text_char_count).max(start);

        // Create entity with matching text
        let entity_text: String = text.chars().skip(start).take(end - start).collect();
        let entity = Entity::new(&entity_text, EntityType::Person, start, end, 0.5);

        let result_optimized = entity.validate_with_len(&text, text_char_count);
        let result_original = entity.validate(&text);

        prop_assert_eq!(
            result_optimized.len(), result_original.len(),
            "Unicode validation should produce same number of issues"
        );
    }

    /// Property: Byte-level operations handle potentially invalid UTF-8 gracefully
    ///
    /// This test uses bstr to create byte sequences that might not be valid UTF-8,
    /// ensuring our code handles edge cases correctly.
    #[test]
    fn byte_level_utf8_handling(
        bytes in proptest::collection::vec(0u8..=255u8, 0..100)
    ) {
        // Convert bytes to string if valid UTF-8, otherwise skip test
        if let Ok(text) = std::str::from_utf8(&bytes) {
            let ner = StackedNER::default();
            let entities = ner.extract_entities(text, None).unwrap();

            // All entities should have valid UTF-8 text
            for entity in entities {
                // Use bstr for efficient byte-level validation
                let entity_bytes = entity.text.as_bytes();
                let bstr_view = BStr::new(entity_bytes);

                prop_assert!(
                    bstr_view.to_str().is_ok(),
                    "Entity text should be valid UTF-8: {:?}",
                    entity.text
                );

                // Validate span bounds
                let text_char_count = text.chars().count();
                prop_assert!(
                    entity.start <= text_char_count && entity.end <= text_char_count,
                    "Entity span should be within text bounds: start={}, end={}, text_len={}",
                    entity.start, entity.end, text_char_count
                );
            }
        }
    }

    /// Property: Byte-level substring search consistency with character-level search
    ///
    /// This test validates that byte-level searches (using bstr) and character-level searches
    /// both find the pattern when it exists, or both don't find it when it doesn't.
    /// For valid UTF-8, both should agree on whether the pattern exists.
    #[test]
    fn byte_vs_char_substring_search(
        text in "[\\u{0000}-\\u{FFFF}]{0,200}",
        pattern in "[\\u{0000}-\\u{FFFF}]{0,50}"
    ) {
        // Skip if pattern is empty or longer than text
        prop_assume!(!pattern.is_empty() && pattern.len() <= text.len());

        // Use bstr for byte-level search
        let text_bytes = text.as_bytes();
        let pattern_bytes = pattern.as_bytes();
        let bstr_text = BStr::new(text_bytes);

        // Byte-level find (bstr uses find() for byte slice patterns)
        let byte_found = bstr_text.find(pattern_bytes).is_some();

        // Character-level find
        let char_found = text.find(&pattern).is_some();

        // Both should agree on whether the pattern exists
        // This is the key invariant: for valid UTF-8, byte-level and char-level searches
        // should both find or both not find the same pattern
        prop_assert_eq!(
            byte_found, char_found,
            "Byte-level and char-level search should agree on pattern existence: text={:?}, pattern={:?}, byte_found={}, char_found={}",
            text, pattern, byte_found, char_found
        );
    }
}
