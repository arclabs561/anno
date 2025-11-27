//! Fuzzing and edge case tests for robustness.
//!
//! These tests focus on:
//! - Unicode handling (RTL, combining chars, emoji, CJK)
//! - Boundary conditions (empty input, max lengths)
//! - Malformed input (invalid spans, overlapping)
//! - Adversarial cases (injection attempts, format strings)

use anno::{Entity, EntityBuilder, EntityType, Model, PatternNER, StatisticalNER};
use proptest::prelude::*;

// =============================================================================
// Unicode Edge Cases
// =============================================================================

#[test]
fn unicode_emoji_in_text() {
    let ner = PatternNER::new();
    let text = "Contact me at test@example.com on January 15, 2024";
    let entities = ner.extract_entities(text, None).unwrap();

    // Should still find email and date
    assert!(entities.iter().any(|e| e.entity_type == EntityType::Email));
    assert!(entities.iter().any(|e| e.entity_type == EntityType::Date));
}

#[test]
fn unicode_rtl_text() {
    // Arabic text mixed with English
    let ner = PatternNER::new();
    let text = "مرحبا test@example.com مرحبا";
    let entities = ner.extract_entities(text, Some("ar")).unwrap();

    // Should find email even in RTL context
    assert!(entities.iter().any(|e| e.entity_type == EntityType::Email));
}

#[test]
fn unicode_cjk_text() {
    let ner = StatisticalNER::new();
    let text = "日本東京 is a beautiful city. Visit Tokyo!";
    let entities = ner.extract_entities(text, None).unwrap();

    // Should handle CJK characters gracefully
    for e in &entities {
        assert!(e.start <= e.end);
        assert!(e.end <= text.len());
    }
}

#[test]
fn unicode_combining_characters() {
    let ner = PatternNER::new();
    // Text with combining diacritical marks: é = e + combining acute
    let text = "Contact me\u{0301} at test@example.com";
    let entities = ner.extract_entities(text, None).unwrap();

    // Should still find email
    assert!(entities.iter().any(|e| e.entity_type == EntityType::Email));
}

#[test]
fn unicode_zero_width_chars() {
    let ner = PatternNER::new();
    // Zero-width space and joiner
    let text = "test\u{200B}@example.com"; // Zero-width space
    let entities = ner.extract_entities(text, None).unwrap();

    // Pattern might or might not match - just shouldn't panic
    for e in &entities {
        assert!(e.start <= e.end);
    }
}

#[test]
fn unicode_surrogate_pairs() {
    let ner = PatternNER::new();
    // Text with characters outside BMP (emoji)
    let text = "📧 test@example.com 🎉 January 15, 2024";
    let entities = ner.extract_entities(text, None).unwrap();

    // Entity offsets are CHARACTER offsets (not byte offsets)
    let char_count = text.chars().count();
    for e in &entities {
        assert!(e.start <= char_count, "Start {} beyond char count {} for {}", e.start, char_count, e.text);
        assert!(e.end <= char_count, "End {} beyond char count {} for {}", e.end, char_count, e.text);
        // Verify extracted text matches
        let extracted: String = text.chars().skip(e.start).take(e.end - e.start).collect();
        assert_eq!(extracted, e.text, "Text mismatch for entity at {}..{}", e.start, e.end);
    }
}

// =============================================================================
// Boundary Conditions
// =============================================================================

#[test]
fn empty_text_input() {
    let ner = PatternNER::new();
    let entities = ner.extract_entities("", None).unwrap();
    assert!(entities.is_empty());

    let ner2 = StatisticalNER::new();
    let entities2 = ner2.extract_entities("", None).unwrap();
    assert!(entities2.is_empty());
}

#[test]
fn whitespace_only_text() {
    let ner = PatternNER::new();
    let entities = ner.extract_entities("   \t\n\r   ", None).unwrap();
    assert!(entities.is_empty());
}

#[test]
fn single_character_text() {
    let ner = PatternNER::new();
    let entities = ner.extract_entities("x", None).unwrap();
    // Should not panic
    for e in &entities {
        assert!(e.start <= e.end);
        assert!(e.end <= 1);
    }
}

#[test]
fn very_long_text() {
    let ner = PatternNER::new();
    // 100KB of repeated text
    let unit = "John Smith visited test@example.com on January 15, 2024. ";
    let text: String = unit.repeat(2000);

    let entities = ner.extract_entities(&text, None).unwrap();

    // Should find many entities
    assert!(!entities.is_empty());

    // All entities should have valid offsets
    for e in &entities {
        assert!(e.start <= e.end);
        assert!(e.end <= text.len());
    }
}

#[test]
fn max_span_length() {
    // Test entity with very long text
    let long_name = "A".repeat(10000);
    let entity = Entity::new(&long_name, EntityType::Person, 0, long_name.len(), 0.9);

    assert_eq!(entity.text.len(), 10000);
    assert_eq!(entity.start, 0);
    assert_eq!(entity.end, 10000);
}

// =============================================================================
// Malformed Input Handling
// =============================================================================

#[test]
fn entity_span_start_equals_end() {
    // Zero-length span
    let entity = Entity::new("", EntityType::Person, 5, 5, 0.9);
    assert_eq!(entity.start, entity.end);
    assert_eq!(entity.total_len(), 0);
}

#[test]
fn entity_confidence_clamping() {
    // Out of range confidence values
    let e1 = Entity::new("Test", EntityType::Person, 0, 4, -0.5);
    let e2 = Entity::new("Test", EntityType::Person, 0, 4, 1.5);
    let e3 = Entity::new("Test", EntityType::Person, 0, 4, f64::INFINITY);
    let e4 = Entity::new("Test", EntityType::Person, 0, 4, f64::NEG_INFINITY);

    assert_eq!(e1.confidence, 0.0);
    assert_eq!(e2.confidence, 1.0);
    assert!(e3.confidence <= 1.0);
    assert!(e4.confidence >= 0.0);

    // Note: NaN.clamp(0.0, 1.0) returns NaN in Rust - this is expected behavior.
    // Applications should validate confidence before constructing entities.
}

#[test]
fn entity_with_mismatched_text_span() {
    // Text doesn't match span length
    let entity = Entity::new("Hello", EntityType::Person, 0, 100, 0.9);

    // Entity should still be valid
    assert_eq!(entity.text, "Hello");
    assert_eq!(entity.end, 100);
}

// =============================================================================
// Adversarial Inputs
// =============================================================================

#[test]
fn format_string_injection() {
    let ner = PatternNER::new();
    let text = "Contact %s at %d or {format} or ${var}";
    let entities = ner.extract_entities(text, None).unwrap();

    // Should not panic or produce weird output
    for e in &entities {
        assert!(!e.text.contains("%s"));
        assert!(!e.text.contains("${"));
    }
}

#[test]
fn regex_metacharacters() {
    let ner = PatternNER::new();
    // Text with regex metacharacters
    let text = "Contact me at test@example.com (or test.*@example.com)";
    let entities = ner.extract_entities(text, None).unwrap();

    // Should find the real email, not match the pattern as regex
    assert!(entities.iter().any(|e| e.text == "test@example.com"));
}

#[test]
fn null_bytes() {
    let ner = PatternNER::new();
    // Rust strings can't have null bytes, but handle gracefully
    let text = "test\0@example.com";
    let entities = ner.extract_entities(text, None).unwrap();

    // Should handle gracefully
    for e in &entities {
        assert!(e.start <= e.end);
    }
}

#[test]
fn repeated_special_chars() {
    let ner = PatternNER::new();
    let text = "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@";
    let entities = ner.extract_entities(text, None).unwrap();

    // Should not produce exponential matches
    assert!(entities.len() < 100);
}

// =============================================================================
// Property-Based Fuzzing
// =============================================================================

proptest! {
    /// PatternNER should never panic on arbitrary ASCII input.
    #[test]
    fn pattern_ner_never_panics_ascii(text in "[ -~]{0,500}") {
        let ner = PatternNER::new();
        let result = ner.extract_entities(&text, None);
        prop_assert!(result.is_ok());
    }

    /// PatternNER should never panic on arbitrary UTF-8 input.
    #[test]
    fn pattern_ner_never_panics_utf8(text in ".{0,200}") {
        let ner = PatternNER::new();
        let result = ner.extract_entities(&text, None);
        prop_assert!(result.is_ok());
    }

    /// StatisticalNER should never panic on arbitrary input.
    #[test]
    fn statistical_ner_never_panics(text in ".{0,200}") {
        let ner = StatisticalNER::new();
        let result = ner.extract_entities(&text, None);
        prop_assert!(result.is_ok());
    }

    /// Entity offsets should always be valid.
    #[test]
    fn entity_offsets_valid(
        start in 0usize..1000,
        len in 1usize..100
    ) {
        let entity = Entity::new("test", EntityType::Person, start, start + len, 0.9);
        prop_assert!(entity.start <= entity.end);
        prop_assert_eq!(entity.end - entity.start, len);
    }

    /// EntityBuilder should produce valid entities.
    #[test]
    fn entity_builder_produces_valid(
        start in 0usize..1000,
        len in 1usize..100,
        conf in 0.0f64..=1.0f64
    ) {
        let entity = EntityBuilder::new("test", EntityType::Person)
            .span(start, start + len)
            .confidence(conf)
            .build();

        prop_assert!(entity.start <= entity.end);
        prop_assert!(entity.confidence >= 0.0);
        prop_assert!(entity.confidence <= 1.0);
    }

    /// Extracted entities should have offsets within text bounds.
    #[test]
    fn extracted_offsets_within_bounds(text in "[A-Za-z0-9 @.]{10,100}") {
        let ner = PatternNER::new();
        if let Ok(entities) = ner.extract_entities(&text, None) {
            for e in &entities {
                prop_assert!(e.start <= e.end, "start > end");
                prop_assert!(e.end <= text.len(), "end > text.len()");
            }
        }
    }

    /// No overlapping entities from single backend.
    #[test]
    fn no_overlapping_entities_single_backend(text in "[A-Za-z0-9 @.]{10,100}") {
        let ner = PatternNER::new();
        if let Ok(entities) = ner.extract_entities(&text, None) {
            for (i, e1) in entities.iter().enumerate() {
                for e2 in entities.iter().skip(i + 1) {
                    let overlaps = e1.start < e2.end && e2.start < e1.end;
                    if overlaps {
                        // Same-type overlaps are not allowed
                        prop_assert!(e1.entity_type != e2.entity_type,
                            "Overlapping same-type entities: {:?} and {:?}", e1, e2);
                    }
                }
            }
        }
    }
}

// =============================================================================
// Mutation Testing Support
// =============================================================================

/// These tests verify specific behaviors that mutation testing would target.
mod mutation_targets {
    use super::*;

    #[test]
    fn confidence_boundary_zero() {
        let e = Entity::new("Test", EntityType::Person, 0, 4, 0.0);
        assert_eq!(e.confidence, 0.0);
    }

    #[test]
    fn confidence_boundary_one() {
        let e = Entity::new("Test", EntityType::Person, 0, 4, 1.0);
        assert_eq!(e.confidence, 1.0);
    }

    #[test]
    fn span_boundary_conditions() {
        // Start at 0
        let e1 = Entity::new("Test", EntityType::Person, 0, 4, 0.9);
        assert_eq!(e1.start, 0);

        // Start equals end (zero-length)
        let e2 = Entity::new("", EntityType::Person, 5, 5, 0.9);
        assert_eq!(e2.total_len(), 0);
    }

    #[test]
    fn entity_type_equality() {
        assert_eq!(EntityType::Person, EntityType::Person);
        assert_ne!(EntityType::Person, EntityType::Organization);
    }

    #[test]
    fn entity_text_preservation() {
        let entity = Entity::new("Exact Text", EntityType::Person, 0, 10, 0.9);
        assert_eq!(entity.text, "Exact Text");
    }

    #[test]
    fn total_len_calculation() {
        // Contiguous
        let e1 = Entity::new("Test", EntityType::Person, 10, 20, 0.9);
        assert_eq!(e1.total_len(), 10);

        // Zero-length
        let e2 = Entity::new("", EntityType::Person, 5, 5, 0.9);
        assert_eq!(e2.total_len(), 0);
    }
}

// =============================================================================
// Coreference Edge Cases
// =============================================================================

#[test]
fn coreference_same_text_different_types() {
    use anno::backends::inference::{resolve_coreferences, CoreferenceConfig};

    // "Apple" as Organization vs "Apple" as Product - should NOT cluster
    let e1 = Entity::new("Apple", EntityType::Organization, 0, 5, 0.9);
    let e2 = Entity::new("Apple", EntityType::Other("Product".to_string()), 20, 25, 0.9);

    let embeddings = vec![0.5f32; 128];
    let config = CoreferenceConfig::default();

    let clusters = resolve_coreferences(&[e1, e2], &embeddings, 64, &config);

    // Different types should not cluster
    assert!(clusters.is_empty() || clusters.iter().all(|c| c.members.len() == 1));
}

#[test]
fn coreference_pronoun_resolution() {
    use anno::backends::inference::{resolve_coreferences, CoreferenceConfig};

    // "John" and "he" - different strings but same type
    let e1 = Entity::new("John", EntityType::Person, 0, 4, 0.95);
    let e2 = Entity::new("he", EntityType::Person, 20, 22, 0.8);

    // Use identical embeddings to force clustering
    let embeddings = vec![0.9f32; 128];
    let config = CoreferenceConfig {
        similarity_threshold: 0.8,
        max_distance: Some(100),
        use_string_match: false, // Don't rely on string match
    };

    let clusters = resolve_coreferences(&[e1, e2], &embeddings, 64, &config);

    // With identical embeddings and low threshold, should cluster
    assert!(!clusters.is_empty());
}

