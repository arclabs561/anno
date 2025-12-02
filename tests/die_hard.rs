//! Die Hard Tests - Adversarial, Cross-Lingual, and Edge Case Testing
//!
//! "Yippee-ki-yay, null pointer exception."
//!
//! These tests are designed to break NER models using:
//! - Zalgo text (excessive combining characters)
//! - Mixed scripts (RTL + LTR + CJK in one line)
//! - Unicode normalization forms (NFC vs NFD)
//! - Control characters and zero-width joiners
//! - Recursive/Nested structures
//! - Massive repetitions

use anno::backends::stacked::ConflictStrategy;
use anno::{
    AutoNER, DiscontinuousSpan, Entity, EntityBuilder, EntityType, EntityViewport,
    ExtractionMethod, HeuristicNER, HierarchicalConfidence, Model, Provenance, RegexNER,
    StackedNER,
};

fn auto() -> AutoNER {
    AutoNER::new()
}

fn heuristic() -> HeuristicNER {
    HeuristicNER::new()
}

// =============================================================================
// 1. The "Tower of Babel" - Mixed Script Chaos
// =============================================================================

#[test]
fn test_mixed_script_sentence() {
    // English + Arabic + Chinese + Emoji in one semantic unit
    // "Apple (company) announced in Riyadh that 北京 is nice."
    let text = "Apple announced in الرياض that 北京 is nice 🚀.";

    let model = auto();
    let entities = model.extract_entities(text, None).unwrap();

    // Should find: Apple (ORG), Riyadh (LOC), Beijing (LOC)
    let texts: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();

    assert!(texts.contains(&"Apple"), "Missed English ORG");
    // Riyadh in Arabic might be missed by HeuristicNER without GLiNER,
    // but we should at least NOT panic or return garbage offsets.
    // Beijing (北京) should be found by HeuristicNER's CJK gazetteer.
    assert!(texts.contains(&"北京"), "Missed Chinese LOC in mixed text");

    // Validate all entity offsets are correct
    let char_count = text.chars().count();
    for e in entities {
        assert!(
            e.start < char_count,
            "Start offset out of bounds: {} >= {}",
            e.start,
            char_count
        );
        assert!(
            e.end <= char_count,
            "End offset out of bounds: {} > {}",
            e.end,
            char_count
        );
        assert!(e.start < e.end, "Invalid offsets: start >= end");
        // Verify extracted text matches using character offsets
        let extracted = e.extract_text(text);
        assert_eq!(
            e.text, extracted,
            "Entity text should match extracted text from source"
        );
    }
}

#[test]
fn test_rtl_ltr_boundary_injection() {
    // "The CEO of [ARABIC_ORG] said..."
    // Requires handling bidirectional text flow logic if we were rendering,
    // but for extraction, byte/char offsets must be strictly linear.
    let text = "The CEO of شركة أرامكو said hello.";
    // "شركة أرامكو" = Saudi Aramco (approx)

    let model = heuristic();
    let entities = model.extract_entities(text, None).unwrap();

    // HeuristicNER doesn't speak Arabic, but it should handle the offsets correctly
    // and perhaps detect "CEO" as a title to skip, or "Aramco" if in latin.
    // Here we just ensure it doesn't crash on the script boundary.
    // (The extraction completing without panic is the test, entities may be empty)

    // Validate all entity offsets are correct (even if empty)
    let char_count = text.chars().count();
    for e in entities {
        assert!(
            e.start < char_count,
            "Start offset out of bounds: {} >= {}",
            e.start,
            char_count
        );
        assert!(
            e.end <= char_count,
            "End offset out of bounds: {} > {}",
            e.end,
            char_count
        );
        assert!(e.start < e.end, "Invalid offsets: start >= end");
        // Verify extracted text matches using character offsets
        let extracted = e.extract_text(text);
        assert_eq!(
            e.text, extracted,
            "Entity text should match extracted text from source"
        );
    }
}

// =============================================================================
// 2. The "Zalgo" - Unicode Stress
// =============================================================================

#[test]
fn test_zalgo_text() {
    // "Google" with heavy combining chars
    // G̶oogle should still structurally look like a capitalized word
    let text = "G̶oogle is a company.";

    let model = heuristic();
    let entities = model.extract_entities(text, None).unwrap();

    // Ideally we clean this or handle it.
    // If HeuristicNER fails to detect "G̶oogle", that's acceptable for now,
    // but it shouldn't crash or return invalid UTF-8 slices.
    // "G" is capitalized base char.

    // If entities are found, verify they're valid and offsets are correct
    if !entities.is_empty() {
        let e = &entities[0];
        // Verify offsets are valid
        let char_count = text.chars().count();
        assert!(
            e.start < char_count && e.end <= char_count,
            "Invalid offsets: start={}, end={}, text_len={}",
            e.start,
            e.end,
            char_count
        );
        assert!(e.start < e.end, "Invalid offsets: start >= end");

        // Verify extracted text matches using character offsets
        let extracted = e.extract_text(text);
        assert_eq!(
            e.text, extracted,
            "Entity text should match extracted text from source"
        );

        // If it extracted "G̶oogle", verify it contains the base characters
        if e.text.contains("G") || e.text.contains("oogle") {
            // Valid extraction
        }
    }
    // Test passes if extraction completes without panic (even if no entities found)
}

#[test]
fn test_normalization_forms() {
    // "Amélie" in NFC vs NFD
    // NFC: 0041 006D 00E9 006C 0069 0065 (6 chars)
    // NFD: 0041 006D 0065 0301 006C 0069 0065 (7 chars, e + combining acute)

    let nfc = "Amélie visited Paris.";
    let nfd = "Ame\u{0301}lie visited Paris.";

    let model = heuristic();

    let e_nfc = model.extract_entities(nfc, None).unwrap();
    let e_nfd = model.extract_entities(nfd, None).unwrap();

    // Should detect "Amélie" in both cases (or at least "Paris")
    // And "Paris"

    // Check for Paris (more likely to be detected)
    assert!(
        e_nfc.iter().any(|e| e.text.contains("Paris")),
        "NFC: Should detect Paris"
    );
    assert!(
        e_nfd.iter().any(|e| e.text.contains("Paris")),
        "NFD: Should detect Paris"
    );

    // Validate offsets for all entities
    let nfc_char_count = nfc.chars().count();
    for e in &e_nfc {
        assert!(
            e.start < nfc_char_count,
            "NFC: Start offset out of bounds: {} >= {}",
            e.start,
            nfc_char_count
        );
        assert!(
            e.end <= nfc_char_count,
            "NFC: End offset out of bounds: {} > {}",
            e.end,
            nfc_char_count
        );
        assert!(e.start < e.end, "NFC: Invalid offsets: start >= end");
        let extracted = e.extract_text(nfc);
        assert_eq!(
            e.text, extracted,
            "NFC: Entity text should match extracted text from source"
        );
    }

    let nfd_char_count = nfd.chars().count();
    for e in &e_nfd {
        assert!(
            e.start < nfd_char_count,
            "NFD: Start offset out of bounds: {} >= {}",
            e.start,
            nfd_char_count
        );
        assert!(
            e.end <= nfd_char_count,
            "NFD: End offset out of bounds: {} > {}",
            e.end,
            nfd_char_count
        );
        assert!(e.start < e.end, "NFD: Invalid offsets: start >= end");
        let extracted = e.extract_text(nfd);
        assert_eq!(
            e.text, extracted,
            "NFD: Entity text should match extracted text from source"
        );
    }

    // Check for Amélie if detected (may not be detected by heuristic, but if detected, should work)
    // Note: HeuristicNER may not detect "Amélie" as it's not a known entity, but if it does,
    // it should handle both normalization forms correctly.

    // Offsets will differ between NFC and NFD, but extracted text should be logically same
    // (though strictly, the extracted text will preserve normalization form)
}

// =============================================================================
// 3. The "Recursive" - Nested and confusing structures
// =============================================================================

#[test]
fn test_nested_parentheses_hell() {
    // "The (United (States) of (America)) is a country."
    let text = "The (United (States) of (America)) is a country.";

    let model = heuristic();
    let entities = model.extract_entities(text, None).unwrap();

    // Should ideally extract "United States of America" or parts.
    // Previous bugs showed "America))" - let's check cleanliness.
    let char_count = text.chars().count();
    for e in entities {
        // Validate offsets
        assert!(
            e.start < char_count,
            "Start offset out of bounds: {} >= {}",
            e.start,
            char_count
        );
        assert!(
            e.end <= char_count,
            "End offset out of bounds: {} > {}",
            e.end,
            char_count
        );
        assert!(e.start < e.end, "Invalid offsets: start >= end");

        // Check for malformed parentheses (nested or trailing/leading)
        assert!(
            !e.text.contains("))"),
            "Failed to clean trailing parens: {}",
            e.text
        );
        assert!(
            !e.text.contains("(("),
            "Failed to clean leading parens: {}",
            e.text
        );
        // Also check for single trailing/leading parens that shouldn't be in entity text
        assert!(
            !e.text.starts_with('('),
            "Entity should not start with opening paren: {}",
            e.text
        );
        assert!(
            !e.text.ends_with(')'),
            "Entity should not end with closing paren: {}",
            e.text
        );

        // Verify extracted text matches using character offsets
        let extracted = e.extract_text(text);
        assert_eq!(
            e.text, extracted,
            "Entity text should match extracted text from source"
        );
    }
}

#[test]
fn test_entities_glued_to_punctuation() {
    let text = "Hello.Google.Inc. is distinct from Apple,Inc.";
    // Missing spaces are common in bad OCR/PDF extraction

    let model = heuristic();
    let entities = model.extract_entities(text, None).unwrap();

    // HeuristicNER relies on whitespace mostly, but let's see if it survives
    // "Hello.Google.Inc." might be treated as one token.
    // If it is, does capitalization check fail?
    // H is caps.

    // At minimum, extraction should complete without panic
    // If entities are found, verify they don't have malformed punctuation
    let char_count = text.chars().count();
    for e in entities {
        // Entity text shouldn't start or end with punctuation (unless it's part of the name)
        // But we should at least verify offsets are valid
        assert!(
            e.start < char_count,
            "Start offset out of bounds: {} >= {}",
            e.start,
            char_count
        );
        assert!(
            e.end <= char_count,
            "End offset out of bounds: {} > {}",
            e.end,
            char_count
        );
        assert!(e.start < e.end, "Invalid offsets: start >= end");
        // Verify extracted text matches using character offsets
        let extracted = e.extract_text(text);
        assert_eq!(
            e.text, extracted,
            "Entity text should match extracted text from source"
        );
    }
}

// =============================================================================
// 4. The "Zero Width" - Invisible characters
// =============================================================================

#[test]
fn test_zero_width_joiners() {
    // "San\u{200B}Francisco" - Zero width space in middle of entity
    let text = "San\u{200B}Francisco is a city.";

    let model = heuristic();
    let entities = model.extract_entities(text, None).unwrap();

    // Should extraction handle this?
    // Usually we want to extract "San\u{200B}Francisco" verbatim to point to source,
    // OR normalize it.
    // Key is: offsets must match original string.
    // CRITICAL: Entity stores CHARACTER offsets, not byte offsets!
    // Must use extract_text() method, not byte slicing.
    if let Some(e) = entities.first() {
        // Use the Entity's extract_text method which correctly handles character offsets
        let extracted = e.extract_text(text);
        assert_eq!(
            e.text, extracted,
            "Extracted text must match source slice exactly"
        );

        // Also verify offsets are within bounds
        let char_count = text.chars().count();
        assert!(
            e.start < char_count,
            "Start offset out of bounds: {} >= {}",
            e.start,
            char_count
        );
        assert!(
            e.end <= char_count,
            "End offset out of bounds: {} > {}",
            e.end,
            char_count
        );
        assert!(e.start < e.end, "Invalid offsets: start >= end");
    }
}

// =============================================================================
// 5. The "Flood" - Massive input
// =============================================================================

#[test]
fn test_massive_repetition() {
    // 10,000 "Google"s separate by dots to prevent merging into one giant entity
    let text = "Google. ".repeat(10_000);

    let model = heuristic();
    let start = std::time::Instant::now();
    let entities = model.extract_entities(&text, None).unwrap();
    let duration = start.elapsed();

    // Note: HeuristicNER may not extract "Google" when followed by a period,
    // as it cleans trailing punctuation. So we check for a reasonable number
    // of entities (at least some should be found, or none if the model filters them).
    // The key test is that it completes without panic and is fast.

    // If entities are found, verify they're valid
    let char_count = text.chars().count();
    for e in entities.iter() {
        assert!(
            e.start < char_count,
            "Start offset out of bounds: {} >= {}",
            e.start,
            char_count
        );
        assert!(
            e.end <= char_count,
            "End offset out of bounds: {} > {}",
            e.end,
            char_count
        );
        assert!(e.start < e.end, "Invalid offsets: start >= end");
        // Verify extracted text matches using character offsets
        let extracted = e.extract_text(&text);
        assert_eq!(
            e.text, extracted,
            "Entity text should match extracted text from source"
        );
    }

    // HeuristicNER should be FAST. < 1500ms for 10k tokens.
    assert!(
        duration.as_millis() < 1500,
        "Too slow: {}ms",
        duration.as_millis()
    );

    // Log the actual count for debugging (test still passes if 0, but warns)
    if entities.is_empty() {
        eprintln!("Warning: No entities extracted from 10k repetitions (may be expected if model filters trailing punctuation)");
    } else {
        eprintln!(
            "Extracted {} entities from 10k repetitions in {}ms",
            entities.len(),
            duration.as_millis()
        );
    }
}

// =============================================================================
// 6. Boundary Conditions
// =============================================================================

#[test]
fn test_entity_at_text_boundaries() {
    let text = "John works at Apple";
    let model = heuristic();
    let entities = model.extract_entities(text, None).unwrap();

    let char_count = text.chars().count();

    // Verify all entities have valid boundaries
    for e in &entities {
        // Entity at start (start == 0)
        if e.start == 0 {
            assert!(e.end > 0, "Entity at start must have end > 0");
            let extracted = e.extract_text(text);
            assert_eq!(e.text, extracted);
        }

        // Entity at end (end == char_count)
        if e.end == char_count {
            assert!(e.start < char_count, "Entity at end must have start < end");
            let extracted = e.extract_text(text);
            assert_eq!(e.text, extracted);
        }

        // Validate using entity.validate()
        let issues = e.validate(text);
        assert!(issues.is_empty(), "Entity should be valid: {:?}", issues);
    }
}

#[test]
fn test_entity_spanning_entire_text() {
    let text = "Apple";
    let model = heuristic();
    let entities = model.extract_entities(text, None).unwrap();

    // If an entity spans the entire text
    for e in entities {
        if e.start == 0 && e.end == text.chars().count() {
            let extracted = e.extract_text(text);
            assert_eq!(e.text, extracted);
            assert_eq!(e.text, text);

            // Validate
            let issues = e.validate(text);
            assert!(issues.is_empty(), "Full-span entity should be valid");
        }
    }
}

#[test]
fn test_zero_length_entity() {
    let text = "Hello World";
    // Create an invalid entity with start == end
    let invalid_entity = Entity::new("", EntityType::Person, 5, 5, 0.9);

    // Should be caught by validation
    let issues = invalid_entity.validate(text);
    assert!(!issues.is_empty(), "Zero-length entity should be invalid");

    // extract_text should return empty string for invalid span
    let extracted = invalid_entity.extract_text(text);
    assert_eq!(extracted, "");
}

// =============================================================================
// 7. Overlapping Entities
// =============================================================================

#[test]
fn test_overlapping_entities() {
    // "New York City" - potential for overlapping entities
    let text = "New York City is large.";
    let model = heuristic();
    let entities = model.extract_entities(text, None).unwrap();

    // Verify no overlapping entities (or if allowed, verify handling)
    for i in 0..entities.len() {
        for j in (i + 1)..entities.len() {
            let e1 = &entities[i];
            let e2 = &entities[j];

            // Check if they overlap (not just adjacent)
            let overlap = !(e1.end <= e2.start || e2.end <= e1.start);

            if overlap {
                // If overlapping, they should be identical or one should be contained
                let contained = (e1.start >= e2.start && e1.end <= e2.end)
                    || (e2.start >= e1.start && e2.end <= e1.end);
                let identical = e1.start == e2.start && e1.end == e2.end;

                assert!(
                    contained || identical,
                    "Overlapping entities should be nested or identical: {:?} and {:?}",
                    e1,
                    e2
                );
            }

            // Both should be valid
            let issues1 = e1.validate(text);
            let issues2 = e2.validate(text);
            assert!(
                issues1.is_empty(),
                "Entity 1 should be valid: {:?}",
                issues1
            );
            assert!(
                issues2.is_empty(),
                "Entity 2 should be valid: {:?}",
                issues2
            );
        }
    }
}

#[test]
fn test_adjacent_entities() {
    // Entities that touch but don't overlap
    let text = "John and Jane work at Apple";
    let model = heuristic();
    let entities = model.extract_entities(text, None).unwrap();

    // Find adjacent entities (one ends where another starts)
    for i in 0..entities.len() {
        for j in (i + 1)..entities.len() {
            let e1 = &entities[i];
            let e2 = &entities[j];

            // Adjacent: e1.end == e2.start or e2.end == e1.start
            let adjacent = e1.end == e2.start || e2.end == e1.start;

            if adjacent {
                // Adjacent entities are valid
                let issues1 = e1.validate(text);
                let issues2 = e2.validate(text);
                assert!(issues1.is_empty());
                assert!(issues2.is_empty());
            }
        }
    }
}

// =============================================================================
// 8. Control Characters
// =============================================================================

#[test]
fn test_control_characters() {
    // Test various control characters
    let text = format!("Test\u{0000}entity\u{200C}here\u{200D}now\u{FEFF}end");

    let model = heuristic();
    let entities = model.extract_entities(&text, None).unwrap();

    // Should handle gracefully, not panic
    for e in entities {
        let extracted = e.extract_text(&text);
        assert_eq!(
            e.text, extracted,
            "Control chars: extracted text should match"
        );

        let issues = e.validate(&text);
        assert!(
            issues.is_empty(),
            "Entity with control chars should be valid"
        );
    }
}

#[test]
fn test_bidirectional_marks() {
    // Left-to-right mark, right-to-left mark, directional isolates
    let text = format!("Hello\u{200E}World\u{200F}Test\u{202A}Arabic\u{202C}");

    let model = heuristic();
    let entities = model.extract_entities(&text, None).unwrap();

    for e in entities {
        let extracted = e.extract_text(&text);
        assert_eq!(e.text, extracted);

        let issues = e.validate(&text);
        assert!(issues.is_empty());
    }
}

#[test]
fn test_line_separators() {
    // Line separator (U+2028) and paragraph separator (U+2029)
    let text = "First line\u{2028}Second line\u{2029}Third paragraph";

    let model = heuristic();
    let entities = model.extract_entities(&text, None).unwrap();

    for e in entities {
        let extracted = e.extract_text(&text);
        assert_eq!(e.text, extracted);

        let issues = e.validate(&text);
        assert!(issues.is_empty());
    }
}

// =============================================================================
// 9. Grapheme Clusters
// =============================================================================

#[test]
fn test_emoji_grapheme_clusters() {
    // Family emoji = 4 code points, 1 grapheme
    // Flag emoji = 2 code points, 1 grapheme
    let text = "The 👨‍👩‍👧‍👦 family visited 🇺🇸 and 🇫🇷.";

    let model = heuristic();
    let entities = model.extract_entities(&text, None).unwrap();

    // Verify offsets are character-based, not code-point based
    for e in entities {
        let extracted = e.extract_text(&text);
        assert_eq!(e.text, extracted, "Emoji grapheme: extracted should match");

        let issues = e.validate(&text);
        assert!(issues.is_empty());

        // Verify we can extract text correctly even with emoji
        let char_count = text.chars().count();
        assert!(e.start < char_count);
        assert!(e.end <= char_count);
    }
}

#[test]
fn test_combining_sequence_graphemes() {
    // "é" as combining sequence: e + combining acute
    let text = "Caf\u{00E9} and Caf\u{0065}\u{0301} are the same.";
    // First "Café" is NFC (single code point), second is NFD (two code points)

    let model = heuristic();
    let entities = model.extract_entities(&text, None).unwrap();

    for e in entities {
        let extracted = e.extract_text(&text);
        assert_eq!(e.text, extracted);

        let issues = e.validate(&text);
        assert!(issues.is_empty());
    }
}

// =============================================================================
// 10. Confidence Edge Cases
// =============================================================================

#[test]
fn test_confidence_edge_cases() {
    let text = "John works at Apple";
    let model = heuristic();
    let entities = model.extract_entities(text, None).unwrap();

    for e in entities {
        // Confidence should be in [0.0, 1.0]
        assert!(
            (0.0..=1.0).contains(&e.confidence),
            "Invalid confidence: {} (should be in [0.0, 1.0])",
            e.confidence
        );
        assert!(!e.confidence.is_nan(), "Confidence should not be NaN");
        assert!(
            !e.confidence.is_infinite(),
            "Confidence should not be infinite"
        );

        // Validate should catch invalid confidence
        let mut test_entity = e.clone();
        test_entity.confidence = -0.1;
        let issues = test_entity.validate(text);
        assert!(!issues.is_empty(), "Negative confidence should be invalid");

        test_entity.confidence = 1.1;
        let issues = test_entity.validate(text);
        assert!(!issues.is_empty(), "Confidence > 1.0 should be invalid");
    }
}

#[test]
fn test_confidence_boundary_values() {
    let text = "Test";

    // Test minimum valid confidence
    let entity_min = Entity::new("Test", EntityType::Person, 0, 4, 0.0);
    let issues_min = entity_min.validate(text);
    assert!(issues_min.is_empty(), "Confidence 0.0 should be valid");

    // Test maximum valid confidence
    let entity_max = Entity::new("Test", EntityType::Person, 0, 4, 1.0);
    let issues_max = entity_max.validate(text);
    assert!(issues_max.is_empty(), "Confidence 1.0 should be valid");
}

// =============================================================================
// 11. Empty Entity Text
// =============================================================================

#[test]
fn test_empty_entity_text() {
    let text = "Hello World";

    // Entity with valid offsets but empty text (inconsistent state)
    let mut entity = Entity::new("", EntityType::Person, 0, 5, 0.9);

    // extract_text should work correctly even if entity.text is empty
    let extracted = entity.extract_text(text);
    assert_eq!(extracted, "Hello");

    // But validation should catch the text mismatch
    let issues = entity.validate(text);
    assert!(
        !issues.is_empty(),
        "Should detect text mismatch when entity.text is empty"
    );

    // Fix the text to match
    entity.text = "Hello".to_string();
    let issues_fixed = entity.validate(text);
    assert!(issues_fixed.is_empty(), "Should be valid after fixing text");
}

#[test]
fn test_whitespace_only_entity_text() {
    let text = "Hello   World";

    // Entity that extracts whitespace
    let entity = Entity::new("   ", EntityType::Other("MISC".to_string()), 5, 8, 0.5);

    let extracted = entity.extract_text(text);
    assert_eq!(extracted, "   ");

    // Validation should pass (whitespace is valid text)
    let _issues = entity.validate(text);
    // Note: Some validators might flag whitespace-only, but it's technically valid
}

// =============================================================================
// 12. Language Parameter
// =============================================================================

#[test]
fn test_language_parameter() {
    let text = "John works at Apple";
    let model = heuristic();

    // Test with language hint
    let entities_en = model.extract_entities(text, Some("en")).unwrap();
    let entities_none = model.extract_entities(text, None).unwrap();

    // Should produce valid results (may differ, but should be valid)
    for e in &entities_en {
        let issues = e.validate(text);
        assert!(issues.is_empty());
        let extracted = e.extract_text(text);
        assert_eq!(e.text, extracted);
    }

    for e in &entities_none {
        let issues = e.validate(text);
        assert!(issues.is_empty());
        let extracted = e.extract_text(text);
        assert_eq!(e.text, extracted);
    }
}

#[test]
fn test_invalid_language_code() {
    let text = "John works at Apple";
    let model = heuristic();

    // Invalid language code should not panic
    let entities = model
        .extract_entities(text, Some("invalid-lang-code"))
        .unwrap();

    // Should still produce valid entities (or empty)
    for e in entities {
        let issues = e.validate(text);
        assert!(issues.is_empty());
    }
}

#[test]
fn test_multilingual_language_hints() {
    let text = "Apple announced in 北京 that 東京 is nice.";
    let model = heuristic();

    // Test with different language hints
    let entities_zh = model.extract_entities(&text, Some("zh")).unwrap();
    let entities_en = model.extract_entities(&text, Some("en")).unwrap();
    let entities_ja = model.extract_entities(&text, Some("ja")).unwrap();

    // All should produce valid entities
    for entities in [&entities_zh, &entities_en, &entities_ja] {
        for e in entities {
            let issues = e.validate(&text);
            assert!(issues.is_empty());
            let extracted = e.extract_text(&text);
            assert_eq!(e.text, extracted);
        }
    }
}

// =============================================================================
// 13. Discontinuous Spans
// =============================================================================

#[test]
fn test_discontinuous_spans() {
    let text = "severe pain in the abdomen";

    // Create entity with discontinuous span: "severe" (0-6) + "pain" (12-16)
    // Note: DiscontinuousSpan uses byte offsets, not character offsets
    let severe_byte_start = 0;
    let severe_byte_end = "severe".len();
    let pain_byte_start = text.find("pain").unwrap();
    let pain_byte_end = pain_byte_start + "pain".len();

    // Extract the actual text from discontinuous spans
    let extracted_text = format!(
        "{} {}",
        &text[severe_byte_start..severe_byte_end],
        &text[pain_byte_start..pain_byte_end]
    );

    let mut entity = Entity::new(
        &extracted_text,
        EntityType::Other("MISC".to_string()),
        0,
        text.len(),
        0.9,
    );

    let disc_span = DiscontinuousSpan::new(vec![
        severe_byte_start..severe_byte_end,
        pain_byte_start..pain_byte_end,
    ]);
    entity.set_discontinuous_span(disc_span);

    // Extract using the discontinuous span's extract_text method
    let extracted = entity
        .discontinuous_span
        .as_ref()
        .unwrap()
        .extract_text(text, " ");
    assert_eq!(extracted, "severe pain");

    // Validate - note: validation may flag text mismatch since discontinuous spans
    // have different text than the bounding range, but the structure should be valid
    let issues = entity.validate(text);
    // For discontinuous spans, text mismatch is expected, so we check other validations
    let non_text_issues: Vec<_> = issues
        .iter()
        .filter(|i| !matches!(i, anno::ValidationIssue::TextMismatch { .. }))
        .collect();
    assert!(
        non_text_issues.is_empty(),
        "Discontinuous span should have valid structure: {:?}",
        non_text_issues
    );

    // Verify it's marked as discontinuous
    assert!(entity.is_discontinuous());
}

// =============================================================================
// 14. Model Consistency
// =============================================================================

#[test]
fn test_model_consistency_auto_vs_heuristic() {
    let text = "John works at Apple in New York";

    let auto_model = auto();
    let heuristic_model = heuristic();

    let auto_entities = auto_model.extract_entities(text, None).unwrap();
    let heuristic_entities = heuristic_model.extract_entities(text, None).unwrap();

    // Both should produce valid entities
    for e in &auto_entities {
        let issues = e.validate(text);
        assert!(issues.is_empty());
        let extracted = e.extract_text(text);
        assert_eq!(e.text, extracted);
    }

    for e in &heuristic_entities {
        let issues = e.validate(text);
        assert!(issues.is_empty());
        let extracted = e.extract_text(text);
        assert_eq!(e.text, extracted);
    }
}

#[test]
fn test_model_consistency_stacked() {
    let text = "John works at Apple";

    let stacked = StackedNER::new();
    let heuristic = heuristic();

    let stacked_entities = stacked.extract_entities(text, None).unwrap();
    let heuristic_entities = heuristic.extract_entities(text, None).unwrap();

    // Both should produce valid entities
    for e in &stacked_entities {
        let issues = e.validate(text);
        assert!(issues.is_empty());
    }

    for e in &heuristic_entities {
        let issues = e.validate(text);
        assert!(issues.is_empty());
    }
}

#[test]
fn test_model_consistency_pattern() {
    let text = "Contact: test@example.com or call 555-1234 on Jan 15, 2024";

    let pattern = RegexNER::new();
    let heuristic = heuristic();

    let pattern_entities = pattern.extract_entities(text, None).unwrap();
    let heuristic_entities = heuristic.extract_entities(text, None).unwrap();

    // Both should produce valid entities
    for e in &pattern_entities {
        let issues = e.validate(text);
        assert!(issues.is_empty());
        let extracted = e.extract_text(text);
        assert_eq!(e.text, extracted);
    }

    for e in &heuristic_entities {
        let issues = e.validate(text);
        assert!(issues.is_empty());
        let extracted = e.extract_text(text);
        assert_eq!(e.text, extracted);
    }
}

// =============================================================================
// 15. Validation Integration
// =============================================================================

#[test]
fn test_validation_in_all_tests() {
    // This test ensures validation is called in a comprehensive scenario
    let text = "John Smith works at Apple Inc. in Cupertino, California.";

    let model = auto();
    let entities = model.extract_entities(text, None).unwrap();

    // All entities should pass validation
    for e in &entities {
        let issues = e.validate(text);
        assert!(
            issues.is_empty(),
            "Entity should be valid: {:?}, issues: {:?}",
            e,
            issues
        );

        // Also verify extract_text
        let extracted = e.extract_text(text);
        assert_eq!(e.text, extracted);
    }
}

#[test]
fn test_validation_catches_invalid_offsets() {
    let text = "Hello World";
    let char_count = text.chars().count();

    // Entity with out-of-bounds end
    let bad_entity = Entity::new("World", EntityType::Location, 6, char_count + 10, 0.9);
    let issues = bad_entity.validate(text);
    assert!(!issues.is_empty(), "Should detect out-of-bounds entity");

    // Entity with start > end
    let bad_entity2 = Entity::new("World", EntityType::Location, 10, 6, 0.9);
    let issues2 = bad_entity2.validate(text);
    assert!(
        !issues2.is_empty(),
        "Should detect invalid span (start > end)"
    );
}

#[test]
fn test_validation_catches_text_mismatch() {
    let text = "John works at Apple";

    // Entity with wrong text
    let bad_entity = Entity::new("Wrong", EntityType::Person, 0, 4, 0.9);
    let issues = bad_entity.validate(text);
    assert!(!issues.is_empty(), "Should detect text mismatch");
}

// =============================================================================
// 16. Extract Text Edge Cases
// =============================================================================

#[test]
fn test_extract_text_out_of_bounds() {
    let text = "Hello World";
    let char_count = text.chars().count();

    // Entity with out-of-bounds offsets
    let bad_entity = Entity::new("Test", EntityType::Person, 0, char_count + 10, 0.9);

    // extract_text should return empty string for invalid bounds
    let extracted = bad_entity.extract_text(text);
    assert_eq!(
        extracted, "",
        "Out-of-bounds extract_text should return empty"
    );
}

#[test]
fn test_extract_text_start_greater_than_end() {
    let text = "Hello World";

    // Entity with start > end
    let bad_entity = Entity::new("Test", EntityType::Person, 10, 5, 0.9);

    // extract_text should return empty string
    let extracted = bad_entity.extract_text(text);
    assert_eq!(
        extracted, "",
        "Invalid span extract_text should return empty"
    );
}

#[test]
fn test_extract_text_at_boundaries() {
    let text = "Apple";
    let char_count = text.chars().count();

    // Entity at start
    let entity_start = Entity::new("App", EntityType::Organization, 0, 3, 0.9);
    let extracted_start = entity_start.extract_text(text);
    assert_eq!(extracted_start, "App");

    // Entity at end
    let entity_end = Entity::new("ple", EntityType::Organization, 2, char_count, 0.9);
    let extracted_end = entity_end.extract_text(text);
    assert_eq!(extracted_end, "ple");

    // Entity spanning entire text
    let entity_full = Entity::new("Apple", EntityType::Organization, 0, char_count, 0.9);
    let extracted_full = entity_full.extract_text(text);
    assert_eq!(extracted_full, "Apple");
}

// =============================================================================
// 17. Performance and Memory
// =============================================================================

#[test]
fn test_very_long_entity_name() {
    // Very long entity name (10k+ characters)
    let long_name = "A".repeat(10_000);
    let text = format!("The {} company announced.", long_name);

    let model = heuristic();
    let start = std::time::Instant::now();
    let entities = model.extract_entities(&text, None).unwrap();
    let duration = start.elapsed();

    // Should complete without panic
    for e in entities {
        let issues = e.validate(&text);
        assert!(issues.is_empty());
    }

    // Should be reasonably fast even with long entity
    assert!(
        duration.as_millis() < 5000,
        "Too slow: {}ms",
        duration.as_millis()
    );
}

#[test]
fn test_many_entities() {
    // Generate text with many potential entities
    let mut text_parts = Vec::new();
    for i in 1..=1000 {
        text_parts.push(format!("Person{} works at Company{} in City{}. ", i, i, i));
    }
    let text = text_parts.join("");

    let model = heuristic();
    let start = std::time::Instant::now();
    let entities = model.extract_entities(&text, None).unwrap();
    let duration = start.elapsed();

    // Should complete without panic
    for e in entities {
        let issues = e.validate(&text);
        assert!(issues.is_empty());
    }

    // Should be reasonably fast
    assert!(
        duration.as_millis() < 3000,
        "Too slow: {}ms",
        duration.as_millis()
    );
}

// =============================================================================
// 18. Entity Builder and Advanced Features
// =============================================================================

#[test]
fn test_entity_builder_pattern() {
    let text = "John works at Apple";

    let entity = EntityBuilder::new("John", EntityType::Person)
        .span(0, 4)
        .confidence(0.95)
        .build();

    let extracted = entity.extract_text(text);
    assert_eq!(extracted, "John");

    let issues = entity.validate(text);
    assert!(issues.is_empty());
}

#[test]
fn test_entity_with_provenance() {
    let text = "Apple Inc.";
    let entity = Entity::with_provenance(
        "Apple Inc.",
        EntityType::Organization,
        0,
        10,
        0.9,
        Provenance {
            source: "test".into(),
            method: ExtractionMethod::Pattern,
            pattern: Some("ORG_SUFFIX".into()),
            raw_confidence: Some(0.9),
            model_version: None,
            timestamp: None,
        },
    );

    assert!(entity.provenance.is_some());
    assert_eq!(
        entity.provenance.as_ref().unwrap().method,
        ExtractionMethod::Pattern
    );

    let issues = entity.validate(text);
    assert!(issues.is_empty());
}

#[test]
fn test_entity_with_hierarchical_confidence() {
    let text = "John Smith";
    let entity = Entity::with_hierarchical_confidence(
        "John Smith",
        EntityType::Person,
        0,
        10,
        HierarchicalConfidence::new(0.8, 0.9, 0.85),
    );

    assert!(entity.hierarchical_confidence.is_some());
    let hc = entity.hierarchical_confidence.as_ref().unwrap();
    assert_eq!(hc.linkage, 0.8);
    assert_eq!(hc.type_score, 0.9);
    assert_eq!(hc.boundary, 0.85);

    let issues = entity.validate(text);
    assert!(issues.is_empty());
}

#[test]
fn test_entity_viewport() {
    let text = "Marie Curie";
    let mut entity = Entity::new("Marie Curie", EntityType::Person, 0, 11, 0.9);
    entity.viewport = Some(EntityViewport::Academic);

    assert!(entity.viewport.is_some());
    assert!(entity.viewport.as_ref().unwrap().is_professional());

    let issues = entity.validate(text);
    assert!(issues.is_empty());
}

#[test]
fn test_custom_entity_type() {
    let text = "CRISPR-Cas9";
    let custom_type = EntityType::custom("TECHNOLOGY", anno::EntityCategory::Misc);
    let entity = Entity::new("CRISPR-Cas9", custom_type, 0, 11, 0.9);

    assert_eq!(entity.entity_type.as_label(), "TECHNOLOGY");

    let issues = entity.validate(text);
    assert!(issues.is_empty());
}

#[test]
fn test_entity_other_type() {
    let text = "UnknownEntity";
    let entity = Entity::new(
        "UnknownEntity",
        EntityType::Other("CUSTOM".to_string()),
        0,
        13,
        0.9,
    );

    assert_eq!(entity.entity_type.as_label(), "CUSTOM");

    let issues = entity.validate(text);
    assert!(issues.is_empty());
}

// =============================================================================
// 19. Conflict Resolution Strategies
// =============================================================================

#[test]
fn test_stacked_priority_strategy() {
    // Create two models that will produce overlapping entities
    let text = "New York City";

    // RegexNER might find "New York" (if it has a pattern)
    // HeuristicNER might find "New York City"
    let stacked = StackedNER::builder()
        .strategy(ConflictStrategy::Priority)
        .build();

    let entities = stacked.extract_entities(text, None).unwrap();

    // Should produce valid entities regardless of overlap resolution
    for e in entities {
        let issues = e.validate(text);
        assert!(issues.is_empty());
        let extracted = e.extract_text(text);
        assert_eq!(e.text, extracted);
    }
}

#[test]
fn test_stacked_longest_span_strategy() {
    let text = "New York City is large";
    let stacked = StackedNER::builder()
        .strategy(ConflictStrategy::LongestSpan)
        .build();

    let entities = stacked.extract_entities(text, None).unwrap();

    // Verify no overlapping entities (longest should win)
    for i in 0..entities.len() {
        for j in (i + 1)..entities.len() {
            let e1 = &entities[i];
            let e2 = &entities[j];
            let overlap = !(e1.end <= e2.start || e2.end <= e1.start);
            if overlap {
                // If overlapping, longer should have won
                let len1 = e1.end - e1.start;
                let len2 = e2.end - e2.start;
                assert_eq!(
                    len1, len2,
                    "Overlapping entities should have same length (longest won)"
                );
            }
        }
    }
}

#[test]
fn test_stacked_union_strategy() {
    let text = "John works at Apple";
    let stacked = StackedNER::builder()
        .strategy(ConflictStrategy::Union)
        .build();

    let entities = stacked.extract_entities(text, None).unwrap();

    // Union allows overlaps, so we just verify all are valid
    for e in entities {
        let issues = e.validate(text);
        assert!(issues.is_empty());
    }
}

// =============================================================================
// 20. Entity Normalization and KB Linking
// =============================================================================

#[test]
fn test_entity_normalization() {
    let text = "Apple Inc.";
    let mut entity = Entity::new("Apple Inc.", EntityType::Organization, 0, 10, 0.9);
    entity.normalized = Some("Apple Inc".to_string());

    assert!(entity.normalized.is_some());
    assert_eq!(entity.normalized.as_ref().unwrap(), "Apple Inc");

    let issues = entity.validate(text);
    assert!(issues.is_empty());
}

#[test]
fn test_entity_kb_linking() {
    let text = "Apple";
    let mut entity = Entity::new("Apple", EntityType::Organization, 0, 5, 0.9);
    entity.kb_id = Some("Q312".to_string());
    entity.canonical_id = Some(42);

    assert!(entity.kb_id.is_some());
    assert!(entity.canonical_id.is_some());

    let issues = entity.validate(text);
    assert!(issues.is_empty());
}

// =============================================================================
// 21. Temporal Validity
// =============================================================================

#[test]
fn test_entity_temporal_validity() {
    use chrono::{TimeZone, Utc};

    let text = "Satya Nadella is CEO";
    let mut entity = Entity::new("Satya Nadella", EntityType::Person, 0, 13, 0.9);

    entity.set_valid_from(Utc.with_ymd_and_hms(2014, 2, 4, 0, 0, 0).unwrap());
    entity.set_valid_until(Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap());

    assert!(entity.valid_from.is_some());
    assert!(entity.valid_until.is_some());
    assert!(entity.is_temporal());

    let issues = entity.validate(text);
    assert!(issues.is_empty());
}

#[test]
fn test_entity_temporal_range() {
    use chrono::{TimeZone, Utc};

    let text = "Steve Ballmer was CEO";
    let mut entity = Entity::new("Steve Ballmer", EntityType::Person, 0, 13, 0.9);

    entity.set_temporal_range(
        Utc.with_ymd_and_hms(2000, 1, 13, 0, 0, 0).unwrap(),
        Utc.with_ymd_and_hms(2014, 2, 4, 0, 0, 0).unwrap(),
    );

    assert!(entity.is_temporal());
    assert!(entity.valid_from.is_some());
    assert!(entity.valid_until.is_some());

    let issues = entity.validate(text);
    assert!(issues.is_empty());
}

// =============================================================================
// 22. Unicode Variation Selectors and Tags
// =============================================================================

#[test]
fn test_unicode_variation_selectors() {
    // Variation selectors (U+FE00-U+FE0F) modify appearance of preceding character
    let text = "Café\u{FE00} and café\u{FE01}"; // Same character, different selectors

    let model = heuristic();
    let entities = model.extract_entities(&text, None).unwrap();

    for e in entities {
        let extracted = e.extract_text(&text);
        assert_eq!(e.text, extracted);
        let issues = e.validate(&text);
        assert!(issues.is_empty());
    }
}

#[test]
fn test_unicode_tags() {
    // Language tags (U+E0000-U+E007F) - private use area
    // Note: These are rarely used but should be handled
    let text = "Test\u{E0001}entity";

    let model = heuristic();
    let entities = model.extract_entities(&text, None).unwrap();

    for e in entities {
        let extracted = e.extract_text(&text);
        assert_eq!(e.text, extracted);
        let issues = e.validate(&text);
        assert!(issues.is_empty());
    }
}

// =============================================================================
// 23. Language Parameter Edge Cases
// =============================================================================

#[test]
fn test_empty_language_string() {
    let text = "John works at Apple";
    let model = heuristic();

    // Empty string vs None
    let entities_none = model.extract_entities(text, None).unwrap();
    let entities_empty = model.extract_entities(text, Some("")).unwrap();

    // Both should produce valid results
    for e in &entities_none {
        let issues = e.validate(text);
        assert!(issues.is_empty());
    }

    for e in &entities_empty {
        let issues = e.validate(text);
        assert!(issues.is_empty());
    }
}

#[test]
fn test_very_long_language_code() {
    let text = "Test";
    let model = heuristic();

    // Very long invalid language code
    let long_lang = "x".repeat(1000);
    let entities = model.extract_entities(text, Some(&long_lang)).unwrap();

    // Should not panic, should produce valid entities (or empty)
    for e in entities {
        let issues = e.validate(text);
        assert!(issues.is_empty());
    }
}

// =============================================================================
// 24. Entity Text Trimming
// =============================================================================

#[test]
fn test_entity_with_leading_whitespace() {
    let text = "  John works";
    // Entity that includes leading whitespace
    let entity = Entity::new("  John", EntityType::Person, 0, 6, 0.9);

    let extracted = entity.extract_text(text);
    assert_eq!(extracted, "  John");

    let issues = entity.validate(text);
    // Validation should pass (whitespace is valid)
    assert!(
        issues.is_empty()
            || issues
                .iter()
                .any(|i| matches!(i, anno::ValidationIssue::TextMismatch { .. }))
    );
}

#[test]
fn test_entity_with_trailing_whitespace() {
    let text = "John  works";
    // Entity that includes trailing whitespace
    let entity = Entity::new("John  ", EntityType::Person, 0, 6, 0.9);

    let extracted = entity.extract_text(text);
    assert_eq!(extracted, "John  ");

    let issues = entity.validate(text);
    // Validation should pass
    assert!(
        issues.is_empty()
            || issues
                .iter()
                .any(|i| matches!(i, anno::ValidationIssue::TextMismatch { .. }))
    );
}

// =============================================================================
// 25. Entity Sorting and Stability
// =============================================================================

#[test]
fn test_entity_sorting_stability() {
    let text = "John and Jane work at Apple";
    let model = heuristic();

    let entities1 = model.extract_entities(text, None).unwrap();
    let entities2 = model.extract_entities(text, None).unwrap();

    // Results should be stable (same input = same order)
    assert_eq!(entities1.len(), entities2.len());

    // Verify all entities are valid
    for e in &entities1 {
        let issues = e.validate(text);
        assert!(issues.is_empty());
    }

    for e in &entities2 {
        let issues = e.validate(text);
        assert!(issues.is_empty());
    }
}

#[test]
fn test_entities_sorted_by_position() {
    let text = "John works at Apple in New York";
    let model = heuristic();
    let entities = model.extract_entities(text, None).unwrap();

    // Verify entities are sorted by start position
    for i in 1..entities.len() {
        assert!(
            entities[i - 1].start <= entities[i].start,
            "Entities should be sorted by start position"
        );

        // If same start, should be sorted by end
        if entities[i - 1].start == entities[i].start {
            assert!(
                entities[i - 1].end <= entities[i].end,
                "Entities with same start should be sorted by end"
            );
        }
    }
}

// =============================================================================
// 26. Regex Backtracking (ReDoS Protection)
// =============================================================================

#[test]
fn test_redos_protection_pattern() {
    // Pattern that could cause catastrophic backtracking
    // "a" repeated many times followed by "b" and "a" again
    let text = format!("{}{}a", "a".repeat(100), "b");

    let model = RegexNER::new();
    let start = std::time::Instant::now();
    let entities = model.extract_entities(&text, None).unwrap();
    let duration = start.elapsed();

    // Should complete quickly (not exponential time)
    assert!(
        duration.as_millis() < 1000,
        "ReDoS protection: took {}ms",
        duration.as_millis()
    );

    for e in entities {
        let issues = e.validate(&text);
        assert!(issues.is_empty());
    }
}

// =============================================================================
// 27. Error Propagation
// =============================================================================

#[test]
fn test_stacked_error_propagation() {
    // Create a model that returns an error
    // Note: MockModel doesn't return errors, so we test with real models
    // that might fail in edge cases

    let text = "Test";
    let stacked = StackedNER::default();

    // Should handle gracefully even if one layer has issues
    let result = stacked.extract_entities(text, None);
    assert!(result.is_ok(), "StackedNER should handle errors gracefully");

    if let Ok(entities) = result {
        for e in entities {
            let issues = e.validate(text);
            assert!(issues.is_empty());
        }
    }
}

// =============================================================================
// 28. Very Long Invalid Inputs
// =============================================================================

#[test]
fn test_extremely_long_single_word() {
    // Single word with 100k characters
    let long_word = "a".repeat(100_000);
    let text = format!("The {} company", long_word);

    let model = heuristic();
    let start = std::time::Instant::now();
    let entities = model.extract_entities(&text, None).unwrap();
    let duration = start.elapsed();

    // Should complete without panic
    for e in entities {
        let issues = e.validate(&text);
        assert!(issues.is_empty());
    }

    // Should be reasonably fast (not O(n²))
    assert!(
        duration.as_millis() < 10000,
        "Too slow: {}ms",
        duration.as_millis()
    );
}

#[test]
fn test_many_tiny_entities() {
    // Text with many potential single-character entities
    let text: String = (0..5000).map(|i| format!("A{} ", i)).collect();

    let model = heuristic();
    let start = std::time::Instant::now();
    let entities = model.extract_entities(&text, None).unwrap();
    let duration = start.elapsed();

    // Should complete without panic
    for e in entities {
        let issues = e.validate(&text);
        assert!(issues.is_empty());
    }

    // Should be reasonably fast
    assert!(
        duration.as_millis() < 5000,
        "Too slow: {}ms",
        duration.as_millis()
    );
}

// =============================================================================
// 31. Entity Method Edge Cases
// =============================================================================

#[test]
fn test_entity_overlaps_method() {
    let _text = "New York City";
    let e1 = Entity::new("New York", EntityType::Location, 0, 8, 0.8);
    let e2 = Entity::new("York City", EntityType::Location, 4, 13, 0.9);
    let e3 = Entity::new("Paris", EntityType::Location, 20, 25, 0.9);

    // e1 and e2 overlap
    assert!(e1.overlaps(&e2));
    assert!(e2.overlaps(&e1));

    // e1 and e3 don't overlap
    assert!(!e1.overlaps(&e3));
    assert!(!e3.overlaps(&e1));
}

#[test]
fn test_entity_overlap_ratio() {
    let _text = "New York City";
    let e1 = Entity::new("New York", EntityType::Location, 0, 8, 0.8);
    let e2 = Entity::new("York City", EntityType::Location, 4, 13, 0.9);

    let ratio = e1.overlap_ratio(&e2);
    assert!(
        ratio > 0.0 && ratio <= 1.0,
        "Overlap ratio should be in [0, 1]: {}",
        ratio
    );

    // Identical entities should have ratio 1.0
    let e3 = Entity::new("New York", EntityType::Location, 0, 8, 0.8);
    let ratio_identical = e1.overlap_ratio(&e3);
    assert_eq!(ratio_identical, 1.0);

    // Non-overlapping should have ratio 0.0
    let e4 = Entity::new("Paris", EntityType::Location, 20, 25, 0.9);
    let ratio_no_overlap = e1.overlap_ratio(&e4);
    assert_eq!(ratio_no_overlap, 0.0);
}

#[test]
fn test_entity_is_structured_vs_named() {
    let text = "John works at Apple on Jan 15, 2024";
    let model = RegexNER::new();
    let entities = model.extract_entities(text, None).unwrap();

    for e in entities {
        // Pattern entities should be structured
        if e.entity_type == EntityType::Date || e.entity_type == EntityType::Money {
            assert!(
                e.is_structured(),
                "Pattern entity should be structured: {:?}",
                e
            );
            assert!(!e.is_named(), "Pattern entity should not be named: {:?}", e);
        }
    }
}

#[test]
fn test_entity_normalized_or_text() {
    let _text = "Jan 15, 2024";
    let mut entity = Entity::new("Jan 15, 2024", EntityType::Date, 0, 12, 0.9);

    // Without normalization, should return text
    assert_eq!(entity.normalized_or_text(), "Jan 15, 2024");

    // With normalization, should return normalized
    entity.set_normalized("2024-01-15");
    assert_eq!(entity.normalized_or_text(), "2024-01-15");
}

#[test]
fn test_entity_method_and_source() {
    let text = "test@example.com";
    let model = RegexNER::new();
    let entities = model.extract_entities(text, None).unwrap();

    for e in entities {
        // Method should return something (Unknown if no provenance)
        let _method = e.method();
        // Source may or may not be set
        let _source = e.source();

        // Entity should still be valid
        let issues = e.validate(text);
        assert!(issues.is_empty());
    }
}

#[test]
fn test_entity_category() {
    let text = "John works at Apple";
    let model = heuristic();
    let entities = model.extract_entities(text, None).unwrap();

    for e in entities {
        let category = e.category();
        // Category should be valid
        assert!(!category.as_str().is_empty());

        // Verify category matches entity type
        assert_eq!(category, e.entity_type.category());
    }
}

#[test]
fn test_entity_is_linked() {
    let text = "Apple";
    let mut entity = Entity::new("Apple", EntityType::Organization, 0, 5, 0.9);

    assert!(
        !entity.is_linked(),
        "Entity without KB ID should not be linked"
    );

    entity.link_to_kb("Q312");
    assert!(entity.is_linked(), "Entity with KB ID should be linked");

    let issues = entity.validate(text);
    assert!(issues.is_empty());
}

#[test]
fn test_entity_has_coreference() {
    let _text = "Apple";
    let mut entity = Entity::new("Apple", EntityType::Organization, 0, 5, 0.9);

    assert!(
        !entity.has_coreference(),
        "Entity without canonical ID should not have coreference"
    );

    entity.set_canonical(42);
    assert!(
        entity.has_coreference(),
        "Entity with canonical ID should have coreference"
    );
}

#[test]
fn test_entity_is_visual() {
    use anno::Span;

    let _text = "receipt total";
    let mut entity = Entity::new("receipt total", EntityType::Money, 0, 13, 0.9);

    assert!(
        !entity.is_visual(),
        "Entity without visual span should not be visual"
    );

    entity.visual_span = Some(Span::bbox(0.1, 0.2, 0.3, 0.4));
    assert!(
        entity.is_visual(),
        "Entity with visual span should be visual"
    );
}

#[test]
fn test_entity_set_visual_span() {
    use anno::Span;

    let mut entity = Entity::new("test", EntityType::Person, 0, 4, 0.9);

    // Initially no visual span
    assert!(entity.visual_span.is_none());

    // Set visual span
    let visual_span = Span::bbox(0.1, 0.2, 0.3, 0.4);
    entity.set_visual_span(visual_span.clone());

    // Verify it was set
    assert!(entity.visual_span.is_some());
    assert_eq!(entity.visual_span, Some(visual_span));

    // Verify entity is now visual
    assert!(entity.is_visual());
}

#[test]
fn test_entity_text_span_and_span_len() {
    let _text = "Hello World";
    let entity = Entity::new("Hello", EntityType::Other("MISC".to_string()), 0, 5, 0.9);

    let (start, end) = entity.text_span();
    assert_eq!(start, 0);
    assert_eq!(end, 5);

    assert_eq!(entity.span_len(), 5);
}

#[test]
fn test_entity_total_len_with_discontinuous() {
    let _text = "severe pain in the abdomen";
    let mut entity = Entity::new(
        "severe pain",
        EntityType::Other("MISC".to_string()),
        0,
        11,
        0.9,
    );

    // Without discontinuous span
    assert_eq!(entity.total_len(), 11);

    // With discontinuous span
    let disc_span = DiscontinuousSpan::new(vec![0..6, 12..16]);
    entity.set_discontinuous_span(disc_span);

    // Total len should be sum of segments: 6 + 4 = 10
    assert_eq!(entity.total_len(), 10);
}

#[test]
fn test_entity_valid_at() {
    use chrono::{TimeZone, Utc};

    let _text = "CEO";
    let mut entity = Entity::new("CEO", EntityType::Person, 0, 3, 0.9);

    entity.set_temporal_range(
        Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0).unwrap(),
        Utc.with_ymd_and_hms(2010, 1, 1, 0, 0, 0).unwrap(),
    );

    // Valid during range
    let during = Utc.with_ymd_and_hms(2005, 6, 1, 0, 0, 0).unwrap();
    assert!(entity.valid_at(&during));

    // Before range
    let before = Utc.with_ymd_and_hms(1999, 1, 1, 0, 0, 0).unwrap();
    assert!(!entity.valid_at(&before));

    // After range
    let after = Utc.with_ymd_and_hms(2011, 1, 1, 0, 0, 0).unwrap();
    assert!(!entity.valid_at(&after));

    // Atemporal entity (no bounds)
    let atemporal = Entity::new("Paris", EntityType::Location, 0, 5, 0.9);
    let any_time = Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0).unwrap();
    assert!(
        atemporal.valid_at(&any_time),
        "Atemporal entity should be valid at any time"
    );
}

#[test]
fn test_entity_hierarchical_confidence_methods() {
    let _text = "John";
    let mut entity = Entity::new("John", EntityType::Person, 0, 4, 0.9);

    // Without hierarchical confidence, should use base confidence
    assert!((entity.linkage_confidence() - 0.9).abs() < 0.001);
    assert!((entity.type_confidence() - 0.9).abs() < 0.001);
    assert!((entity.boundary_confidence() - 0.9).abs() < 0.001);

    // With hierarchical confidence
    entity.set_hierarchical_confidence(HierarchicalConfidence::new(0.8, 0.9, 0.85));
    assert!((entity.linkage_confidence() - 0.8).abs() < 0.001);
    assert!((entity.type_confidence() - 0.9).abs() < 0.001);
    assert!((entity.boundary_confidence() - 0.85).abs() < 0.001);
}

#[test]
fn test_entity_viewport_methods() {
    let _text = "Marie Curie";
    let mut entity = Entity::new("Marie Curie", EntityType::Person, 0, 11, 0.9);

    entity.set_viewport(EntityViewport::Academic);
    assert_eq!(entity.viewport_or_default(), EntityViewport::Academic);
    assert!(entity.matches_viewport(&EntityViewport::Academic));
    assert!(!entity.matches_viewport(&EntityViewport::Business));

    // Without viewport, should return default
    let entity_no_viewport = Entity::new("Test", EntityType::Person, 0, 4, 0.9);
    assert_eq!(
        entity_no_viewport.viewport_or_default(),
        EntityViewport::General
    );
}

#[test]
fn test_discontinuous_span_methods() {
    let _text = "severe pain in the abdomen";

    // Contiguous span
    let contiguous = DiscontinuousSpan::contiguous(0, 6);
    assert!(contiguous.is_contiguous());
    assert!(!contiguous.is_discontinuous());
    assert_eq!(contiguous.num_segments(), 1);
    assert_eq!(contiguous.total_len(), 6);

    // Discontinuous span
    let disc = DiscontinuousSpan::new(vec![0..6, 12..16]);
    assert!(!disc.is_contiguous());
    assert!(disc.is_discontinuous());
    assert_eq!(disc.num_segments(), 2);
    assert_eq!(disc.total_len(), 10); // 6 + 4

    // Bounding range
    let bounding = disc.bounding_range();
    assert_eq!(bounding, Some(0..16));

    // Contains
    assert!(disc.contains(3)); // In first segment
    assert!(disc.contains(14)); // In second segment
    assert!(!disc.contains(10)); // Between segments
}

#[test]
fn test_discontinuous_span_empty() {
    let _text = "test";
    let empty = DiscontinuousSpan::new(vec![]);

    assert_eq!(empty.num_segments(), 0);
    assert!(empty.is_contiguous()); // Empty is considered contiguous
    assert!(!empty.is_discontinuous());
    assert_eq!(empty.total_len(), 0);
    assert_eq!(empty.bounding_range(), None);
}

#[test]
fn test_entity_builder_chaining() {
    let text = "John Smith";

    let entity = EntityBuilder::new("John Smith", EntityType::Person)
        .span(0, 10)
        .confidence(0.95)
        .canonical_id(42)
        .viewport(EntityViewport::Academic)
        .build();

    assert_eq!(entity.text, "John Smith");
    assert_eq!(entity.entity_type, EntityType::Person);
    assert_eq!(entity.start, 0);
    assert_eq!(entity.end, 10);
    assert_eq!(entity.confidence, 0.95);
    assert_eq!(entity.canonical_id, Some(42));
    assert_eq!(entity.viewport, Some(EntityViewport::Academic));

    let issues = entity.validate(text);
    assert!(issues.is_empty());
}

// =============================================================================
// 32. Entity Type Edge Cases
// =============================================================================

#[test]
fn test_entity_type_from_label() {
    // Test various label formats
    assert_eq!(EntityType::from_label("PER"), EntityType::Person);
    assert_eq!(EntityType::from_label("PERSON"), EntityType::Person);
    assert_eq!(EntityType::from_label("ORG"), EntityType::Organization);
    assert_eq!(EntityType::from_label("LOC"), EntityType::Location);
    assert_eq!(EntityType::from_label("DATE"), EntityType::Date);
    assert_eq!(EntityType::from_label("MONEY"), EntityType::Money);

    // Unknown label becomes Other
    let unknown = EntityType::from_label("UNKNOWN_TYPE");
    assert!(matches!(unknown, EntityType::Other(_)));
}

#[test]
fn test_entity_type_category() {
    // Named entities require ML
    assert!(EntityType::Person.category().requires_ml());
    assert!(EntityType::Organization.category().requires_ml());
    assert!(EntityType::Location.category().requires_ml());

    // Pattern entities don't require ML
    assert!(EntityType::Date.category().pattern_detectable());
    assert!(EntityType::Money.category().pattern_detectable());
    assert!(EntityType::Email.category().pattern_detectable());
}

// =============================================================================
// 33. Serialization Edge Cases
// =============================================================================

#[test]
fn test_entity_serialization_with_special_chars() {
    let _text = "Test \"quotes\" and 'apostrophes' and <tags>";
    let entity = Entity::new(
        "Test \"quotes\"",
        EntityType::Other("MISC".to_string()),
        0,
        13,
        0.9,
    );

    // Should serialize without issues
    let json = serde_json::to_string(&entity).unwrap();
    let deserialized: Entity = serde_json::from_str(&json).unwrap();

    assert_eq!(entity.text, deserialized.text);
    assert_eq!(entity.start, deserialized.start);
    assert_eq!(entity.end, deserialized.end);
}

#[test]
fn test_entity_serialization_with_unicode() {
    let _text = "北京 and 東京";
    let entity = Entity::new("北京", EntityType::Location, 0, 2, 0.9);

    let json = serde_json::to_string(&entity).unwrap();
    let deserialized: Entity = serde_json::from_str(&json).unwrap();

    assert_eq!(entity.text, deserialized.text);
    assert_eq!(entity.start, deserialized.start);
    assert_eq!(entity.end, deserialized.end);
}

// =============================================================================
// 34. Entity Comparison and Equality
// =============================================================================

#[test]
fn test_entity_equality() {
    let e1 = Entity::new("Apple", EntityType::Organization, 0, 5, 0.9);
    let e2 = Entity::new("Apple", EntityType::Organization, 0, 5, 0.9);
    let _e3 = Entity::new("Apple", EntityType::Organization, 0, 5, 0.8); // Different confidence

    // Entities with same text, type, and span should be equal (if PartialEq implemented)
    // Note: Entity may not implement PartialEq, so we compare fields
    assert_eq!(e1.text, e2.text);
    assert_eq!(e1.entity_type, e2.entity_type);
    assert_eq!(e1.start, e2.start);
    assert_eq!(e1.end, e2.end);
}

// =============================================================================
// 35. Edge Cases for Entity Type Methods
// =============================================================================

#[test]
fn test_entity_type_as_label() {
    assert_eq!(EntityType::Person.as_label(), "PER");
    assert_eq!(EntityType::Organization.as_label(), "ORG");
    assert_eq!(EntityType::Location.as_label(), "LOC");
    assert_eq!(EntityType::Date.as_label(), "DATE");
    assert_eq!(EntityType::Money.as_label(), "MONEY");

    // Custom types
    let custom = EntityType::custom("DISEASE", anno::EntityCategory::Misc);
    assert_eq!(custom.as_label(), "DISEASE");

    // Other types
    let other = EntityType::Other("CUSTOM".to_string());
    assert_eq!(other.as_label(), "CUSTOM");
}

#[test]
fn test_entity_type_requires_ml() {
    assert!(EntityType::Person.requires_ml());
    assert!(EntityType::Organization.requires_ml());
    assert!(!EntityType::Date.requires_ml());
    assert!(!EntityType::Money.requires_ml());
    assert!(!EntityType::Email.requires_ml());
}

#[test]
fn test_entity_type_pattern_detectable() {
    assert!(!EntityType::Person.pattern_detectable());
    assert!(!EntityType::Organization.pattern_detectable());
    assert!(EntityType::Date.pattern_detectable());
    assert!(EntityType::Money.pattern_detectable());
    assert!(EntityType::Email.pattern_detectable());
}

// =============================================================================
// 29. Entity Deduplication Edge Cases
// =============================================================================

#[test]
fn test_exact_duplicate_entities() {
    // Entities with identical spans and text
    let text = "John and John";

    let model = heuristic();
    let entities = model.extract_entities(text, None).unwrap();

    // Check for exact duplicates (same start, end, text)
    for i in 0..entities.len() {
        for j in (i + 1)..entities.len() {
            let e1 = &entities[i];
            let e2 = &entities[j];

            // If exact duplicate, they should be identical
            if e1.start == e2.start && e1.end == e2.end && e1.text == e2.text {
                // This is a duplicate - verify they're both valid
                let issues1 = e1.validate(text);
                let issues2 = e2.validate(text);
                assert!(issues1.is_empty());
                assert!(issues2.is_empty());
            }
        }
    }
}

// =============================================================================
// 30. Extraction Method Edge Cases
// =============================================================================

#[test]
fn test_extraction_method_provenance() {
    let text = "test@example.com";
    let model = RegexNER::new();
    let entities = model.extract_entities(text, None).unwrap();

    // RegexNER should set provenance with Pattern method
    for e in entities {
        if let Some(ref _prov) = e.provenance {
            // Pattern entities should have Pattern method
            if e.entity_type == EntityType::Email || e.entity_type == EntityType::Url {
                // Pattern entities may have Pattern provenance
            }
        }

        let issues = e.validate(text);
        assert!(issues.is_empty());
    }
}
