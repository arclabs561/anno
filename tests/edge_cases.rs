//! Edge case tests for NER backends.
//!
//! Tests boundary conditions, unusual inputs, and potential failure modes.

use anno::{Entity, EntityType, HeuristicNER, Model, RegexNER, StackedNER};

fn has_type(entities: &[Entity], ty: &EntityType) -> bool {
    entities.iter().any(|e| e.entity_type == *ty)
}

// =============================================================================
// Empty and Whitespace
// =============================================================================

mod empty_input {
    use super::*;

    #[test]
    fn pattern_empty_string() {
        let ner = RegexNER::new();
        let e = ner.extract_entities("", None).unwrap();
        assert!(e.is_empty());
    }

    #[test]
    fn statistical_empty_string() {
        let ner = HeuristicNER::new();
        let e = ner.extract_entities("", None).unwrap();
        assert!(e.is_empty());
    }

    #[test]
    fn stacked_empty_string() {
        let ner = StackedNER::new();
        let e = ner.extract_entities("", None).unwrap();
        assert!(e.is_empty());
    }

    #[test]
    fn whitespace_only() {
        let ner = StackedNER::new();
        assert!(ner.extract_entities("   ", None).unwrap().is_empty());
        assert!(ner.extract_entities("\t\t", None).unwrap().is_empty());
        assert!(ner.extract_entities("\n\n", None).unwrap().is_empty());
        assert!(ner.extract_entities("  \t\n  ", None).unwrap().is_empty());
    }

    #[test]
    fn newlines_only() {
        let ner = StackedNER::new();
        let e = ner.extract_entities("\n\n\n", None).unwrap();
        assert!(e.is_empty());
    }
}

// =============================================================================
// Unicode
// =============================================================================

mod unicode {
    use super::*;

    #[test]
    fn pattern_with_emoji() {
        let ner = RegexNER::new();
        let e = ner.extract_entities("Meeting costs $100 🎉", None).unwrap();
        assert!(has_type(&e, &EntityType::Money));
    }

    #[test]
    fn pattern_with_chinese() {
        let ner = RegexNER::new();
        let e = ner.extract_entities("价格是 $100 美元", None).unwrap();
        assert!(has_type(&e, &EntityType::Money));
    }

    #[test]
    fn pattern_with_arabic() {
        let ner = RegexNER::new();
        let e = ner.extract_entities("السعر $100", None).unwrap();
        assert!(has_type(&e, &EntityType::Money));
    }

    #[test]
    fn pattern_with_mixed_scripts() {
        let ner = RegexNER::new();
        let e = ner
            .extract_entities("日期: 2024-01-15, email: test@example.com", None)
            .unwrap();
        assert!(has_type(&e, &EntityType::Date));
        assert!(has_type(&e, &EntityType::Email));
    }

    #[test]
    fn unicode_character_boundaries() {
        let ner = RegexNER::new();
        // Multi-byte characters before pattern
        let text = "价格：$500";
        let e = ner.extract_entities(text, None).unwrap();
        if !e.is_empty() {
            let entity = &e[0];
            // Entity offsets are CHARACTER offsets, extract using chars()
            let extracted: String = text
                .chars()
                .skip(entity.start)
                .take(entity.end - entity.start)
                .collect();
            assert!(
                extracted.contains("$500") || extracted == "$500",
                "Expected $500, got '{}' at char {}..{}",
                extracted,
                entity.start,
                entity.end
            );
        }
    }

    #[test]
    fn emoji_before_entity() {
        let ner = RegexNER::new();
        let e = ner
            .extract_entities("🚀 Launch on 2024-01-15", None)
            .unwrap();
        assert!(has_type(&e, &EntityType::Date));
    }

    #[test]
    fn emoji_inside_should_not_match() {
        let ner = RegexNER::new();
        // Emoji breaks the date pattern
        let e = ner.extract_entities("2024🎉01-15", None).unwrap();
        // Should NOT match as a valid date
        assert!(!has_type(&e, &EntityType::Date));
    }

    #[test]
    fn statistical_with_accented_names() {
        let ner = HeuristicNER::new();
        // Accented names should still be detected as title case
        let e = ner
            .extract_entities("Meeting with José García", None)
            .unwrap();
        // May or may not detect - heuristic system
        // Just verify no panic
        let _ = e.len(); // Just ensure extraction succeeded
    }

    #[test]
    fn zero_width_characters() {
        let ner = RegexNER::new();
        // Zero-width joiner and other invisible chars
        let e = ner.extract_entities("$100\u{200B}dollars", None).unwrap();
        assert!(has_type(&e, &EntityType::Money));
    }
}

// =============================================================================
// Very Long Input
// =============================================================================

mod long_input {
    use super::*;

    #[test]
    fn very_long_text_no_entities() {
        let ner = StackedNER::new();
        let text = "word ".repeat(10000);
        let e = ner.extract_entities(&text, None).unwrap();
        assert!(e.is_empty());
    }

    #[test]
    fn very_long_text_with_entities() {
        let ner = RegexNER::new();
        let filler = "Lorem ipsum dolor sit amet. ".repeat(1000);
        let text = format!("{}Price: $100. {}", filler, filler);
        let e = ner.extract_entities(&text, None).unwrap();
        assert!(has_type(&e, &EntityType::Money));
    }

    #[test]
    fn many_entities() {
        let ner = RegexNER::new();
        let text = (1..=100)
            .map(|i| format!("Item {}: ${}", i, i * 10))
            .collect::<Vec<_>>()
            .join(". ");
        let e = ner.extract_entities(&text, None).unwrap();
        // Should find most of the money amounts
        let money_count = e
            .iter()
            .filter(|e| e.entity_type == EntityType::Money)
            .count();
        assert!(
            money_count >= 90,
            "Should find most money entities: {}",
            money_count
        );
    }

    #[test]
    fn single_very_long_word() {
        let ner = StackedNER::new();
        let long_word = "a".repeat(10000);
        let e = ner.extract_entities(&long_word, None).unwrap();
        // Should not panic, should return empty
        assert!(e.is_empty());
    }
}

// =============================================================================
// Special Characters
// =============================================================================

mod special_chars {
    use super::*;

    #[test]
    fn html_entities() {
        let ner = RegexNER::new();
        let e = ner.extract_entities("Price: &dollar;100", None).unwrap();
        // HTML entity is NOT a real dollar sign
        assert!(!has_type(&e, &EntityType::Money));
    }

    #[test]
    fn escaped_characters() {
        let ner = RegexNER::new();
        let e = ner.extract_entities("Price: \\$100", None).unwrap();
        // Backslash before $ - depends on regex
        // Just verify no panic
        let _ = e.len(); // Just ensure extraction succeeded
    }

    #[test]
    fn null_byte_handling() {
        let ner = StackedNER::new();
        // Null bytes should not cause panic
        let text = "Price: $100\0extra";
        let result = ner.extract_entities(text, None);
        // Should handle gracefully
        assert!(result.is_ok());
    }

    #[test]
    fn control_characters() {
        let ner = RegexNER::new();
        let e = ner.extract_entities("$100\x01\x02\x03", None).unwrap();
        assert!(has_type(&e, &EntityType::Money));
    }

    #[test]
    fn mixed_line_endings() {
        let ner = RegexNER::new();
        let e = ner
            .extract_entities("$100\r\n2024-01-15\n$200\r$300", None)
            .unwrap();
        let money_count = e
            .iter()
            .filter(|e| e.entity_type == EntityType::Money)
            .count();
        assert!(money_count >= 2);
    }

    #[test]
    fn tabs_in_text() {
        let ner = RegexNER::new();
        let e = ner
            .extract_entities("Price:\t$100\tDate:\t2024-01-15", None)
            .unwrap();
        assert!(has_type(&e, &EntityType::Money));
        assert!(has_type(&e, &EntityType::Date));
    }
}

// =============================================================================
// Boundary Conditions
// =============================================================================

mod boundaries {
    use super::*;

    #[test]
    fn entity_at_start() {
        let ner = RegexNER::new();
        let e = ner.extract_entities("$100 is the price", None).unwrap();
        assert!(has_type(&e, &EntityType::Money));
        assert_eq!(e[0].start, 0);
    }

    #[test]
    fn entity_at_end() {
        let ner = RegexNER::new();
        let e = ner.extract_entities("The price is $100", None).unwrap();
        assert!(has_type(&e, &EntityType::Money));
    }

    #[test]
    fn entity_is_entire_text() {
        let ner = RegexNER::new();
        let e = ner.extract_entities("$100", None).unwrap();
        assert_eq!(e.len(), 1);
        assert_eq!(e[0].start, 0);
        assert_eq!(e[0].end, 4);
    }

    #[test]
    fn adjacent_entities() {
        let ner = RegexNER::new();
        // Two entities right next to each other
        let e = ner.extract_entities("$100$200", None).unwrap();
        // Might be interpreted as one or two entities
        assert!(!e.is_empty());
    }

    #[test]
    fn entities_separated_by_single_space() {
        let ner = RegexNER::new();
        let e = ner.extract_entities("$100 $200", None).unwrap();
        let money_count = e
            .iter()
            .filter(|e| e.entity_type == EntityType::Money)
            .count();
        assert_eq!(money_count, 2);
    }

    #[test]
    fn overlapping_pattern_candidates() {
        let ner = RegexNER::new();
        // Could be interpreted multiple ways
        let e = ner.extract_entities("12/25/2024", None).unwrap();
        // Should pick one interpretation (date)
        assert!(e.len() <= 1);
    }
}

// =============================================================================
// Edge Cases in Patterns
// =============================================================================

mod pattern_edges {
    use super::*;

    #[test]
    fn money_edge_cases() {
        let ner = RegexNER::new();

        // Very large amounts
        let e = ner.extract_entities("$1,000,000,000", None).unwrap();
        assert!(has_type(&e, &EntityType::Money));

        // Small amounts
        let e = ner.extract_entities("$0.01", None).unwrap();
        assert!(has_type(&e, &EntityType::Money));

        // Various currencies (only $ is guaranteed)
        let e = ner.extract_entities("$100", None).unwrap();
        assert!(has_type(&e, &EntityType::Money));
    }

    #[test]
    fn date_edge_cases() {
        let ner = RegexNER::new();

        // Leap year date
        let e = ner.extract_entities("2024-02-29", None).unwrap();
        assert!(has_type(&e, &EntityType::Date));

        // End of year
        let e = ner.extract_entities("December 31, 2024", None).unwrap();
        assert!(has_type(&e, &EntityType::Date));

        // Start of year
        let e = ner.extract_entities("January 1, 2024", None).unwrap();
        assert!(has_type(&e, &EntityType::Date));
    }

    #[test]
    fn email_edge_cases() {
        let ner = RegexNER::new();

        // Subdomain
        let e = ner.extract_entities("test@mail.example.com", None).unwrap();
        assert!(has_type(&e, &EntityType::Email));

        // Plus addressing
        let e = ner.extract_entities("test+tag@example.com", None).unwrap();
        assert!(has_type(&e, &EntityType::Email));

        // Numbers in local part
        let e = ner.extract_entities("test123@example.com", None).unwrap();
        assert!(has_type(&e, &EntityType::Email));
    }

    #[test]
    fn time_edge_cases() {
        let ner = RegexNER::new();

        // Midnight
        let e = ner.extract_entities("at 12:00am", None).unwrap();
        assert!(has_type(&e, &EntityType::Time));

        // Noon
        let e = ner.extract_entities("at 12:00pm", None).unwrap();
        assert!(has_type(&e, &EntityType::Time));

        // 24-hour format
        let e = ner.extract_entities("at 23:59", None).unwrap();
        assert!(has_type(&e, &EntityType::Time));
    }

    #[test]
    fn percent_edge_cases() {
        let ner = RegexNER::new();

        // 100%
        let e = ner.extract_entities("100%", None).unwrap();
        assert!(has_type(&e, &EntityType::Percent));

        // Decimal percent
        let e = ner.extract_entities("3.14%", None).unwrap();
        assert!(has_type(&e, &EntityType::Percent));

        // Large percent
        let e = ner.extract_entities("500%", None).unwrap();
        assert!(has_type(&e, &EntityType::Percent));
    }

    #[test]
    fn phone_edge_cases() {
        let ner = RegexNER::new();

        // Various formats
        let e = ner.extract_entities("(555) 123-4567", None).unwrap();
        assert!(has_type(&e, &EntityType::Phone));

        let e = ner.extract_entities("555-123-4567", None).unwrap();
        assert!(has_type(&e, &EntityType::Phone));
    }
}

// =============================================================================
// Invalid/Malformed Patterns
// =============================================================================

mod invalid_patterns {
    use super::*;

    #[test]
    fn invalid_email_no_at() {
        let ner = RegexNER::new();
        let e = ner.extract_entities("testexample.com", None).unwrap();
        assert!(!has_type(&e, &EntityType::Email));
    }

    #[test]
    fn invalid_email_no_domain() {
        let ner = RegexNER::new();
        let e = ner.extract_entities("test@", None).unwrap();
        assert!(!has_type(&e, &EntityType::Email));
    }

    #[test]
    fn invalid_date_month_13() {
        let ner = RegexNER::new();
        // 13 is not a valid month, but regex might still match format
        let e = ner.extract_entities("2024-13-15", None).unwrap();
        // Regex doesn't validate month ranges, so this might match
        // Just verify no panic
        let _ = e.len(); // Just ensure extraction succeeded
    }

    #[test]
    fn almost_money_no_amount() {
        let ner = RegexNER::new();
        let e = ner.extract_entities("$ only", None).unwrap();
        assert!(!has_type(&e, &EntityType::Money));
    }

    #[test]
    fn almost_url_no_scheme() {
        let ner = RegexNER::new();
        let e = ner.extract_entities("example.com", None).unwrap();
        // Without scheme, might not be detected as URL
        // Just verify behavior is consistent
        let _ = e.len(); // Just ensure extraction succeeded
    }
}

// =============================================================================
// Property: Span Validity
// =============================================================================

mod span_validity {
    use super::*;

    fn check_spans(text: &str, entities: &[Entity]) {
        for entity in entities {
            assert!(
                entity.start <= entity.end,
                "Start {} should be <= end {} for '{}'",
                entity.start,
                entity.end,
                entity.text
            );
            assert!(
                entity.end <= text.len(),
                "End {} should be <= text len {} for '{}'",
                entity.end,
                text.len(),
                entity.text
            );
            // Verify extractable
            let extracted = &text[entity.start..entity.end];
            assert!(
                extracted.contains(&entity.text) || entity.text.contains(extracted),
                "Extracted '{}' should relate to entity text '{}'",
                extracted,
                entity.text
            );
        }
    }

    #[test]
    fn pattern_spans_valid() {
        let ner = RegexNER::new();
        let texts = [
            "$100 is the price",
            "Date: 2024-01-15",
            "Email: test@test.com",
            "Very long text with $500 somewhere in the middle of it all",
        ];
        for text in texts {
            let e = ner.extract_entities(text, None).unwrap();
            check_spans(text, &e);
        }
    }

    #[test]
    fn statistical_spans_valid() {
        let ner = HeuristicNER::new();
        let texts = [
            "Dr. John Smith is here",
            "Apple Inc. announced",
            "Meeting in New York City",
        ];
        for text in texts {
            let e = ner.extract_entities(text, None).unwrap();
            check_spans(text, &e);
        }
    }

    #[test]
    fn stacked_spans_valid() {
        let ner = StackedNER::new();
        let texts = [
            "Dr. Smith charges $100 on 2024-01-15",
            "Contact Apple Inc. at test@apple.com",
            "Meeting in NYC at 3pm costs $50",
        ];
        for text in texts {
            let e = ner.extract_entities(text, None).unwrap();
            check_spans(text, &e);
        }
    }
}
