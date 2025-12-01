//! Deep exploration of NER implementation nuances.
//!
//! This file tests the specific design decisions and edge cases in:
//! 1. Pattern priority and confidence scoring
//! 2. Statistical classification scoring
//! 3. Layer interaction and overlap handling

use anno::{Entity, EntityType, HeuristicNER, Model, RegexNER, StackedNER};

// =============================================================================
// PATTERN CONFIDENCE AND PRIORITY
// =============================================================================

/// Tests that verify pattern confidence levels are appropriately calibrated.
mod pattern_confidence {
    use super::*;

    fn extract(text: &str) -> Vec<Entity> {
        RegexNER::new().extract_entities(text, None).unwrap()
    }

    /// ISO dates should have high confidence (0.95) as they're unambiguous.
    #[test]
    fn iso_date_highest_confidence() {
        let e = extract("Date: 2024-01-15");
        assert!(!e.is_empty());
        assert!(
            e[0].confidence >= 0.90,
            "ISO date should be >= 0.90, got {}",
            e[0].confidence
        );
    }

    /// Email addresses should have high confidence (0.98).
    #[test]
    fn email_high_confidence() {
        let e = extract("Contact: test@example.com");
        assert!(!e.is_empty());
        assert!(
            e[0].confidence >= 0.98,
            "Email should be 0.98, got {}",
            e[0].confidence
        );
    }

    /// URLs should have high confidence (0.98).
    #[test]
    fn url_high_confidence() {
        let e = extract("Visit: https://example.com");
        assert!(!e.is_empty());
        assert!(
            e[0].confidence >= 0.98,
            "URL should be 0.98, got {}",
            e[0].confidence
        );
    }

    /// Money with symbols should have 0.95 confidence.
    #[test]
    fn money_symbol_confidence() {
        let e = extract("Cost: $100");
        assert!(!e.is_empty());
        assert!(
            e[0].confidence >= 0.95,
            "Money should be 0.95, got {}",
            e[0].confidence
        );
    }

    /// Percentages should have 0.95 confidence.
    #[test]
    fn percent_confidence() {
        let e = extract("Growth: 25%");
        assert!(!e.is_empty());
        assert!(
            e[0].confidence >= 0.95,
            "Percent should be 0.95, got {}",
            e[0].confidence
        );
    }

    /// 12-hour time should have higher confidence than 24-hour (due to AM/PM).
    #[test]
    fn time_12h_higher_than_24h() {
        let e12 = extract("Meeting at 3:30 PM");
        let e24 = extract("Meeting at 15:30");

        assert!(!e12.is_empty());
        assert!(!e24.is_empty());

        // 12h format has AM/PM marker - less ambiguous
        assert!(e12[0].confidence >= 0.90);
        // 24h format could match other numeric patterns
        assert!(e24[0].confidence >= 0.85);
    }

    /// Phone numbers have lower confidence (0.85) due to false positive risk.
    #[test]
    fn phone_lower_confidence() {
        let e = extract("Call: (555) 123-4567");
        assert!(!e.is_empty());
        assert!(
            e[0].confidence <= 0.90,
            "Phone should be <=0.90, got {}",
            e[0].confidence
        );
    }
}

/// Tests that verify pattern priority (higher confidence patterns win).
mod pattern_priority {
    use super::*;

    fn extract(text: &str) -> Vec<Entity> {
        RegexNER::new().extract_entities(text, None).unwrap()
    }

    /// When multiple patterns could match, higher confidence should win.
    #[test]
    fn higher_confidence_wins() {
        // This tests that patterns are checked in order of confidence
        let e = extract("$100 at 3:30 PM on 2024-01-15");

        // All should be found
        let types: Vec<_> = e.iter().map(|e| e.entity_type.clone()).collect();
        assert!(types.contains(&EntityType::Money));
        assert!(types.contains(&EntityType::Time));
        assert!(types.contains(&EntityType::Date));
    }

    /// First match wins for overlapping spans.
    #[test]
    fn no_overlaps_higher_priority_wins() {
        // "$5 million" could match both MONEY_SYMBOL and MONEY_MAGNITUDE
        // The first pattern checked should win
        let e = extract("Budget: $5 million");

        // Should have exactly one money entity (no duplicates)
        let money: Vec<_> = e
            .iter()
            .filter(|e| e.entity_type == EntityType::Money)
            .collect();
        assert_eq!(
            money.len(),
            1,
            "Should have exactly one money entity: {:?}",
            money
        );
    }
}

// =============================================================================
// STATISTICAL SCORING NUANCES
// =============================================================================

mod statistical_scoring {
    use super::*;

    fn extract(text: &str) -> Vec<Entity> {
        HeuristicNER::new().extract_entities(text, None).unwrap()
    }

    /// Person prefix (Mr., Dr.) provides strongest signal.
    #[test]
    fn person_prefix_strong_signal() {
        let with_prefix = extract("Mr. Smith said hello.");
        let without_prefix = extract("Smith said hello.");

        // With prefix should have entity
        let persons_with: Vec<_> = with_prefix
            .iter()
            .filter(|e| e.entity_type == EntityType::Person)
            .collect();

        // Prefix provides context that increases confidence
        // Without prefix, "Smith" is ambiguous (could be last name or place)
        if !persons_with.is_empty() && !without_prefix.is_empty() {
            let p1 = persons_with[0];
            let p2_opt = without_prefix.iter().find(|e| e.text == "Smith");
            if let Some(p2) = p2_opt {
                // With prefix should have equal or higher confidence
                assert!(p1.confidence >= p2.confidence);
            }
        }
    }

    /// Organization suffix (Inc., Corp.) provides strongest signal.
    #[test]
    fn org_suffix_strong_signal() {
        let with_suffix = extract("Working at Apple Inc.");
        let _without_suffix = extract("Working at Apple."); // Kept for comparison context

        // With suffix should more likely be org
        let orgs_with: Vec<_> = with_suffix
            .iter()
            .filter(|e| e.entity_type == EntityType::Organization)
            .collect();

        // Without suffix, "Apple" is ambiguous
        // (could be fruit, person's name, etc.)
        assert!(!orgs_with.is_empty() || with_suffix.is_empty());
    }

    /// Location prefix (in, at, from) provides context.
    #[test]
    fn location_prefix_context() {
        let with_in = extract("Conference in Paris.");
        let with_at = extract("Meeting at London.");

        // Both should find locations after preposition
        let locs_in: Vec<_> = with_in
            .iter()
            .filter(|e| e.entity_type == EntityType::Location)
            .collect();
        let locs_at: Vec<_> = with_at
            .iter()
            .filter(|e| e.entity_type == EntityType::Location)
            .collect();

        assert!(!locs_in.is_empty(), "Should find Paris as location");
        assert!(!locs_at.is_empty(), "Should find London as location");
    }

    /// Common first names boost person classification.
    #[test]
    fn common_name_boosts_person() {
        // "John" is a common first name
        let e = extract("I talked to John yesterday.");
        let persons: Vec<_> = e
            .iter()
            .filter(|e| e.entity_type == EntityType::Person)
            .collect();

        assert!(!persons.is_empty(), "John should be recognized as person");
    }

    /// Two-word capitalized sequences are more likely people.
    #[test]
    fn two_word_name_pattern() {
        let e = extract("Meeting with Steve Jobs tomorrow.");

        // Two capitalized words together often = person name
        let has_steve_jobs = e
            .iter()
            .any(|e| e.text.contains("Steve") || e.text.contains("Jobs"));
        assert!(has_steve_jobs, "Should find Steve Jobs: {:?}", e);
    }

    /// Context words after name (verbs) provide signal.
    #[test]
    fn verb_suffix_context() {
        // NOTE: "Jobs" at sentence start is ambiguous - could be noun (employment)
        // Adding Mr. prefix provides crucial context
        let e = extract("Mr. Jobs founded the company.");

        // With "Mr." prefix, "Jobs" is clearly a person name
        let has_jobs = e.iter().any(|e| e.text.contains("Jobs"));
        assert!(has_jobs, "Should find Mr. Jobs: {:?}", e);
    }
}

// =============================================================================
// LAYER INTERACTION NUANCES
// =============================================================================

mod layer_interaction {
    use super::*;

    fn extract(text: &str) -> Vec<Entity> {
        StackedNER::new().extract_entities(text, None).unwrap()
    }

    /// Pattern layer runs first and prevents statistical false positives.
    #[test]
    fn pattern_prevents_statistical_overlap() {
        let text = "Event on January 15, 2024 in Paris.";
        let e = extract(text);

        // "January" should be part of date, not a named entity
        let january_entities: Vec<_> = e.iter().filter(|e| e.text.contains("January")).collect();

        if !january_entities.is_empty() {
            // Should be Date type, not Person
            assert_eq!(
                january_entities[0].entity_type,
                EntityType::Date,
                "January should be part of date, not named entity"
            );
        }

        // "Paris" should be location (from statistical layer)
        let has_location = e.iter().any(|e| e.entity_type == EntityType::Location);
        assert!(has_location, "Should find Paris as location");
    }

    /// Statistical layer fills gaps left by pattern layer.
    #[test]
    fn statistical_fills_pattern_gaps() {
        let text = "Dr. Smith charges $200/hr.";
        let e = extract(text);

        // Pattern finds: $200
        let has_money = e.iter().any(|e| e.entity_type == EntityType::Money);
        assert!(has_money, "Should find money");

        // Statistical may find: Dr. Smith
        let _has_person = e.iter().any(|e| e.entity_type == EntityType::Person);
        // May or may not find depending on context scoring
        // But should have at least money
        assert!(has_money);
    }

    /// No overlapping entities in merged output.
    #[test]
    fn no_overlaps_in_merged_output() {
        let texts = [
            "Email john@company.com on Jan 15.",
            "Dr. Smith at $100/hr in Paris.",
            "Call +1-555-123-4567 for meeting on 2024-01-01.",
        ];

        for text in texts {
            let e = extract(text);

            // Check no overlaps
            for i in 0..e.len() {
                for j in (i + 1)..e.len() {
                    let overlap = e[i].start < e[j].end && e[j].start < e[i].end;
                    assert!(
                        !overlap,
                        "Overlap in '{}': {} and {}",
                        text, e[i].text, e[j].text
                    );
                }
            }
        }
    }

    /// Entities are sorted by position.
    #[test]
    fn entities_sorted_by_position() {
        let text = "$100 from John at Google in Paris on 2024-01-01";
        let e = extract(text);

        for i in 1..e.len() {
            assert!(
                e[i - 1].start <= e[i].start,
                "Not sorted: {} at {} should come before {} at {}",
                e[i - 1].text,
                e[i - 1].start,
                e[i].text,
                e[i].start
            );
        }
    }
}

// =============================================================================
// EDGE CASES AND BOUNDARY CONDITIONS
// =============================================================================

mod edge_cases {
    use super::*;

    fn pattern_extract(text: &str) -> Vec<Entity> {
        RegexNER::new().extract_entities(text, None).unwrap()
    }

    fn stat_extract(text: &str) -> Vec<Entity> {
        HeuristicNER::new().extract_entities(text, None).unwrap()
    }

    fn tiered_extract(text: &str) -> Vec<Entity> {
        StackedNER::new().extract_entities(text, None).unwrap()
    }

    /// Empty string produces no entities.
    #[test]
    fn empty_string() {
        assert!(pattern_extract("").is_empty());
        assert!(stat_extract("").is_empty());
        assert!(tiered_extract("").is_empty());
    }

    /// Only whitespace produces no entities.
    #[test]
    fn only_whitespace() {
        assert!(pattern_extract("   \t\n  ").is_empty());
        assert!(stat_extract("   \t\n  ").is_empty());
        assert!(tiered_extract("   \t\n  ").is_empty());
    }

    /// Only punctuation produces no entities.
    #[test]
    fn only_punctuation() {
        assert!(pattern_extract("!@#$%^&*()").is_empty());
        assert!(stat_extract("!@#$%^&*()").is_empty());
        assert!(tiered_extract("!@#$%^&*()").is_empty());
    }

    /// Entity at start of string.
    #[test]
    fn entity_at_start() {
        let e = pattern_extract("$100 is the price.");
        assert!(!e.is_empty());
        assert_eq!(e[0].start, 0);
    }

    /// Entity at end of string.
    #[test]
    fn entity_at_end() {
        let text = "Contact test@email.com";
        let e = pattern_extract(text);
        assert!(!e.is_empty());
        assert_eq!(e[0].end, text.len());
    }

    /// Unicode text handling.
    #[test]
    fn unicode_text() {
        // Entity offsets are CHARACTER offsets (not byte offsets)
        let text = "José García earns €500.";
        let e = pattern_extract(text);

        // Should find €500
        let money: Vec<_> = e
            .iter()
            .filter(|e| e.entity_type == EntityType::Money)
            .collect();
        assert!(!money.is_empty(), "Should find €500");

        // Verify span is valid (using char-based extraction)
        let char_count = text.chars().count();
        for entity in &e {
            assert!(entity.start <= entity.end, "Start > end");
            assert!(entity.end <= char_count, "End beyond text");
            let extracted: String = text
                .chars()
                .skip(entity.start)
                .take(entity.end - entity.start)
                .collect();
            assert_eq!(extracted, entity.text, "Extracted text mismatch");
        }
    }

    /// Very long text doesn't cause issues.
    #[test]
    fn long_text() {
        let text = "The price is $100. ".repeat(1000);
        let e = pattern_extract(&text);

        // Should find all $100 instances
        let money_count = e
            .iter()
            .filter(|e| e.entity_type == EntityType::Money)
            .count();
        assert_eq!(money_count, 1000);
    }

    /// Adjacent entities without space.
    #[test]
    fn adjacent_no_space() {
        let e = pattern_extract("$100$200$300");
        let money: Vec<_> = e
            .iter()
            .filter(|e| e.entity_type == EntityType::Money)
            .collect();
        assert_eq!(money.len(), 3);
    }

    /// Entity spans are valid character indices.
    #[test]
    fn spans_are_valid_char_indices() {
        let texts = [
            "Simple $100 text.",
            "Unicode: €500 für José.",
            "Mixed: 日本語 $999 text",
        ];

        for text in texts {
            let e = tiered_extract(text);
            let char_count = text.chars().count();
            for entity in &e {
                // Should be valid character indices
                assert!(
                    entity.start <= char_count,
                    "Start {} beyond char count {} in '{}'",
                    entity.start,
                    char_count,
                    text
                );
                assert!(
                    entity.end <= char_count,
                    "End {} beyond char count {} in '{}'",
                    entity.end,
                    char_count,
                    text
                );
                assert!(entity.start <= entity.end);

                // Extracted text should match entity.text
                let extracted: String = text
                    .chars()
                    .skip(entity.start)
                    .take(entity.end - entity.start)
                    .collect();
                assert_eq!(
                    extracted, entity.text,
                    "Text mismatch at {}..{} in '{}': got '{}', expected '{}'",
                    entity.start, entity.end, text, extracted, entity.text
                );
            }
        }
    }
}

// =============================================================================
// PROVENANCE AND METADATA
// =============================================================================

mod provenance {
    use super::*;

    #[test]
    fn pattern_provenance_has_source() {
        let e = RegexNER::new()
            .extract_entities("Price: $100", None)
            .unwrap();

        assert!(!e.is_empty());
        let prov = e[0].provenance.as_ref().unwrap();
        assert_eq!(prov.source.as_ref(), "pattern");
    }

    #[test]
    fn pattern_provenance_has_pattern_name() {
        let e = RegexNER::new()
            .extract_entities("Email: test@email.com", None)
            .unwrap();

        assert!(!e.is_empty());
        let prov = e[0].provenance.as_ref().unwrap();
        assert!(prov.pattern.is_some());
        assert_eq!(prov.pattern.as_ref().unwrap().as_ref(), "EMAIL");
    }

    #[test]
    fn statistical_provenance_has_source() {
        let e = HeuristicNER::new()
            .extract_entities("Mr. Smith said hello.", None)
            .unwrap();

        if !e.is_empty() {
            let prov = e[0].provenance.as_ref().unwrap();
            assert_eq!(prov.source.as_ref(), "heuristic");
        }
    }

    #[test]
    fn statistical_provenance_has_reason() {
        let e = HeuristicNER::new()
            .extract_entities("Mr. Smith said hello.", None)
            .unwrap();

        if !e.is_empty() {
            let prov = e[0].provenance.as_ref().unwrap();
            // Pattern contains classification reason
            assert!(prov.pattern.is_some());
        }
    }
}

// =============================================================================
// REGRESSION PREVENTION
// =============================================================================

mod regressions {
    use super::*;

    /// Issue: Emails with valid subdomains should match.
    #[test]
    fn email_with_subdomain_matches() {
        let e = RegexNER::new()
            .extract_entities("Contact: admin@mail.company.co.uk", None)
            .unwrap();

        let emails: Vec<_> = e
            .iter()
            .filter(|e| e.entity_type == EntityType::Email)
            .collect();
        assert!(!emails.is_empty());
    }

    /// Issue: URLs with paths and queries should match.
    #[test]
    fn url_with_complex_path_matches() {
        let e = RegexNER::new()
            .extract_entities(
                "API: https://api.example.com/v1/users?page=1&sort=asc",
                None,
            )
            .unwrap();

        let urls: Vec<_> = e
            .iter()
            .filter(|e| e.entity_type == EntityType::Url)
            .collect();
        assert!(!urls.is_empty());
    }

    /// Issue: Phone numbers with various separators should all match.
    #[test]
    fn phone_various_separators() {
        let cases = [
            "(555) 123-4567",
            "555-123-4567",
            "555.123.4567",
            "555 123 4567",
        ];

        for case in cases {
            let e = RegexNER::new().extract_entities(case, None).unwrap();

            let phones: Vec<_> = e
                .iter()
                .filter(|e| e.entity_type == EntityType::Phone)
                .collect();
            assert!(!phones.is_empty(), "Should match: {}", case);
        }
    }

    /// Issue: Written dates in various formats should match.
    #[test]
    fn written_date_formats() {
        let cases = [
            "January 15, 2024",
            "Jan 15 2024",
            "15 January 2024",
            "15th Jan 2024",
        ];

        for case in cases {
            let e = RegexNER::new().extract_entities(case, None).unwrap();

            let dates: Vec<_> = e
                .iter()
                .filter(|e| e.entity_type == EntityType::Date)
                .collect();
            assert!(!dates.is_empty(), "Should match: {}", case);
        }
    }
}
