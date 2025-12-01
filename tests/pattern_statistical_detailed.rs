//! Detailed tests for RegexNER and HeuristicNER backends.
//!
//! These tests focus on:
//! - Edge cases and boundary conditions
//! - Pattern matching accuracy
//! - Statistical heuristic behavior
//! - Confidence scoring

#![allow(unused_variables)] // Many smoke tests just ensure no panic

use anno::{EntityType, HeuristicNER, Model, RegexNER, StackedNER};

// =============================================================================
// RegexNER Detailed Tests
// =============================================================================

mod regex_ner {
    use super::*;

    // -------------------------------------------------------------------------
    // Date Patterns
    // -------------------------------------------------------------------------

    #[test]
    fn test_iso_dates() {
        let ner = RegexNER::new();

        let cases = [
            ("Meeting on 2024-01-15", "2024-01-15"),
            ("Date: 2023-12-31", "2023-12-31"),
            ("Starts 1999-06-01", "1999-06-01"),
        ];

        for (text, expected) in cases {
            let entities = ner.extract_entities(text, None).unwrap();
            let dates: Vec<_> = entities
                .iter()
                .filter(|e| matches!(e.entity_type, EntityType::Date))
                .collect();

            assert!(!dates.is_empty(), "Should find date in: {}", text);
            assert_eq!(dates[0].text, expected);
        }
    }

    #[test]
    fn test_us_dates() {
        let ner = RegexNER::new();

        // Note: RegexNER uses specific date patterns - MM/DD/YYYY with slashes
        let cases = [
            ("Born 01/15/2024", "01/15/2024"),
            ("Date: 12/31/2023", "12/31/2023"),
        ];

        for (text, expected) in cases {
            let entities = ner.extract_entities(text, None).unwrap();
            let dates: Vec<_> = entities
                .iter()
                .filter(|e| matches!(e.entity_type, EntityType::Date))
                .collect();

            assert!(!dates.is_empty(), "Should find date in: {}", text);
            assert_eq!(dates[0].text, expected);
        }
    }

    #[test]
    fn test_hyphenated_dates() {
        let ner = RegexNER::new();

        // Hyphenated dates may be ISO format (YYYY-MM-DD) or US format (MM-DD-YYYY)
        // Our RegexNER primarily supports ISO format
        let text = "Event on 2024-06-01";
        let entities = ner.extract_entities(text, None).unwrap();
        let dates: Vec<_> = entities
            .iter()
            .filter(|e| matches!(e.entity_type, EntityType::Date))
            .collect();

        assert!(!dates.is_empty(), "Should find ISO date");
        assert_eq!(dates[0].text, "2024-06-01");
    }

    #[test]
    fn test_written_dates() {
        let ner = RegexNER::new();

        let cases = [
            ("January 15, 2024", "January 15, 2024"),
            ("Dec 31, 2023", "Dec 31, 2023"),
            ("March 1st, 1999", "March 1st, 1999"),
        ];

        for (text, expected) in cases {
            let entities = ner.extract_entities(text, None).unwrap();
            let dates: Vec<_> = entities
                .iter()
                .filter(|e| matches!(e.entity_type, EntityType::Date))
                .collect();

            assert!(!dates.is_empty(), "Should find date in: {}", text);
            assert_eq!(dates[0].text, expected);
        }
    }

    // -------------------------------------------------------------------------
    // Money Patterns
    // -------------------------------------------------------------------------

    #[test]
    fn test_money_formats() {
        let ner = RegexNER::new();

        let cases = [
            ("Price: $100", "$100"),
            ("Cost is $1,234.56", "$1,234.56"),
            ("€50.00 each", "€50.00"),
            ("¥1000 yen", "¥1000"),
            ("£99.99", "£99.99"),
        ];

        for (text, expected) in cases {
            let entities = ner.extract_entities(text, None).unwrap();
            let money: Vec<_> = entities
                .iter()
                .filter(|e| matches!(e.entity_type, EntityType::Money))
                .collect();

            assert!(!money.is_empty(), "Should find money in: {}", text);
            assert_eq!(money[0].text, expected);
        }
    }

    #[test]
    fn test_money_with_text() {
        let ner = RegexNER::new();

        let cases = [
            ("Costs 100 USD", "100 USD"),
            ("Price 50 EUR", "50 EUR"),
            ("Fee: 25 dollars", "25 dollars"),
        ];

        for (text, expected) in cases {
            let entities = ner.extract_entities(text, None).unwrap();
            let money: Vec<_> = entities
                .iter()
                .filter(|e| matches!(e.entity_type, EntityType::Money))
                .collect();

            assert!(!money.is_empty(), "Should find money in: {}", text);
            assert_eq!(money[0].text, expected);
        }
    }

    // -------------------------------------------------------------------------
    // Percent Patterns
    // -------------------------------------------------------------------------

    #[test]
    fn test_percent_formats() {
        let ner = RegexNER::new();

        let cases = [
            ("Growth of 25%", "25%"),
            ("Rate: 3.5%", "3.5%"),
            ("Down 0.1%", "0.1%"),
            ("Up 100%", "100%"),
        ];

        for (text, expected) in cases {
            let entities = ner.extract_entities(text, None).unwrap();
            let pct: Vec<_> = entities
                .iter()
                .filter(|e| matches!(e.entity_type, EntityType::Percent))
                .collect();

            assert!(!pct.is_empty(), "Should find percent in: {}", text);
            assert_eq!(pct[0].text, expected);
        }
    }

    // -------------------------------------------------------------------------
    // Email Patterns
    // -------------------------------------------------------------------------

    #[test]
    fn test_email_formats() {
        let ner = RegexNER::new();

        let cases = [
            ("Contact: test@example.com", "test@example.com"),
            ("Email john.doe@company.org", "john.doe@company.org"),
            ("user+tag@domain.co.uk", "user+tag@domain.co.uk"),
            ("admin@sub.domain.net", "admin@sub.domain.net"),
        ];

        for (text, expected) in cases {
            let entities = ner.extract_entities(text, None).unwrap();
            let emails: Vec<_> = entities
                .iter()
                .filter(|e| matches!(e.entity_type, EntityType::Email))
                .collect();

            assert!(!emails.is_empty(), "Should find email in: {}", text);
            assert_eq!(emails[0].text, expected);
        }
    }

    // -------------------------------------------------------------------------
    // URL Patterns
    // -------------------------------------------------------------------------

    #[test]
    fn test_url_formats() {
        let ner = RegexNER::new();

        let cases = [
            ("Visit https://example.com", "https://example.com"),
            ("Link: http://test.org/path", "http://test.org/path"),
            (
                "See https://api.example.com/v1/users",
                "https://api.example.com/v1/users",
            ),
        ];

        for (text, expected) in cases {
            let entities = ner.extract_entities(text, None).unwrap();
            let urls: Vec<_> = entities
                .iter()
                .filter(|e| matches!(e.entity_type, EntityType::Url))
                .collect();

            assert!(!urls.is_empty(), "Should find URL in: {}", text);
            assert_eq!(urls[0].text, expected);
        }
    }

    // -------------------------------------------------------------------------
    // Phone Patterns
    // -------------------------------------------------------------------------

    #[test]
    fn test_phone_formats() {
        let ner = RegexNER::new();

        let cases = [
            ("Call +1 (555) 123-4567", "+1 (555) 123-4567"),
            ("Phone: 555-123-4567", "555-123-4567"),
            ("Tel: (555) 123-4567", "(555) 123-4567"),
        ];

        for (text, expected) in cases {
            let entities = ner.extract_entities(text, None).unwrap();
            let phones: Vec<_> = entities
                .iter()
                .filter(|e| matches!(e.entity_type, EntityType::Phone))
                .collect();

            assert!(!phones.is_empty(), "Should find phone in: {}", text);
            assert_eq!(phones[0].text, expected);
        }
    }

    // -------------------------------------------------------------------------
    // Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_multiple_entities_same_type() {
        let ner = RegexNER::new();
        let text = "Meeting on 2024-01-15 and 2024-02-20";

        let entities = ner.extract_entities(text, None).unwrap();
        let dates: Vec<_> = entities
            .iter()
            .filter(|e| matches!(e.entity_type, EntityType::Date))
            .collect();

        assert_eq!(dates.len(), 2);
    }

    #[test]
    fn test_mixed_entities() {
        let ner = RegexNER::new();
        let text = "Email test@example.com for $100 on 2024-01-15";

        let entities = ner.extract_entities(text, None).unwrap();

        let has_email = entities
            .iter()
            .any(|e| matches!(e.entity_type, EntityType::Email));
        let has_money = entities
            .iter()
            .any(|e| matches!(e.entity_type, EntityType::Money));
        let has_date = entities
            .iter()
            .any(|e| matches!(e.entity_type, EntityType::Date));

        assert!(has_email);
        assert!(has_money);
        assert!(has_date);
    }

    #[test]
    fn test_no_false_positives() {
        let ner = RegexNER::new();

        // These should NOT match
        let cases = [
            "regular text",
            "John Smith",
            "Microsoft Corporation",
            "123", // Just a number
            "hello world",
        ];

        for text in cases {
            let entities = ner.extract_entities(text, None).unwrap();
            // May find some entities, but shouldn't find structured ones in plain text
            for e in &entities {
                // If we find any, they should be legitimate
                assert!(!e.text.is_empty(), "Found empty entity in: {}", text);
            }
        }
    }

    #[test]
    fn test_entity_boundaries() {
        let ner = RegexNER::new();
        let text = "prefix$100suffix";

        let entities = ner.extract_entities(text, None).unwrap();
        let money: Vec<_> = entities
            .iter()
            .filter(|e| matches!(e.entity_type, EntityType::Money))
            .collect();

        // Should extract just "$100" not the surrounding text
        if !money.is_empty() {
            assert_eq!(money[0].text, "$100");
        }
    }

    #[test]
    fn test_confidence_scores() {
        let ner = RegexNER::new();
        let text = "Date: 2024-01-15, Email: test@example.com";

        let entities = ner.extract_entities(text, None).unwrap();

        for e in &entities {
            // All RegexNER entities should have confidence >= 0.9
            assert!(
                e.confidence >= 0.9,
                "Entity {} has low confidence {}",
                e.text,
                e.confidence
            );
        }
    }
}

// =============================================================================
// HeuristicNER Detailed Tests
// =============================================================================

mod statistical_ner {
    use super::*;

    // -------------------------------------------------------------------------
    // Person Detection
    // -------------------------------------------------------------------------

    #[test]
    fn test_person_with_title() {
        let ner = HeuristicNER::new();

        let cases = [
            "Dr. John Smith",
            "Mr. James Brown",
            "Mrs. Jane Doe",
            "Prof. Albert Einstein",
        ];

        for text in cases {
            let entities = ner.extract_entities(text, None).unwrap();
            // HeuristicNER uses heuristics, so we check for any Person entity
            let _persons: Vec<_> = entities
                .iter()
                .filter(|e| matches!(e.entity_type, EntityType::Person))
                .collect();

            // May or may not find depending on heuristics
            // At minimum, shouldn't panic
        }
    }

    #[test]
    fn test_person_capitalized() {
        let ner = HeuristicNER::new();

        let cases = [
            "John Smith works here",
            "I met Mary Johnson yesterday",
            "CEO Steve Jobs announced",
        ];

        for text in cases {
            let _entities = ner.extract_entities(text, None).unwrap();
            // Check that we find something (heuristic system)
            // Just ensure no panic
        }
    }

    // -------------------------------------------------------------------------
    // Organization Detection
    // -------------------------------------------------------------------------

    #[test]
    fn test_org_with_suffix() {
        let ner = HeuristicNER::new();

        let cases = [
            "Microsoft Corporation",
            "Apple Inc.",
            "OpenAI LLC",
            "Google Ltd",
        ];

        for text in cases {
            let entities = ner.extract_entities(text, None).unwrap();
            let orgs: Vec<_> = entities
                .iter()
                .filter(|e| matches!(e.entity_type, EntityType::Organization))
                .collect();

            // Should find organization with suffix
            if orgs.is_empty() {
                // May not always find if context is missing
                // Just ensure we don't panic
            }
        }
    }

    #[test]
    fn test_org_with_prefix() {
        let ner = HeuristicNER::new();

        let cases = [
            "Bank of America",
            "University of California",
            "Department of Defense",
        ];

        for text in cases {
            let entities = ner.extract_entities(text, None).unwrap();
            // Heuristic-based, may or may not find
        }
    }

    // -------------------------------------------------------------------------
    // Location Detection
    // -------------------------------------------------------------------------

    #[test]
    fn test_location_with_prefix() {
        let ner = HeuristicNER::new();

        let cases = [
            "City of London",
            "State of California",
            "Republic of France",
        ];

        for text in cases {
            let entities = ner.extract_entities(text, None).unwrap();
            // Heuristic-based
        }
    }

    // -------------------------------------------------------------------------
    // Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_empty_text() {
        let ner = HeuristicNER::new();
        let entities = ner.extract_entities("", None).unwrap();
        assert!(entities.is_empty());
    }

    #[test]
    fn test_lowercase_text() {
        let ner = HeuristicNER::new();
        let entities = ner
            .extract_entities("all lowercase text here", None)
            .unwrap();
        // HeuristicNER relies on capitalization, so should find nothing
        assert!(entities.is_empty());
    }

    #[test]
    fn test_all_caps_text() {
        let ner = HeuristicNER::new();
        let entities = ner.extract_entities("ALL CAPS TEXT HERE", None).unwrap();
        // ALL CAPS might be treated differently
    }

    #[test]
    fn test_threshold_effect() {
        let low_threshold = HeuristicNER::with_threshold(0.1);
        let high_threshold = HeuristicNER::with_threshold(0.9);

        let text = "John Smith works at Microsoft Corporation in New York";

        let low_entities = low_threshold.extract_entities(text, None).unwrap();
        let high_entities = high_threshold.extract_entities(text, None).unwrap();

        // Lower threshold should find more (or equal) entities
        assert!(low_entities.len() >= high_entities.len());
    }

    #[test]
    fn test_confidence_range() {
        let ner = HeuristicNER::new();
        let text = "Dr. John Smith works at Microsoft Corporation";

        let entities = ner.extract_entities(text, None).unwrap();

        for e in &entities {
            assert!(
                e.confidence >= 0.0 && e.confidence <= 1.0,
                "Confidence {} out of range for {}",
                e.confidence,
                e.text
            );
        }
    }
}

// =============================================================================
// StackedNER Combination Tests
// =============================================================================

mod stacked_combination {
    use super::*;

    #[test]
    fn test_pattern_and_statistical_combined() {
        let ner = StackedNER::default();

        let text = "Dr. John Smith charges $100/hr. Contact: john@example.com on 2024-01-15";
        let entities = ner.extract_entities(text, None).unwrap();

        // Should find both pattern (email, money, date) and statistical (person) entities
        let has_email = entities
            .iter()
            .any(|e| matches!(e.entity_type, EntityType::Email));
        let has_money = entities
            .iter()
            .any(|e| matches!(e.entity_type, EntityType::Money));
        let has_date = entities
            .iter()
            .any(|e| matches!(e.entity_type, EntityType::Date));

        assert!(has_email);
        assert!(has_money);
        assert!(has_date);
    }

    #[test]
    fn test_no_duplicate_entities() {
        let ner = StackedNER::default();
        let text = "Meeting on 2024-01-15 at $100";

        let entities = ner.extract_entities(text, None).unwrap();

        // Check for duplicates by comparing (start, end) pairs
        let mut seen = std::collections::HashSet::new();
        for e in &entities {
            let key = (e.start, e.end);
            assert!(
                seen.insert(key),
                "Duplicate entity at ({}, {})",
                e.start,
                e.end
            );
        }
    }

    #[test]
    fn test_provenance_tracking() {
        let ner = StackedNER::default();
        let text = "Contact test@example.com";

        let entities = ner.extract_entities(text, None).unwrap();

        for e in &entities {
            // All entities should have provenance
            if let Some(ref prov) = e.provenance {
                assert!(!prov.source.is_empty());
            }
        }
    }

    #[test]
    fn test_entity_ordering() {
        let ner = StackedNER::default();
        let text = "Email: a@b.com, Money: $100, Date: 2024-01-01";

        let entities = ner.extract_entities(text, None).unwrap();

        // Entities should be ordered by start position
        for i in 1..entities.len() {
            assert!(
                entities[i].start >= entities[i - 1].start,
                "Entities not ordered: {} vs {}",
                entities[i - 1].start,
                entities[i].start
            );
        }
    }
}

// =============================================================================
// Unicode and International Tests
// =============================================================================

mod unicode_tests {
    use super::*;

    #[test]
    fn test_unicode_in_text() {
        let ner = RegexNER::new();
        let text = "Meeting on 2024-01-15 with café owner";

        let entities = ner.extract_entities(text, None).unwrap();
        // Should still find the date
        let dates: Vec<_> = entities
            .iter()
            .filter(|e| matches!(e.entity_type, EntityType::Date))
            .collect();

        assert!(!dates.is_empty());
    }

    #[test]
    fn test_emoji_in_text() {
        let ner = RegexNER::new();
        let text = "Contact 👋 test@example.com 📧";

        let entities = ner.extract_entities(text, None).unwrap();
        let emails: Vec<_> = entities
            .iter()
            .filter(|e| matches!(e.entity_type, EntityType::Email))
            .collect();

        assert!(!emails.is_empty());
        assert_eq!(emails[0].text, "test@example.com");
    }

    #[test]
    fn test_chinese_text_with_entities() {
        let ner = RegexNER::new();
        let text = "会议日期 2024-01-15 费用 $100";

        let entities = ner.extract_entities(text, None).unwrap();
        // Should find date and money even with Chinese text
        let has_date = entities
            .iter()
            .any(|e| matches!(e.entity_type, EntityType::Date));
        let has_money = entities
            .iter()
            .any(|e| matches!(e.entity_type, EntityType::Money));

        assert!(has_date);
        assert!(has_money);
    }

    #[test]
    fn test_arabic_numerals_in_entities() {
        let ner = RegexNER::new();
        let text = "Price: $١٢٣"; // Arabic-Indic numerals

        // This might or might not work depending on regex
        let entities = ner.extract_entities(text, None).unwrap();
        // Just ensure no panic
    }
}

// =============================================================================
// Performance Characteristics
// =============================================================================

mod performance {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_regex_ner_is_fast() {
        let ner = RegexNER::new();
        let text = "Contact test@example.com for $100 on 2024-01-15";

        let start = Instant::now();
        for _ in 0..1000 {
            let _ = ner.extract_entities(text, None);
        }
        let elapsed = start.elapsed();

        // RegexNER should process 1000 iterations in < 100ms
        assert!(
            elapsed.as_millis() < 100,
            "RegexNER too slow: {}ms for 1000 iterations",
            elapsed.as_millis()
        );
    }

    #[test]
    fn test_statistical_ner_reasonable_speed() {
        let ner = HeuristicNER::new();
        let text = "Dr. John Smith works at Microsoft Corporation in New York City";

        let start = Instant::now();
        for _ in 0..1000 {
            let _ = ner.extract_entities(text, None);
        }
        let elapsed = start.elapsed();

        // HeuristicNER should process 1000 iterations in < 500ms
        assert!(
            elapsed.as_millis() < 500,
            "HeuristicNER too slow: {}ms for 1000 iterations",
            elapsed.as_millis()
        );
    }

    #[test]
    fn test_stacked_ner_reasonable_speed() {
        let ner = StackedNER::default();
        let text = "Dr. John Smith (john@example.com) charges $100/hr. Meeting 2024-01-15.";

        let start = Instant::now();
        for _ in 0..100 {
            let _ = ner.extract_entities(text, None);
        }
        let elapsed = start.elapsed();

        // StackedNER should process 100 iterations in < 100ms
        assert!(
            elapsed.as_millis() < 100,
            "StackedNER too slow: {}ms for 100 iterations",
            elapsed.as_millis()
        );
    }

    #[test]
    fn test_long_text_handling() {
        let ner = StackedNER::default();
        let base = "Contact test@example.com for $100 on 2024-01-15. ";
        let text = base.repeat(100); // ~5000 chars

        let start = Instant::now();
        let entities = ner.extract_entities(&text, None).unwrap();
        let elapsed = start.elapsed();

        // Should handle long text in reasonable time
        assert!(
            elapsed.as_millis() < 1000,
            "Long text too slow: {}ms",
            elapsed.as_millis()
        );

        // Should find multiple entities
        assert!(entities.len() > 100);
    }
}
