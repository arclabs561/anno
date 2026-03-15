use super::*;

#[test]
fn test_basic_person_detection() {
    let ner = HeuristicNER::new();
    let entities = ner
        .extract_entities("Dr. John Smith met with Mary.", None)
        .unwrap();

    let names: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
    assert!(
        names
            .iter()
            .any(|n| n.contains("John") || n.contains("Smith")),
        "Should detect John Smith: {:?}",
        names
    );
}

#[test]
fn test_organization_suffix_detection() {
    let ner = HeuristicNER::new();
    let entities = ner
        .extract_entities("Apple Inc. announced new products.", None)
        .unwrap();

    let orgs: Vec<_> = entities
        .iter()
        .filter(|e| matches!(e.entity_type, EntityType::Organization))
        .collect();
    assert!(!orgs.is_empty(), "Should detect Apple Inc. as organization");
}

#[test]
fn test_location_preposition_context() {
    let ner = HeuristicNER::new();
    let entities = ner
        .extract_entities("She lived in Paris for years.", None)
        .unwrap();

    let locs: Vec<_> = entities
        .iter()
        .filter(|e| matches!(e.entity_type, EntityType::Location))
        .collect();
    assert!(!locs.is_empty(), "Should detect Paris as location");
}

#[test]
fn test_known_organizations() {
    let ner = HeuristicNER::new();
    let entities = ner
        .extract_entities("Google and Microsoft competed.", None)
        .unwrap();

    let texts: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
    assert!(
        texts.iter().any(|t| t.contains("Google")),
        "Should detect Google"
    );
    assert!(
        texts.iter().any(|t| t.contains("Microsoft")),
        "Should detect Microsoft"
    );
}

#[test]
fn test_cjk_organization_detection() {
    let ner = HeuristicNER::new();
    let entities = ner
        .extract_entities("ソニーが新製品を発表しました。", None)
        .unwrap();

    let orgs: Vec<_> = entities
        .iter()
        .filter(|e| matches!(e.entity_type, EntityType::Organization))
        .collect();
    assert!(
        !orgs.is_empty(),
        "Should detect Sony (ソニー) as organization"
    );
}

#[test]
fn test_cjk_location_detection() {
    let ner = HeuristicNER::new();
    let entities = ner
        .extract_entities("東京オリンピックが開催された。", None)
        .unwrap();

    let locs: Vec<_> = entities
        .iter()
        .filter(|e| matches!(e.entity_type, EntityType::Location))
        .collect();
    assert!(!locs.is_empty(), "Should detect Tokyo (東京) as location");
}

#[test]
fn test_empty_text() {
    let ner = HeuristicNER::new();
    let entities = ner.extract_entities("", None).unwrap();
    assert!(entities.is_empty());
}

#[test]
fn test_no_entities() {
    let ner = HeuristicNER::new();
    let entities = ner
        .extract_entities("the quick brown fox jumps over the lazy dog", None)
        .unwrap();
    // All lowercase, no entities expected
    assert!(
        entities.is_empty(),
        "Lowercase text should have no entities"
    );
}

#[test]
fn test_threshold_filtering() {
    let low_threshold = HeuristicNER::with_threshold(0.1);
    let high_threshold = HeuristicNER::with_threshold(0.9);

    let text = "John works at Google.";
    let low_entities = low_threshold.extract_entities(text, None).unwrap();
    let high_entities = high_threshold.extract_entities(text, None).unwrap();

    // Lower threshold should capture more or equal entities
    assert!(low_entities.len() >= high_entities.len());
}

#[test]
fn test_sentence_starter_filtering() {
    let ner = HeuristicNER::new();
    let entities = ner
        .extract_entities("The dog ran. It was fast.", None)
        .unwrap();

    // "The" and "It" should be filtered as common sentence starters
    let texts: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
    assert!(
        !texts.contains(&"The"),
        "Should filter 'The' as sentence starter"
    );
    assert!(!texts.contains(&"It"), "Should filter 'It' as pronoun");
}

#[test]
fn test_person_prefix_detection() {
    let ner = HeuristicNER::new();
    let entities = ner
        .extract_entities("Prof. Einstein presented the theory.", None)
        .unwrap();

    let persons: Vec<_> = entities
        .iter()
        .filter(|e| matches!(e.entity_type, EntityType::Person))
        .collect();
    assert!(
        !persons.is_empty(),
        "Should detect Prof. Einstein as person"
    );
}

#[test]
fn test_multi_word_organization() {
    let ner = HeuristicNER::new();
    let entities = ner
        .extract_entities("Bank of America provides services.", None)
        .unwrap();

    let orgs: Vec<_> = entities
        .iter()
        .filter(|e| matches!(e.entity_type, EntityType::Organization))
        .collect();
    assert!(!orgs.is_empty(), "Should detect 'Bank of America' pattern");
}

#[test]
fn test_location_indicators() {
    let ner = HeuristicNER::new();
    let entities = ner
        .extract_entities("New Zealand is beautiful.", None)
        .unwrap();

    let locs: Vec<_> = entities
        .iter()
        .filter(|e| matches!(e.entity_type, EntityType::Location))
        .collect();
    assert!(!locs.is_empty(), "Should detect 'New Zealand' as location");
}

#[test]
fn test_model_trait_implementation() {
    let ner = HeuristicNER::new();

    assert_eq!(ner.name(), "heuristic");
    assert!(ner.is_available());
    assert!(!ner.supported_types().is_empty());
    assert!(ner.description().contains("Heuristic"));
}

#[test]
fn test_entity_offsets_are_valid() {
    let ner = HeuristicNER::new();
    let text = "Barack Obama visited Berlin yesterday.";
    let entities = ner.extract_entities(text, None).unwrap();

    let char_count = text.chars().count();
    for entity in &entities {
        assert!(entity.start() <= entity.end(), "start should be <= end");
        assert!(entity.end() <= char_count, "end should be within text");

        // Verify text matches span
        let extracted: String = text
            .chars()
            .skip(entity.start())
            .take(entity.end() - entity.start())
            .collect();
        assert_eq!(
            extracted, entity.text,
            "Extracted text should match entity text"
        );
    }
}

#[test]
fn test_unicode_text_handling() {
    let ner = HeuristicNER::new();
    let text = "François Müller from München met José García.";
    let entities = ner.extract_entities(text, None).unwrap();

    // Should handle diacritics correctly
    for entity in &entities {
        let extracted: String = text
            .chars()
            .skip(entity.start())
            .take(entity.end() - entity.start())
            .collect();
        assert_eq!(extracted, entity.text, "Unicode offsets should be correct");
    }
}

#[test]
fn test_provenance_is_set() {
    let ner = HeuristicNER::new();
    let entities = ner
        .extract_entities("Google announced today.", None)
        .unwrap();

    for entity in &entities {
        if let Some(ref prov) = entity.provenance {
            assert_eq!(prov.source, "heuristic");
            assert!(matches!(prov.method, ExtractionMethod::Heuristic));
        }
    }
}

// =========================================================================
// Acronym signal tests (domain-agnostic, language-agnostic)
// =========================================================================

#[test]
fn test_is_acronym_word_latin() {
    assert!(is_acronym_word("PARC"));
    assert!(is_acronym_word("IBM"));
    assert!(is_acronym_word("NASA"));
    assert!(is_acronym_word("N2K"));
    assert!(is_acronym_word("DARPA."));
    assert!(is_acronym_word("(NATO)"));
    assert!(!is_acronym_word("Xerox"));
    assert!(!is_acronym_word("Lynn"));
    assert!(!is_acronym_word("A"));
    assert!(!is_acronym_word("42"));
    assert!(!is_acronym_word(""));
}

#[test]
fn test_is_acronym_word_cyrillic() {
    assert!(is_acronym_word("\u{041D}\u{0410}\u{0422}\u{041E}")); // НАТО
    assert!(is_acronym_word("\u{041C}\u{0418}\u{0414}")); // МИД
    assert!(!is_acronym_word(
        "\u{041C}\u{043E}\u{0441}\u{043A}\u{0432}\u{0430}"
    )); // Москва
}

#[test]
fn test_is_acronym_word_caseless_scripts() {
    assert!(!is_acronym_word("\u{6771}\u{4EAC}")); // 東京 (CJK)
    assert!(!is_acronym_word("\u{30BD}\u{30CB}\u{30FC}")); // ソニー (Katakana)
    assert!(!is_acronym_word("\u{062D}\u{0645}\u{0627}\u{0633}")); // حماس (Arabic)
}

#[test]
fn test_acronym_in_multi_word_span_signals_org() {
    let ner = HeuristicNER::new();
    let entities = ner
        .extract_entities(
            "Lynn Conway worked at IBM and Xerox PARC in California.",
            None,
        )
        .unwrap();
    let xerox_parc = entities.iter().find(|e| e.text == "Xerox PARC");
    assert!(
        xerox_parc.is_some(),
        "Should detect 'Xerox PARC': {entities:?}"
    );
    assert!(
        matches!(xerox_parc.unwrap().entity_type, EntityType::Organization),
        "Xerox PARC should be ORG, got {:?}",
        xerox_parc.unwrap().entity_type,
    );
}

#[test]
fn test_acronym_no_regression_on_normal_names() {
    let ner = HeuristicNER::new();
    let entities = ner
        .extract_entities("Lynn Conway designed the processor.", None)
        .unwrap();
    let lynn = entities.iter().find(|e| e.text == "Lynn Conway");
    assert!(lynn.is_some(), "Should detect 'Lynn Conway': {entities:?}");
    assert!(
        matches!(lynn.unwrap().entity_type, EntityType::Person),
        "Lynn Conway should remain PER, got {:?}",
        lynn.unwrap().entity_type,
    );
}

#[test]
fn test_single_acronym_signals_org() {
    let ner = HeuristicNER::new();
    let entities = ner
        .extract_entities("She joined DARPA last year.", None)
        .unwrap();
    let darpa = entities.iter().find(|e| e.text == "DARPA");
    assert!(darpa.is_some(), "Should detect 'DARPA': {entities:?}");
    assert!(
        matches!(darpa.unwrap().entity_type, EntityType::Organization),
        "DARPA should be ORG, got {:?}",
        darpa.unwrap().entity_type,
    );
}

#[test]
fn test_known_loc_acronym_still_loc() {
    let ner = HeuristicNER::new();
    let entities = ner
        .extract_entities("She moved to USA last year.", None)
        .unwrap();
    let usa = entities.iter().find(|e| e.text == "USA");
    assert!(usa.is_some(), "Should detect 'USA': {entities:?}");
    assert!(
        matches!(usa.unwrap().entity_type, EntityType::Location),
        "USA should be LOC (gazetteer wins), got {:?}",
        usa.unwrap().entity_type,
    );
}

// =========================================================================
// classify_minimal rule-path tests
// =========================================================================

/// Rule 1: International org suffixes (GmbH, AG, S.A., etc.)
#[test]
fn test_international_org_suffix_gmbh() {
    let ner = HeuristicNER::new();
    let entities = ner
        .extract_entities("Siemens GmbH reported earnings.", None)
        .unwrap();

    let orgs: Vec<_> = entities
        .iter()
        .filter(|e| matches!(e.entity_type, EntityType::Organization))
        .collect();
    assert!(!orgs.is_empty(), "Should detect 'Siemens GmbH' as ORG");
    assert!(
        orgs.iter().any(|e| e.text.contains("GmbH")),
        "Entity text should include GmbH suffix: {orgs:?}"
    );
}

/// classify_minimal skip_word: job titles (CEO, VP) are filtered out.
#[test]
fn test_skip_word_filters_job_titles() {
    let ner = HeuristicNER::with_threshold(0.0);
    let entities = ner
        .extract_entities("the CEO spoke at the event.", None)
        .unwrap();

    let texts: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
    assert!(
        !texts.iter().any(|t| t.eq_ignore_ascii_case("CEO")),
        "CEO should be filtered as skip_word: {texts:?}"
    );
}

/// classify_minimal skip_pronoun: single pronouns at sentence start are
/// filtered even when capitalized.
#[test]
fn test_skip_pronoun_filters_single_pronouns() {
    let ner = HeuristicNER::with_threshold(0.0);
    let entities = ner
        .extract_entities("He ran. She swam. They left.", None)
        .unwrap();

    let texts: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
    for pronoun in &["He", "She", "They"] {
        assert!(
            !texts.contains(pronoun),
            "{pronoun} should be filtered: {texts:?}"
        );
    }
}

/// classify_minimal single_letter: a lone uppercase letter is not an entity.
#[test]
fn test_single_letter_not_entity() {
    let ner = HeuristicNER::with_threshold(0.0);
    let entities = ner
        .extract_entities("variable X was defined.", None)
        .unwrap();

    let texts: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
    assert!(
        !texts.contains(&"X"),
        "Single letter 'X' should be skipped: {texts:?}"
    );
}

/// classify_minimal long_span_org: three+ capitalized words default to ORG.
#[test]
fn test_three_word_span_defaults_to_org() {
    let ner = HeuristicNER::new();
    let entities = ner
        .extract_entities("Global Dynamics Research announced funding.", None)
        .unwrap();

    let span = entities
        .iter()
        .find(|e| e.text == "Global Dynamics Research");
    assert!(
        span.is_some(),
        "Should detect 'Global Dynamics Research': {entities:?}"
    );
    assert!(
        matches!(span.unwrap().entity_type, EntityType::Organization),
        "Three-word span should be ORG, got {:?}",
        span.unwrap().entity_type,
    );
}

/// classify_minimal capitalized default: single capitalized word mid-sentence
/// with no other signal defaults to Person.
#[test]
fn test_single_capitalized_mid_sentence_defaults_person() {
    let ner = HeuristicNER::new();
    let entities = ner
        .extract_entities("I spoke with Valentina about the plan.", None)
        .unwrap();

    let val = entities.iter().find(|e| e.text == "Valentina");
    assert!(val.is_some(), "Should detect 'Valentina': {entities:?}");
    assert!(
        matches!(val.unwrap().entity_type, EntityType::Person),
        "Single capitalized mid-sentence should be PER, got {:?}",
        val.unwrap().entity_type,
    );
}

/// "and" should separate entities, not merge them.
#[test]
fn test_and_separates_entities() {
    let ner = HeuristicNER::new();
    let entities = ner
        .extract_entities("We met Alice and Bob at the event.", None)
        .unwrap();

    let texts: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
    // "Alice and Bob" should NOT be one entity
    assert!(
        !texts.iter().any(|t| t.contains("and")),
        "'and' should separate entities, not join them: {texts:?}"
    );
    assert!(texts.contains(&"Alice"), "Should detect Alice: {texts:?}");
    assert!(texts.contains(&"Bob"), "Should detect Bob: {texts:?}");
}

/// German preposition "aus" triggers location context.
#[test]
fn test_german_preposition_location_context() {
    let ner = HeuristicNER::new();
    let entities = ner
        .extract_entities("Er kommt aus Hamburg zum Meeting.", None)
        .unwrap();

    let locs: Vec<_> = entities
        .iter()
        .filter(|e| matches!(e.entity_type, EntityType::Location))
        .collect();
    assert!(
        locs.iter().any(|e| e.text == "Hamburg"),
        "German preposition 'aus' should signal LOC for Hamburg: {entities:?}"
    );
}

/// Trailing punctuation is stripped from entity text.
#[test]
fn test_trailing_punctuation_stripped() {
    let ner = HeuristicNER::new();
    let entities = ner
        .extract_entities("She met Google, Microsoft, and Tesla.", None)
        .unwrap();

    for entity in &entities {
        assert!(
            !entity.text.ends_with(','),
            "Entity '{}' should not end with comma",
            entity.text
        );
        assert!(
            !entity.text.ends_with('.'),
            "Entity '{}' should not end with period",
            entity.text
        );
    }
}

// =========================================================================
// Fix 2: Span computation -- offsets use original text positions
// =========================================================================

/// Span offsets must match original text, even with multi-space gaps.
#[test]
fn test_span_offsets_with_multiple_spaces() {
    let ner = HeuristicNER::new();
    // Two spaces between words -- joined text would have single space
    let text = "Meeting with  Barack  Obama in Washington.";
    let entities = ner.extract_entities(text, None).unwrap();

    for entity in &entities {
        let extracted: String = text
            .chars()
            .skip(entity.start())
            .take(entity.end() - entity.start())
            .collect();
        // The entity text is joined with single spaces, so it won't match the original
        // multi-space text. But the span boundaries must still be valid character offsets.
        assert!(entity.start() < entity.end(), "start < end");
        assert!(
            entity.end() <= text.chars().count(),
            "end ({}) within text len ({})",
            entity.end(),
            text.chars().count()
        );
        // The extracted span must start and end with the entity's first/last word
        let first_word = entity.text.split_whitespace().next().unwrap();
        let last_word = entity.text.split_whitespace().last().unwrap();
        assert!(
            extracted.starts_with(first_word),
            "Span '{}' should start with '{}' (entity: '{}')",
            extracted,
            first_word,
            entity.text
        );
        assert!(
            extracted.ends_with(last_word),
            "Span '{}' should end with '{}' (entity: '{}')",
            extracted,
            last_word,
            entity.text
        );
    }
}

/// Long multi-word names should not be truncated.
#[test]
fn test_long_names_not_truncated() {
    let ner = HeuristicNER::new();
    let text = "Dr. Emmanuelle Charpentier won the prize.";
    let entities = ner.extract_entities(text, None).unwrap();

    let charpentier = entities.iter().find(|e| e.text.contains("Charpentier"));
    assert!(
        charpentier.is_some(),
        "Should find Charpentier: {:?}",
        entities
    );
    assert!(
        charpentier.unwrap().text.contains("Charpentier"),
        "Name should not be truncated: '{}'",
        charpentier.unwrap().text
    );
}

/// Unicode names should have correct char offsets (not byte offsets).
#[test]
fn test_unicode_name_offsets_correct() {
    let ner = HeuristicNER::new();
    let text = "François Müller presented the results.";
    let entities = ner.extract_entities(text, None).unwrap();

    for entity in &entities {
        let extracted: String = text
            .chars()
            .skip(entity.start())
            .take(entity.end() - entity.start())
            .collect();
        assert_eq!(
            extracted, entity.text,
            "Unicode char offsets must match entity text"
        );
    }
}

/// Leading punctuation trimming uses char count, not byte count.
#[test]
fn test_leading_punct_char_count_not_bytes() {
    let ner = HeuristicNER::new();
    // Opening quote followed by a name
    let text = "She said, \"Alice was there.\"";
    let entities = ner.extract_entities(text, None).unwrap();

    for entity in &entities {
        assert!(
            !entity.text.starts_with('"'),
            "Entity '{}' should not start with quote",
            entity.text
        );
        let extracted: String = text
            .chars()
            .skip(entity.start())
            .take(entity.end() - entity.start())
            .collect();
        assert_eq!(
            extracted, entity.text,
            "Offsets should match after leading punct trim"
        );
    }
}

// =========================================================================
// Fix 3: Day and month names are not entities
// =========================================================================

/// Day names should not be classified as entities.
#[test]
fn test_day_names_not_entities() {
    let ner = HeuristicNER::new();
    let days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ];
    for day in &days {
        let text = format!("{} was a busy day at the office.", day);
        let entities = ner.extract_entities(&text, None).unwrap();
        let texts: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
        assert!(
            !texts.contains(day),
            "'{}' should not be extracted as entity in: '{}' (got: {:?})",
            day,
            text,
            texts
        );
    }
}

/// Month names should not be classified as entities.
#[test]
fn test_month_names_not_entities() {
    let ner = HeuristicNER::new();
    let months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    for month in &months {
        let text = format!("{} earnings exceeded expectations.", month);
        let entities = ner.extract_entities(&text, None).unwrap();
        let texts: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
        assert!(
            !texts.contains(month),
            "'{}' should not be extracted as entity (got: {:?})",
            month,
            texts
        );
    }
}

/// Month names mid-sentence after a location preposition should NOT become LOC.
#[test]
fn test_month_after_preposition_not_loc() {
    let ner = HeuristicNER::new();
    let text = "Sales peaked in March and declined in December.";
    let entities = ner.extract_entities(text, None).unwrap();
    let texts: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
    assert!(
        !texts.iter().any(|t| *t == "March" || *t == "December"),
        "Month names should not be LOC even after 'in': {:?}",
        texts
    );
}

// =========================================================================
// Fix 4: Common acronyms are not entities
// =========================================================================

/// Common tech/science acronyms should not be classified as ORG.
#[test]
fn test_common_acronyms_not_entities() {
    let ner = HeuristicNER::new();
    let acronyms = [
        "LCD", "LED", "USB", "DNA", "RNA", "CPU", "GPU", "HTML", "PDF",
    ];
    for acr in &acronyms {
        let text = format!("The {} technology was revolutionary.", acr);
        let entities = ner.extract_entities(&text, None).unwrap();
        let texts: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
        assert!(
            !texts.contains(acr),
            "'{}' should be filtered as common acronym, got: {:?}",
            acr,
            texts
        );
    }
}

/// Currency code acronyms should not be classified as entities.
#[test]
fn test_currency_codes_not_entities() {
    let ner = HeuristicNER::new();
    let codes = ["EUR", "GBP", "USD", "JPY", "CHF"];
    for code in &codes {
        let text = format!("The {} exchange rate dropped.", code);
        let entities = ner.extract_entities(&text, None).unwrap();
        let texts: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
        assert!(
            !texts.contains(code),
            "'{}' should be filtered as currency code acronym, got: {:?}",
            code,
            texts
        );
    }
}

/// Real entity acronyms (not in common list) should still be detected.
#[test]
fn test_real_acronyms_still_detected() {
    let ner = HeuristicNER::new();
    let entities = ner
        .extract_entities("She joined DARPA and later CERN.", None)
        .unwrap();
    let texts: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
    assert!(texts.contains(&"DARPA"), "DARPA should still be detected");
    assert!(texts.contains(&"CERN"), "CERN should still be detected");
}

/// Hyphenated compounds with common acronym prefix should not be entities.
#[test]
fn test_hyphenated_acronym_compounds_not_entities() {
    let ner = HeuristicNER::new();
    let compounds = [
        "DNA-based",
        "LCD-equipped",
        "USB-powered",
        "GPU-accelerated",
    ];
    for compound in &compounds {
        let text = format!("The {} system performed well.", compound);
        let entities = ner.extract_entities(&text, None).unwrap();
        let texts: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
        assert!(
            !texts.contains(compound),
            "'{}' should be filtered (acronym prefix), got: {:?}",
            compound,
            texts
        );
    }
}

/// Common acronyms in two-word spans should not trigger ORG via acronym_in_span rule.
#[test]
fn test_common_acronym_in_two_word_span_no_acronym_signal() {
    let ner = HeuristicNER::new();
    // "Advanced USB" -- USB is a common acronym. Without the acronym filter,
    // Rule 5.5 would fire and classify as ORG via "acronym_in_span".
    // With the filter, it falls through to Rule 7 (two_word_name -> PER).
    let entities = ner
        .extract_entities("She bought an Advanced USB yesterday.", None)
        .unwrap();
    let usb_span = entities.iter().find(|e| e.text.contains("USB"));
    if let Some(span) = usb_span {
        let prov = span.provenance.as_ref().and_then(|p| p.pattern.as_ref());
        assert!(
            prov.is_none_or(|p| p.as_ref() != "acronym_in_span"),
            "Common acronym USB should not trigger acronym_in_span rule: {:?}",
            span
        );
    }
}

// =========================================================================
// Regression: entity offset validity across all inputs
// =========================================================================

/// All extracted entity offsets must be valid character spans in the original text.
#[test]
fn test_offset_validity_comprehensive() {
    let ner = HeuristicNER::new();
    let texts = [
        "Barack Obama visited Berlin yesterday.",
        "Dr. Emmanuelle Charpentier and Dr. Jennifer Doudna won the Nobel Prize.",
        "Nintendo reported EUR 1.2 million in revenue on Thursday.",
        "The LCD screens use LED backlighting with USB-C connectors.",
        "François Müller from München met José García in São Paulo.",
        "Google, Microsoft, and Tesla announced partnerships.",
        "She said, \"Alice was there.\"",
        "Bank of America reported (Q3) earnings for Apple Inc.",
    ];

    for text in &texts {
        let entities = ner.extract_entities(text, None).unwrap();
        let char_count = text.chars().count();
        for entity in &entities {
            assert!(
                entity.start() < entity.end(),
                "start ({}) < end ({}) for '{}' in '{}'",
                entity.start(),
                entity.end(),
                entity.text,
                text
            );
            assert!(
                entity.end() <= char_count,
                "end ({}) <= text len ({}) for '{}' in '{}'",
                entity.end(),
                char_count,
                entity.text,
                text
            );
        }
    }
}

// =============================================================================
// Title-prefixed name classification (Fix: "CEO X Y" -> PER, not ORG)
// =============================================================================

/// Job title followed by a name should classify as PER, not ORG.
#[test]
fn test_title_prefixed_name_is_person() {
    let ner = HeuristicNER::new();
    let cases = [
        (
            "CEO Shuntaro Furukawa announced the partnership.",
            "CEO Shuntaro Furukawa",
        ),
        (
            "President Barack Obama signed the bill.",
            "President Barack Obama",
        ),
        ("Chairman Li Wei addressed shareholders.", "Chairman Li Wei"),
    ];
    for (text, expected_fragment) in &cases {
        let entities = ner.extract_entities(text, None).unwrap();
        let match_entity = entities
            .iter()
            .find(|e| e.text.contains(expected_fragment) || expected_fragment.contains(&*e.text));
        assert!(
            match_entity.is_some(),
            "Should detect '{}' in '{}', got: {:?}",
            expected_fragment,
            text,
            entities.iter().map(|e| &e.text).collect::<Vec<_>>()
        );
        if let Some(entity) = match_entity {
            assert!(
                matches!(entity.entity_type, EntityType::Person),
                "'{}' should be PER, got {:?}",
                entity.text,
                entity.entity_type
            );
        }
    }
}

/// "Bank of America" (X of Y) should still be ORG, not affected by title rule.
#[test]
fn test_of_pattern_still_org() {
    let ner = HeuristicNER::new();
    let entities = ner
        .extract_entities("Bank of America reported earnings.", None)
        .unwrap();
    let boa = entities.iter().find(|e| e.text.contains("Bank of America"));
    assert!(boa.is_some(), "Should detect Bank of America");
    assert!(
        matches!(boa.unwrap().entity_type, EntityType::Organization),
        "Bank of America should be ORG"
    );
}

// =============================================================================
// Standalone person-prefix skip (Fix: "Dr" alone doesn't create duplicate entity)
// =============================================================================

/// Standalone "Dr" should be skipped when "Dr. X" is also extracted.
#[test]
fn test_standalone_prefix_skipped() {
    let ner = HeuristicNER::new();
    let entities = ner
        .extract_entities("Dr. Jennifer Doudna won the Nobel Prize.", None)
        .unwrap();
    let texts: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
    // Should have "Dr. Jennifer Doudna" (or similar), but NOT a standalone "Dr"
    assert!(
        !texts.iter().any(|t| *t == "Dr" || *t == "Dr."),
        "Standalone 'Dr' should be skipped, got: {:?}",
        texts
    );
    // The full name should be present
    assert!(
        texts
            .iter()
            .any(|t| t.contains("Jennifer") || t.contains("Doudna")),
        "Should detect the full name, got: {:?}",
        texts
    );
}

/// Person prefixes as standalone words should not become entities.
#[test]
fn test_standalone_person_prefixes_skipped() {
    let ner = HeuristicNER::new();
    let prefixes = ["Dr", "Mr", "Mrs", "Prof"];
    for prefix in &prefixes {
        let text = format!("{} went home.", prefix);
        let entities = ner.extract_entities(&text, None).unwrap();
        let has_prefix_entity = entities
            .iter()
            .any(|e| e.text.trim_end_matches('.') == *prefix);
        assert!(
            !has_prefix_entity,
            "Standalone '{}' should be skipped, got: {:?}",
            prefix,
            entities.iter().map(|e| &e.text).collect::<Vec<_>>()
        );
    }
}

#[test]
fn fiscal_quarter_not_tagged_as_entity() {
    let ner = HeuristicNER::new();
    for q in &["Q1", "Q2", "Q3", "Q4"] {
        let text = format!("{} revenue increased by 10%.", q);
        let entities = ner.extract_entities(&text, None).unwrap();
        let has_q = entities.iter().any(|e| e.text == *q);
        assert!(
            !has_q,
            "'{}' should not be tagged as an entity, got: {:?}",
            q,
            entities.iter().map(|e| &e.text).collect::<Vec<_>>()
        );
    }
}

/// N3: Multi-word fiscal quarter patterns like "Q3 FY2025" should not be entities.
#[test]
fn fiscal_quarter_multi_word_not_entity() {
    let ner = super::HeuristicNER::new();
    for pattern in &["Q3 FY2025", "Q1 2024", "Q4 FY2023", "Q2 H1"] {
        let text = format!("The company reported {} earnings grew.", pattern);
        let entities = ner.extract_entities(&text, None).unwrap();
        let has_fiscal = entities
            .iter()
            .any(|e| e.text.starts_with('Q') && e.text.contains(pattern));
        assert!(
            !has_fiscal,
            "'{}' should not be tagged as an entity in: {:?}",
            pattern,
            entities.iter().map(|e| &e.text).collect::<Vec<_>>()
        );
    }
}

/// N4: Common economic acronyms like GDP should not be tagged as ORG.
#[test]
fn economic_acronyms_not_entities() {
    let ner = super::HeuristicNER::new();
    for acronym in &["GDP", "GNP", "CPI", "ROI", "EBITDA", "IPO", "ETF"] {
        let text = format!("The {} grew by 3% this quarter.", acronym);
        let entities = ner.extract_entities(&text, None).unwrap();
        let has_acronym = entities.iter().any(|e| e.text == *acronym);
        assert!(
            !has_acronym,
            "'{}' should not be tagged as an entity, got: {:?}",
            acronym,
            entities.iter().map(|e| &e.text).collect::<Vec<_>>()
        );
    }
}

/// N5: Organization suffixes like "Services", "Technologies" should trigger ORG detection.
#[test]
fn org_suffix_services_technologies() {
    let ner = super::HeuristicNER::new();
    for name in &[
        "Amazon Web Services",
        "Palantir Technologies",
        "General Dynamics Systems",
    ] {
        let text = format!("{} announced a new product.", name);
        let entities = ner.extract_entities(&text, None).unwrap();
        let found = entities
            .iter()
            .any(|e| e.text.contains(name.split_whitespace().last().unwrap()));
        assert!(
            found,
            "Should detect org suffix in '{}', got: {:?}",
            name,
            entities
                .iter()
                .map(|e| (&e.text, &e.entity_type))
                .collect::<Vec<_>>()
        );
    }
}
