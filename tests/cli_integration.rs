//! Integration tests for the anno CLI and grounded module.
//!
//! Tests the full pipeline from text input to entity extraction,
//! validation, and evaluation.

use anno::grounded::{
    render_document_html, render_eval_html, EvalComparison, EvalMatch, GroundedDocument, Location,
    Modality, Quantifier, Signal, SignalValidationError,
};
use anno::{EntityType, HeuristicNER, Model, RegexNER, StackedNER};

// =============================================================================
// Model Coverage Tests
// =============================================================================

#[test]
fn test_pattern_model_dates() {
    let model = RegexNER::new();
    let text = "Meeting on January 15, 2024 at 3pm.";
    let entities = model.extract_entities(text, None).unwrap();

    assert!(entities.iter().any(|e| e.entity_type == EntityType::Date));
    assert!(entities.iter().any(|e| e.entity_type == EntityType::Time));
}

#[test]
fn test_pattern_model_money() {
    let model = RegexNER::new();

    // Various money formats
    let cases = [
        ("$50", true),
        ("$50,000", true),
        ("$1.5 million", true),
        ("€100", true),
        ("£50.99", true),
        ("100 dollars", true),
        ("50 USD", true),
    ];

    for (text, should_find) in cases {
        let entities = model.extract_entities(text, None).unwrap();
        let found = entities.iter().any(|e| e.entity_type == EntityType::Money);
        assert_eq!(found, should_find, "Failed for: {}", text);
    }
}

#[test]
fn test_pattern_model_contact() {
    let model = RegexNER::new();
    let text = "Contact john.doe@example.com or call 555-123-4567 or visit https://example.com";
    let entities = model.extract_entities(text, None).unwrap();

    assert!(
        entities.iter().any(|e| e.entity_type == EntityType::Email),
        "Should find email"
    );
    assert!(
        entities.iter().any(|e| e.entity_type == EntityType::Phone),
        "Should find phone"
    );
    assert!(
        entities.iter().any(|e| e.entity_type == EntityType::Url),
        "Should find URL"
    );
}

#[test]
fn test_statistical_model_persons() {
    let model = HeuristicNER::new();

    // Common names should be detected - these names are in COMMON_FIRST_NAMES
    let cases = [
        ("John Smith is here.", true),
        ("Jane Smith spoke today.", true), // Jane is in dictionary
        ("Barack Obama met world leaders.", true), // Barack is in dictionary
        ("Angela Merkel arrived yesterday.", true), // Angela is in dictionary
    ];

    for (text, should_find) in cases {
        let entities = model.extract_entities(text, None).unwrap();
        let found = entities.iter().any(|e| e.entity_type == EntityType::Person);
        assert!(
            found == should_find,
            "Failed for '{}': expected person={}, got {:?}",
            text,
            should_find,
            entities
        );
    }
}

#[test]
fn test_statistical_model_locations() {
    let model = HeuristicNER::new();
    let text = "Meeting in Berlin, Germany.";
    let entities = model.extract_entities(text, None).unwrap();

    let locations: Vec<_> = entities
        .iter()
        .filter(|e| e.entity_type == EntityType::Location)
        .collect();

    assert!(!locations.is_empty(), "Should find at least one location");
}

#[test]
fn test_stacked_model_combines() {
    let model = StackedNER::default();
    let text = "John Smith paid $100 on January 15th.";
    let entities = model.extract_entities(text, None).unwrap();

    // Should have entities from both pattern and statistical
    assert!(
        entities.iter().any(|e| e.entity_type == EntityType::Person),
        "Should find person"
    );
    assert!(
        entities.iter().any(|e| e.entity_type == EntityType::Money),
        "Should find money"
    );
    assert!(
        entities.iter().any(|e| e.entity_type == EntityType::Date),
        "Should find date"
    );
}

// =============================================================================
// Grounded Document Tests
// =============================================================================

#[test]
fn test_grounded_document_validation() {
    let text = "Marie Curie was a physicist.";
    let mut doc = GroundedDocument::new("test", text);

    // Valid signal
    let signal = Signal::new(0, Location::text(0, 11), "Marie Curie", "PER", 0.95);
    doc.add_signal(signal);

    let errors = doc.validate();
    assert!(
        errors.is_empty(),
        "Valid signal should have no errors: {:?}",
        errors
    );
}

#[test]
fn test_grounded_document_validation_catches_mismatch() {
    let text = "Marie Curie was a physicist.";
    let mut doc = GroundedDocument::new("test", text);

    // Invalid signal - text doesn't match offset
    let signal = Signal::new(
        0,
        Location::text(0, 5),
        "WRONG TEXT", // Doesn't match text[0:5]
        "PER",
        0.95,
    );
    doc.add_signal(signal);

    let errors = doc.validate();
    assert!(!errors.is_empty(), "Should catch text mismatch");
    assert!(matches!(
        errors[0],
        SignalValidationError::TextMismatch { .. }
    ));
}

#[test]
fn test_grounded_document_safe_construction() {
    let text = "Marie Curie won the Nobel Prize.";
    let mut doc = GroundedDocument::new("test", text);

    // Use safe construction
    let id = doc.add_signal_from_text("Marie Curie", "PER", 0.95);
    assert!(id.is_some(), "Should find 'Marie Curie' in text");

    // Verify offsets are correct
    let signal = doc.signals().first().unwrap();
    let (start, end) = signal.text_offsets().unwrap();
    assert_eq!(start, 0);
    assert_eq!(end, 11);
    assert_eq!(signal.surface(), "Marie Curie");

    // Validation should pass
    assert!(doc.is_valid());
}

#[test]
fn test_grounded_document_nth_occurrence() {
    let text = "John met John at John's house.";
    let mut doc = GroundedDocument::new("test", text);

    // Add each occurrence
    let id0 = doc.add_signal_from_text_nth("John", "PER", 0.9, 0);
    let id1 = doc.add_signal_from_text_nth("John", "PER", 0.9, 1);
    let id2 = doc.add_signal_from_text_nth("John", "PER", 0.9, 2);

    assert!(id0.is_some());
    assert!(id1.is_some());
    assert!(id2.is_some());

    // Verify distinct offsets
    let signals: Vec<_> = doc.signals().iter().collect();
    assert_eq!(signals.len(), 3);

    let offsets: Vec<_> = signals.iter().map(|s| s.text_offsets().unwrap()).collect();

    // Each should have different start position
    assert!(offsets[0].0 != offsets[1].0);
    assert!(offsets[1].0 != offsets[2].0);
}

#[test]
fn test_grounded_signal_negation() {
    let _text = "He is not a doctor.";

    let signal = Signal::new(0, Location::text(14, 20), "doctor", "ROLE", 0.8).negated();

    assert!(signal.negated, "Signal should be marked as negated");
}

#[test]
fn test_grounded_signal_quantifier() {
    let _text = "Every employee should attend.";

    let signal = Signal::new(0, Location::text(6, 14), "employee", "ROLE", 0.8)
        .with_quantifier(Quantifier::Universal);

    assert_eq!(signal.quantifier, Some(Quantifier::Universal));
}

#[test]
fn test_grounded_signal_modality() {
    // Text signal should be symbolic
    let text_signal = Signal::new(0, Location::text(0, 10), "some text", "ENT", 0.9)
        .with_modality(Modality::Symbolic);

    assert_eq!(text_signal.modality, Modality::Symbolic);

    // Bounding box signal should be iconic
    let bbox_signal: Signal<Location> =
        Signal::new(1, Location::bbox(0.1, 0.1, 0.2, 0.2), "face", "FACE", 0.9)
            .with_modality(Modality::Iconic);

    assert_eq!(bbox_signal.modality, Modality::Iconic);
}

// =============================================================================
// Spatial Index Tests
// =============================================================================

#[test]
fn test_spatial_index_query() {
    let text = "John Smith met Mary Jones in New York City.";
    let mut doc = GroundedDocument::new("test", text);

    doc.add_signal_from_text("John Smith", "PER", 0.9);
    doc.add_signal_from_text("Mary Jones", "PER", 0.9);
    doc.add_signal_from_text("New York City", "LOC", 0.9);

    // Build index
    let _index = doc.build_text_index();

    // Query range that should contain "Mary Jones" (starts at position 15)
    let results = doc.signals_in_range(10, 30);

    assert!(
        results.iter().any(|s| s.surface() == "Mary Jones"),
        "Should find Mary Jones in range"
    );
}

// =============================================================================
// Eval Comparison Tests
// =============================================================================

#[test]
fn test_eval_comparison_exact_match() {
    let text = "Marie Curie was a physicist.";

    let gold = vec![Signal::new(
        0,
        Location::text(0, 11),
        "Marie Curie",
        "PER",
        1.0,
    )];
    let pred = vec![Signal::new(
        0,
        Location::text(0, 11),
        "Marie Curie",
        "PER",
        0.95,
    )];

    let cmp = EvalComparison::compare(text, gold, pred);

    assert_eq!(cmp.correct_count(), 1);
    assert_eq!(cmp.error_count(), 0);
    assert!((cmp.f1() - 1.0).abs() < 0.001);
}

#[test]
fn test_eval_comparison_type_mismatch() {
    let text = "Apple Inc. is a company.";

    let gold = vec![Signal::new(
        0,
        Location::text(0, 10),
        "Apple Inc.",
        "ORG",
        1.0,
    )];
    let pred = vec![
        Signal::new(0, Location::text(0, 10), "Apple Inc.", "PER", 0.9), // Wrong type
    ];

    let cmp = EvalComparison::compare(text, gold, pred);

    assert_eq!(cmp.correct_count(), 0);
    assert!(cmp
        .matches
        .iter()
        .any(|m| matches!(m, EvalMatch::TypeMismatch { .. })));
}

#[test]
fn test_eval_comparison_boundary_error() {
    let text = "New York City is large.";

    let gold = vec![Signal::new(
        0,
        Location::text(0, 13),
        "New York City",
        "LOC",
        1.0,
    )];
    let pred = vec![
        Signal::new(0, Location::text(0, 8), "New York", "LOC", 0.9), // Missing "City"
    ];

    let cmp = EvalComparison::compare(text, gold, pred);

    assert_eq!(cmp.correct_count(), 0);
    assert!(cmp
        .matches
        .iter()
        .any(|m| matches!(m, EvalMatch::BoundaryError { .. })));
}

#[test]
fn test_eval_comparison_false_positive() {
    let text = "The weather is nice.";

    let gold = vec![]; // No entities
    let pred = vec![
        Signal::new(0, Location::text(4, 11), "weather", "MISC", 0.6), // Spurious
    ];

    let cmp = EvalComparison::compare(text, gold, pred);

    assert_eq!(cmp.correct_count(), 0);
    assert!(cmp
        .matches
        .iter()
        .any(|m| matches!(m, EvalMatch::Spurious { .. })));
}

#[test]
fn test_eval_comparison_false_negative() {
    let text = "Marie Curie was brilliant.";

    let gold = vec![Signal::new(
        0,
        Location::text(0, 11),
        "Marie Curie",
        "PER",
        1.0,
    )];
    let pred = vec![]; // Missed it

    let cmp = EvalComparison::compare(text, gold, pred);

    assert_eq!(cmp.correct_count(), 0);
    assert!(cmp
        .matches
        .iter()
        .any(|m| matches!(m, EvalMatch::Missed { .. })));
}

#[test]
fn test_eval_metrics_calculation() {
    let text = "John met Mary in Paris.";

    // Gold: John (PER), Mary (PER), Paris (LOC)
    let gold = vec![
        Signal::new(0, Location::text(0, 4), "John", "PER", 1.0),
        Signal::new(1, Location::text(9, 13), "Mary", "PER", 1.0),
        Signal::new(2, Location::text(17, 22), "Paris", "LOC", 1.0),
    ];

    // Pred: John (PER), Paris (LOC) - missed Mary
    let pred = vec![
        Signal::new(0, Location::text(0, 4), "John", "PER", 0.9),
        Signal::new(1, Location::text(17, 22), "Paris", "LOC", 0.8),
    ];

    let cmp = EvalComparison::compare(text, gold, pred);

    // 2 correct out of 2 predicted = 100% precision
    // 2 correct out of 3 gold = 66.67% recall
    assert_eq!(cmp.correct_count(), 2);
    assert!((cmp.precision() - 1.0).abs() < 0.001);
    assert!((cmp.recall() - 0.6667).abs() < 0.01);
}

// =============================================================================
// HTML Rendering Tests
// =============================================================================

#[test]
fn test_html_rendering_contains_entities() {
    let text = "John Smith works at Apple.";
    let mut doc = GroundedDocument::new("test", text);

    doc.add_signal_from_text("John Smith", "PER", 0.9);
    doc.add_signal_from_text("Apple", "ORG", 0.8);

    let html = render_document_html(&doc);

    assert!(
        html.contains("John Smith"),
        "HTML should contain entity text"
    );
    assert!(html.contains("Apple"), "HTML should contain entity text");
    assert!(
        html.contains("PER") || html.contains("per"),
        "HTML should contain entity type"
    );
    assert!(html.contains("signals"), "HTML should have signals section");
}

#[test]
fn test_eval_html_rendering() {
    let text = "John met Mary.";

    let gold = vec![
        Signal::new(0, Location::text(0, 4), "John", "PER", 1.0),
        Signal::new(1, Location::text(9, 13), "Mary", "PER", 1.0),
    ];
    let pred = vec![Signal::new(0, Location::text(0, 4), "John", "PER", 0.9)];

    let cmp = EvalComparison::compare(text, gold, pred);
    let html = render_eval_html(&cmp);

    assert!(html.contains("gold"), "Should have gold section");
    assert!(html.contains("predicted"), "Should have predicted section");
    assert!(
        html.contains("50.0%") || html.contains("50%"),
        "Should show recall"
    );
}

// =============================================================================
// End-to-End Pipeline Tests
// =============================================================================

#[test]
fn test_full_pipeline_extract_to_grounded() {
    let text = "John Smith from Boston received $500,000 on March 15, 2024.";
    let model = StackedNER::default();

    // Extract entities
    let entities = model.extract_entities(text, None).unwrap();

    // Convert to grounded document
    let mut doc = GroundedDocument::new("test", text);
    for e in &entities {
        let signal = Signal::new(
            0,
            Location::text(e.start, e.end),
            &e.text,
            e.entity_type.as_label(),
            e.confidence as f32,
        );
        doc.add_signal(signal);
    }

    // Validate
    let errors = doc.validate();
    assert!(
        errors.is_empty(),
        "Pipeline should produce valid signals: {:?}",
        errors
    );

    // Check we got expected entity types - at least money and date from pattern
    let types: Vec<_> = doc.signals().iter().map(|s| s.label()).collect();
    assert!(
        types.iter().any(|t| *t == "MONEY" || *t == "Money"),
        "Should find money. Got: {:?}",
        types
    );
    assert!(
        types.iter().any(|t| *t == "DATE" || *t == "Date"),
        "Should find date. Got: {:?}",
        types
    );
}

#[test]
fn test_pipeline_with_validation() {
    let text = "Marie Curie won the Nobel Prize.";
    let model = StackedNER::default();

    let entities = model.extract_entities(text, None).unwrap();

    let mut doc = GroundedDocument::new("test", text);
    let mut validation_failures = 0;

    for e in &entities {
        let signal = Signal::new(
            0,
            Location::text(e.start, e.end),
            &e.text,
            e.entity_type.as_label(),
            e.confidence as f32,
        );

        // Use validated add
        match doc.add_signal_validated(signal) {
            Ok(_) => {}
            Err(_) => validation_failures += 1,
        }
    }

    assert_eq!(validation_failures, 0, "All model outputs should be valid");
}

// =============================================================================
// Regression Tests
// =============================================================================

#[test]
fn regression_barack_obama_detected() {
    let model = StackedNER::default();
    let text = "Barack Obama was the 44th President.";
    let entities = model.extract_entities(text, None).unwrap();

    let found = entities
        .iter()
        .any(|e| e.text == "Barack Obama" && e.entity_type == EntityType::Person);

    assert!(
        found,
        "Barack Obama should be detected as a person. Got: {:?}",
        entities
    );
}

#[test]
fn regression_money_with_commas() {
    let model = RegexNER::new();
    let text = "Budget: $50,000.";
    let entities = model.extract_entities(text, None).unwrap();

    let found = entities
        .iter()
        .any(|e| e.text == "$50,000" && e.entity_type == EntityType::Money);

    assert!(
        found,
        "Should detect $50,000 with comma. Got: {:?}",
        entities
    );
}

#[test]
fn regression_signal_offsets_match_text() {
    let text = "Marie Curie won the Nobel Prize.";
    let model = StackedNER::default();
    let entities = model.extract_entities(text, None).unwrap();

    for e in &entities {
        let actual_text: String = text.chars().skip(e.start).take(e.end - e.start).collect();

        assert_eq!(
            e.text, actual_text,
            "Entity text '{}' should match text at offsets [{}, {}): '{}'",
            e.text, e.start, e.end, actual_text
        );
    }
}

#[test]
fn regression_eval_comparison_no_panic_on_empty() {
    let text = "No entities here.";
    let gold = vec![];
    let pred = vec![];

    let cmp = EvalComparison::compare(text, gold, pred);

    // Should not panic and metrics should be well-defined
    assert_eq!(cmp.correct_count(), 0);
    // Precision/recall undefined for empty, but should be 0.0
    assert_eq!(cmp.precision(), 0.0);
    assert_eq!(cmp.recall(), 0.0);
    assert_eq!(cmp.f1(), 0.0);
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_unicode_text() {
    let text = "München ist eine Stadt. 東京 is Tokyo.";
    let model = StackedNER::default();

    let entities = model.extract_entities(text, None).unwrap();

    // Just ensure no panics with unicode
    let mut doc = GroundedDocument::new("test", text);
    for e in &entities {
        doc.add_signal(Signal::new(
            0,
            Location::text(e.start, e.end),
            &e.text,
            e.entity_type.as_label(),
            e.confidence as f32,
        ));
    }

    let errors = doc.validate();
    assert!(
        errors.is_empty(),
        "Unicode text should validate: {:?}",
        errors
    );
}

#[test]
fn test_empty_text() {
    let model = StackedNER::default();
    let entities = model.extract_entities("", None).unwrap();
    assert!(entities.is_empty());
}

#[test]
fn test_overlapping_entities() {
    let text = "New York City is in New York State.";
    let mut doc = GroundedDocument::new("test", text);

    // These overlap
    doc.add_signal_from_text("New York City", "LOC", 0.9);
    doc.add_signal_from_text("New York", "LOC", 0.8);

    // Should still be valid (overlapping is allowed)
    assert!(doc.is_valid());
}

// =============================================================================
// Edge Case and Limitation Tests
// =============================================================================

#[test]
fn test_entity_with_punctuation_preserved() {
    let model = StackedNER::default();
    let text = "John Smith lives in Berlin, Germany.";
    let entities = model.extract_entities(text, None).unwrap();

    // Should find "Berlin, Germany" with the comma preserved
    let loc = entities
        .iter()
        .find(|e| e.entity_type == EntityType::Location);
    assert!(loc.is_some(), "Should find location");

    let loc = loc.unwrap();
    // The entity text should include the comma if it spans both words
    if loc.end - loc.start > 6 {
        // More than just "Berlin"
        assert!(
            loc.text.contains(','),
            "Multi-word location should preserve comma"
        );
    }
}

#[test]
fn test_possessive_entity() {
    let model = StackedNER::default();
    let text = "John Smith's company is successful.";
    let entities = model.extract_entities(text, None).unwrap();

    // The model currently includes the possessive - this is a known behavior
    let person = entities
        .iter()
        .find(|e| e.entity_type == EntityType::Person);
    assert!(person.is_some(), "Should find person");
}

#[test]
fn test_signal_validation_catches_text_mismatch() {
    let text = "Hello world";
    let signal = Signal::new(
        0,
        Location::text(0, 5),
        "WRONG", // Doesn't match "Hello"
        "TEST",
        0.9,
    );

    let err = signal.validate_against(text);
    assert!(err.is_some(), "Should catch text mismatch");
    assert!(matches!(
        err.unwrap(),
        SignalValidationError::TextMismatch { .. }
    ));
}

#[test]
fn test_signal_validation_out_of_bounds() {
    let text = "Hello";
    let signal = Signal::new(
        0,
        Location::text(0, 100), // End exceeds text length
        "Hello",
        "TEST",
        0.9,
    );

    let err = signal.validate_against(text);
    assert!(err.is_some(), "Should catch out of bounds");
}

#[test]
fn test_byte_vs_char_offsets_consistency() {
    // This test ensures byte offsets are correctly converted to character offsets
    let model = StackedNER::default();

    // Use text with multi-byte characters
    let text = "Tokyo (東京) is great. John Smith is there.";
    let entities = model.extract_entities(text, None).unwrap();

    for e in &entities {
        // Verify the entity text matches what's at the character offsets
        let chars: Vec<char> = text.chars().collect();
        if e.start < chars.len() && e.end <= chars.len() {
            let extracted: String = chars[e.start..e.end].iter().collect();
            assert_eq!(
                e.text, extracted,
                "Entity text '{}' should match chars[{}..{}]: '{}'",
                e.text, e.start, e.end, extracted
            );
        }
    }
}

#[test]
fn test_document_validation_integration() {
    let text = "Marie Curie won the Nobel Prize.";
    let model = StackedNER::default();
    let entities = model.extract_entities(text, None).unwrap();

    let mut doc = GroundedDocument::new("test", text);
    let mut failed = 0;

    for e in &entities {
        let signal = Signal::new(
            0,
            Location::text(e.start, e.end),
            &e.text,
            e.entity_type.as_label(),
            e.confidence as f32,
        );

        if signal.validate_against(text).is_some() {
            failed += 1;
        } else {
            doc.add_signal(signal);
        }
    }

    // With our fixes, all model entities should now be valid
    assert_eq!(failed, 0, "All model entities should pass validation");
    assert!(doc.is_valid(), "Document should be valid");
}

/// Test that statistical model needs context for location detection
/// This documents the known behavior that locations at sentence start
/// without a prefix like "in", "to", "from" may not be detected.
#[test]
fn test_location_needs_context() {
    let model = StackedNER::default();

    // Without prefix - may not detect
    let text1 = "New York City is great.";
    let entities1 = model.extract_entities(text1, None).unwrap();
    let _has_loc1 = entities1
        .iter()
        .any(|e| e.entity_type == EntityType::Location);

    // With prefix "to" - should detect
    let text2 = "I went to New York City.";
    let entities2 = model.extract_entities(text2, None).unwrap();
    let has_loc2 = entities2
        .iter()
        .any(|e| e.entity_type == EntityType::Location);

    // The statistical model requires context for reliable location detection
    assert!(has_loc2, "Should detect location with context prefix");
    // Note: has_loc1 may be false - this is expected behavior for statistical model
}

// =============================================================================
// Track and Identity Tests
// =============================================================================

#[test]
fn test_track_creation_from_signals() {
    let text = "Marie Curie won the prize.";
    let mut doc = GroundedDocument::new("test", text);

    // Add two signals for the same entity
    let s1 = doc.add_signal(Signal::new(
        0,
        Location::text(0, 11),
        "Marie Curie",
        "PER",
        0.95,
    ));

    // Create a track from these signals
    let track_id = doc.create_track_from_signals("Marie Curie", &[s1]);

    assert!(track_id.is_some(), "Should create track");
    let track = doc.get_track(track_id.unwrap()).unwrap();
    assert_eq!(track.canonical_surface, "Marie Curie");
    assert_eq!(track.signals.len(), 1);
}

#[test]
fn test_add_signal_to_track_updates_index() {
    let text = "Marie Curie was great. She won prizes.";
    let mut doc = GroundedDocument::new("test", text);

    // Add signals
    let s1 = doc.add_signal(Signal::new(
        0,
        Location::text(0, 11),
        "Marie Curie",
        "PER",
        0.95,
    ));
    let s2 = doc.add_signal(Signal::new(0, Location::text(23, 26), "She", "PRON", 0.90));

    // Create track with first signal
    let track_id = doc.create_track_from_signals("Marie Curie", &[s1]).unwrap();

    // Add second signal to track
    assert!(
        doc.add_signal_to_track(s2, track_id, 1),
        "Should add signal to track"
    );

    // Verify signal_to_track index is updated
    let track_for_she = doc.track_for_signal(s2);
    assert!(
        track_for_she.is_some(),
        "Should find track for 'She' signal"
    );
    assert_eq!(track_for_she.unwrap().canonical_surface, "Marie Curie");
}

#[test]
fn test_identity_creation_and_linking() {
    use anno::grounded::Identity;

    let text = "Marie Curie won the Nobel Prize.";
    let mut doc = GroundedDocument::new("test", text);

    // Add signal and track
    let s1 = doc.add_signal(Signal::new(
        0,
        Location::text(0, 11),
        "Marie Curie",
        "PER",
        0.95,
    ));
    let track_id = doc.create_track_from_signals("Marie Curie", &[s1]).unwrap();

    // Create identity with KB link
    let identity = Identity::from_kb(0, "Marie Curie", "wikidata", "Q7186");
    let identity_id = doc.add_identity(identity);

    // Link track to identity
    doc.link_track_to_identity(track_id, identity_id);

    // Verify linking
    let linked_identity = doc.identity_for_track(track_id);
    assert!(
        linked_identity.is_some(),
        "Track should be linked to identity"
    );
    assert_eq!(linked_identity.unwrap().kb_id, Some("Q7186".to_string()));
}

#[test]
fn test_full_signal_track_identity_hierarchy() {
    use anno::grounded::Identity;

    let text = "Barack Obama met Angela Merkel.";
    let mut doc = GroundedDocument::new("test", text);

    // Add signals
    let s1 = doc.add_signal(Signal::new(
        0,
        Location::text(0, 12),
        "Barack Obama",
        "PER",
        0.97,
    ));
    let s2 = doc.add_signal(Signal::new(
        0,
        Location::text(17, 30),
        "Angela Merkel",
        "PER",
        0.95,
    ));

    // Create tracks
    let t1 = doc
        .create_track_from_signals("Barack Obama", &[s1])
        .unwrap();
    let t2 = doc
        .create_track_from_signals("Angela Merkel", &[s2])
        .unwrap();

    // Create identities with KB links
    let id1 = doc.add_identity(Identity::from_kb(0, "Barack Obama", "wikidata", "Q76"));
    let id2 = doc.add_identity(Identity::from_kb(0, "Angela Merkel", "wikidata", "Q567"));

    // Link tracks to identities
    doc.link_track_to_identity(t1, id1);
    doc.link_track_to_identity(t2, id2);

    // Verify full hierarchy
    assert_eq!(doc.signals().len(), 2);
    assert_eq!(doc.tracks().count(), 2);
    assert_eq!(doc.identities().count(), 2);

    // Verify signal -> track -> identity chain
    let identity_for_obama = doc.identity_for_signal(s1);
    assert!(identity_for_obama.is_some());
    assert_eq!(identity_for_obama.unwrap().kb_id, Some("Q76".to_string()));

    let identity_for_merkel = doc.identity_for_signal(s2);
    assert!(identity_for_merkel.is_some());
    assert_eq!(identity_for_merkel.unwrap().kb_id, Some("Q567".to_string()));
}

#[test]
fn test_untracked_signals() {
    let text = "Barack Obama met Angela Merkel in Berlin.";
    let mut doc = GroundedDocument::new("test", text);

    // Add signals
    let s1 = doc.add_signal(Signal::new(
        0,
        Location::text(0, 12),
        "Barack Obama",
        "PER",
        0.97,
    ));
    let _s2 = doc.add_signal(Signal::new(
        0,
        Location::text(17, 30),
        "Angela Merkel",
        "PER",
        0.95,
    ));
    let _s3 = doc.add_signal(Signal::new(
        0,
        Location::text(34, 40),
        "Berlin",
        "LOC",
        0.90,
    ));

    // Only create track for Barack Obama
    let _t1 = doc.create_track_from_signals("Barack Obama", &[s1]);

    // Should have 2 untracked signals
    assert_eq!(doc.untracked_signal_count(), 2);
    let untracked = doc.untracked_signals();
    assert!(untracked.iter().any(|s| s.surface == "Angela Merkel"));
    assert!(untracked.iter().any(|s| s.surface == "Berlin"));
}

#[test]
fn test_document_stats() {
    use anno::grounded::Identity;

    let text = "Marie Curie was great.";
    let mut doc = GroundedDocument::new("test", text);

    let s1 = doc.add_signal(Signal::new(
        0,
        Location::text(0, 11),
        "Marie Curie",
        "PER",
        0.95,
    ));
    let track_id = doc.create_track_from_signals("Marie Curie", &[s1]).unwrap();
    let identity_id = doc.add_identity(Identity::from_kb(0, "Marie Curie", "wikidata", "Q7186"));
    doc.link_track_to_identity(track_id, identity_id);

    let stats = doc.stats();
    assert_eq!(stats.signal_count, 1);
    assert_eq!(stats.track_count, 1);
    assert_eq!(stats.identity_count, 1);
    assert_eq!(stats.linked_track_count, 1);
    assert_eq!(stats.untracked_count, 0);
}

// =============================================================================
// Regression Tests for Stress Testing Bugs
// =============================================================================

#[test]
fn test_unicode_html_rendering_char_offsets() {
    // Bug: Unicode text was rendered with byte offsets instead of char offsets
    // "café ☕ in München" - München was rendered as "in Mün"
    use anno::grounded::{render_document_html, GroundedDocument, Location, Signal};

    let text = "café ☕ in München with François";
    let mut doc = GroundedDocument::new("test", text);

    // Add signal with char offsets for "München" (chars 10-17)
    let signal = Signal::new(0, Location::text(10, 17), "München", "LOC", 0.9);
    doc.add_signal(signal);

    let html = render_document_html(&doc);

    // The HTML should contain the full "München" not truncated
    assert!(
        html.contains(">München<"),
        "Should render full 'München', got truncated"
    );
    assert!(
        !html.contains(">in Mün<"),
        "Should NOT have byte-sliced text"
    );
}

#[test]
fn test_repeated_entity_distinct_offsets() {
    // Bug: All same-text entities got first occurrence offset
    // "John John John" - all showed [0,4)
    use anno::grounded::{GroundedDocument, Location, Signal};

    let text = "John met John and John";
    let mut doc = GroundedDocument::new("test", text);

    // Add signals at different positions (like model would return)
    doc.add_signal(Signal::new(0, Location::text(0, 4), "John", "PER", 0.9));
    doc.add_signal(Signal::new(1, Location::text(9, 13), "John", "PER", 0.9));
    doc.add_signal(Signal::new(2, Location::text(18, 22), "John", "PER", 0.9));

    // Verify each signal has distinct offset
    let offsets: Vec<_> = doc
        .signals()
        .iter()
        .filter_map(|s| s.text_offsets())
        .collect();

    assert_eq!(offsets.len(), 3, "Should have 3 signals");
    assert_eq!(offsets[0], (0, 4), "First John at [0,4)");
    assert_eq!(offsets[1], (9, 13), "Second John at [9,13)");
    assert_eq!(offsets[2], (18, 22), "Third John at [18,22)");
}

// =============================================================================
// Spatial Index Implementation Tests
// =============================================================================

#[test]
fn test_interval_tree_adjacent_no_overlap() {
    // Adjacent (touching) intervals should NOT overlap
    use anno::grounded::{GroundedDocument, Location, Signal};

    let text = "JohnMary";
    let mut doc = GroundedDocument::new("test", text);
    doc.add_signal(Signal::new(0, Location::text(0, 4), "John", "PER", 0.9));
    doc.add_signal(Signal::new(1, Location::text(4, 8), "Mary", "PER", 0.9));

    let index = doc.build_text_index();
    let overlapping = index.query_overlap(0, 4);
    // Should only find John [0,4), not Mary [4,8)
    assert_eq!(
        overlapping.len(),
        1,
        "Adjacent intervals [0,4) and [4,8) should NOT overlap"
    );
}

#[test]
fn test_interval_tree_nested_containing() {
    // Test nested intervals with query_containing
    use anno::grounded::{GroundedDocument, Location, Signal};

    let text = "The New York Times";
    let mut doc = GroundedDocument::new("test", text);
    // Outer: "New York Times" [4,18)
    // Inner: "New York" [4,12)
    doc.add_signal(Signal::new(
        0,
        Location::text(4, 18),
        "New York Times",
        "ORG",
        0.9,
    ));
    doc.add_signal(Signal::new(
        1,
        Location::text(4, 12),
        "New York",
        "LOC",
        0.8,
    ));

    let index = doc.build_text_index();

    // Query for intervals containing [6,10) - "w Yo"
    let containing = index.query_containing(6, 10);
    // Both [4,18) and [4,12) contain [6,10)
    assert_eq!(
        containing.len(),
        2,
        "Both nested intervals should contain [6,10)"
    );
}

#[test]
fn test_interval_tree_empty_range() {
    // Single point query (empty range) should return nothing
    use anno::grounded::{GroundedDocument, Location, Signal};

    let text = "John met Mary";
    let mut doc = GroundedDocument::new("test", text);
    doc.add_signal(Signal::new(0, Location::text(0, 4), "John", "PER", 0.9));

    let index = doc.build_text_index();
    let empty = index.query_overlap(5, 5);
    assert!(empty.is_empty(), "Empty range query should return nothing");
}

#[test]
fn test_interval_tree_query_beyond_text() {
    // Query beyond text bounds should return nothing
    use anno::grounded::{GroundedDocument, Location, Signal};

    let text = "Hello";
    let mut doc = GroundedDocument::new("test", text);
    doc.add_signal(Signal::new(0, Location::text(0, 5), "Hello", "MISC", 0.9));

    let index = doc.build_text_index();
    let beyond = index.query_overlap(100, 200);
    assert!(beyond.is_empty(), "Query beyond text should be empty");
}

#[test]
fn test_interval_tree_contained_in() {
    // Test query_contained_in for finding signals within a range
    use anno::grounded::{GroundedDocument, Location, Signal};

    let text = "In New York, John met Mary at Google headquarters";
    let mut doc = GroundedDocument::new("test", text);
    doc.add_signal(Signal::new(
        0,
        Location::text(3, 11),
        "New York",
        "LOC",
        0.9,
    ));
    doc.add_signal(Signal::new(1, Location::text(13, 17), "John", "PER", 0.9));
    doc.add_signal(Signal::new(2, Location::text(22, 26), "Mary", "PER", 0.9));
    doc.add_signal(Signal::new(3, Location::text(30, 36), "Google", "ORG", 0.9));

    let index = doc.build_text_index();

    // Query for signals contained in [10, 35) - should find John and Mary
    let contained = index.query_contained_in(10, 35);
    assert_eq!(
        contained.len(),
        2,
        "Should find John[13,17) and Mary[22,26) in [10,35)"
    );
}

// =============================================================================
// HeuristicNER Implementation Bug Tests
// =============================================================================

/// Test that title + two-word name works correctly
/// Previously this was a bug where "Dr. John Smith" wasn't detected
#[test]
fn test_heuristic_title_multiword_name_fixed() {
    use anno::HeuristicNER;
    use anno::Model;

    let ner = HeuristicNER::new();

    // Single title + single name
    let e1 = ner.extract_entities("Dr. John said hello.", None).unwrap();
    assert!(!e1.is_empty(), "Dr. + single name should work");

    // Two-word name without title
    let e2 = ner
        .extract_entities("John Smith said hello.", None)
        .unwrap();
    assert!(!e2.is_empty(), "Two-word name should work");

    // Title + two-word name (previously a bug, now fixed)
    let e3 = ner
        .extract_entities("Dr. John Smith said hello.", None)
        .unwrap();
    assert!(!e3.is_empty(), "Title + two-word name should now work");
    assert_eq!(
        e3[0].text, "Dr. John Smith",
        "Should include title and full name"
    );
}

/// Test that validation catches control character mismatches
#[test]
fn test_control_char_validation_mismatch() {
    use anno::grounded::{GroundedDocument, Location, Signal};

    // Text with null byte: "John\x00Smith"
    let text = "John\x00Smith";
    let mut doc = GroundedDocument::new("test", text);

    // Model returned "John Smith" (without null) at [0,10)
    // But text[0:10] is "John\x00Smith" (with null)
    let signal = Signal::new(0, Location::text(0, 10), "John Smith", "PER", 0.9);
    doc.add_signal(signal);

    // Validation should catch this mismatch
    let errors = doc.validate();
    assert!(
        !errors.is_empty(),
        "Should detect text mismatch with control chars"
    );
}

// =============================================================================
// Known Limitations Tests
// =============================================================================

/// GLiNER struggles with entities preceded by certain punctuation
/// This documents a known limitation of the transformer-based model
#[test]
fn test_gliner_punctuation_limitation() {
    use anno::Model;
    use anno::StackedNER;

    // The stacked model (which includes heuristic) handles these cases
    let model = StackedNER::default();

    // These cases are problematic for pure GLiNER:
    let cases = [".John Smith", "(John Smith)", "[John Smith]"];

    for text in cases {
        let entities = model.extract_entities(text, None).unwrap();
        // Stacked model should find something due to heuristic layer
        assert!(
            !entities.is_empty(),
            "Stacked model should find entity in: {}",
            text
        );
    }
}

/// Test overlapping entities from GLiNER (nested NER behavior)
#[test]
fn test_overlapping_entity_detection() {
    use anno::grounded::{GroundedDocument, Location, Signal};

    // GLiNER can return overlapping entities (e.g., "محمد Ali" and "Ali")
    let text = "محمد Ali met Cohen";
    let mut doc = GroundedDocument::new("test", text);

    // Add overlapping signals (simulating GLiNER output)
    doc.add_signal(Signal::new(
        0,
        Location::text(0, 8),
        "محمد Ali",
        "PER",
        0.92,
    ));
    doc.add_signal(Signal::new(1, Location::text(5, 8), "Ali", "PER", 0.68));

    // Query overlapping signals at position 6
    let index = doc.build_text_index();
    let overlapping = index.query_overlap(6, 7);

    // Both signals should overlap at position 6
    assert_eq!(overlapping.len(), 2, "Should find both overlapping signals");
}

// =============================================================================
// HeuristicNER Tests
// =============================================================================

/// HeuristicNER should detect persons from two-word capitalized sequences
#[test]
fn test_heuristic_ner_persons() {
    use anno::{EntityType, HeuristicNER, Model};

    let model = HeuristicNER::new();
    let entities = model
        .extract_entities("John Smith works here.", None)
        .unwrap();

    assert!(entities
        .iter()
        .any(|e| e.text == "John Smith" && e.entity_type == EntityType::Person));
}

/// HeuristicNER should detect known organizations
#[test]
fn test_heuristic_ner_known_orgs() {
    use anno::{EntityType, HeuristicNER, Model};

    let model = HeuristicNER::new();

    for text in [
        "Google announced.",
        "Apple released.",
        "Microsoft competed.",
    ] {
        let entities = model.extract_entities(text, None).unwrap();
        assert!(
            entities
                .iter()
                .any(|e| e.entity_type == EntityType::Organization),
            "Should detect org in: {}",
            text
        );
    }
}

/// HeuristicNER should detect locations from context
#[test]
fn test_heuristic_ner_location_context() {
    use anno::{EntityType, HeuristicNER, Model};

    let model = HeuristicNER::new();
    let entities = model.extract_entities("She lives in Paris.", None).unwrap();

    assert!(
        entities
            .iter()
            .any(|e| e.text == "Paris" && e.entity_type == EntityType::Location),
        "Got entities: {:?}",
        entities
            .iter()
            .map(|e| (&e.text, &e.entity_type))
            .collect::<Vec<_>>()
    );
}

/// HeuristicNER should skip pronouns at sentence start
#[test]
fn test_heuristic_ner_skip_pronouns() {
    use anno::{EntityType, HeuristicNER, Model};

    let model = HeuristicNER::new();
    let entities = model.extract_entities("She went home.", None).unwrap();

    // Should not detect "She" as an entity
    assert!(
        !entities.iter().any(|e| e.text == "She"),
        "Should not detect pronouns as entities"
    );
}
