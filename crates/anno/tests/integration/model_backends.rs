//! Integration tests for model backends.
//!
//! These tests download real models and run inference.
//! Run with: `cargo test -p anno --features onnx --test model_backends -- --ignored`

use anno::{EntityType, HeuristicNER, Model, StackedNER};

// =============================================================================
// StackedNER (requires model download)
// =============================================================================

#[test]
#[ignore]
fn test_stacked_ner_default_loads() {
    // StackedNER::default() with the `onnx` feature will attempt to download
    // and load the BERT ONNX model. This test verifies it completes without panic.
    let ner = StackedNER::default();
    assert!(ner.is_available());
    assert!(
        ner.num_layers() >= 2,
        "expected at least 2 layers (ML + regex/heuristic)"
    );
}

#[test]
#[ignore]
fn test_stacked_ner_predict_basic() {
    let ner = StackedNER::default();
    let entities = ner
        .extract_entities("Alice works at Google in London", None)
        .expect("extract_entities should not fail");

    assert!(
        !entities.is_empty(),
        "expected at least one entity from 'Alice works at Google in London'"
    );

    // Check that we got some recognizable entity texts
    let texts: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
    // At minimum, a decent NER model should find at least one of these
    let found_any_expected = texts
        .iter()
        .any(|t| t.contains("Alice") || t.contains("Google") || t.contains("London"));
    assert!(
        found_any_expected,
        "expected at least one of Alice/Google/London in entity texts, got: {:?}",
        texts
    );
}

#[test]
#[ignore]
fn test_stacked_ner_predict_empty() {
    let ner = StackedNER::default();
    let entities = ner
        .extract_entities("", None)
        .expect("extract_entities on empty string should not fail");

    assert!(
        entities.is_empty(),
        "expected no entities from empty string, got: {:?}",
        entities
    );
}

#[test]
#[ignore]
fn test_stacked_ner_entity_types() {
    let ner = StackedNER::default();
    let entities = ner
        .extract_entities("Alice works at Google in London", None)
        .expect("extract_entities should not fail");

    for entity in &entities {
        match &entity.entity_type {
            EntityType::Person
            | EntityType::Organization
            | EntityType::Location
            | EntityType::Custom { .. }
            // Pattern-layer types (dates, money, etc.)
            | EntityType::Date
            | EntityType::Time
            | EntityType::Money
            | EntityType::Percent
            | EntityType::Email
            | EntityType::Url
            | EntityType::Phone => {}
            other => {
                panic!(
                    "unexpected entity type {:?} for entity {:?}",
                    other, entity.text
                );
            }
        }
    }
}

#[test]
#[ignore]
fn test_stacked_ner_char_offsets_valid() {
    let text = "Alice works at Google in London";
    let text_char_count = text.chars().count();

    let ner = StackedNER::default();
    let entities = ner
        .extract_entities(text, None)
        .expect("extract_entities should not fail");

    for entity in &entities {
        assert!(
            entity.start() < entity.end(),
            "entity {:?}: start ({}) must be < end ({})",
            entity.text,
            entity.start(),
            entity.end()
        );
        assert!(
            entity.end() <= text_char_count,
            "entity {:?}: end ({}) exceeds text length ({})",
            entity.text,
            entity.end(),
            text_char_count
        );
        // Verify the span text matches the entity text (character offsets)
        let extracted: String = text
            .chars()
            .skip(entity.start())
            .take(entity.end() - entity.start())
            .collect();
        assert_eq!(
            extracted.trim(),
            entity.text.trim(),
            "span text mismatch for entity at [{}, {})",
            entity.start(),
            entity.end()
        );
    }
}

#[test]
#[ignore]
fn test_stacked_ner_confidence_range() {
    let ner = StackedNER::default();
    let entities = ner
        .extract_entities("Alice works at Google in London", None)
        .expect("extract_entities should not fail");

    assert!(!entities.is_empty(), "need entities to test confidence");

    for entity in &entities {
        assert!(
            (0.0..=1.0).contains(&entity.confidence),
            "entity {:?}: confidence {} outside [0.0, 1.0]",
            entity.text,
            entity.confidence
        );
    }
}

// =============================================================================
// HeuristicNER (no download required)
// =============================================================================

#[test]
fn test_heuristic_ner_no_model_needed() {
    // HeuristicNER works without any download -- it uses rule-based extraction.
    let ner = HeuristicNER::new();
    assert!(ner.is_available());

    let entities = ner
        .extract_entities("Dr. Smith went to New York", None)
        .expect("HeuristicNER should not fail");

    // Heuristic NER should find at least one entity
    assert!(
        !entities.is_empty(),
        "HeuristicNER should find entities in 'Dr. Smith went to New York'"
    );
}

// =============================================================================
// Model capabilities
// =============================================================================

#[test]
#[ignore]
fn test_model_capabilities() {
    let ner = StackedNER::default();
    let caps = ner.capabilities();

    // StackedNER has no special capabilities by default
    assert!(!caps.relation_capable);
    assert!(!caps.zero_shot);
    assert!(!caps.discontinuous_capable);
}
