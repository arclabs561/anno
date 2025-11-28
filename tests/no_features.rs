//! Tests for the minimal no-features configuration.
//!
//! These tests verify that anno works correctly when compiled with
//! `default = []` (no features enabled). This is the lightest configuration
//! and should provide:
//!
//! - PatternNER (dates, money, email, phone, etc.)
//! - StatisticalNER (basic named entity heuristics)
//! - StackedNER (combination of the above)
//! - Basic eval module (always compiled, not feature-gated)

use anno::{Entity, EntityType, Model, PatternNER, StackedNER, StatisticalNER};

// =============================================================================
// PatternNER (always available)
// =============================================================================

#[test]
fn test_pattern_ner_dates() {
    let model = PatternNER::new();
    let entities = model
        .extract_entities("Meeting on January 15, 2024", None)
        .unwrap();

    assert!(!entities.is_empty(), "Should find date");
    assert!(
        entities.iter().any(|e| e.entity_type == EntityType::Date),
        "Should find DATE type"
    );
}

#[test]
fn test_pattern_ner_money() {
    let model = PatternNER::new();
    let entities = model.extract_entities("Cost: $99.99", None).unwrap();

    assert!(!entities.is_empty(), "Should find money");
    assert!(
        entities.iter().any(|e| e.entity_type == EntityType::Money),
        "Should find MONEY type"
    );
}

#[test]
fn test_pattern_ner_email() {
    let model = PatternNER::new();
    let entities = model
        .extract_entities("Contact: alice@example.com", None)
        .unwrap();

    assert!(!entities.is_empty(), "Should find email");
    assert!(
        entities.iter().any(|e| e.entity_type == EntityType::Email),
        "Should find EMAIL type"
    );
}

#[test]
fn test_pattern_ner_phone() {
    let model = PatternNER::new();
    let entities = model
        .extract_entities("Call 555-123-4567", None)
        .unwrap();

    assert!(!entities.is_empty(), "Should find phone");
    assert!(
        entities.iter().any(|e| e.entity_type == EntityType::Phone),
        "Should find PHONE type"
    );
}

#[test]
fn test_pattern_ner_url() {
    let model = PatternNER::new();
    let entities = model
        .extract_entities("Visit https://example.com", None)
        .unwrap();

    assert!(!entities.is_empty(), "Should find URL");
    assert!(
        entities.iter().any(|e| e.entity_type == EntityType::Url),
        "Should find URL type"
    );
}

#[test]
fn test_pattern_ner_percent() {
    let model = PatternNER::new();
    let entities = model.extract_entities("Got 95% on the test", None).unwrap();

    assert!(!entities.is_empty(), "Should find percent");
    assert!(
        entities.iter().any(|e| e.entity_type == EntityType::Percent),
        "Should find PERCENT type"
    );
}

#[test]
fn test_pattern_ner_time() {
    let model = PatternNER::new();
    let entities = model.extract_entities("Meeting at 3:30 PM", None).unwrap();

    assert!(!entities.is_empty(), "Should find time");
    assert!(
        entities.iter().any(|e| e.entity_type == EntityType::Time),
        "Should find TIME type"
    );
}

#[test]
fn test_pattern_ner_multiple_entities() {
    let model = PatternNER::new();
    let text = "Send $500 to alice@test.com by January 1, 2025";
    let entities = model.extract_entities(text, None).unwrap();

    assert!(entities.len() >= 3, "Should find at least 3 entities");

    let types: Vec<_> = entities.iter().map(|e| &e.entity_type).collect();
    assert!(types.contains(&&EntityType::Money));
    assert!(types.contains(&&EntityType::Email));
    assert!(types.contains(&&EntityType::Date));
}

// =============================================================================
// StatisticalNER (always available)
// =============================================================================

#[test]
fn test_statistical_ner_basic() {
    let model = StatisticalNER::new();
    let entities = model
        .extract_entities("John Smith works at Apple Inc", None)
        .unwrap();

    // StatisticalNER uses heuristics, may or may not find entities
    // but should not panic - just verify it returned successfully
    let _ = entities;
}

#[test]
fn test_statistical_ner_name() {
    let model = StatisticalNER::new();
    let name = model.name();
    assert!(!name.is_empty());
}

// =============================================================================
// StackedNER (always available, recommended default)
// =============================================================================

#[test]
fn test_stacked_ner_default() {
    let model = StackedNER::default();
    let entities = model
        .extract_entities("Dr. Smith charges $100/hr on Mondays", None)
        .unwrap();

    // Should find at least money and possibly person
    assert!(!entities.is_empty(), "StackedNER should find entities");
}

#[test]
fn test_stacked_ner_combines_backends() {
    let model = StackedNER::default();
    let text = "Email bob@test.com about the $500 invoice";
    let entities = model.extract_entities(text, None).unwrap();

    // Should get email and money from PatternNER
    let types: Vec<_> = entities.iter().map(|e| &e.entity_type).collect();
    assert!(types.contains(&&EntityType::Email) || types.contains(&&EntityType::Money));
}

// =============================================================================
// Entity type and confidence
// =============================================================================

#[test]
fn test_entity_has_confidence() {
    let model = PatternNER::new();
    let entities = model.extract_entities("$100", None).unwrap();

    assert!(!entities.is_empty());
    let e = &entities[0];
    assert!(e.confidence >= 0.0 && e.confidence <= 1.0);
}

#[test]
fn test_entity_has_span() {
    let model = PatternNER::new();
    let text = "Pay $50 now";
    let entities = model.extract_entities(text, None).unwrap();

    assert!(!entities.is_empty());
    let e = &entities[0];
    assert!(e.start < e.end);
    assert!(e.end <= text.len());

    // Verify span extracts correct text
    assert_eq!(&text[e.start..e.end], e.text);
}

#[test]
fn test_entity_type_label() {
    let model = PatternNER::new();
    let entities = model.extract_entities("$100", None).unwrap();

    assert!(!entities.is_empty());
    let label = entities[0].entity_type.as_label();
    assert!(!label.is_empty());
}

// =============================================================================
// Model trait
// =============================================================================

#[test]
fn test_model_trait_name() {
    let pattern = PatternNER::new();
    let stacked = StackedNER::default();
    let statistical = StatisticalNER::new();

    assert!(!pattern.name().is_empty());
    assert!(!stacked.name().is_empty());
    assert!(!statistical.name().is_empty());
}

#[test]
fn test_model_trait_supported_types() {
    let model = PatternNER::new();
    let types = model.supported_types();

    // PatternNER should support these
    assert!(types.contains(&EntityType::Date));
    assert!(types.contains(&EntityType::Money));
    assert!(types.contains(&EntityType::Email));
}

// =============================================================================
// Edge cases
// =============================================================================

#[test]
fn test_empty_input() {
    let model = PatternNER::new();
    let entities = model.extract_entities("", None).unwrap();
    assert!(entities.is_empty());
}

#[test]
fn test_whitespace_only() {
    let model = PatternNER::new();
    let entities = model.extract_entities("   \t\n  ", None).unwrap();
    assert!(entities.is_empty());
}

#[test]
fn test_no_entities() {
    let model = PatternNER::new();
    let entities = model
        .extract_entities("The quick brown fox jumps", None)
        .unwrap();
    assert!(entities.is_empty());
}

#[test]
fn test_unicode_text() {
    let model = PatternNER::new();
    // Should handle unicode without panicking
    let entities = model
        .extract_entities("会议在 $500 的价格", None)
        .unwrap();
    // May or may not find entities, but should not panic
    let _ = entities;
}

// =============================================================================
// Eval module (always available, not feature-gated)
// =============================================================================

#[test]
fn test_eval_modes_available() {
    use anno::eval::modes::MultiModeResults;
    use anno::eval::GoldEntity;

    let predicted = vec![Entity::new("John", EntityType::Person, 0, 4, 0.9)];
    let gold = vec![GoldEntity::new("John", EntityType::Person, 0)];

    let results = MultiModeResults::compute(&predicted, &gold);
    assert!(results.strict.f1 >= 0.0);
}

#[test]
fn test_eval_config_available() {
    use anno::eval::modes::EvalConfig;

    let config = EvalConfig::new().with_min_overlap(0.5);
    assert!((config.min_overlap - 0.5).abs() < 0.001);
}

#[test]
fn test_bio_adapter_available() {
    use anno::eval::bio_adapter::{bio_to_entities, BioScheme};

    let tokens = ["John", "works"];
    let tags = ["B-PER", "O"];

    let entities = bio_to_entities(&tokens, &tags, BioScheme::IOB2).unwrap();
    assert_eq!(entities.len(), 1);
}


