//! Tests for MockModel functionality.

use anno::{Entity, EntityCategory, EntityType, MockModel, Model};

#[test]
fn test_mock_model_new() {
    let mock = MockModel::new("test-mock");
    assert_eq!(mock.name(), "test-mock");
    assert!(mock.is_available());
    assert_eq!(mock.description(), "Mock NER model for testing");
}

#[test]
fn test_mock_model_with_entities() {
    // "John works at Apple" - character offsets:
    // "John" = chars 0-4
    // "Apple" = chars 14-19 (after "John works at ")
    // "John works at " = 14 chars, so "Apple" starts at 14
    let text = "John works at Apple";
    let entities = vec![
        Entity::new("John", EntityType::Person, 0, 4, 0.9),
        Entity::new("Apple", EntityType::Organization, 14, 19, 0.95),
    ];

    let mock = MockModel::new("test").with_entities(entities.clone());
    let result = mock.extract_entities(text, None);
    assert!(result.is_ok());
    let extracted = result.unwrap();
    assert_eq!(extracted.len(), 2);
    assert_eq!(extracted[0].text, "John");
    assert_eq!(extracted[1].text, "Apple");
}

#[test]
fn test_mock_model_with_types() {
    let types = vec![EntityType::Person, EntityType::Organization];
    let mock = MockModel::new("test").with_types(types.clone());

    let supported = mock.supported_types();
    assert_eq!(supported.len(), 2);
    assert!(supported.contains(&EntityType::Person));
    assert!(supported.contains(&EntityType::Organization));
}

#[test]
fn test_mock_model_validation_enabled() {
    let entities = vec![Entity::new("John", EntityType::Person, 0, 4, 0.9)];

    let mock = MockModel::new("test").with_entities(entities);

    // Valid text - should work
    let result = mock.extract_entities("John", None);
    assert!(result.is_ok());

    // Invalid text (too short) - should error
    let result = mock.extract_entities("Jo", None);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("exceeds text length"));
}

#[test]
fn test_mock_model_validation_disabled() {
    let entities = vec![Entity::new("John", EntityType::Person, 0, 4, 0.9)];

    let mock = MockModel::new("test")
        .with_entities(entities)
        .without_validation();

    // Even with invalid text, should not error when validation disabled
    let result = mock.extract_entities("Jo", None);
    assert!(result.is_ok());
}

#[test]
fn test_mock_model_text_mismatch_validation() {
    let entities = vec![Entity::new("John", EntityType::Person, 0, 4, 0.9)];

    let mock = MockModel::new("test").with_entities(entities);

    // Text doesn't match entity text at that position
    let result = mock.extract_entities("Jane", None);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("text mismatch"));
}

#[test]
fn test_mock_model_empty_entities() {
    let mock = MockModel::new("test");
    let result = mock.extract_entities("Any text", None);
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
}

#[test]
fn test_mock_model_unicode_validation() {
    let entities = vec![
        Entity::new("北京", EntityType::Location, 0, 2, 0.9), // 2 chars, 6 bytes
    ];

    let mock = MockModel::new("test").with_entities(entities);

    // Valid - text matches
    let result = mock.extract_entities("北京", None);
    assert!(result.is_ok());

    // Invalid - text too short
    let result = mock.extract_entities("北", None);
    assert!(result.is_err());
}

#[test]
#[should_panic(expected = "start")]
fn test_mock_model_with_entities_panics_on_invalid() {
    // MockModel::with_entities should panic if start >= end
    let entities = vec![
        Entity::new("test", EntityType::Person, 5, 5, 0.9), // start == end
    ];
    let _mock = MockModel::new("test").with_entities(entities);
}

#[test]
fn test_mock_model_with_entities_handles_clamped_confidence() {
    // Entity::new clamps confidence to [0.0, 1.0], so 1.5 becomes 1.0
    // MockModel validation should pass because the entity has valid confidence after clamping
    let entities = vec![
        Entity::new("test", EntityType::Person, 0, 4, 1.5), // confidence > 1.0, but clamped to 1.0
    ];
    let mock = MockModel::new("test").with_entities(entities);
    // Should work because Entity::new clamps the confidence
    let result = mock.extract_entities("test", None);
    assert!(result.is_ok());
    let extracted = result.unwrap();
    assert_eq!(extracted.len(), 1);
    assert_eq!(extracted[0].confidence, 1.0); // Clamped to 1.0
}

#[test]
fn test_mock_model_multiple_entities() {
    // Text: "John works at Apple on January 15"
    // Character offsets (not byte offsets):
    // "John" = chars 0-4
    // "Apple" = chars 14-19 (after "John works at ")
    // "January 15" = chars 23-33 (after "John works at Apple on ")
    let text = "John works at Apple on January 15";
    let entities = vec![
        Entity::new("John", EntityType::Person, 0, 4, 0.9),
        Entity::new("Apple", EntityType::Organization, 14, 19, 0.95),
        Entity::new("January 15", EntityType::Date, 23, 33, 0.98),
    ];

    let mock = MockModel::new("test").with_entities(entities.clone());
    let result = mock.extract_entities(text, None);
    assert!(result.is_ok());
    let extracted = result.unwrap();
    assert_eq!(extracted.len(), 3);
}

#[test]
fn test_mock_model_validation_character_offsets() {
    // Test that validation uses character offsets, not byte offsets
    let entities = vec![
        Entity::new("€50", EntityType::Money, 6, 9, 0.9), // "Price " = 6 chars, "€50" = 3 chars
    ];

    let mock = MockModel::new("test").with_entities(entities);
    let text = "Price €50"; // "Price " = 6 chars, "€50" = 3 chars (but 6 bytes for "Price " + 3 bytes for € + 2 bytes for "50")

    let result = mock.extract_entities(text, None);
    assert!(result.is_ok());
}

#[test]
fn test_mock_model_supported_types_empty() {
    let mock = MockModel::new("test");
    let types = mock.supported_types();
    assert!(types.is_empty());
}

#[test]
fn test_mock_model_supported_types_custom() {
    let types = vec![
        EntityType::Person,
        EntityType::Organization,
        EntityType::custom("DISEASE", EntityCategory::Agent),
    ];

    let mock = MockModel::new("test").with_types(types.clone());
    let supported = mock.supported_types();
    assert_eq!(supported.len(), 3);
}
