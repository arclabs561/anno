//! NER evaluation tests
use anno::eval::{entity_type_matches, entity_type_to_string, evaluate_ner_model, GoldEntity};
use anno::{EntityType, PatternNER};

#[test]
fn test_entity_type_matches() {
    assert!(entity_type_matches(
        &EntityType::Person,
        &EntityType::Person
    ));
    assert!(entity_type_matches(
        &EntityType::Organization,
        &EntityType::Organization
    ));
    assert!(entity_type_matches(
        &EntityType::Location,
        &EntityType::Location
    ));
    assert!(!entity_type_matches(
        &EntityType::Person,
        &EntityType::Organization
    ));

    // Test Other variant
    assert!(entity_type_matches(
        &EntityType::Other("misc".to_string()),
        &EntityType::Other("misc".to_string())
    ));
    assert!(!entity_type_matches(
        &EntityType::Other("misc".to_string()),
        &EntityType::Other("other".to_string())
    ));
}

#[test]
fn test_entity_type_to_string() {
    assert_eq!(entity_type_to_string(&EntityType::Person), "PER");
    assert_eq!(entity_type_to_string(&EntityType::Organization), "ORG");
    assert_eq!(entity_type_to_string(&EntityType::Location), "LOC");
    assert_eq!(
        entity_type_to_string(&EntityType::Other("custom".to_string())),
        "custom"
    );
}

#[test]
fn test_evaluate_ner_model_basic() {
    let model = PatternNER::new();

    let test_cases = vec![(
        "Meeting on January 15, 2025 for $100".to_string(),
        vec![
            GoldEntity::with_span("January 15, 2025", EntityType::Date, 11, 27),
            GoldEntity::with_span("$100", EntityType::Money, 32, 36),
        ],
    )];

    let metrics = evaluate_ner_model(&model, &test_cases).unwrap();

    // Should have valid metrics
    assert!(metrics.precision >= 0.0 && metrics.precision <= 1.0);
    assert!(metrics.recall >= 0.0 && metrics.recall <= 1.0);
    assert!(metrics.f1 >= 0.0 && metrics.f1 <= 1.0);
}

#[test]
fn test_evaluate_empty_dataset() {
    let model = PatternNER::new();
    let test_cases: Vec<(String, Vec<GoldEntity>)> = vec![];

    let metrics = evaluate_ner_model(&model, &test_cases).unwrap();

    assert_eq!(metrics.precision, 0.0);
    assert_eq!(metrics.recall, 0.0);
    assert_eq!(metrics.f1, 0.0);
}
