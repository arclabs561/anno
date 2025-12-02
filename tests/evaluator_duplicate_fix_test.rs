//! Test for entity matching double-counting bug fix.
//!
//! This test verifies that the fix in `src/eval/evaluator.rs` correctly prevents
//! duplicate predictions from inflating precision scores. Each gold entity should
//! be matched at most once, even if multiple predictions match it.

use anno::eval::evaluator::{NEREvaluator, StandardNEREvaluator};
use anno::eval::GoldEntity;
use anno::{Entity, EntityType, MockModel};

#[test]
fn test_duplicate_predictions_dont_inflate_precision() {
    // This test verifies the fix for the double-counting bug in evaluator.rs
    // Bug: Multiple predictions matching the same gold entity were all counted as correct
    // Fix: Track which gold entities have been matched, ensuring each is matched at most once

    let evaluator = StandardNEREvaluator::new();

    let text = "John Smith works at Apple Inc.";

    // Create gold: single entity (GoldEntity uses character offsets)
    // "John Smith" = 10 characters, so end = 0 + 10 = 10
    let gold = vec![GoldEntity::new("John Smith", EntityType::Person, 0)];

    // Create model that returns duplicate predictions
    // Entity also uses character offsets (checked in entity.rs)
    let duplicate_entities = vec![
        Entity::new("John Smith", EntityType::Person, 0, 10, 0.9),
        Entity::new("John Smith", EntityType::Person, 0, 10, 0.9), // Duplicate
    ];
    let model = MockModel::new("duplicate-test")
        .with_entities(duplicate_entities)
        .without_validation(); // Disable validation to allow duplicates

    let metrics = evaluator
        .evaluate_test_case(&model, text, &gold, None)
        .unwrap();

    // Verify fix: only one match should be counted
    // With 2 predictions and 1 correct match, precision should be 0.5, not 1.0
    assert_eq!(
        metrics.correct, 1,
        "Duplicate predictions should not inflate correct count. Expected 1, got {}",
        metrics.correct
    );

    assert_eq!(metrics.found, 2, "Should report 2 predictions found");

    // Precision: 1 correct / 2 predicted = 0.5
    assert!(
        (metrics.precision.get() - 0.5).abs() < 0.001,
        "Precision should be 0.5 (1 correct / 2 predicted), not 1.0. Got {}",
        metrics.precision.get()
    );

    // Recall: 1 correct / 1 gold = 1.0
    assert!(
        (metrics.recall.get() - 1.0).abs() < 0.001,
        "Recall should be 1.0 (1 correct / 1 gold). Got {}",
        metrics.recall.get()
    );
}

#[test]
fn test_multiple_gold_entities_with_duplicate_predictions() {
    // Test case: Multiple gold entities, some predictions are duplicates
    let evaluator = StandardNEREvaluator::new();
    let text = "John Smith and Jane Doe work at Apple Inc.";

    // Calculate correct character offsets
    // "John Smith" = 0-10, " and " = 10-15, "Jane Doe" = 15-23, " work at " = 23-32, "Apple Inc." = 32-42
    let gold = vec![
        GoldEntity::new("John Smith", EntityType::Person, 0), // 0-10
        GoldEntity::new("Jane Doe", EntityType::Person, 15),  // 15-23
        GoldEntity::new("Apple Inc.", EntityType::Organization, 32), // 32-42
    ];

    // Predictions: correct matches + one duplicate
    let predicted = vec![
        Entity::new("John Smith", EntityType::Person, 0, 10, 0.9),
        Entity::new("John Smith", EntityType::Person, 0, 10, 0.9), // Duplicate
        Entity::new("Jane Doe", EntityType::Person, 15, 23, 0.9),
        Entity::new("Apple Inc.", EntityType::Organization, 32, 42, 0.9),
    ];

    let model = MockModel::new("duplicate-test")
        .with_entities(predicted)
        .without_validation();
    let metrics = evaluator
        .evaluate_test_case(&model, text, &gold, None)
        .unwrap();

    // Should have 3 correct matches (one for each gold), not 4
    assert_eq!(
        metrics.correct, 3,
        "Should match 3 gold entities (one each), not 4. Got {}",
        metrics.correct
    );

    // Precision: 3 correct / 4 predicted = 0.75
    assert!(
        (metrics.precision.get() - 0.75).abs() < 0.001,
        "Precision should be 0.75 (3 correct / 4 predicted), got {}",
        metrics.precision.get()
    );

    // Recall: 3 correct / 3 gold = 1.0
    assert!(
        (metrics.recall.get() - 1.0).abs() < 0.001,
        "Recall should be 1.0 (3 correct / 3 gold), got {}",
        metrics.recall.get()
    );
}

#[test]
fn test_per_type_stats_with_duplicates() {
    // Test that per-type statistics also prevent double-counting
    let evaluator = StandardNEREvaluator::new();
    let text = "John Smith and John Smith work together.";

    // Calculate correct character offsets
    // "John Smith" = 0-10, " and " = 10-15, "John Smith" = 15-25
    let gold = vec![
        GoldEntity::new("John Smith", EntityType::Person, 0), // 0-10
        GoldEntity::new("John Smith", EntityType::Person, 15), // 15-25
    ];

    // Predictions: correct matches + duplicates
    let predicted = vec![
        Entity::new("John Smith", EntityType::Person, 0, 10, 0.9),
        Entity::new("John Smith", EntityType::Person, 0, 10, 0.9), // Duplicate of first
        Entity::new("John Smith", EntityType::Person, 15, 25, 0.9),
        Entity::new("John Smith", EntityType::Person, 15, 25, 0.9), // Duplicate of second
    ];

    let model = MockModel::new("duplicate-test")
        .with_entities(predicted)
        .without_validation();
    let metrics = evaluator
        .evaluate_test_case(&model, text, &gold, None)
        .unwrap();

    // Should have 2 correct (one for each gold entity), not 4
    assert_eq!(
        metrics.correct, 2,
        "Per-type stats should count 2 correct (one per gold), not 4. Got {}",
        metrics.correct
    );

    // Check per-type metrics
    // The key is created by entity_type_to_string, which uses as_label()
    // EntityType::Person.as_label() returns "PER" (not "Person")
    let person_key = EntityType::Person.as_label();
    let person_metrics = metrics.per_type.get(person_key);
    assert!(
        person_metrics.is_some(),
        "Should have per-type metrics for Person (key: '{}'). Available keys: {:?}",
        person_key,
        metrics.per_type.keys().collect::<Vec<_>>()
    );

    if let Some(person_metrics) = person_metrics {
        // Should have 2 correct (one per gold), not 4
        // TypeMetrics has: found, expected, correct (but correct is not directly exposed)
        // We need to verify through precision/recall or check the counts
        // Actually, TypeMetrics has: found, expected, but correct is calculated from precision/recall
        // Let's verify the counts are correct
        assert_eq!(
            person_metrics.found, 4,
            "Person type should have 4 found (all predictions). Got {}",
            person_metrics.found
        );
        assert_eq!(
            person_metrics.expected, 2,
            "Person type should have 2 expected (both gold entities). Got {}",
            person_metrics.expected
        );
        // Verify precision reflects correct count: 2 correct / 4 found = 0.5
        assert!(
            (person_metrics.precision - 0.5).abs() < 0.001,
            "Person precision should be 0.5 (2 correct / 4 found), got {}",
            person_metrics.precision
        );
    }
}
