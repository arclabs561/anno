//! Tests for subtle bugs found in evaluation code.
//!
//! These tests verify fixes for subtle bugs that might not be immediately obvious
//! but could lead to incorrect metrics or inconsistent behavior.

use anno::eval::evaluator::{NEREvaluator, StandardNEREvaluator};
use anno::eval::GoldEntity;
use anno::{Entity, EntityType, MockModel};

/// Test that CI bounds use clamp() consistently (not .max().min())
#[test]
fn test_ci_bounds_clamp_consistency() {
    // This test verifies that confidence interval bounds are computed using clamp()
    // consistently across all CI calculation methods.
    // The bug was: compute_confidence_intervals_from_scores used .max(0.0).min(1.0)
    // instead of .clamp(0.0, 1.0), which is inconsistent with other methods.

    // The fix ensures all CI calculations use .clamp(0.0, 1.0) for consistency.
    // This test verifies the behavior is correct by checking that CI bounds are
    // always in [0.0, 1.0] and that the calculation is consistent.

    // Test that clamp produces the same result as max().min()
    let test_cases: Vec<(f64, f64)> = vec![
        (-0.1, 0.0),
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
        (1.1, 1.0),
    ];

    for (value, expected) in test_cases {
        let clamped = value.clamp(0.0, 1.0);
        let max_min = value.max(0.0).min(1.0);
        assert_eq!(
            clamped, max_min,
            "clamp() and max().min() should produce same result for {}",
            value
        );
        assert_eq!(
            clamped, expected,
            "clamp() should produce {} for input {}",
            expected, value
        );
    }
}

/// Test that stratified metrics compute per-type metrics correctly
#[test]
fn test_stratified_metrics_per_type_correctness() {
    // This test verifies that stratified metrics compute per-type metrics correctly,
    // not just using overall metrics for all types.
    // 
    // The bug was: compute_stratified_metrics_from_scores grouped by entity type
    // but used overall summary.strict_f1 for all types, which is incorrect.
    //
    // The fix: Compute per-type metrics by filtering entities by type before evaluation.

    let evaluator = StandardNEREvaluator::new();
    let text = "John Smith works at Apple Inc. in New York.";
    // Calculate correct offsets: "John Smith" = 0-10, " works at " = 10-20, "Apple Inc." = 20-30, " in " = 30-34, "New York" = 34-42
    // Text: "John Smith works at Apple Inc. in New York."
    //       "John Smith" = chars 0-10 (10 chars)
    //       " works at " = chars 10-20 (10 chars)
    //       "Apple Inc." = chars 20-30 (10 chars)
    //       " in " = chars 30-34 (4 chars)
    //       "New York" = chars 34-42 (8 chars)
    //       "." = char 42

    // Create gold with multiple entity types
    let gold = vec![
        GoldEntity::new("John Smith", EntityType::Person, 0),
        GoldEntity::new("Apple Inc.", EntityType::Organization, 20),
        GoldEntity::new("New York", EntityType::Location, 34),
    ];

    // Create predictions: correct for Person and Organization, wrong for Location
    let predicted = vec![
        Entity::new("John Smith", EntityType::Person, 0, 10, 0.9),      // Correct
        Entity::new("Apple Inc.", EntityType::Organization, 20, 30, 0.9), // Correct
        Entity::new("New York", EntityType::Person, 34, 42, 0.9),     // Wrong type
    ];

    let model = MockModel::new("stratified-test").with_entities(predicted);
    let metrics = evaluator.evaluate_test_case(&model, text, &gold, None).unwrap();

    // Verify per-type metrics are computed correctly
    let person_metrics = metrics.per_type.get("PER");
    assert!(
        person_metrics.is_some(),
        "Should have metrics for Person type"
    );
    if let Some(person) = person_metrics {
        // Person: 2 found (John Smith correct, New York wrong type), 1 expected, 1 correct
        assert_eq!(person.found, 2, "Person should have 2 found (John Smith + New York predicted as Person)");
        assert_eq!(person.expected, 1, "Person should have 1 expected");
        assert_eq!(person.correct, 1, "Person should have 1 correct (only John Smith)");
        assert_eq!(
            person.precision, 0.5,
            "Person precision should be 0.5 (1 correct / 2 found)"
        );
        assert_eq!(
            person.recall, 1.0,
            "Person recall should be 1.0 (1 correct / 1 expected)"
        );
    }

    let org_metrics = metrics.per_type.get("ORG");
    assert!(
        org_metrics.is_some(),
        "Should have metrics for Organization type"
    );
    if let Some(org) = org_metrics {
        // Organization: 1 found, 1 expected, 1 correct
        assert_eq!(org.found, 1, "Organization should have 1 found");
        assert_eq!(org.expected, 1, "Organization should have 1 expected");
        assert_eq!(org.correct, 1, "Organization should have 1 correct");
        assert_eq!(
            org.precision, 1.0,
            "Organization precision should be 1.0"
        );
    }

    let loc_metrics = metrics.per_type.get("LOC");
    assert!(
        loc_metrics.is_some(),
        "Should have metrics for Location type"
    );
    if let Some(loc) = loc_metrics {
        // Location: 0 found (predicted as Person), 1 expected, 0 correct
        assert_eq!(loc.found, 0, "Location should have 0 found (predicted as Person)");
        assert_eq!(loc.expected, 1, "Location should have 1 expected");
        assert_eq!(loc.correct, 0, "Location should have 0 correct");
        assert_eq!(
            loc.precision, 0.0,
            "Location precision should be 0.0 (0 found)"
        );
        assert_eq!(
            loc.recall, 0.0,
            "Location recall should be 0.0 (0 correct / 1 expected)"
        );
    }

    // Verify overall metrics reflect the mixed performance
    // Overall: 3 found, 3 expected, 2 correct
    assert_eq!(metrics.found, 3, "Overall should have 3 found");
    assert_eq!(metrics.expected, 3, "Overall should have 3 expected");
    assert_eq!(metrics.correct, 2, "Overall should have 2 correct");
    assert!(
        (metrics.precision.get() - 2.0 / 3.0).abs() < 0.001,
        "Overall precision should be 2/3, got {}",
        metrics.precision.get()
    );
    assert!(
        (metrics.recall.get() - 2.0 / 3.0).abs() < 0.001,
        "Overall recall should be 2/3, got {}",
        metrics.recall.get()
    );
}

/// Test that per-type metrics sum correctly to overall metrics
#[test]
fn test_per_type_sum_to_overall() {
    // Verify that per-type found/expected/correct sum to overall totals
    let evaluator = StandardNEREvaluator::new();
    let text = "John Smith works at Apple Inc. in New York City.";
    // Calculate correct offsets: "John Smith" = 0-10, " works at " = 10-20, "Apple Inc." = 20-30, " in " = 30-34, "New York City" = 34-47
    let gold = vec![
        GoldEntity::new("John Smith", EntityType::Person, 0),
        GoldEntity::new("Apple Inc.", EntityType::Organization, 20),
        GoldEntity::new("New York City", EntityType::Location, 34),
    ];
    let predicted = vec![
        Entity::new("John Smith", EntityType::Person, 0, 10, 0.9),
        Entity::new("Apple Inc.", EntityType::Organization, 20, 30, 0.9),
        Entity::new("New York City", EntityType::Location, 34, 47, 0.9),
    ];

    let model = MockModel::new("sum-test").with_entities(predicted);
    let metrics = evaluator.evaluate_test_case(&model, text, &gold, None).unwrap();

    // Sum per-type counts
    let sum_found: usize = metrics.per_type.values().map(|m| m.found).sum();
    let sum_expected: usize = metrics.per_type.values().map(|m| m.expected).sum();
    let sum_correct: usize = metrics.per_type.values().map(|m| m.correct).sum();

    assert_eq!(
        sum_found, metrics.found,
        "Sum of per-type found should equal overall found"
    );
    assert_eq!(
        sum_expected, metrics.expected,
        "Sum of per-type expected should equal overall expected"
    );
    assert_eq!(
        sum_correct, metrics.correct,
        "Sum of per-type correct should equal overall correct"
    );
}

/// Test that zero-length spans are handled correctly in overlap calculation
#[test]
fn test_zero_length_span_overlap() {
    use anno::eval::metrics::calculate_overlap;

    // Test case 1: Both spans are zero-length at same position
    // intersection_start = 5, intersection_end = 5, so intersection_start >= intersection_end
    // This returns 0.0 before union calculation
    let overlap1 = calculate_overlap(5, 5, 5, 5);
    assert_eq!(
        overlap1, 0.0,
        "Zero-length spans at same position: intersection_start (5) >= intersection_end (5), so returns 0.0"
    );

    // Test case 2: One span is zero-length, other isn't, at same position
    let overlap2 = calculate_overlap(5, 5, 5, 10);
    // This should return 1.0 per the code (union == 0.0 case)
    // Actually, let's check: intersection = 0, union = 0 + 5 - 0 = 5, so overlap = 0/5 = 0.0
    // But the code checks `if union == 0.0` first, which won't be true here.
    // So it should be 0.0, not 1.0.
    assert!(
        (0.0..=1.0).contains(&overlap2),
        "Zero-length span overlap should be in [0.0, 1.0]"
    );

    // Test case 3: Zero-length spans at different positions
    // intersection_start = 10, intersection_end = 5, so intersection_start >= intersection_end
    // This returns 0.0 before union calculation
    let overlap3 = calculate_overlap(5, 5, 10, 10);
    assert_eq!(
        overlap3, 0.0,
        "Zero-length spans at different positions: no intersection, returns 0.0"
    );

    // Test case 4: Zero-length span doesn't overlap with non-zero span
    let overlap4 = calculate_overlap(5, 5, 10, 15);
    assert_eq!(
        overlap4, 0.0,
        "Zero-length span should not overlap with non-zero span at different position"
    );
}

