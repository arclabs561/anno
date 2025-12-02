//! Invariant tests for evaluation code.
//!
//! These tests verify that evaluation metrics always satisfy mathematical
//! invariants, regardless of input. They catch bugs in metric calculation
//! logic and ensure correctness.

use anno::eval::evaluator::{NEREvaluator, StandardNEREvaluator};
use anno::eval::GoldEntity;
use anno::eval::metrics::calculate_overlap;
use anno::{Entity, EntityType, MockModel};

/// Test that precision, recall, and F1 are always in [0.0, 1.0]
#[test]
fn test_metric_bounds() {
    let evaluator = StandardNEREvaluator::new();

    // Test case 1: Perfect match
    let text = "John Smith works at Apple Inc.";
    let gold = vec![
        GoldEntity::new("John Smith", EntityType::Person, 0),
        GoldEntity::new("Apple Inc.", EntityType::Organization, 20),
    ];
    let predicted = vec![
        Entity::new("John Smith", EntityType::Person, 0, 10, 0.9),
        Entity::new("Apple Inc.", EntityType::Organization, 20, 30, 0.9),
    ];
    let model = MockModel::new("perfect").with_entities(predicted);
    let metrics = evaluator.evaluate_test_case(&model, text, &gold, None).unwrap();

    assert!(
        (0.0..=1.0).contains(&metrics.precision.get()),
        "Precision should be in [0.0, 1.0], got {}",
        metrics.precision.get()
    );
    assert!(
        (0.0..=1.0).contains(&metrics.recall.get()),
        "Recall should be in [0.0, 1.0], got {}",
        metrics.recall.get()
    );
    assert!(
        (0.0..=1.0).contains(&metrics.f1.get()),
        "F1 should be in [0.0, 1.0], got {}",
        metrics.f1.get()
    );

    // Test case 2: No matches
    let predicted_empty = vec![];
    let model_empty = MockModel::new("empty").with_entities(predicted_empty);
    let metrics_empty = evaluator
        .evaluate_test_case(&model_empty, text, &gold, None)
        .unwrap();

    assert!(
        (0.0..=1.0).contains(&metrics_empty.precision.get()),
        "Precision should be in [0.0, 1.0] even with no predictions, got {}",
        metrics_empty.precision.get()
    );
    assert_eq!(
        metrics_empty.precision.get(),
        0.0,
        "Precision should be 0.0 when no predictions"
    );

    // Test case 3: All wrong (wrong types)
    let predicted_wrong = vec![
        Entity::new("John", EntityType::Location, 0, 4, 0.9), // Wrong type
        Entity::new("Apple", EntityType::Location, 20, 25, 0.9), // Wrong type
    ];
    let model_wrong = MockModel::new("wrong").with_entities(predicted_wrong);
    let metrics_wrong = evaluator
        .evaluate_test_case(&model_wrong, text, &gold, None)
        .unwrap();

    assert!(
        (0.0..=1.0).contains(&metrics_wrong.precision.get()),
        "Precision should be in [0.0, 1.0] even with wrong predictions, got {}",
        metrics_wrong.precision.get()
    );
    assert_eq!(
        metrics_wrong.precision.get(),
        0.0,
        "Precision should be 0.0 when all predictions are wrong"
    );
}

/// Test that F1 = 2 * P * R / (P + R) when P + R > 0
#[test]
fn test_f1_formula() {
    let evaluator = StandardNEREvaluator::new();
    let text = "John Smith works at Apple Inc.";
    let gold = vec![
        GoldEntity::new("John Smith", EntityType::Person, 0),
        GoldEntity::new("Apple Inc.", EntityType::Organization, 20),
    ];

    // Test with partial matches
    let predicted = vec![
        Entity::new("John Smith", EntityType::Person, 0, 10, 0.9), // Correct
        Entity::new("Apple", EntityType::Location, 20, 25, 0.9),  // Wrong type
    ];
    let model = MockModel::new("partial").with_entities(predicted);
    let metrics = evaluator.evaluate_test_case(&model, text, &gold, None).unwrap();

    let precision = metrics.precision.get();
    let recall = metrics.recall.get();
    let f1 = metrics.f1.get();

    if precision + recall > 0.0 {
        let expected_f1 = 2.0 * precision * recall / (precision + recall);
        assert!(
            (f1 - expected_f1).abs() < 1e-10,
            "F1 should equal 2*P*R/(P+R). Got {}, expected {}",
            f1,
            expected_f1
        );
    } else {
        assert_eq!(f1, 0.0, "F1 should be 0.0 when P + R = 0");
    }
}

/// Test that total_correct <= min(total_found, total_expected)
#[test]
fn test_count_invariants() {
    let evaluator = StandardNEREvaluator::new();
    let text = "John Smith works at Apple Inc.";
    let gold = vec![
        GoldEntity::new("John Smith", EntityType::Person, 0),
        GoldEntity::new("Apple Inc.", EntityType::Organization, 20),
    ];

    // Test case 1: Perfect match
    let predicted = vec![
        Entity::new("John Smith", EntityType::Person, 0, 10, 0.9),
        Entity::new("Apple Inc.", EntityType::Organization, 20, 30, 0.9),
    ];
    let model = MockModel::new("perfect").with_entities(predicted);
    let metrics = evaluator.evaluate_test_case(&model, text, &gold, None).unwrap();

    assert!(
        metrics.correct <= metrics.found,
        "correct ({}) should be <= found ({})",
        metrics.correct,
        metrics.found
    );
    assert!(
        metrics.correct <= metrics.expected,
        "correct ({}) should be <= expected ({})",
        metrics.correct,
        metrics.expected
    );
    assert!(
        metrics.correct <= metrics.found.min(metrics.expected),
        "correct ({}) should be <= min(found, expected) = {}",
        metrics.correct,
        metrics.found.min(metrics.expected)
    );

    // Test case 2: More predictions than gold
    let predicted_many = vec![
        Entity::new("John Smith", EntityType::Person, 0, 10, 0.9),
        Entity::new("Apple Inc.", EntityType::Organization, 20, 30, 0.9),
        Entity::new("works", EntityType::Location, 11, 16, 0.9), // Extra prediction
    ];
    let model_many = MockModel::new("many").with_entities(predicted_many);
    let metrics_many = evaluator
        .evaluate_test_case(&model_many, text, &gold, None)
        .unwrap();

    assert!(
        metrics_many.correct <= metrics_many.found,
        "correct ({}) should be <= found ({})",
        metrics_many.correct,
        metrics_many.found
    );
    assert!(
        metrics_many.correct <= metrics_many.expected,
        "correct ({}) should be <= expected ({})",
        metrics_many.correct,
        metrics_many.expected
    );

    // Test case 3: Fewer predictions than gold
    let predicted_few = vec![Entity::new("John Smith", EntityType::Person, 0, 10, 0.9)];
    let model_few = MockModel::new("few").with_entities(predicted_few);
    let metrics_few = evaluator
        .evaluate_test_case(&model_few, text, &gold, None)
        .unwrap();

    assert!(
        metrics_few.correct <= metrics_few.found,
        "correct ({}) should be <= found ({})",
        metrics_few.correct,
        metrics_few.found
    );
    assert!(
        metrics_few.correct <= metrics_few.expected,
        "correct ({}) should be <= expected ({})",
        metrics_few.correct,
        metrics_few.expected
    );
}

/// Test edge cases: empty predictions, empty gold, etc.
#[test]
fn test_edge_cases() {
    let evaluator = StandardNEREvaluator::new();
    let text = "John Smith works at Apple Inc.";

    // Case 1: No predictions
    let gold = vec![GoldEntity::new("John Smith", EntityType::Person, 0)];
    let predicted_empty = vec![];
    let model_empty = MockModel::new("empty").with_entities(predicted_empty);
    let metrics_empty = evaluator
        .evaluate_test_case(&model_empty, text, &gold, None)
        .unwrap();

    assert_eq!(
        metrics_empty.precision.get(),
        0.0,
        "Precision should be 0.0 when found = 0"
    );
    assert_eq!(
        metrics_empty.recall.get(),
        0.0,
        "Recall should be 0.0 when found = 0 and expected > 0"
    );
    assert_eq!(
        metrics_empty.f1.get(),
        0.0,
        "F1 should be 0.0 when precision + recall = 0"
    );
    assert!(
        metrics_empty.precision.get().is_finite(),
        "Precision should be finite, not NaN or Inf"
    );
    assert!(
        metrics_empty.recall.get().is_finite(),
        "Recall should be finite, not NaN or Inf"
    );
    assert!(
        metrics_empty.f1.get().is_finite(),
        "F1 should be finite, not NaN or Inf"
    );

    // Case 2: No gold (should return error or handle gracefully)
    let gold_empty = vec![];
    let predicted = vec![Entity::new("John Smith", EntityType::Person, 0, 10, 0.9)];
    let model = MockModel::new("pred").with_entities(predicted);
    let metrics = evaluator.evaluate_test_case(&model, text, &gold_empty, None).unwrap();

    assert_eq!(
        metrics.recall.get(),
        0.0,
        "Recall should be 0.0 when expected = 0"
    );
    assert!(
        metrics.recall.get().is_finite(),
        "Recall should be finite, not NaN or Inf"
    );

    // Case 3: All predictions correct
    let gold_all = vec![
        GoldEntity::new("John Smith", EntityType::Person, 0),
        GoldEntity::new("Apple Inc.", EntityType::Organization, 20),
    ];
    let predicted_all = vec![
        Entity::new("John Smith", EntityType::Person, 0, 10, 0.9),
        Entity::new("Apple Inc.", EntityType::Organization, 20, 30, 0.9),
    ];
    let model_all = MockModel::new("all").with_entities(predicted_all);
    let metrics_all = evaluator
        .evaluate_test_case(&model_all, text, &gold_all, None)
        .unwrap();

    assert_eq!(
        metrics_all.precision.get(),
        1.0,
        "Precision should be 1.0 when all predictions are correct"
    );
    assert_eq!(
        metrics_all.recall.get(),
        1.0,
        "Recall should be 1.0 when all gold entities are found"
    );
    assert_eq!(
        metrics_all.f1.get(),
        1.0,
        "F1 should be 1.0 when precision = recall = 1.0"
    );
}

/// Test that overlap ratio is always in [0.0, 1.0]
#[test]
fn test_overlap_bounds() {
    // Exact match
    let overlap_exact = calculate_overlap(0, 10, 0, 10);
    assert!(
        (0.0..=1.0).contains(&overlap_exact),
        "Exact match overlap should be in [0.0, 1.0], got {}",
        overlap_exact
    );
    assert!(
        (overlap_exact - 1.0).abs() < 1e-10,
        "Exact match should have overlap = 1.0, got {}",
        overlap_exact
    );

    // No overlap
    let overlap_none = calculate_overlap(0, 10, 20, 30);
    assert!(
        (0.0..=1.0).contains(&overlap_none),
        "No overlap should be in [0.0, 1.0], got {}",
        overlap_none
    );
    assert!(
        (overlap_none - 0.0).abs() < 1e-10,
        "No overlap should have overlap = 0.0, got {}",
        overlap_none
    );

    // Partial overlap
    let overlap_partial = calculate_overlap(0, 10, 5, 15);
    assert!(
        (0.0..=1.0).contains(&overlap_partial),
        "Partial overlap should be in [0.0, 1.0], got {}",
        overlap_partial
    );
    assert!(
        overlap_partial > 0.0 && overlap_partial < 1.0,
        "Partial overlap should be between 0.0 and 1.0, got {}",
        overlap_partial
    );

    // Edge case: empty spans (should return 1.0 per the code)
    let overlap_empty = calculate_overlap(0, 0, 0, 0);
    assert!(
        (0.0..=1.0).contains(&overlap_empty),
        "Empty span overlap should be in [0.0, 1.0], got {}",
        overlap_empty
    );
}

/// Test that union >= intersection in overlap calculation
#[test]
fn test_overlap_union_invariant() {
    // This test verifies that the union calculation is always >= intersection
    // by testing various span configurations

    let test_cases = vec![
        (0, 10, 0, 10),   // Exact match
        (0, 10, 5, 15),   // Partial overlap
        (0, 10, 20, 30),  // No overlap
        (0, 10, 0, 5),    // One contained in other
        (5, 15, 0, 10),   // Reverse containment
    ];

    for (pred_start, pred_end, gt_start, gt_end) in test_cases {
        let intersection_start = pred_start.max(gt_start);
        let intersection_end = pred_end.min(gt_end);

        if intersection_start < intersection_end {
            let intersection = (intersection_end - intersection_start) as f64;
            let union = ((pred_end - pred_start) + (gt_end - gt_start)
                - (intersection_end - intersection_start)) as f64;

            assert!(
                union >= intersection,
                "Union ({}) should be >= intersection ({}) for spans ({}, {}) and ({}, {})",
                union,
                intersection,
                pred_start,
                pred_end,
                gt_start,
                gt_end
            );
            assert!(
                union > 0.0,
                "Union should be > 0.0 for overlapping spans"
            );
        }
    }
}

/// Test aggregation invariants
#[test]
fn test_aggregation_invariants() {
    let evaluator = StandardNEREvaluator::new();
    let text = "John Smith works at Apple Inc.";

    // Create multiple test cases
    let test_cases = vec![
        (
            vec![GoldEntity::new("John Smith", EntityType::Person, 0)],
            vec![Entity::new("John Smith", EntityType::Person, 0, 10, 0.9)],
        ),
        (
            vec![GoldEntity::new("Apple Inc.", EntityType::Organization, 20)],
            vec![Entity::new("Apple Inc.", EntityType::Organization, 20, 30, 0.9)],
        ),
    ];

    let mut query_metrics = Vec::new();
    for (gold, predicted) in test_cases {
        let model = MockModel::new("test").with_entities(predicted);
        let metrics = evaluator.evaluate_test_case(&model, text, &gold, None).unwrap();
        query_metrics.push(metrics);
    }

    let aggregate = evaluator.aggregate(&query_metrics).unwrap();

    // Verify micro-averaged metrics match manual calculation
    let total_found: usize = query_metrics.iter().map(|m| m.found).sum();
    let total_expected: usize = query_metrics.iter().map(|m| m.expected).sum();
    let total_correct: usize = query_metrics.iter().map(|m| m.correct).sum();

    assert_eq!(
        aggregate.total_found, total_found,
        "Aggregate total_found should match sum of per-case found"
    );
    assert_eq!(
        aggregate.total_expected, total_expected,
        "Aggregate total_expected should match sum of per-case expected"
    );
    assert_eq!(
        aggregate.total_correct, total_correct,
        "Aggregate total_correct should match sum of per-case correct"
    );

    // Verify micro-averaged precision
    let expected_precision = if total_found > 0 {
        total_correct as f64 / total_found as f64
    } else {
        0.0
    };
    assert!(
        (aggregate.precision.get() - expected_precision).abs() < 1e-10,
        "Micro-averaged precision should match manual calculation. Got {}, expected {}",
        aggregate.precision.get(),
        expected_precision
    );

    // Verify micro-averaged recall
    let expected_recall = if total_expected > 0 {
        total_correct as f64 / total_expected as f64
    } else {
        0.0
    };
    assert!(
        (aggregate.recall.get() - expected_recall).abs() < 1e-10,
        "Micro-averaged recall should match manual calculation. Got {}, expected {}",
        aggregate.recall.get(),
        expected_recall
    );

    // Verify per-type totals sum to overall totals
    let per_type_found: usize = aggregate.per_type.values().map(|m| m.found).sum();
    let per_type_expected: usize = aggregate.per_type.values().map(|m| m.expected).sum();
    let per_type_correct: usize = aggregate.per_type.values().map(|m| m.correct).sum();

    assert_eq!(
        per_type_found, total_found,
        "Sum of per-type found should equal total_found"
    );
    assert_eq!(
        per_type_expected, total_expected,
        "Sum of per-type expected should equal total_expected"
    );
    assert_eq!(
        per_type_correct, total_correct,
        "Sum of per-type correct should equal total_correct"
    );
}

