//! Tests for NER evaluator trait and implementations.

use anno::eval::evaluator::{NEREvaluator, StandardNEREvaluator};
use anno::eval::types::MetricValue;
use anno::eval::GoldEntity;
use anno::{EntityType, PatternNER};

#[test]
fn test_evaluate_test_case_basic() {
    let evaluator = StandardNEREvaluator::new();
    let model = PatternNER::new();

    // Use entities PatternNER can actually detect
    let text = "Meeting on January 15, 2025 for $100";
    let ground_truth = vec![
        GoldEntity::with_span("January 15, 2025", EntityType::Date, 11, 27),
        GoldEntity::with_span("$100", EntityType::Money, 32, 36),
    ];

    let metrics = evaluator
        .evaluate_test_case(&model, text, &ground_truth, None)
        .unwrap();

    assert!(metrics.precision.get() >= 0.0 && metrics.precision.get() <= 1.0);
    assert!(metrics.recall.get() >= 0.0 && metrics.recall.get() <= 1.0);
    assert!(metrics.f1.get() >= 0.0 && metrics.f1.get() <= 1.0);
    assert!(metrics.tokens_per_second >= 0.0);
}

#[test]
fn test_evaluate_test_case_empty_ground_truth() {
    let evaluator = StandardNEREvaluator::new();
    let model = PatternNER::new();

    let text = "This is a test sentence.";
    let ground_truth = vec![];

    let metrics = evaluator
        .evaluate_test_case(&model, text, &ground_truth, None)
        .unwrap();

    assert_eq!(metrics.expected, 0);
}

#[test]
fn test_aggregate_metrics() {
    let evaluator = StandardNEREvaluator::new();
    let model = PatternNER::new();

    let test_cases = vec![
        (
            "Meeting on January 15, 2025",
            vec![GoldEntity::with_span(
                "January 15, 2025",
                EntityType::Date,
                11,
                27,
            )],
        ),
        (
            "Cost: $500",
            vec![GoldEntity::with_span("$500", EntityType::Money, 6, 10)],
        ),
    ];

    let mut query_metrics = Vec::new();
    for (i, (text, ground_truth)) in test_cases.iter().enumerate() {
        let metrics = evaluator
            .evaluate_test_case(&model, text, ground_truth, Some(&format!("tc_{}", i)))
            .unwrap();
        query_metrics.push(metrics);
    }

    let aggregate = evaluator.aggregate(&query_metrics).unwrap();

    assert!(aggregate.precision.get() >= 0.0);
    assert!(aggregate.recall.get() >= 0.0);
    assert!(aggregate.f1.get() >= 0.0);
    assert_eq!(aggregate.num_test_cases, 2);
}

#[test]
fn test_metric_value_bounds() {
    let v = MetricValue::new(0.5);
    assert!((v.get() - 0.5).abs() < 1e-6);

    // Test clamping
    let high = MetricValue::new(1.5);
    assert!((high.get() - 1.0).abs() < 1e-6);

    let low = MetricValue::new(-0.5);
    assert!((low.get() - 0.0).abs() < 1e-6);
}

#[test]
fn test_metric_value_strict() {
    assert!(MetricValue::try_new(0.5).is_ok());
    assert!(MetricValue::try_new(1.1).is_err());
    assert!(MetricValue::try_new(-0.1).is_err());
}

/// Test that micro-averaging is used correctly.
///
/// Scenario: 2 test cases with very different sizes
/// - Case 1: 1 entity, 1 found, 1 correct -> P=100%, R=100%, F1=100%
/// - Case 2: 100 entities, 100 found, 50 correct -> P=50%, R=50%, F1=50%
///
/// Macro average (equal weight per case): (100% + 50%) / 2 = 75%
/// Micro average (total counts): 51/101 = ~50.5%
///
/// The evaluator should report MICRO average (50.5%) as the primary metric,
/// not macro (75%), because a single perfect prediction shouldn't inflate
/// overall metrics disproportionately.
#[test]
fn test_micro_vs_macro_averaging() {
    use anno::eval::evaluator::NERQueryMetrics;
    use std::collections::HashMap;

    let evaluator = StandardNEREvaluator::new();

    // Simulate Case 1: 1 entity, perfect match
    let case1 = NERQueryMetrics {
        text: "Test 1".to_string(),
        test_case_id: Some("tc1".to_string()),
        precision: MetricValue::new(1.0),
        recall: MetricValue::new(1.0),
        f1: MetricValue::new(1.0),
        per_type: HashMap::new(),
        found: 1,
        expected: 1,
        correct: 1,
        tokens_per_second: 1000.0,
    };

    // Simulate Case 2: 100 entities, 50 correct
    let case2 = NERQueryMetrics {
        text: "Test 2".to_string(),
        test_case_id: Some("tc2".to_string()),
        precision: MetricValue::new(0.5),
        recall: MetricValue::new(0.5),
        f1: MetricValue::new(0.5),
        per_type: HashMap::new(),
        found: 100,
        expected: 100,
        correct: 50,
        tokens_per_second: 1000.0,
    };

    let aggregate = evaluator.aggregate(&[case1, case2]).unwrap();

    // Expected micro-average: 51/101 ≈ 0.505
    let expected_micro = 51.0 / 101.0;
    let actual_precision = aggregate.precision.get();
    let actual_recall = aggregate.recall.get();

    // Verify micro-average is used (around 50.5%, not 75%)
    assert!(
        (actual_precision - expected_micro).abs() < 0.01,
        "Expected micro precision ~{:.3}, got {:.3}",
        expected_micro,
        actual_precision
    );
    assert!(
        (actual_recall - expected_micro).abs() < 0.01,
        "Expected micro recall ~{:.3}, got {:.3}",
        expected_micro,
        actual_recall
    );

    // Verify macro-average is also available for comparison
    let expected_macro = 0.75;
    let actual_macro_precision = aggregate.macro_precision.get();

    assert!(
        (actual_macro_precision - expected_macro).abs() < 0.01,
        "Expected macro precision ~{:.3}, got {:.3}",
        expected_macro,
        actual_macro_precision
    );

    // Verify the totals are correct
    assert_eq!(aggregate.total_found, 101);
    assert_eq!(aggregate.total_expected, 101);
    assert_eq!(aggregate.total_correct, 51);
}

/// Test per-entity-type metrics are tracked correctly.
#[test]
fn test_per_type_metrics() {
    let evaluator = StandardNEREvaluator::new();
    let model = PatternNER::new();

    // Text with multiple entity types
    let text = "Meeting on January 15, 2025 costs $500. Email: test@example.com";
    let ground_truth = vec![
        GoldEntity::with_span("January 15, 2025", EntityType::Date, 11, 27),
        GoldEntity::with_span("$500", EntityType::Money, 34, 38),
        GoldEntity::with_span("test@example.com", EntityType::Email, 47, 63),
    ];

    let metrics = evaluator
        .evaluate_test_case(&model, text, &ground_truth, None)
        .unwrap();

    // Should have per-type breakdown
    assert!(
        !metrics.per_type.is_empty(),
        "Should have per-type metrics"
    );

    // Check that we have metrics for the types we expected
    // (PatternNER should detect these types)
    assert!(
        !metrics.per_type.is_empty(),
        "Should have at least one type in per_type metrics"
    );
}

/// Test evaluation with mixed correct/incorrect predictions.
#[test]
fn test_mixed_predictions() {
    let evaluator = StandardNEREvaluator::new();
    let model = PatternNER::new();

    // PatternNER will find $100 but not "John" (requires ML)
    let text = "John paid $100";
    let ground_truth = vec![
        GoldEntity::with_span("John", EntityType::Person, 0, 4),  // Won't be found
        GoldEntity::with_span("$100", EntityType::Money, 10, 14), // Will be found
    ];

    let metrics = evaluator
        .evaluate_test_case(&model, text, &ground_truth, None)
        .unwrap();

    // Should have partial recall (found money but not person)
    assert!(
        metrics.recall.get() > 0.0 && metrics.recall.get() < 1.0,
        "Recall should be partial (0 < R < 1)"
    );
    assert!(metrics.expected == 2, "Should expect 2 entities");
}

/// Test evaluation with all false positives.
#[test]
fn test_false_positives_only() {
    let evaluator = StandardNEREvaluator::new();
    let model = PatternNER::new();

    // PatternNER will find $100 but we don't expect it
    let text = "Payment: $100";
    let ground_truth: Vec<GoldEntity> = vec![]; // Expect nothing

    let metrics = evaluator
        .evaluate_test_case(&model, text, &ground_truth, None)
        .unwrap();

    // Model will find $100 but it's not in gold
    assert_eq!(metrics.expected, 0);
    // Precision should be 0 (false positive)
    assert!(metrics.found > 0, "Should find entities");
    assert_eq!(metrics.precision.get(), 0.0, "Precision should be 0");
}

/// Test evaluation with all false negatives.
#[test]
fn test_false_negatives_only() {
    let evaluator = StandardNEREvaluator::new();
    let model = PatternNER::new();

    // Expect a person entity that PatternNER can't find
    let text = "John Smith is here";
    let ground_truth = vec![GoldEntity::with_span(
        "John Smith",
        EntityType::Person,
        0,
        10,
    )];

    let metrics = evaluator
        .evaluate_test_case(&model, text, &ground_truth, None)
        .unwrap();

    // PatternNER can't find Person entities
    assert_eq!(metrics.expected, 1);
    assert_eq!(metrics.recall.get(), 0.0, "Recall should be 0");
}
