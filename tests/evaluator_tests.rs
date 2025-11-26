//! Tests for NER evaluator trait and implementations.

use anno::eval::evaluator::{NEREvaluator, StandardNEREvaluator};
use anno::eval::types::MetricValue;
use anno::eval::GoldEntity;
use anno::{Entity, EntityType, Model, PatternNER};

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
