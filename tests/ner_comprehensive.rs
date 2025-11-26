//! Comprehensive tests for NER evaluation framework.
//!
//! Tests cover:
//! - Metrics calculation (precision, recall, F1)
//! - Partial match metrics
//! - Confidence threshold analysis
//! - Edge cases

use anno::eval::metrics::{analyze_confidence_thresholds, calculate_partial_match_metrics};
use anno::eval::GoldEntity;
use anno::{Entity, EntityType, Model, PatternNER};

#[test]
fn test_partial_match_metrics_exact() {
    let predicted = vec![Entity {
        text: "January 15, 2025".to_string(),
        entity_type: EntityType::Date,
        start: 0,
        end: 16,
        confidence: 0.9,
    }];

    let ground_truth = vec![GoldEntity::with_span(
        "January 15, 2025",
        EntityType::Date,
        0,
        16,
    )];

    let metrics = calculate_partial_match_metrics(&predicted, &ground_truth, 0.5);
    assert!((metrics.precision - 1.0).abs() < 0.001);
    assert!((metrics.recall - 1.0).abs() < 0.001);
    assert!((metrics.f1 - 1.0).abs() < 0.001);
}

#[test]
fn test_partial_match_metrics_overlap() {
    let predicted = vec![Entity {
        text: "January 15".to_string(),
        entity_type: EntityType::Date,
        start: 0,
        end: 10,
        confidence: 0.9,
    }];

    let ground_truth = vec![GoldEntity::with_span(
        "January 15, 2025",
        EntityType::Date,
        0,
        16,
    )];

    let metrics = calculate_partial_match_metrics(&predicted, &ground_truth, 0.3);
    // Should have some overlap
    assert!(metrics.precision > 0.0);
    assert!(metrics.recall > 0.0);
}

#[test]
fn test_confidence_threshold_analysis() {
    let predicted = vec![
        Entity {
            text: "$100".to_string(),
            entity_type: EntityType::Money,
            start: 0,
            end: 4,
            confidence: 0.9,
        },
        Entity {
            text: "$50".to_string(),
            entity_type: EntityType::Money,
            start: 10,
            end: 13,
            confidence: 0.3,
        },
    ];

    let ground_truth = vec![GoldEntity::with_span("$100", EntityType::Money, 0, 4)];

    let analysis = analyze_confidence_thresholds(&predicted, &ground_truth, 0.5);
    assert_eq!(analysis.thresholds.len(), 11); // 0.0 to 1.0 in 0.1 steps
    assert!(analysis.metrics_at_threshold.len() == 11);
}

#[test]
fn test_pattern_ner_dates() {
    let ner = PatternNER::new();
    let entities = ner
        .extract_entities("Meeting on 2024-01-15 and January 20, 2024", None)
        .unwrap();

    assert!(entities.len() >= 2);
    for e in &entities {
        assert_eq!(e.entity_type, EntityType::Date);
    }
}

#[test]
fn test_pattern_ner_money() {
    let ner = PatternNER::new();
    let entities = ner
        .extract_entities("Cost is $100 or 50 EUR", None)
        .unwrap();

    // Should find $100 at minimum
    assert!(!entities.is_empty());
    assert!(entities.iter().any(|e| e.entity_type == EntityType::Money));
}

#[test]
fn test_pattern_ner_percent() {
    let ner = PatternNER::new();
    let entities = ner
        .extract_entities("Growth of 15% and 20 percent", None)
        .unwrap();

    assert!(!entities.is_empty());
    assert!(entities
        .iter()
        .any(|e| e.entity_type == EntityType::Percent));
}
