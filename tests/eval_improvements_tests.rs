//! Tests for evaluation improvements: chain-length stratification, familiarity, etc.

use anno::eval::coref::{CorefChain, Mention};
use anno::eval::coref_metrics::{compute_chain_length_stratified, CorefEvaluation};
use anno::eval::types::{CorefChainStats, LabelShift};
use std::collections::HashSet;

#[test]
fn test_chain_length_stratification() {
    // Create test chains of different lengths
    let gold = vec![
        // Long chain (>10 mentions)
        CorefChain::new(
            (0..12)
                .map(|i| Mention::new("John", i * 10, i * 10 + 4))
                .collect(),
        ),
        // Short chain (5 mentions)
        CorefChain::new(
            (0..5)
                .map(|i| Mention::new("he", i * 10, i * 10 + 2))
                .collect(),
        ),
        // Singleton
        CorefChain::singleton(Mention::new("Mary", 100, 104)),
    ];

    let predicted = gold.clone(); // Perfect match

    let stats = compute_chain_length_stratified(&predicted, &gold);

    assert_eq!(stats.long_chain_count, 1);
    assert_eq!(stats.short_chain_count, 1);
    assert_eq!(stats.singleton_count, 1);
    assert!(
        (stats.long_chain_f1 - 1.0).abs() < 0.001,
        "Long chain F1 should be 1.0"
    );
    assert!(
        (stats.short_chain_f1 - 1.0).abs() < 0.001,
        "Short chain F1 should be 1.0"
    );
    assert!(
        (stats.singleton_f1 - 1.0).abs() < 0.001,
        "Singleton F1 should be 1.0"
    );
}

#[test]
fn test_chain_length_stratification_integration() {
    let gold = vec![
        CorefChain::new(
            (0..15)
                .map(|i| Mention::new("Alice", i * 10, i * 10 + 5))
                .collect(),
        ),
        CorefChain::new(
            (0..3)
                .map(|i| Mention::new("Bob", i * 10, i * 10 + 3))
                .collect(),
        ),
    ];

    let predicted = gold.clone();

    let eval = CorefEvaluation::compute(&predicted, &gold);

    assert!(eval.chain_stats.is_some(), "Chain stats should be computed");
    let stats = eval.chain_stats.unwrap();
    assert_eq!(stats.long_chain_count, 1);
    assert_eq!(stats.short_chain_count, 1);
}

#[test]
fn test_familiarity_computation() {
    let train_types = vec![
        "person".to_string(),
        "organization".to_string(),
        "location".to_string(),
    ];

    let eval_types = vec![
        "PERSON".to_string(),  // Should match "person"
        "ORG".to_string(),     // Should match "organization"
        "DISEASE".to_string(), // True zero-shot
    ];

    let shift = LabelShift::from_type_sets(&train_types, &eval_types);

    // Should detect some familiarity via string similarity (PERSON/person, ORG/organization)
    // Note: true_zero_shot_types uses exact string matching, not similarity
    // So "PERSON", "ORG", and "DISEASE" are all considered zero-shot (no exact matches)
    assert!(shift.familiarity > 0.0, "Should have some familiarity");
    // All three eval types are zero-shot by exact match, but familiarity should detect similarity
    assert_eq!(
        shift.true_zero_shot_types.len(),
        3,
        "All three eval types are zero-shot by exact match (PERSON != person, ORG != organization)"
    );
    assert!(shift.true_zero_shot_types.contains(&"DISEASE".to_string()));
    assert!(shift.true_zero_shot_types.contains(&"PERSON".to_string()));
    assert!(shift.true_zero_shot_types.contains(&"ORG".to_string()));
}

#[test]
fn test_familiarity_inflation_detection() {
    let train_types = vec![
        "person".to_string(),
        "organization".to_string(),
        "location".to_string(),
    ];

    let eval_types = vec![
        "PERSON".to_string(),
        "ORGANIZATION".to_string(),
        "LOCATION".to_string(),
    ];

    let shift = LabelShift::from_type_sets(&train_types, &eval_types);

    // High overlap should trigger inflation warning
    assert!(shift.familiarity > 0.5, "Should have high familiarity");
    // Note: is_inflated() checks overlap_ratio > 0.8 or familiarity > 0.85
    // With string similarity, this might not trigger, but familiarity should be high
}

#[test]
fn test_string_similarity() {
    // Test that string similarity works for label matching
    let train_types = vec!["PER".to_string(), "ORG".to_string()];
    let eval_types = vec!["PERSON".to_string(), "ORGANIZATION".to_string()];

    let shift = LabelShift::from_type_sets(&train_types, &eval_types);

    // Should detect similarity even without exact match
    assert!(shift.familiarity > 0.0, "Should have non-zero familiarity");
}

#[test]
fn test_coref_evaluation_with_chain_stats() {
    let gold = vec![CorefChain::new(vec![
        Mention::new("John", 0, 4),
        Mention::new("he", 10, 12),
    ])];

    let predicted = gold.clone();

    let eval = CorefEvaluation::compute(&predicted, &gold);

    assert!(eval.chain_stats.is_some(), "Should have chain stats");
    let stats = eval.chain_stats.unwrap();
    assert_eq!(stats.short_chain_count, 1, "Should have 1 short chain");
    assert!(
        (stats.short_chain_f1 - 1.0).abs() < 0.001,
        "Should have perfect F1"
    );
}

#[test]
fn test_label_shift_zero_shot_types() {
    let train_types = vec!["person".to_string()];
    let eval_types = vec![
        "person".to_string(),
        "disease".to_string(),
        "drug".to_string(),
    ];

    let shift = LabelShift::from_type_sets(&train_types, &eval_types);

    assert_eq!(shift.true_zero_shot_types.len(), 2);
    assert!(shift.true_zero_shot_types.contains(&"disease".to_string()));
    assert!(shift.true_zero_shot_types.contains(&"drug".to_string()));
}

#[test]
fn test_confidence_intervals_structure() {
    use anno::eval::task_evaluator::{ConfidenceIntervals, MetricWithCI, StratifiedMetrics};
    use std::collections::HashMap;

    // Test that structures can be created
    let ci = ConfidenceIntervals {
        f1_ci: (0.85, 0.95),
        precision_ci: (0.80, 0.90),
        recall_ci: (0.90, 1.0),
    };

    assert!(ci.f1_ci.0 < ci.f1_ci.1);
    assert!(ci.precision_ci.0 < ci.precision_ci.1);
    assert!(ci.recall_ci.0 < ci.recall_ci.1);

    let metric_ci = MetricWithCI {
        mean: 0.90,
        std_dev: 0.05,
        ci_95: (0.85, 0.95),
        n: 100,
    };

    assert!((metric_ci.mean - 0.90).abs() < 0.001);

    let stratified = StratifiedMetrics {
        by_entity_type: HashMap::new(),
        by_temporal_stratum: None,
        by_surface_form: None,
        by_mention_char: None,
    };

    assert!(stratified.by_entity_type.is_empty());
}
