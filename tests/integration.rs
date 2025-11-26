//! Integration tests for anno NER evaluation framework.

use anno::eval::{evaluate_ner_model, load_conll2003, GoldEntity};
use anno::{EntityType, Model, PatternNER};
use std::io::Write;

#[test]
fn test_end_to_end_conll_evaluation() {
    // Create a temp CoNLL file
    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join("anno_integration_test.conll");

    let content = "Meeting NNP B-NP O
on IN B-PP O
January NNP B-NP B-DATE
15 CD I-NP I-DATE
, , O I-DATE
2025 CD I-NP I-DATE
for IN B-PP O
$ $ O B-MONEY
100 CD B-NP I-MONEY
. . O O

";
    std::fs::write(&temp_file, content).unwrap();

    // Load and evaluate
    let test_cases = load_conll2003(&temp_file).unwrap();
    assert!(!test_cases.is_empty());

    let model = PatternNER::new();
    let results = evaluate_ner_model(&model, &test_cases).unwrap();

    // Verify results structure
    assert!(results.precision >= 0.0 && results.precision <= 1.0);
    assert!(results.recall >= 0.0 && results.recall <= 1.0);
    assert!(results.f1 >= 0.0 && results.f1 <= 1.0);
    assert!(results.tokens_per_second >= 0.0);

    // Cleanup
    std::fs::remove_file(&temp_file).ok();
}

#[test]
fn test_synthetic_dataset_evaluation() {
    use anno::eval::synthetic::{all_datasets, datasets_by_domain, Domain};

    // Get all synthetic datasets
    let all = all_datasets();
    assert!(!all.is_empty(), "Should have synthetic datasets");

    // Filter by domain
    let news = datasets_by_domain(Domain::News);
    assert!(!news.is_empty(), "Should have news datasets");

    // Convert to test cases and evaluate
    let model = PatternNER::new();

    // PatternNER only detects DATE/MONEY/PERCENT, so we need test cases with those
    let test_cases: Vec<(String, Vec<GoldEntity>)> = vec![
        (
            "Cost is $500 on January 15, 2025".to_string(),
            vec![
                GoldEntity::with_span("$500", EntityType::Money, 8, 12),
                GoldEntity::with_span("January 15, 2025", EntityType::Date, 16, 32),
            ],
        ),
        (
            "Growth of 25% in Q1 2024".to_string(),
            vec![GoldEntity::with_span("25%", EntityType::Percent, 10, 13)],
        ),
    ];

    let results = evaluate_ner_model(&model, &test_cases).unwrap();

    // PatternNER should do well on DATE/MONEY/PERCENT
    assert!(results.f1 > 0.0, "PatternNER should find some entities");
}

#[test]
fn test_metrics_serialization() {
    let model = PatternNER::new();
    let test_cases = vec![(
        "Meeting on January 15, 2025".to_string(),
        vec![GoldEntity::with_span(
            "January 15, 2025",
            EntityType::Date,
            11,
            27,
        )],
    )];

    let results = evaluate_ner_model(&model, &test_cases).unwrap();

    // Should be serializable
    let json = serde_json::to_string(&results).unwrap();
    assert!(json.contains("precision"));
    assert!(json.contains("recall"));
    assert!(json.contains("f1"));

    // Should be deserializable
    let _: anno::eval::NEREvaluationResults = serde_json::from_str(&json).unwrap();
}

#[test]
fn test_per_type_metrics() {
    let model = PatternNER::new();
    let test_cases = vec![
        (
            "Cost: $100".to_string(),
            vec![GoldEntity::with_span("$100", EntityType::Money, 6, 10)],
        ),
        (
            "Date: 2024-01-15".to_string(),
            vec![GoldEntity::with_span("2024-01-15", EntityType::Date, 6, 16)],
        ),
    ];

    let results = evaluate_ner_model(&model, &test_cases).unwrap();

    // Should have per-type breakdown
    // Note: PatternNER recognizes these, so we should have metrics
    assert!(!results.per_type.is_empty() || results.expected == 0);
}
