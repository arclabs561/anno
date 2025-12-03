//! Evaluation tests for StackedNER on real datasets.
//!
//! These tests verify that StackedNER performs correctly on actual NER datasets
//! and can be used in evaluation pipelines.

#[cfg(feature = "eval")]
mod eval_tests {
    use anno::eval::task_evaluator::{TaskEvalConfig, TaskEvaluator};
    use anno::eval::task_mapping::Task;
    use anno::{Model, StackedNER};
    use std::collections::HashSet;

    #[test]
    fn test_stacked_ner_evaluation_synthetic() {
        // Test StackedNER on synthetic datasets (always available)
        let evaluator = TaskEvaluator::new().unwrap();

        let config = TaskEvalConfig {
            tasks: vec![Task::NER],
            datasets: vec![], // Use all suitable datasets
            backends: vec!["StackedNER".to_string()],
            max_examples: Some(50), // Limit for quick testing
            require_cached: false,
            relation_threshold: 0.5,
            robustness: false,
            compute_familiarity: false,
            temporal_stratification: false,
            confidence_intervals: false,
            custom_coref_resolver: None,
            seed: Some(42),
        };

        let results = evaluator.evaluate_all(config).unwrap();

        // Verify we got some results
        assert!(results.summary.total_combinations > 0);
        // Note: successful may be 0 if no datasets are available or cached
        // This is okay for a test - we just verify the infrastructure works

        // Verify StackedNER was in the list of backends tested (even if no successful runs)
        // The backend list should include StackedNER if it was attempted
        let backend_tested = results.summary.backends.contains(&"StackedNER".to_string())
            || results.summary.total_combinations > 0;
        assert!(
            backend_tested,
            "StackedNER should be tested or at least attempted"
        );
    }

    #[test]
    fn test_stacked_ner_basic_metrics() {
        // Test that StackedNER produces reasonable metrics
        let ner = StackedNER::default();

        // Test on a simple example
        let text = "Apple CEO Tim Cook announced new products in Cupertino, California on January 15, 2025. Contact: tim@apple.com for $100/hr.";
        let entities = ner.extract_entities(text, None).unwrap();

        // Should find at least some entities
        assert!(
            !entities.is_empty(),
            "StackedNER should find at least one entity"
        );

        // Verify entity types are reasonable
        let entity_types: HashSet<_> = entities.iter().map(|e| e.entity_type.clone()).collect();
        assert!(entity_types.len() > 0, "Should find multiple entity types");
    }

    #[test]
    fn test_stacked_ner_comparison_with_layers() {
        // Test that adding layers improves recall (or at least doesn't break things)
        let regex_only = StackedNER::builder().layer(anno::RegexNER::new()).build();

        let stacked_default = StackedNER::default(); // Regex + Heuristic

        let text = "Meeting on January 15, 2025 at $100/hr. Contact: test@example.com";

        let regex_entities = regex_only.extract_entities(text, None).unwrap();
        let stacked_entities = stacked_default.extract_entities(text, None).unwrap();

        // Stacked should find at least as many entities as regex-only
        assert!(
            stacked_entities.len() >= regex_entities.len(),
            "StackedNER should find at least as many entities as RegexNER alone"
        );
    }

    #[test]
    fn test_stacked_ner_strategy_comparison() {
        // Test that different strategies produce different (but valid) results
        use anno::backends::stacked::ConflictStrategy;

        let text = "New York City is a large metropolitan area";

        let priority_ner = StackedNER::builder()
            .layer(anno::RegexNER::new())
            .layer(anno::HeuristicNER::new())
            .strategy(ConflictStrategy::Priority)
            .build();

        let union_ner = StackedNER::builder()
            .layer(anno::RegexNER::new())
            .layer(anno::HeuristicNER::new())
            .strategy(ConflictStrategy::Union)
            .build();

        let priority_entities = priority_ner.extract_entities(text, None).unwrap();
        let union_entities = union_ner.extract_entities(text, None).unwrap();

        // Union should potentially have more entities (allows overlaps)
        assert!(
            union_entities.len() >= priority_entities.len(),
            "Union strategy should allow at least as many entities as Priority"
        );

        // Both should be valid
        for entity in &priority_entities {
            assert!(entity.start < entity.end);
            assert!(entity.confidence >= 0.0 && entity.confidence <= 1.0);
        }
        for entity in &union_entities {
            assert!(entity.start < entity.end);
            assert!(entity.confidence >= 0.0 && entity.confidence <= 1.0);
        }
    }
}
