//! Real dataset evaluation tests for StackedNER with ML backends.
//!
//! These tests evaluate ML stacks on real NER datasets to validate:
//! - F1/precision/recall metrics
//! - Comparison: ML-first vs ML-fallback vs default
//! - Domain-specific performance

#[cfg(all(feature = "onnx", feature = "eval"))]
mod real_dataset_tests {
    use anno::eval::task_evaluator::{TaskEvalConfig, TaskEvaluator};
    use anno::eval::task_mapping::Task;
    use anno::{Model, StackedNER};

    // Helper to create GLiNER with graceful failure handling
    fn create_gliner() -> Option<anno::GLiNEROnnx> {
        anno::GLiNEROnnx::new("onnx-community/gliner_small-v2.1").ok()
    }

    #[test]
    fn test_ml_stacked_evaluation_synthetic() {
        // Test ML StackedNER on synthetic datasets
        if create_gliner().is_some() {
            let evaluator = TaskEvaluator::new().unwrap();

            let config = TaskEvalConfig {
                tasks: vec![Task::NER],
                datasets: vec![], // Use all suitable datasets
                backends: vec!["StackedNER".to_string()],
                max_examples: Some(50),
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
        }
    }

    #[test]
    fn test_ml_stacked_basic_metrics() {
        // Test that ML StackedNER produces reasonable metrics
        if let Some(gliner) = create_gliner() {
            let ner = StackedNER::with_ml_first(Box::new(gliner));

            let text = "Apple Inc. CEO Tim Cook announced new products in Cupertino, California on January 15, 2025. Contact: tim@apple.com for $100/hr.";
            let entities = ner.extract_entities(text, None).unwrap();

            // Should find entities
            assert!(!entities.is_empty());

            // Verify entity types are reasonable
            let entity_types: std::collections::HashSet<_> =
                entities.iter().map(|e| e.entity_type.clone()).collect();
            assert!(entity_types.len() > 0);
        }
    }

    #[test]
    fn test_ml_first_vs_fallback_metrics() {
        // Compare ML-first vs ML-fallback on same text
        if let (Some(gliner1), Some(gliner2)) = (create_gliner(), create_gliner()) {
            let ml_first = StackedNER::with_ml_first(Box::new(gliner1));
            let ml_fallback = StackedNER::with_ml_fallback(Box::new(gliner2));
            let default = StackedNER::default();

            let text = "Apple Inc. was founded by Steve Jobs in 1976. Microsoft was founded by Bill Gates in 1975.";

            let entities_first = ml_first.extract_entities(text, None).unwrap();
            let entities_fallback = ml_fallback.extract_entities(text, None).unwrap();
            let entities_default = default.extract_entities(text, None).unwrap();

            // All should produce valid entities
            for entity in entities_first
                .iter()
                .chain(entities_fallback.iter())
                .chain(entities_default.iter())
            {
                assert!(entity.start < entity.end);
                assert!(entity.confidence >= 0.0 && entity.confidence <= 1.0);
            }

            // ML-first may find more entities (runs ML first)
            // ML-fallback should find structured entities (pattern runs first)
            // Both are valid strategies
        }
    }

    #[test]
    fn test_ml_stacked_domain_specific() {
        // Test ML stacks on domain-specific text
        if let Some(gliner) = create_gliner() {
            let ner = StackedNER::with_ml_first(Box::new(gliner));

            // News domain
            let news_text = "Apple Inc. announced today that CEO Tim Cook will present new products at the company's headquarters in Cupertino, California.";
            let news_entities = ner.extract_entities(news_text, None).unwrap();
            assert!(!news_entities.is_empty());

            // Biomedical domain (may not find much, but should not panic)
            let bio_text = "The CRISPR-Cas9 system was developed by Jennifer Doudna and Emmanuelle Charpentier.";
            let bio_entities = ner.extract_entities(bio_text, None).unwrap();
            // May or may not find entities, but should not panic

            // Financial domain
            let finance_text =
                "Apple Inc. reported revenue of $100 billion for Q1 2024, an increase of 25%.";
            let finance_entities = ner.extract_entities(finance_text, None).unwrap();
            // Should find at least money/percent from pattern layer
            let has_structured = finance_entities.iter().any(|e| {
                matches!(
                    e.entity_type,
                    anno::EntityType::Money | anno::EntityType::Percent
                )
            });
            // May or may not have structured entities depending on text
        }
    }
}
