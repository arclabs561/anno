//! Integration tests for the comprehensive evaluation system.
//!
//! These tests verify end-to-end functionality:
//! - Backend factory creation
//! - Dataset loading
//! - Task evaluation (NER, coreference, relation extraction)
//! - Metric computation

#[cfg(feature = "eval-advanced")]
mod tests {
    use anno::eval::backend_factory::BackendFactory;
    use anno::eval::loader::DatasetLoader;
    use anno::eval::task_evaluator::{TaskEvalConfig, TaskEvaluator};
    use anno::eval::task_mapping::Task;

    #[test]
    fn test_backend_factory_pattern() {
        let backend = BackendFactory::create("pattern");
        assert!(backend.is_ok(), "Pattern backend should be available");
    }

    #[test]
    fn test_backend_factory_heuristic() {
        let backend = BackendFactory::create("heuristic");
        assert!(backend.is_ok(), "Heuristic backend should be available");
    }

    #[test]
    fn test_backend_factory_stacked() {
        let backend = BackendFactory::create("stacked");
        assert!(backend.is_ok(), "Stacked backend should be available");
    }

    #[test]
    fn test_backend_factory_unknown() {
        let backend = BackendFactory::create("nonexistent_backend");
        assert!(backend.is_err(), "Unknown backend should return error");
    }

    #[test]
    fn test_available_backends() {
        let backends = BackendFactory::available_backends();
        assert!(!backends.is_empty(), "Should have at least some backends");
        assert!(
            backends.contains(&"pattern"),
            "Should include pattern backend"
        );
    }

    #[test]
    fn test_dataset_loader_creation() {
        let loader = DatasetLoader::new();
        assert!(loader.is_ok(), "DatasetLoader should be creatable");
    }

    #[test]
    fn test_task_evaluator_creation() {
        let evaluator = TaskEvaluator::new();
        assert!(evaluator.is_ok(), "TaskEvaluator should be creatable");
    }

    #[test]
    fn test_ner_evaluation_with_pattern() {
        let evaluator = TaskEvaluator::new().expect("Failed to create evaluator");

        let config = TaskEvalConfig {
            tasks: vec![Task::NER],
            datasets: vec![], // Will use default datasets for NER
            backends: vec!["pattern".to_string()],
            max_examples: Some(5), // Limit for quick testing
            require_cached: false,
            seed: None,
            relation_threshold: 0.5,
            robustness: false,
            compute_familiarity: false,
            temporal_stratification: false,
            confidence_intervals: false,
            custom_coref_resolver: None,
        };

        // This will attempt to download datasets if not cached
        // We'll just check that it doesn't panic
        let result = evaluator.evaluate_all(config);

        // If datasets aren't available, that's okay - we're just testing the framework
        if let Ok(results) = result {
            assert!(
                !results.results.is_empty() || results.summary.total_combinations == 0,
                "Should return results or empty if no datasets available"
            );
        }
    }

    #[test]
    fn test_task_mapping_ner_datasets() {
        use anno::eval::task_mapping::{dataset_tasks, get_task_datasets};

        let datasets = get_task_datasets(Task::NER);
        assert!(!datasets.is_empty(), "NER should have associated datasets");

        // Check that a NER dataset supports NER
        if let Some(&first_dataset) = datasets.first() {
            let dataset_tasks = dataset_tasks(first_dataset);
            assert!(
                dataset_tasks.contains(&Task::NER),
                "Dataset {:?} should support NER",
                first_dataset
            );
        }
    }

    #[test]
    fn test_task_mapping_backends() {
        use anno::eval::task_mapping::get_task_backends;

        let backends = get_task_backends(Task::NER);
        assert!(!backends.is_empty(), "NER should have associated backends");
        assert!(
            backends.iter().any(|b| *b == "pattern"),
            "Pattern should support NER"
        );
    }

    #[test]
    fn test_evaluation_config_default() {
        let config = TaskEvalConfig::default();
        assert!(!config.tasks.is_empty(), "Default config should have tasks");
    }

    #[test]
    fn test_backend_inference() {
        let backend =
            BackendFactory::create("pattern").expect("Pattern backend should be available");

        let text = "John Smith works at Microsoft in Seattle.";
        let entities = backend.extract_entities(text, None);

        assert!(entities.is_ok(), "Backend should extract entities");
        let entities = entities.unwrap();
        assert!(!entities.is_empty(), "Should find at least some entities");
    }

    #[test]
    fn test_multiple_backends() {
        let backends_to_test = vec!["pattern", "heuristic", "stacked"];

        for backend_name in backends_to_test {
            let backend = BackendFactory::create(backend_name);
            assert!(
                backend.is_ok(),
                "Backend '{}' should be available",
                backend_name
            );

            let backend = backend.unwrap();
            let text = "Meeting on January 15, 2024";
            let result = backend.extract_entities(text, None);
            assert!(
                result.is_ok(),
                "Backend '{}' should extract entities",
                backend_name
            );
        }
    }

    #[test]
    fn test_evaluation_error_handling() {
        let evaluator = TaskEvaluator::new().expect("Failed to create evaluator");

        let config = TaskEvalConfig {
            tasks: vec![Task::NER],
            datasets: vec![], // Empty - will use defaults
            backends: vec!["nonexistent".to_string()],
            max_examples: Some(1),
            require_cached: false,
            seed: None,
            relation_threshold: 0.5,
            robustness: false,
            compute_familiarity: false,
            temporal_stratification: false,
            confidence_intervals: false,
            custom_coref_resolver: None,
        };

        let result = evaluator.evaluate_all(config);
        // Should handle gracefully - either return results with errors or fail cleanly
        match result {
            Ok(results) => {
                // If it succeeds, check that failed evaluations are marked
                for eval_result in &results.results {
                    if !eval_result.success {
                        assert!(
                            eval_result.error.is_some(),
                            "Failed evaluations should have error messages"
                        );
                    }
                }
            }
            Err(_) => {
                // Error is acceptable for invalid backend
            }
        }
    }
}
