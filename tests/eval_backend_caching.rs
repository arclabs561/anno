//! Tests for enum-based backend caching in parallel evaluation.

#[cfg(all(feature = "eval-parallel", feature = "onnx"))]
mod tests {
    use anno::eval::task_evaluator::TaskEvaluator;
    use std::sync::Arc;

    #[test]
    fn test_backend_caching_enum_creation() {
        // Test that backends can be created and cached
        let evaluator = TaskEvaluator::new().unwrap();

        // This is an internal function, but we can test via public API
        // The enum-based approach should work without downcast errors
        // We'll verify by running a small evaluation
        let config = anno::eval::task_evaluator::TaskEvalConfig {
            tasks: vec![anno::eval::task_mapping::Task::NER],
            datasets: vec![anno::eval::loader::DatasetId::WikiGold],
            backends: vec!["gliner_onnx".to_string()],
            max_examples: Some(3),
            seed: Some(42),
            require_cached: true,
            relation_threshold: 0.5,
            robustness: false,
            compute_familiarity: false,
            confidence_intervals: false,
            custom_coref_resolver: None,
        };

        // Should not panic or produce downcast errors
        let result = evaluator.evaluate_all(config);
        assert!(
            result.is_ok(),
            "Evaluation should succeed without downcast errors"
        );
    }

    #[test]
    fn test_multiple_backend_types_cached() {
        // Test that different backend types can be cached separately
        // This verifies the enum approach handles multiple backend types
        let evaluator = TaskEvaluator::new().unwrap();

        let config = anno::eval::task_evaluator::TaskEvalConfig {
            tasks: vec![anno::eval::task_mapping::Task::NER],
            datasets: vec![anno::eval::loader::DatasetId::WikiGold],
            backends: vec!["gliner_onnx".to_string(), "nuner".to_string()],
            max_examples: Some(2),
            seed: Some(42),
            require_cached: true,
            relation_threshold: 0.5,
            robustness: false,
            compute_familiarity: false,
            confidence_intervals: false,
            custom_coref_resolver: None,
        };

        let result = evaluator.evaluate_all(config);
        // Should succeed - enum allows different backend types to coexist
        assert!(result.is_ok());
    }
}
