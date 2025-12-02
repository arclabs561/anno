//! Error path tests for StackedNER with ML backends.
//!
//! These tests verify graceful error handling:
//! - ML backend failure scenarios
//! - Model loading failures
//! - Invalid configurations
//! - Partial failure handling

#[cfg(feature = "onnx")]
mod error_tests {
    use anno::{HeuristicNER, Model, RegexNER, StackedNER};

    // Helper to create GLiNER with graceful failure handling
    fn create_gliner() -> Option<anno::GLiNEROnnx> {
        anno::GLiNEROnnx::new("onnx-community/gliner_small-v2.1").ok()
    }

    #[test]
    fn test_ml_backend_failure_handling() {
        // Test that if ML backend fails, pattern/heuristic still work
        // This is hard to test without mocking, but we can test structure
        let default = StackedNER::default();
        let text = "Test $100";

        // Default should work even if ML backends aren't available
        let entities = default.extract_entities(text, None).unwrap();

        // Should find money from pattern layer
        let has_money = entities
            .iter()
            .any(|e| matches!(e.entity_type, anno::EntityType::Money));
        assert!(has_money, "Should find money from pattern layer");
    }

    #[test]
    fn test_invalid_model_config_graceful() {
        // Test behavior with invalid model IDs
        // Should fail gracefully with appropriate error
        let result = anno::GLiNEROnnx::new("invalid/model/id");

        // Should return error, not panic
        assert!(result.is_err());

        // Error message should be informative
        if let Err(e) = result {
            let error_msg = format!("{}", e);
            assert!(!error_msg.is_empty());
        }
    }

    #[test]
    fn test_partial_layer_failure() {
        // Test that if one layer fails, others still work
        // StackedNER should continue with remaining layers
        if let Some(gliner) = create_gliner() {
            let stacked = StackedNER::with_ml_first(Box::new(gliner));

            // Empty text should work fine
            let entities = stacked.extract_entities("", None).unwrap();
            assert!(entities.is_empty());

            // Normal text should work
            let entities = stacked.extract_entities("Test $100", None).unwrap();
            // Should find at least money from pattern layer
            let has_money = entities
                .iter()
                .any(|e| matches!(e.entity_type, anno::EntityType::Money));
            assert!(has_money, "Should find money from pattern layer");
        }
    }

    #[test]
    fn test_empty_stack_handling() {
        // Test that empty stack panics (as expected)
        // This is tested in stacked_ml_composability.rs, but verify here too
        let result = std::panic::catch_unwind(|| {
            let _ = StackedNER::builder().build();
        });

        // Should panic
        assert!(result.is_err());
    }

    #[test]
    fn test_ml_stacked_with_invalid_text() {
        // Test handling of edge case texts
        if let Some(gliner) = create_gliner() {
            let stacked = StackedNER::with_ml_first(Box::new(gliner));

            // Very long text (but not too long to avoid timeout)
            let long_text = "A".repeat(10000);
            let result = stacked.extract_entities(&long_text, None);

            // Should not panic, may or may not find entities
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_ml_stacked_error_propagation() {
        // Test that errors from ML backends are properly handled
        // StackedNER should continue with other layers if one fails
        let default = StackedNER::default();

        // Default stack should always work (no ML dependencies)
        let text = "Test text";
        let result = default.extract_entities(text, None);

        // Should succeed
        assert!(result.is_ok());
    }

    #[test]
    fn test_ml_stacked_graceful_degradation() {
        // Test that ML stacks degrade gracefully when ML unavailable
        // Pattern + Heuristic should still work
        let default = StackedNER::default();
        let text = "Dr. Smith charges $100/hr. Email: smith@test.com";

        let entities = default.extract_entities(text, None).unwrap();

        // Should find entities from pattern layer
        let has_money = entities
            .iter()
            .any(|e| matches!(e.entity_type, anno::EntityType::Money));
        assert!(has_money, "Should find money from pattern layer");

        let has_email = entities
            .iter()
            .any(|e| matches!(e.entity_type, anno::EntityType::Email));
        assert!(has_email, "Should find email from pattern layer");
    }
}
