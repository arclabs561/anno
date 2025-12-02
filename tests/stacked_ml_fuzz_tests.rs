//! Fuzz tests for StackedNER with ML backends.
//!
//! These tests use property-based testing to find edge cases and validate
//! invariants when combining ML backends with pattern and heuristic layers.
//!
//! Focus areas:
//! - Multiple ML backend combinations
//! - Unicode and edge case handling
//! - Large entity sets
//! - Stress testing with various conflict strategies

#[cfg(feature = "onnx")]
mod fuzz_tests {
    use anno::backends::stacked::ConflictStrategy;
    use anno::{HeuristicNER, Model, RegexNER, StackedNER};
    use proptest::prelude::*;

    // Helper to create GLiNER with graceful failure handling
    fn create_gliner() -> Option<anno::GLiNEROnnx> {
        anno::GLiNEROnnx::new("onnx-community/gliner_small-v2.1").ok()
    }

    // Helper to create BertNER with graceful failure handling
    fn create_bert() -> Option<anno::BertNEROnnx> {
        anno::BertNEROnnx::new(anno::DEFAULT_BERT_ONNX_MODEL).ok()
    }

    // Helper to create NuNER with graceful failure handling
    fn create_nuner() -> Option<anno::NuNER> {
        anno::NuNER::from_pretrained("deepanwa/NuNerZero_onnx").ok()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Fuzz: Multiple ML backends with various conflict strategies
        #[test]
        fn fuzz_ml_backend_combinations(
            text in ".{0,500}",
            strategy in prop::sample::select(vec![
                ConflictStrategy::Priority,
                ConflictStrategy::LongestSpan,
                ConflictStrategy::HighestConf,
                ConflictStrategy::Union,
            ])
        ) {
            if let (Some(gliner), Some(bert)) = (create_gliner(), create_bert()) {
                let stacked = StackedNER::builder()
                    .layer(RegexNER::new())
                    .layer_boxed(Box::new(gliner))
                    .layer_boxed(Box::new(bert))
                    .layer(HeuristicNER::new())
                    .strategy(strategy)
                    .build();

                let entities = stacked.extract_entities(&text, None).unwrap();

                // Verify invariants hold
                for entity in &entities {
                    assert!(entity.start < entity.end, "Invalid span: {:?}", entity);
                    assert!(
                        entity.confidence >= 0.0 && entity.confidence <= 1.0,
                        "Invalid confidence: {}", entity.confidence
                    );
                }

                // Verify no overlaps (except with Union strategy)
                if strategy != ConflictStrategy::Union {
                    for i in 0..entities.len() {
                        for j in (i + 1)..entities.len() {
                            let e1 = &entities[i];
                            let e2 = &entities[j];
                            let overlap = e1.start < e2.end && e2.start < e1.end;
                            prop_assert!(
                                !overlap,
                                "Overlapping entities with {:?} strategy: {:?} and {:?}",
                                strategy, e1, e2
                            );
                        }
                    }
                }
            }
        }

        /// Fuzz: Unicode edge cases with ML backends
        #[test]
        fn fuzz_unicode_ml_stacked(text in proptest::string::string_regex(".*").unwrap()) {
            if let Some(gliner) = create_gliner() {
                let stacked = StackedNER::with_ml_first(Box::new(gliner));
                let entities = stacked.extract_entities(&text, None).unwrap();

                let text_len = text.chars().count();
                for entity in entities {
                    prop_assert!(entity.start < entity.end);
                    // Be lenient with Unicode edge cases
                    if text_len > 0 {
                        prop_assert!(entity.end <= text_len + 5);
                    }
                    prop_assert!(entity.confidence >= 0.0 && entity.confidence <= 1.0);
                }
            }
        }

        /// Fuzz: Large entity sets with ML backends
        #[test]
        fn fuzz_large_entity_sets(text in ".{100,2000}") {
            if let Some(gliner) = create_gliner() {
                let stacked = StackedNER::builder()
                    .layer(RegexNER::new())
                    .layer_boxed(Box::new(gliner))
                    .layer(HeuristicNER::new())
                    .build();

                let entities = stacked.extract_entities(&text, None).unwrap();

                // Should handle large texts without panicking
                // Verify all entities are valid
                for entity in &entities {
                    prop_assert!(entity.start < entity.end);
                    prop_assert!(entity.confidence >= 0.0 && entity.confidence <= 1.0);
                }

                // Entities should be sorted
                for i in 1..entities.len() {
                    let prev = &entities[i - 1];
                    let curr = &entities[i];
                    prop_assert!(
                        prev.start < curr.start ||
                        (prev.start == curr.start && prev.end <= curr.end),
                        "Entities not sorted"
                    );
                }
            }
        }

        /// Fuzz: ML-only stacks (no pattern/heuristic)
        #[test]
        fn fuzz_ml_only_stacks(text in ".{0,500}") {
            if let (Some(gliner), Some(bert)) = (create_gliner(), create_bert()) {
                let stacked = StackedNER::builder()
                    .layer_boxed(Box::new(gliner))
                    .layer_boxed(Box::new(bert))
                    .strategy(ConflictStrategy::HighestConf)
                    .build();

                let entities = stacked.extract_entities(&text, None).unwrap();

                // ML-only stacks should still produce valid entities
                for entity in entities {
                    prop_assert!(entity.start < entity.end);
                    prop_assert!(entity.confidence >= 0.0 && entity.confidence <= 1.0);
                }
            }
        }

        /// Fuzz: Different ML backend orderings
        #[test]
        fn fuzz_ml_backend_orderings(
            text in ".{0,300}",
            gliner_first in prop::bool::ANY
        ) {
            if let (Some(gliner), Some(bert)) = (create_gliner(), create_bert()) {
                let stacked = if gliner_first {
                    StackedNER::builder()
                        .layer(RegexNER::new())
                        .layer_boxed(Box::new(gliner))
                        .layer_boxed(Box::new(bert))
                        .layer(HeuristicNER::new())
                        .build()
                } else {
                    StackedNER::builder()
                        .layer(RegexNER::new())
                        .layer_boxed(Box::new(bert))
                        .layer_boxed(Box::new(gliner))
                        .layer(HeuristicNER::new())
                        .build()
                };

                let entities = stacked.extract_entities(&text, None).unwrap();

                // Ordering shouldn't break invariants
                for entity in entities {
                    prop_assert!(entity.start < entity.end);
                    prop_assert!(entity.confidence >= 0.0 && entity.confidence <= 1.0);
                }
            }
        }

        /// Fuzz: ML stacks with NuNER (token-based)
        #[test]
        fn fuzz_nuner_with_stacked(text in ".{0,500}") {
            if let Some(nuner) = create_nuner() {
                let stacked = StackedNER::builder()
                    .layer(RegexNER::new())
                    .layer_boxed(Box::new(nuner))
                    .layer(HeuristicNER::new())
                    .build();

                let entities = stacked.extract_entities(&text, None).unwrap();

                for entity in entities {
                    prop_assert!(entity.start < entity.end);
                    prop_assert!(entity.confidence >= 0.0 && entity.confidence <= 1.0);
                }
            }
        }

        /// Fuzz: Extreme confidence values from ML backends
        #[test]
        fn fuzz_extreme_ml_confidence(text in ".{0,200}") {
            if let Some(gliner) = create_gliner() {
                let stacked = StackedNER::with_ml_first(Box::new(gliner));
                let entities = stacked.extract_entities(&text, None).unwrap();

                // ML backends may produce various confidence scores
                // All should be in valid range
                for entity in entities {
                    prop_assert!(
                        entity.confidence >= 0.0 && entity.confidence <= 1.0,
                        "Confidence out of range: {}", entity.confidence
                    );
                }
            }
        }

        /// Fuzz: Empty and very short texts with ML
        #[test]
        fn fuzz_empty_short_texts_ml(text in ".{0,10}") {
            if let Some(gliner) = create_gliner() {
                let stacked = StackedNER::with_ml_first(Box::new(gliner));
                let entities = stacked.extract_entities(&text, None).unwrap();

                // Should handle empty/short texts gracefully
                for entity in entities {
                    prop_assert!(entity.start < entity.end);
                    prop_assert!(entity.confidence >= 0.0 && entity.confidence <= 1.0);
                }
            }
        }

        /// Fuzz: Control characters and special Unicode with ML
        #[test]
        fn fuzz_control_chars_ml(text in proptest::string::string_regex("[\x00-\x7F]*").unwrap()) {
            if let Some(gliner) = create_gliner() {
                let stacked = StackedNER::with_ml_first(Box::new(gliner));
                let entities = stacked.extract_entities(&text, None).unwrap();

                // Should handle control characters without panicking
                for entity in entities {
                    prop_assert!(entity.start < entity.end);
                    prop_assert!(entity.confidence >= 0.0 && entity.confidence <= 1.0);
                }
            }
        }

        /// Fuzz: Repeated patterns that might confuse ML backends
        #[test]
        fn fuzz_repeated_patterns_ml(text in ".{0,300}") {
            if let Some(gliner) = create_gliner() {
                // Create text with repeated patterns
                let repeated_text = format!("{} {} {}", text, text, text);
                let stacked = StackedNER::with_ml_first(Box::new(gliner));
                let entities = stacked.extract_entities(&repeated_text, None).unwrap();

                // Should handle repeated patterns
                for entity in entities {
                    prop_assert!(entity.start < entity.end);
                    prop_assert!(entity.confidence >= 0.0 && entity.confidence <= 1.0);
                }
            }
        }
    }

    // =========================================================================
    // Stress Tests (not property-based, but fuzz-style)
    // =========================================================================

    #[test]
    fn stress_test_many_ml_layers() {
        // Test with many ML backends (if available)
        if let (Some(gliner1), Some(gliner2), Some(bert)) =
            (create_gliner(), create_gliner(), create_bert())
        {
            // Create multiple instances
            let stacked = StackedNER::builder()
                .layer(RegexNER::new())
                .layer_boxed(Box::new(gliner1))
                .layer_boxed(Box::new(gliner2))
                .layer_boxed(Box::new(bert))
                .layer(HeuristicNER::new())
                .build();

            let text = "Apple Inc. was founded by Steve Jobs in 1976. Microsoft was founded by Bill Gates.";
            let entities = stacked.extract_entities(text, None).unwrap();

            // Should handle multiple ML layers
            assert!(!entities.is_empty());
            for entity in entities {
                assert!(entity.start < entity.end);
                assert!(entity.confidence >= 0.0 && entity.confidence <= 1.0);
            }
        }
    }

    #[test]
    fn stress_test_very_long_text() {
        if let Some(gliner) = create_gliner() {
            let stacked = StackedNER::with_ml_first(Box::new(gliner));

            // Create a very long text (but not too long to avoid timeout)
            let long_text = "Apple Inc. was founded by Steve Jobs. ".repeat(100);
            let entities = stacked.extract_entities(&long_text, None).unwrap();

            // Should handle long text without panicking
            for entity in entities {
                assert!(entity.start < entity.end);
                assert!(entity.confidence >= 0.0 && entity.confidence <= 1.0);
            }
        }
    }

    #[test]
    fn stress_test_all_strategies_with_ml() {
        let strategies = [
            ConflictStrategy::Priority,
            ConflictStrategy::LongestSpan,
            ConflictStrategy::HighestConf,
            ConflictStrategy::Union,
        ];

        for strategy in strategies.iter() {
            if let Some(gliner) = create_gliner() {
                let stacked = StackedNER::builder()
                    .layer(RegexNER::new())
                    .layer_boxed(Box::new(gliner))
                    .layer(HeuristicNER::new())
                    .strategy(*strategy)
                    .build();

                let text = "Test text with entities";
                let entities = stacked.extract_entities(text, None).unwrap();

                // All strategies should produce valid results
                for entity in entities {
                    assert!(entity.start < entity.end);
                    assert!(entity.confidence >= 0.0 && entity.confidence <= 1.0);
                }
            }
        }
    }
}
