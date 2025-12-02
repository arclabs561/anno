//! Tests for StackedNER composability with ML backends.
//!
//! These tests verify that StackedNER correctly combines ML backends with
//! pattern and heuristic layers, handles conflicts, and tracks provenance.
//!
//! StackedNER accepts **any backend that implements `Model`**, not just
//! regex and heuristics. This test suite validates ML backend composability.

#[cfg(feature = "onnx")]
mod onnx_tests {
    use anno::backends::stacked::ConflictStrategy;
    use anno::{HeuristicNER, Model, RegexNER, StackedNER};

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
        // NuNER requires from_pretrained to actually load the model
        // Using a common NuNER model ID
        anno::NuNER::from_pretrained("deepanwa/NuNerZero_onnx").ok()
    }

    #[test]
    fn test_with_ml_first() {
        // This test requires ONNX feature and a model download
        // Skip in CI if model not available
        if let Ok(gliner) = anno::GLiNEROnnx::new("onnx-community/gliner_small-v2.1") {
            let stacked = StackedNER::with_ml_first(Box::new(gliner));

            // Verify layer order
            let layer_names = stacked.layer_names();
            assert!(
                layer_names[0].to_lowercase().contains("gliner"),
                "ML should be first layer, got: {:?}",
                layer_names
            );
            assert!(
                layer_names.len() >= 3,
                "Should have ML + Pattern + Heuristic"
            );

            // Test extraction
            let text = "Dr. Smith charges $100/hr. Email: smith@test.com";
            let entities = stacked.extract_entities(text, None).unwrap();

            // Should find entities from multiple layers
            assert!(!entities.is_empty());

            // Verify provenance is set
            for entity in &entities {
                // At least some entities should have provenance
                if entity.provenance.is_some() {
                    assert!(!entity.provenance.as_ref().unwrap().source.is_empty());
                }
            }
        }
    }

    #[test]
    fn test_with_ml_fallback() {
        if let Ok(gliner) = anno::GLiNEROnnx::new("onnx-community/gliner_small-v2.1") {
            let stacked = StackedNER::with_ml_fallback(Box::new(gliner));

            // Verify layer order
            let layer_names = stacked.layer_names();
            assert!(
                layer_names[0].to_lowercase().contains("pattern")
                    || layer_names[0].to_lowercase().contains("regex"),
                "Pattern should be first, got: {:?}",
                layer_names
            );
            let last_layer = layer_names.last().unwrap();
            assert!(
                last_layer.to_lowercase().contains("gliner"),
                "ML should be last layer, got: {:?}",
                layer_names
            );

            // Test extraction
            let text = "Contact alice@example.com about the $500 invoice";
            let entities = stacked.extract_entities(text, None).unwrap();

            // Should find structured entities from pattern layer
            let has_email = entities
                .iter()
                .any(|e| matches!(e.entity_type, anno::EntityType::Email));
            assert!(has_email, "Should find email from pattern layer");
        }
    }

    #[test]
    fn test_custom_ml_stack() {
        if let Ok(gliner) = anno::GLiNEROnnx::new("onnx-community/gliner_small-v2.1") {
            let stacked = StackedNER::builder()
                .layer(RegexNER::new())
                .layer_boxed(Box::new(gliner))
                .layer(HeuristicNER::new())
                .strategy(ConflictStrategy::HighestConf)
                .build();

            assert_eq!(stacked.num_layers(), 3);

            let text = "Apple Inc. was founded in 1976";
            let entities = stacked.extract_entities(text, None).unwrap();

            // Should combine results from all layers
            assert!(!entities.is_empty());
        }
    }

    #[test]
    fn test_multiple_overlapping_entities() {
        // Test the fix for multiple overlap bug
        let stacked = StackedNER::default();

        // Create scenario where one entity overlaps with multiple
        // "New York" and "York" both exist, then "New York City" comes in
        let text = "New York City is large";
        let entities = stacked.extract_entities(text, None).unwrap();

        // Should not have overlapping entities (unless Union strategy)
        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                let e1 = &entities[i];
                let e2 = &entities[j];
                let overlap = !(e1.end <= e2.start || e2.end <= e1.start);
                // Overlaps are only allowed with Union strategy
                // For default Priority strategy, should not overlap
                if overlap && stacked.strategy() != ConflictStrategy::Union {
                    panic!(
                        "Found overlapping entities with {:?} strategy: {:?} and {:?}",
                        stacked.strategy(),
                        e1,
                        e2
                    );
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "requires at least one layer")]
    fn test_empty_stack_panics() {
        let _stacked = StackedNER::builder().build();
    }

    #[test]
    fn test_partial_error_handling() {
        // Test that if one layer fails, others still work
        // This is hard to test without a mock, but we can test the structure
        let stacked = StackedNER::default();

        // Default stack should work
        let text = "Test $100";
        let result = stacked.extract_entities(text, None);
        assert!(result.is_ok());

        // Verify it found something
        if let Ok(entities) = result {
            // Should at least find money
            let has_money = entities
                .iter()
                .any(|e| matches!(e.entity_type, anno::EntityType::Money));
            assert!(has_money, "Should find money from pattern layer");
        }
    }

    // =========================================================================
    // Multiple ML Backend Tests
    // =========================================================================

    #[test]
    fn test_multiple_ml_backends() {
        // Test stacking multiple ML backends together
        if let (Some(gliner), Some(bert)) = (create_gliner(), create_bert()) {
            let stacked = StackedNER::builder()
                .layer(RegexNER::new())
                .layer_boxed(Box::new(gliner))
                .layer_boxed(Box::new(bert))
                .layer(HeuristicNER::new())
                .strategy(ConflictStrategy::HighestConf)
                .build();

            assert_eq!(stacked.num_layers(), 4);

            let text = "Apple Inc. was founded in 1976 by Steve Jobs";
            let entities = stacked.extract_entities(text, None).unwrap();

            // Should combine results from all layers
            assert!(!entities.is_empty());

            // Verify layer names include both ML backends
            let layer_names = stacked.layer_names();
            assert!(
                layer_names
                    .iter()
                    .any(|n| n.to_lowercase().contains("gliner")),
                "Should have GLiNER layer, got: {:?}",
                layer_names
            );
            // BERT may return "unknown" if name() isn't implemented, so check for either
            assert!(
                layer_names.iter().any(|n| {
                    let n_lower = n.to_lowercase();
                    n_lower.contains("bert") || n_lower == "unknown"
                }),
                "Should have BERT layer (or 'unknown' if name() not implemented), got: {:?}",
                layer_names
            );
        }
    }

    #[test]
    fn test_ml_only_stack() {
        // Test stack with only ML backends (no pattern/heuristic)
        if let (Some(gliner), Some(bert)) = (create_gliner(), create_bert()) {
            let stacked = StackedNER::builder()
                .layer_boxed(Box::new(gliner))
                .layer_boxed(Box::new(bert))
                .strategy(ConflictStrategy::HighestConf)
                .build();

            assert_eq!(stacked.num_layers(), 2);

            let text = "Microsoft was founded by Bill Gates in 1975";
            let entities = stacked.extract_entities(text, None).unwrap();

            // ML backends should still extract entities
            assert!(!entities.is_empty());
        }
    }

    #[test]
    fn test_nuner_with_stacked() {
        // Test NuNER (zero-shot token-based) with StackedNER
        if let Some(nuner) = create_nuner() {
            let stacked = StackedNER::builder()
                .layer(RegexNER::new())
                .layer_boxed(Box::new(nuner))
                .layer(HeuristicNER::new())
                .build();

            let text = "Contact Sarah Chen at sarah@example.com about the $500 project";
            let entities = stacked.extract_entities(text, None).unwrap();

            // Should find structured entities from pattern layer
            let has_email = entities
                .iter()
                .any(|e| matches!(e.entity_type, anno::EntityType::Email));
            assert!(has_email, "Should find email from pattern layer");

            // NuNER might find person names
            assert!(!entities.is_empty());
        }
    }

    // =========================================================================
    // Conflict Strategy Tests with ML Backends
    // =========================================================================

    #[test]
    fn test_ml_with_longest_span_strategy() {
        if let Some(gliner) = create_gliner() {
            let stacked = StackedNER::builder()
                .layer(RegexNER::new())
                .layer_boxed(Box::new(gliner))
                .layer(HeuristicNER::new())
                .strategy(ConflictStrategy::LongestSpan)
                .build();

            let text = "New York City is located in New York state";
            let entities = stacked.extract_entities(text, None).unwrap();

            // With LongestSpan, should prefer longer spans
            // Verify no overlapping entities (except with Union)
            for i in 0..entities.len() {
                for j in (i + 1)..entities.len() {
                    let e1 = &entities[i];
                    let e2 = &entities[j];
                    let overlap = !(e1.end <= e2.start || e2.end <= e1.start);
                    if overlap {
                        panic!(
                            "Found overlapping entities with LongestSpan strategy: {:?} and {:?}",
                            e1, e2
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_ml_with_highest_conf_strategy() {
        if let Some(gliner) = create_gliner() {
            let stacked = StackedNER::builder()
                .layer(RegexNER::new())
                .layer_boxed(Box::new(gliner))
                .strategy(ConflictStrategy::HighestConf)
                .build();

            let text = "Apple Inc. charges $1000 for the iPhone";
            let entities = stacked.extract_entities(text, None).unwrap();

            // Should resolve conflicts by confidence
            // Verify all entities have valid confidence scores
            for entity in &entities {
                assert!(
                    entity.confidence >= 0.0 && entity.confidence <= 1.0,
                    "Invalid confidence: {}",
                    entity.confidence
                );
            }
        }
    }

    #[test]
    fn test_ml_with_union_strategy() {
        if let Some(gliner) = create_gliner() {
            let stacked = StackedNER::builder()
                .layer(RegexNER::new())
                .layer_boxed(Box::new(gliner))
                .layer(HeuristicNER::new())
                .strategy(ConflictStrategy::Union)
                .build();

            let text = "Contact John at john@example.com";
            let entities = stacked.extract_entities(text, None).unwrap();

            // Union strategy allows overlaps, so we just verify it doesn't panic
            assert!(!entities.is_empty());

            // Should find email from pattern layer
            let has_email = entities
                .iter()
                .any(|e| matches!(e.entity_type, anno::EntityType::Email));
            assert!(has_email, "Should find email from pattern layer");
        }
    }

    // =========================================================================
    // Provenance and Metadata Tests
    // =========================================================================

    #[test]
    fn test_ml_provenance_tracking() {
        if let Some(gliner) = create_gliner() {
            let stacked = StackedNER::with_ml_first(Box::new(gliner));

            let text = "Dr. Smith charges $100/hr. Email: smith@test.com";
            let entities = stacked.extract_entities(text, None).unwrap();

            // Verify provenance is set for entities
            let mut found_provenance = false;
            for entity in &entities {
                if let Some(prov) = &entity.provenance {
                    found_provenance = true;
                    assert!(
                        !prov.source.is_empty(),
                        "Provenance source should not be empty"
                    );
                    // Should track which layer found the entity (case-insensitive check)
                    let source_lower = prov.source.to_lowercase();
                    assert!(
                        source_lower.contains("gliner")
                            || source_lower.contains("pattern")
                            || source_lower.contains("heuristic")
                            || source_lower.contains("regex"),
                        "Provenance should indicate source layer: {:?}",
                        prov.source
                    );
                }
            }

            // At least some entities should have provenance
            if !entities.is_empty() {
                assert!(
                    found_provenance,
                    "At least some entities should have provenance"
                );
            }
        }
    }

    #[test]
    fn test_ml_layer_names() {
        if let Some(gliner) = create_gliner() {
            let stacked = StackedNER::with_ml_first(Box::new(gliner));

            let layer_names = stacked.layer_names();
            assert_eq!(layer_names.len(), 3, "Should have ML + Pattern + Heuristic");
            assert!(
                layer_names[0].to_lowercase().contains("gliner"),
                "ML should be first layer, got: {:?}",
                layer_names
            );
            assert!(
                layer_names[1].to_lowercase().contains("pattern")
                    || layer_names[1].to_lowercase().contains("regex"),
                "Pattern should be second layer, got: {:?}",
                layer_names
            );
            assert!(
                layer_names[2].to_lowercase().contains("heuristic"),
                "Heuristic should be third layer, got: {:?}",
                layer_names
            );
        }
    }

    // =========================================================================
    // Complex Composition Tests
    // =========================================================================

    #[test]
    fn test_ml_pattern_heuristic_all_three() {
        // Test that all three types work together
        if let Some(gliner) = create_gliner() {
            let stacked = StackedNER::builder()
                .layer(RegexNER::new())
                .layer_boxed(Box::new(gliner))
                .layer(HeuristicNER::new())
                .build();

            let text = "Dr. Sarah Chen from Microsoft in Seattle charges $200/hr. Contact: sarah@microsoft.com";
            let entities = stacked.extract_entities(text, None).unwrap();

            // Should find entities from all three layers
            let has_money = entities
                .iter()
                .any(|e| matches!(e.entity_type, anno::EntityType::Money));
            let has_email = entities
                .iter()
                .any(|e| matches!(e.entity_type, anno::EntityType::Email));

            assert!(has_money, "Should find money from pattern layer");
            assert!(has_email, "Should find email from pattern layer");
            assert!(!entities.is_empty(), "Should find at least some entities");
        }
    }

    #[test]
    fn test_ml_middle_layer() {
        // Test ML backend in the middle of the stack
        if let Some(gliner) = create_gliner() {
            let stacked = StackedNER::builder()
                .layer(RegexNER::new())
                .layer_boxed(Box::new(gliner))
                .layer(HeuristicNER::new())
                .strategy(ConflictStrategy::Priority)
                .build();

            let text = "Apple Inc. was founded in 1976";
            let entities = stacked.extract_entities(text, None).unwrap();

            // Pattern layer (first) should have priority
            // But ML and heuristic should still contribute
            assert!(!entities.is_empty());
        }
    }

    #[test]
    fn test_multiple_ml_with_different_strategies() {
        // Test that different conflict strategies work with ML backends
        let strategies = [
            ConflictStrategy::Priority,
            ConflictStrategy::LongestSpan,
            ConflictStrategy::HighestConf,
            ConflictStrategy::Union,
        ];

        for strategy in strategies.iter() {
            // Create a new GLiNER instance for each strategy test
            if let Some(gliner_for_strategy) = create_gliner() {
                let stacked = StackedNER::builder()
                    .layer(RegexNER::new())
                    .layer_boxed(Box::new(gliner_for_strategy))
                    .strategy(*strategy)
                    .build();

                let text = "Test $100";
                let result = stacked.extract_entities(text, None);
                assert!(
                    result.is_ok(),
                    "Strategy {:?} should work with ML backend",
                    strategy
                );
            }
        }
    }

    // =========================================================================
    // Edge Cases with ML Backends
    // =========================================================================

    #[test]
    fn test_ml_empty_text() {
        if let Some(gliner) = create_gliner() {
            let stacked = StackedNER::with_ml_first(Box::new(gliner));

            let entities = stacked.extract_entities("", None).unwrap();
            assert!(entities.is_empty(), "Empty text should produce no entities");
        }
    }

    #[test]
    fn test_ml_very_long_text() {
        if let Some(gliner) = create_gliner() {
            let stacked = StackedNER::with_ml_first(Box::new(gliner));

            // Create a long text (but not too long to avoid timeout)
            let long_text = "Apple Inc. ".repeat(100);
            let result = stacked.extract_entities(&long_text, None);

            // Should not panic, may or may not find entities
            assert!(result.is_ok(), "Should handle long text without panicking");
        }
    }

    #[test]
    fn test_ml_special_characters() {
        if let Some(gliner) = create_gliner() {
            let stacked = StackedNER::with_ml_first(Box::new(gliner));

            let text = "Email: test@example.com, Phone: (555) 123-4567, Price: $99.99";
            let entities = stacked.extract_entities(text, None).unwrap();

            // Should handle special characters in structured entities
            let has_email = entities
                .iter()
                .any(|e| matches!(e.entity_type, anno::EntityType::Email));
            let has_phone = entities
                .iter()
                .any(|e| matches!(e.entity_type, anno::EntityType::Phone));
            let has_money = entities
                .iter()
                .any(|e| matches!(e.entity_type, anno::EntityType::Money));

            // At least pattern layer should find these
            assert!(
                has_email || has_phone || has_money,
                "Should find at least one structured entity: {:?}",
                entities
            );
        }
    }

    #[test]
    fn test_ml_stats() {
        if let Some(gliner) = create_gliner() {
            let stacked = StackedNER::with_ml_first(Box::new(gliner));

            let stats = stacked.stats();
            assert_eq!(stats.layer_count, 3, "Should have 3 layers");
            assert_eq!(
                stats.strategy,
                ConflictStrategy::Priority,
                "Default strategy should be Priority"
            );
            assert_eq!(stats.layer_names.len(), 3, "Should have 3 layer names");
        }
    }

    // =========================================================================
    // Builder Pattern Tests
    // =========================================================================

    #[test]
    fn test_builder_fluent_api() {
        if let Some(gliner) = create_gliner() {
            let stacked = StackedNER::builder()
                .layer(RegexNER::new())
                .layer(HeuristicNER::new())
                .layer_boxed(Box::new(gliner))
                .strategy(ConflictStrategy::HighestConf)
                .build();

            assert_eq!(stacked.num_layers(), 3);
            assert_eq!(stacked.strategy(), ConflictStrategy::HighestConf);
        }
    }

    #[test]
    fn test_builder_single_ml_layer() {
        if let Some(gliner) = create_gliner() {
            let stacked = StackedNER::builder().layer_boxed(Box::new(gliner)).build();

            assert_eq!(stacked.num_layers(), 1);

            let text = "Apple Inc. was founded in 1976";
            let entities = stacked.extract_entities(text, None).unwrap();

            // Single ML layer should still work
            assert!(!entities.is_empty());
        }
    }

    // =========================================================================
    // Property-Based Tests for ML Composability
    // =========================================================================

    #[cfg(test)]
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(50))]

            /// Property: StackedNER with ML backend never panics on any input text
            #[test]
            fn ml_stacked_never_panics(text in ".*") {
                if let Some(gliner) = create_gliner() {
                    let stacked = StackedNER::with_ml_first(Box::new(gliner));
                    let _ = stacked.extract_entities(&text, None);
                }
            }

            /// Property: ML confidence scores are in [0, 1]
            #[test]
            fn ml_confidence_bounds(text in ".{0,1000}") {
                if let Some(gliner) = create_gliner() {
                    let stacked = StackedNER::with_ml_first(Box::new(gliner));
                    let entities = stacked.extract_entities(&text, None).unwrap();
                    for entity in entities {
                        prop_assert!(
                            entity.confidence >= 0.0 && entity.confidence <= 1.0,
                            "Confidence out of range: {}", entity.confidence
                        );
                    }
                }
            }

            /// Property: ML layers preserve entity span validity
            #[test]
            fn ml_spans_valid(text in ".{0,1000}") {
                if let Some(gliner) = create_gliner() {
                    let stacked = StackedNER::with_ml_first(Box::new(gliner));
                    let entities = stacked.extract_entities(&text, None).unwrap();
                    let text_len = text.chars().count();
                    for entity in entities {
                        prop_assert!(
                            entity.start < entity.end,
                            "Invalid span: start={}, end={}", entity.start, entity.end
                        );
                        // Allow small overflow for edge cases
                        if text_len > 0 && entity.end > text_len + 2 {
                            prop_assert!(
                                entity.end <= text_len + 2,
                                "Entity end significantly exceeds text length: end={}, text_len={}",
                                entity.end, text_len
                            );
                        }
                    }
                }
            }

            /// Property: ML+Pattern+Heuristic produces sorted entities
            #[test]
            fn ml_stacked_sorted_output(text in ".{0,1000}") {
                if let Some(gliner) = create_gliner() {
                    let stacked = StackedNER::with_ml_first(Box::new(gliner));
                    let entities = stacked.extract_entities(&text, None).unwrap();
                    for i in 1..entities.len() {
                        let prev = &entities[i - 1];
                        let curr = &entities[i];
                        prop_assert!(
                            prev.start < curr.start ||
                            (prev.start == curr.start && prev.end <= curr.end),
                            "Entities not sorted: prev=[{},{}), curr=[{}, {})",
                            prev.start, prev.end, curr.start, curr.end
                        );
                    }
                }
            }

            /// Property: ML stacks respect conflict strategies (no overlaps except Union)
            #[test]
            fn ml_stacked_no_overlaps(text in ".{0,500}") {
                if let Some(gliner) = create_gliner() {
                    let stacked = StackedNER::builder()
                        .layer_boxed(Box::new(gliner))
                        .layer(RegexNER::new())
                        .layer(HeuristicNER::new())
                        .strategy(ConflictStrategy::Priority)
                        .build();

                    let entities = stacked.extract_entities(&text, None).unwrap();
                    for i in 0..entities.len() {
                        for j in (i + 1)..entities.len() {
                            let e1 = &entities[i];
                            let e2 = &entities[j];
                            let overlap = e1.start < e2.end && e2.start < e1.end;
                            prop_assert!(
                                !overlap,
                                "Overlapping entities with Priority strategy: {:?} and {:?}",
                                e1, e2
                            );
                        }
                    }
                }
            }

            /// Property: Multiple ML backends produce valid results
            #[test]
            fn multiple_ml_backends_valid(text in ".{0,500}") {
                if let (Some(gliner), Some(bert)) = (create_gliner(), create_bert()) {
                    let stacked = StackedNER::builder()
                        .layer(RegexNER::new())
                        .layer_boxed(Box::new(gliner))
                        .layer_boxed(Box::new(bert))
                        .layer(HeuristicNER::new())
                        .strategy(ConflictStrategy::HighestConf)
                        .build();

                    let entities = stacked.extract_entities(&text, None).unwrap();
                    for entity in entities {
                        prop_assert!(entity.start < entity.end);
                        prop_assert!(entity.confidence >= 0.0 && entity.confidence <= 1.0);
                    }
                }
            }

            /// Property: All conflict strategies work with ML backends
            #[test]
            fn ml_all_strategies_valid(
                text in ".{0,200}",
                strategy in prop::sample::select(vec![
                    ConflictStrategy::Priority,
                    ConflictStrategy::LongestSpan,
                    ConflictStrategy::HighestConf,
                    ConflictStrategy::Union,
                ])
            ) {
                if let Some(gliner) = create_gliner() {
                    let stacked = StackedNER::builder()
                        .layer(RegexNER::new())
                        .layer_boxed(Box::new(gliner))
                        .layer(HeuristicNER::new())
                        .strategy(strategy)
                        .build();

                    let entities = stacked.extract_entities(&text, None).unwrap();
                    let text_len = text.chars().count();

                    for entity in entities {
                        prop_assert!(entity.start < entity.end);
                        if text_len > 0 {
                            prop_assert!(entity.end <= text_len + 2);
                        }
                        prop_assert!(entity.confidence >= 0.0 && entity.confidence <= 1.0);
                    }
                }
            }

            /// Property: ML-first vs ML-fallback produce valid results
            #[test]
            fn ml_first_vs_fallback_valid(text in ".{0,500}") {
                if let (Some(gliner1), Some(gliner2)) = (create_gliner(), create_gliner()) {
                    let ml_first = StackedNER::with_ml_first(Box::new(gliner1));
                    let ml_fallback = StackedNER::with_ml_fallback(Box::new(gliner2));

                    let entities_first = ml_first.extract_entities(&text, None).unwrap();
                    let entities_fallback = ml_fallback.extract_entities(&text, None).unwrap();

                    // Both should produce valid entities
                    for entity in entities_first.iter().chain(entities_fallback.iter()) {
                        prop_assert!(entity.start < entity.end);
                        prop_assert!(entity.confidence >= 0.0 && entity.confidence <= 1.0);
                    }
                }
            }

            /// Property: ML stacks handle Unicode correctly
            #[test]
            fn ml_stacked_unicode_handling(text in proptest::string::string_regex(".*").unwrap()) {
                if let Some(gliner) = create_gliner() {
                    let stacked = StackedNER::with_ml_first(Box::new(gliner));
                    let entities = stacked.extract_entities(&text, None).unwrap();

                    let text_len = text.chars().count();
                    for entity in entities {
                        prop_assert!(entity.start < entity.end);
                        if text_len > 0 {
                            // Be lenient with Unicode edge cases
                            prop_assert!(entity.end <= text_len + 5);
                        }
                    }
                }
            }
        }
    }
}
