//! Tests for NuNER and W2NER backends.
//!
//! These backends are placeholders demonstrating the API design for:
//! - NuNER: Token-based zero-shot NER (arbitrary-length entities)
//! - W2NER: Word-word relation grids (nested/discontinuous entities)

use anno::{Model, NuNER, W2NER, W2NERConfig, W2NERRelation};
use anno::backends::inference::HandshakingMatrix;

// =============================================================================
// NuNER Tests
// =============================================================================

mod nuner {
    use super::*;

    #[test]
    fn test_nuner_creation() {
        let ner = NuNER::new();
        assert_eq!(ner.model_id(), "numind/NuNER_Zero");
        assert!((ner.threshold() - 0.5).abs() < f64::EPSILON);
        assert_eq!(ner.name(), "nuner");
    }

    #[test]
    fn test_nuner_custom_model() {
        let ner = NuNER::with_model("custom/model-path")
            .with_threshold(0.7)
            .with_labels(vec![
                "technology".to_string(),
                "company".to_string(),
                "product".to_string(),
            ]);

        assert_eq!(ner.model_id(), "custom/model-path");
        assert!((ner.threshold() - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_nuner_threshold_clamping() {
        // Test that threshold is clamped to [0, 1]
        let ner = NuNER::new().with_threshold(1.5);
        assert!((ner.threshold() - 1.0).abs() < f64::EPSILON);

        let ner = NuNER::new().with_threshold(-0.5);
        assert!(ner.threshold().abs() < f64::EPSILON);
    }

    #[test]
    fn test_nuner_supported_types() {
        let ner = NuNER::new();
        let types = ner.supported_types();
        
        // Default labels should map to these types
        assert!(types.iter().any(|t| matches!(t, anno::EntityType::Person)));
        assert!(types.iter().any(|t| matches!(t, anno::EntityType::Organization)));
        assert!(types.iter().any(|t| matches!(t, anno::EntityType::Location)));
    }

    #[test]
    fn test_nuner_empty_input() {
        let ner = NuNER::new();
        let entities = ner.extract_entities("", None).unwrap();
        assert!(entities.is_empty());
    }

    #[test]
    fn test_nuner_is_not_available() {
        // NuNER requires ONNX model files to be downloaded
        let ner = NuNER::new();
        assert!(!ner.is_available());
    }

    #[test]
    fn test_nuner_description() {
        let ner = NuNER::new();
        let desc = ner.description();
        assert!(desc.contains("NuNER"));
        assert!(desc.contains("MIT"));
    }

    #[test]
    fn test_nuner_label_mapping() {
        // Test various label mappings
        let ner = NuNER::new().with_labels(vec![
            "PER".to_string(),      // Should map to Person
            "org".to_string(),      // Should map to Organization (case-insensitive)
            "LOCATION".to_string(), // Should map to Location
            "custom".to_string(),   // Should map to Other
        ]);

        let types = ner.supported_types();
        assert_eq!(types.len(), 4);
    }
}

// =============================================================================
// W2NER Tests
// =============================================================================

mod w2ner {
    use super::*;
    use anno::backends::inference::HandshakingCell;

    #[test]
    fn test_w2ner_creation() {
        let ner = W2NER::new();
        assert_eq!(ner.name(), "w2ner");
    }

    #[test]
    fn test_w2ner_config_defaults() {
        let config = W2NERConfig::default();
        assert!((config.threshold - 0.5).abs() < f64::EPSILON);
        assert!(config.allow_nested);
        assert!(config.allow_discontinuous);
        assert_eq!(config.entity_labels.len(), 3);
        assert!(config.entity_labels.contains(&"PER".to_string()));
        assert!(config.entity_labels.contains(&"ORG".to_string()));
        assert!(config.entity_labels.contains(&"LOC".to_string()));
    }

    #[test]
    fn test_w2ner_custom_config() {
        let config = W2NERConfig {
            threshold: 0.7,
            entity_labels: vec!["GENE".to_string(), "DISEASE".to_string()],
            allow_nested: false,
            allow_discontinuous: true,
            model_id: String::new(),
        };

        let ner = W2NER::with_config(config);
        let types = ner.supported_types();
        
        // Custom types should map to Other
        assert_eq!(types.len(), 2);
    }

    #[test]
    fn test_w2ner_relation_types() {
        // Test relation type conversions
        assert_eq!(W2NERRelation::from_index(0), W2NERRelation::None);
        assert_eq!(W2NERRelation::from_index(1), W2NERRelation::NNW);
        assert_eq!(W2NERRelation::from_index(2), W2NERRelation::THW);
        assert_eq!(W2NERRelation::from_index(99), W2NERRelation::None); // Invalid -> None

        assert_eq!(W2NERRelation::None.to_index(), 0);
        assert_eq!(W2NERRelation::NNW.to_index(), 1);
        assert_eq!(W2NERRelation::THW.to_index(), 2);
    }

    #[test]
    fn test_w2ner_decode_single_token_entity() {
        let ner = W2NER::new();
        let tokens = ["John"];

        // THW at (0,0) means single-token entity
        let matrix = HandshakingMatrix {
            cells: vec![HandshakingCell {
                i: 0,
                j: 0,
                label_idx: W2NERRelation::THW.to_index() as u16,
                score: 0.9,
            }],
            seq_len: 1,
            num_labels: 3,
        };

        let entities = ner.decode_from_matrix(&matrix, &tokens, 0);
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].0, 0); // start
        assert_eq!(entities[0].1, 1); // end
        assert!((entities[0].2 - 0.9).abs() < 0.001); // f32->f64 conversion tolerance
    }

    #[test]
    fn test_w2ner_decode_multi_token_entity() {
        let ner = W2NER::new();
        let tokens = ["New", "York", "City"];

        // THW at (2,0) means entity spans tokens 0-2
        let matrix = HandshakingMatrix {
            cells: vec![HandshakingCell {
                i: 2,
                j: 0,
                label_idx: W2NERRelation::THW.to_index() as u16,
                score: 0.95,
            }],
            seq_len: 3,
            num_labels: 3,
        };

        let entities = ner.decode_from_matrix(&matrix, &tokens, 0);
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].0, 0); // start
        assert_eq!(entities[0].1, 3); // end (exclusive)
    }

    #[test]
    fn test_w2ner_decode_nested_entities() {
        let config = W2NERConfig {
            allow_nested: true,
            ..Default::default()
        };
        let ner = W2NER::with_config(config);
        let tokens = ["University", "of", "California"];

        // Two entities: full span and nested "California"
        let matrix = HandshakingMatrix {
            cells: vec![
                // Full entity: "University of California"
                HandshakingCell {
                    i: 2,
                    j: 0,
                    label_idx: W2NERRelation::THW.to_index() as u16,
                    score: 0.95,
                },
                // Nested: "California"
                HandshakingCell {
                    i: 2,
                    j: 2,
                    label_idx: W2NERRelation::THW.to_index() as u16,
                    score: 0.85,
                },
            ],
            seq_len: 3,
            num_labels: 3,
        };

        let entities = ner.decode_from_matrix(&matrix, &tokens, 0);
        assert_eq!(entities.len(), 2);
    }

    #[test]
    fn test_w2ner_remove_nested() {
        let config = W2NERConfig {
            allow_nested: false,
            ..Default::default()
        };
        let ner = W2NER::with_config(config);
        let tokens = ["University", "of", "California"];

        // Same as above but with allow_nested=false
        let matrix = HandshakingMatrix {
            cells: vec![
                HandshakingCell {
                    i: 2,
                    j: 0,
                    label_idx: W2NERRelation::THW.to_index() as u16,
                    score: 0.95,
                },
                HandshakingCell {
                    i: 2,
                    j: 2,
                    label_idx: W2NERRelation::THW.to_index() as u16,
                    score: 0.85,
                },
            ],
            seq_len: 3,
            num_labels: 3,
        };

        let entities = ner.decode_from_matrix(&matrix, &tokens, 0);
        // Should only return the outer entity
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].0, 0); // start of "University"
    }

    #[test]
    fn test_w2ner_threshold_filtering() {
        let config = W2NERConfig {
            threshold: 0.9, // High threshold
            ..Default::default()
        };
        let ner = W2NER::with_config(config);
        let tokens = ["John", "Smith"];

        let matrix = HandshakingMatrix {
            cells: vec![
                // High confidence entity
                HandshakingCell {
                    i: 1,
                    j: 0,
                    label_idx: W2NERRelation::THW.to_index() as u16,
                    score: 0.95,
                },
                // Low confidence entity (should be filtered)
                HandshakingCell {
                    i: 0,
                    j: 0,
                    label_idx: W2NERRelation::THW.to_index() as u16,
                    score: 0.85,
                },
            ],
            seq_len: 2,
            num_labels: 3,
        };

        let entities = ner.decode_from_matrix(&matrix, &tokens, 0);
        assert_eq!(entities.len(), 1);
        assert!((entities[0].2 - 0.95).abs() < 0.001); // f32->f64 conversion tolerance
    }

    #[test]
    fn test_w2ner_empty_matrix() {
        let ner = W2NER::new();
        let tokens = ["Hello", "world"];

        let matrix = HandshakingMatrix {
            cells: vec![],
            seq_len: 2,
            num_labels: 3,
        };

        let entities = ner.decode_from_matrix(&matrix, &tokens, 0);
        assert!(entities.is_empty());
    }

    #[test]
    fn test_w2ner_invalid_indices() {
        let ner = W2NER::new();
        let tokens = ["Hello"];

        // Invalid: tail < head (backwards)
        let matrix = HandshakingMatrix {
            cells: vec![HandshakingCell {
                i: 0, // tail
                j: 5, // head > tail (invalid)
                label_idx: W2NERRelation::THW.to_index() as u16,
                score: 0.9,
            }],
            seq_len: 1,
            num_labels: 3,
        };

        let entities = ner.decode_from_matrix(&matrix, &tokens, 0);
        // Should filter out invalid spans
        assert!(entities.is_empty());
    }

    #[test]
    fn test_w2ner_empty_input() {
        let ner = W2NER::new();
        let entities = ner.extract_entities("", None).unwrap();
        assert!(entities.is_empty());
    }

    #[test]
    fn test_w2ner_is_not_available() {
        let ner = W2NER::new();
        assert!(!ner.is_available());
    }

    #[test]
    fn test_w2ner_description() {
        let ner = W2NER::new();
        let desc = ner.description();
        assert!(desc.contains("W2NER"));
        assert!(desc.contains("nested") || desc.contains("discontinuous"));
    }
}

// =============================================================================
// HandshakingMatrix Integration Tests
// =============================================================================

mod handshaking {
    use super::*;
    use anno::backends::inference::HandshakingCell;

    #[test]
    fn test_handshaking_matrix_creation() {
        let matrix = HandshakingMatrix {
            cells: vec![
                HandshakingCell {
                    i: 0,
                    j: 0,
                    label_idx: 1,
                    score: 0.9,
                },
                HandshakingCell {
                    i: 1,
                    j: 0,
                    label_idx: 2,
                    score: 0.8,
                },
            ],
            seq_len: 5,
            num_labels: 3,
        };

        assert_eq!(matrix.cells.len(), 2);
        assert_eq!(matrix.seq_len, 5);
        assert_eq!(matrix.num_labels, 3);
    }

    #[test]
    fn test_handshaking_cell_fields() {
        let cell = HandshakingCell {
            i: 3,
            j: 1,
            label_idx: 2,
            score: 0.95,
        };

        assert_eq!(cell.i, 3);
        assert_eq!(cell.j, 1);
        assert_eq!(cell.label_idx, 2);
        assert!((cell.score - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn test_w2ner_with_handshaking_matrix() {
        // Integration test: W2NER decoding with HandshakingMatrix
        let ner = W2NER::new();
        let tokens = ["The", "quick", "brown", "fox"];

        // Create a matrix indicating "quick brown" is an entity
        let matrix = HandshakingMatrix {
            cells: vec![HandshakingCell {
                i: 2, // tail at "brown"
                j: 1, // head at "quick"
                label_idx: W2NERRelation::THW.to_index() as u16,
                score: 0.88,
            }],
            seq_len: 4,
            num_labels: 3,
        };

        let entities = ner.decode_from_matrix(&matrix, &tokens, 0);
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].0, 1); // "quick"
        assert_eq!(entities[0].1, 3); // "brown" + 1
    }
}

// =============================================================================
// Integration Tests (require model downloads)
// =============================================================================

mod integration {
    #[allow(unused_imports)]
    use super::*;

    /// Test NuNER with real model - requires network and onnx feature.
    #[test]
    #[ignore] // Run with: cargo test --features onnx -- --ignored nuner_real
    fn test_nuner_real_model() {
        #[cfg(feature = "onnx")]
        {
            println!("\n=== NuNER Real Model Test ===\n");

            // Load real model with custom labels
            let ner = match NuNER::from_pretrained("numind/NuNerZero") {
                Ok(n) => n.with_labels(vec![
                    "person".to_string(),
                    "organization".to_string(), 
                    "location".to_string(),
                ]),
                Err(e) => {
                    println!("Skipping NuNER test (model not available): {}", e);
                    return;
                }
            };

            assert!(ner.is_available(), "NuNER should be available after loading");

            let test_cases = [
                "Steve Jobs founded Apple in California.",
                "Microsoft acquired GitHub for $7.5 billion.",
                "CRISPR was developed by Jennifer Doudna at UC Berkeley.",
            ];

            for text in test_cases {
                println!("Input: {}", text);

                match ner.extract_entities(text, None) {
                    Ok(entities) => {
                        for e in &entities {
                            println!("  - {} ({}, {:.2})", e.text, e.entity_type.as_label(), e.confidence);
                        }
                        if entities.is_empty() {
                            println!("  (no entities found)");
                        }
                    }
                    Err(e) => println!("  Error: {}", e),
                }
                println!();
            }
        }

        #[cfg(not(feature = "onnx"))]
        println!("Skipping NuNER test (onnx feature not enabled)");
    }

    /// Test W2NER with real model - requires network and onnx feature.
    #[test]
    #[ignore] // Run with: cargo test --features onnx -- --ignored w2ner_real
    fn test_w2ner_real_model() {
        #[cfg(feature = "onnx")]
        {
            use anno::DiscontinuousNER;

            println!("\n=== W2NER Real Model Test ===\n");

            // Load real model using from_pretrained
            let ner = match W2NER::from_pretrained("ljvmiranda921/w2ner-conll2003") {
                Ok(n) => n,
                Err(e) => {
                    println!("Skipping W2NER test (model not available): {}", e);
                    return;
                }
            };

            assert!(ner.is_available(), "W2NER should be available after loading");

            let test_cases = [
                "The European Union met with United States officials.",
                "John Smith and Mary Johnson visited New York City.",
                "Apple CEO Tim Cook announced new products in San Francisco.",
            ];

            for text in test_cases {
                println!("Input: {}", text);

                // Standard extraction
                match ner.extract_entities(text, None) {
                    Ok(entities) => {
                        println!("  Standard entities:");
                        for e in &entities {
                            println!("    - {} ({}, {:.2})", e.text, e.entity_type.as_label(), e.confidence);
                        }
                        if entities.is_empty() {
                            println!("    (no entities found)");
                        }
                    }
                    Err(e) => println!("  Error: {}", e),
                }

                // Discontinuous extraction using the DiscontinuousNER trait
                match ner.extract_discontinuous(text, &["PER", "ORG", "LOC"], 0.5) {
                    Ok(entities) => {
                        if !entities.is_empty() {
                            println!("  Discontinuous entities:");
                            for e in &entities {
                                println!("    - {} ({}, {:.2}) spans: {:?}", 
                                    e.text, e.entity_type, e.confidence, e.spans);
                            }
                        }
                    }
                    Err(e) => println!("  Discontinuous error: {}", e),
                }
                println!();
            }
        }

        #[cfg(not(feature = "onnx"))]
        println!("Skipping W2NER test (onnx feature not enabled)");
    }
}

// =============================================================================
// Property-Based Tests
// =============================================================================

mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn nuner_threshold_always_valid(threshold in -10.0f64..10.0f64) {
            let ner = NuNER::new().with_threshold(threshold);
            let t = ner.threshold();
            prop_assert!((0.0..=1.0).contains(&t));
        }

        #[test]
        fn w2ner_relation_roundtrip(idx in 0usize..5) {
            let relation = W2NERRelation::from_index(idx);
            let back = relation.to_index();
            // Only indices 0-2 are valid, others map to None (0)
            if idx <= 2 {
                prop_assert_eq!(back, idx);
            } else {
                prop_assert_eq!(back, 0);
            }
        }

        #[test]
        fn w2ner_empty_text_never_panics(text in ".*") {
            let ner = W2NER::new();
            let result = ner.extract_entities(&text, None);
            prop_assert!(result.is_ok());
        }

        #[test]
        fn nuner_empty_text_never_panics(text in ".*") {
            let ner = NuNER::new();
            let result = ner.extract_entities(&text, None);
            prop_assert!(result.is_ok());
        }
    }
}

