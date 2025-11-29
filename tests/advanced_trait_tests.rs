//! End-to-end tests for advanced NER traits.
//!
//! Tests cover:
//! - ZeroShotNER trait with GLiNER
//! - RelationExtractor trait (heuristic and model)
//! - DiscontinuousNER trait with W2NER
//! - BiEncoder architecture components
//!
//! Many tests are marked `#[ignore]` because they require model downloads.
//! Run with: `cargo test --test advanced_trait_tests -- --ignored`

use anno::eval::{
    evaluate_discontinuous_ner, evaluate_relations, DiscontinuousEvalConfig, DiscontinuousGold,
    RelationEvalConfig, RelationGold, RelationPrediction,
};
use anno::{DiscontinuousEntity, Entity, EntityType};

// =============================================================================
// ZeroShotNER Trait Tests
// =============================================================================

mod zero_shot_ner {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_zero_shot_trait_definition() {
        // Verify the trait exists and has the right methods
        // This is a compile-time check
        fn _assert_trait<T: anno::ZeroShotNER>() {}
    }

    #[test]
    #[ignore] // Requires GLiNER model download
    fn test_gliner_zero_shot_custom_types() {
        // Test GLiNER with custom entity types
        #[cfg(feature = "onnx")]
        {
            use anno::GLiNEROnnx;
            use anno::ZeroShotNER;

            let model = match GLiNEROnnx::new("onnx-community/gliner_small-v2.1") {
                Ok(m) => m,
                Err(e) => {
                    println!("Skipping GLiNER test: {}", e);
                    return;
                }
            };

            // Custom medical entity types
            let types = &["drug", "disease", "symptom"];
            let text = "The patient was prescribed aspirin for headache.";

            let entities = model.extract_with_types(text, types, 0.5).unwrap();

            println!("Zero-shot entities:");
            for e in &entities {
                println!("  {} [{}] ({:.2})", e.text, e.entity_type, e.confidence);
            }

            // Should find at least aspirin (drug) or headache (symptom/disease)
            assert!(!entities.is_empty(), "Should find at least one entity");
        }
    }

    #[test]
    #[ignore]
    fn test_gliner_zero_shot_descriptions() {
        // Test with natural language descriptions
        #[cfg(feature = "onnx")]
        {
            use anno::GLiNEROnnx;
            use anno::ZeroShotNER;

            let model = match GLiNEROnnx::new("onnx-community/gliner_small-v2.1") {
                Ok(m) => m,
                Err(e) => {
                    println!("Skipping GLiNER test: {}", e);
                    return;
                }
            };

            let descriptions = &[
                "a pharmaceutical compound or medication",
                "a medical condition or illness",
            ];
            let text = "Ibuprofen is commonly used to treat arthritis.";

            let entities = model
                .extract_with_descriptions(text, descriptions, 0.5)
                .unwrap();

            println!("Zero-shot with descriptions:");
            for e in &entities {
                println!("  {} [{}] ({:.2})", e.text, e.entity_type, e.confidence);
            }
        }
    }
}

// =============================================================================
// RelationExtractor Trait Tests
// =============================================================================

mod relation_extractor {
    use super::*;
    use anno::backends::inference::{
        extract_relations, RelationExtractionConfig, SemanticRegistry,
    };

    #[test]
    fn test_relation_extractor_trait_definition() {
        fn _assert_trait<T: anno::RelationExtractor>() {}
    }

    #[test]
    fn test_heuristic_relation_extraction() {
        // Test the built-in heuristic relation extractor
        let entities = vec![
            Entity::new("Steve Jobs", EntityType::Person, 0, 10, 0.95),
            Entity::new("Apple", EntityType::Organization, 20, 25, 0.92),
        ];

        let text = "Steve Jobs founded Apple in 1976.";

        let registry = SemanticRegistry::builder()
            .add_entity("person", "A human being")
            .add_entity("organization", "A company or group")
            .add_relation("FOUNDED", "Founded an organization")
            .add_relation("WORKS_FOR", "Employment relationship")
            .add_relation("LOCATED_IN", "Located in a place")
            .build_placeholder(64);

        let config = RelationExtractionConfig::default();
        let relations = extract_relations(&entities, text, &registry, &config);

        assert!(!relations.is_empty(), "Should find the FOUNDED relation");
        assert_eq!(relations[0].relation_type, "FOUNDED");
        assert_eq!(relations[0].head.text, "Steve Jobs");
        assert_eq!(relations[0].tail.text, "Apple");
    }

    #[test]
    fn test_relation_extraction_evaluation() {
        let gold = vec![RelationGold::new(
            (0, 10),
            "PER",
            "Steve Jobs",
            (20, 25),
            "ORG",
            "Apple",
            "FOUNDED",
        )];

        let pred = vec![RelationPrediction {
            head_span: (0, 10),
            head_type: "PER".to_string(),
            tail_span: (20, 25),
            tail_type: "ORG".to_string(),
            relation_type: "FOUNDED".to_string(),
            confidence: 0.9,
        }];

        let metrics = evaluate_relations(&gold, &pred, &RelationEvalConfig::default());

        assert!(
            (metrics.strict_f1 - 1.0).abs() < 0.001,
            "Should have perfect F1 for exact match"
        );
    }

    #[test]
    fn test_relation_type_filter() {
        // Test that relation types not in registry are filtered
        let entities = vec![
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
            Entity::new("Bob", EntityType::Person, 10, 13, 0.9),
        ];

        let text = "Alice met Bob yesterday.";

        let registry = SemanticRegistry::builder()
            .add_entity("person", "A human being")
            // No relations defined!
            .build_placeholder(64);

        let config = RelationExtractionConfig::default();
        let relations = extract_relations(&entities, text, &registry, &config);

        // No relations should be found since none are in registry
        assert!(
            relations.is_empty(),
            "Should find no relations when none defined in registry"
        );
    }
}

// =============================================================================
// DiscontinuousNER Trait Tests
// =============================================================================

mod discontinuous_ner {
    use super::*;

    #[test]
    fn test_discontinuous_ner_trait_definition() {
        fn _assert_trait<T: anno::DiscontinuousNER>() {}
    }

    #[test]
    fn test_discontinuous_entity_evaluation() {
        let gold = vec![
            DiscontinuousGold::new(vec![(0, 8), (25, 33)], "LOC", "New York airports"),
            DiscontinuousGold::contiguous(13, 24, "LOC", "Los Angeles"),
        ];

        let pred = vec![
            DiscontinuousEntity {
                spans: vec![(0, 8), (25, 33)],
                text: "New York airports".to_string(),
                entity_type: "LOC".to_string(),
                confidence: 0.9,
            },
            DiscontinuousEntity {
                spans: vec![(13, 24)],
                text: "Los Angeles".to_string(),
                entity_type: "LOC".to_string(),
                confidence: 0.85,
            },
        ];

        let metrics = evaluate_discontinuous_ner(&gold, &pred, &DiscontinuousEvalConfig::default());

        assert!(
            (metrics.exact_f1 - 1.0).abs() < 0.001,
            "Should have perfect F1 for exact match: got {}",
            metrics.exact_f1
        );
    }

    #[test]
    #[ignore] // Requires W2NER model download
    fn test_w2ner_discontinuous_extraction() {
        #[cfg(feature = "onnx")]
        {
            use anno::DiscontinuousNER;
            use anno::W2NER;

            let model = match W2NER::from_pretrained("ljynlp/w2ner-bert-base") {
                Ok(m) => m,
                Err(e) => {
                    println!("Skipping W2NER test: {}", e);
                    return;
                }
            };

            let text = "New York and Los Angeles airports have increased security.";
            let types = &["location"];

            let entities = model.extract_discontinuous(text, types, 0.5).unwrap();

            println!("Discontinuous entities:");
            for e in &entities {
                println!(
                    "  {} [{}] spans={:?} (contiguous={})",
                    e.text,
                    e.entity_type,
                    e.spans,
                    e.is_contiguous()
                );
            }
        }
    }

    #[test]
    fn test_discontinuous_entity_properties() {
        // Contiguous entity
        let contiguous = DiscontinuousEntity {
            spans: vec![(0, 10)],
            text: "Steve Jobs".to_string(),
            entity_type: "PER".to_string(),
            confidence: 0.95,
        };

        assert!(contiguous.is_contiguous());
        let entity = contiguous.to_entity().expect("Should convert");
        assert_eq!(entity.start, 0);
        assert_eq!(entity.end, 10);

        // Discontinuous entity
        let discontinuous = DiscontinuousEntity {
            spans: vec![(0, 8), (25, 33)],
            text: "New York airports".to_string(),
            entity_type: "LOC".to_string(),
            confidence: 0.85,
        };

        assert!(!discontinuous.is_contiguous());
        assert!(discontinuous.to_entity().is_none());
    }
}

// =============================================================================
// BiEncoder Architecture Tests
// =============================================================================

mod bi_encoder {
    use anno::backends::inference::{
        DotProductInteraction, LateInteraction, SemanticRegistry, SpanRepConfig,
        SpanRepresentationLayer,
    };
    use anno::{RaggedBatch, SpanCandidate};

    #[test]
    fn test_semantic_registry() {
        let registry = SemanticRegistry::builder()
            .add_entity("person", "A named individual human being")
            .add_entity("organization", "A company, institution, or group")
            .add_entity("location", "A geographical place")
            .add_relation("CEO_OF", "Chief executive of organization")
            .build_placeholder(768);

        assert_eq!(registry.len(), 4);
        assert_eq!(registry.entity_labels().count(), 3);
        assert_eq!(registry.relation_labels().count(), 1);

        // Check embeddings are allocated
        assert_eq!(registry.embeddings.len(), 4 * 768);
    }

    #[test]
    fn test_late_interaction() {
        let interaction = DotProductInteraction::with_temperature(20.0);

        // 2 spans, 3 labels, 64 dim
        let span_embs = vec![0.1f32; 2 * 64];
        let label_embs = vec![0.1f32; 3 * 64];

        let scores = interaction.compute_similarity(&span_embs, 2, &label_embs, 3, 64);

        assert_eq!(scores.len(), 6); // 2 spans x 3 labels

        // All scores should be finite
        for score in &scores {
            assert!(score.is_finite());
        }
    }

    #[test]
    fn test_span_representation_layer() {
        let config = SpanRepConfig {
            hidden_dim: 64,
            max_width: 12,
            use_width_embeddings: true,
            width_emb_dim: 16,
        };

        let layer = SpanRepresentationLayer::new(config);

        // Create fake token embeddings (5 tokens, 64 dim)
        let token_embeddings = vec![0.1f32; 5 * 64];

        // Create candidates
        let candidates = vec![
            SpanCandidate::new(0, 0, 2), // tokens 0-1
            SpanCandidate::new(0, 2, 5), // tokens 2-4
        ];

        // Create ragged batch
        let sequences = vec![vec![0u32; 5]];
        let batch = RaggedBatch::from_sequences(&sequences);

        let span_embs = layer.forward(&token_embeddings, &candidates, &batch);

        assert_eq!(span_embs.len(), 2 * 64); // 2 spans x 64 dim
    }

    #[test]
    fn test_standard_ner_registry() {
        let registry = SemanticRegistry::standard_ner(768);

        assert!(registry.label_index.contains_key("person"));
        assert!(registry.label_index.contains_key("organization"));
        assert!(registry.label_index.contains_key("location"));
        assert!(registry.label_index.contains_key("date"));
        assert!(registry.label_index.contains_key("money"));
    }
}

// =============================================================================
// Integration Tests
// =============================================================================

mod integration {
    use super::*;
    use anno::backends::inference::HandshakingMatrix;

    #[test]
    fn test_handshaking_matrix_to_entities() {
        use anno::backends::inference::HandshakingCell;

        // Create a simple matrix with one entity span
        let matrix = HandshakingMatrix {
            cells: vec![HandshakingCell {
                i: 2,
                j: 0,
                label_idx: 0,
                score: 0.9,
            }],
            seq_len: 5,
            num_labels: 3,
        };

        // Create a registry
        let registry = anno::backends::inference::SemanticRegistry::builder()
            .add_entity("person", "A human being")
            .add_entity("organization", "A company")
            .add_entity("location", "A place")
            .build_placeholder(64);

        let decoded = matrix.decode_entities(&registry);

        // Should decode one entity
        assert_eq!(decoded.len(), 1);
        let (span, label, score) = &decoded[0];
        assert_eq!(span.start, 0);
        assert_eq!(span.end, 3);
        assert_eq!(label.slug, "person");
        assert!((score - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_full_evaluation_pipeline() {
        // Create gold standard
        let gold = vec![
            DiscontinuousGold::contiguous(0, 10, "PER", "Steve Jobs"),
            DiscontinuousGold::contiguous(20, 25, "ORG", "Apple"),
        ];

        // Create predictions
        let pred = vec![
            DiscontinuousEntity {
                spans: vec![(0, 10)],
                text: "Steve Jobs".to_string(),
                entity_type: "PER".to_string(),
                confidence: 0.95,
            },
            DiscontinuousEntity {
                spans: vec![(20, 25)],
                text: "Apple".to_string(),
                entity_type: "ORG".to_string(),
                confidence: 0.92,
            },
        ];

        // Evaluate
        let metrics = evaluate_discontinuous_ner(&gold, &pred, &DiscontinuousEvalConfig::default());

        // Check metrics
        assert!((metrics.exact_f1 - 1.0).abs() < 0.001);
        assert_eq!(metrics.num_predicted, 2);
        assert_eq!(metrics.num_gold, 2);
        assert_eq!(metrics.exact_matches, 2);

        // Check per-type breakdown
        assert!(metrics.per_type.contains_key("PER"));
        assert!(metrics.per_type.contains_key("ORG"));
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
        fn discontinuous_gold_bounding_range_valid(
            start in 0usize..1000,
            len in 1usize..100,
        ) {
            let end = start + len;
            let gold = DiscontinuousGold::contiguous(start, end, "TEST", "test");

            let (min, max) = gold.bounding_range().unwrap();
            prop_assert_eq!(min, start);
            prop_assert_eq!(max, end);
        }

        #[test]
        fn discontinuous_gold_total_length_correct(
            spans in proptest::collection::vec((0usize..1000, 1usize..100), 1..5)
        ) {
            let adjusted: Vec<(usize, usize)> = spans
                .into_iter()
                .map(|(start, len)| (start, start + len))
                .collect();

            let expected_len: usize = adjusted.iter().map(|(s, e)| e - s).sum();

            let gold = DiscontinuousGold::new(adjusted, "TEST", "test");
            prop_assert_eq!(gold.total_length(), expected_len);
        }

        #[test]
        fn relation_gold_creation_valid(
            head_start in 0usize..100,
            head_len in 1usize..20,
            tail_start in 150usize..250,
            tail_len in 1usize..20,
        ) {
            let gold = RelationGold::new(
                (head_start, head_start + head_len),
                "PER",
                "Head",
                (tail_start, tail_start + tail_len),
                "ORG",
                "Tail",
                "RELATION",
            );

            prop_assert_eq!(gold.head_span.0, head_start);
            prop_assert_eq!(gold.head_span.1, head_start + head_len);
            prop_assert_eq!(gold.tail_span.0, tail_start);
            prop_assert_eq!(gold.tail_span.1, tail_start + tail_len);
        }
    }
}
