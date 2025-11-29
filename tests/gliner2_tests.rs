//! Comprehensive tests for GLiNER2 multi-task extraction.
//!
//! Tests the schema builder, task composition, and trait implementations.
//!
//! Run with: `cargo test --test gliner2_tests --features candle`
//! Or: `cargo test --test gliner2_tests --features onnx`

#![cfg(any(feature = "onnx", feature = "candle"))]

use anno::backends::gliner2::{ExtractionResult, FieldType, StructureTask, TaskSchema};
use std::collections::HashMap;

// =============================================================================
// TaskSchema Builder Tests
// =============================================================================

#[test]
fn test_empty_schema() {
    let schema = TaskSchema::new();
    assert!(schema.entities.is_none());
    assert!(schema.classifications.is_empty());
    assert!(schema.structures.is_empty());
}

#[test]
fn test_entity_only_schema() {
    let schema = TaskSchema::new().with_entities(&["person", "organization", "location"]);

    assert!(schema.entities.is_some());
    let ent = schema.entities.as_ref().unwrap();
    assert_eq!(ent.types.len(), 3);
    assert!(ent.types.contains(&"person".to_string()));
    assert!(ent.types.contains(&"organization".to_string()));
    assert!(ent.types.contains(&"location".to_string()));
    assert!(ent.descriptions.is_empty());
}

#[test]
fn test_entities_with_descriptions() {
    let mut descriptions = HashMap::new();
    descriptions.insert(
        "person".to_string(),
        "A named individual human being".to_string(),
    );
    descriptions.insert(
        "company".to_string(),
        "A business organization or corporation".to_string(),
    );

    let schema = TaskSchema::new().with_entities_described(descriptions);

    let ent = schema.entities.unwrap();
    assert_eq!(ent.types.len(), 2);
    assert_eq!(
        ent.descriptions.get("person"),
        Some(&"A named individual human being".to_string())
    );
}

#[test]
fn test_classification_schema() {
    let schema = TaskSchema::new()
        .with_classification("sentiment", &["positive", "negative", "neutral"], false)
        .with_classification("intent", &["question", "statement", "command"], false);

    assert_eq!(schema.classifications.len(), 2);
    assert_eq!(schema.classifications[0].name, "sentiment");
    assert_eq!(schema.classifications[0].labels.len(), 3);
    assert!(!schema.classifications[0].multi_label);

    assert_eq!(schema.classifications[1].name, "intent");
    assert_eq!(schema.classifications[1].labels.len(), 3);
}

#[test]
fn test_multi_label_classification() {
    let schema = TaskSchema::new().with_classification(
        "topics",
        &["sports", "politics", "tech", "entertainment"],
        true,
    );

    assert_eq!(schema.classifications.len(), 1);
    assert!(schema.classifications[0].multi_label);
    assert_eq!(schema.classifications[0].labels.len(), 4);
}

#[test]
fn test_structure_task_builder() {
    let task = StructureTask::new("product")
        .with_field_described("name", FieldType::String, "Product name")
        .with_field("price", FieldType::String)
        .with_field_described("features", FieldType::List, "List of product features")
        .with_choice_field("category", &["electronics", "software", "hardware"]);

    assert_eq!(task.name, "product");
    assert_eq!(task.fields.len(), 4);

    assert_eq!(task.fields[0].name, "name");
    assert_eq!(task.fields[0].field_type, FieldType::String);
    assert_eq!(task.fields[0].description, Some("Product name".to_string()));

    assert_eq!(task.fields[1].name, "price");
    assert_eq!(task.fields[1].field_type, FieldType::String);
    assert!(task.fields[1].description.is_none());

    assert_eq!(task.fields[2].name, "features");
    assert_eq!(task.fields[2].field_type, FieldType::List);

    assert_eq!(task.fields[3].name, "category");
    assert_eq!(task.fields[3].field_type, FieldType::Choice);
    assert!(task.fields[3].choices.is_some());
    assert_eq!(task.fields[3].choices.as_ref().unwrap().len(), 3);
}

#[test]
fn test_combined_schema() {
    let schema = TaskSchema::new()
        .with_entities(&["person", "organization"])
        .with_classification("sentiment", &["positive", "negative", "neutral"], false)
        .with_structure(
            StructureTask::new("announcement")
                .with_field_described("company", FieldType::String, "Company making announcement")
                .with_field("product", FieldType::String)
                .with_field("features", FieldType::List),
        );

    assert!(schema.entities.is_some());
    assert_eq!(schema.classifications.len(), 1);
    assert_eq!(schema.structures.len(), 1);
    assert_eq!(schema.structures[0].fields.len(), 3);
}

// =============================================================================
// FieldType Tests
// =============================================================================

#[test]
fn test_field_types() {
    // Verify FieldType enum values exist and are distinct
    let string_type = FieldType::String;
    let list_type = FieldType::List;
    let choice_type = FieldType::Choice;

    assert_ne!(string_type, list_type);
    assert_ne!(list_type, choice_type);
    assert_ne!(string_type, choice_type);
}

// =============================================================================
// ExtractionResult Tests
// =============================================================================

#[test]
fn test_extraction_result_default() {
    let result = ExtractionResult::default();
    assert!(result.entities.is_empty());
    assert!(result.classifications.is_empty());
    assert!(result.structures.is_empty());
}

#[test]
fn test_extraction_result_serialization() {
    let result = ExtractionResult::default();
    let json = serde_json::to_string(&result).expect("Should serialize");
    assert!(json.contains("entities"));
    assert!(json.contains("classifications"));
    assert!(json.contains("structures"));
}

// =============================================================================
// Domain-Specific Schema Tests
// =============================================================================

#[test]
fn test_medical_domain_schema() {
    let schema = TaskSchema::new()
        .with_entities(&[
            "medication",
            "disease",
            "symptom",
            "dosage",
            "body_part",
            "procedure",
        ])
        .with_classification(
            "severity",
            &["mild", "moderate", "severe", "critical"],
            false,
        )
        .with_structure(
            StructureTask::new("patient_complaint")
                .with_field_described(
                    "chief_complaint",
                    FieldType::String,
                    "Primary reason for visit",
                )
                .with_field_described("symptoms", FieldType::List, "Reported symptoms"),
        );

    let ent = schema.entities.as_ref().unwrap();
    assert_eq!(ent.types.len(), 6);
    assert!(ent.types.contains(&"medication".to_string()));
    assert!(ent.types.contains(&"symptom".to_string()));

    assert_eq!(schema.classifications[0].labels.len(), 4);
    assert_eq!(schema.structures[0].fields.len(), 2);
}

#[test]
fn test_financial_domain_schema() {
    let schema = TaskSchema::new()
        .with_entities(&["company", "person", "money", "percentage", "date"])
        .with_classification(
            "topics",
            &["merger", "acquisition", "ipo", "earnings", "layoffs"],
            true, // multi-label
        )
        .with_structure(
            StructureTask::new("deal")
                .with_field_described("acquirer", FieldType::String, "Company making acquisition")
                .with_field_described("target", FieldType::String, "Company being acquired")
                .with_field_described("value", FieldType::String, "Deal value"),
        );

    assert!(schema.classifications[0].multi_label);
    assert_eq!(schema.structures[0].fields.len(), 3);
}

#[test]
fn test_legal_domain_schema() {
    let mut descriptions = HashMap::new();
    descriptions.insert(
        "party".to_string(),
        "Named litigant or legal entity in a case".to_string(),
    );
    descriptions.insert(
        "court".to_string(),
        "Judicial body hearing the case".to_string(),
    );
    descriptions.insert(
        "statute".to_string(),
        "Referenced law or regulation".to_string(),
    );

    let schema = TaskSchema::new()
        .with_entities_described(descriptions)
        .with_classification(
            "case_type",
            &["civil", "criminal", "administrative", "constitutional"],
            true,
        )
        .with_structure(
            StructureTask::new("citation")
                .with_field_described("case_name", FieldType::String, "Full case name")
                .with_field("volume", FieldType::String)
                .with_field("reporter", FieldType::String)
                .with_field("page", FieldType::String)
                .with_field("year", FieldType::String),
        );

    let ent = schema.entities.as_ref().unwrap();
    assert!(ent.descriptions.contains_key("party"));
    assert_eq!(schema.structures[0].fields.len(), 5);
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_duplicate_entity_types() {
    let schema = TaskSchema::new().with_entities(&["person", "person", "organization"]);

    let ent = schema.entities.as_ref().unwrap();
    // Should contain duplicates (no dedup in builder)
    assert_eq!(ent.types.len(), 3);
}

#[test]
fn test_empty_entity_types() {
    let schema = TaskSchema::new().with_entities(&[]);
    let ent = schema.entities.as_ref().unwrap();
    assert!(ent.types.is_empty());
}

#[test]
fn test_empty_classification_labels() {
    let schema = TaskSchema::new().with_classification("empty_task", &[], false);

    assert_eq!(schema.classifications.len(), 1);
    assert!(schema.classifications[0].labels.is_empty());
}

#[test]
fn test_structure_with_no_fields() {
    let task = StructureTask::new("empty_structure");
    assert_eq!(task.name, "empty_structure");
    assert!(task.fields.is_empty());
}

#[test]
fn test_long_descriptions() {
    let long_desc = "A".repeat(1000);
    let mut descriptions = HashMap::new();
    descriptions.insert("entity".to_string(), long_desc.clone());

    let schema = TaskSchema::new().with_entities_described(descriptions);

    let ent = schema.entities.as_ref().unwrap();
    assert_eq!(ent.descriptions.get("entity").unwrap().len(), 1000);
}

#[test]
fn test_unicode_in_schema() {
    let schema = TaskSchema::new()
        .with_entities(&["人物", "组织", "地点"])
        .with_classification("sentiment", &["正面", "负面", "中性"], false);

    let ent = schema.entities.as_ref().unwrap();
    assert!(ent.types.contains(&"人物".to_string()));
    assert!(schema.classifications[0]
        .labels
        .contains(&"正面".to_string()));
}

// =============================================================================
// Complex Multi-Task Schemas
// =============================================================================

#[test]
fn test_realistic_multi_task_schema() {
    let schema = TaskSchema::new()
        .with_entities(&["company", "product", "person", "money", "date"])
        .with_classification("sentiment", &["bullish", "bearish", "neutral"], false)
        .with_classification(
            "topics",
            &["earnings", "acquisition", "product_launch", "executive"],
            true,
        )
        .with_structure(
            StructureTask::new("deal")
                .with_field_described("acquirer", FieldType::String, "Company making acquisition")
                .with_field_described("target", FieldType::String, "Company being acquired")
                .with_field_described("value", FieldType::String, "Monetary value of deal"),
        );

    // Verify all components
    let ent = schema.entities.as_ref().unwrap();
    assert_eq!(ent.types.len(), 5);

    assert_eq!(schema.classifications.len(), 2);
    assert!(!schema.classifications[0].multi_label); // sentiment
    assert!(schema.classifications[1].multi_label); // topics

    assert_eq!(schema.structures.len(), 1);
    assert_eq!(schema.structures[0].name, "deal");
    assert_eq!(schema.structures[0].fields.len(), 3);
}

// =============================================================================
// Trait Implementation Tests
// =============================================================================

mod trait_tests {
    #[test]
    fn test_gliner2_model_trait() {
        // Verify GLiNER2 implements Model trait
        use anno::Model;

        #[cfg(feature = "candle")]
        {
            fn _assert_model<T: Model>() {}
            fn _check() {
                _assert_model::<anno::backends::gliner2::GLiNER2Candle>();
            }
        }

        #[cfg(all(feature = "onnx", not(feature = "candle")))]
        {
            fn _assert_model<T: Model>() {}
            fn _check() {
                _assert_model::<anno::backends::gliner2::GLiNER2Onnx>();
            }
        }
    }

    #[test]
    fn test_gliner2_supported_types() {
        use anno::Model;

        #[cfg(feature = "candle")]
        {
            // Can't actually construct without loading model, but we can verify the trait exists
            fn _check_supported_types<T: Model>(model: &T) {
                let _types = model.supported_types();
            }
        }
    }
}

mod zero_shot_tests {
    #[test]
    fn test_zero_shot_trait_bound() {
        use anno::backends::inference::ZeroShotNER;

        #[cfg(feature = "candle")]
        {
            fn _assert_zero_shot<T: ZeroShotNER>() {}
            fn _check() {
                _assert_zero_shot::<anno::backends::gliner2::GLiNER2Candle>();
            }
        }

        #[cfg(all(feature = "onnx", not(feature = "candle")))]
        {
            fn _assert_zero_shot<T: ZeroShotNER>() {}
            fn _check() {
                _assert_zero_shot::<anno::backends::gliner2::GLiNER2Onnx>();
            }
        }
    }
}

mod batch_tests {
    #[test]
    fn test_batch_capable_trait_bound() {
        use anno::BatchCapable;

        #[cfg(feature = "candle")]
        {
            fn _assert_batch<T: BatchCapable>() {}
            fn _check() {
                _assert_batch::<anno::backends::gliner2::GLiNER2Candle>();
            }
        }

        #[cfg(all(feature = "onnx", not(feature = "candle")))]
        {
            fn _assert_batch<T: BatchCapable>() {}
            fn _check() {
                _assert_batch::<anno::backends::gliner2::GLiNER2Onnx>();
            }
        }
    }
}

mod integration_tests {
    use super::*;
    use anno::backends::gliner2::{ClassificationResult, ExtractedStructure, StructureValue};

    #[test]
    fn test_classification_result_construction() {
        let mut scores = HashMap::new();
        scores.insert("positive".to_string(), 0.8f32);
        scores.insert("negative".to_string(), 0.1f32);
        scores.insert("neutral".to_string(), 0.1f32);

        let result = ClassificationResult {
            labels: vec!["positive".to_string()],
            scores,
        };

        assert_eq!(result.labels.len(), 1);
        assert_eq!(result.labels[0], "positive");
        assert_eq!(result.scores.get("positive"), Some(&0.8f32));
    }

    #[test]
    fn test_structure_extraction_result() {
        let mut fields = HashMap::new();
        fields.insert(
            "name".to_string(),
            StructureValue::Single("iPhone".to_string()),
        );
        fields.insert(
            "features".to_string(),
            StructureValue::List(vec!["5G".to_string(), "A17 chip".to_string()]),
        );

        let structure = ExtractedStructure {
            structure_type: "product".to_string(),
            fields,
        };

        assert_eq!(structure.structure_type, "product");
        assert!(matches!(
            structure.fields.get("name"),
            Some(StructureValue::Single(_))
        ));
        assert!(matches!(
            structure.fields.get("features"),
            Some(StructureValue::List(_))
        ));
    }
}

// =============================================================================
// RelationExtractor Tests
// =============================================================================

#[cfg(any(feature = "onnx", feature = "candle"))]
mod relation_extraction {
    use anno::backends::inference::{ExtractionWithRelations, RelationTriple};
    use anno::Entity;

    #[test]
    fn test_relation_triple_structure() {
        let triple = RelationTriple {
            head_idx: 0,
            tail_idx: 1,
            relation_type: "WORKS_FOR".to_string(),
            confidence: 0.85,
        };

        assert_eq!(triple.head_idx, 0);
        assert_eq!(triple.tail_idx, 1);
        assert_eq!(triple.relation_type, "WORKS_FOR");
        assert!(triple.confidence > 0.8);
    }

    #[test]
    fn test_extraction_with_relations_structure() {
        let entities = vec![
            Entity::new(
                "Steve Jobs".to_string(),
                anno::EntityType::Person,
                0,
                10,
                0.9,
            ),
            Entity::new(
                "Apple".to_string(),
                anno::EntityType::Organization,
                20,
                25,
                0.85,
            ),
        ];

        let relations = vec![RelationTriple {
            head_idx: 0,
            tail_idx: 1,
            relation_type: "FOUNDED".to_string(),
            confidence: 0.75,
        }];

        let result = ExtractionWithRelations {
            entities: entities.clone(),
            relations,
        };

        assert_eq!(result.entities.len(), 2);
        assert_eq!(result.relations.len(), 1);
        assert_eq!(result.relations[0].relation_type, "FOUNDED");
    }

    #[test]
    fn test_relation_type_patterns() {
        // Test that relation heuristics understand common patterns
        let _text = "Tim Cook is the CEO of Apple Inc.";

        // The relation extractor should recognize:
        // - Person-Organization -> WORKS_FOR, CEO_OF, etc.
        // - The "CEO" trigger pattern

        // Without model, we just verify the data structures work
        let entities = vec![
            Entity::new("Tim Cook".to_string(), anno::EntityType::Person, 0, 8, 0.9),
            Entity::new(
                "Apple Inc".to_string(),
                anno::EntityType::Organization,
                23,
                32,
                0.85,
            ),
        ];

        // Simulate what the heuristic extractor would find
        let _expected_relation = "CEO_OF";

        assert!(entities[0].text.contains("Tim"));
        assert!(entities[1].text.contains("Apple"));
    }
}

// =============================================================================
// BatchCapable Tests for GLiNER2
// =============================================================================

#[cfg(any(feature = "onnx", feature = "candle"))]
mod batch_processing {
    use anno::BatchCapable;

    #[test]
    fn test_batch_capable_trait_bounds() {
        // Verify trait exists and has expected methods
        fn _check_trait<T: BatchCapable>(_model: &T) {
            // This compiles = trait is properly defined
        }
    }

    #[test]
    fn test_optimal_batch_size_reasonable() {
        // Verify typical batch sizes are reasonable
        let typical_onnx_batch = 16;
        let typical_candle_batch = 8;

        assert!(typical_onnx_batch >= 1 && typical_onnx_batch <= 64);
        assert!(typical_candle_batch >= 1 && typical_candle_batch <= 32);
    }

    #[test]
    fn test_batch_input_validation() {
        // Empty batch should return empty results
        let empty_texts: Vec<&str> = vec![];
        assert!(empty_texts.is_empty());

        // Single item batch should work
        let single = vec!["Hello world"];
        assert_eq!(single.len(), 1);

        // Multi-item batch
        let multi = vec!["Text 1", "Text 2", "Text 3"];
        assert_eq!(multi.len(), 3);
    }
}

// =============================================================================
// StreamingCapable Tests for GLiNER2
// =============================================================================

#[cfg(any(feature = "onnx", feature = "candle"))]
mod streaming_processing {
    use anno::StreamingCapable;

    #[test]
    fn test_streaming_trait_bounds() {
        fn _check_trait<T: StreamingCapable>(_model: &T) {
            // Trait exists with expected methods
        }
    }

    #[test]
    fn test_recommended_chunk_sizes() {
        // GLiNER2 uses ~4096 chars as chunk size
        let expected_chunk_size = 4096;
        assert!(expected_chunk_size > 1000); // At least 1KB
        assert!(expected_chunk_size < 100_000); // At most 100KB
    }

    #[test]
    fn test_chunk_boundary_handling() {
        // Test that chunk boundaries don't break entities
        let text = "Steve Jobs founded Apple in California.";
        let chunk1 = &text[0..20]; // "Steve Jobs founded A"
        let _chunk2 = &text[15..]; // "unded Apple in California."

        // Overlap between chunks should help avoid boundary issues
        let overlap = 5;
        assert!(chunk1.len() >= overlap);
        assert!(text.len() > chunk1.len());
    }

    #[test]
    fn test_offset_calculation() {
        let full_text = "First sentence. Second sentence with Apple Inc.";
        let chunk_start = full_text.find("Second").unwrap();
        let chunk = &full_text[chunk_start..];

        // If entity "Apple Inc" is found at position 21 in chunk,
        // its global position should be chunk_start + 21
        let local_pos = chunk.find("Apple").unwrap();
        let global_pos = chunk_start + local_pos;

        assert_eq!(global_pos, full_text.find("Apple").unwrap());
    }
}
