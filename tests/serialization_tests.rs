//! Serialization and deserialization tests.
//!
//! Tests JSON roundtrip, schema stability, and format compatibility.

use anno::{
    Entity, EntityCategory, EntityType, ExtractionMethod, HierarchicalConfidence, Model,
    Provenance, RegexNER, Span, StackedNER,
};

// =============================================================================
// Entity Serialization
// =============================================================================

mod entity_serde {
    use super::*;

    fn sample_entity() -> Entity {
        Entity {
            text: "Apple Inc.".to_string(),
            entity_type: EntityType::Organization,
            start: 0,
            end: 10,
            confidence: 0.95,
            normalized: Some("Apple Inc".to_string()),
            provenance: Some(Provenance {
                source: "test".into(),
                method: ExtractionMethod::Pattern,
                pattern: Some("ORG_SUFFIX".into()),
                raw_confidence: Some(0.95),
                model_version: None,
                timestamp: None,
            }),
            kb_id: Some("Q312".to_string()),
            canonical_id: Some(42),
            hierarchical_confidence: Some(HierarchicalConfidence::new(0.9, 0.95, 0.92)),
            visual_span: None,
            discontinuous_span: None,
            valid_from: None,
            valid_until: None,
            viewport: None,
        }
    }

    #[test]
    fn entity_to_json() {
        let entity = sample_entity();
        let json = serde_json::to_string(&entity).unwrap();
        assert!(json.contains("Apple Inc."));
        assert!(json.contains("Organization"));
    }

    #[test]
    fn entity_from_json() {
        let json = r#"{
            "text": "Apple Inc.",
            "entity_type": "Organization",
            "start": 0,
            "end": 10,
            "confidence": 0.95
        }"#;
        let entity: Entity = serde_json::from_str(json).unwrap();
        assert_eq!(entity.text, "Apple Inc.");
        assert_eq!(entity.entity_type, EntityType::Organization);
    }

    #[test]
    fn entity_roundtrip() {
        let original = sample_entity();
        let json = serde_json::to_string(&original).unwrap();
        let restored: Entity = serde_json::from_str(&json).unwrap();

        assert_eq!(original.text, restored.text);
        assert_eq!(original.entity_type, restored.entity_type);
        assert_eq!(original.start, restored.start);
        assert_eq!(original.end, restored.end);
        assert!((original.confidence - restored.confidence).abs() < 0.001);
    }

    #[test]
    fn entity_pretty_print() {
        let entity = sample_entity();
        let json = serde_json::to_string_pretty(&entity).unwrap();
        assert!(json.contains('\n'));
    }

    #[test]
    fn entity_minimal_json() {
        // Only required fields
        let json = r#"{
            "text": "test",
            "entity_type": "Person",
            "start": 0,
            "end": 4,
            "confidence": 0.5
        }"#;
        let entity: Entity = serde_json::from_str(json).unwrap();
        assert!(entity.provenance.is_none());
        assert!(entity.kb_id.is_none());
    }

    #[test]
    fn entity_list_serialization() {
        let entities = vec![
            Entity::new("John", EntityType::Person, 0, 4, 0.9),
            Entity::new("$100", EntityType::Money, 10, 14, 0.95),
        ];

        let json = serde_json::to_string(&entities).unwrap();
        let restored: Vec<Entity> = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.len(), 2);
    }
}

// =============================================================================
// EntityType Serialization
// =============================================================================

mod entity_type_serde {
    use super::*;

    #[test]
    fn standard_types() {
        let types = vec![
            EntityType::Person,
            EntityType::Organization,
            EntityType::Location,
            EntityType::Date,
            EntityType::Time,
            EntityType::Money,
            EntityType::Percent,
            EntityType::Email,
            EntityType::Url,
            EntityType::Phone,
        ];

        for ty in types {
            let json = serde_json::to_string(&ty).unwrap();
            let restored: EntityType = serde_json::from_str(&json).unwrap();
            assert_eq!(ty, restored);
        }
    }

    #[test]
    fn custom_type() {
        let ty = EntityType::Custom {
            name: "ProductCode".to_string(),
            category: EntityCategory::Misc,
        };
        let json = serde_json::to_string(&ty).unwrap();
        let restored: EntityType = serde_json::from_str(&json).unwrap();
        assert_eq!(ty, restored);
    }

    #[test]
    fn other_type() {
        let ty = EntityType::Other("MISC".to_string());
        let json = serde_json::to_string(&ty).unwrap();
        let restored: EntityType = serde_json::from_str(&json).unwrap();
        assert_eq!(ty, restored);
    }
}

// =============================================================================
// Provenance Serialization
// =============================================================================

mod provenance_serde {
    use super::*;

    #[test]
    fn provenance_roundtrip() {
        let prov = Provenance {
            source: "pattern".into(),
            method: ExtractionMethod::Pattern,
            pattern: Some("EMAIL".into()),
            raw_confidence: Some(0.98),
            model_version: None,
            timestamp: None,
        };

        let json = serde_json::to_string(&prov).unwrap();
        let restored: Provenance = serde_json::from_str(&json).unwrap();

        assert_eq!(prov.source, restored.source);
    }

    #[test]
    #[allow(deprecated)]
    fn extraction_method_all_variants() {
        let methods = vec![
            ExtractionMethod::Pattern,
            ExtractionMethod::Neural,
            ExtractionMethod::SoftLexicon,
            ExtractionMethod::GatedEnsemble,
            ExtractionMethod::Consensus,
            ExtractionMethod::Heuristic,
            ExtractionMethod::Unknown,
            // Legacy variants (deprecated but still need to serialize)
            ExtractionMethod::ML,
            ExtractionMethod::Rule,
            ExtractionMethod::Ensemble,
            ExtractionMethod::Lexicon,
        ];

        for method in methods {
            let json = serde_json::to_string(&method).unwrap();
            let restored: ExtractionMethod = serde_json::from_str(&json).unwrap();
            assert_eq!(method, restored);
        }
    }
}

// =============================================================================
// Confidence Serialization
// =============================================================================

mod confidence_serde {
    use super::*;

    #[test]
    fn hierarchical_confidence_roundtrip() {
        let conf = HierarchicalConfidence::new(0.9, 0.85, 0.88);
        let json = serde_json::to_string(&conf).unwrap();
        let restored: HierarchicalConfidence = serde_json::from_str(&json).unwrap();

        assert!((conf.combined() - restored.combined()).abs() < 0.001);
    }
}

// =============================================================================
// Span Serialization
// =============================================================================

mod span_serde {
    use super::*;

    #[test]
    fn text_span_roundtrip() {
        let span = Span::Text { start: 0, end: 10 };
        let json = serde_json::to_string(&span).unwrap();
        let restored: Span = serde_json::from_str(&json).unwrap();
        assert_eq!(span, restored);
    }

    #[test]
    fn bbox_span_roundtrip() {
        let span = Span::BoundingBox {
            x: 100.0,
            y: 200.0,
            width: 50.0,
            height: 20.0,
            page: Some(1),
        };
        let json = serde_json::to_string(&span).unwrap();
        let restored: Span = serde_json::from_str(&json).unwrap();
        assert_eq!(span, restored);
    }
}

// =============================================================================
// Extracted Entities Serialization
// =============================================================================

mod extraction_results_serde {
    use super::*;

    #[test]
    fn extracted_entities_roundtrip() {
        let ner = RegexNER::new();
        let entities = ner
            .extract_entities("Cost: $100 on 2024-01-15", None)
            .unwrap();

        let json = serde_json::to_string(&entities).unwrap();
        let restored: Vec<Entity> = serde_json::from_str(&json).unwrap();

        assert_eq!(entities.len(), restored.len());
        for (orig, rest) in entities.iter().zip(restored.iter()) {
            assert_eq!(orig.text, rest.text);
            assert_eq!(orig.entity_type, rest.entity_type);
        }
    }

    #[test]
    fn empty_extraction() {
        let ner = RegexNER::new();
        let entities = ner.extract_entities("no entities here", None).unwrap();
        let json = serde_json::to_string(&entities).unwrap();
        assert_eq!(json, "[]");
    }

    #[test]
    fn complex_extraction_roundtrip() {
        let ner = StackedNER::new();
        let text = "Dr. Smith charges $200/hr. Contact: smith@test.com";
        let entities = ner.extract_entities(text, None).unwrap();

        let json = serde_json::to_string_pretty(&entities).unwrap();
        let restored: Vec<Entity> = serde_json::from_str(&json).unwrap();

        assert_eq!(entities.len(), restored.len());
    }
}

// =============================================================================
// Schema Compatibility
// =============================================================================

mod schema_compatibility {
    use super::*;

    #[test]
    fn accept_unknown_fields() {
        // Future-proofing: should ignore unknown fields
        let json = r#"{
            "text": "test",
            "entity_type": "Person",
            "start": 0,
            "end": 4,
            "confidence": 0.5,
            "future_field": "ignored"
        }"#;
        // This should not fail - serde default behavior
        let entity: Entity = serde_json::from_str(json).unwrap();
        assert_eq!(entity.text, "test");
    }

    #[test]
    fn null_optional_fields() {
        let json = r#"{
            "text": "test",
            "entity_type": "Person",
            "start": 0,
            "end": 4,
            "confidence": 0.5,
            "normalized": null,
            "provenance": null,
            "kb_id": null
        }"#;
        let entity: Entity = serde_json::from_str(json).unwrap();
        assert!(entity.normalized.is_none());
    }
}
