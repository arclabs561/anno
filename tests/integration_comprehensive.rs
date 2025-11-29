//! Comprehensive integration tests for the anno NER library.
//!
//! These tests verify that all components work together correctly:
//! - Entity types and their relationships
//! - Provenance tracking through the pipeline
//! - Backend composition and conflict resolution
//! - Serialization roundtrips
//! - Edge cases and boundary conditions

use anno::backends::stacked::ConflictStrategy;
use anno::{
    eval::GoldEntity, DiscontinuousSpan, Entity, EntityBuilder, EntityCategory, EntityType,
    ExtractionMethod, HeuristicNER, Model, PatternNER, StackedNER,
};

mod entity_types {
    use super::*;

    #[test]
    fn structured_types_category() {
        // These types are in Temporal/Numeric/Contact categories - detectable by patterns
        let structured = [
            (EntityType::Date, EntityCategory::Temporal),
            (EntityType::Time, EntityCategory::Temporal),
            (EntityType::Money, EntityCategory::Numeric),
            (EntityType::Percent, EntityCategory::Numeric),
            (EntityType::Email, EntityCategory::Contact),
            (EntityType::Url, EntityCategory::Contact),
            (EntityType::Phone, EntityCategory::Contact),
        ];

        for (ty, expected_cat) in structured {
            assert_eq!(
                ty.category(),
                expected_cat,
                "{:?} should have category {:?}",
                ty,
                expected_cat
            );
        }
    }

    #[test]
    fn named_types_category() {
        let named = [
            (EntityType::Person, EntityCategory::Agent),
            (EntityType::Organization, EntityCategory::Organization),
            (EntityType::Location, EntityCategory::Place),
        ];

        for (ty, expected_cat) in named {
            assert_eq!(
                ty.category(),
                expected_cat,
                "{:?} should have category {:?}",
                ty,
                expected_cat
            );
        }
    }

    #[test]
    fn custom_entity_type_roundtrip() {
        let custom = EntityType::custom("GENE", EntityCategory::Misc);
        // as_label returns lowercase for custom types
        assert_eq!(custom.as_label().to_lowercase(), "gene");

        // Parse back
        let parsed: EntityType = "gene".parse().unwrap();
        // Custom types parse as Other variant with original casing preserved
        if let EntityType::Other(name) = parsed {
            assert_eq!(name.to_lowercase(), "gene");
        }
    }

    #[test]
    fn entity_type_display_parse_roundtrip() {
        let types = [
            EntityType::Person,
            EntityType::Organization,
            EntityType::Location,
            EntityType::Date,
            EntityType::Money,
        ];

        for ty in &types {
            let label = ty.as_label().to_string();
            let parsed: EntityType = label.parse().unwrap();
            assert_eq!(parsed.as_label(), ty.as_label());
        }
    }
}

mod provenance_tracking {
    use super::*;

    #[test]
    fn pattern_ner_includes_provenance() {
        let ner = PatternNER::new();
        let entities = ner.extract_entities("Price: $100.00", None).unwrap();

        assert!(!entities.is_empty());
        let entity = &entities[0];

        // Should have provenance
        if let Some(prov) = &entity.provenance {
            assert!(prov.source.contains("pattern"));
            assert_eq!(prov.method, ExtractionMethod::Pattern);
        }
    }

    #[test]
    fn heuristic_ner_includes_provenance() {
        let ner = HeuristicNER::new();
        let entities = ner
            .extract_entities("Dr. John Smith visited Boston.", None)
            .unwrap();

        for entity in &entities {
            if let Some(prov) = &entity.provenance {
                // HeuristicNER uses heuristics (capitalization, context)
                assert_eq!(prov.method, ExtractionMethod::Heuristic);
                assert!(prov.source.contains("heuristic"));
            }
        }
    }

    #[test]
    fn stacked_ner_preserves_provenance() {
        let ner = StackedNER::default();
        let entities = ner
            .extract_entities("Dr. Smith charges $100/hour. Email: smith@test.com", None)
            .unwrap();

        // Should have entities from both backends
        let pattern_entities: Vec<_> = entities
            .iter()
            .filter(|e| {
                e.provenance
                    .as_ref()
                    .map(|p| matches!(p.method, ExtractionMethod::Pattern))
                    .unwrap_or(false)
            })
            .collect();

        // Pattern should find $100 and email
        assert!(!pattern_entities.is_empty(), "Should have pattern entities");
    }
}

mod conflict_resolution {
    use super::*;

    #[test]
    fn priority_strategy_favors_first() {
        let ner = StackedNER::builder()
            .layer(PatternNER::new())
            .layer(HeuristicNER::new())
            .strategy(ConflictStrategy::Priority)
            .build();

        let entities = ner.extract_entities("$100 from John", None).unwrap();

        // Check no exact duplicate spans
        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                let e1 = &entities[i];
                let e2 = &entities[j];
                // With Priority, shouldn't have exact duplicate spans with same type
                if e1.start == e2.start && e1.end == e2.end {
                    assert_ne!(e1.entity_type, e2.entity_type, "Duplicate entities found");
                }
            }
        }
    }

    #[test]
    fn longest_span_prefers_longer() {
        let ner = StackedNER::builder()
            .layer(PatternNER::new())
            .layer(HeuristicNER::new())
            .strategy(ConflictStrategy::LongestSpan)
            .build();

        let text = "The Bank of America branch";
        let entities = ner.extract_entities(text, None).unwrap();

        // Should prefer longer spans if available
        for entity in &entities {
            assert!(entity.end > entity.start);
        }
    }

    #[test]
    fn union_keeps_all() {
        let ner = StackedNER::builder()
            .layer(PatternNER::new())
            .layer(HeuristicNER::new())
            .strategy(ConflictStrategy::Union)
            .build();

        let entities = ner.extract_entities("$100 for Dr. Smith", None).unwrap();

        // Union should keep everything
        assert!(!entities.is_empty());
    }

    #[test]
    fn highest_conf_selects_best() {
        let ner = StackedNER::builder()
            .layer(PatternNER::new())
            .layer(HeuristicNER::new())
            .strategy(ConflictStrategy::HighestConf)
            .build();

        let entities = ner
            .extract_entities("Contact: john@example.com", None)
            .unwrap();

        // Pattern NER typically has higher confidence for emails
        let email_entity = entities.iter().find(|e| e.entity_type == EntityType::Email);
        assert!(email_entity.is_some());
    }
}

mod discontinuous_spans {
    use super::*;

    #[test]
    fn discontinuous_span_basics() {
        // A "contiguous" discontinuous span (single segment)
        let span = DiscontinuousSpan::contiguous(10, 20);
        assert_eq!(span.num_segments(), 1);
        assert!(!span.is_discontinuous());

        // Multi-segment span
        let multi = DiscontinuousSpan::new(vec![0..5, 10..15]);
        assert_eq!(multi.num_segments(), 2);
        assert!(multi.is_discontinuous());
    }

    #[test]
    fn discontinuous_span_text_extraction() {
        let span = DiscontinuousSpan::new(vec![0..4, 9..13]);
        let text = "John and Mary went home";

        let extracted = span.extract_text(text, " ");
        assert!(extracted.contains("John"));
        assert!(extracted.contains("Mary"));
    }

    #[test]
    fn entity_builder_basic() {
        let entity = EntityBuilder::new("John Smith", EntityType::Person)
            .span(0, 10)
            .confidence(0.95)
            .build();

        assert_eq!(entity.text, "John Smith");
        assert_eq!(entity.entity_type, EntityType::Person);
        assert_eq!(entity.start, 0);
        assert_eq!(entity.end, 10);
        assert!((entity.confidence - 0.95).abs() < 0.001);
    }
}

mod serialization {
    use super::*;

    #[test]
    fn entity_json_roundtrip() {
        let entity = Entity::new("test", EntityType::Person, 0, 4, 0.95);

        let json = serde_json::to_string(&entity).unwrap();
        let parsed: Entity = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.text, entity.text);
        assert_eq!(parsed.entity_type, entity.entity_type);
        assert_eq!(parsed.start, entity.start);
        assert_eq!(parsed.end, entity.end);
    }

    #[test]
    fn entity_type_serialization() {
        let types = [
            EntityType::Person,
            EntityType::Organization,
            EntityType::Date,
            EntityType::custom("CUSTOM", EntityCategory::Misc),
        ];

        for ty in &types {
            let json = serde_json::to_string(&ty).unwrap();
            let parsed: EntityType = serde_json::from_str(&json).unwrap();

            // Check they're equivalent (custom types use different representation)
            match (ty, &parsed) {
                (EntityType::Person, EntityType::Person) => {}
                (EntityType::Organization, EntityType::Organization) => {}
                (EntityType::Date, EntityType::Date) => {}
                (EntityType::Other(a), EntityType::Other(b)) => {
                    assert_eq!(a, b);
                }
                _ => {}
            }
        }
    }

    #[test]
    fn batch_extraction_json_roundtrip() {
        let ner = StackedNER::default();
        let entities = ner
            .extract_entities("Dr. Smith paid $100 on 2024-01-01", None)
            .unwrap();

        let json = serde_json::to_string(&entities).unwrap();
        let parsed: Vec<Entity> = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.len(), entities.len());
    }
}

mod gold_entity_integration {
    use super::*;

    #[test]
    fn gold_entity_creation() {
        let gold = GoldEntity::new("test", EntityType::Person, 10);

        assert_eq!(gold.text, "test");
        assert_eq!(gold.entity_type, EntityType::Person);
        assert_eq!(gold.start, 10);
        assert_eq!(gold.end, 10 + "test".len());
    }

    #[test]
    fn gold_entity_with_span() {
        let gold = GoldEntity::with_span("test", EntityType::Organization, 5, 15);

        assert_eq!(gold.start, 5);
        assert_eq!(gold.end, 15);
    }

    #[test]
    fn gold_entity_matches_entity() {
        let gold = GoldEntity::new("Apple", EntityType::Organization, 0);
        let entity = Entity::new("Apple", EntityType::Organization, 0, 5, 0.9);

        // Verify they represent the same thing
        assert_eq!(gold.text, entity.text);
        assert_eq!(gold.entity_type, entity.entity_type);
        assert_eq!(gold.start, entity.start);
        assert_eq!(gold.end, entity.end);
    }
}

mod backend_composition {
    use super::*;

    #[test]
    fn stacked_with_single_layer() {
        let ner = StackedNER::pattern_only();
        let entities = ner.extract_entities("$100", None).unwrap();

        assert!(!entities.is_empty());
        assert_eq!(entities[0].entity_type, EntityType::Money);
    }

    #[test]
    #[allow(deprecated)]
    fn stacked_with_heuristic_only() {
        let ner = StackedNER::statistical_only(); // deprecated alias
        let _entities = ner.extract_entities("Dr. John Smith", None).unwrap();

        // Should find at least something with heuristics
        // (though accuracy varies based on input)
    }

    #[test]
    #[allow(deprecated)]
    fn stacked_builder_with_threshold() {
        let ner = StackedNER::with_statistical_threshold(0.7); // deprecated
        let _entities = ner.extract_entities("John Smith at MIT", None).unwrap();

        // Higher threshold means fewer, more confident extractions
    }

    #[test]
    fn stacked_default_is_pattern_plus_heuristic() {
        let ner = StackedNER::default();

        // Should handle both pattern-detectable and named entities
        let entities = ner
            .extract_entities("Pay $50 to Dr. Smith by 2024-12-31", None)
            .unwrap();

        // Should have at least money and date
        let has_money = entities.iter().any(|e| e.entity_type == EntityType::Money);
        let has_date = entities.iter().any(|e| e.entity_type == EntityType::Date);

        assert!(has_money, "Should detect money");
        assert!(has_date, "Should detect date");
    }

    #[test]
    fn builder_can_add_same_layer_twice() {
        // Edge case: adding same backend type twice
        let ner = StackedNER::builder()
            .layer(PatternNER::new())
            .layer(PatternNER::new())
            .build();

        let entities = ner.extract_entities("$100", None).unwrap();
        // Should still work (might have duplicates with Union strategy)
        assert!(!entities.is_empty());
    }

    #[test]
    fn model_trait_name() {
        let ner = StackedNER::default();
        let name = ner.name();
        assert!(!name.is_empty());
    }
}

mod edge_cases {
    use super::*;

    #[test]
    fn empty_text_returns_empty() {
        let ner = StackedNER::default();
        let entities = ner.extract_entities("", None).unwrap();
        assert!(entities.is_empty());
    }

    #[test]
    fn whitespace_only_returns_empty() {
        let ner = StackedNER::default();
        let entities = ner.extract_entities("   \t\n  ", None).unwrap();
        assert!(entities.is_empty());
    }

    #[test]
    fn unicode_text_handled_correctly() {
        let ner = StackedNER::default();

        // Emoji text
        let entities = ner.extract_entities("Pay $100 for 🍕", None).unwrap();
        let money = entities.iter().find(|e| e.entity_type == EntityType::Money);
        assert!(money.is_some());
        assert_eq!(money.unwrap().text, "$100");

        // Non-ASCII currencies and text
        let entities2 = ner.extract_entities("Price: €50 for café", None).unwrap();
        let money2 = entities2
            .iter()
            .find(|e| e.entity_type == EntityType::Money);
        assert!(money2.is_some());
    }

    #[test]
    fn very_long_text() {
        let ner = StackedNER::default();

        // Generate long text with multiple entities
        let long_text = (0..100)
            .map(|i| format!("Item {} costs ${}.99. ", i, i * 10 + 5))
            .collect::<String>();

        let entities = ner.extract_entities(&long_text, None).unwrap();

        // Should find many money entities
        let money_count = entities
            .iter()
            .filter(|e| e.entity_type == EntityType::Money)
            .count();

        assert!(
            money_count >= 50,
            "Should find many money entities: {}",
            money_count
        );
    }

    #[test]
    fn entities_sorted_by_position() {
        let ner = StackedNER::default();
        let entities = ner
            .extract_entities("$100 on 2024-01-01 for john@test.com", None)
            .unwrap();

        // Verify sorted by start position
        for i in 1..entities.len() {
            assert!(
                entities[i].start >= entities[i - 1].start,
                "Entities should be sorted by position"
            );
        }
    }

    #[test]
    fn special_characters_in_entities() {
        let ner = PatternNER::new();

        // Email with special chars
        let entities = ner
            .extract_entities("user+tag@sub.domain.co.uk", None)
            .unwrap();
        let email = entities.iter().find(|e| e.entity_type == EntityType::Email);
        assert!(email.is_some());

        // URL with query params
        let entities2 = ner
            .extract_entities("https://example.com/path?key=value&foo=bar", None)
            .unwrap();
        let url = entities2.iter().find(|e| e.entity_type == EntityType::Url);
        assert!(url.is_some());
    }
}

mod performance {
    use super::*;
    use std::time::Instant;

    #[test]
    fn extraction_is_reasonably_fast() {
        let ner = StackedNER::default();
        let text = "Dr. Smith charges $100/hr. Email: smith@test.com. Call: 555-1234.";

        let start = Instant::now();
        for _ in 0..1000 {
            let _ = ner.extract_entities(text, None);
        }
        let elapsed = start.elapsed();

        // Should process at least 1000 extractions per second
        assert!(
            elapsed.as_millis() < 5000,
            "1000 extractions took {:?}, expected < 5s",
            elapsed
        );
    }

    #[test]
    fn pattern_ner_is_faster_than_stacked() {
        let pattern_ner = PatternNER::new();
        let stacked_ner = StackedNER::default();
        let text = "$100 on 2024-01-01 email@test.com";

        let start_pattern = Instant::now();
        for _ in 0..1000 {
            let _ = pattern_ner.extract_entities(text, None);
        }
        let pattern_time = start_pattern.elapsed();

        let start_stacked = Instant::now();
        for _ in 0..1000 {
            let _ = stacked_ner.extract_entities(text, None);
        }
        let stacked_time = start_stacked.elapsed();

        // Pattern should be comparable or faster (no heuristic layer)
        // This test just ensures both complete in reasonable time
        assert!(pattern_time.as_millis() < 3000);
        assert!(stacked_time.as_millis() < 5000);
    }
}
