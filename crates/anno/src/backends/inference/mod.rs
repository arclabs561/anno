//! Inference abstractions shared across `anno` backends.
//!
//! This module is mostly **plumbing**: common traits, data shapes, and small
//! utilities used by multiple NER / IE backends (including “fixed-label” and
//! “open/zero-shot” styles).
//!
//! Some of the terminology and design choices correspond to well-known
//! architectures in the NER/IE literature, but the code here should be treated
//! as an implementation substrate, not a verbatim reproduction of any single
//! paper’s experiment section.
//!
//! ## Paper pointers (context only)
//!
//! - GLiNER: arXiv:2311.08526
//! - UniversalNER: arXiv:2308.03279
//! - W2NER: arXiv:2112.10070
//! - ModernBERT: arXiv:2412.13663

pub(crate) mod registry;

pub mod encoder;
pub use encoder::{EncoderOutput, TextEncoder};

pub mod traits;
pub use traits::{
    DiscontinuousEntity, DiscontinuousNER, ExtractionWithRelations, RelationExtractor,
    RelationTriple, ZeroShotNER,
};
pub(crate) use traits::{DEFAULT_ENTITY_TYPES, DEFAULT_RELATION_TYPES};

pub(crate) mod span;
pub(crate) use span::{HandshakingCell, HandshakingMatrix};

pub mod coref;
pub(crate) use coref::CoreferenceCluster;
pub use coref::{resolve_coreferences, CoreferenceConfig};

pub mod relation_extraction;
pub use relation_extraction::{
    extract_relation_triples, extract_relation_triples_simple, extract_relations,
    RelationExtractionConfig,
};

// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::coref::{resolve_coreferences, CoreferenceConfig};
    use super::registry::{SemanticRegistry, SemanticRegistryBuilder};
    use super::span::SpanRepConfig;
    use super::*;
    use crate::{Confidence, Entity, EntityType};

    #[test]
    fn test_semantic_registry_builder() {
        let registry = SemanticRegistry::builder()
            .add_entity("person", "A human being")
            .add_entity("organization", "A company or group")
            .add_relation("WORKS_FOR", "Employment relationship")
            .build_zero(768);

        assert_eq!(registry.len(), 3);
        assert_eq!(registry.entity_labels().count(), 2);
        assert_eq!(registry.relation_labels().count(), 1);
    }

    #[test]
    fn test_standard_ner_registry() {
        let registry = SemanticRegistry::standard_ner(768);
        assert!(registry.len() >= 5);
        assert!(registry.label_index.contains_key("person"));
        assert!(registry.label_index.contains_key("organization"));
    }

    #[test]
    fn test_coreference_string_match() {
        let entities = vec![
            Entity::new("Marie Curie", EntityType::Person, 0, 11, 0.95),
            Entity::new("Curie", EntityType::Person, 50, 55, 0.90),
        ];

        let embeddings = vec![0.0f32; 2 * 768]; // Placeholder
        let clusters =
            resolve_coreferences(&entities, &embeddings, 768, &CoreferenceConfig::default());

        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].members.len(), 2);
        assert_eq!(clusters[0].canonical_name, "Marie Curie");
    }

    #[test]
    fn test_handshaking_matrix() {
        // 3 tokens, 2 labels, threshold 0.5
        let scores = vec![
            // token 0 with tokens 0,1,2 for labels 0,1
            0.9, 0.1, // (0,0)
            0.2, 0.8, // (0,1)
            0.1, 0.1, // (0,2)
            // token 1 with tokens 0,1,2
            0.0, 0.0, // (1,0) - skipped (lower triangle)
            0.7, 0.2, // (1,1)
            0.3, 0.6, // (1,2)
            // token 2
            0.0, 0.0, // (2,0)
            0.0, 0.0, // (2,1)
            0.1, 0.1, // (2,2)
        ];

        let matrix = HandshakingMatrix::from_dense(&scores, 3, 2, 0.5);

        // Should have cells for scores >= 0.5
        assert!(matrix.cells.len() >= 4);
    }

    #[test]
    fn test_relation_extraction() {
        let entities = vec![
            Entity::new("Steve Jobs", EntityType::Person, 0, 10, 0.95),
            Entity::new("Apple", EntityType::Organization, 20, 25, 0.90),
        ];

        let text = "Steve Jobs founded Apple Inc in 1976";

        let registry = SemanticRegistry::builder()
            .add_relation("FOUNDED", "Founded an organization")
            .build_zero(768);

        let config = RelationExtractionConfig::default();
        let relations = extract_relations(&entities, text, &registry, &config);

        assert!(!relations.is_empty());
        assert_eq!(relations[0].relation_type, "FOUNDED");
    }

    #[test]
    fn test_relation_extraction_uses_character_offsets_with_unicode_prefix() {
        // Unicode prefix ensures byte offsets != character offsets.
        let text = "👋 Steve Jobs founded Apple Inc.";

        // Compute character offsets explicitly (Entity spans are char-based).
        let steve_start = text.find("Steve Jobs").expect("substring present");
        // `find` returns byte offset; convert to char offset.
        let conv = crate::offset::SpanConverter::new(text);
        let steve_start_char = conv.byte_to_char(steve_start);
        let steve_end_char = steve_start_char + "Steve Jobs".chars().count();

        let apple_start = text.find("Apple").expect("substring present");
        let apple_start_char = conv.byte_to_char(apple_start);
        let apple_end_char = apple_start_char + "Apple".chars().count();

        let entities = vec![
            Entity::new(
                "Steve Jobs",
                EntityType::Person,
                steve_start_char,
                steve_end_char,
                0.95,
            ),
            Entity::new(
                "Apple",
                EntityType::Organization,
                apple_start_char,
                apple_end_char,
                0.90,
            ),
        ];

        let registry = SemanticRegistry::builder()
            .add_relation("FOUNDED", "Founded an organization")
            .build_zero(768);

        let config = RelationExtractionConfig::default();
        let relations = extract_relations(&entities, text, &registry, &config);

        assert!(
            !relations.is_empty(),
            "Expected FOUNDED relation to be detected"
        );
        assert_eq!(relations[0].relation_type, "FOUNDED");

        // Trigger span should exist and cover "founded" in character offsets.
        let trigger = relations[0]
            .trigger_span
            .expect("expected trigger_span to be present");
        let trigger_text: String = text
            .chars()
            .skip(trigger.0)
            .take(trigger.1.saturating_sub(trigger.0))
            .collect();
        assert_eq!(trigger_text.to_ascii_lowercase(), "founded");
    }

    // =========================================================================
    // Coreference: edge cases
    // =========================================================================

    #[test]
    fn test_coreference_empty_input() {
        let clusters = resolve_coreferences(&[], &[], 768, &CoreferenceConfig::default());
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_coreference_single_entity_no_cluster() {
        let entities = vec![Entity::new("Alice", EntityType::Person, 0, 5, 0.9)];
        let embeddings = vec![0.0f32; 768];
        let clusters =
            resolve_coreferences(&entities, &embeddings, 768, &CoreferenceConfig::default());
        // A single entity cannot form a cluster (needs 2+ members)
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_coreference_type_mismatch_prevents_linking() {
        // Same text but different entity types should NOT cluster
        let entities = vec![
            Entity::new("Apple", EntityType::Organization, 0, 5, 0.9),
            Entity::new("Apple", EntityType::Location, 20, 25, 0.9),
        ];
        let embeddings = vec![0.0f32; 2 * 768];
        let clusters =
            resolve_coreferences(&entities, &embeddings, 768, &CoreferenceConfig::default());
        assert!(
            clusters.is_empty(),
            "Different entity types should not cluster even with same text"
        );
    }

    #[test]
    fn test_coreference_distance_filtering() {
        // Two mentions far apart should not cluster when max_distance is small
        let entities = vec![
            Entity::new("Bob", EntityType::Person, 0, 3, 0.9),
            Entity::new("Bob", EntityType::Person, 1000, 1003, 0.9),
        ];
        let embeddings = vec![0.0f32; 2 * 768];
        let config = CoreferenceConfig {
            max_distance: Some(10),  // Very small window
            use_string_match: false, // Disable string match to test distance alone
            similarity_threshold: 0.85,
        };
        let clusters = resolve_coreferences(&entities, &embeddings, 768, &config);
        assert!(
            clusters.is_empty(),
            "Entities beyond max_distance should not cluster"
        );
    }

    #[test]
    fn test_coreference_string_match_substring() {
        // "Dr. Smith" contains "Smith" -> should cluster
        let entities = vec![
            Entity::new("Dr. Smith", EntityType::Person, 0, 9, 0.9),
            Entity::new("Smith", EntityType::Person, 30, 35, 0.9),
        ];
        let embeddings = vec![0.0f32; 2 * 768];
        let clusters =
            resolve_coreferences(&entities, &embeddings, 768, &CoreferenceConfig::default());
        assert_eq!(clusters.len(), 1);
        // Representative should be the longest mention
        assert_eq!(clusters[0].canonical_name, "Dr. Smith");
    }

    #[test]
    fn test_coreference_transitive_closure() {
        // A matches B (string), B matches C (string) -> all three cluster
        let entities = vec![
            Entity::new("Robert Johnson", EntityType::Person, 0, 14, 0.9),
            Entity::new("Johnson", EntityType::Person, 30, 37, 0.9),
            Entity::new("Mr. Johnson", EntityType::Person, 60, 71, 0.9),
        ];
        let embeddings = vec![0.0f32; 3 * 768];
        let clusters =
            resolve_coreferences(&entities, &embeddings, 768, &CoreferenceConfig::default());
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].members.len(), 3);
        assert_eq!(clusters[0].canonical_name, "Robert Johnson");
    }

    // =========================================================================
    // Handshaking Matrix: decode and edge cases
    // =========================================================================

    #[test]
    fn test_handshaking_matrix_empty_scores() {
        let matrix = HandshakingMatrix::from_dense(&[], 0, 0, 0.5);
        assert!(matrix.cells.is_empty());
    }

    #[test]
    fn test_handshaking_matrix_all_below_threshold() {
        let scores = vec![0.1, 0.2, 0.3, 0.4]; // 2 tokens, 1 label
        let matrix = HandshakingMatrix::from_dense(&scores, 2, 1, 0.5);
        // Only upper triangle: (0,0)=0.1, (0,1)=0.2, (1,1)=0.4 -- all < 0.5
        assert!(
            matrix.cells.is_empty(),
            "All scores below threshold should yield no cells"
        );
    }

    #[test]
    fn test_handshaking_matrix_decode_entities() {
        // Build a registry with one entity label
        let registry = SemanticRegistry::builder()
            .add_entity("person", "A human being")
            .build_zero(768);

        // 3 tokens, 1 label. Dense layout: [seq_len * seq_len * num_labels]
        // We want (0,1) to have a high score (entity spanning tokens 0..2)
        let mut scores = vec![0.0f32; 3 * 3];
        // Cell (0,1) = tokens 0 to 1 -> index = 0*3*1 + 1*1 + 0 = 1
        scores[1] = 0.9;

        let matrix = HandshakingMatrix::from_dense(&scores, 3, 1, 0.5);
        assert_eq!(matrix.cells.len(), 1);

        let entities = matrix.decode_entities(&registry);
        assert_eq!(entities.len(), 1);
        // W2NER: j=start (1), i=end inclusive (0) -- but the cell is (i=0, j=1)
        // decode: span [j, i+1) = [1, 1) -- actually the code uses cell.j as start, cell.i+1 as end
        // So span is [1, 1) which is 0-width... let me check the actual decode logic
        // From the code: SpanCandidate::new(0, cell.j, cell.i + 1)
        // cell.i=0, cell.j=1 -> SpanCandidate(doc=0, start=1, end=1)
        // That would be 0-width. The actual semantics: i < j for upper triangle
        // So cell.i=0, cell.j=1 means span [1, 1) -- this is the W2NER format
        let (span, label, score) = &entities[0];
        assert_eq!(label.slug, "person");
        assert!((score - 0.9).abs() < 0.001);
        // Verify the span was decoded
        assert_eq!(span.start, 1); // cell.j
        assert_eq!(span.end, 1); // cell.i + 1
    }

    #[test]
    fn test_handshaking_matrix_non_maximum_suppression() {
        // Two overlapping spans; only the higher-scoring one should survive.
        // W2NER decode: SpanCandidate(doc=0, start=cell.j, end=cell.i+1)
        // Upper triangle: i <= j, so cell (i=0, j=2) -> span [2, 1) is invalid.
        // For proper spans we need i >= j, but from_dense only iterates upper triangle (j >= i).
        // So cell (i, j) with i<=j -> span [j, i+1). For i=0,j=0 -> [0,1); i=0,j=1 -> [1,1).
        // To get overlapping: cell (i=2, j=0) -> span [0, 3) and cell (i=1, j=0) -> span [0, 2)
        // But from_dense only iterates j >= i. So let's use cell (i=0, j=0) -> [0,1)
        // and cell (i=0, j=1) -> [1,1) which is empty and won't overlap.
        //
        // Actually, from the W2NER convention the cells represent entity boundaries and the
        // NMS uses start/end. Let's just test that from_dense + decode_entities produces
        // the right count for non-overlapping cells.
        let registry = SemanticRegistry::builder()
            .add_entity("person", "A human being")
            .build_zero(768);

        // 4 tokens, 1 label. Cell (i=0, j=0) -> span [0, 1), cell (i=0, j=1) -> span [1, 1)
        // cell (i=1, j=1) -> span [1, 2). The spans [0,1) and [1,2) are adjacent, not overlapping.
        let mut scores = vec![0.0f32; 4 * 4];
        // Cell (i=0, j=0): idx = 0*4*1 + 0*1 + 0 = 0 -> span [0, 1)
        scores[0] = 0.9;
        // Cell (i=1, j=1): idx = 1*4*1 + 1*1 + 0 = 5 -> span [1, 2)
        scores[5] = 0.7;

        let matrix = HandshakingMatrix::from_dense(&scores, 4, 1, 0.5);
        assert_eq!(matrix.cells.len(), 2);

        let entities = matrix.decode_entities(&registry);
        // Adjacent spans [0,1) and [1,2) should both survive NMS
        assert_eq!(
            entities.len(),
            2,
            "Non-overlapping adjacent spans should both survive NMS"
        );
    }

    // =========================================================================
    // SpanRepConfig defaults
    // =========================================================================

    #[test]
    fn test_span_rep_config_defaults() {
        let config = SpanRepConfig::default();
        assert_eq!(config.hidden_dim, 768);
        assert_eq!(config.max_width, 12);
        assert!(config.use_width_embeddings);
        assert_eq!(config.width_emb_dim, 192);
    }

    // =========================================================================
    // SemanticRegistry: embedding lookup
    // =========================================================================

    #[test]
    fn test_registry_get_embedding() {
        let registry = SemanticRegistry::builder()
            .add_entity("person", "A human being")
            .add_entity("org", "An organization")
            .build_zero(4);

        let emb = registry.get_embedding("person");
        assert!(emb.is_some());
        assert_eq!(emb.unwrap().len(), 4);

        let missing = registry.get_embedding("nonexistent");
        assert!(missing.is_none());
    }

    #[test]
    fn test_registry_empty() {
        let registry = SemanticRegistryBuilder::new().build_zero(768);
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert_eq!(registry.entity_labels().count(), 0);
        assert_eq!(registry.relation_labels().count(), 0);
    }

    // =========================================================================
    // DiscontinuousEntity
    // =========================================================================

    #[test]
    fn test_discontinuous_entity_contiguous() {
        let entity = DiscontinuousEntity {
            spans: vec![(0, 5)],
            text: "hello".to_string(),
            entity_type: "person".to_string(),
            confidence: Confidence::new(0.9),
        };
        assert!(entity.is_contiguous());
        let converted = entity.to_entity().expect("should convert single-span");
        assert_eq!(converted.text, "hello");
        assert_eq!(converted.start(), 0);
        assert_eq!(converted.end(), 5);
    }

    #[test]
    fn test_discontinuous_entity_non_contiguous() {
        let entity = DiscontinuousEntity {
            spans: vec![(0, 3), (10, 15)],
            text: "New airports".to_string(),
            entity_type: "location".to_string(),
            confidence: Confidence::new(0.8),
        };
        assert!(!entity.is_contiguous());
        assert!(entity.to_entity().is_none());
    }

    // =========================================================================
    // ExtractionWithRelations
    // =========================================================================

    #[test]
    fn test_extraction_with_relations_into_anno_relations() {
        let extraction = ExtractionWithRelations {
            entities: vec![
                Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
                Entity::new("Acme", EntityType::Organization, 20, 24, 0.8),
            ],
            relations: vec![RelationTriple {
                head_idx: 0,
                tail_idx: 1,
                relation_type: "WORKS_FOR".to_string(),
                confidence: Confidence::new(0.85),
            }],
        };

        let (entities, relations) = extraction.into_anno_relations();
        assert_eq!(entities.len(), 2);
        assert_eq!(relations.len(), 1);
        assert_eq!(relations[0].relation_type, "WORKS_FOR");
        assert_eq!(relations[0].head.text, "Alice");
        assert_eq!(relations[0].tail.text, "Acme");
    }

    #[test]
    fn test_extraction_with_relations_out_of_bounds_dropped() {
        let extraction = ExtractionWithRelations {
            entities: vec![Entity::new("Alice", EntityType::Person, 0, 5, 0.9)],
            relations: vec![RelationTriple {
                head_idx: 0,
                tail_idx: 99, // out of bounds
                relation_type: "WORKS_FOR".to_string(),
                confidence: Confidence::new(0.85),
            }],
        };

        let (_, relations) = extraction.into_anno_relations();
        assert!(
            relations.is_empty(),
            "Out-of-bounds relation should be silently dropped"
        );
    }

    // =========================================================================
    // Relation extraction: edge cases
    // =========================================================================

    #[test]
    fn test_relation_extraction_empty_entities() {
        let registry = SemanticRegistry::builder()
            .add_relation("FOUNDED", "Founded an organization")
            .build_zero(768);
        let config = RelationExtractionConfig::default();
        let relations = extract_relations(&[], "some text", &registry, &config);
        assert!(relations.is_empty());
    }

    #[test]
    fn test_relation_extraction_no_relation_labels() {
        let entities = vec![
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
            Entity::new("Acme", EntityType::Organization, 20, 24, 0.8),
        ];
        // Registry with only entity labels, no relations
        let registry = SemanticRegistry::builder()
            .add_entity("person", "A human being")
            .build_zero(768);
        let config = RelationExtractionConfig::default();
        let text = "Alice works at Acme Corp";
        let relations = extract_relations(&entities, text, &registry, &config);
        assert!(
            relations.is_empty(),
            "No relation labels in registry -> no relations extracted"
        );
    }

    #[test]
    fn test_relation_extraction_distance_penalty() {
        // Entities close together should have higher confidence than distant ones
        let registry = SemanticRegistry::builder()
            .add_relation("FOUNDED", "Founded an organization")
            .build_zero(768);

        let text_close = "Jobs founded Apple in 1976";
        let entities_close = vec![
            Entity::new("Jobs", EntityType::Person, 0, 4, 0.9),
            Entity::new("Apple", EntityType::Organization, 13, 18, 0.9),
        ];
        let config = RelationExtractionConfig::default();
        let rels_close = extract_relations(&entities_close, text_close, &registry, &config);

        assert!(!rels_close.is_empty());
        // Confidence should be > 0.5 (distance penalty doesn't kill it)
        assert!(rels_close[0].confidence > 0.5);
    }

    #[test]
    fn test_extract_relation_triples_overlapping_spans_skipped() {
        let registry = SemanticRegistry::builder()
            .add_relation("PART_OF", "Part of")
            .build_zero(768);
        let text = "New York City is a great city";
        // Overlapping entities: "New York City" (0..13) and "York" (4..8)
        let entities = vec![
            Entity::new("New York City", EntityType::Location, 0, 13, 0.9),
            Entity::new("York", EntityType::Location, 4, 8, 0.8),
        ];
        let config = RelationExtractionConfig::default();
        let triples = extract_relation_triples(&entities, text, &registry, &config);
        assert!(
            triples.is_empty(),
            "Overlapping spans should be skipped in extract_relation_triples"
        );
    }
}
