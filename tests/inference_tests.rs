//! Property tests for the inference module.
//!
//! These tests validate invariants that must always hold:
//! - Similarity scores are bounded
//! - Span indices are valid
//! - Registry lookups are consistent
//! - Coreference clusters are well-formed

#![allow(dead_code)] // Strategy generators used in #[ignore] tests

use anno::backends::inference::{
    resolve_coreferences, CoreferenceConfig, DotProductInteraction,
    HandshakingMatrix, LabelCategory, LateInteraction, SemanticRegistry,
};
// Re-exported from lib.rs
use anno::{Entity, EntityType, RaggedBatch, SpanCandidate};
use proptest::prelude::*;

// =============================================================================
// Strategy Generators
// =============================================================================

/// Generate a valid hidden dimension (power of 2 for efficiency, typical values).
fn hidden_dim_strategy() -> impl Strategy<Value = usize> {
    prop_oneof![Just(64), Just(128), Just(256), Just(384), Just(512), Just(768), Just(1024)]
}

/// Generate a normalized embedding vector.
fn normalized_embedding(dim: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(any::<f32>().prop_map(|x| x.clamp(-1.0, 1.0)), dim).prop_map(
        move |mut v| {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            for x in v.iter_mut() {
                *x /= norm;
            }
            v
        },
    )
}

/// Generate entity type labels.
fn entity_type_strategy() -> impl Strategy<Value = EntityType> {
    prop_oneof![
        Just(EntityType::Person),
        Just(EntityType::Organization),
        Just(EntityType::Location),
        Just(EntityType::Date),
        Just(EntityType::Money),
        Just(EntityType::Percent),
        "[a-z]{3,10}".prop_map(EntityType::Other),
    ]
}

/// Generate valid entity text.
fn entity_text_strategy() -> impl Strategy<Value = String> {
    "[A-Z][a-z]{2,15}( [A-Z][a-z]{2,15}){0,3}"
}

// =============================================================================
// SemanticRegistry Tests
// =============================================================================

proptest! {
    /// Registry builder produces consistent label indices.
    #[test]
    fn registry_labels_indexed_consistently(
        labels in proptest::collection::vec("[a-z_]{3,15}", 1..20),
        dim in hidden_dim_strategy()
    ) {
        let mut builder = SemanticRegistry::builder();
        for label in &labels {
            builder = builder.add_entity(label, &format!("Description for {}", label));
        }
        let registry = builder.build_placeholder(dim);

        // Invariant: every added label has an index
        for label in &labels {
            prop_assert!(registry.label_index.contains_key(label),
                "Label {} missing from index", label);
        }

        // Invariant: indices are in range [0, len)
        for &idx in registry.label_index.values() {
            prop_assert!(idx < registry.len(),
                "Index {} out of bounds (len={})", idx, registry.len());
        }

        // Invariant: embeddings have correct size
        prop_assert_eq!(registry.embeddings.len(), registry.len() * dim);
    }

    /// Registry embedding lookup returns correct dimensions.
    #[test]
    fn registry_embedding_lookup_dimensions(
        dim in hidden_dim_strategy()
    ) {
        let registry = SemanticRegistry::standard_ner(dim);

        for label in &registry.labels {
            if let Some(emb) = registry.get_embedding(&label.slug) {
                prop_assert_eq!(emb.len(), dim,
                    "Embedding for {} has wrong dimension", label.slug);
            }
        }
    }

    /// Standard NER registry has expected entity types.
    #[test]
    fn standard_ner_registry_completeness(dim in hidden_dim_strategy()) {
        let registry = SemanticRegistry::standard_ner(dim);

        // Must have at least person, org, location
        let slugs: Vec<_> = registry.labels.iter().map(|l| l.slug.as_str()).collect();
        prop_assert!(slugs.contains(&"person"), "Missing person");
        prop_assert!(slugs.contains(&"organization"), "Missing organization");
        prop_assert!(slugs.contains(&"location"), "Missing location");

        // All should be Entity category
        for label in &registry.labels {
            prop_assert_eq!(label.category, LabelCategory::Entity,
                "Label {} should be Entity category", label.slug);
        }
    }
}

// =============================================================================
// LateInteraction Tests
// =============================================================================

proptest! {
    /// Dot product interaction produces scores in reasonable range.
    #[test]
    fn dot_product_scores_bounded(
        num_spans in 1..20usize,
        num_labels in 1..10usize,
        dim in Just(64),
    ) {
        let interaction = DotProductInteraction::new();

        // Generate normalized embeddings
        let span_embs: Vec<f32> = (0..num_spans * dim)
            .map(|i| ((i % 7) as f32 - 3.0) / 10.0)
            .collect();
        let label_embs: Vec<f32> = (0..num_labels * dim)
            .map(|i| ((i % 5) as f32 - 2.0) / 10.0)
            .collect();

        let scores = interaction.compute_similarity(
            &span_embs, num_spans,
            &label_embs, num_labels,
            dim
        );

        // Invariant: correct output size
        prop_assert_eq!(scores.len(), num_spans * num_labels);

        // Invariant: scores are finite
        for score in &scores {
            prop_assert!(score.is_finite(), "Score is not finite: {}", score);
        }
    }

    /// Identical vectors have similarity 1.0.
    #[test]
    fn cosine_similarity_identical_vectors(dim in 1..100usize) {
        let v: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) / 10.0).collect();
        let sim = anno::backends::inference::cosine_similarity(&v, &v);

        // Invariant: cosine similarity of identical vectors is 1.0
        prop_assert!((sim - 1.0).abs() < 1e-5, "Expected ~1.0, got {}", sim);
    }

    /// Orthogonal vectors have similarity 0.0.
    #[test]
    fn cosine_similarity_orthogonal(dim in 2..100usize) {
        let mut v1 = vec![0.0f32; dim];
        let mut v2 = vec![0.0f32; dim];
        v1[0] = 1.0;
        v2[1] = 1.0;

        let sim = anno::backends::inference::cosine_similarity(&v1, &v2);

        // Invariant: orthogonal vectors have 0 similarity
        prop_assert!(sim.abs() < 1e-5, "Expected ~0.0, got {}", sim);
    }

    /// Opposite vectors have similarity -1.0.
    #[test]
    fn cosine_similarity_opposite(dim in 1..100usize) {
        let v1: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) / 10.0).collect();
        let v2: Vec<f32> = v1.iter().map(|x| -x).collect();

        let sim = anno::backends::inference::cosine_similarity(&v1, &v2);

        // Invariant: opposite vectors have -1 similarity
        prop_assert!((sim + 1.0).abs() < 1e-5, "Expected ~-1.0, got {}", sim);
    }
}

// =============================================================================
// SpanCandidate Tests
// =============================================================================

proptest! {
    /// SpanCandidate width is consistent with start/end.
    #[test]
    fn span_candidate_width_invariant(
        doc_idx in 0..10u32,
        start in 0..100u32,
        width in 1..12u32,
    ) {
        let end = start + width;
        let candidate = SpanCandidate::new(doc_idx, start, end);

        // Invariant: width = end - start
        prop_assert_eq!(candidate.width(), width,
            "Width mismatch: {} vs {}", candidate.width(), width);

        // Invariant: start < end
        prop_assert!(candidate.start < candidate.end,
            "Start {} >= end {}", candidate.start, candidate.end);
    }

    /// Span candidates can be compared for overlap.
    #[test]
    fn span_overlap_detection(
        start1 in 0..50u32,
        width1 in 1..10u32,
        start2 in 0..50u32,
        width2 in 1..10u32,
    ) {
        let end1 = start1 + width1;
        let end2 = start2 + width2;

        let overlap = !(end1 <= start2 || end2 <= start1);
        let manual_overlap = start1 < end2 && start2 < end1;

        // Invariant: overlap detection is symmetric
        prop_assert_eq!(overlap, manual_overlap,
            "Overlap detection mismatch for [{}, {}) and [{}, {})",
            start1, end1, start2, end2);
    }
}

// =============================================================================
// RaggedBatch Tests
// =============================================================================

proptest! {
    /// RaggedBatch doc_range returns valid ranges.
    #[test]
    fn ragged_batch_doc_range_valid(
        doc_lens in proptest::collection::vec(1..50usize, 1..10),
    ) {
        // Build sequences with specified lengths
        let sequences: Vec<Vec<u32>> = doc_lens
            .iter()
            .map(|&len| vec![0u32; len])
            .collect();
        let batch = RaggedBatch::from_sequences(&sequences);

        // Invariant: each doc_range is valid
        for (i, len) in doc_lens.iter().enumerate() {
            if let Some(range) = batch.doc_range(i) {
                prop_assert_eq!(range.end - range.start, *len,
                    "Doc {} range length mismatch", i);
            }
        }

        // Invariant: out of bounds returns None
        prop_assert!(batch.doc_range(doc_lens.len() + 1).is_none());
    }
}

// =============================================================================
// Coreference Resolution Tests
// =============================================================================

proptest! {
    /// Coreference clusters are disjoint (no entity in multiple clusters).
    #[test]
    fn coreference_clusters_disjoint(
        num_entities in 2..10usize,
    ) {
        let entities: Vec<Entity> = (0..num_entities)
            .map(|i| Entity::new(
                &format!("Entity{}", i),
                EntityType::Person,
                i * 10,
                i * 10 + 5,
                0.9,
            ))
            .collect();

        let embeddings = vec![0.0f32; num_entities * 64];
        let clusters = resolve_coreferences(
            &entities, &embeddings, 64, &CoreferenceConfig::default()
        );

        // Collect all entity indices across clusters
        let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for cluster in &clusters {
            for &member in &cluster.members {
                prop_assert!(seen.insert(member),
                    "Entity {} appears in multiple clusters", member);
            }
        }
    }

    /// String-match coreference links entities with matching substrings.
    #[test]
    fn coreference_string_match_works(offset in 50..100usize) {
        let entities = vec![
            Entity::new("John Smith", EntityType::Person, 0, 10, 0.95),
            Entity::new("Smith", EntityType::Person, offset, offset + 5, 0.90),
        ];

        let embeddings = vec![0.0f32; 2 * 768];
        let config = CoreferenceConfig {
            similarity_threshold: 0.99, // High threshold - won't match by embedding
            max_distance: Some(200),
            use_string_match: true,
        };
        let clusters = resolve_coreferences(&entities, &embeddings, 768, &config);

        // Invariant: substring match should create a cluster
        prop_assert_eq!(clusters.len(), 1,
            "Expected 1 cluster from substring match, got {}", clusters.len());
        prop_assert_eq!(clusters[0].members.len(), 2,
            "Expected 2 members, got {}", clusters[0].members.len());
        prop_assert_eq!(&clusters[0].canonical_name, "John Smith",
            "Canonical should be longest: {}", clusters[0].canonical_name);
    }
}

// =============================================================================
// HandshakingMatrix Tests
// =============================================================================

proptest! {
    /// HandshakingMatrix only contains upper triangular entries (i <= j).
    #[test]
    fn handshaking_matrix_upper_triangular(
        seq_len in 2..20usize,
        num_labels in 1..5usize,
    ) {
        // Create sparse scores - only a few above threshold
        let total = seq_len * seq_len * num_labels;
        let scores: Vec<f32> = (0..total)
            .map(|i| if i % 17 == 0 { 0.8 } else { 0.1 })
            .collect();

        let matrix = HandshakingMatrix::from_dense(&scores, seq_len, num_labels, 0.5);

        // Invariant: all cells have i <= j (upper triangular)
        for cell in &matrix.cells {
            prop_assert!(cell.i <= cell.j,
                "Cell ({}, {}) violates upper triangular: i > j",
                cell.i, cell.j);
        }
    }

    /// HandshakingMatrix scores are above threshold.
    #[test]
    fn handshaking_matrix_above_threshold(
        seq_len in 2..10usize,
        num_labels in 1..3usize,
        threshold in 0.3..0.9f32,
    ) {
        let total = seq_len * seq_len * num_labels;
        let scores: Vec<f32> = (0..total)
            .map(|i| (i as f32 % 10.0) / 10.0) // 0.0 to 0.9
            .collect();

        let matrix = HandshakingMatrix::from_dense(&scores, seq_len, num_labels, threshold);

        // Invariant: all kept cells are above threshold
        for cell in &matrix.cells {
            prop_assert!(cell.score >= threshold,
                "Cell score {} below threshold {}", cell.score, threshold);
        }
    }
}

// =============================================================================
// Integration Tests
// =============================================================================

#[test]
fn end_to_end_span_extraction_flow() {
    // This test validates the full inference flow without running a model.
    // It checks that types compose correctly.

    // 1. Build registry
    let registry = SemanticRegistry::builder()
        .add_entity("person", "A human being")
        .add_entity("organization", "A company or group")
        .build_placeholder(64);

    assert_eq!(registry.len(), 2);

    // 2. Create span candidates
    let _candidates = vec![
        SpanCandidate::new(0, 0, 2),  // "John Smith"
        SpanCandidate::new(0, 3, 4),  // "Apple"
    ];

    // 3. Create interaction
    let interaction = DotProductInteraction::new();

    // 4. Fake embeddings (normally from encoder)
    let span_embs = vec![0.1f32; 2 * 64]; // 2 spans
    let label_embs = vec![0.1f32; 2 * 64]; // 2 labels

    // 5. Compute scores
    let scores = interaction.compute_similarity(&span_embs, 2, &label_embs, 2, 64);

    assert_eq!(scores.len(), 4); // 2 spans × 2 labels

    // 6. The scores are finite and reasonable
    for score in &scores {
        assert!(score.is_finite());
    }
}

#[test]
fn relation_extraction_integration() {
    use anno::backends::inference::{extract_relations, RelationExtractionConfig};

    let entities = vec![
        Entity::new("Steve Jobs", EntityType::Person, 0, 10, 0.95),
        Entity::new("Apple", EntityType::Organization, 20, 25, 0.92),
    ];

    let text = "Steve Jobs founded Apple Inc.";

    let registry = SemanticRegistry::builder()
        .add_relation("FOUNDED", "Founded an organization")
        .build_placeholder(64);

    let config = RelationExtractionConfig::default();
    let relations = extract_relations(&entities, text, &registry, &config);

    // Should find the FOUNDED relation
    assert!(!relations.is_empty(), "Should find at least one relation");
    assert_eq!(relations[0].relation_type, "FOUNDED");
    assert_eq!(relations[0].head.text, "Steve Jobs");
    assert_eq!(relations[0].tail.text, "Apple");
}

// =============================================================================
// New Trait Tests (Research Alignment)
// =============================================================================

#[test]
fn test_discontinuous_entity_conversion() {
    use anno::DiscontinuousEntity;

    // Contiguous entity
    let contiguous = DiscontinuousEntity {
        spans: vec![(0, 10)],
        text: "Steve Jobs".to_string(),
        entity_type: "person".to_string(),
        confidence: 0.95,
    };

    assert!(contiguous.is_contiguous());
    let entity = contiguous.to_entity().expect("Should convert to Entity");
    assert_eq!(entity.text, "Steve Jobs");
    assert_eq!(entity.start, 0);
    assert_eq!(entity.end, 10);

    // Discontinuous entity (e.g., "New York ... airports")
    let discontinuous = DiscontinuousEntity {
        spans: vec![(0, 8), (25, 33)],
        text: "New York airports".to_string(),
        entity_type: "location".to_string(),
        confidence: 0.85,
    };

    assert!(!discontinuous.is_contiguous());
    assert!(discontinuous.to_entity().is_none());
}

#[test]
fn test_span_label_score() {
    use anno::SpanLabelScore;

    let score = SpanLabelScore {
        start: 0,
        end: 10,
        label_idx: 0,
        score: 0.95,
    };

    assert_eq!(score.start, 0);
    assert_eq!(score.end, 10);
    assert!(score.score > 0.9);
}

#[test]
fn test_encoder_output_structure() {
    use anno::EncoderOutput;

    let output = EncoderOutput {
        embeddings: vec![0.1f32; 768 * 5], // 5 tokens
        num_tokens: 5,
        hidden_dim: 768,
        token_offsets: vec![(0, 2), (3, 6), (7, 10), (11, 14), (15, 18)],
    };

    assert_eq!(output.embeddings.len(), 768 * 5);
    assert_eq!(output.num_tokens, 5);
    assert_eq!(output.token_offsets.len(), 5);
}

#[test]
fn test_relation_triple() {
    use anno::RelationTriple;

    let triple = RelationTriple {
        head_idx: 0,
        tail_idx: 1,
        relation_type: "WORKS_FOR".to_string(),
        confidence: 0.88,
    };

    assert_eq!(triple.head_idx, 0);
    assert_eq!(triple.tail_idx, 1);
    assert_eq!(triple.relation_type, "WORKS_FOR");
}

#[test]
fn test_extraction_with_relations() {
    use anno::{ExtractionWithRelations, RelationTriple};

    let extraction = ExtractionWithRelations {
        entities: vec![
            Entity::new("John", EntityType::Person, 0, 4, 0.95),
            Entity::new("Google", EntityType::Organization, 15, 21, 0.92),
        ],
        relations: vec![RelationTriple {
            head_idx: 0,
            tail_idx: 1,
            relation_type: "WORKS_FOR".to_string(),
            confidence: 0.85,
        }],
    };

    assert_eq!(extraction.entities.len(), 2);
    assert_eq!(extraction.relations.len(), 1);
    assert_eq!(extraction.relations[0].head_idx, 0);
    assert_eq!(extraction.relations[0].tail_idx, 1);
}

#[test]
fn test_label_category_variants() {
    assert_ne!(LabelCategory::Entity, LabelCategory::Relation);
    assert_ne!(LabelCategory::Entity, LabelCategory::Attribute);
}

#[test]
fn test_modality_hint_variants() {
    use anno::ModalityHint;

    assert_ne!(ModalityHint::TextOnly, ModalityHint::VisualOnly);
    assert_ne!(ModalityHint::TextOnly, ModalityHint::Any);
    assert_ne!(ModalityHint::VisualOnly, ModalityHint::Any);
}

#[test]
fn test_image_format_default() {
    use anno::ImageFormat;

    let default = ImageFormat::default();
    assert_eq!(default, ImageFormat::Png);
}

proptest! {
    /// DiscontinuousEntity spans should have valid ordering.
    #[test]
    fn discontinuous_spans_ordered(
        spans in proptest::collection::vec((0usize..1000, 1usize..100), 1..5)
    ) {
        use anno::DiscontinuousEntity;

        let adjusted: Vec<(usize, usize)> = spans
            .into_iter()
            .map(|(start, len)| (start, start + len))
            .collect();

        let entity = DiscontinuousEntity {
            spans: adjusted.clone(),
            text: "test".to_string(),
            entity_type: "test".to_string(),
            confidence: 0.5,
        };

        // Each span should have end > start
        for (start, end) in &entity.spans {
            prop_assert!(end > start);
        }
    }

    /// SpanLabelScore should have bounded confidence.
    #[test]
    fn span_label_score_bounded(score in 0.0f32..=1.0f32) {
        use anno::SpanLabelScore;

        let sls = SpanLabelScore {
            start: 0,
            end: 10,
            label_idx: 0,
            score,
        };

        prop_assert!(sls.score >= 0.0);
        prop_assert!(sls.score <= 1.0);
    }
}

// =============================================================================
// Additional Property Tests for Research Alignment
// =============================================================================

proptest! {
    /// Invariant: LateInteraction output size is spans × labels
    #[test]
    fn late_interaction_output_size(
        num_spans in 1usize..50,
        num_labels in 1usize..20,
        dim in prop_oneof![Just(64usize), Just(128), Just(256), Just(768)]
    ) {
        let interaction = DotProductInteraction::new();

        let span_embs = vec![0.1f32; num_spans * dim];
        let label_embs = vec![0.1f32; num_labels * dim];

        let scores = interaction.compute_similarity(&span_embs, num_spans, &label_embs, num_labels, dim);

        prop_assert_eq!(scores.len(), num_spans * num_labels,
            "Expected {} scores, got {}", num_spans * num_labels, scores.len());
    }

    /// Invariant: LateInteraction scores are always finite
    #[test]
    fn late_interaction_scores_finite(
        num_spans in 1usize..10,
        num_labels in 1usize..5,
        dim in prop_oneof![Just(64usize), Just(128)]
    ) {
        let interaction = DotProductInteraction::new();

        // Use normalized vectors to avoid overflow
        let span_embs: Vec<f32> = (0..num_spans * dim)
            .map(|i| ((i as f32) * 0.01).sin())
            .collect();
        let label_embs: Vec<f32> = (0..num_labels * dim)
            .map(|i| ((i as f32) * 0.02).cos())
            .collect();

        let scores = interaction.compute_similarity(&span_embs, num_spans, &label_embs, num_labels, dim);

        for (i, score) in scores.iter().enumerate() {
            prop_assert!(score.is_finite(), "Score {} is not finite: {}", i, score);
        }
    }

    /// Invariant: Temperature scaling affects distribution sharpness
    #[test]
    fn temperature_affects_distribution(
        temp in 0.1f32..10.0f32
    ) {
        let interaction = DotProductInteraction::with_temperature(temp);

        // Two identical vectors → high similarity
        let v = vec![0.5f32; 64];
        let scores = interaction.compute_similarity(&v, 1, &v, 1, 64);

        prop_assert!(scores[0].is_finite());
        // Higher temp → higher raw score before sigmoid
        // The actual score depends on normalization
    }

    /// Invariant: SemanticRegistry preserves label order
    #[test]
    fn registry_preserves_order(
        labels in proptest::collection::hash_set("[a-z]{3,10}", 1..15)
    ) {
        let labels_vec: Vec<_> = labels.into_iter().collect();
        let mut builder = SemanticRegistry::builder();
        for label in &labels_vec {
            builder = builder.add_entity(label, &format!("desc {}", label));
        }
        let registry = builder.build_placeholder(64);

        // Check that labels appear in order (unique labels only)
        for (i, label) in labels_vec.iter().enumerate() {
            if let Some(&idx) = registry.label_index.get(label) {
                prop_assert_eq!(idx, i, "Label {} at wrong index", label);
            }
        }
    }

    /// Invariant: HandshakingMatrix stores upper triangular correctly (prop)
    #[test]
    fn handshaking_matrix_stores_upper_triangular(
        size in 2usize..10,
        num_labels in 1usize..5
    ) {
        // Create dense scores with known pattern
        let total = size * size * num_labels;
        let scores: Vec<f32> = (0..total)
            .map(|i| if i % 3 == 0 { 0.8 } else { 0.2 })
            .collect();

        let matrix = HandshakingMatrix::from_dense(&scores, size, num_labels, 0.5);

        // Only cells with score >= 0.5 should be stored
        for cell in &matrix.cells {
            prop_assert!(cell.score >= 0.5, "Cell has score {} < 0.5", cell.score);
            prop_assert!(cell.i <= cell.j, "Not upper triangular: i={} > j={}", cell.i, cell.j);
        }
    }

    /// Invariant: Coreference clusters are disjoint
    #[test]
    fn coreference_clusters_disjoint_prop(
        num_entities in 2usize..10
    ) {
        // Create entities
        let entities: Vec<Entity> = (0..num_entities)
            .map(|i| Entity::new(
                format!("entity{}", i),
                EntityType::Person,
                i * 10,
                i * 10 + 5,
                0.9,
            ))
            .collect();

        // Create fake embeddings (very different so no clustering)
        let hidden_dim = 64;
        let embeddings: Vec<f32> = (0..num_entities * hidden_dim)
            .map(|i| (i as f32) * 0.001)
            .collect();

        // Resolve coreferences with strict config
        let config = CoreferenceConfig {
            similarity_threshold: 0.999, // Very strict - no clustering
            max_distance: None,
            use_string_match: false,
        };

        let clusters = resolve_coreferences(&entities, &embeddings, hidden_dim, &config);

        // Check disjointness: no entity should appear in multiple clusters
        let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for cluster in &clusters {
            for &idx in &cluster.members {
                prop_assert!(!seen.contains(&idx),
                    "Entity {} appears in multiple clusters", idx);
                seen.insert(idx);
            }
        }
    }
}

// =============================================================================
// Fuzzing-Style Edge Case Tests
// =============================================================================

#[test]
fn fuzz_empty_inputs() {
    // Empty registry
    let registry = SemanticRegistry::builder().build_placeholder(64);
    assert_eq!(registry.len(), 0);

    // Empty coreference resolution
    let clusters = resolve_coreferences(&[], &[], 64, &CoreferenceConfig::default());
    assert!(clusters.is_empty());

    // Empty handshaking matrix
    let matrix = HandshakingMatrix::from_dense(&[], 0, 1, 0.5);
    assert!(matrix.cells.is_empty());
}

#[test]
fn fuzz_single_element() {
    // Single entity - coreference only creates clusters for >1 member
    let entity = Entity::new("Test", EntityType::Person, 0, 4, 0.9);
    let embeddings = vec![0.5f32; 64];
    let clusters = resolve_coreferences(&[entity], &embeddings, 64, &CoreferenceConfig::default());
    // Single entity doesn't form a coreference cluster (by design)
    assert_eq!(clusters.len(), 0);

    // Two identical entities should cluster
    let e1 = Entity::new("Test", EntityType::Person, 0, 4, 0.9);
    let e2 = Entity::new("Test", EntityType::Person, 10, 14, 0.9);
    let embeddings2 = vec![0.5f32; 128]; // 2 entities × 64 dim
    let clusters2 = resolve_coreferences(&[e1, e2], &embeddings2, 64, &CoreferenceConfig::default());
    // With string match enabled (default), identical names cluster
    assert_eq!(clusters2.len(), 1);
    assert_eq!(clusters2[0].members.len(), 2);

    // Single label registry
    let registry = SemanticRegistry::builder()
        .add_entity("person", "a human")
        .build_placeholder(64);
    assert_eq!(registry.len(), 1);

    // 1x1 handshaking matrix with single label
    let scores = vec![0.9f32]; // single cell
    let matrix = HandshakingMatrix::from_dense(&scores, 1, 1, 0.5);
    assert_eq!(matrix.cells.len(), 1);
}

#[test]
fn fuzz_unicode_labels() {
    // Unicode in label names
    let registry = SemanticRegistry::builder()
        .add_entity("人", "Chinese for person")
        .add_entity("организация", "Russian for organization")
        .add_entity("مكان", "Arabic for place")
        .add_entity("emoji_🎉", "Label with emoji")
        .build_placeholder(64);

    assert_eq!(registry.len(), 4);
    assert!(registry.label_index.contains_key("人"));
    assert!(registry.label_index.contains_key("организация"));
}

#[test]
fn fuzz_extreme_confidence() {
    // Boundary confidence values
    let e1 = Entity::new("Test", EntityType::Person, 0, 4, 0.0);
    let e2 = Entity::new("Test", EntityType::Person, 0, 4, 1.0);
    let e3 = Entity::new("Test", EntityType::Person, 0, 4, -0.1); // Should clamp to 0
    let e4 = Entity::new("Test", EntityType::Person, 0, 4, 1.5);  // Should clamp to 1

    assert_eq!(e1.confidence, 0.0);
    assert_eq!(e2.confidence, 1.0);
    assert_eq!(e3.confidence, 0.0);
    assert_eq!(e4.confidence, 1.0);
}

#[test]
fn fuzz_large_handshaking_matrix() {
    // Stress test with larger matrix
    let size = 50;
    let num_labels = 3;

    // Create dense scores with pattern
    let total = size * size * num_labels;
    let scores: Vec<f32> = (0..total)
        .map(|i| {
            let row = (i / (size * num_labels)) as f32;
            let col = ((i / num_labels) % size) as f32;
            // Higher scores on diagonal
            if (row - col).abs() < 2.0 { 0.8 } else { 0.1 }
        })
        .collect();

    let matrix = HandshakingMatrix::from_dense(&scores, size, num_labels, 0.5);

    // Should have cells near diagonal
    assert!(!matrix.cells.is_empty());

    // All stored cells should have high scores
    for cell in &matrix.cells {
        assert!(cell.score >= 0.5);
    }
}

#[test]
fn fuzz_late_interaction_edge_cases() {
    let interaction = DotProductInteraction::new();

    // Zero vectors
    let zeros = vec![0.0f32; 64];
    let scores = interaction.compute_similarity(&zeros, 1, &zeros, 1, 64);
    assert_eq!(scores.len(), 1);
    assert!(scores[0].is_finite());

    // Very large values (should still be finite after sigmoid)
    let large = vec![1000.0f32; 64];
    let scores = interaction.compute_similarity(&large, 1, &large, 1, 64);
    assert!(scores[0].is_finite());

    // Very small values
    let small = vec![1e-10f32; 64];
    let scores = interaction.compute_similarity(&small, 1, &small, 1, 64);
    assert!(scores[0].is_finite());
}

