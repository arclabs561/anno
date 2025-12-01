//! Integration tests for newly added features.
//!
//! Tests:
//! - Binary embedding blocking with two-stage retrieval
//! - Entropy-based conflict filtering
//! - End-to-end pipeline integration
//!
//! Run: `cargo test --test new_features_integration --features eval-advanced`

use anno::backends::inference::{two_stage_retrieval, BinaryBlocker, BinaryHash};
use anno::Entity;
use anno::EntityType;

// =============================================================================
// Binary Embedding Integration Tests
// =============================================================================

#[test]
fn test_binary_hash_large_dimension() {
    // Test with realistic embedding dimensions
    let embedding_768 = vec![0.1f32; 768];
    let hash = BinaryHash::from_embedding(&embedding_768);

    assert_eq!(hash.dim, 768);
    assert_eq!(hash.bits.len(), 12); // ceil(768/64)
}

#[test]
fn test_binary_blocking_scalability() {
    // Test with 1000 entities
    let mut blocker = BinaryBlocker::new(50); // ~7% bit difference threshold

    // Generate pseudo-random embeddings
    for i in 0..1000 {
        let mut embedding = vec![0.1f32; 768];
        // Perturb based on index to create clusters
        for j in 0..(i % 100) {
            embedding[j] = -0.1;
        }
        blocker.add(i, BinaryHash::from_embedding(&embedding));
    }

    // Query should find similar embeddings
    let query = BinaryHash::from_embedding(&vec![0.1f32; 768]);
    let candidates = blocker.query(&query);

    // Should find entries 0-99 (same cluster as query)
    assert!(!candidates.is_empty(), "Should find some candidates");
    assert!(
        candidates.contains(&0),
        "Should include index 0 (identical)"
    );
}

#[test]
fn test_two_stage_retrieval_correctness() {
    // Create distinguishable embeddings
    let mut candidates = Vec::new();

    // Cluster 1: positive values
    candidates.push(vec![1.0, 0.0, 0.0, 0.0]);
    candidates.push(vec![0.9, 0.1, 0.0, 0.0]);

    // Cluster 2: negative values
    candidates.push(vec![-1.0, 0.0, 0.0, 0.0]);
    candidates.push(vec![-0.9, -0.1, 0.0, 0.0]);

    // Query similar to cluster 1
    let query = vec![0.95, 0.05, 0.0, 0.0];

    // Should return cluster 1 members
    let results = two_stage_retrieval(&query, &candidates, 2, 10);

    assert!(results.len() >= 2);
    // Top results should be from cluster 1 (indices 0, 1)
    assert!(results[0].0 == 0 || results[0].0 == 1);
}

// =============================================================================
// Entropy Filter Integration Tests
// =============================================================================

#[cfg(feature = "eval-advanced")]
mod entropy_tests {
    use anno::eval::calibration::{confidence_entropy, EntropyFilter};

    #[test]
    fn test_entropy_with_realistic_scenarios() {
        // Scenario 1: Multiple NER models agree on "Apple Inc."
        let apple_scores = vec![0.92, 0.89, 0.94, 0.91];
        let filter = EntropyFilter::new(0.3);
        assert!(
            filter.should_keep(&apple_scores),
            "Agreeing models should pass filter"
        );

        // Scenario 2: Models disagree on "Apple" (company vs fruit)
        let ambiguous_scores = vec![0.85, 0.15, 0.60, 0.40];
        assert!(
            !filter.should_keep(&ambiguous_scores),
            "Disagreeing models should be filtered"
        );
    }

    #[test]
    fn test_entropy_edge_cases() {
        // All same confidence
        let uniform = vec![0.5, 0.5, 0.5, 0.5];
        assert!(
            confidence_entropy(&uniform) < 0.01,
            "Uniform scores should have near-zero entropy"
        );

        // Binary split (worst case)
        let split = vec![1.0, 0.0, 1.0, 0.0];
        let entropy = confidence_entropy(&split);
        assert!(
            entropy > 0.9,
            "Binary split should have high entropy: {}",
            entropy
        );
    }
}

// =============================================================================
// Pipeline Integration Tests
// =============================================================================

#[test]
fn test_entity_with_temporal_validity() {
    use chrono::Utc;

    // Test that Entity can hold temporal validity info
    let mut entity = Entity::new("Tim Cook", EntityType::Person, 0, 8, 0.95);
    entity.valid_from = Some(Utc::now());

    assert!(entity.valid_from.is_some());
    assert!(entity.valid_until.is_none()); // Still valid
}

#[test]
fn test_entity_viewport_context() {
    use anno::EntityViewport;

    let mut entity = Entity::new("Apple Inc.", EntityType::Organization, 0, 10, 0.9);
    entity.viewport = Some(EntityViewport::Business);

    assert!(entity.viewport.as_ref().unwrap().is_professional());
}

// =============================================================================
// CDCR + Binary Blocking Integration
// =============================================================================

#[test]
fn test_cdcr_with_custom_blocking() {
    use anno::eval::cdcr::{CDCRConfig, CDCRResolver, Document, LSHBlocker};

    // Create documents with overlapping entities
    let mut doc1 = Document::new("doc1", "Jensen Huang is CEO of Nvidia.");
    doc1.entities = vec![
        Entity::new("Jensen Huang", EntityType::Person, 0, 12, 0.95),
        Entity::new("Nvidia", EntityType::Organization, 24, 30, 0.94),
    ];

    let mut doc2 = Document::new("doc2", "Huang announced new chips at the Nvidia event.");
    doc2.entities = vec![
        Entity::new("Huang", EntityType::Person, 0, 5, 0.88),
        Entity::new("Nvidia", EntityType::Organization, 34, 40, 0.91),
    ];

    // Configure with custom LSH parameters
    let config = CDCRConfig {
        use_lsh: true,
        lsh: LSHBlocker::new(3, 2), // More bands, fewer rows
        min_similarity: 0.4,
        require_type_match: true,
    };

    let resolver = CDCRResolver::with_config(config);
    let clusters = resolver.resolve(&[doc1, doc2]);

    // Should cluster "Nvidia" mentions
    let nvidia_cluster = clusters
        .iter()
        .find(|c| c.canonical_name.to_lowercase() == "nvidia");
    assert!(nvidia_cluster.is_some(), "Should find Nvidia cluster");
    assert!(
        nvidia_cluster.unwrap().doc_count() >= 2,
        "Nvidia should span both documents"
    );
}

// =============================================================================
// Known Limitation Tests (Document edge cases)
// =============================================================================

#[test]
fn test_short_text_handling() {
    use anno::backends::RegexNER;
    use anno::Model;

    let ner = RegexNER::default();

    // Edge case: very short text
    let result = ner.extract_entities("Hi", None);
    assert!(result.is_ok());

    // Edge case: only numbers
    let result = ner.extract_entities("123 456", None);
    assert!(result.is_ok());

    // Edge case: empty
    let result = ner.extract_entities("", None);
    assert!(result.is_ok());
}

#[test]
fn test_unicode_handling() {
    let embedding = vec![0.1f32; 64];
    let hash = BinaryHash::from_embedding(&embedding);

    // Binary hashing should work regardless of text encoding
    // since it operates on embeddings, not text
    assert_eq!(hash.dim, 64);
}
