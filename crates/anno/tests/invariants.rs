//! Invariant tests for entity resolution and similarity functions.
//!
//! These tests verify fundamental properties that should always hold:
//! - Symmetry: sim(a, b) == sim(b, a)
//! - Identity: sim(a, a) == 1.0
//! - Bounded: 0.0 <= sim(a, b) <= 1.0
//! - Triangle inequality (for distance functions)

use ::anno_core::coalesce as anno_coalesce;
use anno_core::core as anno_core;

use anno_coalesce::resolver::{embedding_similarity, string_similarity, Resolver};
use anno_coalesce::similarity::{levenshtein_distance, Similarity};
use anno_core::TypeLabel;
use anno_core::{Corpus, GroundedDocument, Track};

// =============================================================================
// String Similarity Invariants
// =============================================================================

#[test]
fn test_string_similarity_symmetry() {
    let pairs = [
        ("hello", "world"),
        ("Marie Curie", "Curie"),
        ("test", ""),
        ("ABC", "abc"),
        ("北京", "東京"),
    ];

    for (a, b) in pairs {
        let ab = string_similarity(a, b);
        let ba = string_similarity(b, a);
        assert!(
            (ab - ba).abs() < 0.001,
            "Symmetry failed for ({}, {}): {} vs {}",
            a,
            b,
            ab,
            ba
        );
    }
}

#[test]
fn test_string_similarity_identity() {
    let strings = ["hello", "Marie Curie", "", "北京", "🎉"];

    for s in strings {
        let sim = string_similarity(s, s);
        if s.is_empty() {
            assert_eq!(sim, 1.0, "Empty string self-similarity should be 1.0");
        } else {
            assert!(
                (sim - 1.0).abs() < 0.001,
                "Identity failed for '{}': {}",
                s,
                sim
            );
        }
    }
}

// =============================================================================
// Embedding Similarity Invariants
// =============================================================================

#[test]
fn test_embedding_similarity_symmetry() {
    let pairs = [
        (vec![1.0, 0.0], vec![0.0, 1.0]),
        (vec![1.0, 2.0, 3.0], vec![3.0, 2.0, 1.0]),
        (vec![0.5], vec![0.5]),
    ];

    for (a, b) in pairs {
        let ab = embedding_similarity(&a, &b);
        let ba = embedding_similarity(&b, &a);
        assert!(
            (ab - ba).abs() < 0.001,
            "Embedding symmetry failed: {} vs {}",
            ab,
            ba
        );
    }
}

#[test]
fn test_embedding_similarity_identity() {
    let embeddings = [
        vec![1.0, 0.0, 0.0],
        vec![0.5, 0.5],
        vec![1.0],
        vec![-1.0, -1.0],
    ];

    for emb in embeddings {
        let sim = embedding_similarity(&emb, &emb);
        assert!(
            (sim - 1.0).abs() < 0.001,
            "Embedding identity failed: {}",
            sim
        );
    }
}

// =============================================================================
// Levenshtein Distance Invariants
// =============================================================================

#[test]
fn test_string_distance_triangle_inequality() {
    // Triangle inequality: d(a, c) <= d(a, b) + d(b, c)
    let triples = [
        ("kitten", "sitting", "smitten"),
        ("hello", "hallo", "holla"),
        ("abc", "abd", "aed"),
    ];

    for (a, b, c) in triples {
        let ab = levenshtein_distance(a, b);
        let bc = levenshtein_distance(b, c);
        let ac = levenshtein_distance(a, c);

        assert!(
            ac <= ab + bc,
            "Triangle inequality failed: d({},{})={} > d({},{})={} + d({},{})={}",
            a,
            c,
            ac,
            a,
            b,
            ab,
            b,
            c,
            bc
        );
    }
}

// =============================================================================
// Multilingual Similarity Invariants
// =============================================================================

#[test]
fn test_multilingual_similarity_symmetry() {
    let sim = Similarity::new();

    let pairs = [
        ("hello world", "world hello"),
        ("北京市", "上海市"),
        ("Москва", "Санкт"),
        ("test 123", "123 test"),
    ];

    for (a, b) in pairs {
        let ab = sim.compute(a, b);
        let ba = sim.compute(b, a);
        assert!(
            (ab - ba).abs() < 0.001,
            "Multilingual symmetry failed for ({}, {}): {} vs {}",
            a,
            b,
            ab,
            ba
        );
    }
}

// =============================================================================
// Resolver Invariants
// =============================================================================

/// Helper to create a track with entity type
fn create_track(id: u64, surface: &str, entity_type: &str) -> Track {
    let mut track = Track::new(id, surface);
    track.entity_type = Some(TypeLabel::from(entity_type));
    track.cluster_confidence = 0.9;
    track
}

#[test]
fn test_resolution_idempotency() {
    // Running resolution twice should produce consistent results
    let resolver = Resolver::new().with_threshold(0.7);

    let mut corpus1 = Corpus::new();
    let mut doc = GroundedDocument::new("test_doc", "Obama Barack Obama");
    doc.add_track(create_track(1, "Obama", "PERSON"));
    doc.add_track(create_track(2, "Barack Obama", "PERSON"));
    corpus1.add_document(doc);

    let mut corpus2 = corpus1.clone();

    let ids1 = resolver.resolve_inter_doc_coref(&mut corpus1, None, None);
    let ids2 = resolver.resolve_inter_doc_coref(&mut corpus2, None, None);

    assert_eq!(ids1.len(), ids2.len(), "Resolution should be idempotent");
}

#[test]
fn test_resolution_determinism() {
    // Same input should always produce same output
    let resolver = Resolver::new().with_threshold(0.8);

    for _ in 0..5 {
        let mut corpus = Corpus::new();

        let mut doc1 = GroundedDocument::new("doc1", "Marie Curie");
        doc1.add_track(create_track(1, "Marie Curie", "PERSON"));
        corpus.add_document(doc1);

        let mut doc2 = GroundedDocument::new("doc2", "Marie Curie");
        doc2.add_track(create_track(1, "Marie Curie", "PERSON"));
        corpus.add_document(doc2);

        let ids = resolver.resolve_inter_doc_coref(&mut corpus, None, None);
        // With same names and high threshold, should merge
        assert!(!ids.is_empty(), "Should create at least one identity");
    }
}

#[test]
fn test_clustering_transitivity() {
    // If A clusters with B, and B clusters with C, then A should be in same cluster as C
    let resolver = Resolver::new().with_threshold(0.5); // Low threshold to encourage clustering

    let mut corpus = Corpus::new();

    // Three very similar mentions in different docs
    let mut doc1 = GroundedDocument::new("doc1", "John Smith");
    doc1.add_track(create_track(1, "John Smith", "PERSON"));
    corpus.add_document(doc1);

    let mut doc2 = GroundedDocument::new("doc2", "John Smith");
    doc2.add_track(create_track(1, "John Smith", "PERSON"));
    corpus.add_document(doc2);

    let mut doc3 = GroundedDocument::new("doc3", "John Smith");
    doc3.add_track(create_track(1, "John Smith", "PERSON"));
    corpus.add_document(doc3);

    let ids = resolver.resolve_inter_doc_coref(&mut corpus, None, None);

    // All three identical mentions should be in the same cluster
    assert_eq!(
        ids.len(),
        1,
        "Three identical mentions should form one cluster"
    );
}

// =============================================================================
// Property Tests
// =============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        /// String similarity is symmetric
        #[test]
        fn string_sim_symmetric(a in "[a-z ]{0,20}", b in "[a-z ]{0,20}") {
            let ab = string_similarity(&a, &b);
            let ba = string_similarity(&b, &a);
            prop_assert!((ab - ba).abs() < 0.0001,
                "Symmetry: {} vs {}", ab, ba);
        }

        /// String similarity is bounded [0, 1]
        #[test]
        fn string_sim_bounded(a in ".*", b in ".*") {
            let sim = string_similarity(&a, &b);
            prop_assert!((0.0..=1.0).contains(&sim),
                "Bounds: {}", sim);
        }

        /// String similarity identity
        #[test]
        fn string_sim_identity(s in "[a-z]{1,20}") {
            let sim = string_similarity(&s, &s);
            prop_assert!((sim - 1.0).abs() < 0.0001,
                "Identity: {}", sim);
        }

        /// Embedding similarity is symmetric
        #[test]
        fn embedding_sim_symmetric(dim in 1usize..10, seed in any::<u64>()) {
            let mut rng = seed;
            let emb1: Vec<f32> = (0..dim).map(|_| {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                (rng % 100) as f32 / 100.0
            }).collect();
            let emb2: Vec<f32> = (0..dim).map(|_| {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                (rng % 100) as f32 / 100.0
            }).collect();

            let ab = embedding_similarity(&emb1, &emb2);
            let ba = embedding_similarity(&emb2, &emb1);
            prop_assert!((ab - ba).abs() < 0.0001,
                "Embedding symmetry: {} vs {}", ab, ba);
        }

        /// Embedding similarity is bounded [0, 1]
        #[test]
        fn embedding_sim_bounded(dim in 1usize..20, seed in any::<u64>()) {
            let mut rng = seed;
            let emb1: Vec<f32> = (0..dim).map(|_| {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                (rng % 2000) as f32 / 1000.0 - 1.0
            }).collect();
            let emb2: Vec<f32> = (0..dim).map(|_| {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                (rng % 2000) as f32 / 1000.0 - 1.0
            }).collect();

            let sim = embedding_similarity(&emb1, &emb2);
            prop_assert!((0.0..=1.0).contains(&sim),
                "Embedding bounds: {}", sim);
        }

        /// Resolution produces valid identity IDs
        #[test]
        fn resolution_produces_valid_identities(
            name1 in "[A-Za-z ]{3,20}",
            name2 in "[A-Za-z ]{3,20}"
        ) {
            let resolver = Resolver::new().with_threshold(0.7);
            let mut corpus = Corpus::new();

            let mut doc = GroundedDocument::new("test_doc", format!("{} {}", name1, name2));
            doc.add_track(create_track(1, &name1, "PERSON"));
            doc.add_track(create_track(2, &name2, "PERSON"));
            corpus.add_document(doc);

            let ids = resolver.resolve_inter_doc_coref(&mut corpus, None, None);

            // Should produce 1 or 2 identities depending on similarity
            prop_assert!(!ids.is_empty() && ids.len() <= 2,
                "Should produce 1-2 identities, got {}", ids.len());
        }

        /// Resolution is deterministic
        #[test]
        fn resolution_deterministic(name in "[A-Za-z]{5,15}") {
            let resolver = Resolver::new().with_threshold(0.8);

            let results: Vec<usize> = (0..3).map(|_| {
                let mut corpus = Corpus::new();
                let mut doc = GroundedDocument::new("doc", &name);
                doc.add_track(create_track(1, &name, "PERSON"));
                corpus.add_document(doc);
                resolver.resolve_inter_doc_coref(&mut corpus, None, None).len()
            }).collect();

            // All runs should produce same number of identities
            prop_assert!(results.iter().all(|&r| r == results[0]),
                "Resolution not deterministic: {:?}", results);
        }
    }
}
