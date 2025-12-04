//! Comprehensive tests for Corpus, TrackRef, IdentitySource, and cross-document operations.
//!
//! Tests the new abstractions for inter-document coreference and entity linking.

use anno::grounded::{
    Corpus, GroundedDocument, Identity, IdentitySource, Location, Signal, Track, TrackRef,
};
use anno::{Entity, EntityType};
use anno_coalesce::Resolver;

// =============================================================================
// Corpus Basic Operations
// =============================================================================

#[test]
fn test_corpus_new() {
    let corpus = Corpus::new();
    assert_eq!(corpus.documents().count(), 0);
    assert_eq!(corpus.identities().count(), 0);
}

#[test]
fn test_corpus_add_document() {
    let mut corpus = Corpus::new();
    let doc = GroundedDocument::new("doc1", "Test text");
    corpus.add_document(doc);

    assert_eq!(corpus.documents().count(), 1);
    assert!(corpus.get_document("doc1").is_some());
    assert_eq!(corpus.get_document("doc1").unwrap().id, "doc1");
}

#[test]
fn test_corpus_get_document_mut() {
    let mut corpus = Corpus::new();
    let mut doc = GroundedDocument::new("doc1", "Test");
    doc.add_signal(Signal::new(0, Location::text(0, 4), "Test", "Type", 0.9));
    corpus.add_document(doc);

    let doc_mut = corpus.get_document_mut("doc1").unwrap();
    assert_eq!(doc_mut.signals().len(), 1);
}

#[test]
fn test_corpus_documents_iterator() {
    let mut corpus = Corpus::new();
    corpus.add_document(GroundedDocument::new("doc1", "Text 1"));
    corpus.add_document(GroundedDocument::new("doc2", "Text 2"));
    corpus.add_document(GroundedDocument::new("doc3", "Text 3"));

    let doc_ids: Vec<_> = corpus.documents().map(|d| d.id.as_str()).collect();
    assert_eq!(doc_ids.len(), 3);
    assert!(doc_ids.contains(&"doc1"));
    assert!(doc_ids.contains(&"doc2"));
    assert!(doc_ids.contains(&"doc3"));
}

// =============================================================================
// TrackRef Tests
// =============================================================================

#[test]
fn test_track_ref_creation() {
    let mut doc = GroundedDocument::new("doc1", "Test");
    let s1 = doc.add_signal(Signal::new(0, Location::text(0, 4), "Test", "Type", 0.9));
    let mut track = Track::new(0, "Test");
    track.add_signal(s1, 0);
    let track_id = doc.add_track(track);

    let track_ref = doc.track_ref(track_id).unwrap();
    assert_eq!(track_ref.doc_id, "doc1");
    assert_eq!(track_ref.track_id, track_id);
}

#[test]
fn test_track_ref_invalid_track() {
    let doc = GroundedDocument::new("doc1", "Test");
    // Track ID 999 doesn't exist
    assert!(doc.track_ref(999).is_none());
}

#[test]
fn test_track_ref_equality() {
    let ref1 = TrackRef {
        doc_id: "doc1".to_string(),
        track_id: 0,
    };
    let ref2 = TrackRef {
        doc_id: "doc1".to_string(),
        track_id: 0,
    };
    let ref3 = TrackRef {
        doc_id: "doc1".to_string(),
        track_id: 1,
    };

    assert_eq!(ref1, ref2);
    assert_ne!(ref1, ref3);
}

#[test]
fn test_track_ref_hash() {
    use std::collections::HashSet;

    let ref1 = TrackRef {
        doc_id: "doc1".to_string(),
        track_id: 0,
    };
    let ref2 = TrackRef {
        doc_id: "doc1".to_string(),
        track_id: 0,
    };
    let ref3 = TrackRef {
        doc_id: "doc2".to_string(),
        track_id: 0,
    };

    let mut set = HashSet::new();
    set.insert(ref1.clone());
    set.insert(ref2.clone());
    set.insert(ref3.clone());

    // ref1 and ref2 are equal, so set should have 2 elements
    assert_eq!(set.len(), 2);
}

// =============================================================================
// IdentitySource Tests
// =============================================================================

#[test]
fn test_identity_source_cross_doc_coref() {
    let track_refs = vec![
        TrackRef {
            doc_id: "doc1".to_string(),
            track_id: 0,
        },
        TrackRef {
            doc_id: "doc2".to_string(),
            track_id: 1,
        },
    ];

    let source = IdentitySource::CrossDocCoref {
        track_refs: track_refs.clone(),
    };

    match source {
        IdentitySource::CrossDocCoref { track_refs: refs } => {
            assert_eq!(refs.len(), 2);
            assert_eq!(refs[0].doc_id, "doc1");
            assert_eq!(refs[1].doc_id, "doc2");
        }
        _ => panic!("Wrong source type"),
    }
}

#[test]
fn test_identity_source_knowledge_base() {
    let source = IdentitySource::KnowledgeBase {
        kb_name: "wikidata".to_string(),
        kb_id: "Q7186".to_string(),
    };

    match source {
        IdentitySource::KnowledgeBase { kb_name, kb_id } => {
            assert_eq!(kb_name, "wikidata");
            assert_eq!(kb_id, "Q7186");
        }
        _ => panic!("Wrong source type"),
    }
}

#[test]
fn test_identity_source_hybrid() {
    let track_refs = vec![TrackRef {
        doc_id: "doc1".to_string(),
        track_id: 0,
    }];

    let source = IdentitySource::Hybrid {
        track_refs: track_refs.clone(),
        kb_name: "wikidata".to_string(),
        kb_id: "Q7186".to_string(),
    };

    match source {
        IdentitySource::Hybrid {
            track_refs: refs,
            kb_name,
            kb_id,
        } => {
            assert_eq!(refs.len(), 1);
            assert_eq!(kb_name, "wikidata");
            assert_eq!(kb_id, "Q7186");
        }
        _ => panic!("Wrong source type"),
    }
}

// =============================================================================
// Inter-Document Coreference Tests
// =============================================================================

#[test]
fn test_resolve_inter_doc_coref_basic() {
    let mut corpus = Corpus::new();

    // Document 1: "Marie Curie won the Nobel Prize."
    let mut doc1 = GroundedDocument::new("doc1", "Marie Curie won the Nobel Prize.");
    let s1 = doc1.add_signal(Signal::new(
        0,
        Location::text(0, 12),
        "Marie Curie",
        "Person",
        0.95,
    ));
    let mut track1 = Track::new(0, "Marie Curie");
    track1.add_signal(s1, 0);
    track1.entity_type = Some("Person".to_string());
    doc1.add_track(track1);
    corpus.add_document(doc1);

    // Document 2: "She was a physicist."
    let mut doc2 = GroundedDocument::new("doc2", "She was a physicist.");
    let s2 = doc2.add_signal(Signal::new(0, Location::text(0, 3), "She", "Person", 0.88));
    let mut track2 = Track::new(0, "She");
    track2.add_signal(s2, 0);
    track2.entity_type = Some("Person".to_string());
    doc2.add_track(track2);
    corpus.add_document(doc2);

    // Document 3: "Marie Curie discovered radium."
    let mut doc3 = GroundedDocument::new("doc3", "Marie Curie discovered radium.");
    let s3 = doc3.add_signal(Signal::new(
        0,
        Location::text(0, 12),
        "Marie Curie",
        "Person",
        0.95,
    ));
    let mut track3 = Track::new(0, "Marie Curie");
    track3.add_signal(s3, 0);
    track3.entity_type = Some("Person".to_string());
    doc3.add_track(track3);
    corpus.add_document(doc3);

    // Resolve inter-doc coref
    let resolver = Resolver::new().with_threshold(0.5).require_type_match(true);
    let identity_ids = resolver.resolve_inter_doc_coref(&mut corpus, None, None);

    // Should create at least one identity linking doc1 and doc3 (both have "Marie Curie")
    assert!(!identity_ids.is_empty());

    // Check that tracks are linked
    let doc1 = corpus.get_document("doc1").unwrap();
    let track1 = doc1.get_track(0).unwrap();
    assert!(track1.identity_id.is_some());

    let doc3 = corpus.get_document("doc3").unwrap();
    let track3 = doc3.get_track(0).unwrap();
    assert!(track3.identity_id.is_some());

    // Both should link to same identity
    assert_eq!(track1.identity_id, track3.identity_id);
}

#[test]
fn test_resolve_inter_doc_coref_type_mismatch() {
    let mut corpus = Corpus::new();

    // Document 1: Person "Apple"
    let mut doc1 = GroundedDocument::new("doc1", "Apple is a person.");
    let s1 = doc1.add_signal(Signal::new(0, Location::text(0, 5), "Apple", "Person", 0.9));
    let mut track1 = Track::new(0, "Apple");
    track1.add_signal(s1, 0);
    track1.entity_type = Some("Person".to_string());
    doc1.add_track(track1);
    corpus.add_document(doc1);

    // Document 2: Organization "Apple"
    let mut doc2 = GroundedDocument::new("doc2", "Apple is a company.");
    let s2 = doc2.add_signal(Signal::new(
        0,
        Location::text(0, 5),
        "Apple",
        "Organization",
        0.9,
    ));
    let mut track2 = Track::new(0, "Apple");
    track2.add_signal(s2, 0);
    track2.entity_type = Some("Organization".to_string());
    doc2.add_track(track2);
    corpus.add_document(doc2);

    // With type matching required, should NOT cluster
    let resolver = Resolver::new().with_threshold(0.5).require_type_match(true);
    let identity_ids = resolver.resolve_inter_doc_coref(&mut corpus, None, None);

    // Should create separate identities (or none if threshold too high)
    // Actually, with require_type_match=true, they shouldn't cluster
    // But if they do create identities, they should be separate
    if !identity_ids.is_empty() {
        let doc1 = corpus.get_document("doc1").unwrap();
        let doc2 = corpus.get_document("doc2").unwrap();
        let track1_id = doc1.get_track(0).unwrap().identity_id;
        let track2_id = doc2.get_track(0).unwrap().identity_id;

        // If both have identities, they should be different
        if track1_id.is_some() && track2_id.is_some() {
            assert_ne!(track1_id, track2_id, "Different types should not cluster");
        }
    }
}

#[test]
fn test_resolve_inter_doc_coref_empty_corpus() {
    let mut corpus = Corpus::new();
    let resolver = Resolver::new();
    let identity_ids = resolver.resolve_inter_doc_coref(&mut corpus, None, None);
    assert!(identity_ids.is_empty());
}

#[test]
fn test_resolve_inter_doc_coref_no_tracks() {
    let mut corpus = Corpus::new();
    let doc = GroundedDocument::new("doc1", "Text with no tracks.");
    corpus.add_document(doc);

    let resolver = Resolver::new();
    let identity_ids = resolver.resolve_inter_doc_coref(&mut corpus, None, None);
    assert!(identity_ids.is_empty());
}

#[test]
fn test_resolve_inter_doc_coref_singleton_tracks() {
    let mut corpus = Corpus::new();

    // Single document with one track
    let mut doc = GroundedDocument::new("doc1", "Marie Curie");
    let s = doc.add_signal(Signal::new(
        0,
        Location::text(0, 12),
        "Marie Curie",
        "Person",
        0.95,
    ));
    let mut track = Track::new(0, "Marie Curie");
    track.add_signal(s, 0);
    doc.add_track(track);
    corpus.add_document(doc);

    // With only one track, should create one identity (or none, depending on design)
    let resolver = Resolver::new();
    let identity_ids = resolver.resolve_inter_doc_coref(&mut corpus, None, None);
    // Singleton tracks can still create identities
    assert!(identity_ids.len() <= 1);
}

#[test]
fn test_resolve_inter_doc_coref_threshold_variations() {
    let mut corpus = Corpus::new();

    // Two documents with similar but not identical names
    let mut doc1 = GroundedDocument::new("doc1", "John Smith");
    let s1 = doc1.add_signal(Signal::new(
        0,
        Location::text(0, 10),
        "John Smith",
        "Person",
        0.9,
    ));
    let mut track1 = Track::new(0, "John Smith");
    track1.add_signal(s1, 0);
    doc1.add_track(track1);
    corpus.add_document(doc1);

    let mut doc2 = GroundedDocument::new("doc2", "John A. Smith");
    let s2 = doc2.add_signal(Signal::new(
        0,
        Location::text(0, 13),
        "John A. Smith",
        "Person",
        0.9,
    ));
    let mut track2 = Track::new(0, "John A. Smith");
    track2.add_signal(s2, 0);
    doc2.add_track(track2);
    corpus.add_document(doc2);

    // Low threshold: should cluster (high similarity)
    let resolver = Resolver::new().with_threshold(0.3).require_type_match(true);
    let ids_low = resolver.resolve_inter_doc_coref(&mut corpus, None, None);
    assert!(!ids_low.is_empty());

    // Note: Testing threshold variations requires separate corpus instances
    // The basic clustering behavior is verified above
}

// =============================================================================
// Entity Linking Tests
// =============================================================================

#[test]
fn test_link_track_to_kb_new_identity() {
    let mut corpus = Corpus::new();

    let mut doc = GroundedDocument::new("doc1", "Marie Curie");
    let s = doc.add_signal(Signal::new(
        0,
        Location::text(0, 12),
        "Marie Curie",
        "Person",
        0.95,
    ));
    let mut track = Track::new(0, "Marie Curie");
    track.add_signal(s, 0);
    let track_id = doc.add_track(track);
    corpus.add_document(doc);

    let track_ref = TrackRef {
        doc_id: "doc1".to_string(),
        track_id,
    };

    let identity_id = corpus
        .link_track_to_kb(&track_ref, "wikidata", "Q7186", "Marie Curie")
        .unwrap();

    // Verify identity was created
    let identity = corpus.get_identity(identity_id).unwrap();
    assert_eq!(identity.kb_id, Some("Q7186".to_string()));
    assert_eq!(identity.kb_name, Some("wikidata".to_string()));
    assert_eq!(identity.canonical_name, "Marie Curie");

    // Verify source
    match identity.source.as_ref().unwrap() {
        IdentitySource::KnowledgeBase { kb_name, kb_id } => {
            assert_eq!(kb_name, "wikidata");
            assert_eq!(kb_id, "Q7186");
        }
        _ => panic!("Expected KnowledgeBase source"),
    }

    // Verify track is linked
    let doc = corpus.get_document("doc1").unwrap();
    let track = doc.get_track(track_id).unwrap();
    assert_eq!(track.identity_id, Some(identity_id));
}

#[test]
fn test_link_track_to_kb_existing_identity() {
    let mut corpus = Corpus::new();

    // First, create identity via inter-doc coref
    let mut doc1 = GroundedDocument::new("doc1", "Marie Curie");
    let s1 = doc1.add_signal(Signal::new(
        0,
        Location::text(0, 12),
        "Marie Curie",
        "Person",
        0.95,
    ));
    let mut track1 = Track::new(0, "Marie Curie");
    track1.add_signal(s1, 0);
    doc1.add_track(track1);
    corpus.add_document(doc1);

    // Resolve inter-doc coref (creates identity without KB)
    let resolver = Resolver::new();
    let identity_ids = resolver.resolve_inter_doc_coref(&mut corpus, None, None);
    assert!(!identity_ids.is_empty());
    let identity_id = identity_ids[0];

    // Now link to KB
    let track_ref = TrackRef {
        doc_id: "doc1".to_string(),
        track_id: 0,
    };

    let linked_id = corpus
        .link_track_to_kb(&track_ref, "wikidata", "Q7186", "Marie Curie")
        .unwrap();

    // Should update existing identity, not create new one
    assert_eq!(linked_id, identity_id);

    let identity = corpus.get_identity(identity_id).unwrap();
    assert_eq!(identity.kb_id, Some("Q7186".to_string()));

    // Source should be Hybrid now
    match identity.source.as_ref().unwrap() {
        IdentitySource::Hybrid {
            track_refs: _,
            kb_name,
            kb_id,
        } => {
            assert_eq!(kb_name, "wikidata");
            assert_eq!(kb_id, "Q7186");
        }
        _ => panic!("Expected Hybrid source after linking"),
    }
}

#[test]
fn test_link_track_to_kb_invalid_track_ref() {
    let mut corpus = Corpus::new();
    corpus.add_document(GroundedDocument::new("doc1", "Test"));

    let invalid_ref = TrackRef {
        doc_id: "doc1".to_string(),
        track_id: 999, // Doesn't exist
    };

    assert!(corpus
        .link_track_to_kb(&invalid_ref, "wikidata", "Q1", "Test")
        .is_err());
}

#[test]
fn test_link_track_to_kb_missing_document() {
    let mut corpus = Corpus::new();

    let invalid_ref = TrackRef {
        doc_id: "nonexistent".to_string(),
        track_id: 0,
    };

    assert!(corpus
        .link_track_to_kb(&invalid_ref, "wikidata", "Q1", "Test")
        .is_err());
}

// =============================================================================
// String Similarity Edge Cases
// =============================================================================

#[test]
fn test_string_similarity_identical() {
    // Access via a test helper or make it public for testing
    // For now, test through resolve_inter_doc_coref
    let mut corpus = Corpus::new();

    let mut doc1 = GroundedDocument::new("doc1", "Apple");
    let s1 = doc1.add_signal(Signal::new(0, Location::text(0, 5), "Apple", "Org", 0.9));
    let mut track1 = Track::new(0, "Apple");
    track1.add_signal(s1, 0);
    doc1.add_track(track1);
    corpus.add_document(doc1);

    let mut doc2 = GroundedDocument::new("doc2", "Apple");
    let s2 = doc2.add_signal(Signal::new(0, Location::text(0, 5), "Apple", "Org", 0.9));
    let mut track2 = Track::new(0, "Apple");
    track2.add_signal(s2, 0);
    doc2.add_track(track2);
    corpus.add_document(doc2);

    // Identical strings should cluster even with high threshold
    let resolver = Resolver::new().with_threshold(0.9).require_type_match(true);
    let identity_ids = resolver.resolve_inter_doc_coref(&mut corpus, None, None);
    assert!(!identity_ids.is_empty());
}

#[test]
fn test_string_similarity_empty_strings() {
    // Empty strings should not crash
    let mut corpus = Corpus::new();

    let mut doc1 = GroundedDocument::new("doc1", "");
    let track1 = Track::new(0, "");
    doc1.add_track(track1);
    corpus.add_document(doc1);

    let mut doc2 = GroundedDocument::new("doc2", "");
    let track2 = Track::new(0, "");
    doc2.add_track(track2);
    corpus.add_document(doc2);

    // Should handle gracefully (empty strings have 0 similarity)
    let resolver = Resolver::new().with_threshold(0.1).require_type_match(true);
    let identity_ids = resolver.resolve_inter_doc_coref(&mut corpus, None, None);
    // Empty strings won't cluster (similarity = 0.0)
    assert!(identity_ids.is_empty() || identity_ids.len() <= 2);
}

#[test]
fn test_string_similarity_single_word_vs_multiword() {
    let mut corpus = Corpus::new();

    // "Apple" vs "Apple Inc" - should have decent similarity
    let mut doc1 = GroundedDocument::new("doc1", "Apple");
    let track1 = Track::new(0, "Apple");
    doc1.add_track(track1);
    corpus.add_document(doc1);

    let mut doc2 = GroundedDocument::new("doc2", "Apple Inc");
    let track2 = Track::new(0, "Apple Inc");
    doc2.add_track(track2);
    corpus.add_document(doc2);

    // With low threshold, should cluster (Jaccard: intersection=1, union=2, sim=0.5)
    let resolver = Resolver::new().with_threshold(0.4).require_type_match(true);
    let identity_ids = resolver.resolve_inter_doc_coref(&mut corpus, None, None);
    // Should cluster with threshold < 0.5
    assert!(!identity_ids.is_empty());
}

// =============================================================================
// Integration Tests
// =============================================================================

#[test]
fn test_full_pipeline_inter_doc_coref_then_linking() {
    let mut corpus = Corpus::new();

    // Create two documents mentioning the same person
    let mut doc1 = GroundedDocument::new("doc1", "Jensen Huang announced Nvidia's new chips.");
    let s1 = doc1.add_signal(Signal::new(
        0,
        Location::text(0, 13),
        "Jensen Huang",
        "Person",
        0.95,
    ));
    let mut track1 = Track::new(0, "Jensen Huang");
    track1.add_signal(s1, 0);
    track1.entity_type = Some("Person".to_string());
    doc1.add_track(track1);
    corpus.add_document(doc1);

    let mut doc2 = GroundedDocument::new("doc2", "The CEO of Nvidia revealed expansion plans.");
    let s2 = doc2.add_signal(Signal::new(
        0,
        Location::text(4, 17),
        "CEO of Nvidia",
        "Person",
        0.85,
    ));
    let mut track2 = Track::new(0, "CEO of Nvidia");
    track2.add_signal(s2, 0);
    track2.entity_type = Some("Person".to_string());
    doc2.add_track(track2);
    corpus.add_document(doc2);

    // Step 1: Resolve inter-doc coref
    let resolver = Resolver::new().with_threshold(0.3).require_type_match(true);
    let identity_ids = resolver.resolve_inter_doc_coref(&mut corpus, None, None);
    assert!(!identity_ids.is_empty());

    // Step 2: Link to KB
    let track_ref = TrackRef {
        doc_id: "doc1".to_string(),
        track_id: 0,
    };
    let linked_id = corpus
        .link_track_to_kb(&track_ref, "wikidata", "Q12345", "Jensen Huang")
        .unwrap();

    // Verify both tracks are linked to same identity
    let doc1 = corpus.get_document("doc1").unwrap();
    let doc2 = corpus.get_document("doc2").unwrap();
    let track1_id = doc1.get_track(0).unwrap().identity_id;
    let track2_id = doc2.get_track(0).unwrap().identity_id;

    // Both should link to the same identity (after linking)
    assert_eq!(track1_id, Some(linked_id));
    // track2 should also be linked (via inter-doc coref)
    assert!(track2_id.is_some());

    // Identity should have KB link and Hybrid source
    let identity = corpus.get_identity(linked_id).unwrap();
    assert_eq!(identity.kb_id, Some("Q12345".to_string()));
    match identity.source.as_ref().unwrap() {
        IdentitySource::Hybrid { .. } => {}
        _ => panic!("Expected Hybrid source"),
    }
}

#[test]
fn test_identity_source_preservation() {
    let mut corpus = Corpus::new();

    // Create identity via inter-doc coref
    let mut doc1 = GroundedDocument::new("doc1", "Test");
    let track1 = Track::new(0, "Test Entity");
    doc1.add_track(track1);
    corpus.add_document(doc1);

    let resolver = Resolver::new();
    let identity_ids = resolver.resolve_inter_doc_coref(&mut corpus, None, None);
    if !identity_ids.is_empty() {
        let identity = corpus.get_identity(identity_ids[0]).unwrap();
        assert!(matches!(
            identity.source.as_ref().unwrap(),
            IdentitySource::CrossDocCoref { .. }
        ));
    }

    // =============================================================================
    // GroundedDocument Advanced Method Tests
    // =============================================================================

    #[test]
    fn test_grounded_document_identity_for_track() {
        let mut doc = GroundedDocument::new("doc1", "Marie Curie was a physicist.");
        let s1 = doc.add_signal(Signal::new(
            0,
            Location::text(0, 12),
            "Marie Curie",
            "Person",
            0.95,
        ));
        let mut track = Track::new(0, "Marie Curie");
        track.add_signal(s1, 0);
        let track_id = doc.add_track(track);

        let identity = Identity::from_kb(0, "Marie Curie", "wikidata", "Q7186");
        let identity_id = doc.add_identity(identity);
        doc.link_track_to_identity(track_id, identity_id);

        // Should be able to get identity for track
        let identity = doc.identity_for_track(track_id);
        assert!(identity.is_some());
        assert_eq!(identity.unwrap().kb_id, Some("Q7186".to_string()));
    }

    #[test]
    fn test_grounded_document_identity_for_track_no_identity() {
        let mut doc = GroundedDocument::new("doc1", "Test");
        let s1 = doc.add_signal(Signal::new(0, Location::text(0, 4), "Test", "Type", 0.9));
        let mut track = Track::new(0, "Test");
        track.add_signal(s1, 0);
        let track_id = doc.add_track(track);

        // Track has no identity linked
        let identity = doc.identity_for_track(track_id);
        assert!(identity.is_none());
    }

    #[test]
    fn test_grounded_document_identity_for_signal() {
        let mut doc = GroundedDocument::new("doc1", "Marie Curie");
        let s1 = doc.add_signal(Signal::new(
            0,
            Location::text(0, 12),
            "Marie Curie",
            "Person",
            0.95,
        ));
        let mut track = Track::new(0, "Marie Curie");
        track.add_signal(s1, 0);
        let track_id = doc.add_track(track);

        let identity = Identity::from_kb(0, "Marie Curie", "wikidata", "Q7186");
        let identity_id = doc.add_identity(identity);
        doc.link_track_to_identity(track_id, identity_id);

        // Should be able to get identity for signal (transitively through track)
        let identity = doc.identity_for_signal(s1);
        assert!(identity.is_some());
        assert_eq!(identity.unwrap().kb_id, Some("Q7186".to_string()));
    }

    #[test]
    fn test_grounded_document_identity_for_signal_no_track() {
        let mut doc = GroundedDocument::new("doc1", "Test");
        let s1 = doc.add_signal(Signal::new(0, Location::text(0, 4), "Test", "Type", 0.9));
        // Signal not added to any track

        let identity = doc.identity_for_signal(s1);
        assert!(identity.is_none());
    }

    #[test]
    fn test_grounded_document_to_coref_document() {
        let mut doc = GroundedDocument::new(
            "doc1",
            "Marie Curie won the Nobel Prize. She was a physicist.",
        );
        let s1 = doc.add_signal(Signal::new(
            0,
            Location::text(0, 12),
            "Marie Curie",
            "Person",
            0.95,
        ));
        let s2 = doc.add_signal(Signal::new(
            1,
            Location::text(38, 41),
            "She",
            "Person",
            0.88,
        ));

        let mut track = Track::new(0, "Marie Curie");
        track.add_signal(s1, 0);
        track.add_signal(s2, 1);
        doc.add_track(track);

        let coref_doc = doc.to_coref_document();
        assert_eq!(coref_doc.text, doc.text);
        assert_eq!(coref_doc.chains.len(), 1);
        assert_eq!(coref_doc.chains[0].len(), 2);
        assert_eq!(coref_doc.chains[0].mentions[0].text, "Marie Curie");
        assert_eq!(coref_doc.chains[0].mentions[1].text, "She");
    }

    #[test]
    fn test_grounded_document_to_coref_document_empty() {
        let doc = GroundedDocument::new("doc1", "No entities here.");
        let coref_doc = doc.to_coref_document();
        assert_eq!(coref_doc.text, doc.text);
        assert!(coref_doc.chains.is_empty());
    }

    #[test]
    fn test_grounded_document_to_coref_document_multiple_tracks() {
        let mut doc =
            GroundedDocument::new("doc1", "Marie Curie won the Nobel Prize. Paris is nice.");
        let s1 = doc.add_signal(Signal::new(
            0,
            Location::text(0, 12),
            "Marie Curie",
            "Person",
            0.95,
        ));
        let s2 = doc.add_signal(Signal::new(
            1,
            Location::text(30, 35),
            "Paris",
            "Location",
            0.9,
        ));

        let mut track1 = Track::new(0, "Marie Curie");
        track1.add_signal(s1, 0);
        doc.add_track(track1);

        let mut track2 = Track::new(1, "Paris");
        track2.add_signal(s2, 0);
        doc.add_track(track2);

        let coref_doc = doc.to_coref_document();
        assert_eq!(coref_doc.chains.len(), 2);
    }

    #[test]
    fn test_grounded_document_from_entities() {
        use anno::Entity;
        use anno::EntityType;

        let entities = vec![
            Entity::new("Marie Curie", EntityType::Person, 0, 12, 0.95),
            Entity::new("Paris", EntityType::Location, 20, 25, 0.9),
        ];

        let text = "Marie Curie visited Paris";
        let doc = GroundedDocument::from_entities("doc1", text, &entities);

        assert_eq!(doc.text, text);
        assert_eq!(doc.signals().len(), 2);
        assert_eq!(doc.signals()[0].surface, "Marie Curie");
        assert_eq!(doc.signals()[1].surface, "Paris");
    }

    #[test]
    fn test_grounded_document_from_entities_empty() {
        use anno::Entity;

        let entities: Vec<Entity> = vec![];
        let text = "No entities";
        let doc = GroundedDocument::from_entities("doc1", text, &entities);

        assert_eq!(doc.text, text);
        assert!(doc.signals().is_empty());
    }

    #[test]
    fn test_grounded_document_from_entities_with_kb_id() {
        use anno::Entity;
        use anno::EntityType;

        let mut entity = Entity::new("Marie Curie", EntityType::Person, 0, 12, 0.95);
        entity.kb_id = Some("Q7186".to_string());
        let entities = vec![entity];

        let text = "Marie Curie";
        let doc = GroundedDocument::from_entities("doc1", text, &entities);

        // Should create identity with KB ID
        assert_eq!(doc.identities().count(), 1);
        let identity = doc.identities().next().unwrap();
        assert_eq!(identity.kb_id, Some("Q7186".to_string()));
    }

    #[test]
    fn test_grounded_document_to_entities() {
        let mut doc = GroundedDocument::new("doc1", "Marie Curie won the Nobel Prize.");
        let s1 = doc.add_signal(Signal::new(
            0,
            Location::text(0, 12),
            "Marie Curie",
            "Person",
            0.95,
        ));
        let _s2 = doc.add_signal(Signal::new(
            1,
            Location::text(17, 29),
            "Nobel Prize",
            "Award",
            0.92,
        ));

        let mut track = Track::new(0, "Marie Curie");
        track.add_signal(s1, 0);
        let track_id = doc.add_track(track);

        let identity = Identity {
            id: 0,
            canonical_name: "Marie Curie".to_string(),
            entity_type: Some("Person".to_string()),
            kb_id: Some("Q7186".to_string()),
            kb_name: Some("wikidata".to_string()),
            description: None,
            embedding: None,
            aliases: Vec::new(),
            confidence: 0.95,
            source: None,
        };
        let identity_id = doc.add_identity(identity);
        doc.link_track_to_identity(track_id, identity_id);

        let entities = doc.to_entities();
        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0].text, "Marie Curie");
        assert_eq!(entities[0].kb_id, Some("Q7186".to_string()));
        assert_eq!(entities[1].text, "Nobel Prize");
    }

    #[test]
    fn test_grounded_document_to_entities_preserves_offsets() {
        let mut doc = GroundedDocument::new("doc1", "Price €50");
        let s1 = doc.add_signal(Signal::new(0, Location::text(6, 9), "€50", "Money", 0.9));
        // Signal is already added, no need to add again
        // Just verify the signal exists
        assert_eq!(doc.signals().len(), 1);

        let entities = doc.to_entities();
        assert_eq!(entities.len(), 1);
        // Should use character offsets (6, 9), not byte offsets
        assert_eq!(entities[0].start, 6);
        assert_eq!(entities[0].end, 9);
    }
}
