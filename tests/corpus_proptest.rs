//! Property-based tests for Corpus invariants.
//!
//! These tests verify that certain properties hold for ALL valid inputs,
//! not just specific examples. They catch edge cases that unit tests miss.

use anno::grounded::{Corpus, GroundedDocument, Location, Signal, Track};
use anno_coalesce::Resolver;
use proptest::prelude::*;

proptest! {
    #[test]
    fn corpus_identity_ids_are_unique(
        num_docs in 1..=10usize,
        num_tracks_per_doc in 1..=5usize,
    ) {
        let mut corpus = Corpus::new();

        // Add documents with tracks
        for doc_idx in 0..num_docs {
            let mut doc = GroundedDocument::new(format!("doc{}", doc_idx), "Test");
            for track_idx in 0..num_tracks_per_doc {
                let s = doc.add_signal(Signal::new(
                    track_idx as u64,
                    Location::text(0, 4),
                    "Test",
                    "Type",
                    0.9,
                ));
                let mut track = Track::new(track_idx as u64, "Test");
                track.add_signal(s, 0);
                doc.add_track(track);
            }
            corpus.add_document(doc);
        }

        // Resolve inter-doc coref
        let resolver = Resolver::new().with_threshold(0.5).require_type_match(false);
        let identity_ids = resolver.resolve_inter_doc_coref(&mut corpus, None, None);

        // All identity IDs should be unique
        let mut seen = std::collections::HashSet::new();
        for id in &identity_ids {
            assert!(seen.insert(*id), "Duplicate identity ID: {}", id);
        }
    }

    #[test]
    fn corpus_track_linking_is_consistent(
        num_docs in 2..=5usize,
    ) {
        let mut corpus = Corpus::new();

        // Create documents with identical track names (should cluster)
        for doc_idx in 0..num_docs {
            let mut doc = GroundedDocument::new(format!("doc{}", doc_idx), "Test");
            let s = doc.add_signal(Signal::new(0, Location::text(0, 4), "Test", "Type", 0.9));
            let mut track = Track::new(0, "Test");
            track.add_signal(s, 0);
            doc.add_track(track);
            corpus.add_document(doc);
        }

        // Resolve inter-doc coref with low threshold (should cluster all)
        let resolver = Resolver::new().with_threshold(0.1).require_type_match(false);
        let identity_ids = resolver.resolve_inter_doc_coref(&mut corpus, None, None);

        // All tracks should be linked to identities
        for doc in corpus.documents() {
            for track in doc.tracks() {
                // Track should be linked to an identity
                assert!(track.identity_id.is_some(), "Track should be linked to identity");

                // Identity should exist in corpus
                let identity_id = track.identity_id.unwrap();
                assert!(corpus.get_identity(identity_id).is_some(),
                    "Identity {} should exist in corpus", identity_id);
            }
        }
    }

    #[test]
    fn corpus_identity_count_bounds(
        num_docs in 1..=10usize,
        num_tracks_per_doc in 1..=5usize,
    ) {
        let mut corpus = Corpus::new();

        // Add documents
        for doc_idx in 0..num_docs {
            let mut doc = GroundedDocument::new(format!("doc{}", doc_idx), "Test");
            for track_idx in 0..num_tracks_per_doc {
                let s = doc.add_signal(Signal::new(
                    track_idx as u64,
                    Location::text(0, 4),
                    &format!("Track{}", track_idx),
                    "Type",
                    0.9,
                ));
                let mut track = Track::new(track_idx as u64, &format!("Track{}", track_idx));
                track.add_signal(s, 0);
                doc.add_track(track);
            }
            corpus.add_document(doc);
        }

        // Resolve inter-doc coref
        let resolver = Resolver::new().with_threshold(0.5).require_type_match(false);
        let identity_ids = resolver.resolve_inter_doc_coref(&mut corpus, None, None);

        // Number of identities should be between 1 and total tracks
        let total_tracks = num_docs * num_tracks_per_doc;
        assert!(identity_ids.len() >= 1, "Should have at least 1 identity");
        assert!(identity_ids.len() <= total_tracks,
            "Should have at most {} identities, got {}", total_tracks, identity_ids.len());
    }

    #[test]
    fn corpus_identity_source_preservation(
        track_name in "[A-Za-z]+",
    ) {
        let mut corpus = Corpus::new();

        // Create two documents with same track name
        for doc_idx in 0..2 {
            let mut doc = GroundedDocument::new(format!("doc{}", doc_idx), "Test");
            let s = doc.add_signal(Signal::new(0, Location::text(0, 4), &track_name, "Type", 0.9));
            let mut track = Track::new(0, &track_name);
            track.add_signal(s, 0);
            doc.add_track(track);
            corpus.add_document(doc);
        }

        // Resolve inter-doc coref
        let resolver = Resolver::new().with_threshold(0.5).require_type_match(false);
        let identity_ids = resolver.resolve_inter_doc_coref(&mut corpus, None, None);

        // If identities were created, they should have CrossDocCoref source
        for id in identity_ids {
            if let Some(identity) = corpus.get_identity(id) {
                if let Some(source) = &identity.source {
                    match source {
                        anno::grounded::IdentitySource::CrossDocCoref { .. } => {
                            // Good - this is expected
                        }
                        _ => {
                            panic!("Identity created by inter-doc coref should have CrossDocCoref source");
                        }
                    }
                }
            }
        }
    }
}
