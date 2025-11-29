//! Performance tests for Corpus operations on large datasets.

use anno::grounded::{Corpus, GroundedDocument, Location, Signal, Track};
use std::time::Instant;

#[test]
#[ignore] // Only run with --ignored flag
fn test_corpus_performance_large_corpus() {
    let mut corpus = Corpus::new();
    let num_docs = 100;
    let tracks_per_doc = 10;

    // Create large corpus
    let start = Instant::now();
    for doc_idx in 0..num_docs {
        let mut doc = GroundedDocument::new(format!("doc{}", doc_idx), "Test document");
        for track_idx in 0..tracks_per_doc {
            let s = doc.add_signal(Signal::new(
                track_idx as u64,
                Location::text(0, 4),
                &format!("Entity{}", track_idx),
                "Type",
                0.9,
            ));
            let mut track = Track::new(track_idx as u64, &format!("Entity{}", track_idx));
            track.add_signal(s, 0);
            doc.add_track(track);
        }
        corpus.add_document(doc);
    }
    let add_time = start.elapsed();

    // Resolve inter-doc coref
    let start = Instant::now();
    let identity_ids = corpus.resolve_inter_doc_coref(0.5, false);
    let coref_time = start.elapsed();

    println!(
        "Added {} documents with {} tracks each in {:?}",
        num_docs, tracks_per_doc, add_time
    );
    println!(
        "Resolved {} identities in {:?}",
        identity_ids.len(),
        coref_time
    );
    println!("Time per document: {:?}", add_time / num_docs as u32);
    println!(
        "Time per identity: {:?}",
        coref_time / identity_ids.len().max(1) as u32
    );

    // Performance assertions (adjust based on your requirements)
    assert!(add_time.as_secs() < 10, "Adding documents should be fast");
    assert!(
        coref_time.as_secs() < 30,
        "Coref resolution should complete in reasonable time"
    );
}

#[test]
#[ignore]
fn test_corpus_performance_high_similarity() {
    let mut corpus = Corpus::new();
    let num_docs = 50;

    // Create documents with very similar track names (will cluster)
    for doc_idx in 0..num_docs {
        let mut doc = GroundedDocument::new(format!("doc{}", doc_idx), "Test");
        let s = doc.add_signal(Signal::new(
            0,
            Location::text(0, 4),
            "Apple Inc",
            "Org",
            0.9,
        ));
        let mut track = Track::new(0, "Apple Inc");
        track.add_signal(s, 0);
        doc.add_track(track);
        corpus.add_document(doc);
    }

    let start = Instant::now();
    let identity_ids = corpus.resolve_inter_doc_coref(0.5, false);
    let elapsed = start.elapsed();

    // All should cluster into one identity
    assert_eq!(identity_ids.len(), 1);
    println!(
        "Clustered {} documents into 1 identity in {:?}",
        num_docs, elapsed
    );
    assert!(
        elapsed.as_secs() < 5,
        "High similarity clustering should be fast"
    );
}

#[test]
#[ignore]
fn test_corpus_performance_low_similarity() {
    let mut corpus = Corpus::new();
    let num_docs = 50;

    // Create documents with unique track names (won't cluster)
    for doc_idx in 0..num_docs {
        let mut doc = GroundedDocument::new(format!("doc{}", doc_idx), "Test");
        let s = doc.add_signal(Signal::new(
            0,
            Location::text(0, 4),
            &format!("Unique{}", doc_idx),
            "Type",
            0.9,
        ));
        let mut track = Track::new(0, &format!("Unique{}", doc_idx));
        track.add_signal(s, 0);
        doc.add_track(track);
        corpus.add_document(doc);
    }

    let start = Instant::now();
    let identity_ids = corpus.resolve_inter_doc_coref(0.5, false);
    let elapsed = start.elapsed();

    // Each should be its own identity
    assert_eq!(identity_ids.len(), num_docs);
    println!("Created {} separate identities in {:?}", num_docs, elapsed);
    assert!(
        elapsed.as_secs() < 5,
        "Low similarity (no clustering) should be fast"
    );
}
