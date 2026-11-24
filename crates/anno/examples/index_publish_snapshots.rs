// SPDX-License-Identifier: MIT OR Apache-2.0
//! Emit native resolver publications for an original, four-revision search fixture.
//! An empty replacement retires a source document's tracks; the empty envelope remains.
use anno::coalesce::Resolver;
use anno::{Corpus, GroundedDocument, Location, Signal};

fn add(corpus: &mut Corpus, doc_id: &str, text: &str) {
    let mut doc = GroundedDocument::new(doc_id, text);
    let signal = Signal::new(0, Location::text(0, text.len()), text, "PER", 1.0);
    let signal_id = doc.add_signal_validated(signal).unwrap();
    doc.create_track_from_signals(text, &[signal_id]).unwrap();
    corpus.add_document(doc);
}

fn retire(corpus: &mut Corpus, doc_id: &str) {
    corpus.add_document(GroundedDocument::new(doc_id, ""));
}

fn publish(corpus: &mut Corpus, revision: usize) {
    Resolver::new()
        .with_threshold(0.5)
        .resolve_inter_doc_coref(corpus, None, None);
    let mut docs: Vec<_> = corpus.documents().collect();
    docs.sort_by_key(|d| d.id());
    println!("snapshot\t{revision}");
    for doc in docs {
        for signal in doc.signals() {
            let identity = doc.identity_for_signal(signal.id).unwrap();
            // Each live source edition has revision 1; tombstones have no postings.
            println!(
                "posting\t{}\t1\t{}\t{}",
                doc.id(),
                signal.id.get(),
                identity.id.get()
            );
        }
    }
}

fn main() {
    let mut corpus = Corpus::new();
    add(&mut corpus, "a", "ash");
    add(&mut corpus, "bridge", "ash ivy");
    add(&mut corpus, "c", "ivy");
    publish(&mut corpus, 1);
    retire(&mut corpus, "bridge");
    publish(&mut corpus, 2);
    add(&mut corpus, "d", "ivy");
    publish(&mut corpus, 3);
    for id in ["a", "c", "d"] {
        retire(&mut corpus, id);
    }
    publish(&mut corpus, 4);
}
