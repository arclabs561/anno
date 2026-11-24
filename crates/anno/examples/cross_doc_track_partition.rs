// SPDX-License-Identifier: MIT OR Apache-2.0
//! Run the native cross-document resolver on unlinked supplied mention tracks.
//! Stdin is an ordered array of batches, each mapping document IDs to ordered names.
//! Resolve once after each batch. A repeated document ID replaces its prior edition.
//! Stdout contains the corresponding identity numbers from this one resolver run.
use anno::coalesce::Resolver;
use anno::{Corpus, GroundedDocument, Location, Signal};
use std::collections::BTreeMap;
use std::error::Error;
use std::io;

fn document(id: &str, names: &[String]) -> Result<GroundedDocument, Box<dyn Error>> {
    let mut doc = GroundedDocument::new(id, names.join("\n"));
    let mut start = 0;
    for name in names {
        if name.is_empty() || name.contains('\n') {
            return Err("expected one nonempty name per occurrence".into());
        }
        let end = start + name.chars().count();
        let signal = Signal::new(0, Location::text(start, end), name, "PER", 1.0);
        let mention = doc.add_signal_validated(signal)?;
        let track = doc
            .create_track_from_signals(name, &[mention])
            .ok_or("could not form the supplied track")?;
        doc.get_track_mut(track).ok_or("missing track")?.entity_type = Some("PER".into());
        start = end + 1;
    }
    Ok(doc)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let threshold: Option<f32> = args.next().map(|s| s.parse()).transpose()?;
    if args.next().is_some() || threshold.is_some_and(|n| !(0.0..=1.0).contains(&n)) {
        return Err("usage: cross_doc_track_partition [threshold in 0..=1]".into());
    }
    let batches: Vec<BTreeMap<String, Vec<String>>> = serde_json::from_reader(io::stdin().lock())?;
    let mut corpus = Corpus::new();
    let resolver = Resolver::new();
    for batch in batches {
        for (id, names) in batch {
            corpus.add_document(document(&id, &names)?);
        }
        resolver.resolve_inter_doc_coref(&mut corpus, threshold, None);
    }
    let mut predictions = BTreeMap::new();
    for doc in corpus.documents() {
        let groups: Option<Vec<_>> = doc
            .signals()
            .iter()
            .map(|s| {
                doc.identity_for_signal(s.id)
                    .map(|identity| identity.id.get())
            })
            .collect();
        predictions.insert(doc.id(), groups.ok_or("unassigned native output")?);
    }
    serde_json::to_writer(io::stdout().lock(), &predictions)?;
    Ok(())
}
