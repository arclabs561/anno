// SPDX-License-Identifier: MIT OR Apache-2.0
//! Compare publication cadences of anno's native batch resolver on synthetic text.
//! Run `cargo run --release --locked -p anno --no-default-features --example index_reconcile_cost`.
//! Timings cover only calls to resolve_inter_doc_coref. Observation and index diffing
//! run outside the timer. Pair counts are computed from fixture sizes, not instrumented.

use anno::coalesce::Resolver;
use anno::{Corpus, GroundedDocument, IdentityId, Location, Signal, SignalId};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

const PEOPLE: usize = 16;
const REPEATS: usize = 5;

fn name(person: usize) -> String {
    format!("Person{person:02}")
}

fn document(index: usize) -> GroundedDocument {
    let surface = name(index % PEOPLE);
    let mut doc = GroundedDocument::new(format!("doc-{index:04}"), &surface);
    let signal = Signal::new(0, Location::text(0, surface.len()), &surface, "PER", 1.0);
    let id = doc.add_signal_validated(signal).unwrap();
    doc.create_track_from_signals(&surface, &[id]).unwrap();
    doc
}

fn pairs(n: usize) -> usize {
    n * n.saturating_sub(1) / 2
}

fn group_size(total: usize, person: usize) -> usize {
    total / PEOPLE + usize::from(person < total % PEOPLE)
}

struct Run {
    elapsed: Duration,
    calls: usize,
    candidate_pairs: usize,
    support_pairs: usize,
    old_postings_rebound: usize,
    bindings: Vec<IdentityId>,
}

fn run(size: usize, batch: usize) -> Run {
    let resolver = Resolver::new();
    let mut corpus = Corpus::new();
    let mut measured = Run {
        elapsed: Duration::ZERO,
        calls: 0,
        candidate_pairs: 0,
        support_pairs: 0,
        old_postings_rebound: 0,
        bindings: Vec::new(),
    };

    for index in 0..size {
        corpus.add_document(document(index));
        let total = index + 1;
        if total % batch != 0 && total != size {
            continue;
        }
        let start = Instant::now();
        let identities = resolver.resolve_inter_doc_coref(&mut corpus, None, None);
        measured.elapsed += start.elapsed();
        measured.calls += 1;
        measured.candidate_pairs += pairs(total);
        measured.support_pairs += (0..PEOPLE)
            .map(|group| pairs(group_size(total, group)))
            .sum::<usize>();

        // Check every source occurrence against its intended referent. The membership
        // partition and weakest-pair support should agree regardless of ID allocation.
        assert_eq!(identities.len(), total.min(PEOPLE));
        let mut grouped = BTreeMap::new();
        let mut bindings = Vec::with_capacity(total);
        for doc_index in 0..total {
            let person = doc_index % PEOPLE;
            let doc = corpus.get_document(&format!("doc-{doc_index:04}")).unwrap();
            let id = doc.identity_for_signal(SignalId::ZERO).unwrap();
            assert_eq!(id.canonical_name(), name(person));
            assert_eq!(
                id.minimum_pair_similarity.map(|s| s.value()),
                (group_size(total, person) > 1).then_some(1.0)
            );
            if let Some(previous) = grouped.insert(person, id.id()) {
                assert_eq!(previous, id.id());
            }
            bindings.push(id.id());
        }
        let mut distinct: Vec<_> = grouped.into_values().collect();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), total.min(PEOPLE));
        // Every older document has exactly one posting. A changed ID means an already
        // published binding needs repair even though its source document did not change.
        measured.old_postings_rebound += bindings
            .iter()
            .zip(&measured.bindings)
            .filter(|(current, prior)| current != prior)
            .count();
        measured.bindings = bindings;
    }
    measured
}

fn main() {
    let _ = run(16, 16);
    println!("documents,batch,calls,candidate_pairs_formula,support_pairs_formula,old_postings_rebound,final_ids_equal_to_bulk,median_resolve_ms");
    for size in [16, 64, 128, 256] {
        let mut batches = vec![1, size];
        if size > 32 {
            batches.insert(1, 32);
        }
        let mut runs: Vec<Vec<Run>> = batches.iter().map(|_| Vec::new()).collect();
        for repetition in 0..REPEATS {
            for offset in 0..batches.len() {
                let i = (repetition + offset) % batches.len();
                runs[i].push(run(size, batches[i]));
            }
        }
        let bulk_bindings = runs.last().unwrap()[0].bindings.clone();
        for (batch, mut samples) in batches.into_iter().zip(runs) {
            for other in &samples[1..] {
                assert_eq!(samples[0].bindings, other.bindings);
                assert_eq!(samples[0].old_postings_rebound, other.old_postings_rebound);
            }
            samples.sort_by_key(|r| r.elapsed);
            let r = &samples[REPEATS / 2];
            let equal = r
                .bindings
                .iter()
                .zip(&bulk_bindings)
                .filter(|(a, b)| a == b)
                .count();
            println!(
                "{size},{batch},{},{},{},{},{equal},{:.3}",
                r.calls,
                r.candidate_pairs,
                r.support_pairs,
                r.old_postings_rebound,
                r.elapsed.as_secs_f64() * 1000.0
            );
        }
    }
}
