// SPDX-License-Identifier: MIT OR Apache-2.0
//! Offline replay of original source-bound annotations through the native Corpus and Resolver.
use anno::coalesce::Resolver;
use anno::core::grounded::html_source::source_sha256;
use anno::{Corpus, GroundedDocument, Location, Signal, TrackRef};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Edition {
    doc: &'static str,
    revision: u8,
    text: &'static str,
    surface: &'static str,
    span: (usize, usize),
}

// Authored gold: the concert roster's H4 and the pilot register's P8 designate two people.
// Both versions of the manifest and the watch report refer to pilot P8.
const HARP: Edition = Edition {
    doc: "concert",
    revision: 1,
    text: "🎼 Harpist Lira Voss signed roster H4.",
    surface: "Lira Voss",
    span: (10, 19),
};
const MANIFEST: Edition = Edition {
    doc: "manifest",
    revision: 1,
    text: "Pilot Lira Voss Pell signed register P8.",
    surface: "Lira Voss Pell",
    span: (6, 20),
};
const CORRECTED: Edition = Edition {
    revision: 2,
    text: "Pilot Voss Pell signed register P8.",
    surface: "Voss Pell",
    span: (6, 15),
    ..MANIFEST
};
const WATCH: Edition = Edition {
    doc: "watch",
    revision: 1,
    text: "At dusk, Voss Pell renewed register P8.",
    surface: "Voss Pell",
    span: (9, 18),
};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Anchor {
    document: String,
    source_sha256: String,
    span: (usize, usize),
}
type Partition = Vec<Vec<Anchor>>;

impl Edition {
    fn anchor(self) -> Anchor {
        Anchor {
            document: self.doc.into(),
            source_sha256: source_sha256(self.text),
            span: self.span,
        }
    }

    fn document(self) -> GroundedDocument {
        let mut doc = GroundedDocument::new(self.doc, self.text);
        let location = Location::text(self.span.0, self.span.1);
        let signal = doc
            .add_signal_validated(Signal::new(0, location, self.surface, "PER", 1.0))
            .unwrap();
        doc.create_track_from_signals(self.surface, &[signal])
            .unwrap();
        doc
    }
}

fn observed(corpus: &Corpus, member: &TrackRef) -> Anchor {
    let doc = corpus.get_document(&member.doc_id).unwrap();
    let track = doc.get_track(member.track_id).unwrap();
    let [mention] = track.signals() else {
        panic!("expected one mention in this fixture track")
    };
    let signal = doc.get_signal(mention.signal_id).unwrap();
    Anchor {
        document: doc.id().into(),
        source_sha256: source_sha256(doc.text()),
        span: signal.text_offsets().unwrap(),
    }
}

fn partition(corpus: &Corpus) -> Partition {
    let mut groups: Partition = corpus
        .current_coref_clusters()
        .into_iter()
        .map(|(_, members)| {
            let mut anchors: Vec<_> = members.iter().map(|m| observed(corpus, m)).collect();
            anchors.sort();
            anchors
        })
        .collect();
    groups.sort();
    groups
}

fn gold(manifest: Edition) -> Partition {
    vec![vec![HARP.anchor()], vec![manifest.anchor(), WATCH.anchor()]]
}

fn pairs(groups: &Partition) -> BTreeSet<(&Anchor, &Anchor)> {
    let mut pairs = BTreeSet::new();
    for group in groups {
        for (index, left) in group.iter().enumerate() {
            pairs.extend(group[index + 1..].iter().map(|right| (left, right)));
        }
    }
    pairs
}

fn pair_counts(current: &Partition, expected: &Partition) -> (usize, usize, usize) {
    let predicted = pairs(current);
    let truth = pairs(expected);
    (
        predicted.intersection(&truth).count(),
        predicted.difference(&truth).count(),
        truth.difference(&predicted).count(),
    )
}

#[derive(Clone, Copy, PartialEq)]
enum Intake {
    Arrival,
    LatestRevision,
}

#[derive(Default)]
struct Replay {
    corpus: Corpus,
    latest: BTreeMap<&'static str, Edition>,
}

impl Replay {
    // In this caller's fixture, each (document, revision) names one immutable edition.
    // Older revisions are dropped. A repeated current revision must match its payload.
    // Native add_document has no revision and replaces by document ID.
    fn ingest(&mut self, edition: Edition, mode: Intake) -> bool {
        if mode == Intake::LatestRevision {
            if let Some(old) = self.latest.get(edition.doc) {
                if old.revision == edition.revision {
                    assert_eq!(
                        *old, edition,
                        "a duplicate revision must have identical input"
                    );
                    return false;
                }
                if old.revision > edition.revision {
                    return false;
                }
            }
            self.latest.insert(edition.doc, edition);
        }
        self.corpus.add_document(edition.document());
        true
    }

    fn resolve(&mut self) {
        Resolver::new()
            .with_threshold(0.60)
            .resolve_inter_doc_coref(&mut self.corpus, None, None);
    }

    // Buffering changes only the frequency of a full native resolution pass.
    fn run(events: &[Edition], batch_size: usize, mode: Intake) -> Self {
        let mut replay = Self::default();
        for batch in events.chunks(batch_size) {
            for &event in batch {
                replay.ingest(event, mode);
            }
            replay.resolve();
        }
        replay
    }

    fn track(&self, edition: Edition) -> TrackRef {
        let doc = self.corpus.get_document(edition.doc).unwrap();
        doc.track_ref(doc.tracks().next().unwrap().id).unwrap()
    }
}

fn each_order(events: &mut [Edition], fixed: usize, visit: &mut impl FnMut(&[Edition])) {
    if fixed == events.len() {
        visit(events);
        return;
    }
    for next in fixed..events.len() {
        events.swap(fixed, next);
        each_order(events, fixed + 1, visit);
        events.swap(fixed, next);
    }
}

fn bulk(manifest: Edition) -> Partition {
    let reference = Replay::run(&[HARP, manifest, WATCH], 3, Intake::Arrival);
    let groups = partition(&reference.corpus);
    let mut population: Vec<_> = groups.iter().flatten().cloned().collect();
    population.sort();
    assert_eq!(
        population,
        [HARP.anchor(), manifest.anchor(), WATCH.anchor()]
    );
    groups
}

#[test]
fn all_arrival_orders_reproduce_a_fresh_bulk_resolution() {
    let reference = bulk(MANIFEST);
    let mut ids = BTreeSet::new();
    let mut runs = 0;
    each_order(&mut [HARP, MANIFEST, WATCH], 0, &mut |order| {
        for batch in [1, 2, 3] {
            let replay = Replay::run(order, batch, Intake::Arrival);
            assert_eq!(partition(&replay.corpus), reference);
            let doc = replay.corpus.get_document(HARP.doc).unwrap();
            let track = replay.track(HARP);
            ids.insert(doc.identity_for_track(track.track_id).unwrap().id);
            runs += 1;
        }
    });
    assert_eq!(runs, 18);
    let counts = pair_counts(&reference, &gold(MANIFEST));
    println!(
        "{runs} bridge replays match bulk. Concert identity has {} distinct IDs. Bulk (TP, FP, FN) {counts:?}.",
        ids.len(),
    );
}

#[test]
fn correction_delivery_order_and_version_filter_are_distinguishable() {
    let mut native_correct = 0;
    let mut guarded_correct = 0;
    let mut runs = 0;
    let corrected_gold = gold(CORRECTED);
    let corrected = bulk(CORRECTED);
    let original = bulk(MANIFEST);
    each_order(&mut [HARP, MANIFEST, WATCH, CORRECTED], 0, &mut |order| {
        let last = order
            .iter()
            .rev()
            .find(|event| event.doc == MANIFEST.doc)
            .unwrap();
        for batch in [1, 2, order.len()] {
            let native = Replay::run(order, batch, Intake::Arrival);
            let current = partition(&native.corpus);
            assert_eq!(
                native.corpus.get_document(MANIFEST.doc).unwrap().text(),
                last.text
            );
            assert_eq!(
                &current,
                if *last == CORRECTED {
                    &corrected
                } else {
                    &original
                }
            );
            native_correct += usize::from(current == corrected_gold);
            let guarded = Replay::run(order, batch, Intake::LatestRevision);
            let current = partition(&guarded.corpus);
            assert_eq!(
                guarded.corpus.get_document(MANIFEST.doc).unwrap().text(),
                CORRECTED.text
            );
            assert_eq!(current, corrected);
            guarded_correct += usize::from(current == corrected_gold);
            runs += 1;
        }
    });
    assert_eq!(runs, 72);
    println!("Newest authored partition recovered: arrival {native_correct}/{runs}, caller revision filter {guarded_correct}/{runs}");
}

#[test]
fn duplicate_and_late_retries_have_distinct_handle_and_source_effects() {
    let corrected_bulk = bulk(CORRECTED);
    let original_bulk = bulk(MANIFEST);
    for mode in [Intake::Arrival, Intake::LatestRevision] {
        let mut replay = Replay::run(&[HARP, MANIFEST, WATCH], 3, mode);
        let retired = replay.track(MANIFEST);
        assert!(replay.ingest(CORRECTED, mode));
        replay.resolve();
        let corrected = replay.track(CORRECTED);
        assert_ne!(corrected, retired);
        assert_eq!(partition(&replay.corpus), corrected_bulk);
        let accepted = mode == Intake::Arrival;
        assert_eq!(replay.ingest(CORRECTED, mode), accepted);
        replay.resolve();
        assert_eq!(partition(&replay.corpus), corrected_bulk);
        assert_eq!(replay.track(CORRECTED) != corrected, accepted);
        assert_eq!(replay.ingest(MANIFEST, mode), accepted);
        replay.resolve();
        let doc = replay.corpus.get_document(MANIFEST.doc).unwrap();
        assert!(doc.get_track(retired.track_id).is_none());
        assert_eq!(
            doc.text(),
            if accepted {
                MANIFEST.text
            } else {
                CORRECTED.text
            }
        );
        let expected = if accepted {
            &original_bulk
        } else {
            &corrected_bulk
        };
        assert_eq!(&partition(&replay.corpus), expected);
    }
}

#[test]
fn catalog_decisions_preserve_the_members_reviewed_at_each_checkpoint() {
    let events = [HARP, MANIFEST, WATCH];
    for checkpoint in 1..=events.len() {
        let mut replay = Replay::run(&events[..checkpoint], 1, Intake::Arrival);
        let trigger = replay.track(HARP);
        let approved = partition(&replay.corpus)
            .into_iter()
            .find(|members| members.contains(&HARP.anchor()))
            .unwrap();
        let preview = replay
            .corpus
            .preview_kb_assignment(&trigger, "fixture-roster", "H4", HARP.surface)
            .unwrap();
        let receipt = replay.corpus.apply_kb_assignment(preview).unwrap();
        for &event in &events[checkpoint..] {
            replay.ingest(event, Intake::Arrival);
        }
        replay.resolve();
        let actual_scope: Vec<_> = receipt
            .assignment
            .affected_tracks
            .iter()
            .map(|member| observed(&replay.corpus, member))
            .collect();
        assert_eq!(actual_scope, approved);
        let final_groups = partition(&replay.corpus);
        assert!(final_groups.contains(&approved));
        let counts = pair_counts(&final_groups, &gold(MANIFEST));
        println!(
            "Review checkpoint {checkpoint}: {} approved, final (TP, FP, FN) {counts:?}",
            approved.len()
        );
        for edition in events {
            let doc = replay.corpus.get_document(edition.doc).unwrap();
            let track = replay.track(edition);
            let target = doc
                .identity_for_track(track.track_id)
                .unwrap()
                .qualified_kb_ref();
            assert_eq!(
                target,
                approved
                    .contains(&edition.anchor())
                    .then_some(("fixture-roster", "H4"))
            );
        }
    }
}
