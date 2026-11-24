//! Shared byte interpreter for the cargo-fuzz target and stable seed replay.
use anno::coalesce::Resolver;
use anno::core::grounded::{
    Corpus, GroundedDocument, Identity, IdentityId, KbAssignmentError, KbAssignmentPreview,
    SignalRef, Track, TrackRef,
};
use anno::{Location, Signal, SignalId};

pub const MAX_INPUT_BYTES: usize = 48;

/// The observed track binding disagreed with the independent fixture model.
#[derive(Debug)]
pub struct VisibleBindingMismatch;
type Target = (&'static str, &'static str, &'static str);
const ORIGINAL: Target = ("library", "42", "Ada");
const REVIEWED: Target = ("review", "7", "Reviewed");
const COMMANDS: &[u8] = b"012345pABNLmc!r";
const LOCAL_MENTION_STARTS: [usize; 3] = [0, 4, 8];

#[derive(Clone, Copy)]
struct Binding {
    id: IdentityId,
    target: Target,
    corpus_owned: bool,
}

struct Fixture {
    corpus: Corpus,
    tracks: [TrackRef; 6],
    expected: [Option<Binding>; 6],
    local_mentions: [SignalId; 3],
}

fn add_track(doc: &mut GroundedDocument) -> TrackRef {
    let id = doc.add_track(Track::new(0, "Ada"));
    doc.track_ref(id).unwrap()
}

fn add_document(corpus: &mut Corpus, name: &str) -> TrackRef {
    let mut doc = GroundedDocument::new(name, "Ada");
    let track = add_track(&mut doc);
    corpus.add_document(doc);
    track
}

impl Fixture {
    fn new() -> Self {
        let mut corpus = Corpus::new();
        let first = add_document(&mut corpus, "a");
        let peer = add_document(&mut corpus, "b");
        let shared = Resolver::new().resolve_inter_doc_coref(&mut corpus, None, None)[0];
        let (catalog, key, name) = ORIGINAL;
        assert_eq!(
            corpus.link_track_to_kb(&first, catalog, key, name).unwrap(),
            shared
        );
        let independent = add_document(&mut corpus, "c");
        let separate = corpus
            .link_track_to_kb(&independent, catalog, key, name)
            .unwrap();
        assert_ne!(shared, separate);

        let mut doc = GroundedDocument::new("local", "Ada Ada Ada");
        let local = add_track(&mut doc);
        let sibling = add_track(&mut doc);
        // Identical surfaces, distinct occurrences. The third has no assigned track.
        let local_mentions = LOCAL_MENTION_STARTS.map(|start| {
            let mention = Signal::new(0, Location::text(start, start + 3), "Ada", "PERSON", 1.0);
            doc.add_signal_validated(mention).unwrap()
        });
        assert!(doc.add_signal_to_track(local_mentions[0], local.track_id, 0));
        assert!(doc.add_signal_to_track(local_mentions[1], sibling.track_id, 0));
        let id = doc.add_identity(Identity::from_kb(0, name, catalog, key));
        doc.link_track_to_identity(local.track_id, id);
        doc.link_track_to_identity(sibling.track_id, id);
        corpus.add_document(doc);
        // Corpus import can remap a document-local numeric ID. Read its allocated value once.
        let local_id = corpus
            .get_document("local")
            .unwrap()
            .identity_for_track(local.track_id)
            .unwrap()
            .id;
        let unlinked = add_document(&mut corpus, "unlinked");
        let binding = |id, corpus_owned| {
            Some(Binding {
                id,
                target: ORIGINAL,
                corpus_owned,
            })
        };
        Self {
            corpus,
            local_mentions,
            tracks: [first, peer, independent, local, sibling, unlinked],
            expected: [
                binding(shared, true),
                binding(shared, true),
                binding(separate, true),
                binding(local_id, false),
                binding(local_id, false),
                None,
            ],
        }
    }

    // Only the first two tracks can be updated jointly. The two local siblings
    // initially share a record, but a corpus assignment rebinds its trigger alone.
    fn scope(&self, index: usize) -> std::ops::Range<usize> {
        if index < 2 {
            0..2
        } else {
            index..index + 1
        }
    }

    fn check_views(&self, corpus: &Corpus) {
        for (track, expected) in self.tracks.iter().zip(self.expected) {
            let doc = corpus.get_document(&track.doc_id).unwrap();
            assert!(doc.get_track(track.track_id).is_some());
            let visible = doc.identity_for_track(track.track_id).map(|record| {
                (
                    record.id,
                    record.qualified_kb_ref(),
                    record.canonical_name(),
                    corpus.is_shared_identity(record),
                )
            });
            let expected = expected.map(|b| {
                let (catalog, key, name) = b.target;
                (b.id, Some((catalog, key)), name, b.corpus_owned)
            });
            if visible != expected {
                eprintln!(
                    "binding mismatch at {track:?}: expected {expected:?}, observed {visible:?}"
                );
                std::panic::panic_any(VisibleBindingMismatch);
            }
        }

        let doc = corpus.get_document("local").unwrap();
        assert_eq!(doc.signals().len(), self.local_mentions.len());
        let owners = [
            Some(self.tracks[3].track_id),
            Some(self.tracks[4].track_id),
            None,
        ];
        for ((signal_id, start), track_id) in self
            .local_mentions
            .into_iter()
            .zip(LOCAL_MENTION_STARTS)
            .zip(owners)
        {
            let signal = doc.get_signal(signal_id).unwrap();
            assert_eq!(signal.surface, "Ada");
            assert_eq!(signal.location.text_offsets(), Some((start, start + 3)));
            assert_eq!(
                doc.track_for_signal(signal_id).map(|track| track.id),
                track_id
            );
            if let Some(track_id) = track_id {
                assert_eq!(
                    doc.get_track(track_id).unwrap().signals(),
                    &[SignalRef {
                        signal_id,
                        position: 0
                    }]
                );
            }
            // The track-level binding was checked against the oracle above. A caller
            // following the mention must see exactly that identity, or no identity.
            assert_eq!(
                doc.identity_for_signal(signal_id),
                track_id.and_then(|id| doc.identity_for_track(id))
            );
        }
    }

    fn prepare(&self, index: usize) -> KbAssignmentPreview {
        let (catalog, key, name) = REVIEWED;
        let preview = self
            .corpus
            .preview_kb_assignment(&self.tracks[index], catalog, key, name)
            .unwrap();
        let plan = preview.assignment();
        assert_eq!(
            (
                plan.kb_name.as_str(),
                plan.kb_id.as_str(),
                plan.canonical_name.as_str()
            ),
            REVIEWED,
        );
        assert_eq!(plan.trigger, self.tracks[index]);
        assert_eq!(plan.affected_tracks, self.tracks[self.scope(index)]);
        assert_eq!(
            plan.updates_shared_identity,
            self.expected[index].is_some_and(|b| b.corpus_owned)
        );
        assert_eq!(
            plan.previous.as_ref().map(|p| (
                p.identity_id,
                p.kb_name.as_deref(),
                p.kb_id.as_deref(),
                p.canonical_name.as_str()
            )),
            self.expected[index].map(|b| (b.id, Some(b.target.0), Some(b.target.1), b.target.2))
        );
        preview
    }

    fn update_oracle(&mut self, index: usize, id: IdentityId, target: Target) {
        if let Some(previous) = self.expected[index].filter(|b| b.corpus_owned) {
            assert_eq!(id, previous.id);
        } else {
            assert!(self.expected.iter().flatten().all(|b| b.id != id));
        }
        for member in self.scope(index) {
            self.expected[member] = Some(Binding {
                id,
                target,
                corpus_owned: true,
            });
        }
    }

    fn apply(&mut self, index: usize, preview: KbAssignmentPreview, fresh: bool) -> Option<usize> {
        let reviewed = preview.assignment().clone();
        let before_rejection = (!fresh).then(|| self.corpus.identities().clone());
        let result = self.corpus.apply_kb_assignment(preview);
        let size = if fresh {
            let receipt = result.unwrap();
            assert_eq!(receipt.assignment, reviewed);
            self.update_oracle(index, receipt.identity_id, REVIEWED);
            Some(receipt.assignment.affected_tracks.len())
        } else {
            assert!(
                matches!(result, Err(KbAssignmentError::StalePreview)),
                "{result:?}"
            );
            assert_eq!(Some(self.corpus.identities()), before_rejection.as_ref());
            None
        };
        self.check_views(&self.corpus);
        size
    }
}

/// Return the number of tracks in the final applied receipt, or `None` for a stale review.
pub fn replay(bytes: &[u8]) -> Option<usize> {
    replay_observed(bytes, |_, _| {})
}

/// Report checked apply outcomes, including intermediate reviews.
pub fn replay_observed(
    bytes: &[u8],
    mut observe: impl FnMut(usize, Option<usize>),
) -> Option<usize> {
    let mut fixture = Fixture::new();
    let mut selected = 0;
    let mut reviewed = selected;
    let mut pending = fixture.prepare(reviewed);
    let mut fresh = true;

    for &byte in bytes.iter().take(MAX_INPUT_BYTES) {
        let command = if COMMANDS.contains(&byte) || byte.is_ascii_whitespace() {
            byte
        } else {
            COMMANDS[usize::from(byte) % COMMANDS.len()]
        };
        match command {
            b'0'..=b'5' => selected = usize::from(command - b'0'),
            b'p' | b'!' => {
                if command == b'!' {
                    observe(reviewed, fixture.apply(reviewed, pending, fresh));
                }
                reviewed = selected;
                pending = fixture.prepare(reviewed);
                fresh = true;
            }
            b'm' => {
                assert!(fixture
                    .corpus
                    .get_document_mut(&fixture.tracks[selected].doc_id)
                    .is_some());
                fresh = false;
            }
            b'c' => {
                let mut other = fixture.corpus.clone();
                let result = other.apply_kb_assignment(fixture.prepare(selected));
                assert!(matches!(result, Err(KbAssignmentError::StalePreview)));
                assert_eq!(other.identities(), fixture.corpus.identities());
                fixture.check_views(&other);
            }
            b'A' | b'B' | b'N' | b'L' => {
                let target = match command {
                    b'A' => ORIGINAL,
                    b'B' => ("library", "99", "Ada"),
                    b'N' => ("archive", "42", "Ada"),
                    _ => ("library", "42", "Byron"),
                };
                let unchanged = fixture.expected[selected]
                    .is_some_and(|b| b.corpus_owned && b.target == target);
                let (catalog, key, name) = target;
                let id = fixture
                    .corpus
                    .link_track_to_kb(&fixture.tracks[selected], catalog, key, name)
                    .unwrap();
                fixture.update_oracle(selected, id, target);
                fresh &= unchanged;
            }
            _ => {} // 'r' and whitespace leave all bindings observable below.
        }
        fixture.check_views(&fixture.corpus);
    }
    let outcome = fixture.apply(reviewed, pending, fresh);
    observe(reviewed, outcome);
    outcome
}
