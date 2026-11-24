//! Public-API state model used by both stable Cargo and libFuzzer.
//! All documents, source mentions, catalog records and operations are synthetic.
use anno::core::grounded::{
    Corpus, DeliveryError, DeliveryOutcome, DeliveryOwnership, DeliveryReceipt, DocumentDelivery,
    GroundedDocument, Identity, KbAssignmentError, KbAssignmentPreview, Signal, TrackRef,
};
use sha2::{Digest, Sha256};

const ORIGINAL: &str = "17";
const REVIEWED: &str = "23";
const SECOND: &str = "42";
const COMMANDS: &[u8] = b"012345aAcrRxsobvp!";
const MAX_BYTES: usize = 48;

struct Source {
    issuer: &'static str,
    local_id: &'static str,
    origin: Option<&'static str>,
    group: usize,
}

// The source model declares the owner of the single mention in each document.
// Every wire record has the same numeric identity ID, catalog key and name.
const SOURCES: [Source; 6] = [
    Source {
        issuer: "north",
        local_id: "entry",
        origin: Some("vault"),
        group: 0,
    },
    Source {
        issuer: "north",
        local_id: "peer",
        origin: Some("vault"),
        group: 0,
    },
    Source {
        issuer: "north",
        local_id: "fork",
        origin: None,
        group: 2,
    },
    Source {
        issuer: "south",
        local_id: "entry",
        origin: Some("vault"),
        group: 3,
    },
    Source {
        issuer: "north",
        local_id: "annex",
        origin: Some("copy"),
        group: 4,
    },
    Source {
        issuer: "south",
        local_id: "fork",
        origin: None,
        group: 5,
    },
];

#[derive(Clone, Debug, PartialEq, Eq)]
struct Intent {
    slot: usize,
    token: String,
    revision: u64,
    edited_text: bool,
    extra_space: bool,
    change_ownership: bool,
}

impl Intent {
    fn new(slot: usize, revision: u64) -> Self {
        Self {
            slot,
            token: format!("{}-{revision}", SOURCES[slot].local_id),
            revision,
            edited_text: revision.is_multiple_of(2),
            extra_space: false,
            change_ownership: false,
        }
    }

    fn request(&self) -> DocumentDelivery {
        let source = &SOURCES[self.slot];
        let origin = if self.change_ownership {
            source.origin.is_none().then_some("vault")
        } else {
            source.origin
        };
        DocumentDelivery {
            issuer: source.issuer.into(),
            delivery_id: self.token.clone(),
            source_revision: self.revision,
            ownership: origin.map_or(DeliveryOwnership::Independent, |origin| {
                DeliveryOwnership::Restore {
                    origin: origin.into(),
                }
            }),
        }
    }

    fn text(&self) -> &'static str {
        if self.edited_text {
            "Ada amended."
        } else {
            "Ada."
        }
    }

    fn bytes(&self) -> Vec<u8> {
        let mut doc = GroundedDocument::new(SOURCES[self.slot].local_id, self.text());
        let signal = doc.add_signal(Signal::from_text(self.text(), "Ada", "PER", 1.0).unwrap());
        let track = doc.create_track_from_signals("Ada", &[signal]).unwrap();
        let owner = doc.add_identity(Identity::from_kb(0, "Ada", "catalog", ORIGINAL));
        doc.link_track_to_identity(track, owner);
        let mut bytes = serde_json::to_vec(&doc).unwrap();
        if self.extra_space {
            bytes.push(b' ');
        }
        bytes
    }
}

struct Head {
    intent: Intent,
    handle: TrackRef,
}

struct Pending {
    preview: KbAssignmentPreview,
    slot: usize,
    target: &'static str,
    fresh: bool,
}

struct History {
    corpus: Corpus,
    // The expected keys and owner groups never come from a corpus lookup.
    keys: [&'static str; SOURCES.len()],
    heads: [Option<Head>; SOURCES.len()],
    accepted: Vec<(Intent, DeliveryReceipt)>,
    pending: Option<Pending>,
    outcomes: String,
}

impl History {
    fn new() -> Self {
        Self {
            corpus: Corpus::new(),
            keys: [ORIGINAL; SOURCES.len()],
            heads: std::array::from_fn(|_| None),
            accepted: Vec::new(),
            pending: None,
            outcomes: String::new(),
        }
    }

    fn scope(&self, slot: usize) -> Vec<TrackRef> {
        let mut members: Vec<_> = self
            .heads
            .iter()
            .enumerate()
            .filter_map(|(i, head)| {
                head.as_ref()
                    .filter(|_| SOURCES[i].group == SOURCES[slot].group)
                    .map(|h| h.handle.clone())
            })
            .collect();
        members.sort_by(|a, b| a.doc_id.cmp(&b.doc_id));
        members
    }

    fn check_readers(&self) {
        assert_eq!(
            self.corpus.documents().count(),
            self.heads.iter().flatten().count()
        );
        for (slot, head) in self.heads.iter().enumerate() {
            let Some(head) = head else { continue };
            let doc = self.corpus.get_document(&head.handle.doc_id).unwrap();
            assert_eq!(doc.text(), head.intent.text());
            assert_eq!(doc.tracks().count(), 1);
            let [mention] = doc.signals() else {
                panic!("expected one source mention")
            };
            assert_eq!(mention.surface(), "Ada");
            assert_eq!(
                doc.track_for_signal(mention.id).unwrap().id,
                head.handle.track_id
            );
            assert_eq!(
                doc.identity_for_signal(mention.id)
                    .unwrap()
                    .qualified_kb_ref(),
                Some(("catalog", self.keys[SOURCES[slot].group]))
            );
            let preview = self
                .corpus
                .preview_kb_assignment(&head.handle, "catalog", "probe", "Ada")
                .unwrap();
            assert_eq!(preview.assignment().affected_tracks, self.scope(slot));
        }
    }

    fn invalidate(&mut self) {
        if let Some(pending) = self.pending.as_mut() {
            pending.fresh = false;
        }
    }

    fn deliver(&mut self, intent: Intent, authorize: bool) {
        let source = &SOURCES[intent.slot];
        let request = intent.request();
        let bytes = intent.bytes();
        let registry = self.corpus.identities().clone();
        let result = self.corpus.deliver_document(&request, &bytes);
        if let Some((original, receipt)) = self
            .accepted
            .iter()
            .find(|(i, _)| SOURCES[i.slot].issuer == source.issuer && i.token == intent.token)
        {
            if original == &intent {
                let Ok(DeliveryOutcome::PreviouslyApplied(previous)) = result else {
                    panic!("historical receipt missing for {intent:?}: {result:?}");
                };
                assert_eq!(&previous, receipt);
                self.outcomes.push('r');
            } else {
                self.outcomes.push('x');
                assert!(
                    matches!(result, Err(DeliveryError::TokenConflict)),
                    "{intent:?}: {result:?}"
                );
            }
            assert_eq!(self.corpus.identities(), &registry);
            return;
        }
        if self.heads[intent.slot]
            .as_ref()
            .is_some_and(|h| h.intent.revision >= intent.revision)
        {
            assert_eq!(self.corpus.identities(), &registry);
            self.outcomes.push('s');
            assert!(
                matches!(result, Err(DeliveryError::SourceRevisionConflict)),
                "{intent:?}: {result:?}"
            );
            return;
        }
        // The incoming catalog still asserts the original accession. A source
        // replacing a live reviewed member must present that disagreement.
        let edited_review =
            self.heads[intent.slot].is_some() && self.keys[source.group] != ORIGINAL;
        let result = if edited_review {
            let Err(DeliveryError::ReviewRequired(review)) = result else {
                panic!("missing review for {intent:?}: {result:?}");
            };
            let summary = review.summary();
            let previous = self.heads[intent.slot].as_ref().unwrap();
            assert_eq!(summary.request, request);
            assert_eq!(summary.document_id, previous.handle.doc_id);
            assert_eq!(summary.previous_source_revision, previous.intent.revision);
            let [dispute] = summary.affected_tracks.as_slice() else {
                panic!("expected one disputed track")
            };
            assert_eq!(dispute.track, previous.handle);
            assert_eq!(dispute.current_members, self.scope(intent.slot));
            assert_eq!(
                dispute.current.as_ref().unwrap().kb_id.as_deref(),
                Some(self.keys[source.group])
            );
            assert_eq!(
                dispute.incoming.as_ref().unwrap().kb_id.as_deref(),
                Some(ORIGINAL)
            );
            let effective = if source.origin.is_some() {
                &dispute.current
            } else {
                &dispute.incoming
            };
            assert_eq!(&dispute.after, effective);
            assert_eq!(self.corpus.identities(), &registry);
            self.outcomes.push('q');
            // Inspect visible state even if the caller accepts immediately.
            self.check_readers();
            if !authorize {
                return;
            }
            self.corpus.accept_document_replacement(*review)
        } else {
            result
        };
        let Ok(DeliveryOutcome::Applied(receipt)) = result else {
            panic!("expected a new acceptance for {intent:?}: {result:?}");
        };
        assert_eq!(receipt.applied_revision, self.accepted.len() as u64 + 1);
        assert_eq!(receipt.request, request);
        assert_eq!(receipt.source_document_id, source.local_id);
        assert_eq!(
            receipt.payload_sha256,
            Sha256::digest(&bytes)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        );
        let doc = self.corpus.get_document(&receipt.document_id).unwrap();
        let track = doc.tracks().next().unwrap();
        let handle = doc.track_ref(track.id).unwrap();
        if let Some(previous) = &self.heads[intent.slot] {
            assert_eq!(handle.doc_id, previous.handle.doc_id);
            assert!(doc.get_track(previous.handle.track_id).is_none());
        }
        if source.origin.is_none() {
            self.keys[source.group] = ORIGINAL;
        }
        self.heads[intent.slot] = Some(Head {
            intent: intent.clone(),
            handle,
        });
        self.accepted.push((intent, receipt));
        self.outcomes.push('a');
        self.invalidate();
    }

    fn prepare(&mut self, slot: usize) {
        let Some(head) = &self.heads[slot] else {
            return;
        };
        let target = if self.keys[SOURCES[slot].group] == REVIEWED {
            SECOND
        } else {
            REVIEWED
        };
        let preview = self
            .corpus
            .preview_kb_assignment(&head.handle, "catalog", target, "Ada")
            .unwrap();
        assert_eq!(preview.assignment().affected_tracks, self.scope(slot));
        self.outcomes.push('p');
        self.pending = Some(Pending {
            preview,
            slot,
            target,
            fresh: true,
        });
    }

    fn apply(&mut self) {
        let Some(pending) = self.pending.take() else {
            return;
        };
        let result = self.corpus.apply_kb_assignment(pending.preview);
        if pending.fresh {
            self.outcomes.push('!');
            let receipt = result.unwrap();
            assert_eq!(receipt.assignment.affected_tracks, self.scope(pending.slot));
            self.keys[SOURCES[pending.slot].group] = pending.target;
        } else {
            self.outcomes.push('~');
            assert!(
                matches!(result, Err(KbAssignmentError::StalePreview)),
                "{result:?}"
            );
        }
    }

    fn operation(&mut self, slot: usize, op: u8) {
        let head = self.heads[slot].as_ref();
        let revision = head.map_or(0, |head| head.intent.revision);
        let first = self
            .accepted
            .iter()
            .find(|(i, _)| i.slot == slot)
            .map(|(i, _)| i);
        let mut intent = match op {
            b'p' => {
                self.prepare(slot);
                return;
            }
            b'!' => {
                self.apply();
                return;
            }
            b'a' | b'A' | b'c' => Intent::new(slot, revision + 1),
            b's' | b'b' if head.is_some() => {
                let old = if op == b'b' {
                    revision.saturating_sub(1)
                } else {
                    revision
                };
                let mut stale = Intent::new(slot, old);
                stale.token = format!("stale-{slot}-{old}");
                stale
            }
            b'R' => head.map_or_else(|| Intent::new(slot, 1), |h| h.intent.clone()),
            b'r' | b'x' | b'o' | b'v' => first.cloned().unwrap_or_else(|| Intent::new(slot, 1)),
            _ => return,
        };
        match op {
            b'c' => intent.token = "one-token".into(),
            b'x' if first.is_some() => intent.extra_space = true,
            b'o' if first.is_some() => intent.change_ownership = true,
            b'v' if first.is_some() => intent.revision += 1,
            _ => {}
        }
        self.deliver(intent, op == b'A');
    }
}

/// Return checked operation outcomes and final bindings in source-slot order.
pub fn replay(bytes: &[u8]) -> (String, [Option<&'static str>; SOURCES.len()]) {
    let mut history = History::new();
    let mut slot = 0;
    for &byte in bytes
        .iter()
        .take(MAX_BYTES)
        .filter(|b| !b.is_ascii_whitespace())
    {
        let op = if COMMANDS.contains(&byte) {
            byte
        } else {
            COMMANDS[usize::from(byte) % COMMANDS.len()]
        };
        match op {
            b'0'..=b'5' => slot = usize::from(op - b'0'),
            _ => {
                history.operation(slot, op);
                history.check_readers();
            }
        }
    }
    (
        history.outcomes,
        std::array::from_fn(|i| {
            history.heads[i]
                .as_ref()
                .map(|_| history.keys[SOURCES[i].group])
        }),
    )
}
