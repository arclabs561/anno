//! Every sequence of up to four review, edit and apply actions on one shared identity.
use anno::coalesce::Resolver;
use anno::core::grounded::{Corpus, GroundedDocument, KbAssignmentError, Track, TrackRef};

#[derive(Clone, Copy, Debug)]
enum Action {
    Read,
    Assign(&'static str),
    BorrowMutably,
    RetirePeer,
    ApplyOnClone,
    Renew,
    ApplyCompetingPreview,
}

const ACTIONS: [Action; 8] = [
    Action::Read,
    Action::Assign("42"),
    Action::Assign("99"),
    Action::BorrowMutably,
    Action::RetirePeer,
    Action::ApplyOnClone,
    Action::Renew,
    Action::ApplyCompetingPreview,
];

fn add_member(corpus: &mut Corpus, name: &str) -> TrackRef {
    let mut doc = GroundedDocument::new(name, "Mira");
    let id = doc.add_track(Track::new(0, "Mira"));
    let member = doc.track_ref(id).unwrap();
    corpus.add_document(doc);
    member
}

// The oracle records public operations and expected effects independently of anno's revision token.
#[derive(Clone, Copy, PartialEq, Eq)]
struct View {
    key: &'static str,
    has_peer: bool,
}

impl View {
    fn check(self, corpus: &Corpus, trigger: &TrackRef, peer: &TrackRef, trace: &[Action]) {
        for (track, expected) in [
            (trigger, Some(self.key)),
            (peer, self.has_peer.then_some(self.key)),
        ] {
            let observed = corpus
                .get_document(&track.doc_id)
                .unwrap()
                .identity_for_track(track.track_id)
                .and_then(|identity| identity.kb_id());
            assert_eq!(observed, expected, "track {track:?} after {trace:?}");
        }
    }
}

// Whether approving solely on equality of the visible key and membership would accept a stale write.
fn check_sequence(trace: &[Action]) -> bool {
    let mut corpus = Corpus::new();
    let trigger = add_member(&mut corpus, "a");
    let peer = add_member(&mut corpus, "b");
    let shared = Resolver::new().resolve_inter_doc_coref(&mut corpus, None, None)[0];
    corpus
        .link_track_to_kb(&trigger, "kb", "42", "Mira")
        .unwrap();
    let prepare = |corpus: &Corpus, key: &str| {
        corpus
            .preview_kb_assignment(&trigger, "kb", key, "Mira")
            .unwrap_or_else(|error| panic!("prepare after {trace:?}: {error}"))
    };
    let mut pending = prepare(&corpus, "7");
    let mut view = View {
        key: "42",
        has_peer: true,
    };
    let mut reviewed = view;
    let mut current = true;

    for (i, action) in trace.iter().enumerate() {
        let prefix = &trace[..=i];
        match *action {
            Action::Read => assert!(corpus.get_identity(shared).is_some(), "{prefix:?}"),
            Action::Assign(key) => {
                corpus
                    .link_track_to_kb(&trigger, "kb", key, "Mira")
                    .unwrap_or_else(|error| panic!("assign after {prefix:?}: {error}"));
                if key != view.key {
                    current = false;
                }
                view.key = key;
            }
            Action::BorrowMutably => {
                assert!(corpus.get_document_mut("a").is_some(), "{prefix:?}");
                current = false;
            }
            Action::RetirePeer => {
                corpus.add_document(GroundedDocument::new("b", ""));
                view.has_peer = false;
                current = false;
            }
            Action::ApplyOnClone => {
                let mut other = corpus.clone();
                let result = other.apply_kb_assignment(prepare(&corpus, "8"));
                assert!(
                    matches!(result, Err(KbAssignmentError::StalePreview)),
                    "{prefix:?}: {result:?}"
                );
                view.check(&other, &trigger, &peer, prefix);
            }
            Action::Renew => {
                pending = prepare(&corpus, "7");
                reviewed = view;
                current = true;
            }
            Action::ApplyCompetingPreview => {
                let competing = prepare(&corpus, view.key);
                let receipt = corpus
                    .apply_kb_assignment(competing)
                    .unwrap_or_else(|error| {
                        panic!("apply competing preview after {prefix:?}: {error}")
                    });
                assert_eq!(receipt.identity_id, shared, "{prefix:?}");
                current = false;
            }
        }
        view.check(&corpus, &trigger, &peer, prefix);
    }

    let before = corpus.get_identity(shared).unwrap().clone();
    let result = corpus.apply_kb_assignment(pending);
    let snapshot_would_wrongly_accept = reviewed == view && result.is_err();
    if current {
        let receipt =
            result.unwrap_or_else(|error| panic!("current review after {trace:?}: {error}"));
        assert_eq!(receipt.identity_id, shared, "{trace:?}");
        assert_eq!(receipt.assignment.trigger, trigger, "{trace:?}");
        assert_eq!(
            receipt.assignment.previous.unwrap().kb_id.as_deref(),
            Some(view.key),
            "{trace:?}"
        );
        assert_eq!(receipt.assignment.kb_id, "7", "{trace:?}");
        let mut members = vec![trigger.clone()];
        if view.has_peer {
            members.push(peer.clone());
        }
        assert_eq!(receipt.assignment.affected_tracks, members, "{trace:?}");
        view.key = "7";
    } else {
        assert!(
            matches!(result, Err(KbAssignmentError::StalePreview)),
            "stale review after {trace:?}: {result:?}"
        );
        assert_eq!(corpus.get_identity(shared), Some(&before), "{trace:?}");
    }
    view.check(&corpus, &trigger, &peer, trace);
    snapshot_would_wrongly_accept
}

#[test]
fn preview_protocol_agrees_with_the_oracle_for_all_short_sequences() {
    fn visit(trace: &mut Vec<Action>) -> usize {
        check_sequence(trace);
        let mut count = 1;
        if trace.len() < 4 {
            for action in ACTIONS {
                trace.push(action);
                count += visit(trace);
                trace.pop();
            }
        }
        count
    }
    assert_eq!(visit(&mut Vec::new()), 4_681);
}

#[test]
fn value_only_approval_would_accept_a_stale_round_trip() {
    assert!(check_sequence(&[
        Action::Assign("99"),
        Action::Assign("42")
    ]));
}
