//! Reviewed catalog behavior across normal delivery, deliberate replacement, and late retries.
#![allow(clippy::unwrap_used)]
use anno::core::grounded::{
    Corpus, DeliveryError, DeliveryOutcome, DeliveryOwnership, DocumentDelivery, GroundedDocument,
    Identity, Track, TrackRef,
};

fn doc(id: &str, code: &str, tracks: usize) -> Vec<u8> {
    let mut doc = GroundedDocument::new(id, "Reed");
    let identity = doc.add_identity(Identity::from_kb(0, "Reed", "library", code));
    for _ in 0..tracks {
        let track = doc.add_track(Track::new(0, "Reed"));
        doc.link_track_to_identity(track, identity);
    }
    serde_json::to_vec(&doc).unwrap()
}

fn request(token: &str, rev: u64, shared: bool) -> DocumentDelivery {
    DocumentDelivery {
        issuer: "archive".into(),
        delivery_id: token.into(),
        source_revision: rev,
        ownership: if shared {
            DeliveryOwnership::Restore {
                origin: "master".into(),
            }
        } else {
            DeliveryOwnership::Independent
        },
    }
}

fn first(c: &Corpus, id: &str) -> TrackRef {
    let d = c.get_document(id).unwrap();
    let track = d.tracks().map(|t| t.id).min().unwrap();
    d.track_ref(track).unwrap()
}

fn binding<'a>(c: &'a Corpus, r: &TrackRef) -> Option<&'a str> {
    c.get_document(&r.doc_id)
        .unwrap()
        .identity_for_track(r.track_id)
        .and_then(|i| i.kb_id())
}

#[test]
fn full_and_partial_retries_preserve_review_values_scopes_handles_and_original_receipts() {
    for shared in [true, false] {
        for replay in [vec![0], vec![1], vec![1, 0], vec![0, 1]] {
            for after in [true, false] {
                let mut c = Corpus::new();
                let req = [request("a", 1, shared), request("b", 1, shared)];
                let payloads = [doc("A", "17", 1), doc("B", "17", 1)];
                let receipts = [0, 1].map(|n| {
                    c.deliver_document(&req[n], &payloads[n])
                        .unwrap()
                        .receipt()
                        .clone()
                });
                let refs = receipts.each_ref().map(|r| first(&c, &r.document_id));
                let apply = c
                    .preview_kb_assignment(&refs[0], "library", "23", "Reviewed")
                    .unwrap();
                let pending = if after {
                    c.apply_kb_assignment(apply).unwrap();
                    c.preview_kb_assignment(&refs[0], "library", "42", "Further review")
                        .unwrap()
                } else {
                    apply
                };
                for &n in &replay {
                    assert_eq!(
                        c.deliver_document(&req[n], &payloads[n]).unwrap(),
                        DeliveryOutcome::PreviouslyApplied(receipts[n].clone())
                    );
                }
                assert_eq!(binding(&c, &refs[0]), Some(if after { "23" } else { "17" }));
                assert_eq!(
                    binding(&c, &refs[1]),
                    Some(if shared && after { "23" } else { "17" })
                );
                let r = c.apply_kb_assignment(pending).unwrap();
                let expected = if shared { &refs[..] } else { &refs[..1] };
                assert_eq!(r.assignment.affected_tracks, expected);
                for track in expected {
                    assert_eq!(binding(&c, track), Some(if after { "42" } else { "23" }));
                }
            }
        }
    }
}

#[test]
fn issuer_scopes_both_document_keys_and_tokens_and_independent_import_preserves_local_ownership() {
    for order in [[0, 1], [1, 0]] {
        let mut c = Corpus::new();
        let a = request("delivery", 1, false);
        let b = DocumentDelivery {
            issuer: "archive/2".into(),
            ..a.clone()
        };
        let req = [a, b];
        let wire = doc("7", "17", 2);
        for index in order {
            c.deliver_document(&req[index], &wire).unwrap();
        }
        let docs = req.each_ref().map(|r| r.corpus_document_id("7"));
        assert_ne!(docs[0], docs[1]);
        let affected = c
            .preview_kb_assignment(&first(&c, &docs[0]), "library", "23", "Reviewed")
            .unwrap();
        assert_eq!(affected.assignment().affected_tracks.len(), 2);
        assert!(affected
            .assignment()
            .affected_tracks
            .iter()
            .all(|t| t.doc_id == docs[0]));
        c.apply_kb_assignment(affected).unwrap();
        assert_eq!(binding(&c, &first(&c, &docs[1])), Some("17"));
    }
}

#[test]
fn conflicting_tokens_revisions_and_origins_leave_old_reviews_usable() {
    let mut c = Corpus::new();
    let req = request("x", 1, true);
    let wire = doc("A", "17", 1);
    let applied = c.deliver_document(&req, &wire).unwrap();
    let track = first(&c, &applied.receipt().document_id);
    let pending = c
        .preview_kb_assignment(&track, "library", "23", "Reviewed")
        .unwrap();
    let mut whitespace = wire.clone();
    whitespace.push(b' ');
    assert!(matches!(
        c.deliver_document(&req, &whitespace),
        Err(DeliveryError::TokenConflict)
    ));
    assert!(matches!(
        c.deliver_document(&request("x", 1, false), &wire),
        Err(DeliveryError::TokenConflict)
    ));
    assert!(matches!(
        c.deliver_document(&request("late", 1, true), &wire),
        Err(DeliveryError::SourceRevisionConflict)
    ));
    assert!(matches!(
        c.deliver_document(&request("other", 1, true), &doc("B", "55", 1)),
        Err(DeliveryError::OriginConflict)
    ));
    assert!(matches!(
        c.deliver_document(&request("bad", 1, false), b"{"),
        Err(DeliveryError::InvalidPayload(_))
    ));
    assert_eq!(c.documents().count(), 1);
    c.apply_kb_assignment(pending).unwrap();
    let b = c
        .deliver_document(&request("b", 1, true), &doc("B", "17", 1))
        .unwrap();
    assert_eq!(
        binding(&c, &first(&c, &b.receipt().document_id)),
        Some("23")
    );
}

#[test]
fn newer_revision_through_the_secondary_member_requires_a_fresh_inspected_approval() {
    let mut c = Corpus::new();
    let mut tracks = Vec::new();
    for (token, id) in [("a", "A"), ("b", "B")] {
        let r = c
            .deliver_document(&request(token, 1, true), &doc(id, "17", 1))
            .unwrap();
        tracks.push(first(&c, &r.receipt().document_id));
    }
    let correction = c
        .preview_kb_assignment(&tracks[0], "library", "23", "Reviewed")
        .unwrap();
    c.apply_kb_assignment(correction).unwrap();
    let payload = doc("B", "31", 1);
    let req = request("new-B", 2, false);
    let stage = |c: &mut Corpus| match c.deliver_document(&req, &payload) {
        Err(DeliveryError::ReviewRequired(review)) => *review,
        other => panic!("expected review, got {other:?}"),
    };
    let stale = stage(&mut c);
    assert_eq!(stale.summary().affected_tracks.len(), 1);
    let change = &stale.summary().affected_tracks[0];
    assert_eq!(
        change.current.as_ref().unwrap().kb_id.as_deref(),
        Some("23")
    );
    assert_eq!(change.after.as_ref().unwrap().kb_id.as_deref(), Some("31"));
    assert_eq!(change.current_members, tracks);
    assert_eq!(binding(&c, &tracks[1]), Some("23"));
    c.deliver_document(&request("late-C", 1, false), &doc("C", "88", 1))
        .unwrap();
    assert!(matches!(
        c.accept_document_replacement(stale),
        Err(DeliveryError::StaleReview)
    ));
    let active = stage(&mut c);
    let mut copy = c.clone();
    assert!(matches!(
        copy.accept_document_replacement(active),
        Err(DeliveryError::StaleReview)
    ));
    let active = stage(&mut c);
    let accepted = c.accept_document_replacement(active).unwrap();
    assert_eq!(accepted.receipt().applied_revision, 4);
    assert!(binding(&c, &tracks[1]).is_none());
    assert_eq!(binding(&c, &first(&c, &tracks[1].doc_id)), Some("31"));
    assert_eq!(binding(&c, &tracks[0]), Some("23"));
    assert_eq!(
        c.deliver_document(&req, &payload).unwrap().receipt(),
        accepted.receipt()
    );
}

#[test]
fn raw_replacement_claims_its_key_and_old_delivery_receipts_survive() {
    let mut c = Corpus::new();
    let req = request("first", 1, false);
    let payload = doc("A", "17", 1);
    let original = c.deliver_document(&req, &payload).unwrap();
    let key = &original.receipt().document_id;
    c.add_document(serde_json::from_slice(&doc(key, "31", 1)).unwrap());
    assert!(matches!(
        c.deliver_document(&request("new", 2, false), &payload),
        Err(DeliveryError::UnmanagedDocument)
    ));
    assert_eq!(
        c.deliver_document(&req, &payload).unwrap().receipt(),
        original.receipt()
    );
    assert_eq!(binding(&c, &first(&c, key)), Some("31"));
}

#[test]
fn a_late_restored_member_inherits_protection_and_a_new_join_requires_review() {
    let mut c = Corpus::new();
    let old = c
        .deliver_document(&request("a", 1, true), &doc("A", "17", 1))
        .unwrap();
    let a = first(&c, &old.receipt().document_id);
    let p = c
        .preview_kb_assignment(&a, "library", "23", "Reed")
        .unwrap();
    c.apply_kb_assignment(p).unwrap();
    c.deliver_document(&request("b", 1, true), &doc("B", "17", 1))
        .unwrap();
    assert!(matches!(
        c.deliver_document(&request("b2", 2, false), &doc("B", "17", 1)),
        Err(DeliveryError::ReviewRequired(_))
    ));

    let other = c
        .deliver_document(&request("c", 1, false), &doc("C", "17", 1))
        .unwrap();
    let owner = first(&c, &other.receipt().document_id);
    let p = c
        .preview_kb_assignment(&owner, "library", "17", "Reed")
        .unwrap();
    c.apply_kb_assignment(p).unwrap();
    let result = c.deliver_document(&request("c2", 2, true), &doc("C", "17", 1));
    let Err(DeliveryError::ReviewRequired(review)) = result else {
        panic!("expected widened ownership review")
    };
    assert_eq!(
        review.summary().affected_tracks[0].proposed_members.len(),
        3
    );
}

#[test]
fn native_shared_snapshot_restores_members_and_only_complete_similarity_evidence() {
    use anno::coalesce::Resolver;
    use anno::core::grounded::Signal;
    let mut source = Corpus::new();
    for id in ["A", "B"] {
        let mut d = GroundedDocument::new(id, "A. Reed");
        let s = d.add_signal(Signal::from_text("A. Reed", "A. Reed", "PER", 1.).unwrap());
        d.create_track_from_signals("A. Reed", &[s]).unwrap();
        source.add_document(d);
    }
    Resolver::new().resolve_inter_doc_coref(&mut source, None, None);
    let original = source.current_coref_clusters();
    assert_eq!(original[0].1.len(), 2);
    let minimum = source
        .get_identity(original[0].0)
        .unwrap()
        .minimum_pair_similarity
        .unwrap();
    for order in [["A", "B"], ["B", "A"]] {
        let mut restored = Corpus::new();
        for (position, id) in order.into_iter().enumerate() {
            let wire = serde_json::to_vec(source.get_document(id).unwrap()).unwrap();
            restored
                .deliver_document(&request(id, 1, true), &wire)
                .unwrap();
            let groups = restored.current_coref_clusters();
            assert_eq!(groups.len(), 1);
            assert_eq!(groups[0].1.len(), position + 1);
            let identity = restored.get_identity(groups[0].0).unwrap();
            assert_eq!(
                identity.minimum_pair_similarity,
                (position == 1).then_some(minimum)
            );
            for member in &groups[0].1 {
                assert!(restored
                    .get_document(&member.doc_id)
                    .unwrap()
                    .get_track(member.track_id)
                    .is_some());
            }
        }
    }
}
