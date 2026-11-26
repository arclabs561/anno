use anno::coalesce::Resolver;
use anno::core::grounded::{
    Corpus, GroundedDocument, Identity, IdentityId, KbAssignmentError, Track, TrackRef,
};

fn add_track(document: &mut GroundedDocument) -> TrackRef {
    let id = document.add_track(Track::new(0, "Ada"));
    document.track_ref(id).unwrap()
}

fn add_document(corpus: &mut Corpus, id: &str) -> TrackRef {
    let mut document = GroundedDocument::new(id, "Ada");
    let track = add_track(&mut document);
    corpus.add_document(document);
    track
}

fn linked<'a>(corpus: &'a Corpus, track: &TrackRef) -> &'a Identity {
    corpus
        .get_document(&track.doc_id)
        .unwrap()
        .identity_for_track(track.track_id)
        .unwrap()
}

#[test]
fn shared_reassignment_reports_both_documents_and_the_prior_target() {
    let mut corpus = Corpus::new();
    let last = add_document(&mut corpus, "z");
    let first = add_document(&mut corpus, "a");
    let shared = Resolver::new().resolve_inter_doc_coref(&mut corpus, None, None)[0];
    corpus.link_track_to_kb(&last, "old", "42", "Ada").unwrap();
    let preview = corpus
        .preview_kb_assignment(&last, "new", "42", "Ada Byron")
        .unwrap();
    let competing = corpus
        .preview_kb_assignment(&first, "other", "7", "Bo")
        .unwrap();
    let assignment = preview.assignment();
    assert!(assignment.updates_shared_identity);
    assert_eq!(assignment.trigger, last);
    assert_eq!(assignment.affected_tracks, [first.clone(), last.clone()]);
    let prior = assignment.previous.as_ref().unwrap();
    assert_eq!(prior.identity_id, shared);
    assert_eq!(prior.kb_name.as_deref(), Some("old"));
    assert_eq!(prior.kb_id.as_deref(), Some("42"));
    assert_eq!(prior.canonical_name, "Ada");
    assert_eq!(
        linked(&corpus, &first).qualified_kb_ref(),
        Some(("old", "42"))
    );

    let receipt = corpus.apply_kb_assignment(preview).unwrap();
    assert!(matches!(
        corpus.apply_kb_assignment(competing),
        Err(KbAssignmentError::StalePreview)
    ));
    assert_eq!(receipt.identity_id, shared);
    assert_eq!(
        receipt.assignment.affected_tracks,
        [first.clone(), last.clone()]
    );
    assert_eq!(receipt.assignment.kb_name, "new");
    assert_eq!(receipt.assignment.kb_id, "42");
    assert_eq!(receipt.assignment.canonical_name, "Ada Byron");
    for track in [&first, &last] {
        let identity = linked(&corpus, track);
        assert_eq!(identity.id, shared);
        assert_eq!(identity.qualified_kb_ref(), Some(("new", "42")));
        assert_eq!(identity.canonical_name, "Ada Byron");
    }
}

#[test]
fn replacing_a_local_link_keeps_its_sibling_on_the_local_identity() {
    let mut document = GroundedDocument::new("local", "Ada");
    let first = add_track(&mut document);
    let sibling = add_track(&mut document);
    let local = document.add_identity(Identity::from_kb(0, "Local Ada", "old", "42"));
    document.link_track_to_identity(first.track_id, local);
    document.link_track_to_identity(sibling.track_id, local);
    let mut corpus = Corpus::new();
    corpus.add_document(document);

    let preview = corpus
        .preview_kb_assignment(&first, "new", "42", "Ada")
        .unwrap();
    assert!(!preview.assignment().updates_shared_identity);
    assert_eq!(
        preview.assignment().affected_tracks,
        std::slice::from_ref(&first)
    );
    assert_eq!(
        preview.assignment().previous.as_ref().unwrap().identity_id,
        local
    );
    let receipt = corpus.apply_kb_assignment(preview).unwrap();
    assert_ne!(receipt.identity_id, local);
    assert_eq!(linked(&corpus, &first).id, receipt.identity_id);
    assert_eq!(linked(&corpus, &sibling).id, local);
    assert_eq!(
        linked(&corpus, &sibling).qualified_kb_ref(),
        Some(("old", "42"))
    );
}

#[test]
fn label_change_on_a_kb_only_identity_reaches_all_live_tracks() {
    let mut corpus = Corpus::new();
    let first = add_document(&mut corpus, "book");
    let initial = corpus
        .preview_kb_assignment(&first, "kb", "42", "Ada")
        .unwrap();
    assert!(initial.assignment().previous.is_none());
    assert!(!initial.assignment().updates_shared_identity);
    assert_eq!(
        initial.assignment().affected_tracks,
        std::slice::from_ref(&first)
    );
    let id = corpus.apply_kb_assignment(initial).unwrap().identity_id;
    let document = corpus.get_document_mut("book").unwrap();
    let second = add_track(document);
    document.link_track_to_identity(second.track_id, id);
    assert!(corpus.current_coref_clusters().is_empty());

    let preview = corpus
        .preview_kb_assignment(&second, "kb", "42", "Byron")
        .unwrap();
    assert!(preview.assignment().updates_shared_identity);
    assert_eq!(
        preview.assignment().affected_tracks,
        [first.clone(), second.clone()]
    );
    assert_eq!(corpus.apply_kb_assignment(preview).unwrap().identity_id, id);
    assert_eq!(linked(&corpus, &first).canonical_name, "Byron");
    assert_eq!(linked(&corpus, &second).canonical_name, "Byron");
}

#[test]
fn a_rejected_merge_and_an_unchanged_direct_link_preserve_review() {
    let mut corpus = Corpus::new();
    let track = add_document(&mut corpus, "letter");
    let id = corpus.link_track_to_kb(&track, "kb", "42", "Ada").unwrap();
    let preview = corpus
        .preview_kb_assignment(&track, "kb", "42", "Ada")
        .unwrap();
    let competing = corpus
        .preview_kb_assignment(&track, "kb", "99", "Bo")
        .unwrap();

    assert!(corpus.get_document_mut("missing").is_none());
    assert!(corpus
        .merge_tracks("letter", &[track.track_id, track.track_id])
        .is_none());
    assert_eq!(
        corpus.link_track_to_kb(&track, "kb", "42", "Ada").unwrap(),
        id
    );

    // Accepting a reviewed assignment consumes the approval even when the key stays the same.
    assert_eq!(corpus.apply_kb_assignment(preview).unwrap().identity_id, id);
    assert!(matches!(
        corpus.apply_kb_assignment(competing),
        Err(KbAssignmentError::StalePreview)
    ));
}

#[test]
fn returning_to_the_same_kb_target_does_not_revive_a_preview() {
    let mut corpus = Corpus::new();
    let track = add_document(&mut corpus, "letter");
    let id = corpus.link_track_to_kb(&track, "kb", "A", "Ada").unwrap();
    let preview = corpus
        .preview_kb_assignment(&track, "kb", "C", "Cy")
        .unwrap();
    corpus.link_track_to_kb(&track, "kb", "B", "Bo").unwrap();
    corpus.link_track_to_kb(&track, "kb", "A", "Ada").unwrap();

    assert!(matches!(
        corpus.apply_kb_assignment(preview),
        Err(KbAssignmentError::StalePreview)
    ));
    assert_eq!(corpus.identities().len(), 1);
    assert_eq!(linked(&corpus, &track).id, id);
    assert_eq!(
        linked(&corpus, &track).qualified_kb_ref(),
        Some(("kb", "A"))
    );
}

#[test]
fn matching_track_ids_in_a_clone_or_another_corpus_do_not_authorize_a_write() {
    let mut original = Corpus::new();
    let track = add_document(&mut original, "book");
    let valid_preview = original
        .preview_kb_assignment(&track, "kb", "42", "Ada")
        .unwrap();
    let mut unrelated = Corpus::new();
    assert_eq!(add_document(&mut unrelated, "book"), track);
    for mut other in [original.clone(), unrelated] {
        let own_preview = other
            .preview_kb_assignment(&track, "kb", "99", "Bo")
            .unwrap();
        let preview = original
            .preview_kb_assignment(&track, "kb", "42", "Ada")
            .unwrap();
        assert!(matches!(
            other.apply_kb_assignment(preview),
            Err(KbAssignmentError::StalePreview)
        ));
        assert!(other.identities().is_empty());
        other.apply_kb_assignment(own_preview).unwrap();
    }
    original.apply_kb_assignment(valid_preview).unwrap();
    assert_eq!(
        linked(&original, &track).qualified_kb_ref(),
        Some(("kb", "42"))
    );
}

#[test]
fn handing_out_a_mutable_document_retires_its_outstanding_preview() {
    let mut corpus = Corpus::new();
    let track = add_document(&mut corpus, "book");
    let preview = corpus
        .preview_kb_assignment(&track, "kb", "42", "Ada")
        .unwrap();
    assert!(corpus.get_document_mut("book").is_some());
    let error = corpus
        .apply_kb_assignment(preview)
        .expect_err("mutable access must retire the preview");
    assert!(matches!(error, KbAssignmentError::StalePreview));
    assert!(corpus.identities().is_empty());
}

#[test]
fn losing_a_peer_document_retires_the_preview_even_when_its_trigger_survives() {
    let mut corpus = Corpus::new();
    let trigger = add_document(&mut corpus, "kept");
    add_document(&mut corpus, "removed");
    Resolver::new().resolve_inter_doc_coref(&mut corpus, None, None);
    let proposed = corpus
        .preview_kb_assignment(&trigger, "kb", "42", "Ada")
        .unwrap();
    assert_eq!(proposed.assignment().affected_tracks.len(), 2);

    corpus.add_document(GroundedDocument::new("removed", ""));
    assert!(matches!(
        corpus.apply_kb_assignment(proposed),
        Err(KbAssignmentError::StalePreview)
    ));
    assert!(linked(&corpus, &trigger).qualified_kb_ref().is_none());
    let current = corpus
        .preview_kb_assignment(&trigger, "kb", "42", "Ada")
        .unwrap();
    assert_eq!(
        current.assignment().affected_tracks,
        std::slice::from_ref(&trigger)
    );
    let abandoned = corpus
        .preview_kb_assignment(&trigger, "other", "8", "Jo")
        .unwrap();
    drop(abandoned);
    assert_eq!(
        corpus.current_coref_clusters()[0].1,
        std::slice::from_ref(&trigger)
    );
    corpus.apply_kb_assignment(current).unwrap();
    assert_eq!(
        linked(&corpus, &trigger).qualified_kb_ref(),
        Some(("kb", "42"))
    );
}

#[test]
fn rejected_previews_preserve_existing_links() {
    let mut corpus = Corpus::new();
    let track = add_document(&mut corpus, "book");
    let id = corpus.link_track_to_kb(&track, "kb", "42", "Ada").unwrap();
    for (catalog, key) in [("  ", "42"), (" UnKnOwN ", "42"), ("kb", "  ")] {
        assert!(matches!(
            corpus.preview_kb_assignment(&track, catalog, key, "Bo"),
            Err(KbAssignmentError::InvalidCatalogKey)
        ));
    }
    let missing_document = TrackRef {
        doc_id: "missing".into(),
        ..track.clone()
    };
    let missing_track = TrackRef {
        track_id: 99.into(),
        ..track.clone()
    };
    for missing in [missing_document, missing_track] {
        assert!(corpus
            .preview_kb_assignment(&missing, "kb", "B", "Bo")
            .is_err());
    }
    assert_eq!(corpus.identities().len(), 1);
    assert_eq!(linked(&corpus, &track).id, id);
    assert_eq!(linked(&corpus, &track).canonical_name, "Ada");
}

#[test]
fn a_local_identity_matching_the_proposed_shared_record_cannot_join_during_apply() {
    let mut corpus = Corpus::new();
    let a = add_document(&mut corpus, "a");
    let b = add_document(&mut corpus, "b");
    let id = Resolver::new().resolve_inter_doc_coref(&mut corpus, None, None)[0];
    corpus.link_track_to_kb(&a, "kb", "old", "Ada").unwrap();

    // A separately saved corpus already has the proposed catalog assignment.
    let mut other = corpus.clone();
    other.link_track_to_kb(&a, "kb", "42", "Ada Byron").unwrap();
    let future_record = other.get_identity(id).unwrap().clone();
    let c = add_document(&mut corpus, "c");
    let document = corpus.get_document_mut("c").unwrap();
    let local = document.add_identity(future_record.clone());
    assert_eq!(local, id);
    document.link_track_to_identity(c.track_id, local);
    assert!(document.invariants_hold());
    assert!(!corpus.is_shared_identity(linked(&corpus, &c)));

    let initial_shared = corpus.get_identity(id).unwrap().clone();
    let initial_members = vec![(id, vec![a.clone(), b.clone()])];
    assert_eq!(corpus.current_coref_clusters(), initial_members);
    assert!(matches!(
        corpus.preview_kb_assignment(&a, "kb", "42", "Ada Byron"),
        Err(KbAssignmentError::AmbiguousLocalIdentity)
    ));
    assert_eq!(corpus.get_identity(id), Some(&initial_shared));
    assert_eq!(linked(&corpus, &a), &initial_shared);
    assert_eq!(linked(&corpus, &b), &initial_shared);
    assert_eq!(linked(&corpus, &c), &future_record);
    assert_eq!(corpus.current_coref_clusters(), initial_members);

    // Numeric ID collision alone does not prevent a reviewed write.
    let safe = corpus
        .preview_kb_assignment(&a, "kb", "other", "Ada King")
        .unwrap();
    assert_eq!(safe.assignment().affected_tracks, [a.clone(), b.clone()]);
    assert_eq!(corpus.apply_kb_assignment(safe).unwrap().identity_id, id);
    for track in [&a, &b] {
        assert_eq!(
            linked(&corpus, track).qualified_kb_ref(),
            Some(("kb", "other"))
        );
    }
    assert_eq!(linked(&corpus, &c), &future_record);
    assert!(!corpus.is_shared_identity(linked(&corpus, &c)));
    assert_eq!(corpus.current_coref_clusters(), initial_members);
}

#[test]
fn every_track_outside_the_reviewed_scope_keeps_its_catalog_binding() {
    type Binding<'a> = (IdentityId, Option<(&'a str, &'a str)>, &'a str);

    fn binding<'a>(corpus: &'a Corpus, track: &TrackRef) -> Option<Binding<'a>> {
        let document = corpus.get_document(&track.doc_id).unwrap();
        assert!(document.get_track(track.track_id).is_some());
        document.identity_for_track(track.track_id).map(|identity| {
            (
                identity.id(),
                identity.qualified_kb_ref(),
                identity.canonical_name(),
            )
        })
    }

    let mut baseline = Corpus::new();
    let first = add_document(&mut baseline, "a");
    let peer = add_document(&mut baseline, "b");
    Resolver::new().resolve_inter_doc_coref(&mut baseline, None, None);
    baseline
        .link_track_to_kb(&first, "library", "42", "Ada")
        .unwrap();

    // A second corpus identity uses the same catalog key without sharing the first identity.
    let kb_track = add_document(&mut baseline, "c");
    let other = baseline
        .link_track_to_kb(&kb_track, "library", "42", "Ada")
        .unwrap();
    let doc = baseline.get_document_mut("c").unwrap();
    let kb_peer = add_track(doc);
    doc.link_track_to_identity(kb_peer.track_id, other);

    // A shared local record is rebound one track at a time.
    let mut doc = GroundedDocument::new("local", "Ada");
    let local_track = add_track(&mut doc);
    let local_peer = add_track(&mut doc);
    let local = doc.add_identity(Identity::from_kb(0, "Ada", "library", "42"));
    doc.link_track_to_identity(local_track.track_id, local);
    doc.link_track_to_identity(local_peer.track_id, local);
    baseline.add_document(doc);
    let unlinked = add_document(&mut baseline, "unlinked");

    // These scopes follow how the fixture was bound, independently of the API's prediction.
    let scopes = [
        vec![first, peer],
        vec![kb_track, kb_peer],
        vec![local_track],
        vec![local_peer],
        vec![unlinked],
    ];
    for scope in &scopes {
        for trigger in scope {
            for (catalog, key, label) in [
                ("library", "42", "Ada"),
                ("archive", "42", "Ada"),
                ("library", "77", "Ada"),
                ("library", "42", "Byron"),
            ] {
                let mut corpus = baseline.clone();
                let preview = corpus
                    .preview_kb_assignment(trigger, catalog, key, label)
                    .unwrap();
                assert_eq!(&preview.assignment().affected_tracks, scope);
                let receipt = corpus.apply_kb_assignment(preview).unwrap();
                assert_eq!(&receipt.assignment.affected_tracks, scope);

                for track in scopes.iter().flatten() {
                    if scope.contains(track) {
                        assert_eq!(
                            binding(&corpus, track),
                            Some((receipt.identity_id, Some((catalog, key)), label)),
                        );
                    } else {
                        assert_eq!(
                            binding(&corpus, track),
                            binding(&baseline, track),
                            "trigger {trigger:?}, outside track {track:?}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn a_bridge_discovered_during_review_cannot_rebind_a_catalogued_namesake() {
    fn member(corpus: &mut Corpus, doc_id: &str, text: &str, name: &str) -> TrackRef {
        let mut doc = GroundedDocument::new(doc_id, text);
        let id = doc.add_track(Track::new(0, name));
        let member = doc.track_ref(id).unwrap();
        corpus.add_document(doc);
        member
    }

    let mut corpus = Corpus::new();
    let a = member(&mut corpus, "a", "Ada North", "Ada North");
    let b = member(&mut corpus, "b", "Ada North Lee", "Ada North Lee");
    let c = member(&mut corpus, "c", "Iris, also called North Lee", "Iris");
    let resolve = |corpus: &mut Corpus| {
        Resolver::new().resolve_inter_doc_coref(corpus, Some(0.65), None);
    };
    let introduce_alias = |corpus: &mut Corpus| {
        corpus
            .get_document_mut("c")
            .unwrap()
            .get_track_mut(c.track_id)
            .unwrap()
            .canonical_surface = "North Lee".into();
    };

    // With the same lexical evidence and no catalog binding, B joins A to C.
    let mut control = corpus.clone();
    introduce_alias(&mut control);
    resolve(&mut control);
    assert_eq!(linked(&control, &a).id, linked(&control, &c).id);

    corpus.link_track_to_kb(&c, "catalog", "C", "Iris").unwrap();
    resolve(&mut corpus);
    let prepare = |corpus: &Corpus| {
        corpus
            .preview_kb_assignment(&a, "catalog", "AB", "Ada")
            .unwrap()
    };
    let pending = prepare(&corpus);
    assert_eq!(pending.assignment().affected_tracks, [a.clone(), b.clone()]);
    let before = [&a, &b, &c].map(|track| linked(&corpus, track).clone());

    introduce_alias(&mut corpus);
    resolve(&mut corpus);
    assert!(matches!(
        corpus.apply_kb_assignment(pending),
        Err(KbAssignmentError::StalePreview)
    ));
    assert_eq!(
        [&a, &b, &c].map(|track| linked(&corpus, track).clone()),
        before
    );

    let fresh = prepare(&corpus);
    let receipt = corpus.apply_kb_assignment(fresh).unwrap();
    assert_eq!(receipt.assignment.affected_tracks, [a.clone(), b.clone()]);
    for track in [&a, &b] {
        assert_eq!(
            linked(&corpus, track).qualified_kb_ref(),
            Some(("catalog", "AB"))
        );
    }
    assert_eq!(
        linked(&corpus, &c),
        &before[2],
        "C is outside the reviewed scope"
    );
}
