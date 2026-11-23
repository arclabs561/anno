//! Rebuilding a raw-edited document must agree with reading its wire form.

use anno::{GroundedDocument, Identity, IdentityId, Location, Signal, SignalId, Track, TrackId};

fn document() -> GroundedDocument {
    let mut doc = GroundedDocument::new("doc", "Zoë met Jo");
    let zoe = doc.add_signal(Signal::new(0, Location::text(0, 3), "Zoë", "PER", 1.0));
    let jo = doc.add_signal(Signal::new(0, Location::text(8, 10), "Jo", "PER", 1.0));
    let first = doc.create_track_from_signals("Zoë", &[zoe]).unwrap();
    doc.create_track_from_signals("Jo", &[jo]).unwrap();
    let identity = doc.add_identity(Identity::new(0, "Zoë"));
    doc.link_track_to_identity(first, identity);
    doc
}

fn reload(doc: &GroundedDocument) -> serde_json::Result<GroundedDocument> {
    serde_json::from_value(serde_json::to_value(doc).unwrap())
}

#[test]
fn raw_memberships_and_sparse_ids_rebuild_like_reload() {
    let mut doc = document();
    let signal = SignalId::new(17);
    let track = TrackId::new(23);
    let identity = IdentityId::new(29);

    doc.signals_mut()[0].id = signal;
    doc.signals_mut().reverse();
    let mut moved = doc.tracks_map_mut().remove(&TrackId::ZERO).unwrap();
    moved.id = track;
    moved.signals[0].signal_id = signal;
    moved.identity_id = Some(identity);
    doc.tracks_map_mut().insert(track, moved);
    doc.identities_map_mut()
        .insert(identity, Identity::new(identity, "Zoë"));
    doc.get_track_mut(1).unwrap().identity_id = Some(identity);

    let mut restored = reload(&doc).unwrap();
    assert!(!doc.invariants_hold());
    doc.rebuild_indexes();
    for result in [&mut doc, &mut restored] {
        assert!(result.invariants_hold());
        assert!(result.track_for_signal(SignalId::ZERO).is_none());
        assert_eq!(result.get_signal(signal).unwrap().surface, "Zoë");
        assert_eq!(result.track_for_signal(signal).unwrap().id, track);
        assert_eq!(result.identity_for_signal(signal).unwrap().id, identity);
        assert_eq!(
            result.identity_for_track(TrackId::new(1)).unwrap().id,
            identity
        );
        assert_eq!(
            result.add_signal(Signal::new(0, Location::text(0, 3), "Zoë", "PER", 1.0)),
            signal + 1
        );
        assert_eq!(result.add_track(Track::new(0, "empty")), track + 1);
        assert_eq!(result.add_identity(Identity::new(0, "Jo")), identity + 1);
    }
    assert_eq!(
        serde_json::to_value(&doc).unwrap(),
        serde_json::to_value(restored).unwrap()
    );
    let removed = track + 1;
    doc.tracks_map_mut().remove(&removed).unwrap();
    doc.rebuild_indexes();
    assert_eq!(doc.add_track(Track::new(0, "replacement")), removed);
}

fn raw_insert_ids(doc: &mut GroundedDocument, id: u64) {
    doc.signals_mut()
        .push(Signal::new(id, Location::text(0, 3), "Zoë", "PER", 1.0));
    doc.tracks_map_mut()
        .insert(TrackId::new(id), Track::new(id, "empty"));
    doc.identities_map_mut()
        .insert(IdentityId::new(id), Identity::new(id, "Jo"));
}

fn assert_stale_allocators(doc: &GroundedDocument) {
    let errors = doc.validate_invariants().join("\n");
    for kind in ["signal", "track", "identity"] {
        assert!(errors.contains(&format!("next {kind} ID")), "{errors}");
    }
}

#[test]
fn validation_reports_raw_ids_ahead_of_the_allocator() {
    let mut doc = document();
    raw_insert_ids(&mut doc, 99);
    assert_stale_allocators(&doc);
    doc.rebuild_indexes();
    assert!(doc.invariants_hold());
}

#[test]
fn rebuild_reports_unallocatable_ids_in_each_namespace() {
    let mut doc = document();
    raw_insert_ids(&mut doc, u64::MAX);
    assert!(reload(&doc).is_err());
    doc.rebuild_indexes();
    assert_stale_allocators(&doc);
}

#[test]
fn rebuild_keeps_invalid_memberships_and_mismatched_keys_visible() {
    let mut doc = document();
    doc.get_track_mut(1).unwrap().add_signal(0, 1);
    doc.get_track_mut(0).unwrap().id = TrackId::new(20);
    doc.identities_map_mut()
        .get_mut(&IdentityId::ZERO)
        .unwrap()
        .id = IdentityId::new(30);
    doc.rebuild_indexes();
    let errors = doc.validate_invariants().join("\n");
    assert!(errors.contains("multiple track references"));
    assert!(errors.contains("Track map key"));
    assert!(errors.contains("Identity map key"));
    assert!(reload(&doc).is_err());
}

#[test]
fn exhausted_allocations_leave_rebuilt_documents_round_trippable() {
    let mut doc = document();
    raw_insert_ids(&mut doc, u64::MAX - 1);
    doc.rebuild_indexes();
    assert!(doc.invariants_hold());

    let before = serde_json::to_value(&doc).unwrap();
    use std::panic::{catch_unwind, AssertUnwindSafe};
    assert!(catch_unwind(AssertUnwindSafe(|| {
        doc.add_signal(Signal::new(0, Location::text(0, 3), "Zoë", "PER", 1.0))
    }))
    .is_err());
    assert!(doc.try_add_track(Track::new(0, "empty")).is_err());
    assert!(doc
        .merge_tracks(&[TrackId::ZERO, TrackId::new(1)])
        .is_none());
    assert_eq!(doc.merge_tracks(&[TrackId::ZERO]), Some(TrackId::ZERO));
    assert!(catch_unwind(AssertUnwindSafe(|| doc.add_identity(Identity::new(0, "Jo")))).is_err());
    assert_eq!(serde_json::to_value(&doc).unwrap(), before);
    assert!(reload(&doc).unwrap().invariants_hold());
}
