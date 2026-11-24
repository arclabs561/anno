#![cfg(feature = "eval")]

use anno::{CorefRunReport, GroundedDocument};
use std::process::Command;

fn documents(args: &[&str]) -> Vec<GroundedDocument> {
    let cache = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_anno"))
        .args(["pipeline", "--model", "stacked", "--format", "grounded"])
        .args(args)
        .env("XDG_CACHE_HOME", cache.path())
        .env("HF_HOME", cache.path())
        .env("FCOREF_MODEL_PATH", cache.path())
        .env("ANNO_NO_DOWNLOADS", "1")
        .env("HF_HUB_OFFLINE", "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn retain_unclaimed_detections_without_double_counting_coreference_mentions() {
    let first = "We met Alice and Alice. Write roster@example.org.";
    let second = "Contact roster@example.org at noon.";
    let extracted = documents(&[first, second]);
    let resolved = documents(&["--coref", "--cross-doc", first, second]);
    let mut email_identities = Vec::new();
    for (index, doc) in resolved.iter().enumerate() {
        let detections = extracted[index].signals();
        assert_eq!(&doc.signals()[..detections.len()], detections);
        let email = detections
            .iter()
            .find(|s| s.surface() == "roster@example.org")
            .unwrap();
        let track = doc.track_for_signal(email.id).unwrap();
        assert_eq!(track.signals().len(), 1);
        assert_eq!(track.entity_type.as_ref(), Some(&email.label));
        email_identities.push(doc.identity_for_track(track.id).unwrap().id);
        let reported = match doc.coref_run().unwrap() {
            CorefRunReport::Completed { tracks_added, .. } => *tracks_added,
            _ => panic!("expected local coreference to complete"),
        };
        assert_eq!(reported, doc.tracks().count() - 1);
    }
    assert_eq!(email_identities[0], email_identities[1]);
    let doc = &resolved[0];
    let alice_tracks: Vec<_> = doc
        .tracks()
        .filter(|t| t.canonical_surface == "Alice")
        .collect();
    assert_eq!(alice_tracks.len(), 1);
    assert_eq!(alice_tracks[0].signals().len(), 2);
}

#[cfg(feature = "discourse")]
#[test]
fn speaker_only_cross_doc_excludes_unclaimed_detections() {
    let dir = tempfile::tempdir().unwrap();
    let sidecar = dir.path().join("turns.json");
    std::fs::write(&sidecar, "[]").unwrap();
    let docs = documents(&[
        "--coref",
        "--cross-doc",
        "--speaker-turns",
        sidecar.to_str().unwrap(),
        "Contact roster@example.org at noon.",
    ]);
    assert!(docs[0]
        .signals()
        .iter()
        .any(|s| s.surface() == "roster@example.org"));
    assert_eq!(docs[0].tracks().count(), 0);
}
