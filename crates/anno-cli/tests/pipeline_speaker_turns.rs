#![cfg(feature = "discourse")]

use anno::GroundedDocument;
use serde_json::json;
use std::process::Command;

#[test]
fn speaker_tracks_follow_turn_ids_and_exact_character_spans() {
    // Gaps include a speaker label and an unattributed turn.
    let utterances = [
        ("A", "I met you."),
        ("B", "I'm sure you met me."),
        ("A", "That was my idea."),
        ("A", "She said ‘I heard myself.’"),
        ("B", "Elle a dit mine."),
    ];
    let mut text = "Été. I was unattributed.\n".to_owned();
    let mut turns = Vec::new();
    for (i, (speaker, utterance)) in utterances.into_iter().enumerate() {
        text.push_str(&format!("{speaker}: "));
        let start = text.chars().count();
        text.push_str(utterance);
        turns.push(json!({
            "speaker": speaker, "text": utterance,
            "start": start, "end": text.chars().count(),
            "language": if i == 4 { "fr" } else { "en-GB" }
        }));
        text.push('\n');
    }
    let dir = tempfile::tempdir().unwrap();
    let sidecar = dir.path().join("turns.json");
    std::fs::write(&sidecar, serde_json::to_vec(&turns).unwrap()).unwrap();
    let run = || {
        let output = Command::new(env!("CARGO_BIN_EXE_anno"))
            .args([
                "pipeline",
                "--model",
                "heuristic",
                "--format",
                "json",
                "--quiet",
                "--coref",
                "--speaker-turns",
            ])
            .arg(&sidecar)
            .arg(&text)
            .env("ANNO_NO_DOWNLOADS", "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let [doc]: [GroundedDocument; 1] = serde_json::from_slice(&output.stdout).unwrap();
        doc
    };
    let doc = run();
    assert_eq!(doc.text(), text);
    assert_eq!(doc.coref_backend(), Some("speaker-turns"));
    let mut result = std::collections::BTreeMap::new();
    for track in doc.tracks() {
        let members: Vec<_> = track
            .signals()
            .iter()
            .map(|link| {
                let s = doc.get_signal(link.signal_id).unwrap();
                (s.surface(), s.text_offsets().unwrap())
            })
            .collect();
        result.insert(track.canonical_surface.as_str(), members);
    }
    assert_eq!(result.len(), 2);
    assert_eq!(result["A"], [("I", (28, 29)), ("my", (75, 77))]);
    assert_eq!(result["B"], [("I", (42, 43)), ("me", (59, 61))]);

    // An empty sidecar authorizes no mentions, even for repeated names or pronouns.
    std::fs::write(&sidecar, "[]").unwrap();
    let empty = run();
    assert_eq!(empty.coref_backend(), Some("speaker-turns"));
    assert_eq!(empty.tracks().count(), 0);
}
