#![cfg(feature = "discourse")]

use anno::GroundedDocument;
use serde_json::json;
use std::process::Command;

#[test]
fn speaker_tracks_follow_turn_ids_and_exact_character_spans() {
    // Gaps include a speaker label and an unattributed turn.
    let utterances = [
        ("Zoë", "I met you."),
        ("Léa", "I'm sure you met me."),
        ("Zoë", "That was my idea."),
        ("Zoë", "She said ‘I heard myself.’"),
        ("Léa", "Elle a dit mine."),
    ];
    let mut text = "👩‍💻 e\u{301}. Été. I was unattributed.\n".to_owned();
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
    text.push_str("End.");
    let dir = tempfile::tempdir().unwrap();
    let sidecar = dir.path().join("turns.json");
    std::fs::write(&sidecar, serde_json::to_vec(&turns).unwrap()).unwrap();
    let run = || {
        Command::new(env!("CARGO_BIN_EXE_anno"))
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
            .unwrap()
    };
    let read_doc = || {
        let output = run();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let [doc]: [GroundedDocument; 1] = serde_json::from_slice(&output.stdout).unwrap();
        doc
    };
    let doc = read_doc();
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
    assert_eq!(result["Zoë"], [("I", (38, 39)), ("my", (89, 91))]);
    assert_eq!(result["Léa"], [("I", (54, 55)), ("me", (71, 73))]);

    // The final turn uses UTF-16 code units after two non-BMP scalars.
    // Earlier valid turns must not produce a partial document.
    let last = turns.last_mut().unwrap();
    for field in ["start", "end"] {
        let scalar_offset = last[field].as_u64().unwrap() as usize;
        let utf16_offset: usize = text.chars().take(scalar_offset).map(char::len_utf16).sum();
        assert_eq!(utf16_offset, scalar_offset + 2);
        last[field] = json!(utf16_offset);
    }
    std::fs::write(&sidecar, serde_json::to_vec(&turns).unwrap()).unwrap();
    let invalid = run();
    assert!(!invalid.status.success());
    assert!(invalid.stdout.is_empty());
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("exact source text"));

    // An empty sidecar authorizes no mentions, even for repeated names or pronouns.
    std::fs::write(&sidecar, "[]").unwrap();
    let empty = read_doc();
    assert_eq!(empty.coref_backend(), Some("speaker-turns"));
    assert_eq!(empty.tracks().count(), 0);
}
