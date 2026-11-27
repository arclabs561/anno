// SPDX-License-Identifier: MIT OR Apache-2.0
use std::{path::Path, process::Command};

#[test]
fn sqlite_cached_memberships_agree_with_fresh_cli_queries_after_bridge_edits() {
    let script =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../verification/memberships/replay.py");
    let interpreter = if cfg!(windows) { "python" } else { "python3" };
    let status = Command::new(interpreter)
        .arg(script)
        .arg(env!("CARGO_BIN_EXE_anno"))
        .status()
        .expect("run the standard-library SQLite reader");
    assert!(status.success());
}

#[test]
fn explain_submitted_pair_from_cli() {
    use anno::coalesce::memberships::{MembershipDocument, MembershipEdition};
    use anno::{GroundedDocument, Track};
    use serde_json::json;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edition.json");
    let edition = MembershipEdition {
        stream: "records".into(),
        sequence: 4,
        threshold: 0.7,
        require_type_match: true,
        documents: ["-a", "b"]
            .into_iter()
            .map(|id| {
                let mut document = GroundedDocument::new(id, "Kei");
                document.add_track(Track::new(0, "Kei"));
                MembershipDocument {
                    source_revision: None,
                    document,
                }
            })
            .collect(),
    };
    std::fs::write(&path, serde_json::to_vec(&edition).unwrap()).unwrap();
    let query = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_anno"))
            .args(["query", "memberships"])
            .arg(&path)
            .args(args)
            .output()
            .unwrap()
    };
    let json = |args: &[&str]| {
        let output = query(args);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        serde_json::from_slice::<serde_json::Value>(&output.stdout).unwrap()
    };
    let full = json(&[]);
    let result = json(&["--explain-pair", "b", "0", "-a", "0"]);
    assert_eq!(result["checkpoint"], full["checkpoint"]);
    assert_eq!(result["left"]["key"], full["rows"][1][0]);
    assert_eq!(result["right"]["key"], full["rows"][0][0]);
    assert_eq!(result["same_component"], true);
    assert_eq!(
        result["left"]["entity_type"],
        json!({"state": "unspecified"})
    );
    assert_eq!(
        result["decision"],
        json!({"outcome":"accepted", "basis":"word_jaccard", "similarity":1.0})
    );
    assert_eq!(
        json(&["--neighbors", "-a", "0"])["track"],
        result["right"]["key"]
    );
    for repeated in [
        vec!["--explain-pair", "-a", "0", "b", "0"],
        vec!["--neighbors", "-a", "0"],
    ] {
        let output = query(&repeated.repeat(2));
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert!(String::from_utf8_lossy(&output.stderr).contains("cannot be used multiple times"));
    }
    for args in [
        vec!["--explain-pair", "-a", "0", "b", "-1"],
        vec!["--explain-pair", "-a", "0", "b", "999"],
        vec![
            "--explain-pair",
            "-a",
            "0",
            "b",
            "0",
            "--neighbors",
            "-a",
            "0",
        ],
        vec!["--explain-pair", "-a", "0", "b", "0", "--since", "-"],
    ] {
        assert!(!query(&args).status.success(), "{args:?}");
    }
}

#[test]
fn retained_pair_explanation_verifies_only_for_its_edition_and_answer() {
    use anno::coalesce::memberships::{MembershipDocument, MembershipEdition};
    use anno::core::grounded::html_source::source_sha256;
    use anno::{GroundedDocument, Track};
    use serde_json::{json, Value};
    use std::fs;

    let dir = tempfile::tempdir().unwrap();
    let original = dir.path().join("original.json");
    let alternate = dir.path().join("alternate.json");
    let saved = dir.path().join("explanation.json");
    let edition = |right_type: &str| MembershipEdition {
        stream: "archive".into(),
        sequence: 3,
        threshold: 0.75,
        require_type_match: true,
        documents: [("a", "PER", vec![1., 0.]), ("b", right_type, vec![0., 1.])]
            .into_iter()
            .map(|(id, label, vector)| {
                let mut document = GroundedDocument::new(id, "Kei");
                let mut track = Track::new(0, "Kei");
                track.embedding = Some(vector);
                track.entity_type = Some(label.into());
                document.add_track(track);
                MembershipDocument {
                    source_revision: Some("issue-1".into()),
                    document,
                }
            })
            .collect(),
    };
    let mut original_bytes = serde_json::to_vec(&edition("ORG")).unwrap();
    original_bytes.extend_from_slice(b" \n");
    fs::write(&original, &original_bytes).unwrap();
    fs::write(&alternate, serde_json::to_vec(&edition("PER")).unwrap()).unwrap();
    let query = |edition: &Path, args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_anno"))
            .args(["query", "memberships"])
            .arg(edition)
            .args(args)
            .output()
            .unwrap()
    };
    let read = |edition: &Path, args: &[&str]| -> Value {
        let out = query(edition, args);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        serde_json::from_slice(&out.stdout).unwrap()
    };
    let pair = ["--explain-pair", "a", "0", "b", "0"];
    let verify = [
        pair.as_slice(),
        &["--verify-explanation", saved.to_str().unwrap()],
    ]
    .concat();
    let rejects = |edition: &Path, reason: &str| {
        let out = query(edition, &verify);
        assert!(!out.status.success());
        assert!(out.stdout.is_empty());
        assert!(
            String::from_utf8_lossy(&out.stderr).contains(reason),
            "{out:?}"
        );
    };

    assert_eq!(read(&original, &[]), read(&alternate, &[]));
    let recorded = read(&original, &pair);
    assert_eq!(
        recorded["edition_sha256"],
        source_sha256(std::str::from_utf8(&original_bytes).unwrap())
    );
    assert_eq!(recorded["format"], "anno.memberships.pair-explanation.v1");
    assert_eq!(recorded["decision"], json!({"outcome": "type_mismatch"}));
    fs::write(&saved, serde_json::to_vec_pretty(&recorded).unwrap()).unwrap();
    // The library's existing answer type can still consume the expanded CLI output.
    serde_json::from_value::<anno::coalesce::memberships::MembershipPair>(recorded.clone())
        .unwrap();
    assert_eq!(
        read(&alternate, &pair)["decision"]["outcome"],
        "below_threshold"
    );
    rejects(&alternate, "SHA-256");
    assert_eq!(read(&original, &verify), recorded);
    let reversed = query(
        &original,
        &[
            "--explain-pair",
            "b",
            "0",
            "a",
            "0",
            "--verify-explanation",
            saved.to_str().unwrap(),
        ],
    );
    assert!(!reversed.status.success());
    assert!(reversed.stdout.is_empty());

    // Identical answers from different bytes still fail the input pin.
    fs::write(&alternate, serde_json::to_vec(&edition("ORG")).unwrap()).unwrap();
    rejects(&alternate, "SHA-256");
    let valid_json = serde_json::to_string(&recorded).unwrap();
    for (field, conflicting) in [("same_component", "true"), ("outcome", "\"accepted\"")] {
        let key = format!("\"{field}\":");
        let ambiguous = valid_json.replacen(&key, &format!("{key}{conflicting},{key}"), 1);
        fs::write(&saved, ambiguous).unwrap();
        rejects(&original, "duplicate field");
    }
    let mut edited = recorded.clone();
    edited["same_component"] = json!(true);
    fs::write(&saved, serde_json::to_vec(&edited).unwrap()).unwrap();
    rejects(&original, "replayed pair answer");
    // Earlier releases wrote the direct answer without the full-edition fingerprint.
    edited.as_object_mut().unwrap().remove("format");
    edited.as_object_mut().unwrap().remove("edition_sha256");
    fs::write(&saved, serde_json::to_vec(&edited).unwrap()).unwrap();
    rejects(&original, "recording format");
    fs::write(&saved, serde_json::to_vec(&recorded).unwrap()).unwrap();
    fs::remove_file(&original).unwrap();
    rejects(&original, "original.json");
}
