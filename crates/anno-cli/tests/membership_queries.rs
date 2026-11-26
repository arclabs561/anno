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
        documents: ["a", "b"]
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
    let result = json(&["--explain-pair", "b", "0", "a", "0"]);
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
    for args in [
        vec!["--explain-pair", "a", "0", "b", "-1"],
        vec!["--explain-pair", "a", "0", "b", "999"],
        vec![
            "--explain-pair",
            "a",
            "0",
            "b",
            "0",
            "--neighbors",
            "a",
            "0",
        ],
        vec!["--explain-pair", "a", "0", "b", "0", "--since", "-"],
    ] {
        assert!(!query(&args).status.success(), "{args:?}");
    }
}
