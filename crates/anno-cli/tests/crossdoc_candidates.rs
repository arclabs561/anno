#![cfg(feature = "eval")]

use anno::GroundedDocument;
use serde_json::Value;
use std::fs;
use std::path::Path;
use std::process::{Command, Output};

fn cross_doc(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_anno"))
        .arg("cross-doc")
        .args(args)
        .current_dir(dir)
        .env("ANNO_NO_DOWNLOADS", "1")
        .env("HF_HUB_OFFLINE", "1")
        .output()
        .unwrap()
}

fn report(dir: &Path, args: &[&str]) -> Value {
    let output = cross_doc(dir, args);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn raw_candidate_policy_controls_pell_pair_across_size_boundary() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fs::write(dir.join("first.txt"), "Dr. Nara Pell presented a report.").unwrap();
    fs::write(dir.join("second.txt"), "Dr. Pell presented a report.").unwrap();
    for n in 0..98 {
        fs::write(dir.join(format!("empty-{n}.txt")), "quiet.").unwrap();
    }

    for doc_count in [100, 101] {
        if doc_count == 101 {
            fs::write(dir.join("one-more.txt"), "quiet.").unwrap();
        }
        for (requested, actual) in [
            (None, "exhaustive"),
            (Some("exhaustive"), "exhaustive"),
            (Some("lsh"), "lsh"),
        ] {
            let mut args = vec![".", "--model", "heuristic"];
            if let Some(mode) = requested {
                args.extend(["--candidates", mode]);
            }
            let result = report(dir, &args);
            let metadata = &result["metadata"];
            assert_eq!(metadata["documents_processed"], doc_count);
            assert_eq!(metadata["total_entities"], 2);
            assert_eq!(metadata["resolver"], "flat_mentions");
            assert_eq!(metadata["candidate_mode"], actual);
            let clusters = result["clusters"].as_array().unwrap();
            if actual == "exhaustive" {
                assert_eq!(clusters.len(), 1);
                assert_eq!(
                    clusters[0]["mentions"],
                    serde_json::json!([["first", 0], ["second", 0]])
                );
            } else {
                assert_eq!(clusters.len(), 2);
                assert!(clusters
                    .iter()
                    .all(|c| c["mentions"].as_array().unwrap().len() == 1));
            }
        }
    }
}

#[test]
fn raw_auto_mode_counts_mentions_in_a_single_document() {
    let tmp = tempfile::tempdir().unwrap();
    for (mentions, expected) in [(100, "exhaustive"), (101, "lsh")] {
        fs::write(
            tmp.path().join("contacts.txt"),
            "reader@example.org\n".repeat(mentions),
        )
        .unwrap();
        let result = report(tmp.path(), &[".", "--model", "pattern"]);
        assert_eq!(result["metadata"]["documents_processed"], 1);
        assert_eq!(result["metadata"]["total_entities"], mentions);
        assert_eq!(result["metadata"]["candidate_mode"], expected);
    }
}

#[test]
fn imported_tracks_report_their_own_policy_and_reject_raw_options() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let doc = GroundedDocument::new("source", "");
    fs::write(dir.join("doc.json"), serde_json::to_vec(&doc).unwrap()).unwrap();
    let result = report(dir, &["--import", "doc.json"]);
    assert_eq!(result["metadata"]["resolver"], "grounded_tracks");
    assert_eq!(result["metadata"]["candidate_mode"], "exhaustive");

    for input in [
        vec!["--import", "doc.json"],
        vec!["--stdin"],
        vec!["--import", "-"],
    ] {
        let mut args = vec!["--candidates", "lsh"];
        args.extend(input);
        let output = cross_doc(dir, &args);
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("--candidates") && stderr.contains("cannot be used with"),
            "{stderr}"
        );
    }
}
