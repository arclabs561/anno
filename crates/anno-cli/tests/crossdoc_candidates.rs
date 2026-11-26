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
            (
                None,
                if doc_count == 100 {
                    "exhaustive"
                } else {
                    "lsh"
                },
            ),
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

#[test]
fn explain_one_pair_reports_actual_candidate_and_threshold_decisions() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fs::write(dir.join("first.txt"), "Dr. Nara Pell presented a report.").unwrap();
    fs::write(dir.join("-second page.txt"), "Dr. Pell presented a report.").unwrap();
    let run = |mode, cutoff| {
        report(
            dir,
            &[
                ".",
                "--model",
                "heuristic",
                "--candidates",
                mode,
                "--threshold",
                cutoff,
                "--explain-pair",
                "-second page",
                "0",
                "first",
                "0",
                "--min-cluster-size",
                "10",
            ],
        )
    };
    let equal = run("exhaustive", "0.6666666666666666");
    assert_eq!(equal["clusters"], serde_json::json!([]));
    let receipt = &equal["pair_explanation"];
    assert_eq!(receipt["left"]["doc_id"], "-second page");
    assert_eq!(receipt["left"]["text"], "Dr. Pell");
    assert_eq!(receipt["right"]["entity_idx"], 0);
    assert_eq!(receipt["decision"]["outcome"], "accepted");
    assert_eq!(receipt["decision"]["basis"], "string_similarity");
    assert_eq!(receipt["decision"]["similarity"], 2.0 / 3.0);
    assert_eq!(receipt["same_component"], true);

    let higher = run("exhaustive", "0.6666666666666667");
    assert_eq!(
        higher["pair_explanation"]["decision"]["outcome"],
        "below_threshold"
    );
    assert_eq!(
        higher["pair_explanation"]["decision"]["similarity"],
        receipt["decision"]["similarity"]
    );
    assert_eq!(higher["pair_explanation"]["same_component"], false);

    let blocked = run("lsh", "0");
    assert_eq!(
        blocked["pair_explanation"]["decision"],
        serde_json::json!({"outcome":"not_proposed"})
    );
    assert_eq!(blocked["pair_explanation"]["same_component"], false);

    for (args, expected) in [
        (
            vec![
                ".",
                "--model",
                "heuristic",
                "--explain-pair",
                "first",
                "0",
                "-second page",
                "2",
            ],
            "unknown mention",
        ),
        (
            vec![".", "--explain-pair", "first", "a", "-second page", "0"],
            "invalid entity index",
        ),
        (
            vec![
                "--stdin",
                "--explain-pair",
                "first",
                "0",
                "-second page",
                "0",
            ],
            "cannot be used with",
        ),
        (
            vec![
                ".",
                "--format",
                "jsonl",
                "--explain-pair",
                "first",
                "0",
                "-second page",
                "0",
            ],
            "requires --format json",
        ),
    ] {
        let output = cross_doc(dir, &args);
        assert!(!output.status.success());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(error.contains(expected), "{error}");
    }
}
