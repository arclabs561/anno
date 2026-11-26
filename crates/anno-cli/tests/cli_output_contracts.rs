//! Exercise command boundaries with only local models and temporary files.
use serde_json::Value;
use std::{
    fs,
    path::Path,
    process::{Command, Output},
};

fn invoke(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_anno"))
        .args(args)
        .current_dir(dir)
        .env("ANNO_NO_DOWNLOADS", "1")
        .env("HF_HUB_OFFLINE", "1")
        .env("XDG_CACHE_HOME", dir)
        .output()
        .unwrap()
}
fn ok(dir: &Path, args: &[&str]) -> Vec<u8> {
    let out = invoke(dir, args);
    assert!(
        out.status.success(),
        "{args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

#[test]
fn export_preserves_edits_without_starting_the_selected_model() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let saved = ok(
        dir,
        &[
            "extract",
            "--text",
            "Marie works for Acme Inc.",
            "--model",
            "tplinker",
            "--extract-relations",
            "--relation-types",
            "WORKS_FOR",
            "--threshold",
            "0.1",
            "--format",
            "grounded",
        ],
    );
    let mut doc: Value = serde_json::from_slice(&saved).unwrap();
    let edge = &mut doc["relation_evidence"]["mentions"][0]["relation"];
    assert_eq!(edge["relation_type"], "WORKS_FOR");
    edge["relation_type"] = "REVIEWED".into();
    // Both records retain the same original document ID and must get distinct files.
    let row = doc.to_string();
    fs::write(dir.join("review.jsonl"), format!("{row}\n{row}\n")).unwrap();
    for fmt in ["brat", "graph-csv"] {
        ok(
            dir,
            &[
                "export",
                "-i",
                "review.jsonl",
                "-o",
                "out",
                "--model",
                "universal-ner",
                "-f",
                fmt,
            ],
        );
    }
    for number in [1, 2] {
        let prefix = format!("out/review-{number}");
        let ann = fs::read_to_string(dir.join(format!("{prefix}.ann"))).unwrap();
        assert!(ann.contains("REVIEWED Arg1:T1 Arg2:T2"), "{ann}");
        let edges = fs::read_to_string(dir.join(format!("{prefix}-edges.csv"))).unwrap();
        assert!(edges.contains("Marie,org:1_Acme_Inc,REVIEWED,"), "{edges}");
        assert!(!edges.contains("CO_OCCURS"));
    }
}

#[test]
fn direct_and_saved_brat_writers_agree_on_repeated_unicode_spans() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fs::write(dir.join("notice.txt"), "猫 hi@example.org / hi@example.org").unwrap();
    ok(
        dir,
        &[
            "extract",
            "--model",
            "pattern",
            "--file",
            "notice.txt",
            "--format",
            "brat",
            "--export",
            "direct",
        ],
    );
    fs::write(
        dir.join("notice.json"),
        ok(
            dir,
            &[
                "extract",
                "--model",
                "pattern",
                "--file",
                "notice.txt",
                "--format",
                "grounded",
            ],
        ),
    )
    .unwrap();
    ok(
        dir,
        &[
            "export",
            "-i",
            "notice.json",
            "-o",
            "converted",
            "-f",
            "brat",
        ],
    );
    for ext in ["ann", "txt", "anno-pair"] {
        let name = format!("notice.{ext}");
        assert_eq!(
            fs::read(dir.join("direct").join(&name)).unwrap(),
            fs::read(dir.join("converted").join(name)).unwrap()
        );
    }
}

#[test]
fn file_output_and_format_are_observable() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let stdout = ok(
        dir,
        &[
            "pipeline",
            "--model",
            "pattern",
            "--format",
            "jsonl",
            "--output",
            "documents.jsonl",
            "--quiet",
            "first",
            "second",
        ],
    );
    assert!(stdout.is_empty());
    let data = fs::read_to_string(dir.join("documents.jsonl")).unwrap();
    let rows: Vec<Value> = data
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["text"], "first");
    fs::write(dir.join("notice.txt"), "One hi@example.org").unwrap();
    ok(
        dir,
        &[
            "watch",
            "--directory",
            ".",
            "--output",
            "out",
            "--format",
            "jsonl",
            "--model",
            "pattern",
            "--initial",
            "--max-files",
            "1",
            "--quiet",
        ],
    );
    let out = fs::read_to_string(dir.join("out/notice.txt.jsonl")).unwrap();
    assert_eq!(out.lines().count(), 1);
    assert_eq!(
        serde_json::from_str::<Value>(&out).unwrap()["entity_count"],
        1
    );
    assert!(
        !invoke(dir, &["watch", "--directory", ".", "--format", "tsv"])
            .status
            .success()
    );
}

use proptest::prelude::*;

// Derive expected BRAT coordinates from the original characters, independently of
// both serializers. Writer parity alone would miss a shared byte-offset bug.
proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]
    #[test]
    fn saved_and_direct_spans_match_the_source(
        pieces in prop::collection::vec(prop::sample::select(vec!["猫", "👩‍🔬", "e\u{301}", "\r\n", "\t", " "]), 0..16),
        repetitions in 1usize..5,
    ) {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        let email = "hi@example.org";
        let gap = format!(" {} ", pieces.concat());
        let text = format!("{gap}{}{gap}", vec![email; repetitions].join(&gap));
        fs::write(dir.join("record.txt"), &text).unwrap();
        ok(dir, &["extract","-m","pattern","-f","record.txt","--format","brat","--export","direct"]);
        let doc = ok(dir, &["extract","-m","pattern","-f","record.txt","--format","grounded"]);
        fs::write(dir.join("record.json"), doc).unwrap();
        ok(dir, &["export","-i","record.json","-o","saved","-f","brat"]);
        let expected: Vec<_> = text.match_indices(email).map(|(byte, _)| {
            let start = text[..byte].chars().count();
            format!("EMAIL {start} {}\t{email}", start + email.len())
        }).collect();
        for folder in ["direct", "saved"] {
            let source = fs::read_to_string(dir.join(folder).join("record.txt")).unwrap();
            prop_assert_eq!(&source, &text);
            let records = fs::read_to_string(dir.join(folder).join("record.ann")).unwrap();
            let mentions: Vec<_> = records.lines().filter_map(|line| line.split_once('\t'))
                .filter(|(_, body)| body.starts_with("EMAIL ")).map(|(_, body)| body.to_owned()).collect();
            prop_assert_eq!(&mentions, &expected);
        }
    }
}

#[test]
fn rejected_options_exit_before_processing_and_leave_no_output() {
    let tmp = tempfile::tempdir().unwrap();
    for (args, reason) in [
        (
            vec!["batch", "--stdin", "--dir", "missing"],
            "cannot be used with",
        ),
        (
            vec!["domain", "--text", "one", "--input", "missing"],
            "cannot be used with",
        ),
        (vec!["dataset", "check", "--fix"], "unexpected argument"),
        (vec!["eval", "--html", "--json"], "cannot be used with"),
        (
            vec!["compare", "absent", "--format", "yaml"],
            "invalid value",
        ),
        (
            vec!["enhance", "absent", "--format", "tsv"],
            "invalid value",
        ),
        (
            vec!["watch", "--directory", ".", "--interval", "0"],
            "invalid value",
        ),
        (
            vec![
                "pipeline", "--format", "human", "--output", "result", "text",
            ],
            "--output",
        ),
        (
            vec!["debug", "--output", "result", "--text", "text"],
            "--output",
        ),
    ] {
        let output = invoke(tmp.path(), &args);
        assert!(!output.status.success(), "{args:?}");
        assert!(output.stdout.is_empty(), "{args:?} emitted data");
        let diagnostic = String::from_utf8_lossy(&output.stderr);
        assert!(diagnostic.contains(reason), "{args:?}: {diagnostic}");
        assert!(!tmp.path().join("result").exists());
    }
}

#[test]
fn explain_uses_the_selected_occurrence_and_its_actual_rule() {
    let tmp = tempfile::tempdir().unwrap();
    let text = "猫 hi@example.org or hi@example.org";
    let data = ok(
        tmp.path(),
        &["extract", "-m", "pattern", "-t", text, "--format", "json"],
    );
    let compact: Value = serde_json::from_slice(&data).unwrap();
    let hits = compact["entities"].as_array().unwrap();
    assert_eq!(hits.len(), 2);
    assert_ne!(hits[0]["id"], hits[1]["id"]);
    for hit in hits {
        let output = ok(
            tmp.path(),
            &[
                "explain",
                "-m",
                "pattern",
                "-t",
                text,
                "--entity-id",
                hit["id"].as_str().unwrap(),
            ],
        );
        let report = String::from_utf8(output).unwrap();
        assert_eq!(report.matches("character offset)").count(), 2, "{report}");
        assert!(
            report.contains(&format!("start: {} (", hit["start"])),
            "{report}"
        );
        assert!(report.contains("rule: EMAIL"));
        assert!(!report.contains("Features:"));
    }
    assert!(!invoke(
        tmp.path(),
        &[
            "explain",
            "-m",
            "pattern",
            "-t",
            text,
            "--entity-id",
            "unknown"
        ]
    )
    .status
    .success());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]
    #[test]
    fn any_conflicting_artifact_blocks_the_entire_saved_batch(
        n in 2usize..7,
        conflict in 0usize..6,
        extension in prop::sample::select(vec!["ann","txt","anno-pair"]),
        array in any::<bool>(),
    ) {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        let saved = ok(dir, &["extract", "-m", "pattern", "-t", "x@example.org", "--format","grounded"]);
        let document: Value = serde_json::from_slice(&saved).unwrap();
        let documents = vec![document; n];
        let input = if array {
            serde_json::to_string(&documents).unwrap()
        } else {
            documents.iter().map(Value::to_string).collect::<Vec<_>>().join("\n")
        };
        fs::write(dir.join("batch.json"), input).unwrap();
        let out = dir.join("out");
        fs::create_dir(&out).unwrap();
        let existing = out.join(format!("batch-{}.{}", 1 + conflict % n, extension));
        fs::write(&existing, b"reviewed content").unwrap();
        let result = invoke(dir, &["export","-i","batch.json","-o","out","-f","brat"]);
        prop_assert!(!result.status.success());
        prop_assert!(String::from_utf8_lossy(&result.stderr).contains("already exists"));
        prop_assert_eq!(fs::read(&existing).unwrap(), b"reviewed content");
        prop_assert_eq!(out.read_dir().unwrap().count(), 1);
    }
}

#[test]
fn model_comparison_json_and_jsonl_have_the_same_rows() {
    let tmp = tempfile::tempdir().unwrap();
    fs::write(tmp.path().join("input.txt"), "Contact hi@example.org").unwrap();
    let run = |format: &str| {
        ok(
            tmp.path(),
            &[
                "compare",
                "input.txt",
                "--models",
                "--model-list",
                "pattern,heuristic",
                "--format",
                format,
            ],
        )
    };
    let rows: Vec<Value> = serde_json::from_slice(&run("json")).unwrap();
    let lines = String::from_utf8(run("jsonl")).unwrap();
    let streamed: Vec<Value> = lines
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows, streamed);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["model"], "pattern");
    assert_eq!(rows[0]["entities"][0]["text"], "hi@example.org");
}
