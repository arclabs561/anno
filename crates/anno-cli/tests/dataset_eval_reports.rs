#![cfg(feature = "eval")]

use serde_json::Value;
use std::fs;
use std::path::Path;
use std::process::{Command, Output};

fn anno(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_anno"))
        .args(args)
        .current_dir(dir)
        .env("ANNO_NO_DOWNLOADS", "1")
        .env("ANNO_S3_CACHE", "0")
        .env("ANNO_CACHE_DIR", dir.join("cache"))
        .output()
        .unwrap()
}

fn success(output: Output) -> String {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

fn failure(output: Output) -> String {
    assert!(!output.status.success());
    String::from_utf8(output.stderr).unwrap()
}

fn reports(dir: &Path, file: &str) -> Vec<Value> {
    fs::read_to_string(dir.join(file))
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect()
}

#[test]
fn cached_only_uses_each_task_reader_and_identifies_the_file_on_a_miss() {
    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path();
    let cache = dir.join("cache/datasets");
    fs::create_dir_all(&cache).unwrap();
    for (id, task, contents, expected) in [
        ("WikiGold", "ner", "Ada B-PER\n", "Sentences: 1"),
        ("GAP", "coref", concat!(
            "ID\tText\tPronoun\tPronoun-offset\tA\tA-offset\tA-coref\tB\tB-offset\tB-coref\tURL\n",
            "trial\tAda met Ben. She left.\tShe\t13\tAda\t0\tTRUE\tBen\t8\tFALSE\thttps://example.org\n"
        ), "Documents: 1"),
        ("CrossREAI", "relation",
            r#"{"sentence":["Ada","joined","Acme"],"ner":[[0,0,"person"],[2,2,"company"]],"relations":[[0,0,2,2,"works_for"]]}"#,
            "Documents: 1"),
    ] {
        let args = ["dataset", "eval", "--cached-only", "--dataset", id, "--task", task, "--model", "heuristic"];
        let miss = anno(dir, &args);
        let path = cache.join(format!("{id}.cache"));
        let diagnostic = failure(miss);
        assert!(diagnostic.contains(&format!("{id}.cache")), "{diagnostic}");
        assert!(!path.exists());
        fs::write(&path, contents).unwrap();
        let summary = success(anno(dir, &args));
        assert!(summary.contains("local dataset files only") && summary.contains(expected), "{summary}");
    }
}

#[test]
fn every_observation_uses_the_native_single_document_report() {
    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path();
    let summary = success(anno(
        dir,
        &[
            "dataset",
            "eval",
            "--task",
            "ner",
            "--model",
            "pattern",
            "--reports",
            "batch.jsonl",
            "--html",
            "--output",
            "review.html",
            "--max-cases",
            "1",
        ],
    ));
    assert!(summary.contains("Sentences: 8") && summary.contains("Results:"));
    assert!(dir.join("review.html").is_file());
    assert_eq!(fs::read_dir(dir.join("review_files")).unwrap().count(), 1);
    let mut batch = reports(dir, "batch.jsonl");
    assert_eq!(batch.len(), 8);
    for (i, r) in batch.iter().enumerate() {
        assert_eq!(r["pairing"]["document_id"], format!("synthetic:{}", i + 1));
        assert!(r["elapsed_ms"].as_f64().unwrap() >= 0.0);
        assert!(r["label_outcomes"].is_object());
    }
    let mut single: Value = serde_json::from_str(&success(anno(
        dir,
        &[
            "eval",
            "--model",
            "pattern",
            "--text",
            "Contact john@example.com for help.",
            "--gold",
            "john@example.com:EMAIL:8:24",
            "--pair-id",
            "synthetic:8",
            "--json",
        ],
    )))
    .unwrap();
    single.as_object_mut().unwrap().remove("elapsed_ms");
    batch[7].as_object_mut().unwrap().remove("elapsed_ms");
    assert_eq!(batch[7], single);
}

#[test]
fn registry_rows_with_empty_gold_and_duplicate_text_are_preserved_and_pairable() {
    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path();
    let cache = dir.join("cache/datasets");
    fs::create_dir_all(&cache).unwrap();
    fs::write(
        cache.join("WikiGold.cache"),
        "quiet O\n\nquiet O\n\nParis B-LOC\n\nx@y.com O\n",
    )
    .unwrap();
    for (model, target) in [("pattern", "base.jsonl"), ("heuristic", "candidate.jsonl")] {
        let stdout = success(anno(
            dir,
            &[
                "dataset",
                "eval",
                "--dataset",
                "WikiGold",
                "--model",
                model,
                "--reports",
                target,
            ],
        ));
        assert!(stdout.contains("Sentences: 4"));
    }
    let base = reports(dir, "base.jsonl");
    assert_eq!(base.len(), 4);
    assert_eq!(base[0]["gold_count"], 0);
    assert_eq!(base[0]["predicted_count"], 0);
    assert_eq!(base[1]["gold_count"], 0);
    assert_eq!(base[2]["gold_count"], 1);
    assert_eq!(base[3]["gold_count"], 0);
    assert_eq!(base[3]["predicted_count"], 1);
    assert_eq!(
        base[0]["pairing"]["source_sha256"],
        base[1]["pairing"]["source_sha256"]
    );
    assert_eq!(base[0]["pairing"]["document_id"], "WikiGold:1");
    assert_eq!(base[1]["pairing"]["document_id"], "WikiGold:2");
    let error: Value = serde_json::from_str(&success(anno(
        dir,
        &["viz", "errors", "base.jsonl", "--format", "json"],
    )))
    .unwrap();
    assert_eq!(error["reports"], 4);
    let pair: Value = serde_json::from_str(&success(anno(
        dir,
        &[
            "viz",
            "paired",
            "base.jsonl",
            "--against",
            "candidate.jsonl",
            "--format",
            "json",
        ],
    )))
    .unwrap();
    assert_eq!(pair["rows"][0]["observations"], 4);
    assert!(pair["rows"][0]["baseline"]["median_ms"].is_number());
    assert!(pair["rows"][0]["candidate"]["median_ms"].is_number());

    // IDs survive re-running the loader; different text or reference labels cannot be paired.
    let file = cache.join("WikiGold.cache");
    let original = fs::read_to_string(&file).unwrap();
    for changed in [
        original.replace("Paris", "Rome"),
        original.replace("B-LOC", "B-PER"),
    ] {
        fs::write(&file, changed).unwrap();
        success(anno(
            dir,
            &[
                "dataset",
                "eval",
                "--dataset",
                "WikiGold",
                "--model",
                "pattern",
                "--reports",
                "changed.jsonl",
            ],
        ));
        let diagnostic = failure(anno(
            dir,
            &[
                "viz",
                "paired",
                "base.jsonl",
                "--against",
                "changed.jsonl",
                "--format",
                "json",
            ],
        ));
        assert!(
            diagnostic.contains("source or gold reference differs for document WikiGold:3"),
            "{diagnostic}"
        );
    }
}

#[test]
fn incompatible_tasks_and_html_failures_leave_existing_reports_intact() {
    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path();
    fs::write(dir.join("keep"), "existing").unwrap();
    for task in ["coref", "relation"] {
        assert!(failure(anno(
            dir,
            &[
                "dataset",
                "eval",
                "--task",
                task,
                "--dataset",
                "unknown",
                "--reports",
                "keep"
            ]
        ))
        .contains("--reports supports --task ner only"));
    }
    assert!(failure(anno(
        dir,
        &[
            "dataset",
            "eval",
            "--reports",
            "keep",
            "--html",
            "--output",
            "./keep"
        ]
    ))
    .contains("conflicts"));
    assert!(failure(anno(
        dir,
        &[
            "dataset",
            "eval",
            "--reports",
            "review_files",
            "--html",
            "--output",
            "review.html"
        ]
    ))
    .contains("conflicts"));
    fs::create_dir(dir.join("directory")).unwrap();
    assert!(!anno(
        dir,
        &[
            "dataset",
            "eval",
            "--model",
            "pattern",
            "--reports",
            "keep",
            "--html",
            "--output",
            "directory"
        ]
    )
    .status
    .success());
    assert_eq!(fs::read_to_string(dir.join("keep")).unwrap(), "existing");
}

#[cfg(unix)]
#[test]
fn aliased_html_outputs_cannot_overwrite_the_stream_or_an_existing_case() {
    use std::os::unix::fs::symlink;
    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path();
    fs::write(dir.join("keep"), "existing").unwrap();
    fs::hard_link(dir.join("keep"), dir.join("review.html")).unwrap();
    assert!(failure(anno(
        dir,
        &[
            "dataset",
            "eval",
            "--reports",
            "keep",
            "--html",
            "--output",
            "review.html"
        ]
    ))
    .contains("conflicts"));
    symlink("uncreated.jsonl", dir.join("dangling.html")).unwrap();
    assert!(failure(anno(
        dir,
        &[
            "dataset",
            "eval",
            "--reports",
            "uncreated.jsonl",
            "--html",
            "--output",
            "dangling.html"
        ]
    ))
    .contains("conflicts"));
    assert!(!dir.join("uncreated.jsonl").exists());
    fs::create_dir(dir.join("outputs")).unwrap();
    symlink("outputs", dir.join("alias")).unwrap();
    assert!(failure(anno(
        dir,
        &[
            "dataset",
            "eval",
            "--reports",
            "outputs/new",
            "--html",
            "--output",
            "alias/new"
        ]
    ))
    .contains("conflicts"));
    assert!(!dir.join("outputs/new").exists());
    fs::create_dir(dir.join("detail_files")).unwrap();
    symlink("../keep", dir.join("detail_files/case_000000.html")).unwrap();
    assert!(failure(anno(
        dir,
        &[
            "dataset",
            "eval",
            "--model",
            "pattern",
            "--reports",
            "keep",
            "--html",
            "--output",
            "detail.html",
            "--min-errors",
            "0"
        ]
    ))
    .contains("an HTML case page"));
    assert_eq!(fs::read_to_string(dir.join("keep")).unwrap(), "existing");
    assert!(!dir.join("detail.html").exists());
}
