//! Tests for new CLI commands: enhance, pipeline, query, compare, cache, config, batch

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

fn setup_test_file(content: &str) -> (TempDir, String) {
    let dir = tempfile::tempdir().expect("Failed to create temp directory");
    let file_path = dir.path().join("test.txt");
    fs::write(&file_path, content).expect("Failed to write test file");
    (dir, file_path.to_string_lossy().to_string())
}

#[test]
fn test_enhance_command_basic() {
    let dir = tempfile::tempdir().expect("Failed to create temp directory");
    let test_doc = dir.path().join("test-doc.json");
    let enhanced = dir.path().join("enhanced.json");

    // First extract to create a GroundedDocument
    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&[
        "extract",
        "--export",
        test_doc.to_str().unwrap(),
        "Barack Obama met Angela Merkel",
    ])
    .assert()
    .success();

    // Now enhance with coreference
    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&[
        "enhance",
        test_doc.to_str().unwrap(),
        "--coref",
        "--export",
        enhanced.to_str().unwrap(),
    ])
    .assert()
    .success()
    .stderr(predicate::str::contains("Applied coreference resolution"));
}

#[test]
fn test_enhance_command_link_kb() {
    let dir = tempfile::tempdir().expect("Failed to create temp directory");
    let test_doc = dir.path().join("test-doc.json");
    let enhanced = dir.path().join("enhanced.json");

    // Extract and enhance with KB linking
    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&[
        "extract",
        "--export",
        test_doc.to_str().unwrap(),
        "Barack Obama met Angela Merkel",
    ])
    .assert()
    .success();

    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&[
        "enhance",
        test_doc.to_str().unwrap(),
        "--coref",
        "--link-kb",
        "--export",
        enhanced.to_str().unwrap(),
    ])
    .assert()
    .success();
}

#[test]
fn test_pipeline_command_text() {
    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&[
        "pipeline",
        "Apple Inc. was founded by Steve Jobs",
        "--coref",
    ])
    .assert()
    .success()
    .stdout(predicate::str::contains("Apple"));
}

#[test]
fn test_pipeline_command_multiple_texts() {
    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&[
        "pipeline",
        "Apple Inc. was founded by Steve Jobs",
        "Microsoft was founded by Bill Gates",
        "--coref",
    ])
    .assert()
    .success();
}

#[test]
fn test_query_command_single_doc() {
    let dir = tempfile::tempdir().expect("Failed to create temp directory");
    let test_doc = dir.path().join("test-doc.json");

    // First extract to JSON
    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&[
        "extract",
        "--export",
        test_doc.to_str().unwrap(),
        "--format",
        "grounded",
        "John works at Apple",
    ])
    .assert()
    .success();

    // Query for specific type
    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&["query", test_doc.to_str().unwrap(), "--type", "ORG"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Apple"));
}

#[test]
fn test_query_command_entity_search() {
    let dir = tempfile::tempdir().expect("Failed to create temp directory");
    let test_doc = dir.path().join("test-doc.json");

    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&[
        "extract",
        "--export",
        test_doc.to_str().unwrap(),
        "--format",
        "grounded",
        "Apple Inc. CEO Tim Cook",
    ])
    .assert()
    .success();

    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&["query", test_doc.to_str().unwrap(), "--entity", "Apple"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Apple"));
}

#[test]
fn test_compare_command_documents() {
    let dir = tempfile::tempdir().expect("Failed to create temp directory");
    let doc1 = dir.path().join("doc1.json");
    let doc2 = dir.path().join("doc2.json");

    // Create two documents
    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&[
        "extract",
        "--export",
        doc1.to_str().unwrap(),
        "--format",
        "grounded",
        "Apple Inc. CEO Tim Cook",
    ])
    .assert()
    .success();

    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&[
        "extract",
        "--export",
        doc2.to_str().unwrap(),
        "--format",
        "grounded",
        "Microsoft CEO Satya Nadella",
    ])
    .assert()
    .success();

    // Compare them
    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&[
        "compare",
        doc1.to_str().unwrap(),
        doc2.to_str().unwrap(),
        "--format",
        "summary",
    ])
    .assert()
    .success()
    .stdout(predicate::str::contains("Comparison"));
}

#[test]
fn test_cache_command_list() {
    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&["cache", "list"]).assert().success();
}

#[test]
fn test_cache_command_stats() {
    let mut cmd = Command::cargo_bin("anno").unwrap();
    let assert = cmd.args(&["cache", "stats"]).assert().success();
    let output = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    // Cache directory might not exist, so check for either message
    assert!(
        output.contains("Cache Statistics") || output.contains("Cache directory does not exist"),
        "Output: {}",
        output
    );
}

#[test]
fn test_config_command_list() {
    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&["config", "list"]).assert().success();
}

#[test]
#[cfg(all(feature = "cli", feature = "eval"))]
fn test_config_command_save_and_show() {
    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&[
        "config",
        "save",
        "test-workflow",
        "--model",
        "stacked",
        "--coref",
    ])
    .assert()
    .success();

    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&["config", "show", "test-workflow"])
        .assert()
        .success()
        .stdout(predicate::str::contains("model").or(predicate::str::contains("coref")));

    // Cleanup
    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&["config", "delete", "test-workflow"])
        .assert()
        .success();
}

#[test]
fn test_batch_command_directory() {
    let (dir, _) = setup_test_file("Apple Inc. was founded by Steve Jobs.");

    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&["batch", "--dir", dir.path().to_str().unwrap(), "--coref"])
        .assert()
        .success();
}

#[test]
fn test_pipeline_with_cross_doc() {
    #[cfg(feature = "eval-advanced")]
    {
        let (dir, _) = setup_test_file("Apple Inc. was founded by Steve Jobs.");

        // Create second file
        let file2 = dir.path().join("doc2.txt");
        fs::write(&file2, "Microsoft was founded by Bill Gates.").unwrap();

        let mut cmd = Command::cargo_bin("anno").unwrap();
        cmd.args(&[
            "pipeline",
            "--dir",
            dir.path().to_str().unwrap(),
            "--coref",
            "--cross-doc",
        ])
        .assert()
        .success();
    }
}

#[test]
fn test_enhance_from_stdin() {
    let dir = tempfile::tempdir().expect("Failed to create temp directory");
    let test_doc = dir.path().join("test-doc.json");

    // Extract first
    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&[
        "extract",
        "--export",
        test_doc.to_str().unwrap(),
        "--format",
        "grounded",
        "Test text",
    ])
    .assert()
    .success();

    // Enhance from stdin
    let mut cmd = Command::cargo_bin("anno").unwrap();
    let doc_json = fs::read_to_string(test_doc).unwrap();
    cmd.args(&["enhance", "-", "--coref"])
        .write_stdin(doc_json)
        .assert()
        .success();
}

#[test]
fn test_query_min_confidence() {
    let dir = tempfile::tempdir().expect("Failed to create temp directory");
    let test_doc = dir.path().join("test-doc.json");

    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&[
        "extract",
        "--export",
        test_doc.to_str().unwrap(),
        "--format",
        "grounded",
        "John works at Apple",
    ])
    .assert()
    .success();

    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&[
        "query",
        test_doc.to_str().unwrap(),
        "--min-confidence",
        "0.8",
    ])
    .assert()
    .success();
}

#[test]
fn test_compare_models() {
    let dir = tempfile::tempdir().expect("Failed to create temp directory");
    let text_file = dir.path().join("text.txt");
    fs::write(&text_file, "Apple Inc. was founded by Steve Jobs").unwrap();

    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&[
        "compare",
        "--models",
        "--model-list",
        "stacked,heuristic",
        text_file.to_str().unwrap(),
    ])
    .assert()
    .success();
}
