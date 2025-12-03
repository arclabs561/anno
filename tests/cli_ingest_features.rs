//! Tests for new ingest features: URL resolution, preprocessing, graph export

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

#[test]
#[cfg(feature = "eval-advanced")]
fn test_extract_with_url() {
    // Test URL resolution (requires eval-advanced feature)
    // Note: This test may fail if URL is unreachable, but tests the feature exists
    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&["extract", "--url", "https://example.com"])
        .assert()
        .success();
}

#[test]
fn test_extract_with_clean() {
    // Test text cleaning
    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&["extract", "--clean", "Apple  Inc.   was   founded"])
        .assert()
        .success();
}

#[test]
fn test_extract_with_normalize() {
    // Test Unicode normalization
    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&["extract", "--normalize", "Marie Curie won the Nobel Prize"])
        .assert()
        .success();
}

#[test]
fn test_extract_with_detect_lang() {
    // Test language detection
    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&[
        "extract",
        "--detect-lang",
        "Marie Curie won the Nobel Prize",
    ])
    .assert()
    .success();
}

#[test]
fn test_extract_graph_export_neo4j() {
    // Test graph export to Neo4j format
    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&[
        "extract",
        "--export-graph",
        "neo4j",
        "Apple Inc. was founded by Steve Jobs",
    ])
    .assert()
    .success()
    .stdout(predicate::str::contains("CREATE").or(predicate::str::contains("Node")));
}

#[test]
fn test_extract_graph_export_networkx() {
    // Test graph export to NetworkX format
    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&[
        "extract",
        "--export-graph",
        "networkx",
        "Apple Inc. was founded by Steve Jobs",
    ])
    .assert()
    .success()
    .stdout(predicate::str::contains("nodes").or(predicate::str::contains("edges")));
}

#[test]
fn test_debug_with_url() {
    // Test debug command with URL
    #[cfg(feature = "eval-advanced")]
    {
        let mut cmd = Command::cargo_bin("anno").unwrap();
        cmd.args(&["debug", "--url", "https://example.com"])
            .assert()
            .success();
    }
}

#[test]
fn test_debug_with_preprocessing() {
    // Test debug command with preprocessing
    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&[
        "debug",
        "--clean",
        "--normalize",
        "Apple  Inc.   was   founded",
    ])
    .assert()
    .success();
}

#[test]
fn test_debug_graph_export() {
    // Test debug command with graph export
    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&[
        "debug",
        "--export-graph",
        "neo4j",
        "--coref",
        "Apple Inc. was founded by Steve Jobs. The company is based in Cupertino.",
    ])
    .assert()
    .success()
    .stdout(predicate::str::contains("CREATE").or(predicate::str::contains("Node")));
}

#[test]
fn test_enhance_with_graph_export() {
    // Test enhance command can work with graph export
    let dir = tempfile::tempdir().expect("Failed to create temp directory");
    let test_doc = dir.path().join("test-doc.json");

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

    // Enhance and export to graph
    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&[
        "enhance",
        test_doc.to_str().unwrap(),
        "--coref",
        "--export-graph",
        "networkx",
    ])
    .assert()
    .success()
    .stdout(predicate::str::contains("nodes").or(predicate::str::contains("edges")));
}
