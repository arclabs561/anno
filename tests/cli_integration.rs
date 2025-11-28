//! Integration tests for the anno-eval CLI tool.
//!
//! These tests verify the CLI commands work correctly.
//! Run with: cargo test --features eval --test cli_integration

#![cfg(feature = "eval")]

use std::process::Command;

fn cargo_bin() -> Command {
    let mut cmd = Command::new("cargo");
    cmd.args(["run", "--features", "eval", "--bin", "anno-eval", "--"]);
    cmd.current_dir(env!("CARGO_MANIFEST_DIR"));
    cmd
}

// =============================================================================
// Help and Version
// =============================================================================

#[test]
fn test_cli_help() {
    let output = cargo_bin().arg("help").output().expect("Failed to run CLI");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("anno-eval"));
    assert!(stdout.contains("COMMANDS"));
    assert!(stdout.contains("quick"));
    assert!(stdout.contains("bio"));
    assert!(stdout.contains("overlap"));
}

#[test]
fn test_cli_version() {
    let output = cargo_bin()
        .arg("--version")
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("anno-eval"));
}

// =============================================================================
// Quick Command
// =============================================================================

#[test]
fn test_cli_quick() {
    let output = cargo_bin()
        .arg("quick")
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should show evaluation output
    assert!(stdout.contains("Running quick evaluation"));
    assert!(stdout.contains("Strict F1"));
    assert!(stdout.contains("Summary"));
}

// =============================================================================
// BIO Commands
// =============================================================================

#[test]
fn test_cli_bio_validate_valid() {
    let output = cargo_bin()
        .args(["bio", "validate", "B-PER I-PER O B-ORG"])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Valid IOB2 sequence"));
}

#[test]
fn test_cli_bio_validate_invalid() {
    let output = cargo_bin()
        .args(["bio", "validate", "O I-PER I-PER O"])
        .output()
        .expect("Failed to run CLI");

    // Should fail with non-zero exit
    assert!(!output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Invalid IOB2 sequence"));
}

#[test]
fn test_cli_bio_repair() {
    let output = cargo_bin()
        .args(["bio", "repair", "O I-PER I-PER O"])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Original:"));
    assert!(stdout.contains("Repaired:"));
    assert!(stdout.contains("B-PER")); // Should have B-PER after repair
}

#[test]
fn test_cli_bio_convert() {
    let output = cargo_bin()
        .args(["bio", "convert", "B-PER I-PER O B-ORG"])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should show entity information
    assert!(stdout.contains("PER") || stdout.contains("ORG"));
}

// =============================================================================
// Overlap Command
// =============================================================================

#[test]
fn test_cli_overlap_partial() {
    let output = cargo_bin()
        .args(["overlap", "0", "10", "5", "15"])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Span 1:"));
    assert!(stdout.contains("Span 2:"));
    assert!(stdout.contains("Intersection:"));
    assert!(stdout.contains("IoU:"));
    assert!(stdout.contains("33")); // IoU should be ~33%
}

#[test]
fn test_cli_overlap_no_overlap() {
    let output = cargo_bin()
        .args(["overlap", "0", "5", "10", "15"])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("No overlap"));
}

#[test]
fn test_cli_overlap_complete() {
    let output = cargo_bin()
        .args(["overlap", "0", "10", "0", "10"])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("100")); // IoU should be 100%
}

// =============================================================================
// Error Handling
// =============================================================================

#[test]
fn test_cli_unknown_command() {
    let output = cargo_bin()
        .arg("unknown_command_xyz")
        .output()
        .expect("Failed to run CLI");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should mention unknown command in stdout or stderr
    assert!(
        stdout.contains("Unknown command") || stderr.contains("Unknown command"),
        "Expected 'Unknown command' in output"
    );
}

#[test]
fn test_cli_overlap_missing_args() {
    let output = cargo_bin()
        .args(["overlap", "0", "10"]) // Missing 2 args
        .output()
        .expect("Failed to run CLI");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Usage") || stderr.contains("start1"));
}

#[test]
fn test_cli_bio_no_subcommand() {
    let output = cargo_bin()
        .arg("bio")
        .output()
        .expect("Failed to run CLI");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Usage") || stderr.contains("validate"));
}

// =============================================================================
// Short aliases
// =============================================================================

#[test]
fn test_cli_short_quick() {
    let output = cargo_bin().arg("q").output().expect("Failed to run CLI");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Running quick evaluation"));
}

#[test]
fn test_cli_short_bio() {
    let output = cargo_bin()
        .args(["b", "v", "B-PER O"])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());
}

#[test]
fn test_cli_short_overlap() {
    let output = cargo_bin()
        .args(["o", "0", "10", "5", "15"])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());
}


