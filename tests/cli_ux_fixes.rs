//! Tests for CLI UX fixes based on CLI_UX_CRITIQUE.md
//!
//! Tests:
//! - Debug command supports positional args
//! - Info command shows actually available models
//! - Models subcommand works correctly
//! - Cross-doc examples in help text are correct

use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn test_debug_supports_positional_args() {
    // Test that debug command accepts positional text arguments
    // This was a quick win fix - debug should work like extract
    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&["debug", "Marie Curie won the Nobel Prize"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Marie Curie"));
}

#[test]
fn test_debug_supports_text_flag() {
    // Test that debug command still works with --text flag
    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&["debug", "--text", "Barack Obama met Angela Merkel"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Barack Obama"));
}

#[test]
fn test_info_shows_available_models() {
    // Test that info command shows which models are actually available
    let mut cmd = Command::cargo_bin("anno").unwrap();
    let assert = cmd.args(&["info"]).assert().success();

    let output = String::from_utf8(assert.get_output().stdout.clone()).unwrap();

    // Should show model availability status
    assert!(output.contains("Available Models") || output.contains("available"));

    // Should show at least pattern, heuristic, stacked (always available)
    // Note: actual output uses "RegexNER", "StackedNER" etc.
    assert!(
        output.contains("RegexNER") || output.contains("pattern") || output.contains("Pattern")
    );
    assert!(
        output.contains("StackedNER") || output.contains("stacked") || output.contains("Stacked")
    );
}

#[test]
fn test_info_shows_enabled_features() {
    // Test that info command shows enabled features
    let mut cmd = Command::cargo_bin("anno").unwrap();
    let assert = cmd.args(&["info"]).assert().success();

    let output = String::from_utf8(assert.get_output().stdout.clone()).unwrap();

    // Should show "Enabled Features" section
    assert!(output.contains("Enabled Features") || output.contains("Features"));
}

#[test]
fn test_models_list_command() {
    // Test that models list command works
    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&["models", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Available Models"));
}

#[test]
fn test_models_info_command() {
    // Test that models info command works for available models
    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&["models", "info", "stacked"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Model Information"));
}

#[test]
fn test_models_info_unavailable_model() {
    // Test that models info shows helpful message for unavailable models
    let mut cmd = Command::cargo_bin("anno").unwrap();
    let assert = cmd.args(&["models", "info", "gliner"]).assert();

    // Should either succeed (if onnx feature enabled) or show helpful error
    let output = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    let stderr = String::from_utf8(assert.get_output().stderr.clone()).unwrap();
    let combined = output + &stderr;

    // Should mention feature flags or show model info
    assert!(
        combined.contains("feature")
            || combined.contains("onnx")
            || combined.contains("Model Information")
    );
}

#[test]
fn test_models_compare_command() {
    // Test that models compare command works
    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&["models", "compare"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Model Comparison"));
}

#[test]
fn test_cross_doc_help_examples() {
    // Test that cross-doc help shows correct command name (cross-doc not crossdoc)
    let mut cmd = Command::cargo_bin("anno").unwrap();
    let assert = cmd.args(&["cross-doc", "--help"]).assert().success();

    let output = String::from_utf8(assert.get_output().stdout.clone()).unwrap();

    // Examples should use "cross-doc" not "crossdoc"
    assert!(output.contains("anno cross-doc") || output.contains("cross-doc /path"));

    // Should not show incorrect "crossdoc" in examples
    // (allow it in description but not in examples)
    let example_section = output
        .lines()
        .skip_while(|l| !l.contains("Examples"))
        .take_while(|l| !l.is_empty() && !l.starts_with("#"))
        .collect::<Vec<_>>()
        .join("\n");

    // In examples section, should use cross-doc
    if example_section.contains("crossdoc") && !example_section.contains("cross-doc") {
        panic!("Examples should use 'cross-doc' not 'crossdoc'");
    }
}

#[test]
fn test_extract_debug_consistency() {
    // Test that extract and debug both support the same input methods
    let text = "Apple Inc. was founded by Steve Jobs.";

    // Both should work with positional args
    let mut cmd_extract = Command::cargo_bin("anno").unwrap();
    cmd_extract.args(&["extract", text]).assert().success();

    let mut cmd_debug = Command::cargo_bin("anno").unwrap();
    cmd_debug.args(&["debug", text]).assert().success();

    // Both should work with --text flag
    let mut cmd_extract_flag = Command::cargo_bin("anno").unwrap();
    cmd_extract_flag
        .args(&["extract", "--text", text])
        .assert()
        .success();

    let mut cmd_debug_flag = Command::cargo_bin("anno").unwrap();
    cmd_debug_flag
        .args(&["debug", "--text", text])
        .assert()
        .success();
}

#[test]
fn test_models_command_aliases() {
    // Test that models command has alias
    let mut cmd = Command::cargo_bin("anno").unwrap();
    cmd.args(&["m", "list"]).assert().success();
}

#[test]
fn test_models_subcommand_aliases() {
    // Test that models subcommands have aliases
    let mut cmd_list = Command::cargo_bin("anno").unwrap();
    cmd_list.args(&["models", "ls"]).assert().success();

    let mut cmd_info = Command::cargo_bin("anno").unwrap();
    cmd_info
        .args(&["models", "i", "stacked"])
        .assert()
        .success();

    let mut cmd_compare = Command::cargo_bin("anno").unwrap();
    cmd_compare.args(&["models", "c"]).assert().success();
}
