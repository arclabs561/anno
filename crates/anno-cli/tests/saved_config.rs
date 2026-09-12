use std::process::Command;

fn anno() -> Command {
    Command::new(env!("CARGO_BIN_EXE_anno"))
}

#[test]
fn saved_config_applies_and_explicit_command_values_override_it() {
    let temp = tempfile::tempdir().unwrap();
    let config_dir = temp.path();

    let saved = anno()
        .args([
            "config", "save", "workflow", "--model", "pattern", "--format", "json",
        ])
        .env("ANNO_CONFIG_DIR", config_dir)
        .output()
        .unwrap();
    assert!(
        saved.status.success(),
        "config save failed: {}",
        String::from_utf8_lossy(&saved.stderr)
    );

    let from_config = anno()
        .args([
            "extract",
            "--config",
            "workflow",
            "--text",
            "alice@example.com",
        ])
        .env("ANNO_CONFIG_DIR", config_dir)
        .output()
        .unwrap();
    assert!(
        from_config.status.success(),
        "configured extract failed: {}",
        String::from_utf8_lossy(&from_config.stderr)
    );
    let output: serde_json::Value = serde_json::from_slice(&from_config.stdout).unwrap();
    assert_eq!(output["provenance"]["model"], "pattern");

    let override_output = anno()
        .args([
            "extract",
            "--config",
            "workflow",
            "--model",
            "heuristic",
            "--format",
            "jsonl",
            "--text",
            "alice@example.com",
        ])
        .env("ANNO_CONFIG_DIR", config_dir)
        .output()
        .unwrap();
    assert!(
        override_output.status.success(),
        "overridden extract failed: {}",
        String::from_utf8_lossy(&override_output.stderr)
    );
    let first_line = std::str::from_utf8(&override_output.stdout)
        .unwrap()
        .lines()
        .next()
        .unwrap();
    let provenance: serde_json::Value = serde_json::from_str(first_line).unwrap();
    assert_eq!(provenance["provenance"]["model"], "heuristic");
}

#[test]
fn malformed_saved_config_fails_before_execution() {
    let temp = tempfile::tempdir().unwrap();
    std::fs::write(temp.path().join("broken.toml"), "model = 'missing-model'\n").unwrap();

    let output = anno()
        .args([
            "extract",
            "--config",
            "broken",
            "--text",
            "alice@example.com",
        ])
        .env("ANNO_CONFIG_DIR", temp.path())
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Invalid model value 'missing-model' in config 'broken'"));
}

#[test]
fn invalid_saved_and_command_thresholds_fail_before_extraction() {
    let temp = tempfile::tempdir().unwrap();
    let config_dir = temp.path();

    for (name, contents) in [
        ("nan", "threshold = nan\n"),
        ("outside", "threshold = 1.1\n"),
    ] {
        std::fs::write(config_dir.join(format!("{name}.toml")), contents).unwrap();
        let output = anno()
            .args(["extract", "--config", name, "--file", "does-not-exist.txt"])
            .env("ANNO_CONFIG_DIR", config_dir)
            .output()
            .unwrap();

        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("--threshold must be a finite value between 0.0 and 1.0"));
        assert!(!stderr.contains("does-not-exist.txt"));
    }

    let output = anno()
        .args([
            "extract",
            "--relation-threshold=1.1",
            "--file",
            "does-not-exist.txt",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--relation-threshold must be a finite value between 0.0 and 1.0"));
    assert!(!stderr.contains("does-not-exist.txt"));
}

#[test]
fn explicit_false_debug_flags_override_saved_true_values() {
    let temp = tempfile::tempdir().unwrap();
    let config_dir = temp.path();
    let exported = temp.path().join("debug.json");

    let saved = anno()
        .args([
            "config",
            "save",
            "debug-workflow",
            "--model",
            "heuristic",
            "--coref",
            "--link-kb",
        ])
        .env("ANNO_CONFIG_DIR", config_dir)
        .output()
        .unwrap();
    assert!(
        saved.status.success(),
        "config save failed: {}",
        String::from_utf8_lossy(&saved.stderr)
    );

    let output = anno()
        .args([
            "debug",
            "--config",
            "debug-workflow",
            "--coref=false",
            "--link-kb=false",
            "--text",
            "Alice met Alice.",
            "--export",
            exported.to_str().unwrap(),
        ])
        .env("ANNO_CONFIG_DIR", config_dir)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "configured debug failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let document: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&exported).unwrap()).unwrap();
    assert_eq!(document["tracks"], serde_json::json!({}));
    assert_eq!(document["identities"], serde_json::json!({}));
}
