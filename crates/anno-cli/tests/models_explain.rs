//! Exercise model routing preflight with no usable neural weights or interpreters.
use serde_json::Value;
use std::{fs, path::Path, process::Command};
use tempfile::tempdir;

fn plan(root: &Path, args: &[&str]) -> Value {
    let result = Command::new(env!("CARGO_BIN_EXE_anno"))
        .args(args)
        .env("HF_HOME", root)
        .env("HF_HUB_CACHE", root.join("hub"))
        .env("ANNO_CONFIG_DIR", root)
        .env("ANNO_NO_DOWNLOADS", "1")
        .env("HF_HUB_OFFLINE", "1")
        .env("ANNO_GLINEREL_MODEL_DIR", root.join("glinerel"))
        .env("ANNO_GLINEREL_PYTHON", root.join("missing-python"))
        .env("OTTER_CROSS_MODEL_DIR", root.join("otter-cross"))
        .env("OTTER_BI_MODEL_DIR", root.join("otter-bi"))
        .env("GLINER2_PII_MODEL_DIR", root.join("pii"))
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let value: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(value["runtime_checked"], false);
    value
}

#[test]
fn explains_empty_and_prepared_schema_routes_without_starting_workers() {
    let dir = tempdir().unwrap();
    let args = [
        "models",
        "explain",
        "extract",
        "--extract-types",
        "city",
        "--json",
    ];
    let empty = plan(dir.path(), &args);
    assert!(empty["planned_first_attempt"].is_null());
    assert_eq!(empty["candidates"][0]["model"], "otter-cross");
    let checkpoint = dir.path().join("glinerel");
    fs::create_dir_all(checkpoint.join("encoder_config")).unwrap();
    for file in [
        "config.json",
        "tokenizer.json",
        "model.safetensors",
        "encoder_config/config.json",
    ] {
        fs::write(checkpoint.join(file), b"invalid model; preflight only").unwrap();
    }
    let prepared = plan(dir.path(), &args);
    assert_eq!(prepared["planned_first_attempt"], "glinerel-72m");
    let joint = plan(
        dir.path(),
        &[
            "models",
            "explain",
            "extract",
            "--extract-relations",
            "--json",
        ],
    );
    assert_eq!(joint["candidates"][0]["model"], "glinerel-72m");
    let filtered = plan(
        dir.path(),
        &["models", "explain", "extract", "--types", "city", "--json"],
    );
    assert_eq!(filtered["planned_first_attempt"], "stacked");
}

#[test]
fn explains_contact_rules_and_honors_profile_and_cli_precedence() {
    let dir = tempdir().unwrap();
    let contact = plan(
        dir.path(),
        &[
            "models", "explain", "privacy", "--types", "CONTACT", "--json",
        ],
    );
    assert_eq!(contact["planned_first_attempt"], "contact-rules");
    fs::write(dir.path().join("saved.toml"), "model = \"pattern\"\n").unwrap();
    let saved = plan(
        dir.path(),
        &[
            "--config", "saved", "models", "explain", "privacy", "--types", "CONTACT", "--json",
        ],
    );
    assert_eq!(saved["planned_first_attempt"], "pattern");
    assert_eq!(saved["selection_source"], "profile");
    let explicit = plan(
        dir.path(),
        &[
            "--config",
            "saved",
            "models",
            "explain",
            "extract",
            "--model",
            "otter-cross",
            "--json",
        ],
    );
    assert_eq!(explicit["selection_source"], "command_line");
    assert_eq!(explicit["planned_first_attempt"], "otter-cross");
    assert_eq!(explicit["candidates"][0]["eligible"], false);
}
