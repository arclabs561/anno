#![cfg(feature = "eval")]

use std::{fs, process::Command};

#[test]
fn raw_and_unknown_caches_are_checked_locally_before_any_s3_call() {
    let temp = tempfile::tempdir().unwrap();
    let datasets = temp.path().join("datasets");
    fs::create_dir(&datasets).unwrap();
    let entries = ["WikiGold.cache", "unregistered.cache"]
        .map(|name| {
            (
                name,
                serde_json::json!({
                    "dataset_id": name, "source_url": "https://example.org/source",
                    "sha256": "a".repeat(64), "file_size": 0, "downloaded_at": "",
                    "sentence_count": 0, "entity_count": 0, "anno_version": "0.1"
                }),
            )
        })
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();
    let manifest = serde_json::json!({ "version": 1, "entries": entries });
    fs::write(datasets.join("manifest.json"), manifest.to_string()).unwrap();

    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_anno"))
            .args(["cache", "sync-s3"])
            .args(args)
            .env("ANNO_CACHE_DIR", temp.path())
            .env("ANNO_S3_CACHE", "1")
            .env("ANNO_S3_BUCKET", "")
            .env("PATH", temp.path()) // No AWS executable is available.
            .output()
            .unwrap()
    };
    for args in [vec!["--dry-run"], vec!["--bucket", "fixture"]] {
        let result = run(&args);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(String::from_utf8_lossy(&result.stdout).contains("No eligible"));
        let skipped = String::from_utf8_lossy(&result.stderr);
        assert!(skipped.contains("Skip WikiGold.cache"));
        assert!(skipped.contains("Skip unregistered.cache"));
    }
    let missing = run(&[]);
    assert!(!missing.status.success());
    assert!(String::from_utf8_lossy(&missing.stderr).contains("ANNO_S3_BUCKET"));
}
