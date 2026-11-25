// SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(feature = "eval")]

use serde_json::Value;
use std::{collections::BTreeSet, fs, path::Path, process::Command};

fn rows(path: &Path) -> Vec<String> {
    let receipt: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    let rows = receipt["results"].as_array().unwrap();
    assert_eq!(receipt["summary"]["total_combinations"], rows.len());
    rows.iter()
        .map(|r| serde_json::to_string(&(&r["task"], &r["dataset"], &r["backend"])).unwrap())
        .collect()
}

#[test]
fn isolated_backends_accept_default_tasks_and_publish_combined_outputs() {
    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path();
    fs::write(
        dir.join("schema.json"),
        r#"{"entity_types":["person"],"relation_types":["knows"]}"#,
    )
    .unwrap();
    let run = Command::new(env!("CARGO_BIN_EXE_anno"))
        .current_dir(dir)
        .args([
            "benchmark",
            "--backends",
            "gliner_onnx,gliner_multitask",
            "--datasets",
            "CrossREAI",
            "--relation-schema",
            "schema.json",
            "--cached-only",
            "--max-examples",
            "1",
            "--output-json",
            "combined.json",
            "--output",
            "combined.md",
        ])
        .env("ANNO_CACHE_DIR", dir.join("datasets"))
        .env("HF_HOME", dir.join("hub"))
        .env("ANNO_S3_CACHE", "0")
        .env("ANNO_NO_DOWNLOADS", "1")
        .env("HF_HUB_OFFLINE", "1")
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let children: BTreeSet<_> = ["gliner_onnx", "gliner_multitask"]
        .into_iter()
        .flat_map(|backend| rows(&dir.join(format!("combined-{backend}.json"))))
        .collect();
    assert!(!children.is_empty());
    let combined = rows(&dir.join("combined.json"));
    assert_eq!(combined.len(), children.len());
    assert_eq!(combined.into_iter().collect::<BTreeSet<_>>(), children);
    assert!(fs::metadata(dir.join("combined.md")).unwrap().len() > 0);
}
