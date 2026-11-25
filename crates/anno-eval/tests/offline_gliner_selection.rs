#![cfg(all(feature = "eval", feature = "onnx"))]

use serde_json::Value;
use std::{env, fs, path::Path, process::Command};

#[test]
fn offline_matrix_keeps_prepared_gliner_models_and_reports_explicit_failures() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let repo = root.join("hf/hub/models--onnx-community--gliner_small-v2.1");
    let snapshot = repo.join("snapshots/fixture");
    fs::create_dir_all(repo.join("refs")).unwrap();
    fs::create_dir_all(snapshot.join("onnx")).unwrap();
    fs::write(repo.join("refs/main"), "fixture").unwrap();
    fs::create_dir_all(root.join("datasets")).unwrap();
    fs::write(
        root.join("datasets/AnnoNerSmoke.cache"),
        include_bytes!("../../../scripts/qa/fixtures/matrix-smoke.conll"),
    )
    .unwrap();

    let outcomes = |option: &str, backends: &str| {
        let log = root.join("decisions.jsonl");
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_muxer_repeat"));
        for (key, _) in env::vars_os() {
            if key.to_string_lossy().starts_with("ANNO_") {
                cmd.env_remove(key);
            }
        }
        let output = cmd
            .current_dir(root)
            .env("HF_HOME", root.join("hf"))
            .env("ANNO_CACHE_DIR", root)
            .env("ANNO_NO_DOWNLOADS", "1")
            .env("ANNO_ML_IN_MATRIX", "1")
            .env("ANNO_SAMPLE_STRATEGY", "random")
            .args([
                "--task",
                "ner",
                "--fixed-datasets",
                "AnnoNerSmoke",
                option,
                backends,
                "--runs",
                "1",
                "--seed-base",
                "42",
                "--backends-per-run",
                "3",
                "--max-examples",
                "1",
                "--require-cached",
                "--no-agg",
                "--log",
            ])
            .arg(&log)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        fs::read_to_string(log)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .filter(|r: &Value| r["record_type"] == "outcome")
            .collect::<Vec<_>>()
    };
    let automatic = || outcomes("--pin-backend", "heuristic,gliner_onnx,gliner_pii");
    let cold = automatic();
    assert_eq!(cold.len(), 1);
    assert_eq!(cold[0]["backend"], "heuristic");

    let fixed = outcomes("--fixed-backend", "gliner_onnx");
    assert_eq!(fixed.len(), 1);
    assert_eq!(fixed[0]["hard_junk"], true);
    assert!(fixed[0]["primary_f1"].is_null());

    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../anno/tests/bughunt7_data/hf_home/hub/models--bh7--gliner-span/snapshots/fixture");
    fs::copy(
        fixture.join("onnx/model.onnx"),
        snapshot.join("onnx/model_quantized.onnx"),
    )
    .unwrap();
    assert_eq!(automatic().len(), 1);
    for name in ["gliner_config.json", "tokenizer.json"] {
        fs::copy(fixture.join(name), snapshot.join(name)).unwrap();
    }
    let warm = automatic();
    assert_eq!(warm.len(), 2);
    let gliner = warm.iter().find(|r| r["backend"] == "gliner_onnx").unwrap();
    assert_eq!(gliner["hard_junk"], false);
    assert!(gliner["primary_f1"].is_number());

    // When downloads are allowed, an empty Hub cache is a valid starting point.
    // This exercises the preflight alone without fetching weights.
    env::set_var("HF_HOME", root.join("empty"));
    env::remove_var("ANNO_NO_DOWNLOADS");
    assert!(anno::GLiNEROnnx::hub_assets_available(
        anno::DEFAULT_GLINER_MODEL
    ));
}
