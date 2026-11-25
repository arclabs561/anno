#![cfg(all(feature = "eval", feature = "onnx"))]

use serde_json::Value;
use std::{env, fs, path::Path, process::Command};

#[test]
fn offline_matrix_selects_prepared_models_and_reports_explicit_failures() {
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
                "12",
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
    // This integration executable has a single test and its own temporary Hub cache.
    env::set_var("HF_HOME", root.join("hf"));
    env::set_var("ANNO_NO_DOWNLOADS", "1");
    assert_eq!(
        anno::StackedNER::default().layer_names(),
        ["regex", "heuristic"]
    );

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

    // The public default stack must execute the prepared ML layer, too. Its
    // fallback layers can recognize names even if model discovery regresses.
    let report = anno::StackedNER::default()
        .extract_entities_with_report(
            "Call John Smith now",
            None,
            anno::StackedExtractionPolicy::Strict,
        )
        .unwrap();
    assert!(
        report.layer_outcomes().iter().any(|outcome| matches!(
            outcome, anno::StackedLayerOutcome::Succeeded { layer, entity_count }
                if layer == "GLiNER-ONNX" && *entity_count > 0
        )),
        "{:?}",
        report.layer_outcomes()
    );

    // Both strict backends were sampled on CI without their files. A graph
    // alone is insufficient. Only BERT accepts quantized Hub graphs and
    // only BERT also requires a Hub label configuration.
    let hub_snapshot = |model: &str| {
        let dir = root
            .join("hf/hub")
            .join(format!("models--{}", model.replace('/', "--")));
        fs::create_dir_all(dir.join("refs")).unwrap();
        fs::write(dir.join("refs/main"), "fixture").unwrap();
        let snapshot = dir.join("snapshots/fixture");
        fs::create_dir_all(snapshot.join("onnx")).unwrap();
        snapshot
    };
    let strict = || {
        outcomes(
            "--pin-backend",
            "heuristic,bert_onnx,nuner,gliner_multitask",
        )
    };
    let cold = strict();
    assert_eq!(cold.len(), 1);
    assert_eq!(cold[0]["backend"], "heuristic");
    let forced = outcomes("--fixed-backend", "bert_onnx,nuner");
    assert_eq!(forced.len(), 2);
    assert!(forced
        .iter()
        .all(|row| row["hard_junk"] == true && row["primary_f1"].is_null()));
    let bert = hub_snapshot(anno::DEFAULT_BERT_ONNX_MODEL);
    let nuner = hub_snapshot(anno::DEFAULT_NUNER_MODEL);
    let bert_fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../anno/tests/bughunt7_data/bert_cjk");
    fs::copy(
        bert_fixture.join("model.onnx"),
        bert.join("onnx/model_int8.onnx"),
    )
    .unwrap();
    fs::copy(
        bert_fixture.join("tokenizer.json"),
        bert.join("tokenizer.json"),
    )
    .unwrap();
    fs::copy(
        fixture.join("onnx/model.onnx"),
        nuner.join("model_quantized.onnx"),
    )
    .unwrap();
    fs::copy(fixture.join("tokenizer.json"), nuner.join("tokenizer.json")).unwrap();
    assert_eq!(strict().len(), 1);
    fs::copy(bert_fixture.join("config.json"), bert.join("config.json")).unwrap();
    fs::rename(nuner.join("model_quantized.onnx"), nuner.join("model.onnx")).unwrap();
    let warm = strict();
    assert_eq!(warm.len(), 3, "{warm:?}");
    for name in ["bert_onnx", "nuner"] {
        let row = warm.iter().find(|r| r["backend"] == name).unwrap();
        assert_eq!(row["hard_junk"], false, "{row:?}");
        assert!(row["primary_f1"].is_number(), "{row:?}");
    }

    #[cfg(feature = "candle")]
    {
        use anno::backends::{candle::CandleNER, gliner_candle::GLiNERCandle};
        let model = "fixture/candle";
        let dir = hub_snapshot(model);
        assert!(!CandleNER::hub_assets_available(model));
        fs::write(dir.join("pytorch_model.bin"), []).unwrap();
        fs::write(dir.join("vocab.txt"), []).unwrap();
        assert!(!CandleNER::hub_assets_available(model));
        fs::write(dir.join("config.json"), []).unwrap();
        assert!(CandleNER::hub_assets_available(model));
        assert!(!GLiNERCandle::hub_assets_available(model));
        fs::write(dir.join("tokenizer.json"), []).unwrap();
        assert!(GLiNERCandle::hub_assets_available(model));
        fs::remove_file(dir.join("config.json")).unwrap();
        fs::remove_file(dir.join("pytorch_model.bin")).unwrap();
        fs::write(dir.join("gliner_config.json"), []).unwrap();
        assert!(!GLiNERCandle::hub_assets_available(model));
        fs::write(dir.join("gliner_model.safetensors"), []).unwrap();
        assert!(GLiNERCandle::hub_assets_available(model));
        assert!(!CandleNER::hub_assets_available(model));
    }

    // W2NER accepts a prepared local export without a Hub access token.
    let w2ner_dir = bert_fixture.with_file_name("w2ner_tokgrid");
    let w2ner_dir = w2ner_dir.to_str().unwrap();
    use anno::backends::w2ner::W2NER;
    assert!(W2NER::model_files_available(w2ner_dir, false));
    use anno::Model;
    let entities = W2NER::from_pretrained(w2ner_dir)
        .unwrap()
        .extract_entities("John Smith works", None)
        .unwrap();
    assert!(entities.iter().any(|entity| entity.text == "John Smith"));

    // The multitask constructor takes an FP32 graph and its configuration.
    let multi = hub_snapshot(anno::DEFAULT_GLINER_MULTITASK_MODEL);
    fs::copy(fixture.join("onnx/model.onnx"), multi.join("model.onnx")).unwrap();
    fs::copy(fixture.join("tokenizer.json"), multi.join("tokenizer.json")).unwrap();
    use anno::backends::gliner_multitask::GLiNERMultitaskOnnx;
    assert!(!GLiNERMultitaskOnnx::hub_assets_available(
        anno::DEFAULT_GLINER_MULTITASK_MODEL
    ));
    fs::write(multi.join("config.json"), "{}").unwrap();
    assert!(GLiNERMultitaskOnnx::hub_assets_available(
        anno::DEFAULT_GLINER_MULTITASK_MODEL
    ));
    GLiNERMultitaskOnnx::from_pretrained(anno::DEFAULT_GLINER_MULTITASK_MODEL).unwrap();
    // When downloads are allowed, an empty Hub cache is a valid starting point.
    // This exercises the preflight alone without fetching weights.
    env::set_var("HF_HOME", root.join("empty"));
    env::remove_var("ANNO_NO_DOWNLOADS");
    assert!(anno::GLiNEROnnx::hub_assets_available(
        anno::DEFAULT_GLINER_MODEL
    ));
    assert!(W2NER::model_files_available(w2ner_dir, false));
    assert!(!W2NER::model_files_available(
        anno::DEFAULT_W2NER_MODEL,
        false
    ));
    assert!(W2NER::model_files_available(
        anno::DEFAULT_W2NER_MODEL,
        true
    ));
}
