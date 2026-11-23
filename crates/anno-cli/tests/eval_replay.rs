use serde_json::{json, Value};
use std::process::{Command, Output};

const SOURCE: &str = "📬 x@y.com now";
const GOLD: &str = "x@y.com:EMAIL:2:9";

fn cli(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_anno"))
        .args(args)
        .output()
        .unwrap()
}

fn json_output(args: &[&str]) -> Value {
    let output = cli(args);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn extract_once_then_rescore_the_same_unicode_source() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("predictions.json");
    let filename = file.to_str().unwrap();
    let mut saved = json_output(&[
        "extract", "--model", "pattern", "--text", SOURCE, "--format", "grounded",
    ]);
    std::fs::write(&file, saved.to_string()).unwrap();

    for metric in ["entity", "char-masking"] {
        let mut args = vec![
            "eval", "--text", SOURCE, "--gold", GOLD, "--json", "--metric", metric,
        ];
        let mut live = args.clone();
        live.extend(["--model", "pattern"]);
        let live = json_output(&live);
        args.extend(["--predictions", filename]);
        let replay = json_output(&args);
        for key in ["f1", "precision", "recall", "matches", "char_masking"] {
            assert_eq!(replay[key], live[key], "{key}");
        }
        assert_eq!(replay["f1"], 1.0);
        assert!(replay["model"].is_null());
        assert!(replay["elapsed_ms"].is_null());
        assert_eq!(replay["predictions"]["document_id"], "extract");
        assert_eq!(
            replay["predictions"]["source_sha256"],
            "154527a734e13c9a3e2a5735553f36eebd46c6457513317e17bae2aa5ca5434e"
        );
    }

    // A saved producer may assign nonsequential IDs.
    saved["signals"][0]["id"] = json!(43);
    std::fs::write(&file, saved.to_string()).unwrap();
    let changed_gold = json_output(&[
        "eval",
        "--text",
        SOURCE,
        "--gold",
        "x@y.com:ORG:2:9",
        "--json",
        "--predictions",
        filename,
    ]);
    assert_eq!(changed_gold["f1"], 0.0);
    assert_eq!(changed_gold["matches"][0]["pred_id"], 43);

    let rejection = |text: &str, expected: &str| {
        let run = cli(&[
            "eval",
            "--text",
            text,
            "--gold",
            GOLD,
            "--json",
            "--predictions",
            filename,
        ]);
        assert!(!run.status.success());
        assert!(run.stdout.is_empty());
        assert!(
            String::from_utf8_lossy(&run.stderr).contains(expected),
            "{}",
            String::from_utf8_lossy(&run.stderr)
        );
    };
    // Same mention and offsets, different surrounding source.
    rejection("📬 x@y.com NEW", "source text differs");
    saved["signals"][0]["surface"] = json!("z@y.com");
    std::fs::write(&file, saved.to_string()).unwrap();
    rejection(SOURCE, "signal locations or surfaces");
    saved["signals"] = json!([]);
    std::fs::write(&file, saved.to_string()).unwrap();
    let no_predictions = json_output(&[
        "eval",
        "--text",
        SOURCE,
        "--gold",
        GOLD,
        "--json",
        "--predictions",
        filename,
    ]);
    assert_eq!(no_predictions["predicted_count"], 0);
    assert_eq!(no_predictions["f1"], 0.0);
    let both_sources = cli(&[
        "eval",
        "--text",
        SOURCE,
        "--gold",
        GOLD,
        "--model",
        "pattern",
        "--predictions",
        filename,
    ]);
    assert!(!both_sources.status.success());
    assert!(String::from_utf8_lossy(&both_sources.stderr).contains("--model"));
    std::fs::write(&file, "{}").unwrap();
    rejection(SOURCE, "parse GroundedDocument JSON");
}

#[test]
fn coincident_labels_keep_exact_matches_before_type_and_boundary_diagnostics() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("predictions.json");
    let filename = file.to_str().unwrap();
    let doc = json_output(&[
        "extract", "--model", "pattern", "--text", SOURCE, "--format", "grounded",
    ]);
    let gold_org = "x@y.com:ORG:2:9";
    for gold in [[GOLD, gold_org], [gold_org, GOLD]] {
        let base = [
            "eval", "--text", SOURCE, "--gold", gold[0], "--gold", gold[1], "--json",
        ];
        std::fs::write(&file, doc.to_string()).unwrap();
        for source in [["--model", "pattern"], ["--predictions", filename]] {
            let score = json_output(&[base.as_slice(), source.as_slice()].concat());
            assert_eq!(score["correct"], 1);
            assert_eq!(score["precision"], 1.0);
            assert_eq!(score["recall"], 0.5);
            assert_eq!(score["f1"], 2.0 / 3.0);
        }

        let email = doc["signals"][0].clone();
        let mut wrong = email.clone();
        wrong["id"] = json!(7);
        wrong["label"] = json!("DATE");
        let mut partial = email.clone();
        partial["id"] = json!(11);
        partial["location"]["Text"]["end"] = json!(3);
        partial["surface"] = json!("x");
        for predictions in [
            vec![wrong.clone(), partial.clone(), email.clone()],
            vec![email.clone(), partial.clone(), wrong.clone()],
        ] {
            let mut saved = doc.clone();
            saved["signals"] = json!(predictions);
            std::fs::write(&file, saved.to_string()).unwrap();
            let score = json_output(&[base.as_slice(), &["--predictions", filename]].concat());
            let matches = score["matches"].as_array().unwrap();
            assert_eq!(matches.len(), 3);
            assert!(matches
                .iter()
                .any(|m| m["type"] == "correct" && m["pred_id"] == 0));
            assert!(matches.iter().any(|m| m["type"] == "type_mismatch"
                && m["pred_id"] == 7
                && m["gold_label"] == "ORG"));
            assert!(matches
                .iter()
                .any(|m| m["type"] == "false_positive" && m["pred_id"] == 11));
        }
    }
}

#[test]
fn gold_records_bind_to_the_evaluated_source_and_saved_document_when_identified() {
    let dir = tempfile::tempdir().unwrap();
    let predictions = dir.path().join("predictions.json");
    let gold = dir.path().join("gold.jsonl");
    let mut doc = json_output(&[
        "extract", "--model", "pattern", "--text", SOURCE, "--format", "grounded",
    ]);
    doc["id"] = json!("revision-a");
    std::fs::write(&predictions, doc.to_string()).unwrap();
    let args = [
        "eval",
        "--text",
        SOURCE,
        "--predictions",
        predictions.to_str().unwrap(),
        "--gold-file",
        gold.to_str().unwrap(),
        "--json",
    ];
    let valid = json!({"id":"revision-a", "text":SOURCE,
        "source_sha256":"154527a734e13c9a3e2a5735553f36eebd46c6457513317e17bae2aa5ca5434e",
        "entities":[{"text":"x@y.com","label":"EMAIL","start":2,"end":9}]});
    std::fs::write(&gold, valid.to_string()).unwrap();
    assert_eq!(json_output(&args)["f1"], 1.0);
    for (field, value, error) in [
        ("text", "📬 x@y.com NEW", "gold source text differs"),
        ("id", "revision-b", "Gold document ID differs"),
        ("source_sha256", "incorrect", "gold source_sha256 differs"),
    ] {
        let mut wrong = valid.clone();
        wrong[field] = json!(value);
        std::fs::write(&gold, wrong.to_string()).unwrap();
        let result = cli(&args);
        assert!(!result.status.success());
        assert!(result.stdout.is_empty());
        assert!(String::from_utf8_lossy(&result.stderr).contains(error));
    }
    let other = json!({"document_id":"revision-b","text":SOURCE,"entities":[]});
    std::fs::write(&gold, format!("{valid}\n{other}")).unwrap();
    let result = cli(&args);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("conflicting gold document IDs"));
    // String and integer document IDs are common in existing JSONL corpora.
    doc["id"] = json!("42");
    std::fs::write(&predictions, doc.to_string()).unwrap();
    let mut numbered = valid;
    numbered["id"] = json!(42);
    numbered["doc_id"] = json!("42");
    std::fs::write(&gold, numbered.to_string()).unwrap();
    assert_eq!(json_output(&args)["f1"], 1.0);
}

#[test]
fn replay_score_uses_only_covered_characters_for_segmented_mentions() {
    let dir = tempfile::tempdir().unwrap();
    let prediction_file = dir.path().join("prediction.json");
    let gold_file = dir.path().join("gold.jsonl");
    let text = "😀A B C"; // A, B, C begin at scalar indices 1, 3, 5.
    let mut doc = json_output(&[
        "extract", "--model", "pattern", "--text", SOURCE, "--format", "grounded",
    ]);
    doc["text"] = json!(text);
    doc["signals"][0]["location"] = json!({"Discontinuous":{"segments":[[1,2],[5,6]]}});
    doc["signals"][0]["surface"] = json!("A C");
    doc["signals"][0]["label"] = json!("PER");
    std::fs::write(&prediction_file, doc.to_string()).unwrap();
    let score = |entity: Value| {
        let gold = json!({"id":"extract","text":text,"entities":[entity]});
        std::fs::write(&gold_file, gold.to_string()).unwrap();
        cli(&[
            "eval",
            "--text",
            text,
            "--gold-file",
            gold_file.to_str().unwrap(),
            "--predictions",
            prediction_file.to_str().unwrap(),
            "--metric",
            "char-masking",
            "--json",
        ])
    };
    let entity = |surface: &str, start: usize, end: usize, ranges: &[(usize, usize)]| {
        let mut e = json!({"label":"PER","text":surface,"start":start,"end":end});
        if !ranges.is_empty() {
            e["segments"] = json!(ranges
                .iter()
                .map(|(s, e)| json!({"start":s,"end":e}))
                .collect::<Vec<_>>());
        }
        e
    };
    for (gold, correct, iou, tp) in [
        (entity("A C", 1, 6, &[(1, 2), (5, 6)]), 1, None, 2),
        (entity("A B C", 1, 6, &[(1, 2), (5, 6)]), 1, None, 2),
        (entity("A B C", 1, 6, &[]), 0, Some(2.0 / 5.0), 2),
        (
            entity("A B C", 1, 6, &[(1, 2), (3, 4), (5, 6)]),
            0,
            Some(2.0 / 3.0),
            2,
        ),
        (entity("B", 3, 4, &[]), 0, None, 0),
    ] {
        let output = score(gold);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let value: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(value["correct"], correct);
        assert_eq!(value["char_masking"]["strict_span_f1"], correct as f64);
        assert_eq!(value["char_masking"]["counts"]["true_positive_chars"], tp);
        assert_eq!(value["char_masking"]["counts"]["predicted_chars"], 2);
        if let Some(expected) = iou {
            assert_eq!(value["matches"][0]["type"], "boundary_error");
            assert!((value["matches"][0]["iou"].as_f64().unwrap() - expected).abs() < 1e-12);
        } else if correct == 0 {
            assert_eq!(value["matches"].as_array().unwrap().len(), 2);
        }
    }
    // BRAT newline anchors count as zero coverage. A null field keeps the legacy contiguous form.
    let null = {
        let mut row = entity("B", 3, 4, &[]);
        row["segments"] = Value::Null;
        row
    };
    assert!(score(null).status.success());
    assert!(score(entity("A", 1, 2, &[(1, 2)])).status.success());
    let anchored = score(entity(" A C ", 0, 6, &[(0, 0), (1, 2), (5, 6), (6, 6)]));
    assert!(anchored.status.success());
    let anchored: Value = serde_json::from_slice(&anchored.stdout).unwrap();
    assert_eq!(anchored["correct"], 1);
    assert_eq!(anchored["char_masking"]["counts"]["gold_chars"], 2);
    assert!(score(entity("A C", 1, 6, &[(1, 2), (5, 6)]))
        .status
        .success());
    let html = cli(&[
        "eval",
        "--text",
        text,
        "--gold-file",
        gold_file.to_str().unwrap(),
        "--predictions",
        prediction_file.to_str().unwrap(),
        "--html",
    ]);
    assert!(html.status.success());
    let html = String::from_utf8(html.stdout).unwrap();
    assert!(html.contains(">A</span> B <span"), "gap was highlighted");
    assert!(html.contains("data-eids=\"P0\""));
    #[cfg(feature = "eval")]
    {
        let mut wrong = entity("A B C", 1, 6, &[]);
        wrong["label"] = json!("ORG");
        assert!(score(wrong).status.success());
        let detailed = cli(&[
            "eval",
            "--text",
            text,
            "--gold-file",
            gold_file.to_str().unwrap(),
            "--predictions",
            prediction_file.to_str().unwrap(),
            "--verbose",
        ]);
        assert!(detailed.status.success());
        assert!(String::from_utf8(detailed.stdout)
            .unwrap()
            .contains("BoundaryAndType: 1"));
    }
    for invalid in [
        entity("A C", 1, 6, &[(1, 2), (4, 6)]),
        entity("A C", 1, 5, &[(1, 2), (5, 6)]),
        entity("A C", 1, 6, &[(1, 4), (3, 6)]),
        entity(" ", 1, 6, &[(1, 1), (6, 6)]),
    ] {
        let rejected = score(invalid);
        assert!(!rejected.status.success());
        assert!(rejected.stdout.is_empty());
    }
}
