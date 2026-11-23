use std::process::{Command, Output};

fn evaluate(options: &[&str]) -> Output {
    evaluate_text("🌍 Email x@y.com", options)
}

fn evaluate_text(text: &str, options: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_anno"))
        .args(["eval", "--model", "pattern", "--text", text, "--json"])
        .args(options)
        .output()
        .unwrap()
}

#[test]
fn mixed_inline_gold_rejects_the_entire_evaluation() {
    let valid = "x@y.com:EMAIL:8:15";
    let run = evaluate(&["--gold", valid]);
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let score: serde_json::Value = serde_json::from_slice(&run.stdout).unwrap();
    assert_eq!(score["gold_count"], 1);
    assert_eq!(score["f1"], 1.0);
    for (bad, error) in [
        ("malformed", "Invalid --gold #2"),
        ("x@y.com:EMAIL:8:20", "Invalid gold annotation #2"),
        ("z@y.com:EMAIL:8:15", "surface text does not match"),
    ] {
        let run = evaluate(&["--gold", valid, "--gold", bad]);
        assert!(!run.status.success());
        assert!(run.stdout.is_empty());
        assert!(String::from_utf8_lossy(&run.stderr).contains(error));
    }
}

#[test]
fn mixed_file_gold_identifies_malformed_records() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("gold.jsonl");
    let first = r#"{"entities":[{"text":"x@y.com","type":"EMAIL","start":8,"end":15}]}"#;
    std::fs::write(&path, first).unwrap();
    let run = evaluate(&["--gold-file", path.to_str().unwrap()]);
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let score: serde_json::Value = serde_json::from_slice(&run.stdout).unwrap();
    assert_eq!(score["gold_count"], 1);
    assert_eq!(score["f1"], 1.0);
    let conflicting = evaluate(&["--gold-file", path.to_str().unwrap(), "--gold", "malformed"]);
    assert!(!conflicting.status.success());
    assert!(conflicting.stdout.is_empty());
    for (second, error) in [
        (r#"{}"#, "gold.jsonl:2: expected an entities array"),
        (
            r#"{"entities":[{"text":"x","label":"EMAIL","end":9}]}"#,
            "gold.jsonl:2: entity[0] has missing or invalid start",
        ),
    ] {
        std::fs::write(&path, format!("{first}\n{second}")).unwrap();
        let run = evaluate(&["--gold-file", path.to_str().unwrap()]);
        assert!(!run.status.success());
        assert!(run.stdout.is_empty());
        assert!(String::from_utf8_lossy(&run.stderr).contains(error));
    }
}

#[test]
fn annotated_negatives_count_spurious_predictions_and_masked_characters() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("negative.jsonl");
    for (text, expected_spans, expected_chars) in [("🌍 Email x@y.com", 1, 7), ("quiet", 0, 0)] {
        let gold = serde_json::json!({"id": "negative", "text": text, "entities": []});
        std::fs::write(&path, gold.to_string()).unwrap();
        let run = evaluate_text(
            text,
            &[
                "--gold-file",
                path.to_str().unwrap(),
                "--metric",
                "char-masking",
            ],
        );
        assert!(
            run.status.success(),
            "{}",
            String::from_utf8_lossy(&run.stderr)
        );
        let score: serde_json::Value = serde_json::from_slice(&run.stdout).unwrap();
        assert_eq!(score["gold_count"], 0);
        assert_eq!(score["predicted_count"], expected_spans);
        assert_eq!(score["errors"], expected_spans);
        let matches = score["matches"].as_array().unwrap();
        assert_eq!(matches.len(), expected_spans);
        assert!(matches.iter().all(|m| m["type"] == "false_positive"));
        let masking = &score["char_masking"];
        assert_eq!(masking["counts"]["true_positive_chars"], 0);
        assert_eq!(masking["counts"]["predicted_chars"], expected_chars);
        assert_eq!(masking["counts"]["non_pii_chars"], text.chars().count());
        assert_eq!(
            masking["false_positive_rate"].as_f64().unwrap(),
            expected_chars as f64 / text.chars().count() as f64
        );
    }
}

#[test]
fn an_empty_gold_file_is_not_a_labeled_document() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("blank.jsonl");
    std::fs::write(&path, "\n  \n").unwrap();
    let run = evaluate(&["--gold-file", path.to_str().unwrap()]);
    assert!(!run.status.success());
    assert!(run.stdout.is_empty());
    assert!(String::from_utf8_lossy(&run.stderr).contains("expected a gold document"));
}
