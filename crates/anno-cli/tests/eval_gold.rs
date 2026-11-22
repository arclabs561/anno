use std::process::{Command, Output};

fn evaluate(options: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_anno"))
        .args([
            "eval",
            "--model",
            "pattern",
            "--text",
            "🌍 Email x@y.com",
            "--json",
        ])
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
