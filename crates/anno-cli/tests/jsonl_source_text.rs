use serde_json::{json, Value};
use std::{fs, path::Path, process::Command};

#[test]
fn checked_mention_import_preserves_labels_and_validates_original_character_ranges() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("post.txt");
    let input = dir.path().join("mentions.jsonl");
    let output = dir.path().join("verified.jsonl");
    fs::write(&source, "🦊 Zoë\r\nZoë e\u{301} met Li.").unwrap();
    let run = |path: &Path, checked: bool| {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_anno"));
        cmd.args(["dataset", "import", "--format", "jsonl", "--input"])
            .arg(&input)
            .arg("--output")
            .arg(path);
        if checked {
            cmd.arg("--source-text").arg(&source);
        }
        cmd.output().unwrap()
    };
    let cases = [
        json!({"text":"Zoë","type":"performer","start":7,"end":10,"source":"producer","confidence":0.75}),
        json!({"text":"Zoë","type":"PER","start":7,"end":10}),
        json!({"text":"e\u{301}","type":"custom","start":11,"end":13}),
        json!({"text":"Zoë Li","type":"PER","start":7,"end":20,"segments":[{"start":7,"end":10},{"start":18,"end":20}]}),
        json!({"text":"Zoë\r\nZoë","type":"PER","start":2,"end":10,"segments":[{"start":2,"end":5},{"start":7,"end":10}]}),
        json!({"text":"\r\n","type":"line_break","start":5,"end":7,"segments":[{"start":5,"end":5},{"start":7,"end":7}],"confidence":null}),
    ];
    let lines = cases.map(|v| v.to_string()).join("\n");
    fs::write(&input, &lines).unwrap();
    let ok = run(&output, true);
    assert!(
        ok.status.success(),
        "{}",
        String::from_utf8_lossy(&ok.stderr)
    );
    let verified = fs::read_to_string(&output).unwrap();
    let actual: Vec<Value> = verified
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    assert_eq!(actual.len(), 6);
    assert_eq!(actual[0]["type"], "performer");
    assert_eq!(actual[0]["source"], "producer");
    assert_eq!(actual[0]["confidence"], 0.75);
    assert_eq!(actual[1]["type"], "PER");
    assert!(run(&output, false).status.success());
    assert_eq!(fs::read_to_string(&output).unwrap(), verified);

    for bad in [
        json!({"text":"Ana","type":"PER","start":7,"end":10}),
        json!({"text":" ","type":"PER","start":7,"end":10,"segments":[{"start":7,"end":7},{"start":10,"end":10}]}),
        json!({"text":"Zoë","type":"PER","start":7,"end":10,"source":{"producer":"model"}}),
        json!({"text":"Zoë","type":"PER","start":7,"end":10,"confidence":[0.5]}),
        json!({"text":"é","type":"custom","start":11,"end":13}),
        json!({"text":"Li","type":"ORG","start":18,"end":99}),
        json!({"text":"","type":"PER","start":7,"end":7}),
        json!({"text":"🦊","type":"PER","end":1}),
        json!({"text":"Zoë","type":"   ","start":7,"end":10}),
        json!({"text":"Zoë e\u{301} met Li","type":"PER","start":7,"end":20,"segments":[{"start":7,"end":10},{"start":18,"end":20}]}),
    ] {
        fs::write(&input, format!("{}\n{bad}", actual[0])).unwrap();
        let failed = run(&output, true);
        assert!(!failed.status.success(), "accepted {bad}");
        assert!(String::from_utf8_lossy(&failed.stderr).contains("JSONL line 2"));
        assert_eq!(fs::read_to_string(&output).unwrap(), verified);
    }
    fs::write(&input, "").unwrap();
    let empty = run(&output, true);
    assert!(empty.status.success());
    assert!(String::from_utf8_lossy(&empty.stderr).contains("Imported 0 annotations from 1 files"));
    assert_eq!(fs::read(&output).unwrap(), b"");
    assert!(!run(&source, true).status.success());
}

#[test]
fn grounded_export_round_trips_discontinuous_surfaces_through_checked_jsonl() {
    let dir = tempfile::tempdir().unwrap();
    let text = "🦊A B C\r\nD";
    let original = dir.path().join("original.txt");
    let grounded = dir.path().join("doc.json");
    let exported = dir.path().join("export");
    let checked = dir.path().join("checked.jsonl");
    fs::write(&original, text).unwrap();
    let mut doc = anno::GroundedDocument::new("example", text);
    let lexical = anno::Location::Discontinuous {
        segments: vec![(1, 2), (5, 6)],
    };
    let newline = anno::Location::Discontinuous {
        segments: vec![(5, 6), (8, 9)],
    };
    for (surface, location) in [
        ("A C", lexical.clone()),
        ("A B C", lexical),
        ("C\r\nD", newline),
        ("B", anno::Location::text(3, 4)),
    ] {
        doc.add_signal(anno::Signal::new(0, location, surface, "PER", 1.0));
    }
    assert!(doc.validate().is_empty());
    fs::write(&grounded, serde_json::to_vec(&doc).unwrap()).unwrap();
    let run = Command::new(env!("CARGO_BIN_EXE_anno"))
        .args([
            "dataset", "export", "--model", "pattern", "--format", "jsonl", "--input",
        ])
        .arg(&grounded)
        .arg("--output")
        .arg(&exported)
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let run = Command::new(env!("CARGO_BIN_EXE_anno"))
        .args(["dataset", "import", "--format", "jsonl", "--input"])
        .arg(exported.join("doc.jsonl"))
        .arg("--source-text")
        .arg(&original)
        .arg("--output")
        .arg(&checked)
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let rows: Vec<Value> = fs::read_to_string(checked)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 4);
    for row in &rows[..2] {
        assert_eq!(row["text"], "A C");
        assert_eq!(
            row["segments"],
            json!([{"start":1,"end":2},{"start":5,"end":6}])
        );
    }
    assert_eq!(rows[2]["text"], "C\r\nD");
    assert_eq!(
        rows[2]["segments"],
        json!([{"start":5,"end":6},{"start":8,"end":9}])
    );
    assert_eq!(rows[3]["text"], "B");
    assert!(rows[3].get("segments").is_none());
}
