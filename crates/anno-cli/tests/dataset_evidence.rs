use serde_json::{json, Value};
use std::fs;
use std::path::Path;
use std::process::{Command, Output};

fn evidence(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_anno"))
        .current_dir(dir)
        .env_remove("ANNO_FIGURE_CATALOG")
        .env("ANNO_NO_DOWNLOADS", "1")
        .args(["dataset", "evidence"])
        .args(args)
        .output()
        .unwrap()
}

fn json_output(output: Output) -> Value {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn dataset(id: &str, records: Vec<Value>) -> Value {
    json!({"id": id, "name": id, "domain": "news", "measurement": "Other corpus counts",
        "coverage_action": "Inspect measured scope", "declared_tasks": ["ner"],
        "declared_splits": ["train", "test"], "records": records})
}

#[test]
fn checkout_and_unpacked_catalogs_connect_recorded_inputs_to_studies() {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("project/subdir");
    let published = root.path().join("project/docs/eval");
    fs::create_dir_all(&dir).unwrap();
    fs::create_dir_all(&published).unwrap();
    let missing = evidence(&dir, &[]);
    assert!(!missing.status.success());
    assert!(String::from_utf8_lossy(&missing.stderr).contains("--catalog"));
    let measured =
        json!({"task":"ner", "backend":"fixture", "status":"Scored", "source":"receipt-1"});
    let catalog = json!({"schema":"anno.figure-catalog/v1", "ci_survey_day":"2026-11-27",
        "datasets":[dataset("WikiGold", vec![measured]),dataset("ENer",vec![]),dataset("ENER",vec![])],
        "studies":[{"id":"wikigold-routing-paired","datasets":["WikiGold"],"guide":"qa.md","report":{}}]});
    fs::write(published.join("figure-catalog.json"), catalog.to_string()).unwrap();
    let result = json_output(evidence(
        &dir,
        &["WikiGold", "--status", "scored", "--json"],
    ));
    assert_eq!(result["matched_ids"], 1);
    let wiki = &result["datasets"][0];
    assert_eq!(wiki["id"], "WikiGold");
    assert!(wiki["studies"]
        .as_array()
        .unwrap()
        .contains(&json!("wikigold-routing-paired")));
    let records = wiki["records"].as_array().unwrap();
    assert!(!records.is_empty());
    assert!(records
        .iter()
        .all(|r| r["status"] == "Scored" && r["source"].as_str().is_some_and(|s| !s.is_empty())));
    let result = json_output(evidence(&dir, &["--status", "unindexed", "--json"]));
    assert!(result["matched_ids"].as_u64().unwrap() > 0);
    assert!(result["datasets"]
        .as_array()
        .unwrap()
        .iter()
        .all(|d| d["records"] == json!([])));
    for id in ["ENer", "ENER"] {
        let row = json_output(evidence(&dir, &[id, "--json"]));
        assert_eq!(row["matched_ids"], 1);
        assert_eq!(row["datasets"][0]["id"], id);
    }
}

#[test]
fn filters_must_agree_on_one_observation_and_keep_missing_evidence_distinct() {
    let dir = tempfile::tempdir().unwrap();
    let record = |backend: &str, split: &str| {
        json!({
            "task": "ner", "backend": backend, "reported_backend": format!("actual-{backend}"), "split": split,
            "status": "Scored", "source": "test-receipt", "examples": 2, "unit": "sentences"
        })
    };
    let catalog = json!({"schema":"anno.figure-catalog/v1","ci_survey_day":"2026-11-27","studies":[],
        "datasets":[dataset("A", vec![record("alpha", "test"),record("beta","train")]),dataset("B", vec![])]});
    fs::write(dir.path().join("evidence.json"), catalog.to_string()).unwrap();
    for (backend, split, expected) in [("alpha", "train", 0), ("actual-alpha", "test", 1)] {
        let output = json_output(evidence(
            dir.path(),
            &[
                "--catalog",
                "evidence.json",
                "--backend",
                backend,
                "--split",
                split,
                "--json",
            ],
        ));
        assert_eq!(output["matched_ids"], expected);
        if expected == 1 {
            assert_eq!(output["datasets"][0]["records"][0]["examples"], 2);
        }
    }
    let empty = json_output(evidence(
        dir.path(),
        &[
            "--catalog",
            "evidence.json",
            "--status",
            "unindexed",
            "--task",
            "ner",
            "--json",
        ],
    ));
    assert_eq!(empty["matched_ids"], 0);
}
