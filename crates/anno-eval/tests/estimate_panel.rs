#![cfg(feature = "eval")]

use serde_json::Value;
use std::{fs, process::Command};

#[test]
fn estimate_uses_final_datasets_for_eligibility_and_history() {
    for (extra, dataset, domain) in [
        (&[][..], "CoNLL2003Sample", "news"),
        (
            &["--pin-domain", "social_media"][..],
            "Wnut17",
            "social_media",
        ),
        (
            &["--fixed-backend", "heuristic", "--datasets-per-run", "2"][..],
            "CoNLL2003Sample",
            "news",
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let datasets = dir.path().join("datasets");
        fs::create_dir(&datasets).unwrap();
        // Both loaders accept CoNLL; WNUT's label schema excludes the heuristic arm.
        fs::write(datasets.join("Wnut17.cache"), "Paris B-location\n. O\n").unwrap();
        fs::write(datasets.join("CoNLL2003Sample.cache"), "Paris B-LOC\n. O\n").unwrap();
        let log = dir.path().join("decisions.jsonl");
        let pools = dir.path().join("pools.jsonl");
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_muxer_repeat"));
        for (key, _) in std::env::vars_os() {
            if key.to_string_lossy().starts_with("ANNO_") {
                cmd.env_remove(key);
            }
        }
        let result = cmd
            .current_dir(dir.path())
            .env("ANNO_CACHE_DIR", dir.path())
            .env("ANNO_MATRIX_DATASET_POOLS_FILE", &pools)
            .env("ANNO_ML_IN_MATRIX", "0")
            .env("ANNO_NO_DOWNLOADS", "1")
            .env("ANNO_S3_CACHE", "0")
            .env("ANNO_MATRIX_INCLUDE_NON_AUTOMATABLE", "1")
            .env("ANNO_SAMPLE_STRATEGY", "estimate")
            .args([
                "--task",
                "ner",
                "--runs",
                "1",
                "--seed-base",
                "3",
                "--datasets-per-run",
                "1",
                "--backends-per-run",
                "1",
                "--require-cached",
                "--no-agg",
                "--pin-backend",
                "heuristic,stacked",
            ])
            .args(extra)
            .arg("--log")
            .arg(&log)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let rows: Vec<Value> = fs::read_to_string(&log)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let receipt: Value = serde_json::from_str(&fs::read_to_string(&pools).unwrap()).unwrap();
        let pool = &receipt["pool"];
        assert_eq!(receipt["pass"], "initial");
        assert_eq!(receipt["require_cached"], true);
        assert_eq!(pool["selected"], serde_json::json!([dataset]));
        assert_eq!(
            pool["eligible_count"],
            if domain == "social_media" { 1 } else { 2 }
        );
        if !extra.is_empty() && extra[0] == "--fixed-backend" {
            assert_eq!(pool["backend_incompatible_count"], 1);
        }
        let decision = &rows[0];
        assert_eq!(decision["datasets"], serde_json::json!([dataset]));
        let arms = decision["remaining"].as_array().unwrap();
        assert_eq!(arms.iter().any(|b| b == "heuristic"), domain == "news");
        let slice = format!("ner.lang=en.dom={domain}");
        assert_eq!(decision["slice"], slice);
        let outcome = rows.iter().find(|r| r["record_type"] == "outcome").unwrap();
        assert_eq!(outcome["slice"], slice);
        assert!(dir
            .path()
            .join(format!("muxer_history.{slice}.json"))
            .is_file());
    }
}
