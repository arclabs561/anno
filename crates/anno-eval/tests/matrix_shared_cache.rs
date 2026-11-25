#![cfg(all(feature = "eval", unix))]

use serde_json::Value;
use std::{env, fs, os::unix::fs::PermissionsExt, path::Path, process::Command};

#[test]
fn shared_cache_staging_is_opt_in_read_only_and_usable_by_task_readers() {
    let root = tempfile::tempdir().unwrap();
    let aws = root.path().join("aws");
    fs::write(
        &aws,
        r#"#!/bin/sh
echo "$*" >> "$CALLS"
if [ "$1 $2 $4" != "s3 cp -" ]; then exit 2; fi
case "$3" in
    s3://fixture/datasets/WikiGold.cache) echo 'Mira B-PER';;
    s3://fixture/datasets/GAP.cache) cat "$GAP_INPUT";;
    *) echo '(NoSuchKey)' >&2; exit 1;;
esac
"#,
    )
    .unwrap();
    fs::set_permissions(&aws, fs::Permissions::from_mode(0o755)).unwrap();
    let gap = root.path().join("source.tsv");
    fs::write(&gap, "ID\tText\tPronoun\tPronoun-offset\tA\tA-offset\tA-coref\tB\tB-offset\tB-coref\n1\tAlice saw Bob and she waved\tshe\t18\tAlice\t0\tTRUE\tBob\t10\tFALSE\n").unwrap();
    let path = env::join_paths(
        std::iter::once(root.path().to_path_buf())
            .chain(env::split_paths(&env::var_os("PATH").unwrap())),
    )
    .unwrap();
    for (name, task, backend, staging, offline, selected) in [
        ("local", "ner", "heuristic", "0", "", 1),
        ("offline", "ner", "heuristic", "1", "ANNO_NO_DOWNLOADS", 1),
        ("hub-offline", "ner", "heuristic", "1", "HF_HUB_OFFLINE", 2),
        (
            "empty-offline",
            "ner",
            "heuristic",
            "1",
            "ANNO_NO_DOWNLOADS",
            0,
        ),
        ("shared", "ner", "heuristic", "1", "", 2),
        ("coreference", "intra-coref", "coref_resolver", "1", "", 1),
    ] {
        let cache = root.path().join(name);
        let datasets = cache.join("datasets");
        fs::create_dir_all(&datasets).unwrap();
        if name != "empty-offline" {
            fs::copy(
                Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/matrix-smoke.conll"),
                datasets.join("AnnoNerSmoke.cache"),
            )
            .unwrap();
        }
        let calls = cache.join("calls");
        fs::write(&calls, "").unwrap();
        let log = cache.join("decisions.jsonl");
        let pools = cache.join("pools.jsonl");
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_muxer_repeat"));
        for key in env::vars_os().map(|(k, _)| k) {
            if key.to_string_lossy().starts_with("ANNO_") || key == "HF_HUB_OFFLINE" {
                cmd.env_remove(key);
            }
        }
        if !offline.is_empty() {
            cmd.env(offline, "1");
        }
        if name == "offline" {
            cmd.env("ANNO_MATRIX_REQUIRE_CACHED", "0");
        } else if name != "empty-offline" {
            cmd.arg("--require-cached");
        }
        let result = cmd
            .env_remove("GITHUB_ACTIONS") // Exercise the ordinary local default in CI as well.
            .current_dir(&cache)
            .env("PATH", &path)
            .env("CALLS", &calls)
            .env("GAP_INPUT", &gap)
            .env("ANNO_CACHE_DIR", &cache)
            .env("ANNO_S3_CACHE", "1")
            .env("ANNO_S3_BUCKET", "fixture")
            .env("ANNO_MATRIX_STAGE_FROM_S3", staging)
            .env("ANNO_MATRIX_DATASET_POOLS_FILE", &pools)
            .env("ANNO_SAMPLE_STRATEGY", "random")
            .env(
                "ANNO_MATRIX_TRY_DOWNLOAD_ON_EMPTY",
                if name == "empty-offline" { "1" } else { "0" },
            )
            .args([
                "--task",
                task,
                "--pin-backend",
                backend,
                "--runs",
                "1",
                "--seed-base",
                "729",
                "--max-examples",
                "1",
                "--datasets-per-run",
                "2",
                "--backends-per-run",
                "1",
                "--no-agg",
                "--log",
            ])
            .arg(&log)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let pool: Value = serde_json::from_str(&fs::read_to_string(&pools).unwrap()).unwrap();
        let contents = if selected == 0 && !log.exists() {
            String::new()
        } else {
            fs::read_to_string(&log).unwrap()
        };
        let rows: Vec<Value> = contents
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .filter(|r: &Value| r["record_type"] == "outcome")
            .collect();
        assert_eq!(
            pool["pool"]["selected"].as_array().unwrap().len(),
            selected,
            "{name}"
        );
        assert_eq!(pool["require_cached"], true);
        assert_eq!(
            pool["stage_from_s3"],
            staging == "1" && offline != "ANNO_NO_DOWNLOADS"
        );
        assert_eq!(pool["target_datasets"], 2);
        assert_eq!(rows.len(), selected, "{name}");
        assert!(rows
            .iter()
            .all(|r| r["cost_units"] == 1 && r["hard_junk"] == false));
        let reads = fs::read_to_string(calls).unwrap();
        if staging == "1" && offline != "ANNO_NO_DOWNLOADS" {
            let remote_dataset = if task == "ner" { "WikiGold" } else { "GAP" };
            if task == "ner" {
                let rejected = &pool["pool"]["load_rejections"];
                let total: u64 = rejected
                    .as_object()
                    .unwrap()
                    .values()
                    .map(|v| v.as_u64().unwrap())
                    .sum();
                let examples = pool["pool"]["rejection_examples"].as_array().unwrap();
                assert!(total > 0);
                assert_eq!(examples.len(), total.min(3) as usize);
                for example in examples {
                    let dataset = example["dataset"].as_str().unwrap();
                    assert!(reads.contains(&format!("/datasets/{dataset}.cache -")));
                    assert_eq!(example["reason"], "S3 cache error");
                }
            }
            assert_eq!(pool["pool"]["selected_sources"][remote_dataset], "S3Cache");
            assert!(reads.contains(&format!(
                "s3 cp s3://fixture/datasets/{remote_dataset}.cache -"
            )));
            assert!(reads
                .lines()
                .all(|line| line.starts_with("s3 cp s3://fixture/") && line.ends_with(" -")));
            assert!(rows
                .iter()
                .any(|r| r["dataset"] == remote_dataset && r["dataset_sha256"].as_str().is_some()));
        } else {
            assert!(reads.is_empty(), "{name}: {reads}");
        }
    }
}
