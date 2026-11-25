#![cfg(feature = "eval")]

use anno_eval::eval::{dataset_registry::DatasetId, loader::LoadableDatasetId};
use std::process::Command;

fn catalog(args: &[&str]) -> String {
    let run = Command::new(env!("CARGO_BIN_EXE_anno"))
        .arg("dataset")
        .args(args)
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8(run.stdout).unwrap()
}

#[test]
fn catalog_selection_counts_the_displayed_standard_loaders() {
    let domain = "wikipedia";
    let expected: Vec<_> = DatasetId::all()
        .iter()
        .filter(|id| {
            id.domain().to_lowercase().contains(domain) && LoadableDatasetId::try_from(**id).is_ok()
        })
        .map(|id| id.name())
        .collect();
    let output = catalog(&["list", "--loadable", "--domain", domain]);
    let actual: Vec<_> = output
        .lines()
        .filter_map(|line| line.strip_prefix("    "))
        .collect();
    assert_eq!(actual, expected);
    assert!(output.contains(&format!(
        "{} entries supported by the standard loader",
        expected.len()
    )));
}

#[test]
fn docred_is_reported_as_catalog_only() {
    let output = catalog(&["check", "--dataset", "DocRED"]);
    assert!(output.contains("DocRED: catalog only"));
    assert!(output.contains("Supported by the standard loader: 0"));
    assert!(!output.contains("Warnings"));
}

#[test]
fn sampler_explains_cached_sources_filtered_by_policy_and_pins() {
    let cache = tempfile::tempdir().unwrap();
    let datasets = cache.path().join("datasets");
    std::fs::create_dir(&datasets).unwrap();
    std::fs::write(datasets.join("Wnut17.cache"), "data\tO\n").unwrap();
    let history = cache.path().join("bad-history.json");
    std::fs::write(&history, "{incomplete").unwrap();

    let inspect = |allow: bool, domain: &str| {
        let result = Command::new(env!("CARGO_BIN_EXE_anno"))
            .arg("sampler")
            .arg("--history-file")
            .arg(&history)
            .args(["eligibility", "--task", "ner", "--require-cached"])
            .env("ANNO_CACHE_DIR", cache.path())
            .env("ANNO_S3_CACHE", "0")
            .env(
                "ANNO_MATRIX_INCLUDE_NON_AUTOMATABLE",
                if allow { "1" } else { "0" },
            )
            .env("ANNO_MUXER_PIN_DOMAIN", domain)
            .env("ANNO_MUXER_FILTER_DOMAIN", "")
            .env("ANNO_MUXER_PIN_LANG", "")
            .env("ANNO_MUXER_FILTER_LANG", "")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
        let rows = report["datasets"].as_array().unwrap();
        assert_eq!(
            report["eligible_count"],
            rows.iter().filter(|r| r["eligible"] == true).count()
        );
        assert_eq!(report["require_cached"], true);
        let wnut = rows.iter().find(|r| r["dataset"] == "Wnut17").unwrap();
        assert_eq!(wnut["locally_cached"], true);
        let wiki = rows.iter().find(|r| r["dataset"] == "WikiGold").unwrap();
        assert_eq!(wiki["locally_cached"], false);
        assert_eq!(wiki["exclusion"], "cache_required");
        (
            wnut["eligible"].as_bool().unwrap(),
            wnut["exclusion"].clone(),
        )
    };
    assert_eq!(inspect(false, ""), (false, "non_automatable_source".into()));
    assert_eq!(inspect(true, ""), (true, serde_json::Value::Null));
    assert_eq!(
        inspect(true, "unmatched_domain"),
        (false, "facet_pin".into())
    );
}

#[test]
fn googlere_download_loads_snippets_and_relation_eval_explains_missing_offsets() {
    let cache = tempfile::tempdir().unwrap();
    let datasets = cache.path().join("datasets");
    std::fs::create_dir(&datasets).unwrap();
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_anno"))
            .arg("dataset")
            .args(args)
            .env("ANNO_CACHE_DIR", cache.path())
            .env("ANNO_NO_DOWNLOADS", "1")
            .env("ANNO_S3_CACHE", "0")
            .output()
            .unwrap()
    };
    let evaluation = run(&["eval", "--dataset", "GoogleRE", "--task", "relation"]);
    assert!(!evaluation.status.success());
    assert!(String::from_utf8_lossy(&evaluation.stderr)
        .contains("without gold subject/object character spans"));
    std::fs::write(datasets.join("GoogleRE.cache"),
        r#"{"pred":"/people/person/place_of_birth","sub":"/m/person","obj":"/m/place","evidences":[{"url":"https://example.org/story","snippet":"Mara was born in Oslo."}],"judgments":[{"rater":"r1","judgment":"yes"}]}"#).unwrap();
    let download = run(&["download", "--dataset", "GoogleRE"]);
    assert!(
        download.status.success(),
        "{}",
        String::from_utf8_lossy(&download.stderr)
    );
    assert!(String::from_utf8_lossy(&download.stdout).contains("Google-RE (sentences=1)"));
}

#[test]
fn publisher_load_respects_the_explicit_trust_bundle_before_fetching() {
    let cache = tempfile::tempdir().unwrap();
    let ca = cache.path().join("empty.pem");
    std::fs::write(&ca, "").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_anno"))
        .args(["dataset", "download", "--dataset", "FewRel"])
        .env("SSL_CERT_FILE", ca)
        .env("ANNO_CACHE_DIR", cache.path())
        .env("ANNO_S3_CACHE", "0")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("No certificates in SSL_CERT_FILE"));
    assert!(!cache.path().join("datasets/FewRel.cache").exists());
}
