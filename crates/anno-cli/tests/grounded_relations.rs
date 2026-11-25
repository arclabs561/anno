use serde_json::Value;
use std::path::Path;
use std::process::Command;

fn run(cache: &Path, args: &[&str]) -> Value {
    let output = Command::new(env!("CARGO_BIN_EXE_anno"))
        .env("ANNO_NO_DOWNLOADS", "1")
        .env("XDG_CACHE_HOME", cache)
        .env("HF_HOME", cache)
        .args(args)
        .output()
        .expect("run anno");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn extract_and_enhance_preserve_edge_endpoints_and_scorer() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("grounded.json");
    let path_str = path.to_str().unwrap();
    let flat = run(
        temp.path(),
        &[
            "extract",
            "--model",
            "tplinker",
            "--text",
            "Marie works for Acme Inc.",
            "--extract-relations",
            "--relation-types",
            "WORKS_FOR",
            "--threshold",
            "0.1",
            "--format",
            "json",
            "--export",
            path_str,
            "--quiet",
        ],
    );
    let saved: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(flat["relations"].as_array().unwrap().len(), 1);
    let edge = &saved["relation_evidence"]["mentions"][0];
    assert_eq!(edge["provenance"]["source"], "tplinker-heuristic");
    assert_eq!(edge["relation"]["relation_type"], "WORKS_FOR");
    assert_eq!(
        edge["relation"]["confidence"],
        flat["relations"][0]["confidence"]
    );
    for role in ["head", "tail"] {
        let id = &edge[format!("{role}_signal_id")];
        let signal = saved["signals"]
            .as_array()
            .unwrap()
            .iter()
            .find(|signal| signal["id"] == *id)
            .unwrap();
        assert_eq!(signal["surface"], flat["relations"][0][role]["text"]);
        assert_eq!(signal["provenance"]["source"], "tplinker");
        assert_eq!(
            signal["location"]["Text"]["start"],
            edge["relation"][role]["start"]
        );
    }
    let loaded = run(
        temp.path(),
        &["enhance", path_str, "--format", "grounded", "--quiet"],
    );
    assert_eq!(loaded["relation_evidence"], saved["relation_evidence"]);
    let queried = run(temp.path(), &["query", "relations", path_str]);
    assert_eq!(queried["mentions"][0]["evidence"], *edge);
}
