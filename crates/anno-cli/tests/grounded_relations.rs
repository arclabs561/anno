use serde_json::Value;
use std::path::Path;
use std::process::{Command, Output};

fn invoke(cache: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_anno"))
        .env("ANNO_NO_DOWNLOADS", "1")
        .env("XDG_CACHE_HOME", cache)
        .env("HF_HOME", cache)
        .env("HF_HUB_CACHE", cache.join("hub"))
        .args(args)
        .output()
        .expect("run anno")
}

fn stdout(cache: &Path, args: &[&str]) -> Vec<u8> {
    let output = invoke(cache, args);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

fn run(cache: &Path, args: &[&str]) -> Value {
    serde_json::from_slice(&stdout(cache, args)).unwrap()
}

fn extract_args<'a>(model: &'a str, relation: &'a str, format: &'a str) -> Vec<&'a str> {
    vec![
        "extract",
        "--model",
        model,
        "--text",
        "Marie works for Acme Inc.",
        "--extract-relations",
        "--relation-types",
        relation,
        "--threshold",
        "0.1",
        "--format",
        format,
        "--quiet",
    ]
}

#[test]
fn extract_and_enhance_preserve_edge_endpoints_and_scorer() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("grounded.json");
    let path_str = path.to_str().unwrap();
    let mut flat_args = extract_args("tplinker", "WORKS_FOR", "json");
    flat_args.extend(["--export", path_str]);
    let flat = run(temp.path(), &flat_args);
    let stream_args = extract_args("tplinker", "WORKS_FOR", "jsonl");
    let stream = stdout(temp.path(), &stream_args);
    let streamed_edge = serde_json::Deserializer::from_slice(&stream)
        .into_iter::<Value>()
        .map(Result::unwrap)
        .find_map(|row| row.get("relation").cloned())
        .expect("JSONL relation row");
    let saved: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(flat["relations"].as_array().unwrap().len(), 1);
    let edge = &saved["relation_evidence"]["mentions"][0];
    assert_eq!(edge["provenance"]["source"], "tplinker-heuristic");
    assert_eq!(edge["provenance"]["method"], "Heuristic");
    assert_eq!(flat["relations"][0]["provenance"], edge["provenance"]);
    assert_eq!(streamed_edge["provenance"], edge["provenance"]);
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

#[test]
fn unmatched_relation_returns_entities_and_an_empty_list() {
    let cache = tempfile::tempdir().unwrap();
    let result = run(
        cache.path(),
        &extract_args("tplinker", "LOCATED_IN", "json"),
    );
    assert_eq!(result["relations"], serde_json::json!([]));
    let entities = result["entities"].as_array().unwrap();
    for name in ["Marie", "Acme Inc"] {
        assert!(entities.iter().any(|entity| entity["text"] == name));
    }
}

#[cfg(feature = "onnx")]
#[test]
fn missing_explicit_relex_export_fails_without_emitting_predictions() {
    let cache = tempfile::tempdir().unwrap();
    let result = invoke(
        cache.path(),
        &extract_args("gliner-relex", "WORKS_FOR", "json"),
    );
    assert!(!result.status.success());
    assert!(result.stdout.is_empty(), "failed extraction emitted output");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("GLiNER-RelEx") && stderr.contains("export"),
        "{stderr}"
    );
}
