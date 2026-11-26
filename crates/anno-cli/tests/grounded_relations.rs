use serde_json::Value;
use std::path::Path;
use std::process::Command;

fn stdout(cache: &Path, args: &[&str]) -> Vec<u8> {
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
    output.stdout
}

fn run(cache: &Path, args: &[&str]) -> Value {
    serde_json::from_slice(&stdout(cache, args)).unwrap()
}

#[test]
fn extract_and_enhance_preserve_edge_endpoints_and_scorer() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("grounded.json");
    let path_str = path.to_str().unwrap();
    let extract = [
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
        "--quiet",
    ];
    let mut flat_args = extract.to_vec();
    flat_args.extend(["--format", "json", "--export", path_str]);
    let flat = run(temp.path(), &flat_args);
    let mut stream_args = extract.to_vec();
    stream_args.extend(["--format", "jsonl"]);
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
    let entities_only = run(
        temp.path(),
        &[
            "extract",
            "--model",
            "tplinker",
            "--text",
            "Marie works for Acme Inc.",
            "--format",
            "json",
        ],
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
        let source = entities_only["entities"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entity| {
                entity["start"] == edge["relation"][role]["start"]
                    && entity["end"] == edge["relation"][role]["end"]
            })
            .unwrap();
        assert!(signal["provenance"]["source"].is_string());
        assert_eq!(signal["provenance"], source["provenance"]);
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
fn empty_relation_results_still_identify_the_scorer() {
    let cache = tempfile::tempdir().unwrap();
    for (backend, scorer) in [
        ("pattern", "relation-heuristic"),
        ("tplinker", "tplinker-heuristic"),
    ] {
        let args = [
            "extract",
            "--model",
            backend,
            "--extract-relations",
            "--text",
            "東京。",
            "--format",
        ];
        let mut json_args = args.to_vec();
        json_args.push("json");
        let json = run(cache.path(), &json_args);
        assert_eq!(json["relations"], serde_json::json!([]));
        assert_eq!(json["provenance"]["relation_producer"]["source"], scorer);
        let mut stream_args = args.to_vec();
        stream_args.push("jsonl");
        let output = stdout(cache.path(), &stream_args);
        let header = serde_json::Deserializer::from_slice(&output)
            .into_iter::<Value>()
            .next()
            .unwrap()
            .unwrap();
        assert_eq!(
            header["provenance"]["relation_producer"],
            json["provenance"]["relation_producer"]
        );
    }
}

#[test]
fn character_distance_controls_the_direct_and_joint_heuristic_routes() {
    let cache = tempfile::tempdir().unwrap();
    let text = format!("東京 — Marie works for {}Acme Inc.", "près ".repeat(24));
    let head = text[..text.find("Marie").unwrap()].chars().count();
    let tail = text[..text.find("Acme").unwrap()].chars().count();
    let distance = tail - (head + "Marie".chars().count());
    assert!(distance > 120);
    for backend in ["heuristic", "tplinker"] {
        for (limit, expected) in [(distance - 1, 0), (distance, 1)] {
            let report = run(
                cache.path(),
                &[
                    "extract",
                    "--model",
                    backend,
                    "--extract-relations",
                    "--relation-types",
                    "WORKS_FOR",
                    "--relation-threshold",
                    "0.1",
                    "--relation-max-span-distance",
                    &limit.to_string(),
                    "--text",
                    &text,
                    "--format",
                    "json",
                ],
            );
            let edges = report["relations"].as_array().unwrap();
            assert_eq!(edges.len(), expected, "{backend}, {limit}");
            if let Some(edge) = edges.first() {
                assert_eq!(edge["head"]["start"], head);
                assert_eq!(edge["tail"]["start"], tail);
            }
        }
    }
}
