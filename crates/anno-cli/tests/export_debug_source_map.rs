use anno::{SourceKind, SourceMap};
use serde_json::Value;
use std::fs;
use std::process::Command;

fn run(args: &[&str]) -> Vec<u8> {
    let output = Command::new(env!("CARGO_BIN_EXE_anno"))
        .env("ANNO_NO_DOWNLOADS", "1")
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

#[test]
fn debug_exports_original_offsets_in_every_format() {
    let temp = tempfile::tempdir().unwrap();
    let source = " \t👩‍🔬  e\u{301}  ada@example.com \r\n ada@example.com  ";
    let input = temp.path().join("utf8.txt");
    fs::write(&input, source).unwrap();
    let mut baseline = None;
    for format in ["full", "signals", "minimal"] {
        let saved = temp.path().join(format!("{format}.json"));
        run(&[
            "debug",
            "--model",
            "pattern",
            "--file",
            input.to_str().unwrap(),
            "--clean",
            "--normalize",
            "--export",
            saved.to_str().unwrap(),
            "--export-format",
            format,
            "--quiet",
        ]);
        let doc: Value = serde_json::from_slice(&fs::read(saved).unwrap()).unwrap();
        let analyzed = doc["text"].as_str().unwrap();
        assert!(analyzed.starts_with("👩🔬 é "));
        let map: SourceMap = serde_json::from_value(doc["source_map"].clone()).unwrap();
        assert_eq!(map.source_sha256, SourceMap::identity(source).source_sha256);
        if let Some(previous) = &baseline {
            assert_eq!(&map, previous);
        }
        let signal_count = doc["signals"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|s| s["surface"] == "ada@example.com")
            .count();
        assert_eq!(signal_count, 2);
        for ((analyzed_byte, email), (original_byte, _)) in analyzed
            .match_indices("ada@example.com")
            .zip(source.match_indices("ada@example.com"))
        {
            let runs = map
                .project_bytes(analyzed, analyzed_byte..analyzed_byte + email.len())
                .unwrap();
            assert_eq!(runs.len(), 1);
            assert_eq!(runs[0].kind, SourceKind::Direct);
            assert_eq!(
                runs[0].source_bytes,
                Some(original_byte..original_byte + email.len())
            );
        }
        baseline = Some(map);
    }
}

#[test]
fn debug_identifies_the_input_it_selected() {
    let temp = tempfile::tempdir().unwrap();
    let saved = temp.path().join("debug.json");
    let text = "  Ada  ";
    run(&[
        "debug",
        "--model",
        "pattern",
        "--text",
        text,
        "--file",
        "unused.txt",
        "--export",
        saved.to_str().unwrap(),
        "--quiet",
    ]);
    let doc: Value = serde_json::from_slice(&fs::read(saved).unwrap()).unwrap();
    assert_eq!(doc["id"], "debug");
    assert_eq!(doc["text"], text);
    let map: SourceMap = serde_json::from_value(doc["source_map"].clone()).unwrap();
    assert_eq!(map, SourceMap::identity(text));
}

#[test]
fn pipeline_preserves_the_complete_utf8_file() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("original.txt");
    let original = "\r\n👩‍🔬  ada@example.com\t\r\n";
    fs::write(&input, original).unwrap();
    let output = run(&[
        "pipeline",
        "--model",
        "pattern",
        "--files",
        input.to_str().unwrap(),
        "--format",
        "grounded",
        "--quiet",
    ]);
    let docs: Vec<Value> = serde_json::from_slice(&output).unwrap();
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0]["text"], original);
    let map: SourceMap = serde_json::from_value(docs[0]["source_map"].clone()).unwrap();
    assert_eq!(map, SourceMap::identity(original));
}
