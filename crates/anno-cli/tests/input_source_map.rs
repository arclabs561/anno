use serde_json::Value;
use std::fs;
use std::process::{Command, Output};

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_anno"))
        .env("ANNO_NO_DOWNLOADS", "1")
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn external_cleaning_map_survives_a_second_extraction_and_rejects_a_stale_edition() {
    let dir = tempfile::tempdir().unwrap();
    let saved = dir.path().join("earlier.json");
    let bare = dir.path().join("map.json");
    let original = "  Cafe\u{301}   a@example.com";
    let prefix = ["extract", "--model", "pattern", "--format", "grounded"];
    let first = run(&[prefix.as_slice(), &["--text", original, "--clean"]].concat());
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    fs::write(&saved, &first.stdout).unwrap();
    let first: Value = serde_json::from_slice(&first.stdout).unwrap();
    fs::write(&bare, first["source_map"].to_string()).unwrap();
    for path in [&bare, &saved] {
        let path = path.to_str().unwrap();
        let args = ["--normalize", "--input-source-map", path, "--text"];
        let second = run(&[prefix.as_slice(), &args, &[first["text"].as_str().unwrap()]].concat());
        assert!(
            second.status.success(),
            "{}",
            String::from_utf8_lossy(&second.stderr)
        );
        let doc: anno::GroundedDocument = serde_json::from_slice(&second.stdout).unwrap();
        assert_eq!(doc.text(), "Café a@example.com");
        let start = original.find("a@example.com").unwrap();
        assert_eq!(
            doc.project_source_span(5..18).unwrap().unwrap()[0].source_bytes,
            Some(start..original.len())
        );
        let stale = run(&[prefix.as_slice(), &args, &["Xafe\u{301} a@example.com"]].concat());
        assert!(!stale.status.success());
    }
}
