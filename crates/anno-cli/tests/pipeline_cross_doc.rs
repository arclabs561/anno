#![cfg(feature = "eval")]

use anno::GroundedDocument;
use serde_json::{json, Value};
use std::fs;
use std::process::{Command, Output};

fn cli(args: &[&str]) -> Output {
    let cache = tempfile::tempdir().unwrap();
    Command::new(env!("CARGO_BIN_EXE_anno"))
        .args(args)
        .env("XDG_CACHE_HOME", cache.path())
        .env("HF_HOME", cache.path())
        .env("FCOREF_MODEL_PATH", cache.path())
        .env("ANNO_NO_DOWNLOADS", "1")
        .env("HF_HUB_OFFLINE", "1")
        .output()
        .unwrap()
}

fn read(args: &[&str]) -> Value {
    let out = cli(args);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}

#[test]
fn other_documents_can_change_coreference_without_changing_catalog_postings() {
    let views = |other: &str| {
        let base = ["pipeline", "--model", "pattern", "--cross-doc", "--format"];
        let args = |format| {
            [
                base.as_slice(),
                &[format, "Contact same@example.org today.", other],
            ]
            .concat()
        };
        let docs: Vec<GroundedDocument> = serde_json::from_value(read(&args("grounded"))).unwrap();
        let first = &docs[0];
        assert_eq!(first.id(), "text1");
        let signal = first
            .signals()
            .iter()
            .find(|s| s.surface() == "same@example.org")
            .unwrap();
        let identity = first.identity_for_signal(signal.id).unwrap();
        assert!(identity.qualified_kb_ref().is_none());
        let members = identity.source.as_ref().unwrap().track_refs().len();
        let clusters = read(&args("json"));
        let cluster = clusters
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["canonical_name"] == "same@example.org")
            .unwrap();
        assert_eq!(cluster["documents"].as_array().unwrap().len(), members);
        (serde_json::to_value(first).unwrap(), members)
    };
    let (before, together) = views("Also same@example.org.");
    let (after, apart) = views("Also other@example.net.");
    assert_eq!((together, apart), (2, 1));

    let dir = tempfile::tempdir().unwrap();
    let revision = |doc, seq: u64| {
        let path = dir.path().join(format!("{seq}.json"));
        fs::write(
            &path,
            serde_json::to_vec(&json!({"stream":"inbox","sequence":seq,"document":doc})).unwrap(),
        )
        .unwrap();
        path
    };
    let old = revision(before, 1);
    let new = revision(after, 2);
    let diff = read(&[
        "query",
        "publication",
        new.to_str().unwrap(),
        "--since",
        old.to_str().unwrap(),
    ]);
    assert_eq!(diff["removed"], json!([]));
    assert_eq!(diff["upserted"], json!([]));
}

#[test]
fn coreference_and_demo_linking_survive_cross_document_rendering() {
    let args = [
        "pipeline",
        "--model",
        "heuristic",
        "--coref",
        "--link-kb",
        "--cross-doc",
        "--format",
        "grounded",
        "We met Alice and Alice.",
        "We saw Alice and Alice.",
    ];
    let docs: Vec<GroundedDocument> = serde_json::from_value(read(&args)).unwrap();
    for doc in &docs {
        assert_eq!(doc.coref_backend(), Some("mention-ranking"));
        assert!(doc.tracks().any(|track| track.signals.len() > 1
            && doc
                .identity_for_track(track.id)
                .and_then(|i| i.qualified_kb_ref())
                == Some(("demo", "demo:alice"))));
    }
    let mut report_args = args;
    report_args[7] = "json";
    let report = read(&report_args);
    let cluster = report
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["kb_id"] == "demo:alice")
        .unwrap();
    assert_eq!(cluster["documents"], json!(["text1", "text2"]));
    assert!(cluster["mentions"].as_array().unwrap().len() >= 4);
}

#[test]
fn rejects_colliding_file_and_positional_document_ids() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("text1.txt");
    fs::write(&path, "another source").unwrap();
    let out = cli(&[
        "pipeline",
        "--cross-doc",
        "--files",
        path.to_str().unwrap(),
        "original source",
    ]);
    let error = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert!(
        error.contains("Duplicate document ID")
            && error.contains("positional input 1")
            && error.contains("text1.txt"),
        "{error}"
    );
}
