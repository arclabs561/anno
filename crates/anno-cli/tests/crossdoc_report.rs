#![cfg(feature = "eval")]

use anno::{GroundedDocument, Identity, Location, Signal};
use serde_json::{json, Value};
use std::fs;
use std::path::Path;
use std::process::{Command, Output};

fn anno(working_dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_anno"))
        .args(args)
        .current_dir(working_dir)
        .env("ANNO_NO_DOWNLOADS", "1")
        .env("HF_HUB_OFFLINE", "1")
        .output()
        .unwrap()
}

fn success(output: Output) -> Vec<u8> {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

fn linked_doc(id: &str, catalog: &str, accession: &str) -> GroundedDocument {
    let mut doc = GroundedDocument::new(id, "x Ari");
    doc.add_signal(Signal::new(0, Location::text(0, 1), "x", "OTHER", 0.8));
    let signal = doc.add_signal(Signal::new(0, Location::text(2, 5), "Ari", "PER", 0.9));
    let track = doc.create_track_from_signals("Ari", &[signal]).unwrap();
    let identity = doc.add_identity(Identity::from_kb(0, "Ari", catalog, accession));
    doc.link_track_to_identity(track, identity);
    doc
}

#[test]
fn query_reads_the_produced_report_and_matches_the_exact_catalog_record() {
    let dir = tempfile::tempdir().unwrap();
    let dir = dir.path();
    let mut args = vec!["cross-doc"];
    for (id, file, catalog, accession) in [
        ("cast", "cast.json", "stage", "17"),
        ("tour", "tour.json", "stage", "17"),
        ("crew", "crew.json", "harbor", "17"),
        ("guest", "guest.json", "stage", "27"),
        ("note", "note.json", "unknown", "17"),
    ] {
        fs::write(
            dir.join(file),
            serde_json::to_vec(&linked_doc(id, catalog, accession)).unwrap(),
        )
        .unwrap();
        args.extend(["--import", file]);
    }
    args.extend(["--output", "report.json"]);
    success(anno(dir, &args));

    let report: Value =
        serde_json::from_slice(&fs::read(dir.join("report.json")).unwrap()).unwrap();
    assert_eq!(report["metadata"]["documents_processed"], 5);
    let all = &report["clusters"];
    assert_eq!(all.as_array().unwrap().len(), 4);
    fs::write(dir.join("array.json"), serde_json::to_vec(all).unwrap()).unwrap();

    for file in ["report.json", "array.json"] {
        let query = |filters: &[&str]| -> Value {
            let mut args = vec!["query", file, "--format", "json"];
            args.extend_from_slice(filters);
            serde_json::from_slice(&success(anno(dir, &args))).unwrap()
        };
        assert_eq!(&query(&[]), all);
        let records = query(&["--catalog-record", "stage", "17", "--entity", "ari"]);
        assert_eq!(records.as_array().unwrap().len(), 1);
        assert_eq!(records[0]["mentions"], json!([["cast", 1], ["tour", 1]]));
        assert_eq!(query(&["--catalog-record", "unused", "17"]), json!([]));
    }

    for filter in [
        "--linked",
        "--negated",
        "--min-confidence=0.1",
        "--start-offset=0",
    ] {
        let result = anno(dir, &["query", "report.json", filter]);
        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr).contains("GroundedDocument input"));
    }
    let result = anno(dir, &["query", "report.json", "--format", "grounded"]);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("unsupported for cluster queries"));

    let stdout = success(anno(
        dir,
        &[
            "query",
            "report.json",
            "--catalog-record",
            "stage",
            "17",
            "--format",
            "tree",
            "--output",
            "tree.txt",
        ],
    ));
    assert!(stdout.is_empty());
    let tree = fs::read_to_string(dir.join("tree.txt")).unwrap();
    assert!(tree.contains("entity[1] (doc: tour)"));
    assert!(!tree.contains("crew"));
}

#[test]
fn recursive_directory_rejects_ambiguous_mentions_before_writing_output() {
    let dir = tempfile::tempdir().unwrap();
    let dir = dir.path();
    for source in ["harbor", "studio"] {
        fs::create_dir(dir.join(source)).unwrap();
        fs::write(
            dir.join(source).join("report.txt"),
            format!("Contact {source} at archive@example.org."),
        )
        .unwrap();
    }
    fs::write(dir.join("result.json"), "prior output").unwrap();
    let output = anno(
        dir,
        &[
            "cross-doc",
            ".",
            "--model",
            "pattern",
            "--recursive",
            "--output",
            "result.json",
        ],
    );
    assert!(!output.status.success());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("Duplicate document ID"), "{error}");
    for source in ["harbor", "studio"] {
        let path = Path::new(source).join("report.txt");
        assert!(error.contains(path.to_str().unwrap()), "{error}");
    }
    assert_eq!(
        fs::read_to_string(dir.join("result.json")).unwrap(),
        "prior output"
    );

    fs::rename(dir.join("studio/report.txt"), dir.join("studio/notes.txt")).unwrap();
    let output = success(anno(
        dir,
        &[
            "cross-doc",
            ".",
            "--model",
            "pattern",
            "--recursive",
            "--cross-doc-only",
        ],
    ));
    let report: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(report["metadata"]["documents_processed"], 2);
    let clusters = report["clusters"].as_array().unwrap();
    assert_eq!(clusters.len(), 1);
    let mentions = clusters[0]["mentions"].as_array().unwrap();
    assert_eq!(mentions.len(), 2);
    assert!(mentions.contains(&json!(["notes", 0])) && mentions.contains(&json!(["report", 0])));
}
