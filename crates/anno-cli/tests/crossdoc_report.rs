#![cfg(feature = "eval")]

use anno::{GroundedDocument, Identity, Location, Signal};
use serde_json::{json, Value};
use std::fs;
use std::path::Path;
use std::process::{Command, Output};

fn command(working_dir: &Path, args: &[&str]) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_anno"));
    cmd.args(args)
        .current_dir(working_dir)
        .env("ANNO_NO_DOWNLOADS", "1")
        .env("HF_HUB_OFFLINE", "1");
    cmd
}

fn anno(working_dir: &Path, args: &[&str]) -> Output {
    command(working_dir, args).output().unwrap()
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
    assert!(String::from_utf8_lossy(&result.stderr).contains("invalid value"));

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

#[test]
fn mixed_imports_apply_same_document_replacements_in_argument_order() {
    let dir = tempfile::tempdir().unwrap();
    let dir = dir.path();
    let input_file = dir.join("initial.json");
    fs::write(
        input_file,
        serde_json::to_vec(&[
            linked_doc("notice", "record", "initial"),
            linked_doc("other", "record", "constant"),
        ])
        .unwrap(),
    )
    .unwrap();
    let stream = dir.join("changes.jsonl");
    let mut lines = String::new();
    for accession in ["draft", "updated"] {
        lines.push_str(&serde_json::to_string(&linked_doc("notice", "record", accession)).unwrap());
        lines.push('\n');
    }
    fs::write(&stream, lines).unwrap();
    for (first, second, expected) in [
        ("initial.json", "-", "updated"),
        ("-", "initial.json", "initial"),
    ] {
        let stdout = success(
            command(dir, &["cross-doc", "--import", first, "--import", second])
                .stdin(fs::File::open(&stream).unwrap())
                .output()
                .unwrap(),
        );
        let report: Value = serde_json::from_slice(&stdout).unwrap();
        assert_eq!(report["metadata"]["documents_processed"], 2);
        let clusters = report["clusters"].as_array().unwrap();
        assert_eq!(clusters.len(), 2);
        let notice = clusters
            .iter()
            .find(|cluster| cluster["documents"] == json!(["notice"]))
            .unwrap();
        assert_eq!(notice["kb_id"], expected);
    }
    let conflict = anno(dir, &["cross-doc", "--stdin", "--import", "initial.json"]);
    assert!(!conflict.status.success());
    let message = String::from_utf8_lossy(&conflict.stderr);
    assert!(
        message.contains("--stdin") && message.contains("--import"),
        "{message}"
    );
}

#[test]
fn resolved_grounded_array_can_be_imported_and_its_members_regrouped() {
    use anno::{coalesce::Resolver, Corpus};

    let dir = tempfile::tempdir().unwrap();
    let mut corpus = Corpus::new();
    for id in ["minutes", "program"] {
        let mut doc = GroundedDocument::new(id, "Ari");
        let signal = doc.add_signal(Signal::new(0, Location::text(0, 3), "Ari", "PER", 0.8));
        doc.create_track_from_signals("Ari", &[signal]).unwrap();
        corpus.add_document(doc);
    }
    assert_eq!(
        Resolver::new()
            .resolve_inter_doc_coref(&mut corpus, None, None)
            .len(),
        1
    );
    let docs: Vec<_> = corpus.documents().collect();
    fs::write(
        dir.path().join("grounded.json"),
        serde_json::to_vec_pretty(&docs).unwrap(),
    )
    .unwrap();
    let stdout = success(anno(
        dir.path(),
        &["cross-doc", "--import", "grounded.json", "--cross-doc-only"],
    ));
    let report: Value = serde_json::from_slice(&stdout).unwrap();
    let clusters = report["clusters"].as_array().unwrap();
    assert_eq!(clusters.len(), 1);
    assert_eq!(
        clusters[0]["mentions"],
        json!([["minutes", 0], ["program", 0]])
    );
    assert_eq!(clusters[0]["minimum_pair_similarity"], 1.0);
}

#[test]
fn delivery_manifests_scope_documents_and_replay_original_receipts_in_the_report() {
    let temp = tempfile::tempdir().unwrap();
    let directory = temp.path();
    let inputs = directory.join("inputs");
    fs::create_dir(&inputs).unwrap();
    let source = linked_doc("7", "library", "17");
    fs::write(
        inputs.join("one.json"),
        serde_json::to_vec(&source).unwrap(),
    )
    .unwrap();
    let member = |issuer: &str| {
        json!({
            "request": {"issuer": issuer, "delivery_id": "edition-1/7", "source_revision": 1,
                        "ownership": {"mode": "independent"}},
            "path": "one.json"
        })
    };
    let manifest = json!([member("east"), member("west"), member("east")]);
    fs::write(
        inputs.join("deliveries.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let output = success(anno(
        directory,
        &["cross-doc", "--delivery-manifest", "inputs/deliveries.json"],
    ));
    let report: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(report["metadata"]["documents_processed"], 2);
    let receipts = &report["deliveries"];
    assert_eq!(receipts[0]["applied"], receipts[2]["previously_applied"]);
    assert_eq!(receipts[0]["applied"]["applied_revision"], 1);
    assert_eq!(receipts[1]["applied"]["applied_revision"], 2);
    assert_ne!(
        receipts[0]["applied"]["document_id"],
        receipts[1]["applied"]["document_id"]
    );

    let mut conflict = member("east");
    conflict["request"]["source_revision"] = json!(2);
    fs::write(
        inputs.join("conflict.json"),
        serde_json::to_vec(&json!([member("east"), conflict])).unwrap(),
    )
    .unwrap();
    let failed = anno(
        directory,
        &[
            "cross-doc",
            "--delivery-manifest",
            "inputs/conflict.json",
            "--output",
            "unpublished.json",
        ],
    );
    assert!(!failed.status.success());
    assert!(String::from_utf8_lossy(&failed.stderr).contains("different bytes or intent"));
    assert!(!directory.join("unpublished.json").exists());
}

#[test]
fn requested_cross_doc_sources_succeed_together_or_preserve_the_destination() {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path();
    fs::write(
        dir.join("doc.json"),
        serde_json::to_vec(&linked_doc("d", "a", "b")).unwrap(),
    )
    .unwrap();
    fs::write(dir.join("out"), "prior output").unwrap();
    for (sources, expected) in [
        (vec![".", "--import", "doc.json"], "cannot be used with"),
        (vec![".", "--stdin"], "cannot be used with"),
        (
            vec!["--import", "doc.json", "--import", "none-*.json"],
            "none-*.json",
        ),
    ] {
        let mut args = vec!["cross-doc", "--output", "out"];
        args.extend(sources);
        let result = anno(dir, &args);
        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr).contains(expected));
        assert_eq!(fs::read(dir.join("out")).unwrap(), b"prior output");
    }
}

#[test]
fn trees_preserve_source_gaps_and_singletons_count_mentions() {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path();
    let mut doc = GroundedDocument::new("d", "À wrong 京. À wrong 京. Z");
    let parts = |offset: usize| Location::Discontinuous {
        segments: vec![(offset, offset + 1), (offset + 8, offset + 9)],
    };
    let first = doc.add_signal(Signal::new(0, parts(0), "À 京", "PER", 0.9));
    let second = doc.add_signal(Signal::new(0, parts(11), "À 京", "PER", 0.9));
    let track = doc
        .create_track_from_signals("À 京", &[first, second])
        .unwrap();
    let identity = doc.add_identity(Identity::from_kb(0, "À 京", "people", "paired"));
    doc.link_track_to_identity(track, identity);
    let last = doc.add_signal(Signal::new(0, Location::text(22, 23), "Z", "ORG", 0.9));
    let track = doc.create_track_from_signals("Z", &[last]).unwrap();
    let identity = doc.add_identity(Identity::from_kb(1, "Z", "groups", "solo"));
    doc.link_track_to_identity(track, identity);
    fs::write(dir.join("doc.json"), serde_json::to_vec(&doc).unwrap()).unwrap();
    let display = |format| {
        String::from_utf8(success(anno(
            dir,
            &[
                "--color",
                "never",
                "cross-doc",
                "--verbose",
                "--import",
                "doc.json",
                "--format",
                format,
            ],
        )))
        .unwrap()
    };
    let tree = display("tree");
    assert!(tree.contains("[À] wrong [京]"), "{tree}");
    assert!(tree.contains("2 (0 cross-doc, 1 singleton)"), "{tree}");
    assert!(display("summary").contains("Singleton clusters: 1"));
}

#[test]
fn clustering_threshold_rejects_invalid_units_before_input() {
    let root = tempfile::tempdir().unwrap();
    for command in ["cross-doc", "pipeline"] {
        for value in ["75", "-0.1", "NaN", "inf"] {
            let result = anno(root.path(), &[command, &format!("--threshold={value}")]);
            assert!(!result.status.success());
            assert!(String::from_utf8_lossy(&result.stderr).contains("within 0..=1"));
        }
    }
    fs::write(
        root.path().join("doc.json"),
        serde_json::to_vec(&linked_doc("d", "a", "b")).unwrap(),
    )
    .unwrap();
    for value in ["0", "1"] {
        let output = success(anno(
            root.path(),
            &["cross-doc", "--import", "doc.json", "--threshold", value],
        ));
        let report: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(
            report["metadata"]["threshold"],
            value.parse::<f64>().unwrap()
        );
    }
}
