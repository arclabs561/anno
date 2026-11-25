use std::fs;
use std::path::Path;
use std::process::Command;

fn export(input: &Path, out: &Path, format: &str, document: Option<&str>) -> (String, String) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_anno"));
    command
        .arg("export")
        .arg("--input")
        .arg(input)
        .arg("--output")
        .arg(out)
        .args([
            "--format",
            format,
            "--model",
            "pattern",
            "--overwrite",
            "--quiet",
        ]);
    if let Some(iri) = document {
        command.args(["--document-iri", iri]);
    }
    let result = command.output().expect("run anno export");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );

    let mut output = out.join(input.file_stem().unwrap());
    output.set_extension(if format == "jsonld" { "jsonld" } else { "nt" });
    let content = fs::read_to_string(output).unwrap();
    if format == "jsonld" {
        let graph: serde_json::Value = serde_json::from_str(&content).unwrap();
        let mention = &graph["@graph"][0];
        return (
            mention["@id"].as_str().unwrap().to_owned(),
            mention["prov:hadPrimarySource"]["@id"]
                .as_str()
                .unwrap()
                .to_owned(),
        );
    }
    let line = content
        .lines()
        .find(|line| line.contains("<http://www.w3.org/ns/prov#hadPrimarySource>"))
        .unwrap();
    let mut fields = line.split_whitespace();
    let mention = fields.next().unwrap().trim_matches(['<', '>']);
    let doc = fields.nth(1).unwrap().trim_matches(['<', '>']);
    (mention.to_owned(), doc.to_owned())
}

#[test]
fn rdf_exports_preserve_source_revision_and_share_occurrence_ids() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let out = root.join("out");
    let first = root.join("a/report.txt");
    let second = root.join("b/report.txt");
    let content = "🦊 alice@example.com";
    for document in [&first, &second] {
        fs::create_dir(document.parent().unwrap()).unwrap();
        fs::write(document, content).unwrap();
    }
    let implicit = export(&first, &out, "ntriples", None);
    let other = export(&second, &out, "ntriples", None);
    assert_ne!(implicit.0, other.0);
    assert_ne!(implicit.1, other.1);
    fs::write(&first, format!("{content}.")).unwrap();
    let edited = export(&first, &out, "ntriples", None);
    assert_ne!(edited.0, implicit.0);
    assert_ne!(edited.1, implicit.1);
    fs::write(&first, content).unwrap();

    let iri = "https://example.org/revision/Zoë/1";
    let explicit = export(&first, &out, "ntriples", Some(iri));
    assert_eq!(explicit.1, iri);
    assert_eq!(export(&second, &out, "ntriples", Some(iri)), explicit);
    assert_eq!(export(&first, &out, "jsonld", None), implicit);
    assert_eq!(export(&first, &out, "jsonld", Some(iri)), explicit);
    #[cfg(feature = "graph")]
    {
        assert_eq!(export(&first, &out, "graph-ntriples", None), implicit);
        assert_eq!(export(&first, &out, "graph-ntriples", Some(iri)), explicit);
    }
}

#[test]
fn grounded_input_preserves_discontinuous_mention_identity() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("report.json");
    let out = temp.path().join("out");
    let iri = "urn:corpus:report:v1";
    let span = anno::Entity::new("A B C", anno::EntityType::Person, 0, 5, 1.0);
    // GroundedDocument validates the saved surface against the full bounding interval.
    let mut split = span.clone();
    split.set_discontinuous_span(anno::DiscontinuousSpan::new(vec![0..1, 4..5]));
    let occurrence = |entity| {
        let doc = anno::GroundedDocument::from_entity_signals("report", "A B C", &[entity]);
        fs::write(&input, serde_json::to_vec(&doc).unwrap()).unwrap();
        export(&input, &out, "jsonld", Some(iri)).0
    };
    assert_ne!(occurrence(span), occurrence(split));
}

#[test]
fn discontinuous_export_preserves_brat_segments_and_rejects_flat_bio() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("report.json");
    let out = temp.path().join("out");
    let source = "🦊 Ada met Zoë";
    // Older grounded files retain the entire bounding surface.
    let mut entity = anno::Entity::new("Ada met Zoë", anno::EntityType::Person, 2, 13, 1.0);
    entity.set_discontinuous_span(anno::DiscontinuousSpan::new(vec![2..5, 10..13]));
    let doc = anno::GroundedDocument::from_entity_signals("report", source, &[entity]);
    fs::write(&input, serde_json::to_vec(&doc).unwrap()).unwrap();
    let run = |format| {
        Command::new(env!("CARGO_BIN_EXE_anno"))
            .arg("export")
            .arg("--input")
            .arg(&input)
            .arg("--output")
            .arg(&out)
            .args(["--model", "pattern", "--format", format, "--overwrite"])
            .output()
            .unwrap()
    };

    let brat = run("brat");
    assert!(
        brat.status.success(),
        "{}",
        String::from_utf8_lossy(&brat.stderr)
    );
    assert_eq!(fs::read_to_string(out.join("report.txt")).unwrap(), source);
    assert_eq!(
        fs::read_to_string(out.join("report.ann")).unwrap(),
        "T1\tPER 2 5;10 13\tAda Zoë"
    );

    let conll_path = out.join("report.conll");
    fs::write(&conll_path, "existing corpus").unwrap();
    let conll = run("conll");
    assert!(!conll.status.success());
    assert!(String::from_utf8_lossy(&conll.stderr)
        .contains("CoNLL BIO cannot represent a discontinuous mention"));
    assert_eq!(fs::read_to_string(conll_path).unwrap(), "existing corpus");
}

#[test]
fn document_iri_must_identify_one_rdf_input() {
    let temp = tempfile::tempdir().unwrap();
    let file = temp.path().join("text.txt");
    fs::write(&file, "Ada.").unwrap();
    let out = temp.path().join("out");
    for (input, format, iri) in [
        (file.as_path(), "ntriples", "relative"),
        (file.as_path(), "jsonld", "https://example.org/bad>iri"),
        (temp.path(), "ntriples", "urn:doc:rev"),
        (file.as_path(), "brat", "urn:doc:rev"),
    ] {
        let result = Command::new(env!("CARGO_BIN_EXE_anno"))
            .arg("export")
            .arg("--input")
            .arg(input)
            .arg("--output")
            .arg(&out)
            .args([
                "--format",
                format,
                "--document-iri",
                iri,
                "--model",
                "pattern",
            ])
            .output()
            .unwrap();
        assert!(!result.status.success(), "{input:?} {format} {iri}");
        assert!(String::from_utf8_lossy(&result.stderr).contains("document"));
        assert!(!out.exists());
    }
}

#[test]
fn conflicting_batch_names_fail_before_writing_any_files() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    let output = temp.path().join("export");
    fs::create_dir(&input).unwrap();
    fs::write(input.join("first.txt"), "other@example.org").unwrap();
    fs::write(input.join("report.txt"), "alice@example.org").unwrap();
    let doc = anno::GroundedDocument::new("report", "bob@example.org");
    fs::write(input.join("report.json"), serde_json::to_vec(&doc).unwrap()).unwrap();

    for format in ["ntriples", "brat", "graph-csv"] {
        for overwrite in [false, true] {
            let mut command = Command::new(env!("CARGO_BIN_EXE_anno"));
            command
                .arg("export")
                .arg("--input")
                .arg(&input)
                .arg("--output")
                .arg(&output)
                .args(["--format", format, "--model", "pattern"]);
            if overwrite {
                command.arg("--overwrite");
            }
            let result = command.output().unwrap();
            assert!(!result.status.success());
            let error = String::from_utf8(result.stderr).unwrap();
            assert!(
                error.contains("report.json") && error.contains("report.txt"),
                "{error}"
            );
            assert!(error.contains("same output"), "{error}");
            assert!(!output.exists());
        }
    }

    fs::rename(input.join("report.json"), input.join("notes.json")).unwrap();
    fs::create_dir(&output).unwrap();
    fs::write(output.join("report.nt"), "previous revision").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_anno"))
        .arg("export")
        .arg("--input")
        .arg(&input)
        .arg("--output")
        .arg(&output)
        .args(["--format", "ntriples", "--model", "pattern", "--overwrite"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_ne!(
        fs::read_to_string(output.join("report.nt")).unwrap(),
        "previous revision"
    );
    assert!(output.join("first.nt").is_file());
    assert!(output.join("notes.nt").is_file());
}

#[test]
fn export_preflights_aliases_and_all_companions() {
    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path();
    let input = dir.join("doc.jsonl");
    let original = r#"{"id":"v1","text":"unchanged","signals":[],"tracks":{},"identities":{}}"#;
    fs::write(&input, original).unwrap();
    let run = |src: &Path, dst: &Path, format: &str, overwrite: bool| {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_anno"));
        cmd.arg("export")
            .arg("-i")
            .arg(src)
            .arg("-o")
            .arg(dst)
            .args(["--format", format, "--model", "pattern", "--quiet"]);
        if overwrite {
            cmd.arg("--overwrite");
        }
        cmd.output().unwrap().status.success()
    };
    assert!(!run(&input, dir, "jsonl", true));
    assert_eq!(fs::read_to_string(&input).unwrap(), original);
    let out = dir.join("out");
    fs::create_dir(&out).unwrap();
    fs::hard_link(&input, out.join("doc.jsonl")).unwrap();
    assert!(!run(&input, &out, "jsonl", true));
    assert_eq!(fs::read_to_string(&input).unwrap(), original);
    fs::write(out.join("doc.txt"), "old").unwrap();
    assert!(!run(&input, &out, "brat", false));
    assert!(!out.join("doc.ann").exists());
    fs::write(dir.join("other.txt"), "new").unwrap();
    fs::write(out.join("other.nt"), "old").unwrap();
    assert!(!run(dir, &out, "ntriples", false));
    assert!(!out.join("doc.nt").exists());
}

#[cfg(unix)]
#[test]
fn missing_directory_source_fails_before_creating_any_export() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("sources");
    let out = temp.path().join("out");
    fs::create_dir(&input).unwrap();
    fs::write(input.join("available.txt"), "Ada").unwrap();
    std::os::unix::fs::symlink("absent", input.join("missing.txt")).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_anno"))
        .arg("export")
        .arg("--input")
        .arg(input)
        .arg("--output")
        .arg(&out)
        .args(["--model", "pattern", "--format", "jsonl"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("missing.txt"));
    assert!(!out.exists());
}

#[test]
fn rdf_segment_coordinates_roundtrip_even_with_an_opaque_occurrence_iri() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("report.json");
    let out = temp.path().join("out");
    let rows = temp.path().join("rows.jsonl");
    let mut entity = anno::Entity::new("Ada met Zoë", anno::EntityType::Person, 2, 13, 1.0);
    entity.set_discontinuous_span(anno::DiscontinuousSpan::new(vec![2..5, 10..13]));
    let doc = anno::GroundedDocument::from_entity_signals("report", "🦊 Ada met Zoë", &[entity]);
    fs::write(&input, serde_json::to_vec(&doc).unwrap()).unwrap();
    let revision = "urn:document:version-2";
    for format in [
        "jsonld",
        "ntriples",
        #[cfg(feature = "graph")]
        "graph-ntriples",
    ] {
        let (mention, _) = export(&input, &out, format, Some(revision));
        let path = out.join(if format == "jsonld" {
            "report.jsonld"
        } else {
            "report.nt"
        });
        let content = fs::read_to_string(&path).unwrap();
        assert!(content.contains("sourceSegments"));
        fs::write(&path, content.replace(&mention, "urn:opaque:mention")).unwrap();
        let imported = Command::new(env!("CARGO_BIN_EXE_anno"))
            .args(["dataset", "import", "--format", format, "--input"])
            .arg(path)
            .arg("--output")
            .arg(&rows)
            .output()
            .unwrap();
        assert!(
            imported.status.success(),
            "{}",
            String::from_utf8_lossy(&imported.stderr)
        );
        let row: serde_json::Value = serde_json::from_slice(&fs::read(&rows).unwrap()).unwrap();
        assert_eq!(row["source"], revision);
        assert_eq!(row["text"], "Ada Zoë");
        assert_eq!(
            row["segments"],
            serde_json::json!([{"start":2,"end":5},{"start":10,"end":13}])
        );
    }
}
