use anno::core::grounded::html_source::source_sha256;
use serde_json::Value;
use std::fs;
use std::path::Path;
use std::process::{Command, Output};

fn anno(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_anno"))
        .env("ANNO_NO_DOWNLOADS", "1")
        .args(args)
        .output()
        .expect("run anno")
}

fn successful(args: &[&str]) -> Value {
    let output = anno(args);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn extract_and_reload(source_file: &Path, saved: &Path, extra: &[&str]) -> Value {
    let mut args = vec![
        "extract",
        "--model",
        "pattern",
        "--file",
        source_file.to_str().unwrap(),
        "--format",
        "grounded",
        "--export",
        saved.to_str().unwrap(),
        "--quiet",
    ];
    args.extend_from_slice(extra);
    let generated = successful(&args);
    let serialized: Value = serde_json::from_slice(&fs::read(saved).unwrap()).unwrap();
    assert_eq!(generated, serialized);
    assert_eq!(
        serialized,
        successful(&[
            "enhance",
            saved.to_str().unwrap(),
            "--format",
            "grounded",
            "--quiet"
        ])
    );
    generated
}

fn bounds(value: &Value) -> std::ops::Range<usize> {
    value["start"].as_u64().unwrap() as usize..value["end"].as_u64().unwrap() as usize
}

/// Exercise public projection and original-file verification after a grounded reload.
fn origin(
    doc: &Value,
    saved: &Path,
    source: &Path,
    byte: usize,
) -> (String, std::ops::Range<usize>) {
    let text = doc["text"].as_str().unwrap();
    let character = text[..byte].chars().count();
    let result = successful(&[
        "query",
        "source-span",
        saved.to_str().unwrap(),
        &character.to_string(),
        &(character + 1).to_string(),
        "--original",
        source.to_str().unwrap(),
    ]);
    assert_eq!(bounds(&result["analyzed_chars"]), character..character + 1);
    let runs = result["fragments"].as_array().unwrap();
    assert_eq!(runs.len(), 1, "{result}");
    let run = &runs[0];
    (
        run["kind"].as_str().unwrap().to_owned(),
        bounds(&run["source_bytes"]),
    )
}

fn verify_domain(doc: &Value, source: &str) {
    let map = &doc["source_map"];
    let analyzed = doc["text"].as_str().unwrap();
    assert_eq!(map["source_sha256"], source_sha256(source));
    assert_eq!(map["analyzed_sha256"], source_sha256(analyzed));
    assert_eq!(map["source_byte_length"], source.len());
    assert_eq!(map["analyzed_byte_length"], analyzed.len());
}

fn check_repeated_signals(doc: &Value, email: &str) -> Vec<usize> {
    let text = doc["text"].as_str().unwrap();
    let occurrences: Vec<_> = text.match_indices(email).map(|(i, _)| i).collect();
    assert_eq!(occurrences.len(), 2);
    let signals: Vec<_> = doc["signals"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|s| s["surface"] == email)
        .collect();
    assert_eq!(signals.len(), 2);
    assert_ne!(signals[0]["id"], signals[1]["id"]);
    for (signal, &byte) in signals.iter().zip(&occurrences) {
        let start = text[..byte].chars().count();
        assert_eq!(
            bounds(&signal["location"]["Text"]),
            start..start + email.chars().count()
        );
    }
    occurrences
}

#[test]
fn clean_and_nfc_bind_two_equal_mentions_to_their_own_source_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let source_file = dir.path().join("input.txt");
    let saved = dir.path().join("grounded.json");
    let email = "alice@example.org";
    let raw = "\t  🦊  A\u{30a}   al\u{200b}ice@example.org\r\nalice@example.org  ";
    fs::write(&source_file, raw).unwrap();
    let doc = extract_and_reload(&source_file, &saved, &["--clean", "--normalize"]);
    verify_domain(&doc, raw);
    let text = doc["text"].as_str().unwrap();
    let positions = check_repeated_signals(&doc, email);
    let (kind, source) = origin(&doc, &saved, &source_file, text.find('Å').unwrap());
    assert_eq!(kind, "transformed");
    assert_eq!(&raw[source], "A\u{30a}");
    let (_, i) = origin(&doc, &saved, &source_file, positions[0] + 2);
    assert_eq!(i.start, raw.find("ice@example.org").unwrap());
    let (_, a) = origin(&doc, &saved, &source_file, positions[1]);
    assert_eq!(a.start, raw.rfind(email).unwrap());

    let other = dir.path().join("other.txt");
    fs::write(&other, raw.replace("🦊", "🐈")).unwrap();
    let mismatch = anno(&[
        "query",
        "source-span",
        saved.to_str().unwrap(),
        "0",
        "1",
        "--original",
        other.to_str().unwrap(),
    ]);
    assert!(!mismatch.status.success());
    assert!(mismatch.stdout.is_empty());

    // The validator must reject reattaching runs to an edition of the same byte length.
    let mut edited = doc;
    edited["text"] = Value::String(edited["text"].as_str().unwrap().replace("🦊", "🐈"));
    fs::write(&saved, serde_json::to_vec(&edited).unwrap()).unwrap();
    let rejected = anno(&[
        "enhance",
        saved.to_str().unwrap(),
        "--format",
        "grounded",
        "--quiet",
    ]);
    assert!(!rejected.status.success());
    assert!(rejected.stdout.is_empty());
}

#[test]
fn html_entity_and_split_tag_origins_survive_cleanup_and_reload() {
    let dir = tempfile::tempdir().unwrap();
    let source_file = dir.path().join("input.html");
    let saved = dir.path().join("grounded.json");
    let raw = "<p>🦊 A\u{30a}   al\u{200b}<b>ice</b>&#64;example.org</p><p>alice@example.org</p>";
    fs::write(&source_file, raw).unwrap();
    let doc = extract_and_reload(
        &source_file,
        &saved,
        &["--html-source-spans", "--clean", "--normalize"],
    );
    verify_domain(&doc, raw);
    let positions = check_repeated_signals(&doc, "alice@example.org");
    let (_, i) = origin(&doc, &saved, &source_file, positions[0] + 2);
    assert_eq!(i.start, raw.find("ice</b>").unwrap());
    let (kind, at) = origin(&doc, &saved, &source_file, positions[0] + 5);
    assert_ne!(kind, "direct");
    assert_eq!(&raw[at], "&#64;");
    let (kind, literal) = origin(&doc, &saved, &source_file, positions[1] + 5);
    assert_eq!(kind, "direct");
    assert_eq!(literal.start, raw.rfind('@').unwrap());
}

#[test]
fn ordinary_html_renderer_preserves_source_identity_and_reports_unmapped_origins() {
    let dir = tempfile::tempdir().unwrap();
    let source_file = dir.path().join("unmapped.html");
    let saved = dir.path().join("grounded.json");
    let paragraph = "Local volunteers surveyed the river each week and recorded the locations where they observed changes. Their notes describe the study area, the instruments they used, and the conditions during each visit.";
    let raw = format!("<html><head><title>River survey</title></head><body><article><h1>River survey</h1><p>{paragraph}</p><p>🔬 Contact <b>alice@example.org</b> to receive the report.</p><p>{paragraph}</p><p>{paragraph}</p></article></body></html>");
    fs::write(&source_file, &raw).unwrap();
    let doc = extract_and_reload(&source_file, &saved, &["--clean"]);
    verify_domain(&doc, &raw);
    assert!(doc["signals"]
        .as_array()
        .unwrap()
        .iter()
        .any(|s| s["surface"] == "alice@example.org"));
    let runs = doc["source_map"]["runs"].as_array().unwrap();
    assert!(!runs.is_empty());
    assert!(runs
        .iter()
        .all(|r| r["kind"] == "unmapped" && r["source_bytes"].is_null()));
}

#[test]
fn flat_json_and_full_export_share_the_analyzed_edition() {
    let dir = tempfile::tempdir().unwrap();
    let raw = "🦊  al\u{200b}ice@example.org";
    let saved = dir.path().join("full.json");
    let flat = successful(&[
        "extract",
        "--model",
        "pattern",
        "--text",
        raw,
        "--normalize",
        "--clean",
        "--format",
        "json",
        "--export",
        saved.to_str().unwrap(),
        "--quiet",
    ]);
    let doc = successful(&[
        "enhance",
        saved.to_str().unwrap(),
        "--format",
        "grounded",
        "--quiet",
    ]);
    verify_domain(&doc, raw);
    assert_eq!(flat["source_map"], doc["source_map"]);
    let text = doc["text"].as_str().unwrap();
    assert_eq!(text, "🦊 alice@example.org");
    let entity = &flat["entities"][0];
    assert_eq!(entity["text"], "alice@example.org");
    assert_eq!(entity["start"], 2);
    assert_eq!(
        doc["signals"][0]["location"]["Text"]["start"],
        entity["start"]
    );
}

#[test]
fn one_normalized_scalar_projects_to_disjoint_html_contributors() {
    let dir = tempfile::tempdir().unwrap();
    let source_file = dir.path().join("compound.html");
    let saved = dir.path().join("grounded.json");
    let raw = "<p>🦊 <b>e</b><i>&#x301;</i>!</p>";
    fs::write(&source_file, raw).unwrap();
    let doc = extract_and_reload(
        &source_file,
        &saved,
        &["--html-source-spans", "--normalize"],
    );
    let text = doc["text"].as_str().unwrap();
    let byte = text.find('é').unwrap();
    let start = text[..byte].chars().count();
    let projected = successful(&[
        "query",
        "source-span",
        saved.to_str().unwrap(),
        &start.to_string(),
        &(start + 1).to_string(),
        "--original",
        source_file.to_str().unwrap(),
    ]);
    let fragments = projected["fragments"].as_array().unwrap();
    assert_eq!(fragments.len(), 1, "{projected}");
    let run = &fragments[0];
    assert_eq!(bounds(&run["analyzed_bytes"]), byte..byte + 'é'.len_utf8());
    assert_eq!(run["kind"], "transformed");
    assert!(run["source_bytes"].is_null());
    let ranges: Vec<_> = run["source_ranges"]
        .as_array()
        .unwrap()
        .iter()
        .map(bounds)
        .collect();
    let letter = raw.find("e</b>").unwrap();
    let accent = raw.find("&#x301;").unwrap();
    assert_eq!(
        ranges,
        [letter..letter + 1, accent..accent + "&#x301;".len()]
    );
}
