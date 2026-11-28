use anno::{GroundedDocument, Location, Signal};
use serde_json::{json, Value};
use std::io::Write;
use std::process::{Command, Output, Stdio};

fn run(args: &[&str], input: &[u8]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_anno"))
        .args(args)
        .env("ANNO_NO_DOWNLOADS", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(input).unwrap();
    child.wait_with_output().unwrap()
}

fn success(out: Output) -> Vec<u8> {
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

fn document(text: &str, signals: Vec<(Location, &str, &str)>) -> Value {
    let mut doc = GroundedDocument::new("private-document-id", text);
    for (location, surface, label) in signals {
        doc.add_signal(Signal::new(0, location, surface, label, 0.9));
    }
    let mut value = serde_json::to_value(doc).unwrap();
    for (i, signal) in value["signals"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        signal["id"] = json!(9_001 - i * 11);
    }
    value
}

fn summary(bytes: &[u8]) -> Value {
    serde_json::from_slice(&success(run(
        &["viz", "overlap", "--format", "json"],
        bytes,
    )))
    .unwrap()
}

#[test]
fn unicode_segments_duplicate_signals_and_unannotated_documents_pool_by_character() {
    let docs = vec![
        document(
            "🍊ABéCλDXY",
            vec![
                (
                    Location::Discontinuous {
                        segments: vec![(0, 2), (5, 7)],
                    },
                    "🍊A λD",
                    "Outer<&>",
                ),
                (Location::text(1, 4), "ABé", "Repeated"),
                (Location::text(1, 4), "ABé", "Repeated"),
                (Location::text(4, 5), "C", "Gap"),
                (Location::text(2, 3), "B", "Repeated"),
                (Location::text(3, 6), "éCλ", "Crossing"),
            ],
        ),
        document(
            "🍐abc",
            vec![
                (
                    Location::Discontinuous {
                        segments: vec![(0, 1), (3, 4)],
                    },
                    "🍐 c",
                    "Outer<&>",
                ),
                (Location::text(1, 3), "ab", "Gap"),
            ],
        ),
        document("", vec![]),
        document("xyz", vec![]),
    ];
    let encoded = serde_json::to_vec(&docs).unwrap();
    let jsonl = docs
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    let aggregate = summary(&encoded);
    assert_eq!(aggregate, summary(jsonl.as_bytes()));
    let grounded: Vec<GroundedDocument> = serde_json::from_slice(&encoded).unwrap();
    let native = anno::viz::overlap::Overlap::from_documents(&grounded).unwrap();
    assert_eq!(
        aggregate,
        serde_json::from_str::<Value>(&native.to_json().unwrap()).unwrap()
    );
    assert_eq!(aggregate["schema"], "anno.overlap/v1");
    assert_eq!(aggregate["documents"], 4);
    assert_eq!(aggregate["documents_with_overlap"], 1);
    assert_eq!(aggregate["signals"], 8);
    assert_eq!(aggregate["signals_with_overlap"], 6);
    assert_eq!(aggregate["characters"], 16);
    assert_eq!(
        aggregate["coverage"],
        json!({"zero":5,"one":6,"two":2,"three_or_more":3})
    );
    assert_eq!(
        aggregate["labels"]["Outer<&>"],
        json!({"signals":2,"signals_with_overlap":1})
    );
    assert_eq!(
        aggregate["labels"]["Repeated"],
        json!({"signals":3,"signals_with_overlap":3})
    );
    let compact = serde_json::to_vec(&aggregate).unwrap();
    let svg = success(run(&["viz", "overlap"], &compact));
    assert_eq!(svg, success(run(&["viz", "overlap"], &encoded)));
    let svg = String::from_utf8(svg).unwrap();
    assert!(svg.contains("Outer&lt;&amp;&gt;"));
    assert!(svg.contains("1 / 4 documents (25.0%)") && svg.contains("6 / 8 signals (75.0%)"));
    let html = String::from_utf8(success(run(
        &["viz", "overlap", "--format", "html"],
        &compact,
    )))
    .unwrap();
    assert!(html.contains(r#"<th scope="row">Repeated</th><td data-label="Sharing">3</td><td data-label="All signals">3</td><td data-label="Share of label">100.0%</td>"#));
    for artifact in [&svg, &html, &aggregate.to_string()] {
        for secret in ["private-document-id", "🍊", "🍐", "ABé", "éCλ", "xyz"] {
            assert!(!artifact.contains(secret));
        }
    }
}

#[test]
fn legacy_segments_of_the_same_signal_and_empty_populations() {
    let doc = document(
        "𝕏abcd",
        vec![(
            Location::Discontinuous {
                segments: vec![(2, 4), (0, 3)],
            },
            "𝕏abc",
            "Segment",
        )],
    );
    let result = summary(&serde_json::to_vec(&doc).unwrap());
    assert_eq!(
        result["coverage"],
        json!({"zero":1,"one":4,"two":0,"three_or_more":0})
    );
    assert_eq!(result["signals_with_overlap"], 0);
    let empty = summary(b"[]");
    assert_eq!(empty, summary(b""));
    assert_eq!(empty["documents"], 0);
    assert_eq!(empty["signals"], 0);
    let html =
        String::from_utf8(success(run(&["viz", "overlap", "--format", "html"], b"[]"))).unwrap();
    assert!(html.contains(r#"<th scope="row">Documents</th><td data-label="Sharing">0</td><td data-label="All">0</td><td data-label="Share">—</td>"#));
    assert!(!html.contains("NaN"));
}

#[test]
fn rare_depth_and_label_rankings_use_their_named_denominators() {
    let report = json!({
        "schema": "anno.overlap/v1", "title": "Release overlap",
        "documents": 1, "documents_with_overlap": 1,
        "signals": 1050, "signals_with_overlap": 65, "characters": 100000,
        "coverage": {"zero": 99000, "one": 950, "two": 49, "three_or_more": 1},
        "labels": {
            "Common": {"signals": 1000, "signals_with_overlap": 60},
            "Rare": {"signals": 50, "signals_with_overlap": 5}
        }
    });
    let bytes = serde_json::to_vec(&report).unwrap();
    let html = String::from_utf8(success(run(
        &["viz", "overlap", "--format", "html"],
        &bytes,
    )))
    .unwrap();
    assert!(html.contains(r#"<th scope="row">3+ signals</th><td data-label="Characters">1</td><td data-label="Of all text">under 0.01%</td><td data-label="Of shared text">2.00%</td>"#));
    let table = |name: &str| {
        let pos = html.find(name).unwrap();
        &html[pos..pos + html[pos..].find("</table>").unwrap()]
    };
    let share = table("<caption>Labels by affected share</caption>");
    let count = table("<caption>Labels by affected count</caption>");
    assert!(
        share.find(r#"<th scope="row">Rare</th>"#).unwrap()
            < share.find(r#"<th scope="row">Common</th>"#).unwrap()
    );
    assert!(
        count.find(r#"<th scope="row">Common</th>"#).unwrap()
            < count.find(r#"<th scope="row">Rare</th>"#).unwrap()
    );
}

#[test]
fn actual_extract_and_saved_pipeline_documents_render_and_round_trip() {
    let text = "📬 Write to ada@example.org.";
    let extract = success(run(
        &[
            "extract", "--model", "pattern", "--format", "grounded", "--text", text,
        ],
        &[],
    ));
    let measured = summary(&extract);
    assert_eq!(measured["signals"], 1);
    assert_eq!(measured["characters"], text.chars().count());
    assert_eq!(measured["coverage"]["one"], 15);
    let dir = tempfile::tempdir().unwrap();
    for format in ["json", "jsonl"] {
        let saved = dir.path().join(format);
        success(run(
            &[
                "pipeline",
                "--model",
                "pattern",
                "--format",
                format,
                "-o",
                saved.to_str().unwrap(),
                text,
                "quiet",
            ],
            &[],
        ));
        let report = dir.path().join("counts.json");
        success(run(
            &[
                "viz",
                "overlap",
                saved.to_str().unwrap(),
                "--title",
                "Mail study",
                "-o",
                report.to_str().unwrap(),
            ],
            &[],
        ));
        let counts: Value = serde_json::from_slice(&std::fs::read(&report).unwrap()).unwrap();
        assert_eq!(counts["title"], "Mail study");
        assert_eq!(counts["documents"], 2);
        assert_eq!(counts["signals"], 1);
        assert_eq!(counts["characters"], text.chars().count() + 5);
        let svg = success(run(&["viz", "overlap", report.to_str().unwrap()], &[]));
        assert!(String::from_utf8(svg).unwrap().contains("Mail study"));
    }
}

#[test]
fn inconsistent_summaries_and_malformed_documents_do_not_publish_partial_charts() {
    let dir = tempfile::tempdir().unwrap();
    let chart = dir.path().join("existing.svg");
    std::fs::write(&chart, b"previous figure").unwrap();
    let mut wrong = summary(b"[]");
    wrong["coverage"]["zero"] = json!(1);
    let mut wrong_label = summary(b"[]");
    wrong_label["labels"] = json!({"X":{"signals":1,"signals_with_overlap":0}});
    let mut bad_doc = document("é", vec![(Location::text(0, 1), "é", "Letter")]);
    bad_doc["signals"][0]["location"]["Text"]["end"] = json!(2);
    for input in [
        wrong.to_string(),
        wrong_label.to_string(),
        bad_doc.to_string(),
        format!("{}\n{}", summary(b"[]"), document("", vec![])),
    ] {
        let out = run(
            &["viz", "overlap", "-o", chart.to_str().unwrap()],
            input.as_bytes(),
        );
        assert!(!out.status.success());
        assert!(out.stdout.is_empty());
        assert_eq!(std::fs::read(&chart).unwrap(), b"previous figure");
    }
}
