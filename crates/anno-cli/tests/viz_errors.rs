use anno::core::grounded::{EvalComparison, EvalMatch};
use anno::viz::errors::EvalErrors;
use anno::{Location, Signal, SignalId};
use serde_json::{json, Value};
use std::io::Write;
use std::process::{Command, Output, Stdio};

fn eval(args: &[&str]) -> Vec<u8> {
    let out = Command::new(env!("CARGO_BIN_EXE_anno"))
        .args(["eval", "--json", "--model", "pattern"])
        .args(args)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

fn viz(report: &[u8], args: &[&str]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_anno"))
        .args(["viz", "errors"])
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(report).unwrap();
    child.wait_with_output().unwrap()
}

fn success(run: Output) -> String {
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8(run.stdout).unwrap()
}

#[test]
fn real_eval_separates_gold_coverage_and_extra_predictions() {
    let report = eval(&[
        "--text",
        "Mail x@y.com to Ada.",
        "--gold",
        "x@y.com:ORG:5:12",
        "--gold",
        "Ada:PER:16:19",
    ]);
    let svg = success(viz(&report, &[]));
    assert!(svg.starts_with("<svg "));
    assert!(svg.contains("0 exact matches, 1 same-span label errors, 0 partial overlaps, 1 missed gold entities and 0 extra predictions"));
    assert!(svg.contains("Model · pattern"));
    assert!(svg.contains("Same-span label confusions"));
    assert!(!svg.contains("Character masking"));

    let html = success(viz(&report, &["--format", "html"]));
    assert!(html.contains("Gold entities (2)</th>"));
    assert!(html.contains("Predicted entities (1)</th>"));
    assert!(html.contains("Same span, wrong label</th><td>1 (50.0%)</td><td>1 (100.0%)</td>"));
    assert!(html.contains("Missed gold / extra prediction</th><td>1 (50.0%)</td><td>0 (0.0%)</td>"));
    assert!(html.contains(">ORG</th><td>EMAIL</td><td>1</td>"));

    let gold_empty = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(
        gold_empty.path(),
        r#"{"text":"Mail x@y.com", "entities":[]}"#,
    )
    .unwrap();
    let no_gold = eval(&[
        "--text",
        "Mail x@y.com",
        "--gold-file",
        gold_empty.path().to_str().unwrap(),
    ]);
    let html = success(viz(&no_gold, &["--format", "html"]));
    assert!(html.contains("Missed gold / extra prediction</th><td>0 (—)</td><td>1 (100.0%)</td>"));
}

#[test]
fn masking_uses_unicode_union_counts_and_keeps_entity_diagnostics() {
    let report = eval(&[
        "--text",
        "📬 x@y.com now",
        "--gold",
        "x@y:ORG:2:5",
        "--metric",
        "char-masking",
    ]);
    let html = success(viz(&report, &["--format", "html"]));
    assert!(html.contains("Partial span overlap</th><td>1 (100.0%)</td><td>1 (100.0%)</td>"));
    assert!(html.contains("A partial span overlap may also involve a different label."));
    assert!(!html.contains("Same-span label confusions"));
    assert!(html.contains("Gold mask characters (3)</th>"));
    assert!(html.contains("Predicted mask characters (7)</th>"));
    assert!(html.contains("Covered by both masks</th><td>3 (100.0%)</td><td>3 (42.9%)</td>"));
    assert!(html.contains("Missed gold / extra masking</th><td>0 (0.0%)</td><td>4 (57.1%)</td>"));
    assert!(html.contains("13 Unicode scalar values in the text, 10 outside the gold mask"));

    let gold_empty = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(gold_empty.path(), r#"{"text":"🌍 hi", "entities":[]}"#).unwrap();
    let report = eval(&[
        "--text",
        "🌍 hi",
        "--gold-file",
        gold_empty.path().to_str().unwrap(),
        "--metric",
        "char-masking",
    ]);
    let doc = success(viz(&report, &["--format", "html"]));
    assert!(doc.contains("4 Unicode scalar values in the text, 4 outside the gold mask"));
    assert!(doc.contains("Exact span and label</th><td>0 (—)</td><td>0 (—)</td>"));
    assert!(doc.contains("Covered by both masks</th><td>0 (—)</td><td>0 (—)</td>"));
    assert!(!doc.contains("NaN"));
}

#[test]
fn malformed_reports_cannot_replace_an_existing_figure() {
    let report: Value = serde_json::from_slice(&eval(&[
        "--text",
        "Mail x@y.com to Ada.",
        "--gold",
        "x@y.com:ORG:5:12",
        "--gold",
        "Ada:PER:16:19",
        "--metric",
        "char-masking",
    ]))
    .unwrap();
    let destination = tempfile::NamedTempFile::new().unwrap();
    let path = destination.path();
    std::fs::write(path, b"previous figure").unwrap();
    for (pointer, value, expected) in [
        ("/matches/1/gold_id", json!(0), "duplicate gold ID"),
        ("/matches/0/pred_id", json!(null), "invalid"),
        (
            "/matches/0/gold_label",
            json!("EMAIL"),
            "identical gold and predicted labels",
        ),
        (
            "/matches/1",
            json!({"type":"false_positive","pred_id":0}),
            "duplicate predicted ID",
        ),
        ("/matches/1/type", json!("other"), "unknown variant"),
        ("/gold_count", json!(3), "gold_count is 3"),
        ("/predicted_count", json!(0), "predicted_count is 0"),
        ("/errors", json!(1), "errors is 1"),
        ("/correct", json!(1), "correct is 1"),
        ("/metric", json!("entity"), "entity metric contains"),
        (
            "/char_masking/counts/true_positive_chars",
            json!(8),
            "inconsistent char_masking",
        ),
        (
            "/char_masking/counts/non_pii_chars",
            json!(0),
            "inconsistent char_masking",
        ),
        ("/char_masking", Value::Null, "requires char_masking.counts"),
        (
            "/matches/0",
            json!({"type":"boundary_error","pred_id":0,"gold_id":0,"iou":0}),
            "iou",
        ),
    ] {
        let mut bad = report.clone();
        *bad.pointer_mut(pointer).unwrap() = value;
        let run = viz(
            &serde_json::to_vec(&bad).unwrap(),
            &["-o", path.to_str().unwrap()],
        );
        assert!(!run.status.success(), "{pointer}");
        assert!(
            String::from_utf8_lossy(&run.stderr).contains(expected),
            "{}: {}",
            pointer,
            String::from_utf8_lossy(&run.stderr)
        );
        assert!(run.stdout.is_empty());
        assert_eq!(std::fs::read(path).unwrap(), b"previous figure");
    }
}

#[test]
fn sparse_ids_and_escaped_labels_work_in_the_native_library() {
    let signals = |id, label| vec![Signal::new(id, Location::text(0, 1), "A", label, 1.0)];
    let mut comparison =
        EvalComparison::compare("A", signals(17_u64, "P<&"), signals(1024, "OTHER"));
    let chart = EvalErrors::from_comparison(&comparison).unwrap();
    assert!(chart
        .render_html()
        .contains(">P&lt;&amp;</th><td>OTHER</td><td>1</td>"));
    assert!(chart.render_svg().contains("P&lt;&amp;"));
    comparison.matches[0] = EvalMatch::Correct {
        gold_id: SignalId::new(17),
        pred_id: SignalId::new(0),
    };
    assert!(EvalErrors::from_comparison(&comparison)
        .unwrap_err()
        .contains("predicted signals"));
}

#[test]
fn report_stream_pools_counts_and_keeps_ids_local_to_each_evaluation() {
    let first = eval(&[
        "--text",
        "Mail x@y.com to Ada.",
        "--gold",
        "x@y.com:ORG:5:12",
        "--gold",
        "Ada:PER:16:19",
    ]);
    let second = eval(&["--text", "Mail x@y.com", "--gold", "x@y.com:EMAIL:5:12"]);
    let pooled = [first.as_slice(), second.as_slice()].concat();
    let html = success(viz(&pooled, &["--format", "html"]));
    assert!(html.contains("2 evaluation reports · Model · pattern"));
    assert!(html.contains("Gold entities (3)</th>"));
    assert!(html.contains("Predicted entities (2)</th>"));
    assert!(html.contains("Exact span and label</th><td>1 (33.3%)</td><td>1 (50.0%)</td>"));
    assert!(html.contains("Same span, wrong label</th><td>1 (33.3%)</td><td>1 (50.0%)</td>"));
    assert!(html.contains("Missed gold / extra prediction</th><td>1 (33.3%)</td><td>0 (0.0%)</td>"));

    let mask = eval(&[
        "--text",
        "📬 x@y.com now",
        "--gold",
        "x@y:ORG:2:5",
        "--metric",
        "char-masking",
    ]);
    let mixed = [second.as_slice(), mask.as_slice()].concat();
    let failure = viz(&mixed, &[]);
    assert!(!failure.status.success());
    assert!(failure.stdout.is_empty());
    assert!(String::from_utf8_lossy(&failure.stderr)
        .contains("report 2: cannot pool entity and char-masking"));

    let repeated = [mask.as_slice(), mask.as_slice()].concat();
    let html = success(viz(&repeated, &["--format", "html"]));
    assert!(html.contains("Gold mask characters (6)</th>"));
    assert!(html.contains("Predicted mask characters (14)</th>"));
    assert!(html.contains("26 Unicode scalar values in the text, 20 outside the gold mask"));
    assert!(html.contains("Covered by both masks</th><td>6 (100.0%)</td><td>6 (42.9%)</td>"));
    let mut extreme: Value = serde_json::from_slice(&mask).unwrap();
    extreme["char_masking"]["counts"] = json!({
        "total_chars": u64::MAX, "gold_chars": 0, "non_pii_chars": u64::MAX,
        "true_positive_chars": 0, "predicted_chars": 0
    });
    let overflow = [serde_json::to_vec(&extreme).unwrap(), mask].concat();
    let failure = viz(&overflow, &[]);
    assert!(!failure.status.success());
    assert!(String::from_utf8_lossy(&failure.stderr).contains("pooled character count exceeds"));
}

#[test]
fn portable_summary_round_trips_the_real_stream_and_rejects_changed_counts() {
    let report = eval(&[
        "--text",
        "Mail x@y.com to Ada.",
        "--gold",
        "x@y.com:ORG:5:12",
        "--gold",
        "Ada:PER:16:19",
        "--metric",
        "char-masking",
    ]);
    let reports = [report.as_slice(), report.as_slice()].concat();
    let destination = tempfile::tempdir().unwrap();
    let path = destination.path().join("errors.json");
    assert!(success(viz(&reports, &["-o", path.to_str().unwrap()])).is_empty());
    let portable = std::fs::read(&path).unwrap();
    let mut summary: Value = serde_json::from_slice(&portable).unwrap();
    assert_eq!(summary["schema"], "anno.errors/v1");
    assert_eq!(summary["reports"], 2);
    assert_eq!(
        summary["entities"],
        json!({
            "gold_count": 4, "predicted_count": 2, "exact": 0, "same_span_wrong_label": 2,
            "partial_overlap": 0, "missed_gold": 2, "extra_predictions": 0
        })
    );
    assert_eq!(
        summary["same_span_label_confusions"],
        json!([{"gold":"ORG", "predicted":"EMAIL", "pairs":2}])
    );
    assert!(summary.get("matches").is_none());
    assert!(!String::from_utf8_lossy(&portable).contains("gold_id"));
    assert_eq!(success(viz(&reports, &[])), success(viz(&portable, &[])));
    assert_eq!(
        success(viz(&reports, &["--format", "html"])),
        success(viz(&portable, &["--format", "html"]))
    );

    let titled = success(viz(
        &reports,
        &["--title", "MISC <&> ORG", "--format", "json"],
    ));
    let named: Value = serde_json::from_str(&titled).unwrap();
    assert_eq!(named["title"], "MISC <&> ORG");
    summary["title"] = named["title"].clone();
    summary["metadata"] = json!({"occurrences":2});
    let custom = serde_json::to_vec(&summary).unwrap();
    assert!(success(viz(&custom, &[]))
        .contains("<title id=\"eval-title\">MISC &lt;&amp;&gt; ORG</title>"));
    let exported: Value =
        serde_json::from_str(&success(viz(&custom, &["--format", "json"]))).unwrap();
    assert_eq!(exported, summary);
    let doubled = [custom.as_slice(), custom.as_slice()].concat();
    let combined: Value =
        serde_json::from_str(&success(viz(&doubled, &["--format", "json"]))).unwrap();
    assert_eq!(combined["reports"], 4);
    assert!(combined.get("metadata").is_none());
    let pairs = summary["same_span_label_confusions"][0].clone();
    for (key, value, expected) in [
        (
            "/schema",
            json!("anno.errors/v9"),
            "unsupported error summary schema",
        ),
        ("/reports", json!(0), "at least one"),
        ("/entities/predicted_count", json!(3), "do not reconcile"),
        ("/entities/exact", json!(u64::MAX), "counts overflow"),
        ("/same_span_label_confusions", json!([]), "do not sum"),
        (
            "/same_span_label_confusions",
            json!([pairs.clone(), pairs]),
            "duplicate label confusion",
        ),
        (
            "/same_span_label_confusions/0/pairs",
            json!(0),
            "positive count",
        ),
        (
            "/same_span_label_confusions/0/gold",
            json!("EMAIL"),
            "different labels",
        ),
        (
            "/char_masking/counts/predicted_chars",
            json!(u64::MAX),
            "inconsistent char_masking",
        ),
    ] {
        let mut bad = summary.clone();
        *bad.pointer_mut(key).unwrap() = value;
        let run = viz(&serde_json::to_vec(&bad).unwrap(), &[]);
        assert!(!run.status.success(), "{key}");
        assert!(run.stdout.is_empty());
        assert!(
            String::from_utf8_lossy(&run.stderr).contains(expected),
            "{key}: {}",
            String::from_utf8_lossy(&run.stderr)
        );
    }
}
