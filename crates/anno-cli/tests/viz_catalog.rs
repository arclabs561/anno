//! Discover and render local figures from an ordinary directory with no model access.
use serde_json::{json, Value};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Output, Stdio};

fn viz(dir: &Path, args: &[&str], input: &[u8], catalog: Option<&str>) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_anno"));
    command
        .current_dir(dir)
        .args(["viz"])
        .args(args)
        .env("ANNO_NO_DOWNLOADS", "1")
        .env_remove("ANNO_FIGURE_CATALOG");
    if let Some(catalog) = catalog {
        command.env("ANNO_FIGURE_CATALOG", catalog);
    }
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(input).unwrap();
    child.wait_with_output().unwrap()
}

fn success(result: Output) -> Vec<u8> {
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    result.stdout
}

fn trial() -> Value {
    json!({"schema":"anno.tradeoff/v1","title":"Local study","unit":"rows","metadata":{"source_hash":"test-sha256"},
        "series":[{"label":"Dev","observations":2,"gold":1,"points":[{"label":"Stock","correct":1,"unmatched":0}]}]})
}

#[test]
fn unbundled_cli_renders_files_and_stdin_with_no_catalog() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path();
    for args in [vec!["list"], vec!["render", "unknown-study"]] {
        let failure = viz(path, &args, &[], None);
        assert!(!failure.status.success());
        let message = String::from_utf8_lossy(&failure.stderr);
        assert!(
            message.contains("dataset-figures") && message.contains("--catalog"),
            "{message}"
        );
    }
    for (view, portable) in [
        ("tradeoff", trial()),
        (
            "paired",
            json!({"schema":"anno.paired/v1","title":"Change","unit":"rows","baseline":"Stock","candidate":"Trial",
            "rows":[{"label":"Dev","role":"Development","observations":2,"baseline":{"correct":1,"unmatched":0},
                     "candidate":{"correct":1,"unmatched":1},"gained":0,"lost":0}]}),
        ),
        (
            "overlap",
            json!({"schema":"anno.overlap/v1","title":"Spans","documents":1,"documents_with_overlap":0,
            "signals":1,"signals_with_overlap":0,"characters":1,"coverage":{"zero":0,"one":1,"two":0,"three_or_more":0},
            "labels":{"X":{"signals":1,"signals_with_overlap":0}}}),
        ),
        (
            "errors",
            json!({"schema":"anno.errors/v1","title":"Errors","reports":1,"metric":"entity",
            "prediction_source":{"kind":"saved_documents"},"entities":{"gold_count":1,"predicted_count":1,"exact":1,
            "same_span_wrong_label":0,"partial_overlap":0,"missed_gold":0,"extra_predictions":0},
            "same_span_label_confusions":[]}),
        ),
    ] {
        let data = serde_json::to_vec(&portable).unwrap();
        let svg = success(viz(path, &[view], &data, None));
        assert_eq!(success(viz(path, &["render"], &data, None)), svg);
        fs::write(path.join("counts.json"), &data).unwrap();
        // A broken catalog override has no bearing on an explicitly supplied report.
        assert_eq!(
            success(viz(
                path,
                &["render", "counts.json"],
                &[],
                Some("missing.json")
            )),
            svg
        );
    }
}

#[test]
fn published_studies_render_from_an_external_catalog() {
    let reports = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/eval");
    // The crate archive intentionally excludes the repository's research data.
    // The standalone fixtures above cover every schema in either distribution.
    if !reports.is_dir() {
        return;
    }
    let published = reports.join("figure-catalog.json");
    assert!(
        published.is_file(),
        "expected published figure catalog in the repository"
    );
    let catalog = published.to_str();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path();
    let list: Vec<Value> = serde_json::from_slice(&success(viz(
        path,
        &[
            "list",
            "--dataset",
            "wikigold",
            "--json",
            "--catalog",
            published.to_str().unwrap(),
        ],
        &[],
        None,
    )))
    .unwrap();
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let discovered: Vec<Value> = serde_json::from_slice(&success(viz(
        &checkout,
        &["list", "--dataset", "wikigold", "--json"],
        &[],
        None,
    )))
    .unwrap();
    assert_eq!(discovered, list);
    assert!(list.iter().any(|s| s["id"] == "wikigold-routing-paired"));
    for study in &list {
        assert!(study["datasets"]
            .as_array()
            .unwrap()
            .contains(&json!("WikiGold")));
        assert!(!study["title"].as_str().unwrap().is_empty());
        assert!(study["guide"].as_str().unwrap().starts_with("docs/"));
        assert!(study["report"].is_null());
    }
    let none: Vec<Value> = serde_json::from_slice(&success(viz(
        path,
        &["list", "--json", "--dataset", "absent corpus"],
        &[],
        catalog,
    )))
    .unwrap();
    assert!(none.is_empty());
    let alias = success(viz(path, &["list", "--dataset", "WNUT-17"], &[], catalog));
    assert!(String::from_utf8_lossy(&alias).contains("wikigold-routing-paired"));

    for (id, view) in [
        ("wikigold-entity-errors", "errors"),
        ("genia-span-overlap", "overlap"),
        ("crossre-full-test-paired", "paired"),
        ("crossre-full-test-tradeoff", "tradeoff"),
    ] {
        let data = success(viz(path, &["render", id, "--format", "json"], &[], catalog));
        let report: Value = serde_json::from_slice(&data).unwrap();
        assert_eq!(report["schema"], format!("anno.{view}/v1"));
        let svg = success(viz(path, &[view], &data, None));
        assert_eq!(success(viz(path, &["render"], &data, None)), svg);
        assert_eq!(success(viz(path, &["render", id], &[], catalog)), svg);
        let html = success(viz(path, &[view, "--format", "html"], &data, None));
        assert!(success(viz(
            path,
            &["render", "-", "-o", "figure.HTML"],
            &data,
            None
        ))
        .is_empty());
        assert_eq!(fs::read(path.join("figure.HTML")).unwrap(), html);
    }
}

#[test]
fn downloaded_catalog_file_precedence_and_safe_output() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path();
    let report = trial();
    let minimal =
        json!({"schema":"anno.figure-catalog/v1", "studies":[{"id":"personal","report":report}]});
    fs::write(path.join("personal.json"), minimal.to_string()).unwrap();
    assert_eq!(
        success(viz(
            path,
            &["list", "--catalog", "personal.json"],
            &[],
            None
        )),
        b"personal\n  Local study\n"
    );
    let catalog = json!({"schema":"anno.figure-catalog/v1","ci_survey_day":"2026-11-27","datasets":[],
        "studies":[{"id":"local.v1","datasets":["WikiGold", "ENer"],"guide":"docs/QA.md","report":report},
                   {"id":"other","datasets":["ENER"],"guide":"docs/QA.md","report":report}]});
    fs::write(path.join("catalog.json"), catalog.to_string()).unwrap();
    let exact = success(viz(
        path,
        &["list", "--dataset", "ENER", "--json"],
        &[],
        Some("catalog.json"),
    ));
    let exact: Vec<Value> = serde_json::from_slice(&exact).unwrap();
    assert_eq!(exact.len(), 1);
    assert_eq!(exact[0]["id"], "other");
    let listed = success(viz(
        path,
        &["list", "--catalog", "catalog.json"],
        &[],
        Some("missing.json"),
    ));
    assert!(String::from_utf8_lossy(&listed).contains("local.v1"));
    let output = success(viz(
        path,
        &["render", "local.v1", "--format", "json"],
        &[],
        Some("catalog.json"),
    ));
    assert_eq!(serde_json::from_slice::<Value>(&output).unwrap(), report);

    fs::write(path.join("figure-catalog.json"), catalog.to_string()).unwrap();
    assert_eq!(success(viz(path, &["list"], &[], None)), listed);
    assert_eq!(
        success(viz(
            path,
            &["render", "local.v1", "--format", "json"],
            &[],
            None
        )),
        output
    );
    let not_found = viz(path, &["render", "unknown-study"], &[], None);
    assert!(!not_found.status.success());
    assert!(String::from_utf8_lossy(&not_found.stderr).contains("anno viz list"));

    // Aliases to the auto-discovered bundle are protected even if the output
    // uses a different name. The named input consults that same catalog.
    fs::hard_link(
        path.join("figure-catalog.json"),
        path.join("catalog-alias.json"),
    )
    .unwrap();
    for (target, configured) in [
        ("figure-catalog.json", None),
        ("catalog-alias.json", None),
        ("catalog.json", Some("catalog.json")),
    ] {
        let bytes = fs::read(path.join(target)).unwrap();
        let failure = viz(path, &["render", "other", "-o", target], &[], configured);
        assert!(
            !failure.status.success()
                && String::from_utf8_lossy(&failure.stderr).contains("same file")
        );
        assert_eq!(fs::read(path.join(target)).unwrap(), bytes);
    }
    let mut changed = report.clone();
    changed["title"] = json!("Existing input wins");
    fs::write(path.join("local.v1"), changed.to_string()).unwrap();
    let rendered = success(viz(
        path,
        &["render", "local.v1"],
        &[],
        Some("missing.json"),
    ));
    assert!(String::from_utf8_lossy(&rendered).contains("Existing input wins"));
    let same = viz(path, &["render", "local.v1", "-o", "local.v1"], &[], None);
    assert!(!same.status.success() && String::from_utf8_lossy(&same.stderr).contains("same file"));
    assert_eq!(
        fs::read_to_string(path.join("local.v1")).unwrap(),
        changed.to_string()
    );
}

#[test]
fn invalid_reports_and_names_explain_the_next_command_and_preserve_outputs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path();
    fs::write(path.join("saved.svg"), "last good chart").unwrap();
    for (input, expected) in [
        (r#"{"schema":"anno.other/v3"}"#, "Unknown figure schema"),
        (
            r#"{"schema":"anno.tradeoff/v1","title":"Bad","unit":"rows","series":[{"label":"Test","observations":1,"gold":0,"points":[{"label":"Trial","correct":1,"unmatched":0}]}]}"#,
            "points[0].correct",
        ),
        (r#"{"reports":[]}"#, "anno viz errors"),
        ("{}\n{}", "anno viz paired --against"),
        ("[]", "anno viz overlap"),
    ] {
        let output = viz(
            path,
            &["render", "--format", "json", "-o", "saved.svg"],
            input.as_bytes(),
            None,
        );
        assert!(!output.status.success() && output.stdout.is_empty());
        assert!(String::from_utf8_lossy(&output.stderr).contains(expected));
        assert_eq!(
            fs::read_to_string(path.join("saved.svg")).unwrap(),
            "last good chart"
        );
    }
}
