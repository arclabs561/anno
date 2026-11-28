use anno::core::grounded::GroundedDocument;
use anno::viz::paired::{Paired, PairedRow};
use anno::{Location, Signal};
use serde_json::{json, Value};
use std::fs;
use std::path::Path;
use std::process::{Command, Output};

fn cli(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_anno"))
        .current_dir(dir)
        .env("ANNO_NO_DOWNLOADS", "1")
        .args(args)
        .output()
        .unwrap()
}

fn success(result: Output) -> String {
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    String::from_utf8(result.stdout).unwrap()
}

fn portable() -> Value {
    json!({"schema":"anno.paired/v1","title":"Review <&>", "unit":"sentences",
      "baseline":"Stock","candidate":"Trial","rows":[{"label":"Test","role":"Fresh holdout","observations":3,
      "baseline":{"correct":2,"unmatched":1,"calls":0,"median_ms":1.5},
      "candidate":{"correct":2,"unmatched":2,"calls":3,"median_ms":3.5},"gained":1,"lost":1,
      "interval":{"lower":-1,"upper":2,"method":"95% paired resampling"}}]})
}

#[test]
fn portable_counts_persist_and_formats_name_the_operating_costs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path();
    fs::write(path.join("study.json"), portable().to_string()).unwrap();
    let misplaced_unit = cli(
        path,
        &["viz", "paired", "study.json", "--unit", "sentences"],
    );
    assert!(!misplaced_unit.status.success());
    assert!(String::from_utf8_lossy(&misplaced_unit.stderr).contains("--against"));
    let svg = success(cli(path, &["viz", "paired", "study.json"]));
    for name in [
        "Review &lt;&amp;&gt;",
        "Baseline Stock compared with candidate Trial",
        "Fresh holdout",
        "Retained 1 of 2",
        "Gained 1 · lost 1",
        "Total suggestions",
        "Model calls",
        "Median latency (ms)",
        "95% paired resampling",
    ] {
        assert!(svg.contains(name), "{name}");
    }
    success(cli(
        path,
        &["viz", "paired", "study.json", "-o", "report.html"],
    ));
    let html = fs::read_to_string(path.join("report.html")).unwrap();
    assert!(
        html.contains("<th scope=\"col\">Baseline<br>Stock</th>")
            && html.contains(
                "<th scope=\"row\">Total suggestions</th><td>3</td><td>4</td><td>+1</td>"
            )
    );
    let json = success(cli(
        path,
        &[
            "viz",
            "paired",
            "study.json",
            "--title",
            "New title",
            "--format",
            "json",
        ],
    ));
    let mut expected = portable();
    expected["title"] = json!("New title");
    assert_eq!(
        Paired::from_json(json.as_bytes()).unwrap(),
        Paired::from_json(expected.to_string().as_bytes()).unwrap()
    );
}

#[test]
fn signed_match_labels_agree_at_one_zero_and_count_limits() {
    for (before, after, label) in [
        (0, 1, "+1 net exact match"),
        (1, 0, "−1 net exact match"),
        (1, 1, "0 net exact matches"),
        (0, u64::MAX, "+18,446,744,073,709,551,615 net exact matches"),
        (u64::MAX, 0, "−18,446,744,073,709,551,615 net exact matches"),
    ] {
        let mut v = portable();
        let row = &mut v["rows"][0];
        row["observations"] = json!(1);
        row["baseline"] = json!({"correct":before,"unmatched":0});
        row["candidate"] = json!({"correct":after,"unmatched":0});
        row["gained"] = json!(after.saturating_sub(before));
        row["lost"] = json!(before.saturating_sub(after));
        row.as_object_mut().unwrap().remove("interval");
        let report = Paired::from_json(v.to_string().as_bytes()).unwrap();
        for view in [report.render_svg().unwrap(), report.render_html().unwrap()] {
            assert!(view.contains(&format!("{label} ·")), "{label}");
            assert!(!view.contains("1 net exact matches"));
            assert!(view.contains("1 sentence") && !view.contains("1 sentences"));
        }
    }
}

#[test]
fn invalid_paired_claims_fail_before_render() {
    for (field, value, diagnostic) in [
        ("/rows/0/lost", json!(3), "lost exceeds"),
        ("/rows/0/gained", json!(15), "plus gained"),
        ("/rows/0/baseline/unmatched", json!(u64::MAX), "exceeds u64"),
        ("/rows/0/candidate/median_ms", json!(-0.1), "nonnegative"),
        ("/rows/0/interval/lower", json!(3), "ordered"),
        ("/rows/0/observations", json!(0), "empty cohorts"),
    ] {
        let mut v = portable();
        *v.pointer_mut(field).unwrap() = value;
        assert!(
            Paired::from_json(v.to_string().as_bytes())
                .unwrap_err()
                .contains(diagnostic),
            "{field}"
        );
    }
    let mut v = portable();
    let row = &mut v["rows"][0];
    row["baseline"] = json!({"correct":0,"unmatched":0});
    row["candidate"] = row["baseline"].clone();
    for key in ["gained", "lost", "observations"] {
        row[key] = json!(0);
    }
    row.as_object_mut().unwrap().remove("interval");
    let svg = Paired::from_json(v.to_string().as_bytes())
        .unwrap()
        .render_svg()
        .unwrap();
    assert!(svg.contains("0 of 0 baseline matches (undefined share)"));
    assert!(!svg.contains("Model calls"));
}

fn scored(dir: &Path, path: &str, who: &str, start: usize, gold: &[&str]) -> Value {
    let text = "Ada met Zoë.";
    let mut doc = GroundedDocument::new("same-edition", text);
    doc.add_signal(Signal::new(
        0,
        Location::text(start, start + 3),
        who,
        "PER",
        1.0,
    ));
    fs::write(dir.join(path), serde_json::to_vec(&doc).unwrap()).unwrap();
    let mut args = vec!["eval", "--text", text, "--predictions", path, "--json"];
    for spec in gold {
        args.extend(["--gold", spec]);
    }
    serde_json::from_str(&success(cli(dir, &args))).unwrap()
}

#[test]
fn native_eval_pairs_by_source_and_gold_across_reordered_streams() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path();
    let ada = "Ada:PER:0:3";
    let zoe = "Zoë:PER:8:11";
    let base = scored(path, "base.grounded", "Ada", 0, &[ada, ada, zoe]);
    let new = scored(path, "new.grounded", "Zoë", 8, &[zoe, ada, ada]);
    assert_eq!(base["pairing"]["document_id"], "same-edition");
    assert_eq!(
        base["pairing"]["gold_keys"]["0"],
        new["pairing"]["gold_keys"]["1"]
    );
    assert_eq!(
        base["pairing"]["gold_keys"]["0"],
        base["pairing"]["gold_keys"]["1"]
    );
    assert!(!base["pairing"].to_string().contains("Zoë"));
    let live: Value = serde_json::from_str(&success(cli(
        path,
        &[
            "eval",
            "--model",
            "pattern",
            "--text",
            "x@y.com",
            "--gold",
            "x@y.com:EMAIL:0:7",
            "--json",
            "--pair-id",
            "another",
        ],
    )))
    .unwrap();
    assert_eq!(live["pairing"]["document_id"], "another");
    fs::write(path.join("stock.jsonl"), format!("{base}\n{live}")).unwrap();
    fs::write(path.join("trial.jsonl"), format!("{live}\n{new}")).unwrap();
    let summary = success(cli(
        path,
        &[
            "viz",
            "paired",
            "stock.jsonl",
            "--against",
            "trial.jsonl",
            "--cohort",
            "Test",
            "--unit",
            "sentences",
            "--role",
            "Fresh",
            "--baseline-name",
            "Stock",
            "--candidate-name",
            "Trial",
            "--format",
            "json",
        ],
    ));
    let report = Paired::from_json(summary.as_bytes()).unwrap();
    assert_eq!(report.unit, "sentences");
    let row = &report.rows[0];
    assert_eq!(
        (
            row.observations,
            row.baseline.correct,
            row.candidate.correct,
            row.gained,
            row.lost
        ),
        (2, 2, 2, 1, 1)
    );
    // A saved-predictions report carries no latency, so a partial median is omitted.
    assert!(row.baseline.median_ms.is_none() && row.interval.is_none());
    let invalid = cli(
        path,
        &[
            "viz",
            "paired",
            "stock.jsonl",
            "--against",
            "trial.jsonl",
            "--unit",
            " ",
        ],
    );
    assert!(!invalid.status.success());
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("unit must contain a label"));
    let original = fs::read(path.join("trial.jsonl")).unwrap();
    let alias = cli(
        path,
        &[
            "viz",
            "paired",
            "stock.jsonl",
            "--against",
            "trial.jsonl",
            "-o",
            "trial.jsonl",
        ],
    );
    assert!(!alias.status.success());
    assert_eq!(fs::read(path.join("trial.jsonl")).unwrap(), original);

    let base = base.to_string();
    let reordered = scored(path, "reordered.grounded", "Ada", 0, &[zoe, ada, ada]);
    let unchanged = PairedRow::from_eval_reports(
        "T",
        "Eval",
        base.as_bytes(),
        reordered.to_string().as_bytes(),
    )
    .unwrap();
    assert_eq!((unchanged.gained, unchanged.lost), (0, 0));
    let duplicate_key = base.replacen(
        "\"gold_keys\":{",
        &format!("\"gold_keys\":{{\"0\":\"{}\",", "0".repeat(64)),
        1,
    );
    let err = PairedRow::from_eval_reports("T", "Eval", duplicate_key.as_bytes(), base.as_bytes())
        .unwrap_err();
    assert!(err.contains("duplicate fingerprint"), "{err}");
    let measured = live.to_string();
    let timed = PairedRow::from_eval_reports("T", "Eval", measured.as_bytes(), measured.as_bytes())
        .unwrap();
    assert!(
        timed.baseline.median_ms.is_some() && timed.baseline.median_ms == timed.candidate.median_ms
    );
    for (candidate, hint) in [
        (format!("{new}\n{new}"), "duplicate"),
        (String::new(), "missing baseline"),
        (live.to_string(), "unknown candidate"),
        (
            json!({"gold_count":0,"predicted_count":0,"correct":0,"errors":0,"matches":[]})
                .to_string(),
            "regenerate",
        ),
    ] {
        let err = PairedRow::from_eval_reports("T", "Eval", base.as_bytes(), candidate.as_bytes())
            .unwrap_err();
        assert!(err.contains(hint), "{err}");
    }
    for field in ["/pairing/source_sha256", "/pairing/gold_keys/0"] {
        let mut changed = new.clone();
        *changed.pointer_mut(field).unwrap() = json!("0".repeat(64));
        let err = PairedRow::from_eval_reports(
            "T",
            "Eval",
            base.as_bytes(),
            changed.to_string().as_bytes(),
        )
        .unwrap_err();
        assert!(err.contains("source or gold reference differs"), "{err}");
    }
}
