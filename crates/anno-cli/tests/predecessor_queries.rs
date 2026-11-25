// SPDX-License-Identifier: MIT OR Apache-2.0
use anno::core::grounded::{GroundedDocument, Track};
use serde_json::{json, Value};
use std::{
    fs,
    path::Path,
    process::{Command, Output},
};
use tempfile::tempdir;

fn document(id: &str, text: &str, surface: &str, revision: Option<&str>) -> Value {
    let mut doc = serde_json::to_value(GroundedDocument::new(id, text)).unwrap();
    // Keep the submitted handle distinguishable from a freshly allocated zero-based ID.
    doc["tracks"]["7"] = json!(Track::new(7, surface));
    json!({"source_revision":revision,"document":doc})
}

fn edition(sequence: u64, documents: Vec<Value>) -> Value {
    json!({"stream":"registry","sequence":sequence,"threshold":0.7,
        "require_type_match":true,"documents":documents})
}

fn save(dir: &Path, name: &str, value: &Value) {
    fs::write(dir.join(name), serde_json::to_vec(value).unwrap()).unwrap();
}

fn invoke(dir: &Path, doc_id: &str, track_id: &str, options: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_anno"))
        .current_dir(dir)
        .args([
            "query",
            "predecessors",
            "current.json",
            "--prior",
            "prior.json",
            "--doc-id",
            doc_id,
            "--track-id",
            track_id,
        ])
        .args(options)
        .output()
        .unwrap()
}

fn lookup(dir: &Path, options: &[&str]) -> Value {
    let out = invoke(dir, "query", "7", options);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}

fn assert_error(out: Output) {
    assert!(!out.status.success());
    assert!(
        out.stdout.is_empty(),
        "invalid input published partial results"
    );
    assert!(!out.stderr.is_empty());
}

#[test]
fn strict_default_includes_exact_threshold_and_expansion_keeps_only_positive_scores() {
    let temp = tempdir().unwrap();
    let dir = temp.path();
    let current = edition(
        8,
        vec![document("query", "Alex Reed", "Alex Reed", Some("vol-8"))],
    );
    let prior = edition(
        2,
        vec![
            document("alone", "Reed", "Reed", None),
            document("overlap", "Alex Noor", "Alex Noor", Some("vol-2")),
            document("blank", "", " ", None),
            document("different", "Vale", "Vale", None),
        ],
    );
    save(dir, "current.json", &current);
    save(dir, "prior.json", &prior);
    let strict = lookup(dir, &[]);
    assert_eq!(strict["stream"], "registry");
    assert_eq!(strict["prior_sequence"], 2);
    assert_eq!(strict["current_sequence"], 8);
    assert_eq!(strict["scorer"], "word_jaccard_v1");
    assert_eq!(strict["limit"], 1);
    assert_eq!(strict["minimum_score"], 0.5);
    assert_eq!(strict["candidates"].as_array().unwrap().len(), 1);
    let best = &strict["candidates"][0];
    assert_eq!(best["score"], 0.5);
    assert_eq!(best["track"]["doc_id"], "alone");
    assert_eq!(best["track"]["track_id"], 7);
    assert_eq!(best["track"].get("source_revision"), Some(&Value::Null));
    assert_eq!(strict["truncated"], false);
    assert_eq!(strict.get("withheld_tie"), Some(&Value::Null));

    let wider = lookup(dir, &["--minimum-score", "0", "--limit", "5"]);
    let rows = wider["candidates"].as_array().unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(&rows[0], best);
    assert_eq!(rows[1]["track"]["doc_id"], "overlap");
    let close = rows[1]["score"].as_f64().unwrap();
    assert!((close - 1.0 / 3.0).abs() < 1e-7);

    let blank = edition(9, vec![document("query", "", " ", None)]);
    save(dir, "current.json", &blank);
    assert_eq!(
        lookup(dir, &["--minimum-score", "0"])["candidates"],
        json!([])
    );
}

#[test]
fn score_groups_are_order_independent_and_a_tie_never_fills_only_one_slot() {
    let temp = tempdir().unwrap();
    let dir = temp.path();
    let x = document("x", "Reed", "Reed", None);
    let y = document("y", "Alex", "Alex", None);
    let z = document("z", "Alex Reed", "Alex Reed", None);
    save(
        dir,
        "current.json",
        &edition(2, vec![document("query", "Alex Reed", "Alex Reed", None)]),
    );
    save(dir, "prior.json", &edition(1, vec![y.clone(), x.clone()]));
    let tied = lookup(dir, &[]);
    assert_eq!(tied["candidates"], json!([]));
    assert_eq!(tied["withheld_tie"], json!({"score":0.5,"count":2}));
    assert_eq!(tied["truncated"], true);

    save(
        dir,
        "prior.json",
        &edition(1, vec![z.clone(), y.clone(), x.clone()]),
    );
    let exact_cap = lookup(dir, &[]);
    assert_eq!(exact_cap["candidates"][0]["track"]["doc_id"], "z");
    assert_eq!(exact_cap["truncated"], true);
    assert_eq!(exact_cap["withheld_tie"], Value::Null);

    let partial = lookup(dir, &["--limit", "2"]);
    assert_eq!(partial["candidates"], exact_cap["candidates"]);
    assert_eq!(partial["withheld_tie"], tied["withheld_tie"]);
    let full = lookup(dir, &["--limit", "3"]);
    let ids: Vec<_> = full["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["track"]["doc_id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["z", "x", "y"]);
    assert_eq!(full["truncated"], false);
    save(dir, "prior.json", &edition(1, vec![x, z, y]));
    assert_eq!(lookup(dir, &["--limit", "3"]), full);
    assert_eq!(lookup(dir, &["--limit", "2"]), partial);
}

#[test]
fn retained_evidence_explains_the_old_winner_after_a_canonical_name_correction() {
    let temp = tempdir().unwrap();
    let dir = temp.path();
    let unchanged_text = "Rae Vale Chan and Rae Fenn Chan";
    let prior = edition(
        2,
        vec![
            document("record-a", "Rae Vale", "Rae Vale", Some("v1")),
            document("record-b", "Rae Fenn", "Rae Fenn", Some("v1")),
        ],
    );
    let old = edition(
        3,
        vec![document(
            "query",
            unchanged_text,
            "Rae Vale Chan",
            Some("q1"),
        )],
    );
    save(dir, "prior.json", &prior);
    save(dir, "current.json", &old);
    let old_bytes = fs::read(dir.join("current.json")).unwrap();
    let prior_bytes = fs::read(dir.join("prior.json")).unwrap();
    let captured = lookup(dir, &[]);
    assert_eq!(fs::read(dir.join("current.json")).unwrap(), old_bytes);
    let hash = "a6c1b7fd5b357e8b268b7f338dd326a0373c28d77ae18303b63440d74efaa3e6";
    assert_eq!(captured["query"]["text_sha256"], hash);
    assert_eq!(captured["query"]["source_revision"], "q1");
    assert_eq!(captured["query"]["surface"], "Rae Vale Chan");
    assert_eq!(captured["candidates"][0]["track"]["doc_id"], "record-a");
    assert_eq!(captured["candidates"][0]["track"]["surface"], "Rae Vale");
    assert_eq!(captured["candidates"][0]["track"]["source_revision"], "v1");

    let corrected = edition(
        4,
        vec![document(
            "query",
            unchanged_text,
            "Rae Fenn Chan",
            Some("q2"),
        )],
    );
    save(dir, "current.json", &corrected);
    let corrected_bytes = fs::read(dir.join("current.json")).unwrap();
    let fresh = lookup(dir, &[]);
    assert_eq!(
        fresh["query"]["text_sha256"],
        captured["query"]["text_sha256"]
    );
    assert_eq!(fresh["query"]["surface"], "Rae Fenn Chan");
    assert_eq!(fresh["query"]["source_revision"], "q2");
    assert_eq!(fresh["current_sequence"], 4);
    assert_eq!(fresh["candidates"][0]["track"]["doc_id"], "record-b");
    assert_eq!(fs::read(dir.join("current.json")).unwrap(), corrected_bytes);
    assert_eq!(fs::read(dir.join("prior.json")).unwrap(), prior_bytes);

    fs::write(dir.join("current.json"), old_bytes).unwrap();
    assert_eq!(lookup(dir, &[]), captured);
    assert_eq!(fs::read_dir(dir).unwrap().count(), 2);
}

#[test]
fn rejects_mixed_streams_bad_chronology_and_ambiguous_or_missing_source_handles() {
    let temp = tempdir().unwrap();
    let dir = temp.path();
    let query = document("query", "Reed", "Reed", None);
    let one = document("one", "Reed", "Reed", None);
    let valid = edition(1, vec![one.clone()]);
    save(dir, "current.json", &edition(3, vec![query.clone()]));
    save(dir, "prior.json", &valid);
    assert_error(invoke(dir, "query", "99", &[]));
    assert_error(invoke(dir, "absent", "7", &[]));
    for options in [
        &["--limit", "0"][..],
        &["--limit", "129"],
        &["--minimum-score", "NaN"],
    ] {
        assert_error(invoke(dir, "query", "7", options));
    }
    let mut wrong_stream = valid.clone();
    wrong_stream["stream"] = json!("another-registry");
    let mut bad_revision = valid.clone();
    bad_revision["documents"][0]["source_revision"] = json!(" ");
    let mut bad_track = valid.clone();
    bad_track["documents"][0]["document"]["tracks"]["7"]["id"] = json!(8);
    for invalid in [
        wrong_stream,
        edition(3, vec![]),
        edition(4, vec![]),
        edition(1, vec![one.clone(), one]),
        bad_revision,
        bad_track,
    ] {
        save(dir, "prior.json", &invalid);
        assert_error(invoke(dir, "query", "7", &[]));
    }
}

#[test]
fn a_reused_local_track_handle_in_an_older_edition_is_eligible_with_either_input_on_stdin() {
    let temp = tempdir().unwrap();
    let dir = temp.path();
    save(
        dir,
        "current.json",
        &edition(
            2,
            vec![document("query", "Alex Reed", "Alex Reed", Some("now"))],
        ),
    );
    save(
        dir,
        "prior.json",
        &edition(1, vec![document("query", "Reed", "Reed", Some("before"))]),
    );
    let expected = lookup(dir, &[]);
    assert_eq!(
        expected["candidates"][0]["track"]["source_revision"],
        "before"
    );
    assert_eq!(
        expected["candidates"][0]["track"]["doc_id"],
        expected["query"]["doc_id"]
    );
    assert_eq!(
        expected["candidates"][0]["track"]["track_id"],
        expected["query"]["track_id"]
    );

    let piped = |current: &str, prior: &str, source: &str| {
        Command::new(env!("CARGO_BIN_EXE_anno"))
            .current_dir(dir)
            .args([
                "query",
                "predecessors",
                current,
                "--prior",
                prior,
                "--doc-id",
                "query",
                "--track-id",
                "7",
            ])
            .stdin(fs::File::open(dir.join(source)).unwrap())
            .output()
            .unwrap()
    };
    for output in [
        piped("-", "prior.json", "current.json"),
        piped("current.json", "-", "prior.json"),
    ] {
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            serde_json::from_slice::<Value>(&output.stdout).unwrap(),
            expected
        );
    }
    assert_error(piped("-", "-", "prior.json"));
}
