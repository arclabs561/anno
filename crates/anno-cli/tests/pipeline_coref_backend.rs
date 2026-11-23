use anno::GroundedDocument;
use serde_json::Value;
use std::path::Path;
use std::process::Command;

fn pipeline(text: &str, coref: bool, model_dir: &Path) -> Value {
    let cache = tempfile::tempdir().unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_anno"));
    cmd.args([
        "pipeline",
        "--model",
        "heuristic",
        "--format",
        "json",
        "--quiet",
    ])
    .env("HOME", cache.path())
    .env("XDG_CACHE_HOME", cache.path())
    .env("HF_HOME", cache.path())
    .env("FCOREF_MODEL_PATH", model_dir)
    .env("ANNO_NO_DOWNLOADS", "1");
    if coref {
        cmd.arg("--coref");
    }
    let output = cmd.arg(text).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let [doc]: [Value; 1] = serde_json::from_slice(&output.stdout).unwrap();
    doc
}

#[test]
fn pipeline_reports_the_successful_offline_resolver_even_for_zero_tracks() {
    let missing_model = tempfile::tempdir().unwrap();
    let run = |text, coref| pipeline(text, coref, missing_model.path());
    let without = run("Alice saw Alice.", false);
    assert!(without.get("coref_backend").is_none());
    assert!(without.get("coref_run").is_none());
    let old: GroundedDocument = serde_json::from_value(without).unwrap();
    assert!(serde_json::to_value(old)
        .unwrap()
        .get("coref_backend")
        .is_none());

    for (text, has_tracks) in [("We met Alice and Alice.", true), ("quiet", false)] {
        let doc = run(text, true);
        assert_eq!(doc["coref_backend"], "mention-ranking");
        assert_eq!(!doc["tracks"].as_object().unwrap().is_empty(), has_tracks);
        let report = &doc["coref_run"];
        assert_eq!(report["status"], "completed");
        assert_eq!(report["backend"], "mention-ranking");
        assert_eq!(
            report["tracks_added"],
            doc["tracks"].as_object().unwrap().len()
        );
        #[cfg(feature = "onnx")]
        assert_eq!(
            report["prior_failures"].as_array().unwrap().last().unwrap(),
            &serde_json::json!({"stage": "load", "backend": "fcoref"})
        );
        let mut inconsistent = doc.clone();
        inconsistent["coref_backend"] = "other-backend".into();
        assert!(serde_json::from_value::<GroundedDocument>(inconsistent).is_err());
        let round_trip: GroundedDocument = serde_json::from_value(doc).unwrap();
        assert_eq!(
            serde_json::to_value(round_trip).unwrap()["coref_backend"],
            "mention-ranking"
        );
    }
}

#[test]
fn failed_retry_preserves_last_success_and_replaces_attempt_evidence() {
    use anno::{CorefAttemptFailure, CorefRunReport};
    let mut doc = GroundedDocument::new("doc", "");
    doc.set_coref_backend("earlier");
    let legacy: GroundedDocument =
        serde_json::from_value(serde_json::to_value(doc).unwrap()).unwrap();
    assert_eq!(legacy.coref_backend(), Some("earlier"));
    assert!(legacy.coref_run().is_none());

    let mut doc = legacy;
    let report = CorefRunReport::Failed {
        last_failure: CorefAttemptFailure::Inference {
            backend: "attempted".into(),
        },
        prior_failures: Vec::new(),
    };
    doc.record_coref_run(report.clone());
    let read: GroundedDocument =
        serde_json::from_value(serde_json::to_value(doc).unwrap()).unwrap();
    assert_eq!(read.coref_run(), Some(&report));
    assert_eq!(read.coref_backend(), Some("earlier"));
}

#[test]
#[ignore = "requires a real FCoref export in FCOREF_MODEL_PATH"]
fn neural_pipeline_links_exact_source_mentions_and_records_fcoref() {
    let model =
        std::env::var_os("FCOREF_MODEL_PATH").expect("set FCOREF_MODEL_PATH to a model export");
    let text = "À Paris, Marie Curie studied physics. She later won the Nobel Prize.";
    let doc: GroundedDocument =
        serde_json::from_value(pipeline(text, true, Path::new(&model))).unwrap();
    assert_eq!(doc.coref_backend(), Some("fcoref"));
    assert_eq!(doc.text(), text);
    let linked = doc.tracks().any(|track| {
        let has = |surface, offsets| {
            track.signals().iter().any(|member| {
                let signal = doc.get_signal(member.signal_id).unwrap();
                signal.surface() == surface && signal.text_offsets() == Some(offsets)
            })
        };
        has("Marie Curie", (9, 20)) && has("She", (38, 41))
    });
    assert!(linked, "expected the name and pronoun in the same track");
}
