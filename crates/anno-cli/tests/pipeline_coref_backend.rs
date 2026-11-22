use anno::GroundedDocument;
use serde_json::Value;
use std::process::Command;

#[test]
fn pipeline_reports_the_successful_offline_resolver_even_for_zero_tracks() {
    let cache = tempfile::tempdir().unwrap();
    let run = |text: &str, coref: bool| {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_anno"));
        cmd.args([
            "pipeline",
            "--model",
            "heuristic",
            "--format",
            "json",
            "--quiet",
        ])
        .env_remove("HOME")
        .env("XDG_CACHE_HOME", cache.path())
        .env("HF_HOME", cache.path())
        .env("FCOREF_MODEL_PATH", cache.path())
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
        let mut docs: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
        docs.remove(0)
    };

    let without = run("Alice saw Alice.", false);
    assert!(without.get("coref_backend").is_none());
    let old: GroundedDocument = serde_json::from_value(without).unwrap();
    assert!(serde_json::to_value(old)
        .unwrap()
        .get("coref_backend")
        .is_none());

    for (text, has_tracks) in [("We met Alice and Alice.", true), ("quiet", false)] {
        let doc = run(text, true);
        assert_eq!(doc["coref_backend"], "mention-ranking");
        assert_eq!(!doc["tracks"].as_object().unwrap().is_empty(), has_tracks);
        let round_trip: GroundedDocument = serde_json::from_value(doc).unwrap();
        assert_eq!(
            serde_json::to_value(round_trip).unwrap()["coref_backend"],
            "mention-ranking"
        );
    }
}
