use serde_json::{json, Value};
use std::process::{Command, Output};

const SOURCE: &str = "📬 x@y.com now";
const GOLD: &str = "x@y.com:EMAIL:2:9";

fn cli(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_anno"))
        .args(args)
        .output()
        .unwrap()
}

fn json_output(args: &[&str]) -> Value {
    let output = cli(args);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn extract_once_then_rescore_the_same_unicode_source() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("predictions.json");
    let filename = file.to_str().unwrap();
    let mut saved = json_output(&[
        "extract", "--model", "pattern", "--text", SOURCE, "--format", "grounded",
    ]);
    std::fs::write(&file, saved.to_string()).unwrap();

    for metric in ["entity", "char-masking"] {
        let mut args = vec![
            "eval", "--text", SOURCE, "--gold", GOLD, "--json", "--metric", metric,
        ];
        let mut live = args.clone();
        live.extend(["--model", "pattern"]);
        let live = json_output(&live);
        args.extend(["--predictions", filename]);
        let replay = json_output(&args);
        for key in ["f1", "precision", "recall", "matches", "char_masking"] {
            assert_eq!(replay[key], live[key], "{key}");
        }
        assert_eq!(replay["f1"], 1.0);
        assert!(replay["model"].is_null());
        assert!(replay["elapsed_ms"].is_null());
        assert_eq!(replay["predictions"]["document_id"], "extract");
        assert_eq!(
            replay["predictions"]["source_sha256"],
            "154527a734e13c9a3e2a5735553f36eebd46c6457513317e17bae2aa5ca5434e"
        );
    }

    // A saved producer may assign nonsequential IDs.
    saved["signals"][0]["id"] = json!(43);
    std::fs::write(&file, saved.to_string()).unwrap();
    let changed_gold = json_output(&[
        "eval",
        "--text",
        SOURCE,
        "--gold",
        "x@y.com:ORG:2:9",
        "--json",
        "--predictions",
        filename,
    ]);
    assert_eq!(changed_gold["f1"], 0.0);
    assert_eq!(changed_gold["matches"][0]["pred_id"], 43);

    let rejection = |text: &str, expected: &str| {
        let run = cli(&[
            "eval",
            "--text",
            text,
            "--gold",
            GOLD,
            "--json",
            "--predictions",
            filename,
        ]);
        assert!(!run.status.success());
        assert!(run.stdout.is_empty());
        assert!(
            String::from_utf8_lossy(&run.stderr).contains(expected),
            "{}",
            String::from_utf8_lossy(&run.stderr)
        );
    };
    // Same mention and offsets, different surrounding source.
    rejection("📬 x@y.com NEW", "source text differs");
    saved["signals"][0]["surface"] = json!("z@y.com");
    std::fs::write(&file, saved.to_string()).unwrap();
    rejection(SOURCE, "signal locations or surfaces");
    saved["signals"] = json!([]);
    std::fs::write(&file, saved.to_string()).unwrap();
    let no_predictions = json_output(&[
        "eval",
        "--text",
        SOURCE,
        "--gold",
        GOLD,
        "--json",
        "--predictions",
        filename,
    ]);
    assert_eq!(no_predictions["predicted_count"], 0);
    assert_eq!(no_predictions["f1"], 0.0);
    let both_sources = cli(&[
        "eval",
        "--text",
        SOURCE,
        "--gold",
        GOLD,
        "--model",
        "pattern",
        "--predictions",
        filename,
    ]);
    assert!(!both_sources.status.success());
    assert!(String::from_utf8_lossy(&both_sources.stderr).contains("--model"));
    std::fs::write(&file, "{}").unwrap();
    rejection(SOURCE, "parse GroundedDocument JSON");
}
