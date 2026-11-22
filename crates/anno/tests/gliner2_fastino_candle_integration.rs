//! Phase 4 integration tests against `fastino/gliner2-multi-v1` (the
//! PyTorch repo, not the supported GLiNER2 ONNX export).
//!
//! All tests are `#[ignore]`-gated because they require the model
//! cached locally (~280M base, plus tokenizer/config). Run with:
//!
//! ```bash
//! hf download fastino/gliner2-multi-v1 --revision ce747d79a8e362d3dee0b0b26d1201f7f1a8615a
//! cargo test -p anno --features gliner2-fastino-candle-cuda \
//!     --test gliner2_fastino_candle_integration -- --ignored --nocapture
//! ```

#![cfg(feature = "gliner2-fastino-candle")]

use anno::backends::gliner2_fastino::{GLiNER2Fastino, SUPPORTED_GLINER2_FASTINO_MODEL};
use anno::backends::gliner2_fastino_candle::GLiNER2FastinoCandle;
use anno::backends::inference::ZeroShotNER;

#[test]
#[ignore]
fn from_pretrained_smoke() {
    // Phase 4 M4 smoke test. Download fastino/gliner2-multi-v1 from HF
    // Hub, instantiate the engine, verify the encoder loaded successfully.
    // No inference — that's M5+M6's parity test.
    let model = GLiNER2FastinoCandle::from_pretrained("fastino/gliner2-multi-v1")
        .expect("load fastino/gliner2-multi-v1");
    eprintln!("loaded: {model:?}");
    assert_eq!(
        model.active_adapter(),
        None,
        "no adapter active on fresh load"
    );
}

#[test]
#[ignore]
fn extraction_matches_python_spans_and_scores() {
    // Recorded with gliner2 2.2.0, CPU float32, at this checkpoint.
    let revision = "ce747d79a8e362d3dee0b0b26d1201f7f1a8615a";
    let snapshot = hf_hub::Cache::from_env()
        .model("fastino/gliner2-multi-v1".into())
        .pointer_path(revision);
    let check = |model: &dyn ZeroShotNER| {
        let mut entities = model
            .extract_with_types(
                "Élodie met Ada Lovelace in Zürich.",
                &["person", "organization", "location"],
                0.5,
            )
            .unwrap();
        entities.sort_by_key(|e| e.start());
        let expected = [
            ("Élodie", "PER", 0, 6, 0.9998294),
            ("Ada Lovelace", "PER", 11, 23, 0.9997054),
            ("Zürich", "LOC", 27, 33, 0.9995928),
        ];
        assert_eq!(entities.len(), expected.len(), "{entities:?}");
        for (entity, (text, label, start, end, probability)) in entities.iter().zip(expected) {
            assert_eq!(
                (
                    entity.text.as_str(),
                    entity.entity_type.as_label(),
                    entity.start(),
                    entity.end()
                ),
                (text, label, start, end)
            );
            assert!(
                (entity.confidence.value() - probability).abs() < 5e-3,
                "{entity:?}"
            );
        }
    };
    check(&GLiNER2FastinoCandle::from_local(&snapshot).expect("load pinned Candle checkpoint"));
    check(&GLiNER2Fastino::from_pretrained(SUPPORTED_GLINER2_FASTINO_MODEL).expect("load ONNX"));
}

#[test]
#[ignore]
fn parity_onnx_candle_long_text_many_types() {
    // Validation: stress the pipeline with longer text + more entity
    // types than the basic parity test. If parity holds here, the
    // port works on realistic inputs.
    let text = "On January 15, 2026, Apple CEO Tim Cook met Google CEO Sundar Pichai in \
                Seattle. The deal, worth EUR 3.2 billion, closes on March 15, 2026. \
                Contact press@apple.com or call +1-555-867-5309. The European Central \
                Bank, headquartered in Frankfurt, raised interest rates by 25 basis \
                points. ECB President Christine Lagarde announced the decision at a \
                press conference. Germany's Bundeskanzler Olaf Scholz called the \
                decision necessary for stability. Meanwhile, Federal Reserve Chair \
                Jerome Powell signaled the Fed would hold rates steady at its next \
                meeting in Washington, DC on February 1, 2026.";
    let types = [
        "person",
        "organization",
        "location",
        "date",
        "money",
        "email",
        "phone",
        "title",
        "currency",
        "percentage",
    ];

    let onnx_result = GLiNER2Fastino::from_pretrained(SUPPORTED_GLINER2_FASTINO_MODEL)
        .expect("load ONNX")
        .extract_with_types(text, &types, 0.5)
        .expect("ONNX extract long");
    let candle_result = GLiNER2FastinoCandle::from_pretrained("fastino/gliner2-multi-v1")
        .expect("load Candle")
        .extract_with_types(text, &types, 0.5)
        .expect("Candle extract long");

    eprintln!(
        "long-text onnx ({}) candle ({})",
        onnx_result.len(),
        candle_result.len()
    );

    let mut onnx_sorted = onnx_result.clone();
    let mut candle_sorted = candle_result.clone();
    onnx_sorted.sort_by_key(|e| (e.start(), e.end(), e.text.clone()));
    candle_sorted.sort_by_key(|e| (e.start(), e.end(), e.text.clone()));

    assert_eq!(
        onnx_sorted.len(),
        candle_sorted.len(),
        "long-text entity count mismatch: onnx={}, candle={}",
        onnx_sorted.len(),
        candle_sorted.len()
    );

    let mut max_diff: f64 = 0.0;
    for (o, c) in onnx_sorted.iter().zip(candle_sorted.iter()) {
        assert_eq!(
            o.start(),
            c.start(),
            "start mismatch on long text: {o:?} vs {c:?}"
        );
        assert_eq!(
            o.end(),
            c.end(),
            "end mismatch on long text: {o:?} vs {c:?}"
        );
        assert_eq!(o.text, c.text, "text mismatch on long text: {o:?} vs {c:?}");
        let diff = (o.confidence.value() - c.confidence.value()).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    eprintln!("long-text ONNX↔Candle max_abs_diff: {max_diff}");
    assert!(
        max_diff < 5e-3,
        "long-text parity broken: max_abs_diff = {max_diff}"
    );
}

#[test]
#[ignore]
fn parity_onnx_candle_classify() {
    // Validation: the Candle classify path (encoder → schema_gather →
    // count_pred → classifier → softmax) must match the ONNX classify
    // path. Without this test, half of the Candle backend (the classify
    // half) is unverified.
    let text = "I absolutely loved every minute of the show — wonderful experience!";
    let labels = ["positive", "negative", "neutral"];

    let onnx_result = GLiNER2Fastino::from_pretrained(SUPPORTED_GLINER2_FASTINO_MODEL)
        .expect("load ONNX")
        .classify(text, &labels, 0.5)
        .expect("ONNX classify");
    let candle_result = GLiNER2FastinoCandle::from_pretrained("fastino/gliner2-multi-v1")
        .expect("load Candle")
        .classify(text, &labels, 0.5)
        .expect("Candle classify");

    eprintln!("onnx classify: {:?}", onnx_result);
    eprintln!("candle classify: {:?}", candle_result);

    // Top-ranked label must match.
    assert_eq!(
        onnx_result[0].0, candle_result[0].0,
        "top label diverged: onnx={}, candle={}",
        onnx_result[0].0, candle_result[0].0
    );

    // Per-label probabilities within tolerance.
    let onnx_map: std::collections::HashMap<String, f32> = onnx_result.into_iter().collect();
    let candle_map: std::collections::HashMap<String, f32> = candle_result.into_iter().collect();
    let mut max_diff: f64 = 0.0;
    for (label, p_onnx) in &onnx_map {
        let p_candle = candle_map.get(label).copied().unwrap_or(0.0);
        let diff = (*p_onnx - p_candle).abs() as f64;
        if diff > max_diff {
            max_diff = diff;
        }
    }
    eprintln!("classify ONNX↔Candle max_abs_diff: {max_diff}");
    assert!(
        max_diff < 5e-3,
        "classify parity broken: max_abs_diff = {max_diff}"
    );
}
