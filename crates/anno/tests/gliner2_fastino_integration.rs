//! Tier-2 integration tests for `gliner2_fastino`. `#[ignore]`-gated since
//! they download the SemplificaAI/gliner2-multi-v1-onnx model (~6 GB) on
//! first run and require a working multi-session pipeline (Phase 3).
//!
//! Run locally with:
//!
//!     cargo test -p anno --features gliner2-fastino \
//!         --test gliner2_fastino_integration -- --ignored

#![cfg(feature = "gliner2-fastino")]

use anno::backends::gliner2_fastino::GLiNER2Fastino;
use anno::backends::inference::ZeroShotNER;

const FIXTURE: &str = "Acme Corp signed a deal with Globex in Paris on January 5th.";

#[test]
#[ignore]
fn fastino_multi_v1_extracts_org_and_loc() {
    let model = GLiNER2Fastino::from_pretrained("SemplificaAI/gliner2-multi-v1-onnx")
        .expect("load gliner2-multi-v1");
    let ents = model
        .extract_with_types(FIXTURE, &["organization", "location"], 0.5)
        .expect("extract");

    eprintln!("entities: {ents:#?}");

    // Loose assertions — the model's exact tokenization-driven output
    // varies, but Acme Corp + Paris are clearly correct labels.
    let acme = ents.iter().find(|e| e.text.contains("Acme"));
    let paris = ents
        .iter()
        .find(|e| e.text == "Paris" || e.text.contains("Paris"));
    assert!(acme.is_some(), "expected an Acme org entity, got {ents:#?}");
    assert!(paris.is_some(), "expected a Paris entity, got {ents:#?}");
}

#[test]
#[ignore]
fn fastino_extract_with_label_descriptions() {
    // Phase 1.5 M1.3: verify the [DESCRIPTION]-emitting prompt path runs
    // end-to-end against the real model and returns expected entities.
    // The actual accuracy boost vs labels-only isn't measured here —
    // that's a benchmark concern. This just exercises the pipeline.
    let model = GLiNER2Fastino::from_pretrained("SemplificaAI/gliner2-multi-v1-onnx")
        .expect("load gliner2-multi-v1");
    let labeled: Vec<(&str, &str)> = vec![
        ("organization", "a company, corporation, or institution"),
        ("location", "a geographic place, city, country, or region"),
    ];
    let ents = model
        .extract_with_label_descriptions(FIXTURE, &labeled, 0.5)
        .expect("extract_with_label_descriptions");

    eprintln!("entities (with descriptions): {ents:#?}");
    assert!(
        ents.iter().any(|e| e.text.contains("Acme")),
        "expected an Acme org entity, got {ents:#?}",
    );
    assert!(
        ents.iter()
            .any(|e| e.text == "Paris" || e.text.contains("Paris")),
        "expected a Paris location entity, got {ents:#?}",
    );
}

#[test]
#[ignore]
fn fastino_batch_per_sample_labels() {
    // Phase 1.5 M4.2: each text in the batch carries its own label set.
    // Text 0 only looks for orgs; text 1 only looks for people + places.
    use anno::backends::gliner2_fastino::BatchSchemaMode;

    let model = GLiNER2Fastino::from_pretrained("SemplificaAI/gliner2-multi-v1-onnx")
        .expect("load gliner2-multi-v1");
    let texts: Vec<&str> = vec!["Acme Corp signed a deal.", "Marie Curie worked in France."];
    let labels_per_text: Vec<Vec<&str>> = vec![vec!["organization"], vec!["person", "location"]];

    let results = model
        .batch_extract_with_schema_mode(&texts, BatchSchemaMode::PerSample(&labels_per_text), 0.5)
        .expect("batch extract");

    assert_eq!(results.len(), 2);
    // Text 0 should detect Acme as organization (or at least produce non-empty).
    assert!(
        results[0]
            .iter()
            .any(|e| matches!(e.entity_type, anno::EntityType::Organization)),
        "expected an Organization in results[0], got {:#?}",
        results[0],
    );
    // Text 1 should detect Marie as person OR France as location.
    assert!(
        results[1].iter().any(|e| matches!(
            e.entity_type,
            anno::EntityType::Person | anno::EntityType::Location
        )),
        "expected a Person or Location in results[1], got {:#?}",
        results[1],
    );
}

#[test]
#[ignore]
fn fastino_batch_per_sample_length_mismatch_errors() {
    // Defensive: PerSample with mismatched outer-slice length returns
    // a typed Backend error, not a panic.
    use anno::backends::gliner2_fastino::BatchSchemaMode;

    let model =
        GLiNER2Fastino::from_pretrained("SemplificaAI/gliner2-multi-v1-onnx").expect("load");
    let texts: Vec<&str> = vec!["one", "two"];
    let labels_per_text: Vec<Vec<&str>> = vec![vec!["organization"]]; // 1 entry, not 2.

    let err = model
        .batch_extract_with_schema_mode(&texts, BatchSchemaMode::PerSample(&labels_per_text), 0.5)
        .expect_err("expected length-mismatch error");
    assert!(
        err.to_string().contains("PerSample label count")
            && err.to_string().contains("!= texts count"),
        "got: {err}",
    );
}

#[test]
#[ignore]
fn fastino_batch_extract_streaming_fires_callbacks_in_order() {
    // Phase 1.5 M3.2: drives 5 short texts through batch_extract_streaming
    // with batch_size = 2, so we get chunks (0..2), (2..4), (4..5). Each
    // text's callback fires sequentially with the right index. Asserts:
    //   1. All five indices are seen, in order.
    //   2. Total entities >= number of texts (each one has at least one).
    let model = GLiNER2Fastino::from_pretrained("SemplificaAI/gliner2-multi-v1-onnx")
        .expect("load gliner2-multi-v1");
    let texts: Vec<&str> = vec![
        "Acme Corp signed a deal in Paris.",
        "Globex acquired Hooli.",
        "Marie Curie worked in France.",
        "Apple is based in California.",
        "Tokyo is the capital of Japan.",
    ];

    let mut indices_seen: Vec<usize> = Vec::new();
    let mut total_entities = 0usize;

    model
        .batch_extract_streaming(
            &texts,
            &["organization", "location", "person"],
            0.5,
            2, // batch_size
            |idx, ents| {
                indices_seen.push(idx);
                total_entities += ents.len();
            },
        )
        .expect("batch_extract_streaming");

    assert_eq!(indices_seen, vec![0, 1, 2, 3, 4]);
    assert!(
        total_entities >= 5,
        "expected at least 5 total entities across 5 texts, got {total_entities}",
    );
}

#[test]
#[ignore]
fn fastino_batch_extract_streaming_rejects_zero_batch_size() {
    // Defensive: typed Backend error rather than silent loop / divide-by-zero.
    let model =
        GLiNER2Fastino::from_pretrained("SemplificaAI/gliner2-multi-v1-onnx").expect("load");
    let texts: &[&str] = &["irrelevant"];
    let result = model.batch_extract_streaming(texts, &["organization"], 0.5, 0, |_, _| {
        panic!("callback should not fire when batch_size = 0")
    });
    let err = result.expect_err("expected an error for batch_size = 0");
    assert!(
        err.to_string().contains("batch_size must be > 0"),
        "got: {err}",
    );
}

#[test]
#[ignore]
fn fastino_classify_smoke() {
    let model =
        GLiNER2Fastino::from_pretrained("SemplificaAI/gliner2-multi-v1-onnx").expect("load");
    let scores = model
        .classify(
            "This product is wonderful, I love it.",
            &["positive", "negative", "neutral"],
            0.0,
        )
        .expect("classify");
    assert_eq!(scores.len(), 3);
    eprintln!("classify scores: {scores:?}");
    // Top-ranked should be 'positive' for this clearly-positive text.
    assert_eq!(
        scores[0].0, "positive",
        "expected 'positive' top-ranked, got {scores:?}"
    );
}

#[test]
#[ignore]
fn fastino_extract_structure_invoice_single_instance() {
    // Phase 2 M5: single-instance structure extraction.
    use anno::backends::gliner2_fastino::schema::{
        FieldType, StructureTask, StructureValue, TaskSchema,
    };

    let model = GLiNER2Fastino::from_pretrained("SemplificaAI/gliner2-multi-v1-onnx")
        .expect("load gliner2-multi-v1");
    let schema = TaskSchema::new().with_structure(
        StructureTask::new("invoice")
            .with_field("vendor", FieldType::String)
            .with_field("amount", FieldType::String),
    );

    let text = "Invoice from Acme Corp for $4,250.00 dated January 15, 2026.";
    let result = model
        .extract_structure(text, &schema, 0.5)
        .expect("extract");

    eprintln!("invoice extraction: {result:#?}");

    // Loose assertions: at least one instance returned, and "Acme" surfaces
    // as the vendor (or somewhere in the result). We don't pin to exact
    // count because the model's count predictor can return 1 or more
    // depending on tokenization.
    assert!(
        !result.is_empty(),
        "expected at least 1 instance, got {result:#?}"
    );
    let serialized = serde_json::to_string(&result).expect("serialize");
    assert!(
        serialized.contains("Acme"),
        "expected 'Acme' somewhere in result, got {serialized}",
    );

    // structure_type is the task name.
    assert_eq!(result[0].structure_type, "invoice");

    // If vendor is present, it should be a Single value.
    if let Some(v) = result[0].fields.get("vendor") {
        assert!(
            matches!(v, StructureValue::Single(_)),
            "vendor should be Single, got {v:?}",
        );
    }
}

#[test]
#[ignore]
fn fastino_extract_structure_multi_instance_people() {
    // Phase 2 M5: multi-instance structure extraction. Two clear people in
    // the text → expect at least 2 instances.
    use anno::backends::gliner2_fastino::schema::{FieldType, StructureTask, TaskSchema};

    let model =
        GLiNER2Fastino::from_pretrained("SemplificaAI/gliner2-multi-v1-onnx").expect("load");
    let schema = TaskSchema::new().with_structure(
        StructureTask::new("person_record")
            .with_field("name", FieldType::String)
            .with_field("role", FieldType::String),
    );

    let text = "Marie Curie was a physicist. Albert Einstein was also a physicist.";
    let result = model
        .extract_structure(text, &schema, 0.3)
        .expect("extract");

    eprintln!("multi-instance: {result:#?}");
    assert!(
        result.len() >= 2,
        "expected at least 2 person_record instances, got {result:#?}",
    );

    // Every result has structure_type == "person_record".
    for r in &result {
        assert_eq!(r.structure_type, "person_record");
    }
}

#[test]
#[ignore]
fn fastino_extract_structure_empty_schema_returns_empty() {
    // Phase 2 M5: defensive — empty schema → empty vec, no inference passes.
    use anno::backends::gliner2_fastino::schema::TaskSchema;

    let model =
        GLiNER2Fastino::from_pretrained("SemplificaAI/gliner2-multi-v1-onnx").expect("load");
    let schema = TaskSchema::new(); // no structures
    let result = model
        .extract_structure("anything", &schema, 0.5)
        .expect("extract");
    assert!(result.is_empty());
}

// =============================================================================
// Phase 3.5 — Standard ≡ IoBinding parity tests.
//
// These run the same input through both ExecutionMode paths and verify the
// outputs match within a tolerance. Require the SemplificaAI/gliner2-multi-v1-onnx
// snapshot cached locally (~6 GB).
//
// Tolerance: max_abs_diff < 1e-4 on per-entity confidence scores. Looser than
// the spec's 1e-5 for fp32 because:
//   1. IoBinding's MemoryInfo allocator can introduce micro-differences in
//      memory layout that interact with FMA fusion.
//   2. ort's run() vs run_binding() take subtly different paths through the
//      ONNX Runtime C API.
// 1e-4 is well below any user-visible threshold (typical decoder thresholds
// are 0.5+).
// =============================================================================

#[test]
#[ignore]
fn parity_standard_iobinding_extract_with_types() {
    use anno::backends::gliner2_fastino::{ExecutionMode, GLiNER2FastinoConfig};

    let model_id = "SemplificaAI/gliner2-multi-v1-onnx";

    let standard = GLiNER2Fastino::from_pretrained_with_config(
        model_id,
        GLiNER2FastinoConfig::default().with_execution_mode(ExecutionMode::Standard),
    )
    .expect("standard load");

    let iobinding = GLiNER2Fastino::from_pretrained_with_config(
        model_id,
        GLiNER2FastinoConfig::default().with_execution_mode(ExecutionMode::IoBinding),
    )
    .expect("iobinding load");

    let text = "Marie Curie won the Nobel Prize in Physics in 1903.";
    let types = ["person", "award", "year"];

    let std_result =
        ZeroShotNER::extract_with_types(&standard, text, &types, 0.5).expect("standard extract");
    let io_result =
        ZeroShotNER::extract_with_types(&iobinding, text, &types, 0.5).expect("iobinding extract");

    eprintln!("standard ({}): {:#?}", std_result.len(), std_result);
    eprintln!("iobinding ({}): {:#?}", io_result.len(), io_result);

    assert_eq!(
        std_result.len(),
        io_result.len(),
        "entity count mismatch: standard={}, iobinding={}",
        std_result.len(),
        io_result.len()
    );

    // Sort both sides by (start, end, text) for deterministic pairwise
    // comparison; NMS isn't guaranteed to produce identical ordering
    // across paths.
    let mut std_sorted = std_result.clone();
    let mut io_sorted = io_result.clone();
    std_sorted.sort_by_key(|e| (e.start(), e.end(), e.text.clone()));
    io_sorted.sort_by_key(|e| (e.start(), e.end(), e.text.clone()));

    let mut max_diff: f64 = 0.0;
    for (s, i) in std_sorted.iter().zip(io_sorted.iter()) {
        assert_eq!(
            s.start(),
            i.start(),
            "start mismatch: std={:?}, io={:?}",
            s,
            i
        );
        assert_eq!(s.end(), i.end(), "end mismatch: std={:?}, io={:?}", s, i);
        assert_eq!(s.text, i.text, "text mismatch: std={:?}, io={:?}", s, i);
        let diff = (s.confidence.value() - i.confidence.value()).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    eprintln!("max_abs_diff on confidence: {max_diff}");
    assert!(
        max_diff < 1e-4,
        "Standard ≡ IoBinding parity broken: max_abs_diff = {max_diff} > 1e-4"
    );
}

#[test]
#[ignore]
fn parity_standard_iobinding_classify() {
    use anno::backends::gliner2_fastino::{ExecutionMode, GLiNER2FastinoConfig};

    let model_id = "SemplificaAI/gliner2-multi-v1-onnx";

    let standard = GLiNER2Fastino::from_pretrained_with_config(
        model_id,
        GLiNER2FastinoConfig::default().with_execution_mode(ExecutionMode::Standard),
    )
    .expect("standard load");

    let iobinding = GLiNER2Fastino::from_pretrained_with_config(
        model_id,
        GLiNER2FastinoConfig::default().with_execution_mode(ExecutionMode::IoBinding),
    )
    .expect("iobinding load");

    let text = "I absolutely loved every minute of the show — wonderful experience!";
    let labels = ["positive", "negative", "neutral"];

    let std_result = standard.classify(text, &labels, 0.5).expect("std classify");
    let io_result = iobinding.classify(text, &labels, 0.5).expect("io classify");

    eprintln!("standard: {:?}", std_result);
    eprintln!("iobinding: {:?}", io_result);

    assert_eq!(
        std_result[0].0, io_result[0].0,
        "top label diverged: standard={}, iobinding={}",
        std_result[0].0, io_result[0].0
    );

    let std_map: std::collections::HashMap<String, f32> = std_result.into_iter().collect();
    let io_map: std::collections::HashMap<String, f32> = io_result.into_iter().collect();
    let mut max_diff: f64 = 0.0;
    for (label, p_std) in &std_map {
        let p_io = io_map.get(label).copied().unwrap_or(0.0);
        let diff = (*p_std - p_io).abs() as f64;
        if diff > max_diff {
            max_diff = diff;
        }
    }
    eprintln!("classify max_abs_diff: {max_diff}");
    assert!(
        max_diff < 1e-4,
        "classify Standard ≡ IoBinding parity broken: max_abs_diff = {max_diff} > 1e-4"
    );
}

#[cfg(feature = "gliner2-fastino-cuda")]
#[test]
#[ignore]
fn smoke_iobinding_cuda() {
    // Phase 3.5 M13: CUDA + IoBinding smoke test. Verifies that the
    // IoBinding chain works end-to-end with prefer_cuda=true. On a
    // CPU-only host this fails at session load (no CUDA EP) — the
    // `#[ignore]` gate means it only runs opt-in.
    //
    // GPU validation (parity vs CPU IoBinding, latency benchmark) is
    // deferred — once a GPU host is available, extend this test to
    // assert max_abs_diff < 1e-3 vs CPU IoBinding (looser tolerance
    // because fused matmul kernels on CUDA differ from CPU).
    use anno::backends::gliner2_fastino::{ExecutionMode, GLiNER2FastinoConfig};
    use anno::OnnxSessionConfig;

    // OnnxSessionConfig is #[non_exhaustive] — mutate fields on a
    // default() instance rather than constructing with struct literal.
    let mut onnx = OnnxSessionConfig::default();
    onnx.prefer_cuda = true;
    let cfg = GLiNER2FastinoConfig::default()
        .with_execution_mode(ExecutionMode::IoBinding)
        .with_onnx(onnx);

    let model =
        GLiNER2Fastino::from_pretrained_with_config("SemplificaAI/gliner2-multi-v1-onnx", cfg)
            .expect("load with CUDA + IoBinding");

    let text = "Marie Curie won the Nobel Prize in Physics in 1903.";
    let result = ZeroShotNER::extract_with_types(&model, text, &["person", "award", "year"], 0.5)
        .expect("CUDA IoBinding extract");

    eprintln!("CUDA + IoBinding result ({}): {:#?}", result.len(), result);
    assert!(
        !result.is_empty(),
        "CUDA + IoBinding produced no entities; expected at least one for {text:?}"
    );
}
