/// Run a multi-task extraction (NER + classification + structure) using the
/// `gliner_multitask` backend.
///
/// The backend loads `onnx-community/gliner-multitask-large-v0.5` (GLiNER v1
/// with task-conditioned label prompts; Stepanov & Shtopko 2024,
/// arXiv:2406.12925). Note: this is NOT the fastino-ai GLiNER2 architecture —
/// `fastino/gliner2-*` model IDs will be rejected at load time. See issue #17.
///
/// Requires the `onnx` feature. Downloads the model on first run (~700 MB).
///
/// ```sh
/// cargo run --example gliner_multitask --features onnx
/// ```
///
/// Expected output (illustrative; exact spans depend on model build):
///
/// ```text
/// === Entities ===
///   person: Marie Curie (0,11) 0.97
///   substance: radium (22,28) 0.91
///   place: Paris (32,37) 0.85
///
/// === Classification (sentiment) ===
///   neutral: 0.73
///
/// === Structure (event) ===
///   instance 0:
///     actor   : Marie Curie
///     action  : discovered
///     target  : radium
/// ```
#[cfg(feature = "onnx")]
fn main() -> anno::Result<()> {
    use anno::backends::gliner_multitask::{GLiNERMultitaskOnnx, TaskSchema};
    use anno::DEFAULT_GLINER_MULTITASK_MODEL;

    let model = GLiNERMultitaskOnnx::from_pretrained(DEFAULT_GLINER_MULTITASK_MODEL)?;

    let text = "Marie Curie discovered radium in Paris.";

    // A schema combines NER, classification, and structure extraction in one
    // call. Each task type is optional; empty schemas skip the corresponding
    // forward-pass logic.
    let schema = TaskSchema::new()
        .with_entities(&["person", "substance", "place"])
        .with_classification(
            "sentiment",
            &["positive", "negative", "neutral"],
            /* multi_label */ false,
        );

    let result = model.extract(text, &schema)?;

    println!("=== Entities ===");
    for e in &result.entities {
        println!(
            "  {}: {} ({},{}) {:.2}",
            e.entity_type,
            e.text,
            e.start(),
            e.end(),
            e.confidence
        );
    }

    if let Some(class) = result.classifications.get("sentiment") {
        println!("\n=== Classification (sentiment) ===");
        for (label, score) in &class.scores {
            println!("  {label}: {score:.2}");
        }
    }

    Ok(())
}

#[cfg(not(feature = "onnx"))]
fn main() {
    eprintln!("This example requires the `onnx` feature. Run with:");
    eprintln!("  cargo run --example gliner_multitask --features onnx");
    std::process::exit(1);
}
