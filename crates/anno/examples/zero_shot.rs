/// Extract entities with caller-defined types using GLiNER.
///
/// Requires the `onnx` feature (default) and downloads the GLiNER model
/// on first run (~100 MB from HuggingFace).
///
/// ```sh
/// cargo run --example zero_shot
/// ```
///
/// Expected output:
///
/// ```text
/// drug: Aspirin (0,7) 0.94
/// symptom: headaches (25,34) 0.87
/// symptom: fever (46,51) 0.82
/// ```
#[cfg(feature = "onnx")]
fn main() -> anno::Result<()> {
    use anno::GLiNEROnnx;

    let m = GLiNEROnnx::new("onnx-community/gliner_small-v2.1")?;
    let text = "Aspirin can treat headaches and reduce fever.";
    let ents = m.extract(text, &["drug", "symptom"], 0.5)?;

    for e in &ents {
        println!(
            "{}: {} ({},{}) {:.2}",
            e.entity_type,
            e.text,
            e.start(),
            e.end(),
            e.confidence
        );
    }
    Ok(())
}

#[cfg(not(feature = "onnx"))]
fn main() {
    eprintln!(
        "This example requires the `onnx` feature: cargo run --example zero_shot --features onnx"
    );
}
