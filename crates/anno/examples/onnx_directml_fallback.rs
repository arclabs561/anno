//! Check explicit DirectML selection against a CPU distribution of ONNX Runtime.
//! The committed constant-logits fixture is a positive control for runtime and
//! model loading. This probe exercises the registration failure path only.
use anno::{create_onnx_session_with_provider, OnnxExecutionProvider, OnnxSessionConfig};
use ort::ep::ExecutionProvider;
use ort::value::Tensor;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/bughunt7_data/bert_cjk/model.onnx");
    let config = OnnxSessionConfig::default();
    let mut cpu = create_onnx_session_with_provider(&path, config, OnnxExecutionProvider::Cpu)?;
    let ids = Tensor::from_array(([1, 9], vec![0_i64; 9]))?;
    let mask = Tensor::from_array(([1, 9], vec![1_i64; 9]))?;
    let outputs = cpu.run(ort::inputs!["input_ids" => ids, "attention_mask" => mask])?;
    let (shape, scores) = outputs["logits"].try_extract_tensor::<f32>()?;
    assert_eq!(&shape[..], &[1, 9, 9]);
    for (row, tag) in scores.chunks_exact(9).zip([0, 0, 0, 7, 8, 0, 0, 0, 0]) {
        for (i, &score) in row.iter().enumerate() {
            assert_eq!(score, if i == tag { 10.0 } else { -10.0 });
        }
    }
    assert!(
        !ort::ep::DirectML::default().is_available()?,
        "expected a CPU runtime with no DirectML provider"
    );
    let error = create_onnx_session_with_provider(
        &path,
        OnnxSessionConfig::default(),
        OnnxExecutionProvider::DirectMl,
    )
    .expect_err("explicit DirectML unexpectedly returned a session on a CPU-only runtime");
    let message = error.to_string();
    assert!(
        message.contains("ONNX execution providers"),
        "expected a provider-registration failure: {message}"
    );
    println!(
        "{}",
        serde_json::json!({
            "ort_build": ort::info(), "cpu_logits_verified": scores.len(),
            "directml_advertised": false, "directml_registration_error": message,
        })
    );
    Ok(())
}
