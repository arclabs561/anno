//! ONNX CoreML EP smoke test.
//!
//! Single-job: verify the `onnx-coreml` feature compiles, links, and creates
//! a session that registers the CoreML execution provider. Compiles with
//! `--features onnx,onnx-coreml`. Run locally on Apple Silicon (M1/M2/M3+) --
//! the underlying `ort/coreml` feature requires macOS at link time.
//!
//! Per-EP smoke layout (one file per EP, no shared module):
//! - CUDA       -> `onnx_cuda_smoke.rs` (AWS g4dn.xlarge)
//! - TensorRT   -> `onnx_tensorrt_smoke.rs` (deferred, same AWS hardware)
//! - DirectML   -> `onnx_directml_smoke.rs` (deferred, Windows EC2)
//! - ROCm       -> not on AWS (no AMD GPUs in EC2)
//! - CoreML     -> this file (`onnx_coreml_smoke.rs`)
//!
//! ## What this validates
//!
//! - The `onnx-coreml` cargo feature compiles cleanly.
//! - `ort/coreml` links cleanly against the host's CoreML.framework.
//! - `OnnxSessionConfig::prefer_coreml = true` flows through
//!   `create_onnx_session` into ort's
//!   `with_execution_providers([CoreMLExecutionProvider::default().build()])`
//!   without erroring.
//! - The model loads under the resulting session.
//!
//! ## What this does NOT validate (deferred -- same gap as onnx_cuda_smoke)
//!
//! - **Silent CPU fallback.** Detecting this requires inference timing on the
//!   same model with CoreML on/off. See `onnx_cuda_smoke.rs` for the same
//!   open question and the path forward (route through anno's GLiNEROnnx
//!   backend with a CoreML toggle, or vendor a tiny known-input ONNX fixture).
//!
//! ## Running
//!
//! ```bash
//! cargo run --example onnx_coreml_smoke --features onnx,onnx-coreml
//! ```
//!
//! No EC2/AWS plumbing -- this runs on the dev macOS box.
//!
//! Exit code: 0 on session-creation success, non-zero on any earlier failure.

#[cfg(not(all(feature = "onnx", feature = "onnx-coreml")))]
fn main() {
    eprintln!(
        "onnx_coreml_smoke requires --features onnx,onnx-coreml; rebuild with those flags on macOS"
    );
    std::process::exit(2);
}

#[cfg(all(feature = "onnx", feature = "onnx-coreml"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use anno::{create_onnx_session, OnnxSessionConfig};
    use hf_hub::api::sync::Api;

    if !cfg!(target_os = "macos") {
        eprintln!("onnx_coreml_smoke must run on macOS (CoreML.framework dependency). Skipping.");
        std::process::exit(2);
    }

    // Same model anno's gliner_onnx backend exercises -- if this loads, the
    // real backend's path works too.
    const MODEL_REPO: &str = "onnx-community/gliner_small-v2.1";
    const MODEL_FILE: &str = "onnx/model.onnx";

    eprintln!("[smoke] downloading {}/{}", MODEL_REPO, MODEL_FILE);
    let api = Api::new()?;
    let repo = api.model(MODEL_REPO.to_string());
    let model_path = repo.get(MODEL_FILE)?;
    eprintln!("[smoke] model at {}", model_path.display());

    // Mutate-default for non_exhaustive OnnxSessionConfig.
    let mut cfg = OnnxSessionConfig::default();
    cfg.prefer_coreml = true;
    cfg.use_cpu_provider = true; // CPU as fallback so session loads even if CoreML op-coverage is incomplete

    eprintln!("[smoke] building session with prefer_coreml=true");
    let session = create_onnx_session(&model_path, cfg)?;

    eprintln!("[smoke] session ready. inputs:");
    for input in session.inputs().iter() {
        eprintln!("  - {}", input.name());
    }

    eprintln!("[smoke] PASS (session-creation validated; inference benchmark deferred)");
    Ok(())
}
