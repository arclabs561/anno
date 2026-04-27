//! ONNX CUDA EP smoke test.
//!
//! Single-job: verify the `onnx-cuda` feature compiles, links, and creates a
//! session that registers the CUDA execution provider. Compiles with
//! `--features onnx,onnx-cuda`. Run on a real NVIDIA GPU host.
//!
//! Per-EP smoke layout (one file per EP, no shared module):
//! - CUDA       -> this file (`onnx_cuda_smoke.rs`)
//! - TensorRT   -> `onnx_tensorrt_smoke.rs` (deferred, same AWS hardware)
//! - DirectML   -> `onnx_directml_smoke.rs` (deferred, Windows EC2)
//! - ROCm       -> not on AWS (no AMD GPUs in EC2); separate cloud, separate driver
//! - CoreML     -> validated locally on Apple Silicon (already shipped, f891a31)
//!
//! ## What this validates
//!
//! - The `onnx-cuda` cargo feature compiles cleanly (rust side).
//! - `ort/cuda` links cleanly against the host's CUDA runtime (we caught a
//!   glibc 2.35 vs 2.38 mismatch on Ubuntu 22.04 here -- the AWS smoke uses
//!   Ubuntu 24.04 / glibc 2.39).
//! - `OnnxSessionConfig::prefer_cuda = true` flows through `create_onnx_session`
//!   into ort's `with_execution_providers([CUDAExecutionProvider::default().build()])`
//!   without erroring -- meaning ort accepts the EP registration request.
//! - The model loads successfully under the resulting session.
//!
//! ## What this does NOT validate (deferred)
//!
//! - **Silent CPU fallback.** ort 2.0's known failure mode: feature compiles,
//!   EP "loads" without erroring, but actual ops dispatch to CPU because the
//!   underlying runtime library cannot handle the model. Detecting this needs
//!   a timing comparison (CUDA-on vs CUDA-off) on the same model -- but doing
//!   that reliably requires getting model input shapes + types exactly right,
//!   which is brittle for general models. A future iteration should either
//!   route through anno's `GLiNEROnnx` backend (which knows the inputs) with
//!   a `prefer_cuda` toggle, or vendor a tiny known-input ONNX fixture for
//!   this purpose.
//!
//! Exit code: 0 on session-creation success, non-zero on any earlier failure.

#[cfg(not(all(feature = "onnx", feature = "onnx-cuda")))]
fn main() {
    eprintln!(
        "onnx_cuda_smoke requires --features onnx,onnx-cuda; rebuild with those flags on a CUDA host"
    );
    std::process::exit(2);
}

#[cfg(all(feature = "onnx", feature = "onnx-cuda"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use anno::{create_onnx_session, OnnxSessionConfig};
    use hf_hub::api::sync::Api;

    // Same model anno's gliner_onnx backend exercises -- if this loads, the
    // real backend's path works too.
    const MODEL_REPO: &str = "onnx-community/gliner_small-v2.1";
    const MODEL_FILE: &str = "onnx/model.onnx";

    eprintln!("[smoke] downloading {}/{}", MODEL_REPO, MODEL_FILE);
    let api = Api::new()?;
    let repo = api.model(MODEL_REPO.to_string());
    let model_path = repo.get(MODEL_FILE)?;
    eprintln!("[smoke] model at {}", model_path.display());

    // OnnxSessionConfig is `#[non_exhaustive]` so struct-literal construction
    // does not work from outside the `anno` crate. Mutate a default instead --
    // this is the pattern external users adopt.
    let mut cfg = OnnxSessionConfig::default();
    cfg.prefer_cuda = true;
    cfg.use_cpu_provider = true; // CPU as fallback so session loads even if CUDA op-coverage is incomplete

    eprintln!("[smoke] building session with prefer_cuda=true");
    let session = create_onnx_session(&model_path, cfg)?;

    // List input names for visibility -- if these surface, the model graph
    // parsed and the session is ready. Inputs are model-specific; for
    // gliner_small-v2.1 we expect six (input_ids, attention_mask, words_mask,
    // text_lengths, span_idx, span_mask).
    eprintln!("[smoke] session ready. inputs:");
    for input in session.inputs().iter() {
        eprintln!("  - {}", input.name());
    }

    eprintln!("[smoke] PASS (session-creation validated; inference benchmark deferred)");
    Ok(())
}
