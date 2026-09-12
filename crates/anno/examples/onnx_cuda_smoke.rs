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
//! - `OnnxExecutionProvider::Cuda` flows through
//!   `create_onnx_session_with_provider` into ort's
//!   `CUDAExecutionProvider::default().build().error_on_failure()`. CUDA must
//!   register or the smoke exits with an error; it cannot silently fall back
//!   during provider registration.
//! - The model loads successfully under the resulting session.
//!
//! ## What this does NOT validate
//!
//! - **Full graph placement.** Successful EP registration proves CUDA was
//!   available to ONNX Runtime. It does not prove every model node runs on the
//!   GPU, because ONNX Runtime may assign unsupported nodes to CPU.
//!
//! Exit code: 0 on session-creation success, non-zero on any earlier failure.
//!
//! ## Model source
//!
//! Pass one optional positional path to load a local `.onnx` file without a
//! network request. With no argument, the smoke retains its default download
//! of `onnx-community/gliner_small-v2.1`.

#[cfg(not(all(feature = "onnx", feature = "onnx-cuda")))]
fn main() {
    eprintln!(
        "onnx_cuda_smoke requires --features onnx,onnx-cuda; rebuild with those flags on a CUDA host"
    );
    std::process::exit(2);
}

#[cfg(all(feature = "onnx", feature = "onnx-cuda"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use anno::{create_onnx_session_with_provider, OnnxExecutionProvider, OnnxSessionConfig};
    use hf_hub::api::sync::Api;

    // Same model anno's gliner_onnx backend exercises -- if this loads, the
    // real backend's path works too.
    const MODEL_REPO: &str = "onnx-community/gliner_small-v2.1";
    const MODEL_FILE: &str = "onnx/model.onnx";

    let mut args = std::env::args_os();
    let _program = args.next();
    let model_path = match (args.next(), args.next()) {
        (Some(path), None) => {
            let path = std::path::PathBuf::from(path);
            if !path.is_file() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("local ONNX model is not a file: {}", path.display()),
                )
                .into());
            }
            path
        }
        (Some(_), Some(_)) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "usage: onnx_cuda_smoke [path/to/model.onnx]",
            )
            .into());
        }
        (None, _) => {
            eprintln!("[smoke] downloading {}/{}", MODEL_REPO, MODEL_FILE);
            let api = Api::new()?;
            let repo = api.model(MODEL_REPO.to_string());
            repo.get(MODEL_FILE)?
        }
    };
    eprintln!("[smoke] model at {}", model_path.display());

    eprintln!("[smoke] building session with CUDA registration required");
    let session = create_onnx_session_with_provider(
        &model_path,
        OnnxSessionConfig::default(),
        OnnxExecutionProvider::Cuda,
    )?;

    // List input names for visibility -- if these surface, the model graph
    // parsed and the session is ready. Inputs are model-specific; for
    // gliner_small-v2.1 we expect six (input_ids, attention_mask, words_mask,
    // text_lengths, span_idx, span_mask).
    eprintln!("[smoke] session ready. inputs:");
    for input in session.inputs().iter() {
        eprintln!("  - {}", input.name());
    }

    eprintln!("[smoke] PASS (CUDA provider registered; graph placement is model-dependent)");
    Ok(())
}
