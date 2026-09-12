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
//! - `OnnxExecutionProvider::CoreMl` flows through
//!   `create_onnx_session_with_provider` into ort's
//!   `CoreMLExecutionProvider::default().build().error_on_failure()`. CoreML
//!   must register or the smoke exits with an error.
//! - The model loads under the resulting session.
//!
//! ## What this does NOT validate
//!
//! - **Full graph placement.** Successful EP registration proves CoreML was
//!   available to ONNX Runtime. It does not prove every model node runs on the
//!   accelerator, because ONNX Runtime may assign unsupported nodes to CPU.
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
//!
//! ## Model source
//!
//! Pass one optional positional path to load a local `.onnx` file without a
//! network request. With no argument, the smoke retains its default download
//! of `onnx-community/gliner_small-v2.1`.

#[cfg(not(all(feature = "onnx", feature = "onnx-coreml")))]
fn main() {
    eprintln!(
        "onnx_coreml_smoke requires --features onnx,onnx-coreml; rebuild with those flags on macOS"
    );
    std::process::exit(2);
}

#[cfg(all(feature = "onnx", feature = "onnx-coreml"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use anno::{create_onnx_session_with_provider, OnnxExecutionProvider, OnnxSessionConfig};
    use hf_hub::api::sync::Api;

    if !cfg!(target_os = "macos") {
        eprintln!("onnx_coreml_smoke must run on macOS (CoreML.framework dependency). Skipping.");
        std::process::exit(2);
    }

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
                "usage: onnx_coreml_smoke [path/to/model.onnx]",
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

    eprintln!("[smoke] building session with CoreML registration required");
    let session = create_onnx_session_with_provider(
        &model_path,
        OnnxSessionConfig::default(),
        OnnxExecutionProvider::CoreMl,
    )?;

    eprintln!("[smoke] session ready. inputs:");
    for input in session.inputs().iter() {
        eprintln!("  - {}", input.name());
    }

    eprintln!("[smoke] PASS (CoreML provider registered; graph placement is model-dependent)");
    Ok(())
}
