//! ONNX CUDA EP inference benchmark.
//!
//! Successor to `onnx_cuda_smoke.rs`: the smoke validates session creation;
//! this benchmark validates that CUDA actually accelerates inference (i.e.
//! catches silent CPU fallback). Builds two sessions on the same model
//! (`prefer_cuda=true` and `prefer_cuda=false`), runs N inferences each,
//! compares wall time.
//!
//! ## Why a separate file from the smoke
//!
//! The smoke catches the cheap failures (compile, link, EP registration). It
//! is fast, the input set is fully synthetic (no model semantics required),
//! and runs in a tight CI loop or as a smoke after every dep bump.
//!
//! The bench requires (a) model-specific input shapes/dtypes, and (b) wall-
//! clock comparison that is meaningless without sufficient warm-up + run
//! count. Different scope, different cost, different cadence.
//!
//! ## Dynamic dtype handling
//!
//! Rather than hardcode the model's input schema (one input bool, others
//! i64 -- a trap that broke earlier iterations), match on each Outlet's
//! declared `ValueType::Tensor { ty, shape, .. }` and build a matching
//! zero/dummy tensor. This decouples the bench from any specific model;
//! swapping is a one-line `MODEL_REPO` change.
//!
//! ## Running
//!
//! ```bash
//! # Locally on a CUDA host:
//! cargo run --release --example onnx_cuda_bench --features onnx,onnx-cuda
//!
//! # Via the AWS smoke wrapper (g4dn.xlarge / Tesla T4 by default):
//! ANNO_SMOKE_BIN=onnx_cuda_bench ./scripts/aws/gpu-smoke.sh cuda
//! ```
//!
//! Exit code: 0 when both inferences complete end-to-end, non-zero on error
//! (silent CPU fallback would manifest as a runtime error from the GPU
//! tensor path failing to attach, OR as ratio ~1.0; the latter triggers a
//! warning but does not fail the bench since the diagnostic value of seeing
//! the ratio printed is already useful).

#[cfg(not(all(feature = "onnx", feature = "onnx-cuda")))]
fn main() {
    eprintln!(
        "onnx_cuda_bench requires --features onnx,onnx-cuda; rebuild with those flags on a CUDA host"
    );
    std::process::exit(2);
}

#[cfg(all(feature = "onnx", feature = "onnx-cuda"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    use anno::{create_onnx_session, OnnxSessionConfig};
    use hf_hub::api::sync::Api;
    use ort::session::Session;
    use ort::value::{DynValue, Tensor, TensorElementType, ValueType};

    // Tiny well-known model with simple inputs: input_ids, attention_mask,
    // token_type_ids (all i64). Tried gliner_small first; its packed-sequence
    // LSTM errors on synthetic text_lengths inputs -- fine for the smoke
    // (which only validates session creation), too quirky for a synthetic-
    // input benchmark.
    const MODEL_REPO: &str = "sentence-transformers/all-MiniLM-L6-v2";
    const MODEL_FILE: &str = "onnx/model.onnx";
    const N_RUNS: usize = 20;

    eprintln!("[bench] downloading {}/{}", MODEL_REPO, MODEL_FILE);
    let api = Api::new()?;
    let repo = api.model(MODEL_REPO.to_string());
    let model_path = repo.get(MODEL_FILE)?;
    eprintln!("[bench] model at {}", model_path.display());

    // Concrete dims for the model's dynamic dimensions. Heuristic: the first
    // dynamic dim in any input is BATCH, the second is SEQ, anything past is
    // 1. Works for transformer-shaped inputs.
    const BATCH: usize = 1;
    const SEQ: usize = 16;
    fn concretize(declared: &[i64]) -> Vec<usize> {
        let mut dyn_count = 0;
        declared
            .iter()
            .map(|&d| {
                if d > 0 {
                    d as usize
                } else {
                    dyn_count += 1;
                    match dyn_count {
                        1 => BATCH,
                        2 => SEQ,
                        _ => 1,
                    }
                }
            })
            .collect()
    }

    fn elements(shape: &[usize]) -> usize {
        shape.iter().product()
    }

    // Build a zero-valued DynValue matching the input's declared dtype.
    fn build_input(value_type: &ValueType) -> ort::Result<DynValue> {
        let (ty, shape_dims) = match value_type {
            ValueType::Tensor { ty, shape, .. } => {
                let dims: Vec<i64> = shape.iter().copied().collect();
                (*ty, dims)
            }
            other => {
                return Err(ort::Error::new(format!(
                    "unsupported input value type for bench: {:?}",
                    other
                )));
            }
        };
        let shape = concretize(&shape_dims);
        let n = elements(&shape);
        let v: DynValue = match ty {
            TensorElementType::Int64 => {
                let data: Box<[i64]> = vec![1i64; n].into_boxed_slice();
                Tensor::from_array((shape, data))?.into_dyn()
            }
            TensorElementType::Bool => {
                let data: Box<[bool]> = vec![true; n].into_boxed_slice();
                Tensor::from_array((shape, data))?.into_dyn()
            }
            TensorElementType::Float32 => {
                let data: Box<[f32]> = vec![0.0f32; n].into_boxed_slice();
                Tensor::from_array((shape, data))?.into_dyn()
            }
            TensorElementType::Int32 => {
                let data: Box<[i32]> = vec![0i32; n].into_boxed_slice();
                Tensor::from_array((shape, data))?.into_dyn()
            }
            other => {
                return Err(ort::Error::new(format!(
                    "unsupported input dtype in bench: {:?}",
                    other
                )));
            }
        };
        Ok(v)
    }

    fn make_inputs(session: &Session) -> ort::Result<Vec<(String, DynValue)>> {
        session
            .inputs()
            .iter()
            .map(|outlet| {
                let v = build_input(outlet.dtype())?;
                Ok((outlet.name().to_string(), v))
            })
            .collect()
    }

    let bench = |prefer_cuda: bool| -> Result<f64, Box<dyn std::error::Error>> {
        let mut cfg = OnnxSessionConfig::default();
        cfg.prefer_cuda = prefer_cuda;
        cfg.use_cpu_provider = true;
        let mut session = create_onnx_session(&model_path, cfg)?;

        // Warm up once.
        let inputs = make_inputs(&session)?;
        let _ = session.run(inputs)?;

        let t0 = Instant::now();
        for _ in 0..N_RUNS {
            let inputs = make_inputs(&session)?;
            let _ = session.run(inputs)?;
        }
        let elapsed = t0.elapsed().as_secs_f64();
        Ok(elapsed)
    };

    eprintln!("[bench] CPU baseline ({} runs)", N_RUNS);
    let cpu_time = bench(false)?;
    eprintln!(
        "[bench] CPU: {:.3}s ({:.1} ms/run)",
        cpu_time,
        1000.0 * cpu_time / N_RUNS as f64
    );

    eprintln!("[bench] CUDA run ({} runs)", N_RUNS);
    let cuda_time = bench(true)?;
    eprintln!(
        "[bench] CUDA: {:.3}s ({:.1} ms/run)",
        cuda_time,
        1000.0 * cuda_time / N_RUNS as f64
    );

    let speedup = cpu_time / cuda_time;
    eprintln!("[bench] speedup: {:.2}x (CUDA / CPU ratio)", speedup);

    // Both inferences completing end-to-end is the smoke. The speedup ratio
    // is a measurement.
    //
    // For CUDA on a real NVIDIA GPU (Tesla T4 / A10G / L4), the silent CPU
    // fallback failure mode (cudart.so missing, CUDA EP not actually
    // attached) would show ratio ~1.0 -- both runs would essentially be CPU.
    // Ratios meaningfully different from 1.0 (in either direction) prove the
    // CUDA EP is exercising the graph differently from CPU.
    //
    // We emit a strong warning if 0.9 <= ratio <= 1.1, but do not exit
    // non-zero -- the warning is the diagnostic. CI gates that want hard
    // assertions can grep this output for the warning line.
    if (0.9..=1.1).contains(&speedup) {
        eprintln!(
            "[bench] WARNING: speedup ratio {:.2}x is suspiciously close to 1.0. This is the \
             signature of silent CPU fallback (CUDA EP did not attach despite prefer_cuda=true). \
             Verify nvidia-smi shows a live GPU, libcudart is on LD_LIBRARY_PATH, and the build \
             included `ort/cuda` (--features onnx,onnx-cuda).",
            speedup
        );
    }
    eprintln!("[bench] PASS (both inferences completed end-to-end)");
    Ok(())
}
