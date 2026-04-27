//! ONNX CUDA EP smoke test.
//!
//! Single-job: verify the `onnx-cuda` feature is not silently CPU-falling-back.
//! Compiles with `--features onnx,onnx-cuda`. Run on a real NVIDIA GPU host.
//!
//! Per-EP smoke layout (one file per EP, no shared module):
//! - CUDA       -> this file (`onnx_cuda_smoke.rs`)
//! - TensorRT   -> `onnx_tensorrt_smoke.rs` (deferred, same AWS hardware)
//! - DirectML   -> `onnx_directml_smoke.rs` (deferred, Windows EC2)
//! - ROCm       -> not on AWS (no AMD GPUs in EC2); separate cloud, separate driver
//! - CoreML     -> validated locally on Apple Silicon (already shipped, f891a31)
//!
//! Non-goals (do not extend this file):
//! - End-to-end backend coverage (GLiNER, NuNER, etc) -- that is `anno-eval`'s job.
//! - Benchmarking or perf regression tracking -- that is `/perf`'s job.
//! - Multi-EP dispatch in one binary -- one file per EP, see layout above.
//!
//! Validation: builds two ONNX sessions on the same model, one with `prefer_cuda=true`,
//! one with `prefer_cuda=false`. Runs N=20 inferences on each, compares wall time.
//! Asserts `cpu_time / cuda_time >= MIN_SPEEDUP` (default 3x). On a g5.xlarge
//! against gliner_small the typical ratio is 8-15x; falling below 3x almost
//! always means the CUDA EP didn't actually attach and ort silently fell back
//! to CPU -- the failure mode this smoke exists to catch.
//!
//! Exit code: 0 on success, non-zero on any failure (download, session build,
//! inference, or speedup-below-threshold).

#[cfg(not(all(feature = "onnx", feature = "onnx-cuda")))]
fn main() {
    eprintln!(
        "onnx_cuda_smoke requires --features onnx,onnx-cuda; rebuild with those flags on a CUDA host"
    );
    std::process::exit(2);
}

#[cfg(all(feature = "onnx", feature = "onnx-cuda"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    use anno::backends::hf_loader::{create_onnx_session, download_model_file, OnnxSessionConfig};

    // Small known-good ONNX model. ~50MB; downloads once and caches under HF hub
    // default cache. Picked because it is the same model the gliner_onnx backend
    // exercises -- if this works, the real backends work.
    const MODEL_REPO: &str = "onnx-community/gliner_small-v2.1";
    const MODEL_FILE: &str = "onnx/model.onnx";
    const N_RUNS: usize = 20;
    const MIN_SPEEDUP: f64 = 3.0;

    eprintln!("[smoke] downloading {}/{}", MODEL_REPO, MODEL_FILE);
    let model_path = download_model_file(MODEL_REPO, MODEL_FILE)?;
    eprintln!("[smoke] model at {}", model_path.display());

    // Build an input tensor matching the model's input signature. gliner_small uses
    // input_ids [batch, seq], attention_mask [batch, seq], words_mask [batch, seq],
    // text_lengths [batch, 1]. We feed a tiny synthetic batch -- correctness of the
    // output is not the smoke; latency under CUDA vs CPU is.
    let batch: usize = 1;
    let seq: usize = 16;

    let bench = |prefer_cuda: bool| -> Result<f64, Box<dyn std::error::Error>> {
        use ndarray::Array2;
        use ort::value::Value;

        let cfg = OnnxSessionConfig {
            prefer_cuda,
            use_cpu_provider: true,
            ..Default::default()
        };
        let mut session = create_onnx_session(&model_path, cfg)?;

        let input_ids = Array2::<i64>::from_elem((batch, seq), 1);
        let attention_mask = Array2::<i64>::from_elem((batch, seq), 1);
        let words_mask = Array2::<i64>::from_elem((batch, seq), 1);
        let text_lengths = Array2::<i64>::from_elem((batch, 1), seq as i64);

        // Warm up once (JIT, allocator, kernel cache).
        let _ = session.run(ort::inputs![
            "input_ids" => Value::from_array(input_ids.clone())?,
            "attention_mask" => Value::from_array(attention_mask.clone())?,
            "words_mask" => Value::from_array(words_mask.clone())?,
            "text_lengths" => Value::from_array(text_lengths.clone())?,
        ])?;

        let t0 = Instant::now();
        for _ in 0..N_RUNS {
            let _ = session.run(ort::inputs![
                "input_ids" => Value::from_array(input_ids.clone())?,
                "attention_mask" => Value::from_array(attention_mask.clone())?,
                "words_mask" => Value::from_array(words_mask.clone())?,
                "text_lengths" => Value::from_array(text_lengths.clone())?,
            ])?;
        }
        let elapsed = t0.elapsed().as_secs_f64();
        Ok(elapsed)
    };

    eprintln!("[smoke] CPU baseline ({} runs)", N_RUNS);
    let cpu_time = bench(false)?;
    eprintln!(
        "[smoke] CPU: {:.3}s ({:.1} ms/run)",
        cpu_time,
        1000.0 * cpu_time / N_RUNS as f64
    );

    eprintln!("[smoke] CUDA run ({} runs)", N_RUNS);
    let cuda_time = bench(true)?;
    eprintln!(
        "[smoke] CUDA: {:.3}s ({:.1} ms/run)",
        cuda_time,
        1000.0 * cuda_time / N_RUNS as f64
    );

    let speedup = cpu_time / cuda_time;
    eprintln!(
        "[smoke] speedup: {:.2}x (require >= {:.1}x)",
        speedup, MIN_SPEEDUP
    );

    if speedup < MIN_SPEEDUP {
        eprintln!(
            "[smoke] FAIL: speedup {:.2}x below threshold {:.1}x. CUDA EP likely did not attach \
             (silent CPU fallback). Check `nvidia-smi`, libcudart presence, and that the build \
             included `ort/cuda` (--features onnx,onnx-cuda).",
            speedup, MIN_SPEEDUP
        );
        std::process::exit(1);
    }

    eprintln!("[smoke] PASS");
    Ok(())
}
