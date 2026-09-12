//! Profile a real cached BERT NER graph through CPU and CoreML execution providers.
//!
//! This probe is deliberately cache-only: it accepts a local model directory
//! containing `model.onnx`, `tokenizer.json`, and `config.json`, and never
//! invokes Hugging Face. It uses anno's BERT diagnostic trace for the exact
//! tokenizer inputs and CPU decoding, then compares the same tokenized graph
//! output under CPU and CoreML. ONNX Runtime profiling records the provider
//! named for every executed node, including CPU fallback nodes.
//!
//! ```text
//! cargo run -p anno --release --example onnx_bert_provider_probe \
//!   --features onnx,onnx-coreml -- \
//!   /path/to/protectai-snapshot /tmp/anno-bert-provider-probe
//! ```
//!
//! The JSON receipt is a diagnostic artifact, not a benchmark score. It
//! records provider assignment and warm timings for one fixed input; repeat it
//! on a representative workload before using it for a provider decision.

#[cfg(not(all(feature = "onnx", feature = "onnx-coreml")))]
fn main() {
    eprintln!(
        "onnx_bert_provider_probe requires --features onnx,onnx-coreml; run it on macOS with a local model directory"
    );
    std::process::exit(2);
}

#[cfg(all(feature = "onnx", feature = "onnx-coreml"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::{
        fs,
        path::{Path, PathBuf},
    };

    use anno::{BertNEROnnx, BertNerTrace};
    use serde::Serialize;

    const WARM_RUNS: usize = 3;
    const TIMED_RUNS: usize = 12;
    // This is the original strict numerical gate. A future, looser bound must
    // be justified with multiple fixed inputs and preserved strict receipts.
    const MAX_ABS_LOGIT_DELTA: f32 = 1e-4;
    const TEXT: &str = "Apple hired Alice Johnson in Paris on 12 September 2026.";

    if !cfg!(target_os = "macos") {
        return Err("onnx_bert_provider_probe requires macOS/CoreML".into());
    }

    let mut args = std::env::args_os();
    let program = args.next().unwrap_or_default();
    let model_dir = args.next().map(PathBuf::from).ok_or_else(|| {
        format!(
            "usage: {} /path/to/model-dir /path/to/output-dir",
            Path::new(&program).display()
        )
    })?;
    let output_dir = args.next().map(PathBuf::from).ok_or_else(|| {
        format!(
            "usage: {} /path/to/model-dir /path/to/output-dir",
            Path::new(&program).display()
        )
    })?;
    if args.next().is_some() {
        return Err("expected exactly a model directory and output directory".into());
    }
    for file in ["model.onnx", "tokenizer.json", "config.json"] {
        if !model_dir.join(file).is_file() {
            return Err(format!("missing {} in {}", file, model_dir.display()).into());
        }
    }
    fs::create_dir_all(&output_dir)?;

    // This diagnostic is the production BERT tokenizer and decoder path. Keep
    // it separate from direct sessions below so the receipt can distinguish
    // anno's CPU result from the provider comparison.
    let anno_model = BertNEROnnx::new(
        model_dir
            .to_str()
            .ok_or("model directory is not valid UTF-8")?,
    )?;
    let trace = anno_model.debug_trace(TEXT)?;
    let graph = anno_model.artifact_paths().graph.clone();

    let cpu_profile_prefix = output_dir.join("bert-cpu-profile");
    let coreml_profile_prefix = output_dir.join("bert-coreml-profile");
    let mut cpu = build_session(&graph, &cpu_profile_prefix, false)?;
    let mut coreml = build_session(&graph, &coreml_profile_prefix, true)?;

    let cpu_logits = run_logits(&mut cpu, &trace)?;
    let coreml_logits = run_logits(&mut coreml, &trace)?;
    let max_abs_logit_delta = max_abs_delta(&cpu_logits, &coreml_logits)?;
    let label_count = trace
        .logits
        .first()
        .map(Vec::len)
        .ok_or("anno BERT trace returned no logit rows")?;
    let argmax_labels_match =
        argmaxes(&cpu_logits, label_count)? == argmaxes(&coreml_logits, label_count)?;
    let cpu_min_argmax_margin = min_argmax_margin(&cpu_logits, label_count)?;
    let coreml_min_argmax_margin = min_argmax_margin(&coreml_logits, label_count)?;

    // Warm sessions independently before timing them. The comparison is
    // intentionally small and reports rather than asserts a speedup.
    for _ in 0..WARM_RUNS {
        let _ = run_logits(&mut cpu, &trace)?;
        let _ = run_logits(&mut coreml, &trace)?;
    }
    let cpu_warm_ms = mean_run_ms(&mut cpu, &trace, TIMED_RUNS)?;
    let coreml_warm_ms = mean_run_ms(&mut coreml, &trace, TIMED_RUNS)?;

    let cpu_profile = PathBuf::from(cpu.end_profiling()?);
    let coreml_profile = PathBuf::from(coreml.end_profiling()?);
    let cpu_assignments = summarize_profile(&cpu_profile)?;
    let coreml_assignments = summarize_profile(&coreml_profile)?;
    let coreml_nodes = coreml_assignments
        .by_provider
        .get("CoreMLExecutionProvider")
        .copied()
        .unwrap_or(0);
    let strict_logit_tolerance_pass = max_abs_logit_delta <= MAX_ABS_LOGIT_DELTA;
    let provider_assignment_pass = coreml_nodes > 0;

    #[derive(Serialize)]
    struct Receipt {
        schema_version: u32,
        text: &'static str,
        model_dir: PathBuf,
        graph: PathBuf,
        tokenizer: PathBuf,
        config: Option<PathBuf>,
        trace: BertNerTrace,
        strict_logit_tolerance_pass: bool,
        argmax_labels_match: bool,
        cpu_min_argmax_margin: f32,
        coreml_min_argmax_margin: f32,
        provider_assignment_pass: bool,
        max_abs_logit_delta: f32,
        max_abs_logit_delta_limit: f32,
        warm_runs: usize,
        timed_runs: usize,
        cpu_mean_ms: f64,
        coreml_mean_ms: f64,
        cpu_profile: PathBuf,
        coreml_profile: PathBuf,
        cpu_assignments: ProviderAssignments,
        coreml_assignments: ProviderAssignments,
    }

    let paths = anno_model.artifact_paths();
    let receipt = Receipt {
        schema_version: 1,
        text: TEXT,
        model_dir,
        graph,
        tokenizer: paths.tokenizer.clone(),
        config: paths.config.clone(),
        trace,
        strict_logit_tolerance_pass,
        argmax_labels_match,
        cpu_min_argmax_margin,
        coreml_min_argmax_margin,
        provider_assignment_pass,
        max_abs_logit_delta,
        max_abs_logit_delta_limit: MAX_ABS_LOGIT_DELTA,
        warm_runs: WARM_RUNS,
        timed_runs: TIMED_RUNS,
        cpu_mean_ms: cpu_warm_ms,
        coreml_mean_ms: coreml_warm_ms,
        cpu_profile,
        coreml_profile,
        cpu_assignments,
        coreml_assignments,
    };
    let receipt_path = output_dir.join("receipt.json");
    fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt)?)?;
    println!("provider probe receipt: {}", receipt_path.display());
    if !strict_logit_tolerance_pass || !argmax_labels_match || !provider_assignment_pass {
        return Err(format!(
            "provider probe failed (strict_logit_tolerance_pass={strict_logit_tolerance_pass}, \
             argmax_labels_match={argmax_labels_match}, \
             provider_assignment_pass={provider_assignment_pass}); receipt: {}",
            receipt_path.display()
        )
        .into());
    }
    Ok(())
}

#[cfg(all(feature = "onnx", feature = "onnx-coreml"))]
fn build_session(
    graph: &std::path::Path,
    profile_prefix: &std::path::Path,
    coreml: bool,
) -> Result<ort::session::Session, Box<dyn std::error::Error>> {
    use ort::{
        execution_providers::{CPUExecutionProvider, CoreMLExecutionProvider},
        session::{builder::GraphOptimizationLevel, Session},
    };

    let mut builder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .with_profiling(profile_prefix)?;
    builder = if coreml {
        builder.with_execution_providers([
            CoreMLExecutionProvider::default()
                .build()
                .error_on_failure(),
            CPUExecutionProvider::default().build(),
        ])?
    } else {
        builder.with_execution_providers([CPUExecutionProvider::default().build()])?
    };
    Ok(builder.commit_from_file(graph)?)
}

#[cfg(all(feature = "onnx", feature = "onnx-coreml"))]
fn run_logits(
    session: &mut ort::session::Session,
    trace: &anno::BertNerTrace,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use ort::value::Tensor;

    let sequence_len = trace.input_ids.len();
    let ids: Vec<i64> = trace.input_ids.iter().map(|&id| i64::from(id)).collect();
    let mask: Vec<i64> = trace
        .attention_mask
        .iter()
        .map(|&value| i64::from(value))
        .collect();
    let input_ids =
        Tensor::from_array(([1_usize, sequence_len], ids.into_boxed_slice()))?.into_dyn();
    let attention_mask =
        Tensor::from_array(([1_usize, sequence_len], mask.into_boxed_slice()))?.into_dyn();
    let has_type_ids = session
        .inputs()
        .iter()
        .any(|input| input.name() == "token_type_ids");
    let outputs = if has_type_ids {
        let type_ids = Tensor::from_array((
            [1_usize, sequence_len],
            vec![0_i64; sequence_len].into_boxed_slice(),
        ))?
        .into_dyn();
        session.run(ort::inputs![
            "input_ids" => input_ids,
            "attention_mask" => attention_mask,
            "token_type_ids" => type_ids,
        ])?
    } else {
        session.run(ort::inputs![
            "input_ids" => input_ids,
            "attention_mask" => attention_mask,
        ])?
    };
    let logits = outputs
        .get("logits")
        .ok_or("BERT graph did not produce a logits output")?;
    Ok(logits.try_extract_tensor::<f32>()?.1.to_vec())
}

#[cfg(all(feature = "onnx", feature = "onnx-coreml"))]
fn mean_run_ms(
    session: &mut ort::session::Session,
    trace: &anno::BertNerTrace,
    runs: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    let started = std::time::Instant::now();
    for _ in 0..runs {
        let _ = run_logits(session, trace)?;
    }
    Ok(started.elapsed().as_secs_f64() * 1000.0 / runs as f64)
}

#[cfg(any(test, all(feature = "onnx", feature = "onnx-coreml")))]
fn max_abs_delta(left: &[f32], right: &[f32]) -> Result<f32, Box<dyn std::error::Error>> {
    if left.is_empty() || right.is_empty() {
        return Err("empty BERT logits".into());
    }
    if left.iter().chain(right).any(|value| !value.is_finite()) {
        return Err("BERT logits contain a non-finite value".into());
    }
    if left.len() != right.len() {
        return Err(format!("logit lengths differ: {} != {}", left.len(), right.len()).into());
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f32, f32::max))
}

#[cfg(all(feature = "onnx", feature = "onnx-coreml"))]
fn argmaxes(logits: &[f32], label_count: usize) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    if label_count == 0 || !logits.len().is_multiple_of(label_count) {
        return Err(format!(
            "cannot split {} logits into rows with {label_count} labels",
            logits.len()
        )
        .into());
    }
    Ok(logits
        .chunks_exact(label_count)
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(index, _)| index)
                .expect("label_count is nonzero")
        })
        .collect())
}

#[cfg(all(feature = "onnx", feature = "onnx-coreml"))]
fn min_argmax_margin(
    logits: &[f32],
    label_count: usize,
) -> Result<f32, Box<dyn std::error::Error>> {
    if label_count < 2 || !logits.len().is_multiple_of(label_count) {
        return Err(format!(
            "cannot compute argmax margins for {} logits and {label_count} labels",
            logits.len()
        )
        .into());
    }
    logits
        .chunks_exact(label_count)
        .map(|row| {
            let mut top = f32::NEG_INFINITY;
            let mut second = f32::NEG_INFINITY;
            for &value in row {
                if value > top {
                    second = top;
                    top = value;
                } else if value > second {
                    second = value;
                }
            }
            top - second
        })
        .min_by(f32::total_cmp)
        .ok_or_else(|| "cannot compute argmax margin for empty logits".into())
}

#[cfg(all(feature = "onnx", feature = "onnx-coreml"))]
#[derive(Debug, serde::Serialize)]
struct ProviderAssignments {
    total_node_events: usize,
    unassigned_node_events: usize,
    by_provider: std::collections::BTreeMap<String, usize>,
}

#[cfg(all(feature = "onnx", feature = "onnx-coreml"))]
fn summarize_profile(
    path: &std::path::Path,
) -> Result<ProviderAssignments, Box<dyn std::error::Error>> {
    let events: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    let events = events
        .as_array()
        .ok_or("ORT profiling output is not a JSON array")?;
    let mut result = ProviderAssignments {
        total_node_events: 0,
        unassigned_node_events: 0,
        by_provider: std::collections::BTreeMap::new(),
    };
    for event in events {
        if event.get("cat").and_then(serde_json::Value::as_str) != Some("Node") {
            continue;
        }
        result.total_node_events += 1;
        match event
            .get("args")
            .and_then(|args| args.get("provider"))
            .and_then(serde_json::Value::as_str)
        {
            Some(provider) => *result.by_provider.entry(provider.to_string()).or_default() += 1,
            None => result.unassigned_node_events += 1,
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::max_abs_delta;

    #[test]
    fn numerical_gate_rejects_invalid_logits() {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(max_abs_delta(&[invalid], &[invalid]).is_err());
            assert!(max_abs_delta(&[1.0, invalid], &[1.0, 0.0]).is_err());
            assert!(max_abs_delta(&[1.0, 0.0], &[1.0, invalid]).is_err());
        }
        assert!(max_abs_delta(&[], &[]).is_err());
        assert!(max_abs_delta(&[1.0], &[1.0, 2.0]).is_err());
        assert_eq!(max_abs_delta(&[1.0, -2.0], &[1.25, -1.5]).unwrap(), 0.5);
    }
}
