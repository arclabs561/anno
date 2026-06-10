//! Phase 3.5: IoBinding-mode 8-session inference pipeline.
#![allow(missing_docs)] // implementation internals; public API is on GLiNER2Fastino in mod.rs
//!
//! Adapted from SemplificaAI/gliner2-rs/rust_component/src/lib_v2.rs:285-660
//! (`Gliner2EngineV2::extract_iobinding`, Apache-2.0). See
//! `docs/dev-notes/gliner2-iobinding-port-notes.md` for the symbol map.
//!
//! Each `run_*_io` function takes the prerequisite `Value`s from the
//! prior session and runs its session via `IoBinding`. Inter-session
//! tensors stay device-resident (CPU on stock Phase 3.5; future GPU
//! providers via `OnnxSessionConfig` will keep them in EP-allocated
//! buffers). Only the final `scorer` output is read back to host as
//! `ndarray::Array4<f32>` — the cost is amortised over the 7 chained
//! device-resident outputs.
//!
//! `MemoryInfo` is created per-call (not cached on the engine) because
//! `ort::memory::MemoryInfo` is `!Send + !Sync` (contains a `NonNull`).
//! Caching it on the engine would break the engine's `Send + Sync`
//! requirement for the `Model` and `ZeroShotNER` traits. Per-call cost
//! is negligible — pure metadata, no buffer allocation.

#![allow(dead_code)] // Phase 3.5 in-progress: these are wired up by M11.

use crate::backends::gliner2_fastino::errors::Error;
use crate::backends::gliner2_fastino::processor::ProcessedRecord;
use crate::backends::gliner2_fastino::sessions::Sessions;
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::value::DynValue;

/// Build the device `MemoryInfo` used for chained inter-session outputs.
///
/// Phase 3.5 ships CPU-only. CUDA / CoreML providers will widen this
/// in Phase 4 once an `OnnxSessionConfig::prefer_cuda`-aware variant
/// is wired in.
pub(crate) fn device_memory_info() -> ort::Result<MemoryInfo> {
    MemoryInfo::new(
        AllocationDevice::CPU,
        0,
        AllocatorType::Device,
        MemoryType::Default,
    )
}

/// Build the CPU-output `MemoryInfo` used for outputs that must be
/// read back to host (e.g. `count_pred_argmax`'s scalar i64 result).
pub(crate) fn cpu_output_memory_info() -> ort::Result<MemoryInfo> {
    MemoryInfo::new(
        AllocationDevice::CPU,
        0,
        AllocatorType::Device,
        MemoryType::CPUOutput,
    )
}

/// Output of the encoder step, kept as a device-resident `DynValue`.
/// The wrapped value owns its buffer and is safe to thread through
/// later `run_*_io` functions as input.
pub(crate) struct EncoderOutputIo {
    pub hidden_states: DynValue,
    /// Cached output tensor name (varies across exports —
    /// `last_hidden_state` / `hidden_states` / `output`). Kept so callers
    /// don't have to re-resolve it.
    pub output_name: String,
}

/// Run the encoder graph via IoBinding. Inputs (`input_ids`,
/// `attention_mask`) come from `record`; the output is bound to
/// `device_mem` and returned as a device-resident `DynValue`.
pub(crate) fn run_encoder_io(
    sessions: &Sessions,
    record: &ProcessedRecord,
    device_mem: &MemoryInfo,
) -> Result<EncoderOutputIo, Error> {
    use ndarray::Array2;

    let seq_len = record.input_ids.len();
    let input_ids: Array2<i64> = Array2::from_shape_vec((1, seq_len), record.input_ids.clone())
        .map_err(|e| Error::Tokenizer(format!("encoder input_ids reshape: {e}")))?;
    let attn_mask: Array2<i64> =
        Array2::from_shape_vec((1, seq_len), record.attention_mask.clone())
            .map_err(|e| Error::Tokenizer(format!("encoder attn reshape: {e}")))?;

    let input_ids_t = crate::backends::ort_compat::tensor_from_ndarray(input_ids)
        .map_err(|e| Error::Tokenizer(format!("encoder input_ids tensor: {e}")))?;
    let attn_mask_t = crate::backends::ort_compat::tensor_from_ndarray(attn_mask)
        .map_err(|e| Error::Tokenizer(format!("encoder attn tensor: {e}")))?;

    sessions
        .encoder
        .with_session(|s| -> Result<EncoderOutputIo, Error> {
            // Resolve output name. Different fastino exports ship different
            // names — match the standard pipeline's priority order.
            let output_name =
                resolve_output_name(s, &["hidden_states", "last_hidden_state", "output"]);

            let mut binding = s
                .create_binding()
                .map_err(|e| Error::Tokenizer(format!("encoder create_binding: {e}")))?;
            binding
                .bind_input("input_ids", &input_ids_t)
                .map_err(|e| Error::Tokenizer(format!("encoder bind input_ids: {e}")))?;
            binding
                .bind_input("attention_mask", &attn_mask_t)
                .map_err(|e| Error::Tokenizer(format!("encoder bind attention_mask: {e}")))?;
            binding
                .bind_output_to_device(&output_name, device_mem)
                .map_err(|e| Error::Tokenizer(format!("encoder bind_output_to_device: {e}")))?;

            let outputs = s
                .run_binding(&binding)
                .map_err(|e| Error::Tokenizer(format!("encoder run_binding: {e}")))?;

            // Take the bound output by name. The returned `Value` owns its
            // buffer and is safe to return out of the with_session closure.
            let hidden_states = outputs
                .into_iter()
                .find_map(|(name, val)| {
                    if name == output_name.as_str() {
                        Some(val)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| {
                    Error::Tokenizer(format!(
                        "encoder: output '{output_name}' not present in run_binding result"
                    ))
                })?;

            Ok(EncoderOutputIo {
                hidden_states,
                output_name,
            })
        })
}

/// Look up an output's name in priority order. Falls back to the
/// session's first output if none of the candidates match — matches
/// the standard pipeline's behavior so IoBinding works on any export
/// the standard pipeline does.
fn resolve_output_name(session: &ort::session::Session, candidates: &[&str]) -> String {
    let session_outputs: Vec<String> = session
        .outputs()
        .iter()
        .map(|o| o.name().to_string())
        .collect();
    for &c in candidates {
        if session_outputs.iter().any(|n| n == c) {
            return c.to_string();
        }
    }
    session_outputs
        .into_iter()
        .next()
        .unwrap_or_else(|| candidates.first().copied().unwrap_or("output").to_string())
}

/// Output of token_gather: word-level embeddings.
/// Shape: `[1, num_words, H]`. Device-resident DynValue.
pub(crate) struct TokenGatherOutputIo {
    pub text_embs: DynValue,
    pub output_name: String,
}

/// Output of span_rep: span-level embeddings.
/// Shape: `[1, num_words, MAX_WIDTH, H]`. Device-resident DynValue.
pub(crate) struct SpanRepOutputIo {
    pub span_embs: DynValue,
    pub output_name: String,
}

/// Output of schema_gather: prompt + per-field embeddings.
/// Both DynValues are device-resident.
pub(crate) struct SchemaGatherOutputIo {
    /// Shape: `[1, H]` — prompt-context embedding (the [P]-token's row).
    pub pc_emb: DynValue,
    /// Shape: `[M, H]` — per-field/per-label embeddings.
    pub field_embs: DynValue,
    pub pc_output_name: String,
    pub field_output_name: String,
}

/// Output of count_lstm_fixed: struct projection used by scorer.
/// Shape: `[MAX_COUNT, M, H]`. Device-resident DynValue.
pub(crate) struct CountLstmOutputIo {
    pub struct_proj: DynValue,
    pub output_name: String,
}

/// Run token_gather via IoBinding. Inputs:
/// - `last_hidden_state` from [`EncoderOutputIo`] (device-resident).
/// - `word_indices` built from `record.word_to_token_maps[*].0`.
///
/// Output is bound to `device_mem` and returned as a device-resident
/// DynValue for chaining into [`run_span_rep_io`] (M7).
pub(crate) fn run_token_gather_io(
    sessions: &Sessions,
    encoder_out: &EncoderOutputIo,
    record: &ProcessedRecord,
    device_mem: &MemoryInfo,
) -> Result<TokenGatherOutputIo, Error> {
    use ndarray::Array1;

    let num_words = record.word_to_token_maps.len();
    if num_words == 0 {
        return Err(Error::Tokenizer(
            "token_gather_io: 0 words in record".into(),
        ));
    }
    let word_starts: Vec<i64> = record
        .word_to_token_maps
        .iter()
        .map(|&(start, _)| start as i64)
        .collect();
    let word_idx_arr: Array1<i64> = Array1::from_vec(word_starts);
    let word_idx_t = crate::backends::ort_compat::tensor_from_ndarray(word_idx_arr)
        .map_err(|e| Error::Tokenizer(format!("token_gather_io word_idx tensor: {e}")))?;

    sessions
        .token_gather
        .with_session(|s| -> Result<TokenGatherOutputIo, Error> {
            let output_name = resolve_output_name(s, &["text_embs"]);
            let mut binding = s
                .create_binding()
                .map_err(|e| Error::Tokenizer(format!("token_gather_io create_binding: {e}")))?;
            binding
                .bind_input("last_hidden_state", &encoder_out.hidden_states)
                .map_err(|e| {
                    Error::Tokenizer(format!("token_gather_io bind last_hidden_state: {e}"))
                })?;
            binding
                .bind_input("word_indices", &word_idx_t)
                .map_err(|e| Error::Tokenizer(format!("token_gather_io bind word_indices: {e}")))?;
            binding
                .bind_output_to_device(&output_name, device_mem)
                .map_err(|e| Error::Tokenizer(format!("token_gather_io bind_output: {e}")))?;
            let outputs = s
                .run_binding(&binding)
                .map_err(|e| Error::Tokenizer(format!("token_gather_io run_binding: {e}")))?;
            let text_embs = outputs
                .into_iter()
                .find_map(|(name, val)| {
                    if name == output_name.as_str() {
                        Some(val)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| {
                    Error::Tokenizer(format!(
                        "token_gather_io: output '{output_name}' not present"
                    ))
                })?;
            Ok(TokenGatherOutputIo {
                text_embs,
                output_name,
            })
        })
}

/// Run span_rep via IoBinding. Inputs:
/// - `hidden_states`: device-resident text_embs from [`TokenGatherOutputIo`].
/// - `span_idx`: i64 Array3 of shape `[1, num_words * MAX_WIDTH, 2]` built by
///   [`super::pipeline::build_span_idx`] (shared with the standard pipeline).
///
/// Output: span_embs DynValue of shape `[1, num_words, MAX_WIDTH, H]`,
/// device-resident, chained into M10's run_scorer_io.
pub(crate) fn run_span_rep_io(
    sessions: &Sessions,
    tg_out: &TokenGatherOutputIo,
    num_words: usize,
    device_mem: &MemoryInfo,
) -> Result<SpanRepOutputIo, Error> {
    let span_idx = super::pipeline::build_span_idx(num_words);
    let span_idx_t = crate::backends::ort_compat::tensor_from_ndarray(span_idx)
        .map_err(|e| Error::Tokenizer(format!("span_rep_io span_idx tensor: {e}")))?;

    sessions
        .span_rep
        .with_session(|s| -> Result<SpanRepOutputIo, Error> {
            let output_name = resolve_output_name(s, &["span_embeddings", "span_embs"]);
            let mut binding = s
                .create_binding()
                .map_err(|e| Error::Tokenizer(format!("span_rep_io create_binding: {e}")))?;
            binding
                .bind_input("hidden_states", &tg_out.text_embs)
                .map_err(|e| Error::Tokenizer(format!("span_rep_io bind hidden_states: {e}")))?;
            binding
                .bind_input("span_idx", &span_idx_t)
                .map_err(|e| Error::Tokenizer(format!("span_rep_io bind span_idx: {e}")))?;
            binding
                .bind_output_to_device(&output_name, device_mem)
                .map_err(|e| Error::Tokenizer(format!("span_rep_io bind_output: {e}")))?;
            let outputs = s
                .run_binding(&binding)
                .map_err(|e| Error::Tokenizer(format!("span_rep_io run_binding: {e}")))?;
            let span_embs = outputs
                .into_iter()
                .find_map(|(name, val)| {
                    if name == output_name.as_str() {
                        Some(val)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| {
                    Error::Tokenizer(format!("span_rep_io: output '{output_name}' not present"))
                })?;
            Ok(SpanRepOutputIo {
                span_embs,
                output_name,
            })
        })
}

/// Run schema_gather via IoBinding. Inputs:
/// - `last_hidden_state`: device-resident from [`EncoderOutputIo`].
/// - `schema_indices`: i64 Array1 built from
///   `[task.prompt_tok_idx, task.field_tok_indices...]`.
///
/// Outputs (both bound to device):
/// - `pc_emb`: `[1, H]` — prompt-context embedding.
/// - `field_embs`: `[M, H]` — per-field/per-label embeddings.
///
/// Output names follow upstream's convention: the schema_gather export
/// has its outputs in the order `pc_emb`, `field_embs` — we resolve by
/// position rather than by name to be robust.
pub(crate) fn run_schema_gather_io(
    sessions: &Sessions,
    encoder_out: &EncoderOutputIo,
    task: &crate::backends::gliner2_fastino::processor::TaskMapping,
    device_mem: &MemoryInfo,
) -> Result<SchemaGatherOutputIo, Error> {
    use ndarray::Array1;

    let mut indices: Vec<i64> = Vec::with_capacity(1 + task.field_tok_indices.len());
    indices.push(task.prompt_tok_idx as i64);
    indices.extend(task.field_tok_indices.iter().map(|&i| i as i64));
    let idx_arr: Array1<i64> = Array1::from_vec(indices);
    let idx_t = crate::backends::ort_compat::tensor_from_ndarray(idx_arr)
        .map_err(|e| Error::Tokenizer(format!("schema_gather_io idx tensor: {e}")))?;

    sessions
        .schema_gather
        .with_session(|s| -> Result<SchemaGatherOutputIo, Error> {
            // schema_gather has 2 outputs in fixed order: pc_emb (idx 0),
            // field_embs (idx 1). Resolve by position via the session's
            // outputs() metadata.
            let outs: Vec<String> = s.outputs().iter().map(|o| o.name().to_string()).collect();
            if outs.len() < 2 {
                return Err(Error::Tokenizer(format!(
                    "schema_gather_io: expected 2 outputs, found {}",
                    outs.len()
                )));
            }
            let pc_output_name = outs[0].clone();
            let field_output_name = outs[1].clone();

            let mut binding = s
                .create_binding()
                .map_err(|e| Error::Tokenizer(format!("schema_gather_io create_binding: {e}")))?;
            binding
                .bind_input("last_hidden_state", &encoder_out.hidden_states)
                .map_err(|e| {
                    Error::Tokenizer(format!("schema_gather_io bind last_hidden_state: {e}"))
                })?;
            binding.bind_input("schema_indices", &idx_t).map_err(|e| {
                Error::Tokenizer(format!("schema_gather_io bind schema_indices: {e}"))
            })?;
            binding
                .bind_output_to_device(&pc_output_name, device_mem)
                .map_err(|e| Error::Tokenizer(format!("schema_gather_io bind pc_emb: {e}")))?;
            binding
                .bind_output_to_device(&field_output_name, device_mem)
                .map_err(|e| Error::Tokenizer(format!("schema_gather_io bind field_embs: {e}")))?;
            let outputs = s
                .run_binding(&binding)
                .map_err(|e| Error::Tokenizer(format!("schema_gather_io run_binding: {e}")))?;

            // Drain both outputs by name. Order isn't guaranteed in
            // SessionOutputs iteration, so collect into HashMap then take.
            let mut by_name: std::collections::HashMap<String, DynValue> = outputs
                .into_iter()
                .map(|(n, v)| (n.to_string(), v))
                .collect();
            let pc_emb = by_name.remove(&pc_output_name).ok_or_else(|| {
                Error::Tokenizer(format!(
                    "schema_gather_io: pc_emb output '{pc_output_name}' missing from result"
                ))
            })?;
            let field_embs = by_name.remove(&field_output_name).ok_or_else(|| {
                Error::Tokenizer(format!(
                    "schema_gather_io: field_embs output '{field_output_name}' missing from result"
                ))
            })?;

            Ok(SchemaGatherOutputIo {
                pc_emb,
                field_embs,
                pc_output_name,
                field_output_name,
            })
        })
}

/// Run `count_pred_argmax` via IoBinding. Returns the predicted instance
/// count (already argmaxed in-graph; the i64 output is a scalar).
///
/// Output is bound to **CPU memory** (not device) — we need the value
/// on host immediately to drive control flow (e.g. early-return on
/// pred_count == 0). This is the only host-bound output in the
/// IoBinding chain; the cost (single i64 copy) is negligible.
pub(crate) fn run_count_pred_argmax_io(
    sessions: &Sessions,
    sg_out: &SchemaGatherOutputIo,
    cpu_out_mem: &MemoryInfo,
) -> Result<usize, Error> {
    sessions
        .count_pred_argmax
        .with_session(|s| -> Result<usize, Error> {
            let output_name = resolve_output_name(s, &["count"]);
            let mut binding = s
                .create_binding()
                .map_err(|e| Error::Tokenizer(format!("count_pred_io create_binding: {e}")))?;
            binding
                .bind_input("pc_emb", &sg_out.pc_emb)
                .map_err(|e| Error::Tokenizer(format!("count_pred_io bind pc_emb: {e}")))?;
            binding
                .bind_output_to_device(&output_name, cpu_out_mem)
                .map_err(|e| Error::Tokenizer(format!("count_pred_io bind_output: {e}")))?;
            let outputs = s
                .run_binding(&binding)
                .map_err(|e| Error::Tokenizer(format!("count_pred_io run_binding: {e}")))?;

            // Extract the scalar i64 from the bound output. The output is
            // CPU-resident so try_extract_tensor returns a host slice
            // without device-to-host copy.
            let count_val = outputs
                .into_iter()
                .find_map(|(name, val)| {
                    if name == output_name.as_str() {
                        Some(val)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| {
                    Error::Tokenizer(format!("count_pred_io: output '{output_name}' not present"))
                })?;
            let (_, cow) = count_val
                .try_extract_tensor::<i64>()
                .map_err(|e| Error::Tokenizer(format!("count_pred_io extract: {e}")))?;
            let val = cow.iter().next().copied().unwrap_or(0);
            Ok(val.max(0) as usize)
        })
}

/// Run count_lstm_fixed via IoBinding. Input: device-resident
/// `field_embs` from [`SchemaGatherOutputIo`]. Output: `struct_proj`
/// of shape `[MAX_COUNT, M, H]`, device-resident, chained into M10's
/// run_scorer_io as one of its two inputs.
pub(crate) fn run_count_lstm_fixed_io(
    sessions: &Sessions,
    sg_out: &SchemaGatherOutputIo,
    device_mem: &MemoryInfo,
) -> Result<CountLstmOutputIo, Error> {
    sessions
        .count_lstm_fixed
        .with_session(|s| -> Result<CountLstmOutputIo, Error> {
            let output_name = resolve_output_name(s, &["struct_proj"]);
            let mut binding = s
                .create_binding()
                .map_err(|e| Error::Tokenizer(format!("count_lstm_io create_binding: {e}")))?;
            binding
                .bind_input("field_embs", &sg_out.field_embs)
                .map_err(|e| Error::Tokenizer(format!("count_lstm_io bind field_embs: {e}")))?;
            binding
                .bind_output_to_device(&output_name, device_mem)
                .map_err(|e| Error::Tokenizer(format!("count_lstm_io bind_output: {e}")))?;
            let outputs = s
                .run_binding(&binding)
                .map_err(|e| Error::Tokenizer(format!("count_lstm_io run_binding: {e}")))?;
            let struct_proj = outputs
                .into_iter()
                .find_map(|(name, val)| {
                    if name == output_name.as_str() {
                        Some(val)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| {
                    Error::Tokenizer(format!("count_lstm_io: output '{output_name}' not present"))
                })?;
            Ok(CountLstmOutputIo {
                struct_proj,
                output_name,
            })
        })
}

/// Run scorer via IoBinding. Inputs:
/// - `span_embeddings`: device-resident from [`SpanRepOutputIo`].
/// - `struct_proj`: device-resident from [`CountLstmOutputIo`].
///
/// **Reads back to host** as [`super::pipeline::ScorerOutput`]
/// (`Array4<f32>` of shape `[MAX_COUNT, num_words, MAX_WIDTH, M]`) so
/// the existing decoder family (`decode_entities`,
/// `decode_entities_with_thresholds`, `decode_structure`) can consume
/// it without a separate IoBinding-mode decoder fork.
///
/// This is the only device→host copy in the IoBinding chain. The cost
/// (`O(MAX_COUNT × num_words × MAX_WIDTH × M)` f32s) is amortised over
/// the 7 chained device-resident inter-session outputs that IoBinding
/// eliminated.
pub(crate) fn run_scorer_io(
    sessions: &Sessions,
    sr_out: &SpanRepOutputIo,
    cl_out: &CountLstmOutputIo,
    device_mem: &MemoryInfo,
) -> Result<super::pipeline::ScorerOutput, Error> {
    let result: ndarray::ArrayD<f32> = sessions.scorer.with_session(|s| -> Result<_, Error> {
        let output_name = resolve_output_name(s, &["entity_scores", "scores"]);
        let mut binding = s
            .create_binding()
            .map_err(|e| Error::Tokenizer(format!("scorer_io create_binding: {e}")))?;
        binding
            .bind_input("span_embeddings", &sr_out.span_embs)
            .map_err(|e| Error::Tokenizer(format!("scorer_io bind span_embeddings: {e}")))?;
        binding
            .bind_input("struct_proj", &cl_out.struct_proj)
            .map_err(|e| Error::Tokenizer(format!("scorer_io bind struct_proj: {e}")))?;
        binding
            .bind_output_to_device(&output_name, device_mem)
            .map_err(|e| Error::Tokenizer(format!("scorer_io bind_output: {e}")))?;
        let outputs = s
            .run_binding(&binding)
            .map_err(|e| Error::Tokenizer(format!("scorer_io run_binding: {e}")))?;
        let scores_val = outputs
            .into_iter()
            .find_map(|(name, val)| {
                if name == output_name.as_str() {
                    Some(val)
                } else {
                    None
                }
            })
            .ok_or_else(|| {
                Error::Tokenizer(format!("scorer_io: output '{output_name}' not present"))
            })?;
        // Read back to host. CPU-allocated device_mem means this is
        // already host-side; for a future GPU device_mem this would
        // be the implicit device→host copy.
        let (shape, cow) = scores_val
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Tokenizer(format!("scorer_io extract: {e}")))?;
        let data: Vec<f32> = cow.to_vec();
        let shape_usize: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
        ndarray::ArrayD::from_shape_vec(shape_usize, data)
            .map_err(|e| Error::Tokenizer(format!("scorer_io reshape: {e}")))
    })?;

    let scores: ndarray::Array4<f32> = result
        .into_dimensionality::<ndarray::Ix4>()
        .map_err(|e| Error::Tokenizer(format!("scorer_io dim: {e}")))?;
    Ok(super::pipeline::ScorerOutput { scores })
}

/// Run the classifier session via IoBinding for the `classify` path.
///
/// The classifier requires a padded `[1, num_labels, MAX_WIDTH, H]`
/// tensor built from `field_embs` — that padding is host-side. So we
/// extract `field_embs` from the device-resident DynValue, build the
/// padded tensor on host, then run the classifier session via
/// IoBinding. Matches upstream gliner2-rs's pattern (lib_v2.rs:229+).
///
/// Output: `Vec<f32>` of length `num_labels`, softmax-normalised.
pub(crate) fn run_classifier_io(
    sessions: &Sessions,
    sg_out: &SchemaGatherOutputIo,
    device_mem: &MemoryInfo,
) -> Result<Vec<f32>, Error> {
    use super::pipeline::MAX_WIDTH;
    use ndarray::Array4;

    // Extract field_embs to host (small tensor: [num_labels, H]).
    let (fe_shape, fe_cow) = sg_out
        .field_embs
        .try_extract_tensor::<f32>()
        .map_err(|e| Error::Tokenizer(format!("classifier_io extract field_embs: {e}")))?;
    let fe_data: Vec<f32> = fe_cow.to_vec();
    if fe_shape.len() != 2 {
        return Err(Error::Tokenizer(format!(
            "classifier_io: field_embs expected 2D, got shape {:?}",
            fe_shape
        )));
    }
    let num_labels = fe_shape[0] as usize;
    let hidden_size = fe_shape[1] as usize;

    // Pad to [1, num_labels, MAX_WIDTH, hidden_size]: only position 0
    // along the MAX_WIDTH axis is filled with the field embedding.
    let mut padded: Array4<f32> = Array4::zeros((1, num_labels, MAX_WIDTH, hidden_size));
    for m in 0..num_labels {
        for d in 0..hidden_size {
            padded[[0, m, 0, d]] = fe_data[m * hidden_size + d];
        }
    }
    let pad_t = crate::backends::ort_compat::tensor_from_ndarray(padded)
        .map_err(|e| Error::Tokenizer(format!("classifier_io padded tensor: {e}")))?;

    let logits: ndarray::ArrayD<f32> =
        sessions.classifier.with_session(|s| -> Result<_, Error> {
            let output_name = resolve_output_name(s, &["cls_logits", "logits"]);
            let mut binding = s
                .create_binding()
                .map_err(|e| Error::Tokenizer(format!("classifier_io create_binding: {e}")))?;
            binding.bind_input("span_embeddings", &pad_t).map_err(|e| {
                Error::Tokenizer(format!("classifier_io bind span_embeddings: {e}"))
            })?;
            binding
                .bind_output_to_device(&output_name, device_mem)
                .map_err(|e| Error::Tokenizer(format!("classifier_io bind_output: {e}")))?;
            let outputs = s
                .run_binding(&binding)
                .map_err(|e| Error::Tokenizer(format!("classifier_io run_binding: {e}")))?;
            let logits_val = outputs
                .into_iter()
                .find_map(|(name, val)| {
                    if name == output_name.as_str() {
                        Some(val)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| {
                    Error::Tokenizer(format!("classifier_io: output '{output_name}' not present"))
                })?;
            let (shape, cow) = logits_val
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Tokenizer(format!("classifier_io extract: {e}")))?;
            let data: Vec<f32> = cow.to_vec();
            let shape_usize: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
            ndarray::ArrayD::from_shape_vec(shape_usize, data)
                .map_err(|e| Error::Tokenizer(format!("classifier_io reshape: {e}")))
        })?;

    // logits shape is [1, num_labels, MAX_WIDTH, 1]. Take position 0
    // along MAX_WIDTH and softmax over the labels axis.
    let mut exps = Vec::with_capacity(num_labels);
    let mut exp_sum = 0.0f32;
    for m in 0..num_labels {
        let l = logits[[0, m, 0, 0]];
        let e = l.exp();
        exp_sum += e;
        exps.push(e);
    }
    if exp_sum > 0.0 {
        for e in &mut exps {
            *e /= exp_sum;
        }
    }
    Ok(exps)
}

/// Top-level orchestrator: run the 8-session IoBinding pipeline (encoder
/// → token_gather → span_rep → schema_gather → count_pred_argmax →
/// count_lstm_fixed → scorer) for entity-/structure-decoding callers
/// (extract_ner, extract_with_label_descriptions,
/// extract_with_label_thresholds, extract_structure).
///
/// Returns `(ScorerOutput, pred_count)`. On `pred_count == 0`, returns
/// an empty ScorerOutput and skips the count_lstm + scorer steps —
/// matches the standard pipeline's early-return optimization.
pub(crate) fn run_pipeline_for_decoding(
    sessions: &Sessions,
    record: &ProcessedRecord,
    task: &crate::backends::gliner2_fastino::processor::TaskMapping,
) -> Result<(super::pipeline::ScorerOutput, usize), Error> {
    let device_mem = device_memory_info()
        .map_err(|e| Error::Tokenizer(format!("pipeline_io device_mem: {e}")))?;
    let cpu_out_mem = cpu_output_memory_info()
        .map_err(|e| Error::Tokenizer(format!("pipeline_io cpu_out_mem: {e}")))?;

    let enc = run_encoder_io(sessions, record, &device_mem)?;
    let tg = run_token_gather_io(sessions, &enc, record, &device_mem)?;
    let num_words = record.word_to_token_maps.len();
    let sr = run_span_rep_io(sessions, &tg, num_words, &device_mem)?;
    let sg = run_schema_gather_io(sessions, &enc, task, &device_mem)?;
    let pred_count = run_count_pred_argmax_io(sessions, &sg, &cpu_out_mem)?;
    if pred_count == 0 {
        // Empty 4D array — decoder loops over 0..pred_count so this is
        // a no-op for every consumer.
        return Ok((
            super::pipeline::ScorerOutput {
                scores: ndarray::Array4::zeros((0, 0, 0, 0)),
            },
            0,
        ));
    }
    let cl = run_count_lstm_fixed_io(sessions, &sg, &device_mem)?;
    let scorer_out = run_scorer_io(sessions, &sr, &cl, &device_mem)?;
    Ok((scorer_out, pred_count))
}

/// Top-level orchestrator: run the 4-session IoBinding pipeline
/// (encoder → schema_gather → count_pred_argmax → classifier) for the
/// `classify` path. Returns `Vec<f32>` of length `num_labels`,
/// softmax-normalised. Returns all-zeros on `pred_count == 0`.
pub(crate) fn run_classify_pipeline(
    sessions: &Sessions,
    record: &ProcessedRecord,
    task: &crate::backends::gliner2_fastino::processor::TaskMapping,
) -> Result<Vec<f32>, Error> {
    let device_mem = device_memory_info()
        .map_err(|e| Error::Tokenizer(format!("classify_pipeline_io device_mem: {e}")))?;
    let cpu_out_mem = cpu_output_memory_info()
        .map_err(|e| Error::Tokenizer(format!("classify_pipeline_io cpu_out_mem: {e}")))?;

    let enc = run_encoder_io(sessions, record, &device_mem)?;
    let sg = run_schema_gather_io(sessions, &enc, task, &device_mem)?;
    let pred_count = run_count_pred_argmax_io(sessions, &sg, &cpu_out_mem)?;
    if pred_count == 0 {
        return Ok(vec![0.0; task.labels.len()]);
    }
    run_classifier_io(sessions, &sg, &device_mem)
}

// =============================================================================
// Mode dispatch wrappers
//
// These are the two entry points used by the public extract_*  methods on
// GLiNER2Fastino. They route to the Standard chain (Phase 3) or the IoBinding
// chain (Phase 3.5) depending on the engine's ExecutionMode.
// =============================================================================

/// Dispatch wrapper for the 8-session decoder pipeline. Used by every
/// extract method that consumes a [`super::pipeline::ScorerOutput`]:
/// extract_ner, extract_with_label_descriptions,
/// extract_with_label_thresholds, extract_structure.
pub(crate) fn run_pipeline_dispatch(
    sessions: &Sessions,
    record: &ProcessedRecord,
    task: &crate::backends::gliner2_fastino::processor::TaskMapping,
    mode: super::ExecutionMode,
) -> Result<(super::pipeline::ScorerOutput, usize), Error> {
    match mode {
        super::ExecutionMode::Standard => {
            let enc = super::pipeline::run_encoder(sessions, record)?;
            let tg = super::pipeline::run_token_gather(sessions, &enc, record)?;
            let num_words = record.word_to_token_maps.len();
            let sr = super::pipeline::run_span_rep(sessions, &tg, num_words)?;
            let sg = super::pipeline::run_schema_gather(sessions, &enc, task)?;
            let pred_count = super::pipeline::run_count_pred_argmax(sessions, &sg)?;
            if pred_count == 0 {
                return Ok((
                    super::pipeline::ScorerOutput {
                        scores: ndarray::Array4::zeros((0, 0, 0, 0)),
                    },
                    0,
                ));
            }
            let cl = super::pipeline::run_count_lstm_fixed(sessions, &sg)?;
            let scorer_out = super::pipeline::run_scorer(sessions, &sr, &cl)?;
            Ok((scorer_out, pred_count))
        }
        super::ExecutionMode::IoBinding => run_pipeline_for_decoding(sessions, record, task),
    }
}

/// Dispatch wrapper for the 4-session classify pipeline. Used by the
/// `classify` public method.
pub(crate) fn run_classify_dispatch(
    sessions: &Sessions,
    record: &ProcessedRecord,
    task: &crate::backends::gliner2_fastino::processor::TaskMapping,
    mode: super::ExecutionMode,
) -> Result<Vec<f32>, Error> {
    match mode {
        super::ExecutionMode::Standard => {
            let enc = super::pipeline::run_encoder(sessions, record)?;
            let sg = super::pipeline::run_schema_gather(sessions, &enc, task)?;
            let pred_count = super::pipeline::run_count_pred_argmax(sessions, &sg)?;
            if pred_count == 0 {
                return Ok(vec![0.0; task.labels.len()]);
            }
            super::pipeline::run_classifier(sessions, &sg)
        }
        super::ExecutionMode::IoBinding => run_classify_pipeline(sessions, record, task),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_memory_info_constructs_on_cpu() {
        // Phase 3.5 M5: smoke test — CPU-only MemoryInfo creation must
        // succeed on every host. Ensures the runtime API works in our
        // ort 2.0.0-rc.12 pin.
        let mi = device_memory_info().expect("CPU MemoryInfo creation should succeed");
        let _ = format!("{:?}", mi);
    }

    #[test]
    fn cpu_output_memory_info_constructs() {
        // Phase 3.5 M5: same for the CPUOutput variant used by
        // count_pred_argmax (M9).
        let mi = cpu_output_memory_info().expect("CPUOutput MemoryInfo creation");
        let _ = format!("{:?}", mi);
    }
}
