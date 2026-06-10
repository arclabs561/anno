//! Standard-mode 8-session inference pipeline for `gliner2_fastino`.
#![allow(missing_docs)] // implementation internals; public API is on GLiNER2Fastino in mod.rs
//!
//! Adapted from SemplificaAI/gliner2-rs (Apache-2.0):
//! https://github.com/SemplificaAI/gliner2-rs/blob/main/rust_component/src/lib_v2.rs
//! Specifically: `Gliner2EngineV2::extract_standard` (lines ~660-897).
//! Original: Copyright 2026 Dario Finardi, Semplifica s.r.l.
//!
//! Phase 3 standard mode (this module) does NOT implement IOBinding.
//! The IOBinding-mode pipeline (lib_v2.rs:285-660) keeps tensors in
//! a single ort allocator across session boundaries for 2-3× speedup;
//! that's a Phase 3.5 follow-up.

use crate::backends::gliner2_fastino::errors::Error;
use crate::backends::gliner2_fastino::processor::ProcessedRecord;
use crate::backends::gliner2_fastino::sessions::Sessions;
use ndarray::{Array2, Array3};

/// Maximum span width baked into the v2 export. Spans wider than this
/// can't be scored. Hardcoded in `count_lstm_fixed` and `scorer` ONNX
/// graphs.
pub const MAX_WIDTH: usize = 8;

/// Maximum predicted instance count baked into the v2 export. Used by
/// the scorer's first dimension (struct_proj is `[MAX_COUNT, M, H]`).
pub const MAX_COUNT: usize = 20;

/// Output of the encoder step. Owned f32 ndarray of shape `[1, L, H]`.
pub(crate) struct EncoderOutput {
    pub hidden_states: Array3<f32>,
}

/// Run the encoder graph. Tries output names in priority order
/// (`hidden_states`, `last_hidden_state`, `output`) — different fastino
/// exports use different names.
pub(crate) fn run_encoder(
    sessions: &Sessions,
    record: &ProcessedRecord,
) -> Result<EncoderOutput, Error> {
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

    let hs: ndarray::ArrayD<f32> = sessions.encoder.with_session(|s| -> Result<_, Error> {
        let outputs = s
            .run(ort::inputs![
                "input_ids"      => input_ids_t.into_dyn(),
                "attention_mask" => attn_mask_t.into_dyn(),
            ])
            .map_err(|e| Error::Tokenizer(format!("encoder run: {e}")))?;

        for name in ["hidden_states", "last_hidden_state", "output"] {
            if let Some(v) = outputs.get(name) {
                let (shape, cow) = v
                    .try_extract_tensor::<f32>()
                    .map_err(|e| Error::Tokenizer(format!("encoder extract: {e}")))?;
                let data: Vec<f32> = cow.to_vec();
                let shape_usize: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
                return ndarray::ArrayD::from_shape_vec(shape_usize, data)
                    .map_err(|e| Error::Tokenizer(format!("encoder array reshape: {e}")));
            }
        }
        // Fallback: take the first output.
        let first = outputs
            .values()
            .next()
            .ok_or_else(|| Error::Tokenizer("encoder: no outputs".into()))?;
        let (shape, cow) = first
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Tokenizer(format!("encoder extract first: {e}")))?;
        let data: Vec<f32> = cow.to_vec();
        let shape_usize: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
        ndarray::ArrayD::from_shape_vec(shape_usize, data)
            .map_err(|e| Error::Tokenizer(format!("encoder array reshape: {e}")))
    })?;

    // hs is dynamic; convert to fixed [1, L, H] Array3.
    let shape = hs.shape().to_vec();
    if shape.len() != 3 || shape[0] != 1 {
        return Err(Error::Tokenizer(format!(
            "encoder output shape {:?}: expected [1, L, H]",
            shape
        )));
    }
    let hidden_states: Array3<f32> = hs
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|e| Error::Tokenizer(format!("encoder dim convert: {e}")))?;
    Ok(EncoderOutput { hidden_states })
}

/// Output of token_gather: word-level embeddings extracted from the
/// encoder's hidden states using `word_to_token_maps`.
pub(crate) struct TokenGatherOutput {
    /// Shape: `[1, num_words, H]`
    pub text_embs: Array3<f32>,
}

pub(crate) fn run_token_gather(
    sessions: &Sessions,
    encoder_out: &EncoderOutput,
    record: &ProcessedRecord,
) -> Result<TokenGatherOutput, Error> {
    use ndarray::Array1;

    let num_words = record.word_to_token_maps.len();
    if num_words == 0 {
        return Err(Error::Tokenizer("token_gather: 0 words in record".into()));
    }
    let word_starts: Vec<i64> = record
        .word_to_token_maps
        .iter()
        .map(|&(start, _)| start as i64)
        .collect();
    let word_idx_arr: Array1<i64> = Array1::from_vec(word_starts);

    let hs_t = crate::backends::ort_compat::tensor_from_ndarray(encoder_out.hidden_states.clone())
        .map_err(|e| Error::Tokenizer(format!("token_gather hs tensor: {e}")))?;
    let word_idx_t = crate::backends::ort_compat::tensor_from_ndarray(word_idx_arr)
        .map_err(|e| Error::Tokenizer(format!("token_gather idx tensor: {e}")))?;

    let result: ndarray::ArrayD<f32> =
        sessions
            .token_gather
            .with_session(|s| -> Result<_, Error> {
                let outputs = s
                    .run(ort::inputs![
                        "last_hidden_state" => hs_t.into_dyn(),
                        "word_indices"      => word_idx_t.into_dyn(),
                    ])
                    .map_err(|e| Error::Tokenizer(format!("token_gather run: {e}")))?;
                let v = outputs
                    .values()
                    .next()
                    .ok_or_else(|| Error::Tokenizer("token_gather: no outputs".into()))?;
                let (shape, cow) = v
                    .try_extract_tensor::<f32>()
                    .map_err(|e| Error::Tokenizer(format!("token_gather extract: {e}")))?;
                let data: Vec<f32> = cow.to_vec();
                let shape_usize: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
                ndarray::ArrayD::from_shape_vec(shape_usize, data)
                    .map_err(|e| Error::Tokenizer(format!("token_gather array reshape: {e}")))
            })?;

    let text_embs: Array3<f32> = result
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|e| Error::Tokenizer(format!("token_gather dim: {e}")))?;
    Ok(TokenGatherOutput { text_embs })
}

/// Output of span_rep: span-level embeddings.
///
/// Real shape per the SemplificaAI export: `[1, num_words, MAX_WIDTH, H]`.
/// (4D, with the max-span-width dimension explicit.)
pub(crate) struct SpanRepOutput {
    pub span_embs: ndarray::Array4<f32>,
}

/// Build the span-index tensor used by span_rep.
///
/// For each (start_word, width_idx) pair where `width_idx` ∈ 0..MAX_WIDTH,
/// emit (start, start + width_idx). If end exceeds `num_words`, emit
/// `[0, 0]` as zero-padding (matches upstream's behavior — those spans
/// are masked out by the model).
pub(crate) fn build_span_idx(num_words: usize) -> Array3<i64> {
    let num_spans = num_words * MAX_WIDTH;
    let mut data = Vec::with_capacity(num_spans * 2);
    for start in 0..num_words {
        for width in 0..MAX_WIDTH {
            let end = start + width;
            if end >= num_words {
                data.extend_from_slice(&[0_i64, 0_i64]);
            } else {
                data.push(start as i64);
                data.push(end as i64);
            }
        }
    }
    Array3::from_shape_vec((1, num_spans, 2), data)
        .expect("span_idx shape consistent by construction")
}

pub(crate) fn run_span_rep(
    sessions: &Sessions,
    tg_out: &TokenGatherOutput,
    num_words: usize,
) -> Result<SpanRepOutput, Error> {
    let span_idx = build_span_idx(num_words);

    let hs_t = crate::backends::ort_compat::tensor_from_ndarray(tg_out.text_embs.clone())
        .map_err(|e| Error::Tokenizer(format!("span_rep hs tensor: {e}")))?;
    let idx_t = crate::backends::ort_compat::tensor_from_ndarray(span_idx)
        .map_err(|e| Error::Tokenizer(format!("span_rep idx tensor: {e}")))?;

    let result: ndarray::ArrayD<f32> = sessions.span_rep.with_session(|s| -> Result<_, Error> {
        let outputs = s
            .run(ort::inputs![
                "hidden_states" => hs_t.into_dyn(),
                "span_idx"      => idx_t.into_dyn(),
            ])
            .map_err(|e| Error::Tokenizer(format!("span_rep run: {e}")))?;
        let v = outputs
            .values()
            .next()
            .ok_or_else(|| Error::Tokenizer("span_rep: no outputs".into()))?;
        let (shape, cow) = v
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Tokenizer(format!("span_rep extract: {e}")))?;
        let data: Vec<f32> = cow.to_vec();
        let shape_usize: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
        ndarray::ArrayD::from_shape_vec(shape_usize, data)
            .map_err(|e| Error::Tokenizer(format!("span_rep array reshape: {e}")))
    })?;

    let span_embs: ndarray::Array4<f32> = result
        .into_dimensionality::<ndarray::Ix4>()
        .map_err(|e| Error::Tokenizer(format!("span_rep dim: {e}")))?;
    Ok(SpanRepOutput { span_embs })
}

/// Output of schema_gather: per-task pc_emb + field_embs.
pub(crate) struct SchemaGatherOutput {
    /// Shape: `[1, H]` — the [P]-token embedding (prompt context).
    pub pc_emb: Array2<f32>,
    /// Shape: `[M, H]` where M = number of fields/labels for this task.
    pub field_embs: Array2<f32>,
}

pub(crate) fn run_schema_gather(
    sessions: &Sessions,
    encoder_out: &EncoderOutput,
    task: &crate::backends::gliner2_fastino::processor::TaskMapping,
) -> Result<SchemaGatherOutput, Error> {
    use ndarray::Array1;

    let mut indices: Vec<i64> = Vec::with_capacity(1 + task.field_tok_indices.len());
    indices.push(task.prompt_tok_idx as i64);
    indices.extend(task.field_tok_indices.iter().map(|&i| i as i64));
    let idx_arr: Array1<i64> = Array1::from_vec(indices);

    let hs_t = crate::backends::ort_compat::tensor_from_ndarray(encoder_out.hidden_states.clone())
        .map_err(|e| Error::Tokenizer(format!("schema_gather hs tensor: {e}")))?;
    let idx_t = crate::backends::ort_compat::tensor_from_ndarray(idx_arr)
        .map_err(|e| Error::Tokenizer(format!("schema_gather idx tensor: {e}")))?;

    type SchemaResult = (ndarray::ArrayD<f32>, ndarray::ArrayD<f32>);
    let (pc, fields): SchemaResult =
        sessions
            .schema_gather
            .with_session(|s| -> Result<_, Error> {
                let outputs = s
                    .run(ort::inputs![
                        "last_hidden_state" => hs_t.into_dyn(),
                        "schema_indices"    => idx_t.into_dyn(),
                    ])
                    .map_err(|e| Error::Tokenizer(format!("schema_gather run: {e}")))?;
                let mut iter = outputs.values();
                let pc_v = iter
                    .next()
                    .ok_or_else(|| Error::Tokenizer("schema_gather: missing pc_emb".into()))?;
                let fields_v = iter
                    .next()
                    .ok_or_else(|| Error::Tokenizer("schema_gather: missing field_embs".into()))?;
                let (pc_shape, pc_cow) = pc_v
                    .try_extract_tensor::<f32>()
                    .map_err(|e| Error::Tokenizer(format!("schema_gather pc extract: {e}")))?;
                let (fields_shape, fields_cow) = fields_v
                    .try_extract_tensor::<f32>()
                    .map_err(|e| Error::Tokenizer(format!("schema_gather fields extract: {e}")))?;
                let pc_data: Vec<f32> = pc_cow.to_vec();
                let pc_shape_usize: Vec<usize> = pc_shape.iter().map(|&s| s as usize).collect();
                let pc_arr = ndarray::ArrayD::from_shape_vec(pc_shape_usize, pc_data)
                    .map_err(|e| Error::Tokenizer(format!("schema_gather pc reshape: {e}")))?;
                let fields_data: Vec<f32> = fields_cow.to_vec();
                let fields_shape_usize: Vec<usize> =
                    fields_shape.iter().map(|&s| s as usize).collect();
                let fields_arr = ndarray::ArrayD::from_shape_vec(fields_shape_usize, fields_data)
                    .map_err(|e| {
                    Error::Tokenizer(format!("schema_gather fields reshape: {e}"))
                })?;
                Ok((pc_arr, fields_arr))
            })?;

    let pc_emb: Array2<f32> = pc
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| Error::Tokenizer(format!("schema_gather pc dim: {e}")))?;
    let field_embs: Array2<f32> = fields
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| Error::Tokenizer(format!("schema_gather fields dim: {e}")))?;
    Ok(SchemaGatherOutput { pc_emb, field_embs })
}

/// Run `count_pred_argmax`. Returns the predicted instance count
/// (already argmaxed in-graph; the i64 output is a scalar).
pub(crate) fn run_count_pred_argmax(
    sessions: &Sessions,
    sg_out: &SchemaGatherOutput,
) -> Result<usize, Error> {
    let pc_t = crate::backends::ort_compat::tensor_from_ndarray(sg_out.pc_emb.clone())
        .map_err(|e| Error::Tokenizer(format!("count_pred pc tensor: {e}")))?;

    let count: ndarray::ArrayD<i64> =
        sessions
            .count_pred_argmax
            .with_session(|s| -> Result<_, Error> {
                let outputs = s
                    .run(ort::inputs![
                        "pc_emb" => pc_t.into_dyn(),
                    ])
                    .map_err(|e| Error::Tokenizer(format!("count_pred run: {e}")))?;
                let v = outputs
                    .values()
                    .next()
                    .ok_or_else(|| Error::Tokenizer("count_pred_argmax: no outputs".into()))?;
                let (shape, cow) = v
                    .try_extract_tensor::<i64>()
                    .map_err(|e| Error::Tokenizer(format!("count_pred extract: {e}")))?;
                let data: Vec<i64> = cow.to_vec();
                let shape_usize: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
                ndarray::ArrayD::from_shape_vec(shape_usize, data)
                    .map_err(|e| Error::Tokenizer(format!("count_pred reshape: {e}")))
            })?;

    let val = count.iter().next().copied().unwrap_or(0);
    Ok(val.max(0) as usize)
}

/// Output of count_lstm_fixed: struct projection used by scorer.
/// Shape: `[MAX_COUNT, M, H]`.
pub(crate) struct CountLstmOutput {
    pub struct_proj: Array3<f32>,
}

pub(crate) fn run_count_lstm_fixed(
    sessions: &Sessions,
    sg_out: &SchemaGatherOutput,
) -> Result<CountLstmOutput, Error> {
    let fields_t = crate::backends::ort_compat::tensor_from_ndarray(sg_out.field_embs.clone())
        .map_err(|e| Error::Tokenizer(format!("count_lstm tensor: {e}")))?;

    let proj: ndarray::ArrayD<f32> =
        sessions
            .count_lstm_fixed
            .with_session(|s| -> Result<_, Error> {
                let outputs = s
                    .run(ort::inputs![
                        "field_embs" => fields_t.into_dyn(),
                    ])
                    .map_err(|e| Error::Tokenizer(format!("count_lstm run: {e}")))?;
                let v = outputs
                    .values()
                    .next()
                    .ok_or_else(|| Error::Tokenizer("count_lstm_fixed: no outputs".into()))?;
                let (shape, cow) = v
                    .try_extract_tensor::<f32>()
                    .map_err(|e| Error::Tokenizer(format!("count_lstm extract: {e}")))?;
                let data: Vec<f32> = cow.to_vec();
                let shape_usize: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
                ndarray::ArrayD::from_shape_vec(shape_usize, data)
                    .map_err(|e| Error::Tokenizer(format!("count_lstm reshape: {e}")))
            })?;

    let struct_proj: Array3<f32> = proj
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|e| Error::Tokenizer(format!("count_lstm dim: {e}")))?;
    Ok(CountLstmOutput { struct_proj })
}

/// Output of scorer: per-instance per-span per-label entity scores.
/// Shape: `[MAX_COUNT, num_words, MAX_WIDTH, M]`.
/// Already-sigmoided per upstream (`extract_standard` line ~825 comment:
/// "Scorer — restituisce probabilità sigmoid già calcolate").
pub(crate) struct ScorerOutput {
    pub scores: ndarray::Array4<f32>,
}

pub(crate) fn run_scorer(
    sessions: &Sessions,
    sr_out: &SpanRepOutput,
    cl_out: &CountLstmOutput,
) -> Result<ScorerOutput, Error> {
    let span_t = crate::backends::ort_compat::tensor_from_ndarray(sr_out.span_embs.clone())
        .map_err(|e| Error::Tokenizer(format!("scorer span tensor: {e}")))?;
    let proj_t = crate::backends::ort_compat::tensor_from_ndarray(cl_out.struct_proj.clone())
        .map_err(|e| Error::Tokenizer(format!("scorer proj tensor: {e}")))?;

    let result: ndarray::ArrayD<f32> = sessions.scorer.with_session(|s| -> Result<_, Error> {
        let outputs = s
            .run(ort::inputs![
                "span_embeddings" => span_t.into_dyn(),
                "struct_proj"     => proj_t.into_dyn(),
            ])
            .map_err(|e| Error::Tokenizer(format!("scorer run: {e}")))?;
        let v = outputs
            .values()
            .next()
            .ok_or_else(|| Error::Tokenizer("scorer: no outputs".into()))?;
        let (shape, cow) = v
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Tokenizer(format!("scorer extract: {e}")))?;
        let data: Vec<f32> = cow.to_vec();
        let shape_usize: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
        ndarray::ArrayD::from_shape_vec(shape_usize, data)
            .map_err(|e| Error::Tokenizer(format!("scorer reshape: {e}")))
    })?;

    let scores: ndarray::Array4<f32> = result
        .into_dimensionality::<ndarray::Ix4>()
        .map_err(|e| Error::Tokenizer(format!("scorer dim: {e}")))?;
    Ok(ScorerOutput { scores })
}

/// Decode the scorer's [MAX_COUNT, num_words, MAX_WIDTH, M] tensor to
/// `Vec<Entity>` (with character offsets in the original text), apply
/// per-label thresholds, then NMS.
///
/// Phase 1.5: a label not present in `label_thresholds` is dropped
/// entirely (its threshold is treated as `+∞`). This allows callers to
/// score for many labels but only keep a subset, without round-tripping
/// the whole prompt+inference.
pub(crate) fn decode_entities_with_thresholds(
    text: &str,
    record: &ProcessedRecord,
    task: &crate::backends::gliner2_fastino::processor::TaskMapping,
    scorer_out: &ScorerOutput,
    pred_count: usize,
    label_thresholds: &[(&str, f32)],
    flat_ner: bool,
) -> Vec<crate::Entity> {
    // Build a fast lookup keyed by label-index in `task.labels`. Any
    // label not in the input list gets `+∞`, dropping every candidate.
    let thresholds: Vec<f32> = task
        .labels
        .iter()
        .map(|label| {
            label_thresholds
                .iter()
                .find(|(l, _)| *l == label.as_str())
                .map(|(_, t)| *t)
                .unwrap_or(f32::INFINITY)
        })
        .collect();

    let num_words = record.word_to_char_maps.len();
    let num_labels = task.labels.len();
    let scores = &scorer_out.scores;

    let mut candidates: Vec<crate::Entity> = Vec::new();
    for c_idx in 0..pred_count.min(MAX_COUNT) {
        for start in 0..num_words {
            for width_idx in 0..MAX_WIDTH {
                let end_word = (start + width_idx + 1).min(num_words);
                for m in 0..num_labels {
                    let prob = scores[[c_idx, start, width_idx, m]];
                    if prob <= thresholds[m] {
                        continue;
                    }
                    let (byte_start, _) = record.word_to_char_maps[start];
                    let (_, byte_end) = record.word_to_char_maps[end_word - 1];
                    if byte_end > text.len() || byte_start > byte_end {
                        continue;
                    }
                    let surface = text[byte_start..byte_end].trim();
                    if surface.is_empty() {
                        continue;
                    }
                    let etype = crate::schema::map_to_canonical(&task.labels[m], None);
                    // Convert byte offsets to char offsets (anno convention).
                    let (cs, ce) = crate::offset::bytes_to_chars(text, byte_start, byte_end);
                    candidates.push(crate::Entity::new(surface, etype, cs, ce, prob));
                }
            }
        }
    }
    super::nms::greedy_nms(candidates, flat_ner)
}

/// Decode the scorer's [MAX_COUNT, num_words, MAX_WIDTH, M] tensor with
/// a single global threshold applied to every label. Thin wrapper over
/// [`decode_entities_with_thresholds`] (DRY).
pub(crate) fn decode_entities(
    text: &str,
    record: &ProcessedRecord,
    task: &crate::backends::gliner2_fastino::processor::TaskMapping,
    scorer_out: &ScorerOutput,
    pred_count: usize,
    threshold: f32,
    flat_ner: bool,
) -> Vec<crate::Entity> {
    let label_thresholds: Vec<(&str, f32)> = task
        .labels
        .iter()
        .map(|l| (l.as_str(), threshold))
        .collect();
    decode_entities_with_thresholds(
        text,
        record,
        task,
        scorer_out,
        pred_count,
        &label_thresholds,
        flat_ner,
    )
}

/// Decode the scorer's `[MAX_COUNT, num_words, MAX_WIDTH, num_fields]`
/// tensor as a structure-extraction result. Walks the `MAX_COUNT` axis
/// as the instance axis: for each predicted instance `c_idx ∈ 0..pred_count`,
/// pick the best span for each field and assemble one
/// [`crate::backends::gliner2_fastino::schema::ExtractedStructure`].
///
/// Phase 2: ships `FieldType::String` only. `List` / `Choice` field types
/// receive the same single-best-span treatment as `String` — see the
/// `// TODO(Phase 2.5)` markers below for where they'd specialize.
///
/// Threshold semantics: a (instance, field) candidate is dropped only if
/// its best score is `<= threshold`. An instance with all fields dropped
/// becomes an empty `fields` map; the caller decides whether to keep
/// such instances (this fn keeps them — see `extract_structure` for the
/// emptiness filter).
pub(crate) fn decode_structure(
    text: &str,
    record: &ProcessedRecord,
    task: &crate::backends::gliner2_fastino::processor::TaskMapping,
    scorer_out: &ScorerOutput,
    pred_count: usize,
    threshold: f32,
    fields: &[(String, crate::backends::gliner2_fastino::schema::FieldType)],
) -> Vec<crate::backends::gliner2_fastino::schema::ExtractedStructure> {
    use crate::backends::gliner2_fastino::schema::{ExtractedStructure, StructureValue};
    use std::collections::HashMap;

    let num_words = record.word_to_char_maps.len();
    let num_fields = task.labels.len();
    debug_assert_eq!(
        num_fields,
        fields.len(),
        "decode_structure: task.labels.len() = {} but fields.len() = {}",
        num_fields,
        fields.len(),
    );
    let scores = &scorer_out.scores;

    let mut out: Vec<ExtractedStructure> = Vec::with_capacity(pred_count);
    for c_idx in 0..pred_count.min(MAX_COUNT) {
        let mut field_values: HashMap<String, StructureValue> = HashMap::new();
        for (m, (field_name, _ftype)) in fields.iter().enumerate().take(num_fields) {
            // Find the best (start, width_idx) for this (instance, field).
            let mut best: Option<(f32, usize, usize)> = None;
            for start in 0..num_words {
                for width_idx in 0..MAX_WIDTH {
                    let prob = scores[[c_idx, start, width_idx, m]];
                    if prob <= threshold {
                        continue;
                    }
                    let end_word = (start + width_idx + 1).min(num_words);
                    let (byte_start, _) = record.word_to_char_maps[start];
                    let (_, byte_end) = record.word_to_char_maps[end_word - 1];
                    if byte_end > text.len() || byte_start > byte_end {
                        continue;
                    }
                    let surface = text[byte_start..byte_end].trim();
                    if surface.is_empty() {
                        continue;
                    }
                    match best {
                        Some((b, _, _)) if b >= prob => {}
                        _ => best = Some((prob, start, width_idx)),
                    }
                }
            }
            if let Some((_prob, start, width_idx)) = best {
                let end_word = (start + width_idx + 1).min(num_words);
                let (byte_start, _) = record.word_to_char_maps[start];
                let (_, byte_end) = record.word_to_char_maps[end_word - 1];
                let surface = text[byte_start..byte_end].trim().to_string();
                // Phase 2: every field, regardless of FieldType, becomes
                // StructureValue::Single. TODO(Phase 2.5): branch on
                // _ftype here for List (collect top-K) / Choice (snap
                // surface to nearest choice via edit distance).
                field_values.insert(field_name.clone(), StructureValue::Single(surface));
            }
        }
        out.push(ExtractedStructure {
            structure_type: task.task_name.clone(),
            fields: field_values,
        });
    }
    out
}

/// Run the classifier head on a single task's field_embs.
/// Returns label scores (softmax probabilities, sum to 1).
///
/// Internal mechanics: pad `field_embs` to `[1, num_labels, MAX_WIDTH,
/// hidden_size]` with first-position-only set, convert to fp16, run,
/// softmax over the label axis.
pub(crate) fn run_classifier(
    sessions: &Sessions,
    sg_out: &SchemaGatherOutput,
) -> Result<Vec<f32>, Error> {
    use ndarray::Array4;

    let num_labels = sg_out.field_embs.shape()[0];
    let hidden_size = sg_out.field_embs.shape()[1];

    // Pad to [1, num_labels, MAX_WIDTH, hidden_size] in fp16,
    // then convert to f32 for ort.
    let mut padded_fp16: Array4<half::f16> = Array4::from_elem(
        (1, num_labels, MAX_WIDTH, hidden_size),
        half::f16::from_f32(0.0),
    );
    for m in 0..num_labels {
        for d in 0..hidden_size {
            padded_fp16[[0, m, 0, d]] = half::f16::from_f32(sg_out.field_embs[[m, d]]);
        }
    }
    // Convert fp16 padding to f32 for ort tensor compatibility.
    let padded: Array4<f32> = padded_fp16.mapv(|v| v.to_f32());

    let pad_t = crate::backends::ort_compat::tensor_from_ndarray(padded)
        .map_err(|e| Error::Tokenizer(format!("classifier tensor: {e}")))?;

    let logits: ndarray::ArrayD<f32> =
        sessions.classifier.with_session(|s| -> Result<_, Error> {
            let outputs = s
                .run(ort::inputs![
                    "span_embeddings" => pad_t.into_dyn(),
                ])
                .map_err(|e| Error::Tokenizer(format!("classifier run: {e}")))?;
            let v = outputs
                .values()
                .next()
                .ok_or_else(|| Error::Tokenizer("classifier: no outputs".into()))?;
            let (shape, cow) = v
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Tokenizer(format!("classifier extract: {e}")))?;
            let data: Vec<f32> = cow.to_vec();
            let shape_usize: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
            ndarray::ArrayD::from_shape_vec(shape_usize, data)
                .map_err(|e| Error::Tokenizer(format!("classifier reshape: {e}")))
        })?;

    // logits shape is [1, num_labels, MAX_WIDTH, 1]. Take position 0.
    let mut exps = Vec::with_capacity(num_labels);
    let mut exp_sum = 0.0f32;
    for m in 0..num_labels {
        let l = logits[[0, m, 0, 0]];
        let e = l.exp();
        exp_sum += e;
        exps.push(e);
    }
    Ok(exps.into_iter().map(|e| e / exp_sum.max(1e-12)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_span_idx_basic_shape() {
        let arr = build_span_idx(3);
        assert_eq!(arr.shape(), &[1, 3 * MAX_WIDTH, 2]);
    }

    #[test]
    fn decode_entities_respects_per_label_thresholds() {
        use crate::backends::gliner2_fastino::processor::{ProcessedRecord, TaskMapping};
        use ndarray::Array4;

        // Build a synthetic ProcessedRecord with 2 words ("Acme Corp").
        let record = ProcessedRecord {
            input_ids: vec![],
            attention_mask: vec![],
            tasks: vec![],
            text_start: 0,
            text_end: 0,
            word_to_token_maps: vec![(0, 1), (1, 2)],
            word_to_char_maps: vec![(0, 4), (5, 9)],
        };
        let task = TaskMapping {
            task_name: "entities".to_string(),
            task_type: "entities".to_string(),
            labels: vec!["organization".into(), "location".into()],
            prompt_tok_idx: 0,
            field_tok_indices: vec![0, 0],
        };
        // Scorer output: [MAX_COUNT=20, num_words=2, MAX_WIDTH=8, num_labels=2].
        // Set scores so:
        //   span (0,0) label=org      score=0.9
        //   span (1,1) label=location score=0.6
        let mut scores = Array4::<f32>::zeros((MAX_COUNT, 2, MAX_WIDTH, 2));
        scores[[0, 0, 0, 0]] = 0.9; // org at word 0
        scores[[0, 1, 0, 1]] = 0.6; // location at word 1
        let scorer_out = ScorerOutput { scores };

        let text = "Acme Corp";

        // Both labels at threshold 0.5: both candidates pass.
        let ents = decode_entities_with_thresholds(
            text,
            &record,
            &task,
            &scorer_out,
            1,
            &[("organization", 0.5), ("location", 0.5)],
            false,
        );
        assert_eq!(ents.len(), 2, "expected 2 entities, got {ents:#?}");

        // Tighten location threshold above 0.6: only org passes.
        let ents = decode_entities_with_thresholds(
            text,
            &record,
            &task,
            &scorer_out,
            1,
            &[("organization", 0.5), ("location", 0.7)],
            false,
        );
        assert_eq!(ents.len(), 1, "expected 1 entity (only org), got {ents:#?}");
        assert!(
            matches!(ents[0].entity_type, crate::EntityType::Organization),
            "expected Organization, got {:?}",
            ents[0].entity_type
        );

        // Omit a label entirely from the threshold list: it's dropped.
        let ents = decode_entities_with_thresholds(
            text,
            &record,
            &task,
            &scorer_out,
            1,
            &[("organization", 0.5)],
            false,
        );
        assert_eq!(
            ents.len(),
            1,
            "expected 1 entity (location dropped via missing threshold), got {ents:#?}",
        );

        // Sanity: the original decode_entities (single threshold) still works
        // and matches the all-labels-same-threshold case.
        let ents_global = decode_entities(text, &record, &task, &scorer_out, 1, 0.5, false);
        assert_eq!(ents_global.len(), 2);
    }

    #[test]
    fn build_span_idx_zero_pads_overflow() {
        // 2 words, MAX_WIDTH=8.
        let arr = build_span_idx(2);
        // First row is start=0 width=0 → [0,0].
        assert_eq!(arr[[0, 0, 0]], 0);
        assert_eq!(arr[[0, 0, 1]], 0);
        // Second row is start=0 width=1 → [0,1].
        assert_eq!(arr[[0, 1, 0]], 0);
        assert_eq!(arr[[0, 1, 1]], 1);
        // 9th row (index MAX_WIDTH = 8) is start=1 width=0 → [1,1].
        assert_eq!(arr[[0, MAX_WIDTH, 0]], 1);
        assert_eq!(arr[[0, MAX_WIDTH, 1]], 1);
        // 10th row is start=1 width=1 → would be (1,2) but 2 >= num_words=2,
        // so padded to [0,0].
        assert_eq!(arr[[0, MAX_WIDTH + 1, 0]], 0);
        assert_eq!(arr[[0, MAX_WIDTH + 1, 1]], 0);
    }

    #[test]
    fn decode_structure_single_instance_picks_best_span_per_field() {
        use crate::backends::gliner2_fastino::processor::{ProcessedRecord, TaskMapping};
        use crate::backends::gliner2_fastino::schema::{FieldType, StructureValue};
        use ndarray::Array4;

        // 3 words: "Acme Corp Paris" (indices 0, 1, 2 with byte ranges).
        let record = ProcessedRecord {
            input_ids: vec![],
            attention_mask: vec![],
            tasks: vec![],
            text_start: 0,
            text_end: 0,
            word_to_token_maps: vec![(0, 1), (1, 2), (2, 3)],
            word_to_char_maps: vec![(0, 4), (5, 9), (10, 15)],
        };
        let task = TaskMapping {
            task_name: "company_loc".to_string(),
            task_type: "structures".to_string(),
            labels: vec!["vendor".into(), "city".into()],
            prompt_tok_idx: 0,
            field_tok_indices: vec![0, 0],
        };
        // Scorer: [MAX_COUNT, num_words=3, MAX_WIDTH, num_fields=2].
        // Instance 0:
        //   field 0 (vendor) best at start=0, width=1 ("Acme Corp"): 0.9
        //   field 1 (city)   best at start=2, width=0 ("Paris"):     0.85
        let mut scores = Array4::<f32>::zeros((MAX_COUNT, 3, MAX_WIDTH, 2));
        scores[[0, 0, 1, 0]] = 0.9;
        scores[[0, 2, 0, 1]] = 0.85;
        let scorer_out = ScorerOutput { scores };

        let fields = vec![
            ("vendor".to_string(), FieldType::String),
            ("city".to_string(), FieldType::String),
        ];
        let result = decode_structure(
            "Acme Corp Paris",
            &record,
            &task,
            &scorer_out,
            /* pred_count = */ 1,
            /* threshold  = */ 0.5,
            &fields,
        );

        assert_eq!(result.len(), 1, "expected 1 instance, got {}", result.len());
        let inst = &result[0];
        assert_eq!(inst.structure_type, "company_loc");
        match inst.fields.get("vendor") {
            Some(StructureValue::Single(s)) => assert_eq!(s, "Acme Corp"),
            other => panic!("expected vendor=Single(\"Acme Corp\"), got {other:?}"),
        }
        match inst.fields.get("city") {
            Some(StructureValue::Single(s)) => assert_eq!(s, "Paris"),
            other => panic!("expected city=Single(\"Paris\"), got {other:?}"),
        }
    }

    #[test]
    fn decode_structure_zero_pred_count_returns_empty() {
        use crate::backends::gliner2_fastino::processor::{ProcessedRecord, TaskMapping};
        use crate::backends::gliner2_fastino::schema::FieldType;
        use ndarray::Array4;

        let record = ProcessedRecord {
            input_ids: vec![],
            attention_mask: vec![],
            tasks: vec![],
            text_start: 0,
            text_end: 0,
            word_to_token_maps: vec![(0, 1)],
            word_to_char_maps: vec![(0, 4)],
        };
        let task = TaskMapping {
            task_name: "x".to_string(),
            task_type: "structures".to_string(),
            labels: vec!["a".into()],
            prompt_tok_idx: 0,
            field_tok_indices: vec![0],
        };
        let scorer_out = ScorerOutput {
            scores: Array4::<f32>::zeros((MAX_COUNT, 1, MAX_WIDTH, 1)),
        };
        let fields = vec![("a".to_string(), FieldType::String)];

        let result = decode_structure("Acme", &record, &task, &scorer_out, 0, 0.5, &fields);
        assert!(
            result.is_empty(),
            "expected 0 instances when pred_count=0, got {result:?}"
        );
    }

    #[test]
    fn decode_structure_multi_instance_separates_by_c_idx() {
        use crate::backends::gliner2_fastino::processor::{ProcessedRecord, TaskMapping};
        use crate::backends::gliner2_fastino::schema::{FieldType, StructureValue};
        use ndarray::Array4;

        // 3 words: "Marie Albert physicist".
        let record = ProcessedRecord {
            input_ids: vec![],
            attention_mask: vec![],
            tasks: vec![],
            text_start: 0,
            text_end: 0,
            word_to_token_maps: vec![(0, 1), (1, 2), (2, 3)],
            word_to_char_maps: vec![(0, 5), (6, 12), (13, 22)],
        };
        let task = TaskMapping {
            task_name: "person".to_string(),
            task_type: "structures".to_string(),
            labels: vec!["name".into()],
            prompt_tok_idx: 0,
            field_tok_indices: vec![0],
        };
        let mut scores = Array4::<f32>::zeros((MAX_COUNT, 3, MAX_WIDTH, 1));
        scores[[0, 0, 0, 0]] = 0.9; // instance 0, name = "Marie"
        scores[[1, 1, 0, 0]] = 0.8; // instance 1, name = "Albert"
        let scorer_out = ScorerOutput { scores };
        let fields = vec![("name".to_string(), FieldType::String)];

        let result = decode_structure(
            "Marie Albert physicist",
            &record,
            &task,
            &scorer_out,
            2,
            0.5,
            &fields,
        );

        assert_eq!(result.len(), 2, "expected 2 instances");
        let names: Vec<&String> = result
            .iter()
            .filter_map(|s| match s.fields.get("name") {
                Some(StructureValue::Single(n)) => Some(n),
                _ => None,
            })
            .collect();
        assert_eq!(names, vec![&"Marie".to_string(), &"Albert".to_string()]);
    }

    #[test]
    fn decode_structure_below_threshold_drops_field() {
        use crate::backends::gliner2_fastino::processor::{ProcessedRecord, TaskMapping};
        use crate::backends::gliner2_fastino::schema::FieldType;
        use ndarray::Array4;

        let record = ProcessedRecord {
            input_ids: vec![],
            attention_mask: vec![],
            tasks: vec![],
            text_start: 0,
            text_end: 0,
            word_to_token_maps: vec![(0, 1)],
            word_to_char_maps: vec![(0, 4)],
        };
        let task = TaskMapping {
            task_name: "t".to_string(),
            task_type: "structures".to_string(),
            labels: vec!["f".into()],
            prompt_tok_idx: 0,
            field_tok_indices: vec![0],
        };
        let mut scores = Array4::<f32>::zeros((MAX_COUNT, 1, MAX_WIDTH, 1));
        scores[[0, 0, 0, 0]] = 0.4; // below threshold 0.5
        let scorer_out = ScorerOutput { scores };
        let fields = vec![("f".to_string(), FieldType::String)];

        let result = decode_structure("Acme", &record, &task, &scorer_out, 1, 0.5, &fields);
        assert_eq!(
            result.len(),
            1,
            "instance is still emitted (with empty fields)"
        );
        assert!(
            result[0].fields.is_empty(),
            "field below threshold should be dropped, got {:?}",
            result[0].fields,
        );
    }
}
