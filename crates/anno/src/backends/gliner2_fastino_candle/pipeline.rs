//! 8-step pipeline orchestration for the Candle backend.
//!
//! Per `docs/dev-notes/gliner2-multi-v1-forward-pass.md`, this composes:
//!   encoder → token_gather (or schema_gather) → span_rep → count_pred
//!   → count_lstm → scorer
//! and packages the final scores as a [`super::decoder::ScorerOutput`]
//! matching the ONNX backend's contract so the shared decoder family
//! decodes it identically.

use candle_core::Tensor;
use candle_nn::ops;
use ndarray::Array4;

use super::decoder::{ScorerOutput, MAX_COUNT, MAX_WIDTH};
use super::heads::{schema_gather::SchemaGather, scorer::Scorer, token_gather::TokenGather};
use super::processor::{ProcessedRecord, TaskMapping};
use crate::backends::gliner2_fastino::pipeline::build_span_idx;
use crate::Error;

/// Span path: encoder → token_gather → span_rep → schema_gather →
/// count_pred → count_lstm → scorer → [`ScorerOutput`].
///
/// Returns `(scorer_out, pred_count)`. On `pred_count == 0`, returns an
/// empty `ScorerOutput` like the ONNX backend's `run_pipeline_dispatch`.
pub(crate) fn run_pipeline_candle(
    model: &super::GLiNER2FastinoCandle,
    record: &ProcessedRecord,
    task: &TaskMapping,
) -> crate::Result<(ScorerOutput, usize)> {
    let device = &model.device;

    // 1. Build input tensors.
    let seq_len = record.input_ids.len();
    let input_ids = Tensor::from_slice(&record.input_ids[..], (1, seq_len), device)
        .map_err(|e| Error::Backend(format!("candle input_ids: {e}")))?;
    let attn_mask: Vec<u32> = record.attention_mask.iter().map(|&v| v as u32).collect();
    let attention_mask = Tensor::from_slice(&attn_mask[..], (1, seq_len), device)
        .map_err(|e| Error::Backend(format!("candle attn_mask: {e}")))?;

    // 2. Encode.
    let hidden = model
        .encoder
        .forward(&input_ids, &attention_mask, None)
        .map_err(|e| Error::Backend(format!("candle encoder.forward: {e}")))?;
    // hidden: [1, S, H]

    // 3. Token gather → text_emb.
    let num_words = record.word_to_token_maps.len();
    if num_words == 0 {
        return Ok((empty_scorer_output(), 0));
    }
    let word_starts: Vec<u32> = record
        .word_to_token_maps
        .iter()
        .map(|&(start, _)| start as u32)
        .collect();
    let word_indices = Tensor::from_slice(&word_starts[..], (num_words,), device)
        .map_err(|e| Error::Backend(format!("candle word_indices: {e}")))?;
    let text_emb = TokenGather
        .forward(&hidden, &word_indices)
        .map_err(|e| Error::Backend(format!("candle token_gather: {e}")))?;
    // text_emb: [1, num_words, H]

    // 4. Span rep.
    let span_idx_arr = build_span_idx(num_words); // ndarray Array3<i64> shape [1, T*W, 2]
    let span_idx_data: Vec<i64> = span_idx_arr.iter().copied().collect();
    let span_idx = Tensor::from_slice(&span_idx_data[..], (1, num_words * MAX_WIDTH, 2), device)
        .map_err(|e| Error::Backend(format!("candle span_idx: {e}")))?;
    let span_rep_out = model
        .heads
        .span_rep
        .forward(&text_emb, &span_idx)
        .map_err(|e| Error::Backend(format!("candle span_rep: {e}")))?;
    // span_rep_out: [1, num_words, MAX_WIDTH, H]

    // 5. Schema gather: [P] index + per-field indices.
    let mut schema_idx: Vec<u32> = Vec::with_capacity(1 + task.field_tok_indices.len());
    schema_idx.push(task.prompt_tok_idx as u32);
    schema_idx.extend(task.field_tok_indices.iter().map(|&i| i as u32));
    let schema_idx_t = Tensor::from_slice(&schema_idx[..], (schema_idx.len(),), device)
        .map_err(|e| Error::Backend(format!("candle schema_idx: {e}")))?;
    let sg_out = SchemaGather
        .forward(&hidden, &schema_idx_t)
        .map_err(|e| Error::Backend(format!("candle schema_gather: {e}")))?;
    // sg_out.pc_emb: [1, H], sg_out.field_embs: [F, H]

    // 6. Count pred.
    let pred_count = model
        .heads
        .count_pred
        .forward(&sg_out.pc_emb)
        .map_err(|e| Error::Backend(format!("candle count_pred: {e}")))?;
    if pred_count == 0 {
        return Ok((empty_scorer_output(), 0));
    }

    // 7. Count LSTM (GRU): produces struct_proj [pred_count, F, H].
    let struct_proj = model
        .heads
        .count_lstm
        .forward(&sg_out.field_embs, pred_count, device)
        .map_err(|e| Error::Backend(format!("candle count_lstm: {e}")))?;

    // 8. Scorer: [pred_count, F, num_words, MAX_WIDTH] sigmoid scores.
    // span_rep_out is [1, num_words, MAX_WIDTH, H] — squeeze batch.
    let span_rep_per_sample = span_rep_out
        .squeeze(0)
        .map_err(|e| Error::Backend(format!("candle span_rep squeeze: {e}")))?;
    let scores = Scorer
        .forward(&span_rep_per_sample, &struct_proj)
        .map_err(|e| Error::Backend(format!("candle scorer: {e}")))?;
    // scores: [pred_count, F, num_words, MAX_WIDTH]

    // 9. Convert to ScorerOutput format (ndarray Array4):
    //    ONNX shape: [MAX_COUNT, num_words, MAX_WIDTH, M].
    //    Our scores: [pred_count, F, num_words, MAX_WIDTH] — different
    //    axis order. Permute to [pred_count, num_words, MAX_WIDTH, F]
    //    then pad first dim to MAX_COUNT.
    let scores = scores
        .permute((0, 2, 3, 1))
        .map_err(|e| Error::Backend(format!("candle scores permute: {e}")))?
        .contiguous()
        .map_err(|e| Error::Backend(format!("candle scores contiguous: {e}")))?;
    // scores: [pred_count, num_words, MAX_WIDTH, F]

    let num_fields = task.labels.len();
    let scores_padded: Tensor = if pred_count < MAX_COUNT {
        let pad_shape = (MAX_COUNT - pred_count, num_words, MAX_WIDTH, num_fields);
        let pad = Tensor::zeros(pad_shape, scores.dtype(), device)
            .map_err(|e| Error::Backend(format!("candle scores pad: {e}")))?;
        Tensor::cat(&[&scores, &pad], 0)
            .map_err(|e| Error::Backend(format!("candle scores cat: {e}")))?
    } else {
        scores
    };
    // scores_padded: [MAX_COUNT, num_words, MAX_WIDTH, num_fields]

    // 10. Read back to host as Array4<f32>.
    let scores_vec: Vec<f32> = scores_padded
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| Error::Backend(format!("candle scores readback: {e}")))?;
    let scores_arr =
        Array4::from_shape_vec((MAX_COUNT, num_words, MAX_WIDTH, num_fields), scores_vec)
            .map_err(|e| Error::Backend(format!("candle scores reshape: {e}")))?;

    Ok((ScorerOutput { scores: scores_arr }, pred_count))
}

/// Classify path: encoder → schema_gather → count_pred (host-only;
/// classify ignores `pred_count`) → classifier softmax.
pub(crate) fn run_classify_pipeline_candle(
    model: &super::GLiNER2FastinoCandle,
    record: &ProcessedRecord,
    task: &TaskMapping,
) -> crate::Result<Vec<f32>> {
    let device = &model.device;
    let seq_len = record.input_ids.len();
    let input_ids = Tensor::from_slice(&record.input_ids[..], (1, seq_len), device)
        .map_err(|e| Error::Backend(format!("candle input_ids: {e}")))?;
    let attn_mask: Vec<u32> = record.attention_mask.iter().map(|&v| v as u32).collect();
    let attention_mask = Tensor::from_slice(&attn_mask[..], (1, seq_len), device)
        .map_err(|e| Error::Backend(format!("candle attn_mask: {e}")))?;

    let hidden = model
        .encoder
        .forward(&input_ids, &attention_mask, None)
        .map_err(|e| Error::Backend(format!("candle encoder.forward: {e}")))?;

    let mut schema_idx: Vec<u32> = Vec::with_capacity(1 + task.field_tok_indices.len());
    schema_idx.push(task.prompt_tok_idx as u32);
    schema_idx.extend(task.field_tok_indices.iter().map(|&i| i as u32));
    let schema_idx_t = Tensor::from_slice(&schema_idx[..], (schema_idx.len(),), device)
        .map_err(|e| Error::Backend(format!("candle schema_idx: {e}")))?;
    let sg_out = SchemaGather
        .forward(&hidden, &schema_idx_t)
        .map_err(|e| Error::Backend(format!("candle schema_gather: {e}")))?;

    let pred_count = model
        .heads
        .count_pred
        .forward(&sg_out.pc_emb)
        .map_err(|e| Error::Backend(format!("candle count_pred: {e}")))?;
    if pred_count == 0 || task.labels.is_empty() {
        return Ok(vec![0.0; task.labels.len()]);
    }

    // Classifier: field_embs → [num_labels] logits, then softmax.
    let logits = model
        .heads
        .classifier
        .forward(&sg_out.field_embs)
        .map_err(|e| Error::Backend(format!("candle classifier: {e}")))?;
    let probs = ops::softmax(&logits, candle_core::D::Minus1)
        .map_err(|e| Error::Backend(format!("candle softmax: {e}")))?;

    probs
        .to_vec1::<f32>()
        .map_err(|e| Error::Backend(format!("candle probs readback: {e}")))
}

fn empty_scorer_output() -> ScorerOutput {
    ScorerOutput {
        scores: Array4::zeros((0, 0, 0, 0)),
    }
}
