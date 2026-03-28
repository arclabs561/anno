//! GLiNER ONNX configuration and prompt cache types.

use super::*;

pub struct GLiNERConfig {
    /// Prefer quantized models (INT8) for faster CPU inference.
    pub prefer_quantized: bool,
    /// ONNX optimization level (1-3, default 3).
    pub optimization_level: u8,
    /// Number of threads for inference (0 = auto).
    pub num_threads: usize,
    /// Cache size for prompt encodings (0 = disabled, default 100).
    ///
    /// The prompt cache stores encoded prompts keyed by (text, entity_types, model_id).
    /// This can materially reduce repeated work in evaluation loops and API usage patterns
    /// where the same text is queried with multiple type sets.
    pub prompt_cache_size: usize,
    /// Force bi-encoder mode. When `None`, auto-detect from ONNX input names.
    /// When `Some(true)`, assume bi-encoder layout. When `Some(false)`, force uni-encoder.
    pub bi_encoder: Option<bool>,
}

/// Whether the loaded ONNX model uses uni-encoder or bi-encoder architecture.
///
/// Bi-encoder GLiNER (Stepanov et al., arXiv:2602.18487) separates text and label
/// encoding for higher throughput at large label counts. The model is detected as
/// bi-encoder when the ONNX session has a `label_input_ids` input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EncoderMode {
    /// Standard GLiNER: labels and text concatenated into a single prompt.
    Uni,
    /// Bi-encoder GLiNER: text and labels encoded independently, matched via similarity.
    Bi,
}

/// Pre-computed label embedding for bi-encoder mode.
///
/// Cached per label string so that label encoding is amortized across calls.
#[cfg(feature = "onnx")]
#[derive(Debug, Clone)]
pub(crate) struct LabelEmbedding {
    /// The dense embedding vector produced by the label encoder.
    /// Currently empty (placeholder) until bi-encoder ONNX exports are available.
    #[allow(dead_code)]
    pub(crate) embedding: Vec<f32>,
}

#[cfg(feature = "onnx")]
impl Default for GLiNERConfig {
    fn default() -> Self {
        Self {
            prefer_quantized: true,
            optimization_level: 3,
            num_threads: 4,
            prompt_cache_size: 100,
            bi_encoder: None,
        }
    }
}

/// Cache key for prompt encodings.
///
/// Keyed by (text_hash, entity_types_hash, model_id) to ensure cache hits
/// only when text, entity types, and model are identical.
#[cfg(feature = "onnx")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct PromptCacheKey {
    pub(super) text_hash: u64,
    pub(super) entity_types_hash: u64,
    pub(super) model_id: String,
}

/// Cached prompt encoding result.
#[cfg(feature = "onnx")]
#[derive(Debug, Clone)]
pub(super) struct PromptCacheValue {
    pub(super) input_ids: Vec<i64>,
    pub(super) attention_mask: Vec<i64>,
    pub(super) words_mask: Vec<i64>,
    pub(super) text_lengths: i64,
    pub(super) entity_count: usize,
}
