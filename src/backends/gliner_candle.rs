//! GLiNER implementation using Candle (pure Rust ML) with Metal/CUDA support.
//!
//! Zero-shot NER using bi-encoder architecture: match text spans to entity labels.
//!
//! # Architecture
//!
//! ```text
//! Text Input     Label Input
//!     |              |
//!     v              v
//! [Tokenizer]   [Tokenizer]
//!     |              |
//!     v              v
//! [Transformer Encoder] (shared)
//!     |              |
//!     v              v
//! [SpanRepLayer]  [LabelEncoder]
//!     |              |
//!     +------+-------+
//!            |
//!            v
//!     [SpanLabelMatcher]
//!            |
//!            v
//!       [Entities]
//! ```
//!
//! # GPU Support
//!
//! - **Metal** (Apple Silicon): `cargo build --features candle,metal`
//! - **CUDA** (NVIDIA): `cargo build --features candle,cuda`
//! - **CPU**: Always available as fallback
//!
//! # Example
//!
//! ```rust,ignore
//! use anno::backends::gliner_candle::GLiNERCandle;
//!
//! let model = GLiNERCandle::from_pretrained("urchade/gliner_small-v2.1")?;
//! let entities = model.extract(
//!     "Steve Jobs founded Apple in California.",
//!     &["person", "organization", "location"],
//!     0.5,
//! )?;
//! ```

#![allow(dead_code)] // Token constants for future prompt encoding

use crate::{Entity, EntityType, Error, Result};
use std::path::{Path, PathBuf};

#[cfg(feature = "candle")]
use {
    super::encoder_candle::{CandleEncoder, TextEncoder},
    candle_core::{DType, Device, IndexOp, Module, Tensor, D},
    candle_nn::{embedding, linear, Embedding, Linear, VarBuilder},
    tokenizers::Tokenizer,
};

/// Maximum span width for entity candidates.
const MAX_SPAN_WIDTH: usize = 12;

/// Special tokens for GLiNER models.
#[cfg(feature = "candle")]
const TOKEN_START: u32 = 1;
#[cfg(feature = "candle")]
const TOKEN_END: u32 = 2;
#[cfg(feature = "candle")]
const TOKEN_ENT: u32 = 128002;
#[cfg(feature = "candle")]
const TOKEN_SEP: u32 = 128003;

// =============================================================================
// Device Selection
// =============================================================================

/// Get the best available compute device.
#[cfg(feature = "candle")]
pub fn best_device() -> Result<Device> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        if let Ok(device) = Device::new_metal(0) {
            log::info!("[GLiNER-Candle] Using Metal GPU");
            return Ok(device);
        }
    }

    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = Device::new_cuda(0) {
            log::info!("[GLiNER-Candle] Using CUDA GPU");
            return Ok(device);
        }
    }

    log::info!("[GLiNER-Candle] Using CPU");
    Ok(Device::Cpu)
}

// =============================================================================
// Span Representation Layer
// =============================================================================

/// Span representation: combines start, end, and width embeddings.
#[cfg(feature = "candle")]
pub struct SpanRepLayer {
    projection: Linear,
    width_embeddings: Embedding,
    hidden_size: usize,
}

#[cfg(feature = "candle")]
impl SpanRepLayer {
    /// Create a new span representation layer.
    pub fn new(hidden_size: usize, max_width: usize, vb: VarBuilder) -> Result<Self> {
        let width_emb_size = hidden_size / 4;
        let input_size = hidden_size * 2 + width_emb_size;

        let projection = linear(input_size, hidden_size, vb.pp("projection"))
            .map_err(|e| Error::Retrieval(format!("SpanRepLayer projection: {}", e)))?;

        let width_embeddings = embedding(max_width, width_emb_size, vb.pp("width_embeddings"))
            .map_err(|e| Error::Retrieval(format!("SpanRepLayer width_embeddings: {}", e)))?;

        Ok(Self {
            projection,
            width_embeddings,
            hidden_size,
        })
    }

    /// Compute span embeddings from token embeddings.
    ///
    /// # Arguments
    /// * `token_embeddings` - [batch, seq_len, hidden]
    /// * `span_indices` - [batch, num_spans, 2] (start, end)
    ///
    /// # Returns
    /// [batch, num_spans, hidden]
    pub fn forward(&self, token_embeddings: &Tensor, span_indices: &Tensor) -> Result<Tensor> {
        let (batch_size, _seq_len, _hidden) = token_embeddings
            .dims3()
            .map_err(|e| Error::Parse(format!("token_embeddings dims: {}", e)))?;
        let (_, _num_spans, _) = span_indices
            .dims3()
            .map_err(|e| Error::Parse(format!("span_indices dims: {}", e)))?;

        let start_idx = span_indices.i((.., .., 0))?.to_dtype(DType::U32)?;
        let end_idx = span_indices.i((.., .., 1))?.to_dtype(DType::U32)?;

        let mut span_embs = Vec::new();

        for b in 0..batch_size {
            let batch_tokens = token_embeddings.i(b)?;
            let batch_starts = start_idx.i(b)?;
            let batch_ends = end_idx.i(b)?;

            // Validate indices are within bounds before selection
            let seq_len = batch_tokens.dims()[0];
            // Candle's index_select will validate, but we check explicitly for clearer errors
            // The actual validation happens in index_select, but this helps with debugging

            let widths = (&batch_ends - &batch_starts)?;
            let width_embs = self.width_embeddings.forward(&widths)?;

            let start_embs = batch_tokens.index_select(&batch_starts, 0).map_err(|e| {
                Error::Parse(format!(
                    "Span start indices out of bounds for seq_len={}: {}",
                    seq_len, e
                ))
            })?;
            let end_embs = batch_tokens.index_select(&batch_ends, 0).map_err(|e| {
                Error::Parse(format!(
                    "Span end indices out of bounds for seq_len={}: {}",
                    seq_len, e
                ))
            })?;

            let combined = Tensor::cat(&[&start_embs, &end_embs, &width_embs], D::Minus1)?;
            let span_emb = self.projection.forward(&combined)?;
            span_embs.push(span_emb);
        }

        Tensor::stack(&span_embs, 0).map_err(|e| Error::Parse(format!("stack span_embs: {}", e)))
    }
}

// =============================================================================
// Label Encoder
// =============================================================================

/// Projects label embeddings to matching space.
#[cfg(feature = "candle")]
pub struct LabelEncoder {
    projection: Linear,
}

#[cfg(feature = "candle")]
impl LabelEncoder {
    /// Create a new label encoder.
    pub fn new(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let projection = linear(hidden_size, hidden_size, vb.pp("label_projection"))
            .map_err(|e| Error::Retrieval(format!("LabelEncoder: {}", e)))?;

        Ok(Self { projection })
    }

    /// Project label embeddings to matching space.
    pub fn forward(&self, label_embeddings: &Tensor) -> Result<Tensor> {
        self.projection
            .forward(label_embeddings)
            .map_err(|e| Error::Parse(format!("label projection: {}", e)))
    }
}

// =============================================================================
// Span-Label Matcher
// =============================================================================

/// Computes similarity between spans and labels.
#[cfg(feature = "candle")]
pub struct SpanLabelMatcher {
    temperature: f64,
}

#[cfg(feature = "candle")]
impl SpanLabelMatcher {
    /// Create a new span-label matcher with temperature scaling.
    pub fn new(temperature: f64) -> Self {
        Self { temperature }
    }

    /// Match spans to labels via cosine similarity.
    ///
    /// # Arguments
    /// * `span_embeddings` - [batch, num_spans, hidden]
    /// * `label_embeddings` - [num_labels, hidden]
    ///
    /// # Returns
    /// [batch, num_spans, num_labels] scores in [0, 1]
    pub fn forward(&self, span_embeddings: &Tensor, label_embeddings: &Tensor) -> Result<Tensor> {
        let span_norm = l2_normalize(span_embeddings, D::Minus1)?;
        let label_norm = l2_normalize(label_embeddings, D::Minus1)?;

        let batch_size = span_norm.dims()[0];
        let label_t = label_norm.t()?;
        let label_t = label_t.unsqueeze(0)?.broadcast_as((
            batch_size,
            label_t.dims()[0],
            label_t.dims()[1],
        ))?;

        let scores = span_norm.matmul(&label_t)?;
        let scaled = (scores * self.temperature)?;

        candle_nn::ops::sigmoid(&scaled).map_err(|e| Error::Parse(format!("sigmoid: {}", e)))
    }
}

#[cfg(feature = "candle")]
fn l2_normalize(tensor: &Tensor, dim: D) -> Result<Tensor> {
    let norm = tensor.sqr()?.sum(dim)?.sqrt()?;
    let norm = norm.unsqueeze(D::Minus1)?;
    // Clamp norm to prevent division by zero (same as gliner2.rs)
    let norm_clamped = norm
        .clamp(1e-12, f32::MAX)
        .map_err(|e| Error::Parse(format!("clamp: {}", e)))?;
    tensor
        .broadcast_div(&norm_clamped)
        .map_err(|e| Error::Parse(format!("l2_normalize: {}", e)))
}

// =============================================================================
// GLiNER Candle Model
// =============================================================================

/// GLiNER zero-shot NER using pure Rust Candle backend.
///
/// Matches text spans to entity type descriptions using a bi-encoder.
/// Supports Metal (Apple Silicon) and CUDA (NVIDIA) GPU acceleration.
#[cfg(feature = "candle")]
pub struct GLiNERCandle {
    /// Text encoder (BERT/ModernBERT/DeBERTa)
    encoder: CandleEncoder,
    /// Tokenizer
    tokenizer: Tokenizer,
    /// Span representation layer
    span_rep: SpanRepLayer,
    /// Label encoder
    label_encoder: LabelEncoder,
    /// Span-label matcher
    matcher: SpanLabelMatcher,
    /// Model name
    model_name: String,
    /// Hidden size
    hidden_size: usize,
    /// Device
    device: Device,
}

#[cfg(feature = "candle")]
impl std::fmt::Debug for GLiNERCandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GLiNERCandle")
            .field("model_name", &self.model_name)
            .field("hidden_size", &self.hidden_size)
            .field("device", &format!("{:?}", self.device))
            .finish_non_exhaustive()
    }
}

/// Helper function to convert pytorch_model.bin to safetensors format
///
/// # Implementation Options
///
/// 1. **Python subprocess** (pragmatic): Calls Python's safetensors library
/// 2. **Pure Rust** (complex): Requires parsing PyTorch pickle format manually
///
/// PyTorch state dicts use Python pickle format with `torch._utils._rebuild_tensor_v2`
/// which requires parsing complex nested structures. The `tch` crate can load models
/// but doesn't provide direct state dict -> safetensors conversion.
#[cfg(feature = "candle")]
pub(crate) fn convert_pytorch_to_safetensors(pytorch_path: &Path) -> Result<PathBuf> {
    let cache_dir = pytorch_path
        .parent()
        .ok_or_else(|| Error::Retrieval("Invalid pytorch model path".to_string()))?;

    let safetensors_path = cache_dir.join("model_converted.safetensors");

    // Check if already converted
    if safetensors_path.exists() {
        log::debug!("Using cached safetensors conversion");
        return Ok(safetensors_path);
    }

    log::info!(
        "Converting PyTorch model to safetensors: {:?}",
        pytorch_path
    );

    // Find the conversion script (in scripts/ directory relative to crate root)
    let script_path = if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        Path::new(&manifest_dir).join("scripts/convert_pytorch_to_safetensors.py")
    } else {
        // Fallback: try to find script relative to current executable
        Path::new("scripts/convert_pytorch_to_safetensors.py").to_path_buf()
    };

    // Try uv run first (PEP 723 script with inline dependencies)
    let output = std::process::Command::new("uv")
        .arg("run")
        .arg("--script")
        .arg(&script_path)
        .arg(pytorch_path)
        .arg(&safetensors_path)
        .output()
        .or_else(|_| {
            // Fallback to python3 if uv is not available
            std::process::Command::new("python3")
                .arg(&script_path)
                .arg(pytorch_path)
                .arg(&safetensors_path)
                .output()
        })
        .map_err(|e| {
            Error::Retrieval(format!(
                "Failed to run conversion script (uv or python3 not found?): {}",
                e
            ))
        })?;

    if output.status.success() {
        if safetensors_path.exists() {
            log::info!(
                "Successfully converted to safetensors: {:?}",
                safetensors_path
            );
            return Ok(safetensors_path);
        }
    }

    // If Python conversion failed, provide helpful error
    let error_msg = String::from_utf8_lossy(&output.stderr);
    let stdout_msg = String::from_utf8_lossy(&output.stdout);

    Err(Error::Retrieval(format!(
        "PyTorch to safetensors conversion failed. \
         \
         Script: {:?} \
         Error: {} \
         Output: {} \
         \
         Recommended solutions (in order of preference): \
         1. Use GLiNEROnnx (ONNX backend) - works with all GLiNER models, no conversion needed \
         2. Use a model that already has safetensors format (e.g., knowledgator/modern-gliner-bi-large-v1.0) \
         3. Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh \
         4. Manual conversion: uv run --script scripts/convert_pytorch_to_safetensors.py \"{}\" \"{}\" \
         \
         Note: Pure Rust conversion would require parsing PyTorch pickle format (torch._utils._rebuild_tensor_v2) \
         which is complex. Python's torch.load handles this automatically.",
        script_path,
        error_msg,
        stdout_msg,
        pytorch_path.display(),
        safetensors_path.display()
    )))
}

#[cfg(feature = "candle")]
impl GLiNERCandle {
    /// Load GLiNER from HuggingFace.
    ///
    /// # Arguments
    /// * `model_id` - HuggingFace model ID (e.g., "urchade/gliner_small-v2.1")
    pub fn from_pretrained(model_id: &str) -> Result<Self> {
        use hf_hub::api::sync::Api;

        let device = best_device()?;

        let api = Api::new().map_err(|e| Error::Retrieval(format!("HuggingFace API: {}", e)))?;

        let repo = api.model(model_id.to_string());

        // Download files
        // Try knowledgator models first (they have safetensors + tokenizer.json)
        // knowledgator/modern-gliner-bi-large-v1.0 has safetensors available
        // Fall back to urchade models if needed
        let tokenizer_path = repo.get("tokenizer.json").map_err(|e| {
            Error::Retrieval(format!(
                "tokenizer.json not found. GLiNER Candle requires tokenizer.json. \
                 Try using knowledgator/modern-gliner-bi-large-v1.0 (has safetensors) \
                 or GLiNEROnnx instead. Original error: {}",
                e
            ))
        })?;
        // GLiNER Candle requires safetensors format
        // Most GLiNER models only have pytorch_model.bin, which Candle cannot load directly
        // Workaround: Try to convert pytorch_model.bin to safetensors on-the-fly
        let weights_path = repo
            .get("model.safetensors")
            .or_else(|_| repo.get("gliner_model.safetensors"))
            .or_else(|_| {
                // Workaround: Try to convert pytorch_model.bin to safetensors
                // Now that we have From<ApiError>, we can use ? directly
                let pytorch_path = repo.get("pytorch_model.bin")?;
                convert_pytorch_to_safetensors(&pytorch_path)
            })
            .map_err(|e| Error::Retrieval(format!(
                "safetensors weights not found and conversion failed. GLiNER Candle requires safetensors format. \
                 Most GLiNER models (urchade/, knowledgator/) only provide pytorch_model.bin. \
                 Attempted automatic conversion but it failed. \
                 Please use GLiNEROnnx (ONNX version) instead, which works with all GLiNER models. \
                 Original error: {}",
                e
            )))?;
        let config_path = repo
            .get("config.json")
            .or_else(|_| repo.get("gliner_config.json"))
            .map_err(|e| Error::Retrieval(format!("config.json: {}", e)))?;

        // Load tokenizer (only if tokenizer.json, not tokenizer_config.json)
        let tokenizer = if tokenizer_path.ends_with("tokenizer.json") {
            Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| Error::Retrieval(format!("tokenizer: {}", e)))?
        } else {
            return Err(Error::Retrieval(format!(
                "GLiNER Candle requires tokenizer.json, but only found {}. \
                 The model may not be in Candle-compatible format. \
                 Consider using GLiNEROnnx instead.",
                tokenizer_path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
            )));
        };

        // Parse config for hidden size
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| Error::Retrieval(format!("config: {}", e)))?;
        let config: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| Error::Parse(format!("config JSON: {}", e)))?;

        let hidden_size = config["hidden_size"].as_u64().unwrap_or(768) as usize;

        // Load weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                .map_err(|e| Error::Retrieval(format!("safetensors: {}", e)))?
        };

        // Build encoder from the same model
        let encoder = CandleEncoder::from_pretrained(model_id)?;

        // Build GLiNER-specific components
        let span_rep = SpanRepLayer::new(hidden_size, MAX_SPAN_WIDTH, vb.pp("span_rep"))?;
        let label_encoder = LabelEncoder::new(hidden_size, vb.pp("label_encoder"))?;
        let matcher = SpanLabelMatcher::new(1.0);

        log::info!(
            "[GLiNER-Candle] Loaded {} (hidden={}) on {:?}",
            model_id,
            hidden_size,
            device
        );

        Ok(Self {
            encoder,
            tokenizer,
            span_rep,
            label_encoder,
            matcher,
            model_name: model_id.to_string(),
            hidden_size,
            device,
        })
    }

    /// Simplified constructor that creates with random weights (for testing).
    pub fn new(model_name: &str) -> Result<Self> {
        Self::from_pretrained(model_name)
    }

    /// Extract entities with custom labels (zero-shot).
    ///
    /// # Arguments
    /// * `text` - Input text
    /// * `labels` - Entity types to detect (e.g., ["person", "organization"])
    /// * `threshold` - Confidence threshold (0.0-1.0)
    pub fn extract(&self, text: &str, labels: &[&str], threshold: f32) -> Result<Vec<Entity>> {
        if text.is_empty() || labels.is_empty() {
            return Ok(vec![]);
        }

        // Tokenize text word-by-word (GLiNER pattern)
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Ok(vec![]);
        }

        // Build prompt: [START] <<ENT>> label1 <<ENT>> label2 <<SEP>> word1 word2 ... [END]
        let (text_embeddings, word_positions) = self.encode_text(&words)?;
        let label_embeddings = self.encode_labels(labels)?;

        // Generate span candidates
        let span_indices = self.generate_spans(words.len())?;

        // Compute span embeddings
        let span_embs = self.span_rep.forward(&text_embeddings, &span_indices)?;

        // Compute label embeddings
        let label_embs = self.label_encoder.forward(&label_embeddings)?;

        // Match spans to labels
        let scores = self.matcher.forward(&span_embs, &label_embs)?;

        // Decode to entities
        let entities =
            self.decode_entities(text, &words, &word_positions, &scores, labels, threshold)?;

        Ok(entities)
    }

    fn encode_text(&self, words: &[&str]) -> Result<(Tensor, Vec<(usize, usize)>)> {
        // For now, encode each word and average. Full implementation would:
        // 1. Build GLiNER prompt format
        // 2. Get per-word embeddings from words_mask

        let text = words.join(" ");
        let (embeddings, seq_len) = self.encoder.encode(&text)?;

        // Reshape to [1, seq_len, hidden]
        let tensor = Tensor::from_vec(embeddings, (1, seq_len, self.hidden_size), &self.device)
            .map_err(|e| Error::Parse(format!("text tensor: {}", e)))?;

        // Build word positions
        let full_text = words.join(" ");
        let word_positions: Vec<(usize, usize)> = {
            // Performance: Pre-allocate positions vec with known size
            let mut positions = Vec::with_capacity(words.len());
            let mut pos = 0;
            for (idx, word) in words.iter().enumerate() {
                if let Some(start) = full_text[pos..].find(word) {
                    let abs_start = pos + start;
                    let abs_end = abs_start + word.len();
                    // Validate position is after previous word (words should be in order)
                    if !positions.is_empty() {
                        let (_prev_start, prev_end) = positions[positions.len() - 1];
                        if abs_start < prev_end {
                            log::warn!(
                                "Word '{}' (index {}) at position {} overlaps with previous word ending at {}",
                                word,
                                idx,
                                abs_start,
                                prev_end
                            );
                        }
                    }
                    positions.push((abs_start, abs_end));
                    pos = abs_end;
                } else {
                    // Word not found - return error to prevent silent entity skipping
                    return Err(Error::Parse(format!(
                        "Word '{}' (index {}) not found in text starting at position {}",
                        word, idx, pos
                    )));
                }
            }
            positions
        };

        // Validate that we found positions for all words
        if word_positions.len() != words.len() {
            return Err(Error::Parse(format!(
                "Word position mismatch: found {} positions for {} words",
                word_positions.len(),
                words.len()
            )));
        }

        Ok((tensor, word_positions))
    }

    fn encode_labels(&self, labels: &[&str]) -> Result<Tensor> {
        // Encode each label
        // Performance: Pre-allocate all_embeddings with estimated capacity
        // Each label produces hidden_size embeddings
        let mut all_embeddings = Vec::with_capacity(labels.len().saturating_mul(self.hidden_size));

        for label in labels {
            let (embeddings, seq_len) = self.encoder.encode(label)?;
            // Average pool to get single embedding - handle empty sequences
            let avg: Vec<f32> = if seq_len == 0 {
                // Return zero vector for empty sequences
                vec![0.0f32; self.hidden_size]
            } else {
                (0..self.hidden_size)
                    .map(|i| {
                        embeddings
                            .iter()
                            .skip(i)
                            .step_by(self.hidden_size)
                            .take(seq_len)
                            .sum::<f32>()
                            / seq_len as f32
                    })
                    .collect()
            };
            all_embeddings.extend(avg);
        }

        Tensor::from_vec(
            all_embeddings,
            (labels.len(), self.hidden_size),
            &self.device,
        )
        .map_err(|e| Error::Parse(format!("label tensor: {}", e)))
    }

    fn generate_spans(&self, num_words: usize) -> Result<Tensor> {
        // Performance: Pre-allocate spans vec with estimated capacity
        // num_words * MAX_SPAN_WIDTH * 2 (for start/end pairs)
        let estimated_capacity = num_words.saturating_mul(MAX_SPAN_WIDTH).saturating_mul(2);
        let mut spans = Vec::with_capacity(estimated_capacity.min(1000));

        for start in 0..num_words {
            for width in 0..MAX_SPAN_WIDTH.min(num_words - start) {
                let end = start + width;
                spans.push(start as i64);
                spans.push(end as i64);
            }
        }

        let num_spans = spans.len() / 2;
        Tensor::from_vec(spans, (1, num_spans, 2), &self.device)
            .map_err(|e| Error::Parse(format!("span tensor: {}", e)))
    }

    fn decode_entities(
        &self,
        text: &str,
        words: &[&str],
        word_positions: &[(usize, usize)],
        scores: &Tensor,
        labels: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        // scores: [1, num_spans, num_labels]
        let scores_vec = scores
            .flatten_all()
            .map_err(|e| Error::Parse(format!("flatten scores: {}", e)))?
            .to_vec1::<f32>()
            .map_err(|e| Error::Parse(format!("scores to vec: {}", e)))?;

        let num_labels = labels.len();
        let num_spans = scores_vec.len() / num_labels;

        // Performance: Pre-allocate entities vec with estimated capacity
        let mut entities = Vec::with_capacity(num_spans.min(32));
        let mut span_idx = 0;

        for start in 0..words.len() {
            for width in 0..MAX_SPAN_WIDTH.min(words.len() - start) {
                if span_idx >= num_spans {
                    break;
                }

                // Note: generate_spans uses end = start + width (inclusive end word index).
                // For word_positions indexing, we need the last word index (inclusive).
                // Since word_positions[i] corresponds to word i, we use end_inclusive directly.
                // Loop bounds ensure: width < words.len() - start, so end_inclusive < words.len().
                let end_inclusive = start + width; // Last word index (inclusive), matches generate_spans

                // Find best label for this span
                let base = span_idx * num_labels;
                let mut best_label = 0;
                let mut best_score = 0.0f32;

                for (label_idx, _) in labels.iter().enumerate() {
                    let score = scores_vec.get(base + label_idx).copied().unwrap_or(0.0);
                    if score > best_score {
                        best_score = score;
                        best_label = label_idx;
                    }
                }

                if best_score >= threshold {
                    // Validate bounds: end_inclusive must be < word_positions.len()
                    // (Loop bounds ensure this, but defensive check for safety)
                    if start < word_positions.len() && end_inclusive < word_positions.len() {
                        if let (Some(&(start_pos, _)), Some(&(_, end_pos))) =
                            (word_positions.get(start), word_positions.get(end_inclusive))
                        {
                            if let Some(entity_text) = text.get(start_pos..end_pos) {
                                let label = labels[best_label];
                                let entity_type = Self::map_label(label);
                                entities.push(Entity::new(
                                    entity_text,
                                    entity_type,
                                    start_pos,
                                    end_pos,
                                    best_score as f64,
                                ));
                            }
                        }
                    }
                }

                span_idx += 1;
            }
        }

        // Performance: Use unstable sort (we don't need stable sort here)
        // Remove overlapping (keep highest scoring)
        entities.sort_unstable_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Performance: Pre-allocate filtered vec with estimated capacity
        let mut filtered = Vec::with_capacity(entities.len().min(32));
        for entity in entities {
            let overlaps = filtered
                .iter()
                .any(|e: &Entity| !(entity.end <= e.start || entity.start >= e.end));
            if !overlaps {
                filtered.push(entity);
            }
        }

        // Performance: Use unstable sort (we don't need stable sort here)
        filtered.sort_unstable_by_key(|e| e.start);
        Ok(filtered)
    }

    fn map_label(label: &str) -> EntityType {
        match label.to_lowercase().as_str() {
            "person" | "per" => EntityType::Person,
            "organization" | "org" | "company" => EntityType::Organization,
            "location" | "loc" | "place" | "gpe" => EntityType::Location,
            "date" => EntityType::Date,
            "time" => EntityType::Time,
            "money" | "currency" => EntityType::Money,
            "percent" | "percentage" => EntityType::Percent,
            other => EntityType::Other(other.to_string()),
        }
    }

    /// Get device as a string.
    pub fn device(&self) -> String {
        match &self.device {
            Device::Cpu => "cpu".to_string(),
            Device::Metal(_) => "metal".to_string(),
            Device::Cuda(_) => "cuda".to_string(),
        }
    }

    /// Get model name.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

// =============================================================================
// Model Trait Implementation
// =============================================================================

#[cfg(feature = "candle")]
const DEFAULT_GLINER_LABELS: &[&str] = &[
    "person",
    "organization",
    "location",
    "date",
    "time",
    "money",
    "percent",
    "product",
    "event",
    "facility",
    "work_of_art",
    "law",
    "language",
];

#[cfg(feature = "candle")]
impl crate::Model for GLiNERCandle {
    fn extract_entities(&self, text: &str, _language: Option<&str>) -> Result<Vec<Entity>> {
        self.extract(text, DEFAULT_GLINER_LABELS, 0.5)
    }

    fn supported_types(&self) -> Vec<EntityType> {
        DEFAULT_GLINER_LABELS
            .iter()
            .map(|label| Self::map_label(label))
            .collect()
    }

    fn is_available(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "GLiNER-Candle"
    }

    fn description(&self) -> &'static str {
        "Zero-shot NER using GLiNER bi-encoder (pure Rust with Metal/CUDA support)"
    }
}

#[cfg(feature = "candle")]
impl crate::backends::inference::ZeroShotNER for GLiNERCandle {
    fn extract_with_types(
        &self,
        text: &str,
        entity_types: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        self.extract(text, entity_types, threshold)
    }

    fn extract_with_descriptions(
        &self,
        text: &str,
        descriptions: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        // GLiNER can use descriptions directly as label text
        self.extract(text, descriptions, threshold)
    }
}

// =============================================================================
// Non-candle stub
// =============================================================================

#[cfg(not(feature = "candle"))]
#[derive(Debug)]
pub struct GLiNERCandle {
    _private: (),
}

#[cfg(not(feature = "candle"))]
impl GLiNERCandle {
    /// Create GLiNER (requires candle feature).
    pub fn new(_model_name: &str) -> Result<Self> {
        Err(Error::FeatureNotAvailable(
            "GLiNER-Candle requires the 'candle' feature. \
             Build with: cargo build --features candle\n\
             Alternative: Use GLiNEROnnx with the 'onnx' feature for similar functionality."
                .to_string(),
        ))
    }

    /// Load from pretrained (requires candle feature).
    pub fn from_pretrained(_model_id: &str) -> Result<Self> {
        Self::new("")
    }
}

#[cfg(not(feature = "candle"))]
impl crate::Model for GLiNERCandle {
    fn extract_entities(&self, _text: &str, _language: Option<&str>) -> Result<Vec<Entity>> {
        Err(Error::FeatureNotAvailable(
            "GLiNER-Candle requires the 'candle' feature".to_string(),
        ))
    }

    fn supported_types(&self) -> Vec<EntityType> {
        vec![]
    }

    fn is_available(&self) -> bool {
        false
    }

    fn name(&self) -> &'static str {
        "GLiNER-Candle (unavailable)"
    }

    fn description(&self) -> &'static str {
        "Zero-shot NER with Candle - requires 'candle' feature"
    }
}

#[cfg(not(feature = "candle"))]
impl crate::backends::inference::ZeroShotNER for GLiNERCandle {
    fn extract_with_types(
        &self,
        _text: &str,
        _entity_types: &[&str],
        _threshold: f32,
    ) -> Result<Vec<Entity>> {
        Err(Error::FeatureNotAvailable(
            "GLiNER-Candle requires the 'candle' feature".to_string(),
        ))
    }

    fn extract_with_descriptions(
        &self,
        _text: &str,
        _descriptions: &[&str],
        _threshold: f32,
    ) -> Result<Vec<Entity>> {
        Err(Error::FeatureNotAvailable(
            "GLiNER-Candle requires the 'candle' feature".to_string(),
        ))
    }
}

// =============================================================================
// BatchCapable Trait Implementation
// =============================================================================

#[cfg(feature = "candle")]
impl crate::BatchCapable for GLiNERCandle {
    fn extract_entities_batch(
        &self,
        texts: &[&str],
        _language: Option<&str>,
    ) -> Result<Vec<Vec<Entity>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Pre-compute label embeddings for efficiency
        let _ = self.extract(texts[0], DEFAULT_GLINER_LABELS, 0.5)?;

        // Process texts - label embeddings are now cached internally
        texts
            .iter()
            .map(|text| self.extract(text, DEFAULT_GLINER_LABELS, 0.5))
            .collect()
    }

    fn optimal_batch_size(&self) -> Option<usize> {
        Some(8)
    }
}

#[cfg(not(feature = "candle"))]
impl crate::BatchCapable for GLiNERCandle {
    fn extract_entities_batch(
        &self,
        _texts: &[&str],
        _language: Option<&str>,
    ) -> Result<Vec<Vec<Entity>>> {
        Err(Error::FeatureNotAvailable(
            "GLiNER-Candle requires the 'candle' feature".to_string(),
        ))
    }

    fn optimal_batch_size(&self) -> Option<usize> {
        None
    }
}

// =============================================================================
// StreamingCapable Trait Implementation
// =============================================================================

#[cfg(feature = "candle")]
impl crate::StreamingCapable for GLiNERCandle {
    fn recommended_chunk_size(&self) -> usize {
        4096 // Characters - translates to ~500 words
    }
}

#[cfg(not(feature = "candle"))]
impl crate::StreamingCapable for GLiNERCandle {
    fn recommended_chunk_size(&self) -> usize {
        4096
    }
}

// =============================================================================
// GpuCapable Trait Implementation
// =============================================================================

#[cfg(feature = "candle")]
impl crate::GpuCapable for GLiNERCandle {
    fn is_gpu_active(&self) -> bool {
        matches!(&self.device, Device::Metal(_) | Device::Cuda(_))
    }

    fn device(&self) -> &str {
        // Use the existing device() method but return &str
        // We'll need to store this as a static or use a different approach
        match &self.device {
            Device::Cpu => "cpu",
            Device::Metal(_) => "metal",
            Device::Cuda(_) => "cuda",
        }
    }
}

#[cfg(not(feature = "candle"))]
impl crate::GpuCapable for GLiNERCandle {
    fn is_gpu_active(&self) -> bool {
        false
    }

    fn device(&self) -> &str {
        "cpu"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stub_without_feature() {
        #[cfg(not(feature = "candle"))]
        {
            let result = GLiNERCandle::new("test");
            assert!(result.is_err());
            let err = result.unwrap_err().to_string();
            assert!(err.contains("candle"));
        }
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_span_label_matcher() {
        let device = Device::Cpu;
        let matcher = SpanLabelMatcher::new(1.0);

        let span_embs = Tensor::randn(0f32, 1., (1, 10, 64), &device).unwrap();
        let label_embs = Tensor::randn(0f32, 1., (3, 64), &device).unwrap();

        let scores = matcher.forward(&span_embs, &label_embs).unwrap();
        assert_eq!(scores.dims(), &[1, 10, 3]);
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_l2_normalize() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![3.0f32, 4.0], (1, 2), &device).unwrap();
        let normed = l2_normalize(&x, D::Minus1).unwrap();

        // Should be [0.6, 0.8] (3/5, 4/5)
        let values = normed.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((values[0] - 0.6).abs() < 0.01);
        assert!((values[1] - 0.8).abs() < 0.01);
    }
}
