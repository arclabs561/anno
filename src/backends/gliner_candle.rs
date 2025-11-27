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
        let (batch_size, _seq_len, _hidden) = token_embeddings.dims3()
            .map_err(|e| Error::Parse(format!("token_embeddings dims: {}", e)))?;
        let (_, _num_spans, _) = span_indices.dims3()
            .map_err(|e| Error::Parse(format!("span_indices dims: {}", e)))?;

        let start_idx = span_indices.i((.., .., 0))?.to_dtype(DType::U32)?;
        let end_idx = span_indices.i((.., .., 1))?.to_dtype(DType::U32)?;

        let mut span_embs = Vec::new();
        
        for b in 0..batch_size {
            let batch_tokens = token_embeddings.i(b)?;
            let batch_starts = start_idx.i(b)?;
            let batch_ends = end_idx.i(b)?;

            let widths = (&batch_ends - &batch_starts)?;
            let width_embs = self.width_embeddings.forward(&widths)?;

            let start_embs = batch_tokens.index_select(&batch_starts, 0)?;
            let end_embs = batch_tokens.index_select(&batch_ends, 0)?;

            let combined = Tensor::cat(&[&start_embs, &end_embs, &width_embs], D::Minus1)?;
            let span_emb = self.projection.forward(&combined)?;
            span_embs.push(span_emb);
        }

        Tensor::stack(&span_embs, 0)
            .map_err(|e| Error::Parse(format!("stack span_embs: {}", e)))
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
        self.projection.forward(label_embeddings)
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
        let label_t = label_t.unsqueeze(0)?
            .broadcast_as((batch_size, label_t.dims()[0], label_t.dims()[1]))?;
        
        let scores = span_norm.matmul(&label_t)?;
        let scaled = (scores * self.temperature)?;

        candle_nn::ops::sigmoid(&scaled)
            .map_err(|e| Error::Parse(format!("sigmoid: {}", e)))
    }
}

#[cfg(feature = "candle")]
fn l2_normalize(tensor: &Tensor, dim: D) -> Result<Tensor> {
    let norm = tensor.sqr()?.sum(dim)?.sqrt()?;
    let norm = norm.unsqueeze(D::Minus1)?;
    tensor.broadcast_div(&norm)
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
impl GLiNERCandle {
    /// Load GLiNER from HuggingFace.
    ///
    /// # Arguments
    /// * `model_id` - HuggingFace model ID (e.g., "urchade/gliner_small-v2.1")
    pub fn from_pretrained(model_id: &str) -> Result<Self> {
        use hf_hub::api::sync::Api;

        let device = best_device()?;
        
        let api = Api::new().map_err(|e| {
            Error::Retrieval(format!("HuggingFace API: {}", e))
        })?;

        let repo = api.model(model_id.to_string());

        // Download files
        let tokenizer_path = repo.get("tokenizer.json")
            .map_err(|e| Error::Retrieval(format!("tokenizer.json: {}", e)))?;
        let weights_path = repo.get("model.safetensors")
            .or_else(|_| repo.get("gliner_model.safetensors"))
            .map_err(|e| Error::Retrieval(format!("weights: {}", e)))?;
        let config_path = repo.get("config.json")
            .map_err(|e| Error::Retrieval(format!("config.json: {}", e)))?;
        
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Retrieval(format!("tokenizer: {}", e)))?;

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
            model_id, hidden_size, device
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
        let entities = self.decode_entities(
            text, &words, &word_positions, &scores, labels, threshold
        )?;

        Ok(entities)
    }

    fn encode_text(&self, words: &[&str]) -> Result<(Tensor, Vec<(usize, usize)>)> {
        // For now, encode each word and average. Full implementation would:
        // 1. Build GLiNER prompt format
        // 2. Get per-word embeddings from words_mask
        
        let text = words.join(" ");
        let (embeddings, seq_len) = self.encoder.encode(&text)?;
        
        // Reshape to [1, seq_len, hidden]
        let tensor = Tensor::from_vec(
            embeddings, 
            (1, seq_len, self.hidden_size), 
            &self.device
        ).map_err(|e| Error::Parse(format!("text tensor: {}", e)))?;

        // Build word positions
        let full_text = words.join(" ");
        let word_positions: Vec<(usize, usize)> = {
            let mut positions = Vec::new();
            let mut pos = 0;
            for word in words {
                if let Some(start) = full_text[pos..].find(word) {
                    let abs_start = pos + start;
                    positions.push((abs_start, abs_start + word.len()));
                    pos = abs_start + word.len();
                }
            }
            positions
        };

        Ok((tensor, word_positions))
    }

    fn encode_labels(&self, labels: &[&str]) -> Result<Tensor> {
        // Encode each label
        let mut all_embeddings = Vec::new();
        
        for label in labels {
            let (embeddings, seq_len) = self.encoder.encode(label)?;
            // Average pool to get single embedding
            let avg: Vec<f32> = (0..self.hidden_size)
                .map(|i| {
                    embeddings.iter()
                        .skip(i)
                        .step_by(self.hidden_size)
                        .take(seq_len)
                        .sum::<f32>() / seq_len as f32
                })
                .collect();
            all_embeddings.extend(avg);
        }

        Tensor::from_vec(
            all_embeddings,
            (labels.len(), self.hidden_size),
            &self.device
        ).map_err(|e| Error::Parse(format!("label tensor: {}", e)))
    }

    fn generate_spans(&self, num_words: usize) -> Result<Tensor> {
        let mut spans = Vec::new();
        
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
        let scores_vec = scores.flatten_all()
            .map_err(|e| Error::Parse(format!("flatten scores: {}", e)))?
            .to_vec1::<f32>()
            .map_err(|e| Error::Parse(format!("scores to vec: {}", e)))?;

        let num_labels = labels.len();
        let num_spans = scores_vec.len() / num_labels;

        let mut entities = Vec::new();
        let mut span_idx = 0;

        for start in 0..words.len() {
            for width in 0..MAX_SPAN_WIDTH.min(words.len() - start) {
                if span_idx >= num_spans {
                    break;
                }

                let end = start + width + 1;
                
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
                    if let (Some(&(start_pos, _)), Some(&(_, end_pos))) = 
                        (word_positions.get(start), word_positions.get(end - 1)) 
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

                span_idx += 1;
            }
        }

        // Remove overlapping (keep highest scoring)
        entities.sort_by(|a, b| {
            b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut filtered = Vec::new();
        for entity in entities {
            let overlaps = filtered.iter().any(|e: &Entity| {
                !(entity.end <= e.start || entity.start >= e.end)
            });
            if !overlaps {
                filtered.push(entity);
            }
        }

        filtered.sort_by_key(|e| e.start);
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

    /// Get device.
    pub fn device(&self) -> &Device {
        &self.device
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
    "person", "organization", "location", "date", "time", "money", "percent",
    "product", "event", "facility", "work_of_art", "law", "language",
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

// =============================================================================
// Non-candle stub
// =============================================================================

#[cfg(not(feature = "candle"))]
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
             Alternative: Use GLiNEROnnx with the 'onnx' feature for similar functionality.".to_string()
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
            "GLiNER-Candle requires the 'candle' feature".to_string()
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
