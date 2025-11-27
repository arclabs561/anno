//! GLiNER implementation using Candle (pure Rust ML) with Metal support.
//!
//! This provides native Rust GLiNER inference with GPU acceleration on Apple Silicon.
//!
//! # Architecture
//!
//! GLiNER consists of:
//! 1. **Encoder** (BERT/DeBERTa/ModernBERT): Transforms text to contextual embeddings
//! 2. **Span Matching Head**: Computes similarity between span embeddings and label embeddings
//!
//! # Metal Support
//!
//! On macOS with Apple Silicon, this uses Metal for GPU acceleration:
//! ```bash
//! cargo build --features candle,metal
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use anno::backends::gliner_candle::GLiNERCandle;
//!
//! let model = GLiNERCandle::new("answerdotai/ModernBERT-base")?;
//! let entities = model.extract(
//!     "Steve Jobs founded Apple in California.",
//!     &["person", "organization", "location"],
//!     0.5,
//! )?;
//! ```

#![allow(dead_code)]
#![allow(unused_imports)]

use crate::{Entity, EntityType, Error, Result};

#[cfg(feature = "candle")]
use {
    candle_core::{DType, Device, IndexOp, Module, Tensor, D},
    candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, VarBuilder},
    std::collections::HashMap,
    std::sync::Mutex,
};

#[cfg(feature = "candle")]
use tokenizers::Tokenizer;

/// Maximum span width for entity candidates (from GLiNER config).
const MAX_SPAN_WIDTH: usize = 12;

/// Special tokens for GLiNER.
const TOKEN_ENT: u32 = 128002;
const TOKEN_SEP: u32 = 128003;

// =============================================================================
// Device Selection (Metal/CUDA/CPU)
// =============================================================================

/// Get the best available device for computation.
#[cfg(feature = "candle")]
pub fn best_device() -> Result<Device> {
    // Try Metal first (Apple Silicon)
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        if let Ok(device) = Device::new_metal(0) {
            log::info!("[GLiNER-Candle] Using Metal device");
            return Ok(device);
        }
    }

    // Try CUDA (NVIDIA)
    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = Device::new_cuda(0) {
            log::info!("[GLiNER-Candle] Using CUDA device");
            return Ok(device);
        }
    }

    // Fallback to CPU
    log::info!("[GLiNER-Candle] Using CPU device");
    Ok(Device::Cpu)
}

// =============================================================================
// Span Representation Layer
// =============================================================================

/// Span representation layer: projects token embeddings to span embeddings.
///
/// For a span [i, j], combines:
/// - Start token embedding
/// - End token embedding  
/// - Width embedding (learned)
#[cfg(feature = "candle")]
pub struct SpanRepLayer {
    /// Projects concatenated [start, end, width] to span embedding
    projection: Linear,
    /// Width embeddings (learned, indexed by span width)
    width_embeddings: Embedding,
    /// Hidden size
    hidden_size: usize,
}

#[cfg(feature = "candle")]
impl SpanRepLayer {
    /// Create a new span representation layer.
    pub fn new(hidden_size: usize, max_width: usize, vb: VarBuilder) -> Result<Self> {
        // Input: start (hidden) + end (hidden) + width_emb (hidden/4)
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

    /// Compute span representations from token embeddings.
    ///
    /// # Arguments
    /// * `token_embeddings` - Shape: [batch, seq_len, hidden]
    /// * `span_indices` - Shape: [batch, num_spans, 2] (start, end indices)
    ///
    /// # Returns
    /// Span embeddings with shape: [batch, num_spans, hidden]
    pub fn forward(&self, token_embeddings: &Tensor, span_indices: &Tensor) -> Result<Tensor> {
        let (batch_size, _seq_len, hidden) = token_embeddings.dims3()
            .map_err(|e| Error::Parse(format!("token_embeddings dims: {}", e)))?;
        let (_, num_spans, _) = span_indices.dims3()
            .map_err(|e| Error::Parse(format!("span_indices dims: {}", e)))?;

        let device = token_embeddings.device();

        // Extract start and end indices
        let start_idx = span_indices.i((.., .., 0))?
            .to_dtype(DType::U32)?;
        let end_idx = span_indices.i((.., .., 1))?
            .to_dtype(DType::U32)?;

        // Gather start and end embeddings
        // For simplicity, process batch by batch
        let mut span_embs = Vec::new();
        
        for b in 0..batch_size {
            let batch_tokens = token_embeddings.i(b)?; // [seq_len, hidden]
            let batch_starts = start_idx.i(b)?; // [num_spans]
            let batch_ends = end_idx.i(b)?; // [num_spans]

            // Compute widths
            let widths = (&batch_ends - &batch_starts)?;
            let width_embs = self.width_embeddings.forward(&widths)?; // [num_spans, width_emb_size]

            // Gather start embeddings
            let start_embs = batch_tokens.index_select(&batch_starts, 0)?; // [num_spans, hidden]
            
            // Gather end embeddings
            let end_embs = batch_tokens.index_select(&batch_ends, 0)?; // [num_spans, hidden]

            // Concatenate [start, end, width]
            let combined = Tensor::cat(&[&start_embs, &end_embs, &width_embs], D::Minus1)?;

            // Project to span embedding
            let span_emb = self.projection.forward(&combined)?;
            span_embs.push(span_emb);
        }

        // Stack batches
        Tensor::stack(&span_embs, 0)
            .map_err(|e| Error::Parse(format!("stack span_embs: {}", e)))
    }
}

// =============================================================================
// Label Encoder
// =============================================================================

/// Encodes entity type labels into embeddings.
///
/// Uses the same transformer encoder as text, ensuring labels and text
/// live in the same embedding space (key insight of GLiNER bi-encoder).
#[cfg(feature = "candle")]
pub struct LabelEncoder {
    /// Shared projection layer
    projection: Linear,
}

#[cfg(feature = "candle")]
impl LabelEncoder {
    /// Create a new label encoder.
    pub fn new(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let projection = linear(hidden_size, hidden_size, vb.pp("label_projection"))
            .map_err(|e| Error::Retrieval(format!("LabelEncoder projection: {}", e)))?;

        Ok(Self { projection })
    }

    /// Encode label embeddings.
    ///
    /// # Arguments
    /// * `label_embeddings` - Raw embeddings from transformer [num_labels, hidden]
    ///
    /// # Returns
    /// Projected label embeddings [num_labels, hidden]
    pub fn forward(&self, label_embeddings: &Tensor) -> Result<Tensor> {
        self.projection.forward(label_embeddings)
            .map_err(|e| Error::Parse(format!("label projection: {}", e)))
    }
}

// =============================================================================
// Span-Label Matcher
// =============================================================================

/// Computes matching scores between span embeddings and label embeddings.
///
/// This is the core of GLiNER: treating NER as a bi-encoder matching problem.
#[cfg(feature = "candle")]
pub struct SpanLabelMatcher {
    /// Temperature for softmax (learned)
    temperature: f64,
}

#[cfg(feature = "candle")]
impl SpanLabelMatcher {
    /// Create a new span-label matcher.
    pub fn new(temperature: f64) -> Self {
        Self { temperature }
    }

    /// Compute matching scores between spans and labels.
    ///
    /// # Arguments
    /// * `span_embeddings` - Shape: [batch, num_spans, hidden]
    /// * `label_embeddings` - Shape: [num_labels, hidden]
    ///
    /// # Returns
    /// Matching scores: [batch, num_spans, num_labels]
    pub fn forward(&self, span_embeddings: &Tensor, label_embeddings: &Tensor) -> Result<Tensor> {
        // Normalize embeddings for cosine similarity
        let span_norm = l2_normalize(span_embeddings, D::Minus1)?;
        let label_norm = l2_normalize(label_embeddings, D::Minus1)?;

        // Compute cosine similarity via matmul
        // span_norm: [batch, num_spans, hidden]
        // label_norm: [num_labels, hidden]
        // We need: [batch, num_spans, num_labels]
        
        // Get batch size for broadcasting
        let batch_size = span_norm.dims()[0];
        
        // Transpose labels: [num_labels, hidden] -> [hidden, num_labels]
        let label_t = label_norm.t()?;
        
        // Unsqueeze to [1, hidden, num_labels] then broadcast to [batch, hidden, num_labels]
        let label_t = label_t.unsqueeze(0)?
            .broadcast_as((batch_size, label_t.dims()[0], label_t.dims()[1]))?;
        
        // Batched matmul: [batch, num_spans, hidden] @ [batch, hidden, num_labels]
        let scores = span_norm.matmul(&label_t)?; // [batch, num_spans, num_labels]

        // Apply temperature scaling
        let scaled = (scores * self.temperature)?;

        // Sigmoid to get probability scores
        candle_nn::ops::sigmoid(&scaled)
            .map_err(|e| Error::Parse(format!("sigmoid: {}", e)))
    }
}

/// L2 normalize along a dimension.
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

/// GLiNER implementation using Candle with Metal/CUDA support.
///
/// # Device Selection
///
/// Automatically selects the best available device:
/// 1. Metal (Apple Silicon)
/// 2. CUDA (NVIDIA GPU)
/// 3. CPU (fallback)
///
/// # Architecture
///
/// ```text
/// Text Input     Label Input
///     |              |
///     v              v
/// [Tokenizer]   [Tokenizer]
///     |              |
///     v              v
/// [Transformer Encoder] (shared)
///     |              |
///     v              v
/// [SpanRepLayer]  [LabelEncoder]
///     |              |
///     +------+-------+
///            |
///            v
///     [SpanLabelMatcher]
///            |
///            v
///       [Entities]
/// ```
#[cfg(feature = "candle")]
pub struct GLiNERCandle {
    /// Tokenizer
    tokenizer: Tokenizer,
    /// Device (Metal/CUDA/CPU)
    device: Device,
    /// Model name
    model_name: String,
    /// Hidden size
    hidden_size: usize,
    /// Span representation layer
    span_rep: SpanRepLayer,
    /// Label encoder
    label_encoder: LabelEncoder,
    /// Span-label matcher
    matcher: SpanLabelMatcher,
    // Note: Full transformer encoder would go here
    // For now, this is a skeleton showing the architecture
}

#[cfg(feature = "candle")]
impl GLiNERCandle {
    /// Create a new GLiNER model with Candle backend.
    ///
    /// # Arguments
    /// * `model_name` - HuggingFace model ID (e.g., "answerdotai/ModernBERT-base")
    ///
    /// # Device Selection
    ///
    /// Automatically uses:
    /// - Metal on Apple Silicon (with `metal` feature)
    /// - CUDA on NVIDIA GPUs (with `cuda` feature)
    /// - CPU as fallback
    pub fn new(model_name: &str) -> Result<Self> {
        use hf_hub::api::sync::Api;

        let device = best_device()?;
        
        let api = Api::new().map_err(|e| {
            Error::Retrieval(format!("HuggingFace API init failed: {}", e))
        })?;

        let repo = api.model(model_name.to_string());

        // Download tokenizer
        let tokenizer_path = repo.get("tokenizer.json")
            .map_err(|e| Error::Retrieval(format!("tokenizer.json download: {}", e)))?;
        
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Retrieval(format!("tokenizer load: {}", e)))?;

        // For a full implementation, we would:
        // 1. Download model.safetensors
        // 2. Load weights into VarBuilder
        // 3. Build transformer encoder (BERT/DeBERTa/ModernBERT)
        //
        // This skeleton shows the GLiNER-specific components.
        // The transformer encoder follows standard Candle patterns (see candle-transformers).

        let hidden_size = 768; // Base model

        // Create VarBuilder (placeholder - would load from safetensors)
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let span_rep = SpanRepLayer::new(hidden_size, MAX_SPAN_WIDTH, vb.pp("span_rep"))?;
        let label_encoder = LabelEncoder::new(hidden_size, vb.pp("label_encoder"))?;
        let matcher = SpanLabelMatcher::new(1.0);

        Ok(Self {
            tokenizer,
            device,
            model_name: model_name.to_string(),
            hidden_size,
            span_rep,
            label_encoder,
            matcher,
        })
    }

    /// Extract entities from text using GLiNER.
    ///
    /// # Arguments
    /// * `text` - Input text
    /// * `labels` - Entity types to detect (e.g., ["person", "organization"])
    /// * `threshold` - Confidence threshold (0.0-1.0)
    pub fn extract(&self, text: &str, labels: &[&str], threshold: f32) -> Result<Vec<Entity>> {
        // This is a skeleton showing the inference pipeline.
        // A full implementation would:
        //
        // 1. Tokenize text and labels
        // 2. Run through transformer encoder
        // 3. Generate span candidates
        // 4. Compute span embeddings via SpanRepLayer
        // 5. Compute label embeddings via LabelEncoder
        // 6. Match spans to labels via SpanLabelMatcher
        // 7. Decode high-confidence matches to entities

        log::warn!("[GLiNER-Candle] Full implementation pending - skeleton only");
        
        // Placeholder: return empty for now
        Ok(vec![])
    }

    /// Get the device being used.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the model name.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

// =============================================================================
// Model Trait Implementation
// =============================================================================

/// Default entity types for zero-shot GLiNER when used via the Model trait.
#[cfg(feature = "candle")]
const DEFAULT_GLINER_LABELS: &[&str] = &[
    "person", "organization", "location", "date", "time", "money", "percent",
    "product", "event", "facility", "work_of_art", "law", "language",
];

#[cfg(feature = "candle")]
impl crate::Model for GLiNERCandle {
    fn extract_entities(&self, text: &str, _language: Option<&str>) -> Result<Vec<Entity>> {
        // Use default labels for the Model trait interface
        // For custom labels, use the extract() method directly
        self.extract(text, DEFAULT_GLINER_LABELS, 0.5)
    }

    fn supported_types(&self) -> Vec<EntityType> {
        // GLiNER supports any type via zero-shot - return the defaults
        DEFAULT_GLINER_LABELS
            .iter()
            .map(|label| EntityType::Custom {
                name: (*label).to_string(),
                category: crate::entity::EntityCategory::Misc,
            })
            .collect()
    }

    fn is_available(&self) -> bool {
        true // If we got this far, it's available
    }

    fn name(&self) -> &'static str {
        "GLiNER-Candle"
    }

    fn description(&self) -> &'static str {
        "Zero-shot NER using GLiNER with pure Rust Candle backend (Metal/CUDA/CPU)"
    }
}

// =============================================================================
// Non-candle stub
// =============================================================================

#[cfg(not(feature = "candle"))]
pub struct GLiNERCandle;

#[cfg(not(feature = "candle"))]
impl GLiNERCandle {
    /// Create a new GLiNER model (stub - requires candle feature).
    pub fn new(_model_name: &str) -> Result<Self> {
        Err(Error::InvalidInput(
            "GLiNER-Candle requires the 'candle' feature. \
             Build with: cargo build --features candle".to_string()
        ))
    }
}

#[cfg(not(feature = "candle"))]
impl crate::Model for GLiNERCandle {
    fn extract_entities(&self, _text: &str, _language: Option<&str>) -> Result<Vec<Entity>> {
        Err(Error::InvalidInput(
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
        "GLiNER with Candle backend - requires 'candle' feature"
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
        }
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_span_label_matcher() {
        let device = Device::Cpu;
        let matcher = SpanLabelMatcher::new(1.0);

        // Create dummy embeddings
        let span_embs = Tensor::randn(0f32, 1., (1, 10, 64), &device).unwrap();
        let label_embs = Tensor::randn(0f32, 1., (3, 64), &device).unwrap();

        let scores = matcher.forward(&span_embs, &label_embs).unwrap();
        
        // Check output shape: [batch=1, num_spans=10, num_labels=3]
        assert_eq!(scores.dims(), &[1, 10, 3]);
        
        // Scores should be in [0, 1] after sigmoid
        // Flatten and check bounds
        let flat_scores = scores.flatten_all().unwrap();
        let min_val = flat_scores.min(0).unwrap().to_scalar::<f32>().unwrap();
        let max_val = flat_scores.max(0).unwrap().to_scalar::<f32>().unwrap();
        assert!(min_val >= 0.0, "min {} should be >= 0", min_val);
        assert!(max_val <= 1.0, "max {} should be <= 1", max_val);
    }
}

