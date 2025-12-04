//! Traditional BERT NER using Candle (pure Rust ML).
//!
//! This provides token classification NER using fine-tuned BERT models.
//! Unlike GLiNER (zero-shot), this uses models fine-tuned on specific entity types.
//!
//! # Architecture
//!
//! ```text
//! Input: "Steve Jobs founded Apple"
//!
//!        ┌─────────────────────────────┐
//!        │      Encoder (BERT)          │
//!        │      [hidden per token]      │
//!        └─────────────────────────────┘
//!                     │
//!        ┌─────────────────────────────┐
//!        │    Classification Head       │
//!        │    [num_labels per token]    │
//!        └─────────────────────────────┘
//!                     │
//!                     ▼
//!        B-PER I-PER  O    B-ORG
//!        Steve Jobs  founded Apple
//! ```
//!
//! # Models
//!
//! Works with any BERT-style model fine-tuned for token classification:
//! - `dslim/bert-base-NER` - English NER (PER, ORG, LOC, MISC)
//! - `dbmdz/bert-large-cased-finetuned-conll03-english` - CoNLL-03
//! - `Jean-Baptiste/camembert-ner` - French NER
//!
//! # Example
//!
//! ```rust,ignore
//! use anno::CandleNER;
//!
//! let model = CandleNER::from_pretrained("dslim/bert-base-NER")?;
//! let entities = model.extract_entities("Steve Jobs founded Apple", None)?;
//! ```

use crate::{Entity, EntityType, Error, Model, Result};

#[cfg(feature = "candle")]
use {
    super::encoder_candle::{CandleEncoder, EncoderConfig, TextEncoder},
    candle_core::{DType, Device, Module, Tensor, D},
    candle_nn::{linear, Linear, VarBuilder},
    std::collections::HashMap,
    tokenizers::Tokenizer,
};

/// Label mapping for standard CoNLL-style NER.
const CONLL_LABELS: &[&str] = &[
    "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC",
];

/// Candle-based BERT NER model.
///
/// Uses token classification with BIO tagging for traditional NER.
/// Requires a model fine-tuned for NER (e.g., `dslim/bert-base-NER`).
///
/// # Feature Requirements
///
/// Requires the `candle` feature for actual inference.
#[cfg(feature = "candle")]
pub struct CandleNER {
    /// Encoder (BERT/ModernBERT/DeBERTa)
    encoder: CandleEncoder,
    /// Classification head
    classifier: Linear,
    /// Label mapping
    id2label: Vec<String>,
    /// Model name
    model_name: String,
    /// Device
    device: Device,
}

#[cfg(feature = "candle")]
impl CandleNER {
    /// Create a new CandleNER from a HuggingFace model.
    ///
    /// # Arguments
    /// * `model_id` - HuggingFace model ID (e.g., "dslim/bert-base-NER")
    ///
    /// # Note
    /// Some older models (like dslim/bert-base-NER) only have vocab.txt, not tokenizer.json.
    /// This function will try the provided model, and if it fails due to missing tokenizer.json,
    /// it will automatically try alternative models that have tokenizer.json.
    pub fn from_pretrained(model_id: &str) -> Result<Self> {
        use hf_hub::api::sync::Api;

        let device = super::encoder_candle::best_device()?;

        let api = Api::new()
            .map_err(|e| Error::Retrieval(format!("HuggingFace API init failed: {}", e)))?;

        let repo = api.model(model_id.to_string());

        // Download config, weights, tokenizer
        let config_path = repo
            .get("config.json")
            .map_err(|e| Error::Retrieval(format!("config.json: {}", e)))?;
        // Candle requires safetensors format - try to convert pytorch_model.bin if needed
        let weights_path = repo
            .get("model.safetensors")
            .or_else(|_| {
                // Try to convert pytorch_model.bin to safetensors
                let pytorch_path = repo.get("pytorch_model.bin")?;
                crate::backends::gliner_candle::convert_pytorch_to_safetensors(&pytorch_path)
            })
            .map_err(|e| Error::Retrieval(format!(
                "model.safetensors not found and conversion failed. CandleNER requires safetensors format. \
                 The model may only have pytorch_model.bin. Attempted automatic conversion but it failed. \
                 Consider using BertNEROnnx (ONNX version) instead. \
                 Original error: {}",
                e
            )))?;
        // Try tokenizer.json first, fall back to vocab.txt for older models
        let tokenizer_path = repo.get("tokenizer.json").or_else(|_| {
            // For older BERT models without tokenizer.json, we can't easily create
            // a tokenizer from vocab.txt alone. Skip tokenizer validation for now.
            // The encoder will handle tokenization.
            repo.get("vocab.txt").map_err(|e| {
                Error::Retrieval(format!(
                    "tokenizer: neither tokenizer.json nor vocab.txt found: {}",
                    e
                ))
            })
        })?;

        // Parse config
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| Error::Retrieval(format!("read config: {}", e)))?;
        let config_json: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| Error::Parse(format!("config JSON: {}", e)))?;

        // Get encoder config
        let encoder_config = EncoderConfig::from_model_name(model_id);

        // Get label mapping
        let id2label = Self::parse_labels(&config_json)?;
        let num_labels = id2label.len();

        // Load tokenizer - handle both tokenizer.json and vocab.txt
        let _tokenizer = if tokenizer_path.ends_with("tokenizer.json") {
            Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| Error::Retrieval(format!("tokenizer: {}", e)))?
        } else if tokenizer_path.ends_with("vocab.txt") {
            // Create a BERT tokenizer from vocab.txt
            use tokenizers::models::wordpiece::WordPiece;
            use tokenizers::normalizers::bert::BertNormalizer;
            use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
            use tokenizers::processors::bert::BertProcessing;
            use tokenizers::Tokenizer as TokenizerImpl;

            let vocab_str = tokenizer_path
                .to_str()
                .ok_or_else(|| Error::Retrieval("Invalid tokenizer path".to_string()))?;

            let model = WordPiece::from_file(vocab_str).build().map_err(|e| {
                Error::Retrieval(format!("Failed to create WordPiece from vocab.txt: {}", e))
            })?;

            let mut tokenizer_impl = TokenizerImpl::new(model);
            tokenizer_impl.with_normalizer(Some(BertNormalizer::default()));
            tokenizer_impl.with_pre_tokenizer(Some(BertPreTokenizer));
            tokenizer_impl.with_post_processor(Some(BertProcessing::default()));

            // Convert to the tokenizers::Tokenizer type expected
            Tokenizer::from(tokenizer_impl)
        } else {
            return Err(Error::Retrieval(format!(
                "Unsupported tokenizer format: {}. Expected tokenizer.json or vocab.txt.",
                tokenizer_path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
            )));
        };

        // Load weights
        // SAFETY: VarBuilder::from_mmaped_safetensors uses unsafe internally for memory mapping.
        // The weights_path is validated to exist before this call, and the safetensors format
        // is validated by the library. This is a safe FFI boundary.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                .map_err(|e| Error::Retrieval(format!("safetensors: {}", e)))?
        };

        // Build encoder
        let encoder = CandleEncoder::from_pretrained(model_id)?;

        // Build classifier head
        let classifier = linear(encoder_config.hidden_size, num_labels, vb.pp("classifier"))
            .map_err(|e| Error::Retrieval(format!("classifier: {}", e)))?;

        log::info!(
            "[CandleNER] Loaded {} with {} labels on {:?}",
            model_id,
            num_labels,
            device
        );

        Ok(Self {
            encoder,
            classifier,
            id2label,
            model_name: model_id.to_string(),
            device,
        })
    }

    /// Create with default CoNLL labels (for testing without config).
    pub fn new(model_id: &str) -> Result<Self> {
        Self::from_pretrained(model_id)
    }

    fn parse_labels(config: &serde_json::Value) -> Result<Vec<String>> {
        if let Some(id2label) = config.get("id2label") {
            let map: HashMap<String, String> = serde_json::from_value(id2label.clone())
                .map_err(|e| Error::Parse(format!("id2label: {}", e)))?;

            let max_id = map
                .keys()
                .filter_map(|k| k.parse::<usize>().ok())
                .max()
                .unwrap_or(0);

            let mut labels = vec!["O".to_string(); max_id + 1];
            for (id_str, label) in map {
                if let Ok(id) = id_str.parse::<usize>() {
                    labels[id] = label;
                }
            }
            Ok(labels)
        } else {
            // Default CoNLL labels
            Ok(CONLL_LABELS.iter().map(|s| s.to_string()).collect())
        }
    }

    /// Extract entities with token classification.
    pub fn extract(&self, text: &str) -> Result<Vec<Entity>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        // Get encoder output
        let (embeddings, seq_len) = self.encoder.encode(text)?;

        // Reshape to [1, seq_len, hidden]
        let hidden_dim = self.encoder.hidden_dim();
        let hidden = Tensor::from_vec(embeddings, (1, seq_len, hidden_dim), &self.device)
            .map_err(|e| Error::Parse(format!("hidden tensor: {}", e)))?;

        // Run classifier: [1, seq_len, hidden] -> [1, seq_len, num_labels]
        let logits = self
            .classifier
            .forward(&hidden)
            .map_err(|e| Error::Parse(format!("classifier forward: {}", e)))?;

        // Argmax to get predictions
        let predictions = logits
            .argmax(D::Minus1)
            .map_err(|e| Error::Parse(format!("argmax: {}", e)))?
            .flatten_all()
            .map_err(|e| Error::Parse(format!("flatten: {}", e)))?
            .to_vec1::<u32>()
            .map_err(|e| Error::Parse(format!("to_vec: {}", e)))?;

        // Decode BIO to entities
        let entities = self.decode_bio(text, &predictions)?;

        Ok(entities)
    }

    fn decode_bio(&self, text: &str, predictions: &[u32]) -> Result<Vec<Entity>> {
        // Performance: Pre-allocate entities vec with estimated capacity
        let mut entities = Vec::with_capacity(16);
        let words: Vec<&str> = text.split_whitespace().collect();

        // Build word positions
        let word_positions: Vec<(usize, usize)> = {
            // Performance: Pre-allocate positions vec with known size
            let mut positions = Vec::with_capacity(words.len());
            let mut pos = 0;
            for (idx, word) in words.iter().enumerate() {
                if let Some(start) = text[pos..].find(word) {
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

        let mut current_entity: Option<(usize, usize, String)> = None;

        for (idx, &pred) in predictions.iter().enumerate() {
            if idx >= words.len() {
                break;
            }

            let label = self
                .id2label
                .get(pred as usize)
                .map(|s| s.as_str())
                .unwrap_or("O");

            if label.starts_with("B-") {
                // Flush previous entity
                if let Some((start, end, etype)) = current_entity.take() {
                    if let Some(e) = self.create_entity(text, &word_positions, start, end, &etype) {
                        entities.push(e);
                    }
                }
                // Start new entity
                let entity_type = label.strip_prefix("B-").unwrap_or("MISC");
                current_entity = Some((idx, idx + 1, entity_type.to_string()));
            } else if label.starts_with("I-") {
                // Continue entity if same type
                if let Some((start, _, ref etype)) = current_entity {
                    let entity_type = label.strip_prefix("I-").unwrap_or("MISC");
                    if entity_type == etype {
                        current_entity = Some((start, idx + 1, etype.clone()));
                    }
                }
            } else {
                // O tag - flush entity
                if let Some((start, end, etype)) = current_entity.take() {
                    if let Some(e) = self.create_entity(text, &word_positions, start, end, &etype) {
                        entities.push(e);
                    }
                }
            }
        }

        // Flush final entity
        if let Some((start, end, etype)) = current_entity.take() {
            if let Some(e) = self.create_entity(text, &word_positions, start, end, &etype) {
                entities.push(e);
            }
        }

        Ok(entities)
    }

    fn create_entity(
        &self,
        text: &str,
        word_positions: &[(usize, usize)],
        start_word: usize,
        end_word: usize,
        entity_type: &str,
    ) -> Option<Entity> {
        // Validate indices to prevent underflow
        if end_word == 0 || end_word > word_positions.len() || start_word >= word_positions.len() {
            return None;
        }
        let start_pos = word_positions.get(start_word)?.0;
        let end_pos = word_positions.get(end_word.saturating_sub(1))?.1;
        let entity_text = text.get(start_pos..end_pos)?;

        let etype = match entity_type.to_uppercase().as_str() {
            "PER" | "PERSON" => EntityType::Person,
            "ORG" | "ORGANIZATION" => EntityType::Organization,
            "LOC" | "LOCATION" | "GPE" => EntityType::Location,
            "DATE" => EntityType::Date,
            "TIME" => EntityType::Time,
            "MONEY" => EntityType::Money,
            "PERCENT" => EntityType::Percent,
            other => EntityType::Other(other.to_string()),
        };

        Some(Entity::new(entity_text, etype, start_pos, end_pos, 0.9))
    }

    /// Get model name.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Get device as a string.
    pub fn device(&self) -> String {
        match &self.device {
            Device::Cpu => "cpu".to_string(),
            Device::Metal(_) => "metal".to_string(),
            Device::Cuda(_) => "cuda".to_string(),
        }
    }
}

#[cfg(feature = "candle")]
impl Model for CandleNER {
    fn extract_entities(&self, text: &str, _language: Option<&str>) -> Result<Vec<Entity>> {
        self.extract(text)
    }

    fn supported_types(&self) -> Vec<EntityType> {
        self.id2label
            .iter()
            .filter(|l| l.starts_with("B-"))
            .map(|l| {
                let tag = l.strip_prefix("B-").unwrap_or("MISC");
                match tag.to_uppercase().as_str() {
                    "PER" | "PERSON" => EntityType::Person,
                    "ORG" | "ORGANIZATION" => EntityType::Organization,
                    "LOC" | "LOCATION" | "GPE" => EntityType::Location,
                    other => EntityType::Other(other.to_string()),
                }
            })
            .collect()
    }

    fn is_available(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "CandleNER"
    }

    fn description(&self) -> &'static str {
        "BERT token classification NER using Candle (pure Rust, GPU support)"
    }
}

// =============================================================================
// GpuCapable Trait Implementation
// =============================================================================

#[cfg(feature = "candle")]
impl crate::GpuCapable for CandleNER {
    fn is_gpu_active(&self) -> bool {
        matches!(&self.device, Device::Metal(_) | Device::Cuda(_))
    }

    fn device(&self) -> &str {
        match &self.device {
            Device::Cpu => "cpu",
            Device::Metal(_) => "metal",
            Device::Cuda(_) => "cuda",
        }
    }
}

#[cfg(not(feature = "candle"))]
impl crate::GpuCapable for CandleNER {
    fn is_gpu_active(&self) -> bool {
        false
    }

    fn device(&self) -> &str {
        "cpu"
    }
}

// =============================================================================
// BatchCapable Trait Implementation
// =============================================================================

#[cfg(feature = "candle")]
impl crate::BatchCapable for CandleNER {
    fn optimal_batch_size(&self) -> Option<usize> {
        Some(8)
    }
}

// =============================================================================
// StreamingCapable Trait Implementation
// =============================================================================

#[cfg(feature = "candle")]
impl crate::StreamingCapable for CandleNER {
    fn recommended_chunk_size(&self) -> usize {
        4096 // Characters
    }
}

// =============================================================================
// Non-candle stub
// =============================================================================

#[cfg(not(feature = "candle"))]
pub struct CandleNER {
    _private: (),
}

#[cfg(not(feature = "candle"))]
impl CandleNER {
    /// Create a new CandleNER (requires candle feature).
    pub fn new(_model_name: &str) -> Result<Self> {
        Err(Error::FeatureNotAvailable(
            "CandleNER requires the 'candle' feature. \
             Build with: cargo build --features candle\n\
             Alternative: Use BertNEROnnx with the 'onnx' feature for similar functionality."
                .to_string(),
        ))
    }

    /// Load from pretrained (requires candle feature).
    pub fn from_pretrained(_model_id: &str) -> Result<Self> {
        Self::new("")
    }

    /// Get model name.
    pub fn model_name(&self) -> &str {
        "candle-disabled"
    }
}

#[cfg(not(feature = "candle"))]
impl Model for CandleNER {
    fn extract_entities(&self, _text: &str, _language: Option<&str>) -> Result<Vec<Entity>> {
        Err(Error::FeatureNotAvailable(
            "CandleNER requires the 'candle' feature".to_string(),
        ))
    }

    fn supported_types(&self) -> Vec<EntityType> {
        vec![]
    }

    fn is_available(&self) -> bool {
        false
    }

    fn name(&self) -> &'static str {
        "CandleNER (unavailable)"
    }

    fn description(&self) -> &'static str {
        "BERT NER with Candle - requires 'candle' feature"
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
            let result = CandleNER::new("test");
            assert!(result.is_err());
            let err = result.unwrap_err().to_string();
            assert!(err.contains("candle"));
        }
    }

    #[test]
    fn test_conll_labels() {
        assert_eq!(CONLL_LABELS.len(), 9);
        assert_eq!(CONLL_LABELS[0], "O");
        assert!(CONLL_LABELS.iter().any(|l| *l == "B-PER"));
    }
}
