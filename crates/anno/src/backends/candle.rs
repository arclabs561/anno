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

use crate::{Entity, EntityCategory, EntityType, Error, Language, Model, Result};

#[cfg(feature = "candle")]
use {
    super::encoder_candle::{CandleEncoder, CandleTextEncoder},
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
    /// Automatically loads `.env` for HF_TOKEN if present.
    ///
    /// # Arguments
    /// * `model_id` - HuggingFace model ID (e.g., "dslim/bert-base-NER")
    ///
    /// # Note
    /// Some older models (like dslim/bert-base-NER) only have vocab.txt, not tokenizer.json.
    /// This function will try the provided model, and if it fails due to missing tokenizer.json,
    /// it will automatically try alternative models that have tokenizer.json.
    pub fn from_pretrained(model_id: &str) -> Result<Self> {
        let device = super::encoder_candle::best_device()?;

        let api = crate::backends::hf_loader::hf_api()?;
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

        // Get encoder config from the model's config.json (not defaults!)
        let encoder_config = CandleEncoder::parse_config(&config_str)?;

        // Get label mapping
        let id2label = Self::parse_labels(&config_json)?;
        let num_labels = id2label.len();

        // Load weights
        // SAFETY: VarBuilder::from_mmaped_safetensors uses unsafe internally for memory mapping.
        // The weights_path is validated to exist before this call, and the safetensors format
        // is validated by the library. This is a safe FFI boundary.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                .map_err(|e| Error::Retrieval(format!("safetensors: {}", e)))?
        };

        // Build encoder from shared VarBuilder (encoder weights are under "bert" prefix)
        // Load tokenizer for encoder
        let encoder_tokenizer = if tokenizer_path.ends_with("tokenizer.json") {
            Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| Error::Retrieval(format!("tokenizer: {}", e)))?
        } else if tokenizer_path.ends_with("vocab.txt") {
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
            // Use cased normalizer for NER - case is important for detecting names!
            // BertNormalizer::default() lowercases which breaks NER
            tokenizer_impl.with_normalizer(Some(BertNormalizer::new(
                false, // clean_text
                true,  // handle_chinese_chars
                None,  // strip_accents - None means don't strip
                false, // lowercase - CRITICAL: keep case for NER!
            )));
            tokenizer_impl.with_pre_tokenizer(Some(BertPreTokenizer));
            tokenizer_impl.with_post_processor(Some(BertProcessing::default()));

            tokenizer_impl
        } else {
            return Err(Error::Retrieval("Unsupported tokenizer format".to_string()));
        };

        let encoder = CandleEncoder::from_vb(
            encoder_config.clone(),
            vb.pp("bert"),
            encoder_tokenizer,
            device.clone(),
        )?;

        // Build classifier head (classifier weights are under "classifier" prefix)
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

        // Get encoder output with token offsets
        let (embeddings, seq_len, offsets) = self.encoder.encode_with_offsets(text)?;

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

        // Decode BIO to entities using token offsets (like BertNEROnnx)
        self.decode_with_offsets(text, &predictions, &offsets)
    }

    /// Decode BIO predictions using token offsets.
    /// This properly handles subword tokenization by using the exact character offsets.
    fn decode_with_offsets(
        &self,
        text: &str,
        predictions: &[u32],
        offsets: &[(usize, usize)],
    ) -> Result<Vec<Entity>> {
        let mut entities = Vec::with_capacity(16);
        let mut current_entity: Option<(usize, usize, String, f64)> = None;
        // `tokenizers::Encoding::get_offsets()` are byte offsets. `Entity` expects char offsets.
        // Build once so conversion is O(1) per entity.
        let span_converter = crate::offset::SpanConverter::new(text);

        for (token_idx, &pred) in predictions.iter().enumerate() {
            if token_idx >= offsets.len() {
                break;
            }

            let (byte_start, byte_end) = offsets[token_idx];

            // Skip special tokens (offset 0,0)
            if byte_start == byte_end {
                // Finalize current entity if any
                if let Some((start, end, etype, conf)) = current_entity.take() {
                    if let Some(e) = self.create_entity_from_offsets(
                        text,
                        &span_converter,
                        start,
                        end,
                        &etype,
                        conf,
                    ) {
                        entities.push(e);
                    }
                }
                continue;
            }

            let label = self
                .id2label
                .get(pred as usize)
                .map(|s| s.as_str())
                .unwrap_or("O");

            if label == "O" {
                // Outside label - finalize current entity
                if let Some((start, end, etype, conf)) = current_entity.take() {
                    if let Some(e) = self.create_entity_from_offsets(
                        text,
                        &span_converter,
                        start,
                        end,
                        &etype,
                        conf,
                    ) {
                        entities.push(e);
                    }
                }
            } else if label.starts_with("B-") {
                // Begin new entity - finalize previous if any
                if let Some((start, end, etype, conf)) = current_entity.take() {
                    if let Some(e) = self.create_entity_from_offsets(
                        text,
                        &span_converter,
                        start,
                        end,
                        &etype,
                        conf,
                    ) {
                        entities.push(e);
                    }
                }
                let entity_type = label.strip_prefix("B-").unwrap_or("MISC");
                current_entity = Some((byte_start, byte_end, entity_type.to_string(), 0.9));
            } else if label.starts_with("I-") {
                // Inside continuation
                let entity_type = label.strip_prefix("I-").unwrap_or("MISC");
                if let Some((start, _, ref etype, conf)) = current_entity {
                    if entity_type == etype {
                        // Continue entity
                        current_entity = Some((start, byte_end, etype.clone(), conf));
                    } else {
                        // Type mismatch - start new entity
                        if let Some((s, e, t, c)) = current_entity.take() {
                            if let Some(ent) =
                                self.create_entity_from_offsets(text, &span_converter, s, e, &t, c)
                            {
                                entities.push(ent);
                            }
                        }
                        current_entity = Some((byte_start, byte_end, entity_type.to_string(), 0.9));
                    }
                } else {
                    // No current entity - start new one (treat I- as B-)
                    current_entity = Some((byte_start, byte_end, entity_type.to_string(), 0.9));
                }
            }
        }

        // Flush final entity
        if let Some((start, end, etype, conf)) = current_entity.take() {
            if let Some(e) =
                self.create_entity_from_offsets(text, &span_converter, start, end, &etype, conf)
            {
                entities.push(e);
            }
        }

        Ok(entities)
    }

    /// Create an entity from tokenizer byte offsets, converting to character offsets for `Entity`.
    fn create_entity_from_offsets(
        &self,
        text: &str,
        span_converter: &crate::offset::SpanConverter,
        byte_start: usize,
        byte_end: usize,
        entity_type: &str,
        confidence: f64,
    ) -> Option<Entity> {
        if byte_start >= byte_end || byte_end > text.len() {
            return None;
        }

        // Extract text using byte offsets (tokenizers use byte indices in Rust).
        let entity_text = text.get(byte_start..byte_end)?;

        // Skip empty or whitespace-only entities
        if entity_text.trim().is_empty() {
            return None;
        }

        let char_start = span_converter.byte_to_char(byte_start);
        let char_end = span_converter.byte_to_char(byte_end);

        let etype = match entity_type.to_uppercase().as_str() {
            "PER" | "PERSON" => EntityType::Person,
            "ORG" | "ORGANIZATION" => EntityType::Organization,
            "LOC" | "LOCATION" | "GPE" => EntityType::Location,
            "DATE" => EntityType::Date,
            "TIME" => EntityType::Time,
            "MONEY" => EntityType::Money,
            "PERCENT" => EntityType::Percent,
            "MISC" => EntityType::custom("MISC", EntityCategory::Misc),
            other => EntityType::custom(other, EntityCategory::Misc),
        };

        Some(Entity::new(
            entity_text.trim().to_string(),
            etype,
            char_start,
            char_end,
            confidence,
        ))
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
    fn extract_entities(&self, text: &str, _language: Option<Language>) -> Result<Vec<Entity>> {
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
                    other => EntityType::custom(other, EntityCategory::Misc),
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

    fn version(&self) -> String {
        format!("candle-ner-{}-{}", self.model_name, self.device())
    }

    fn capabilities(&self) -> crate::ModelCapabilities {
        crate::ModelCapabilities::default()
    }
}

// =============================================================================
// Non-candle stub
// =============================================================================

crate::backends::macros::define_feature_stub! {
    struct CandleNER;
    feature = "candle";
    name = "CandleNER (unavailable)";
    description = "BERT NER with Candle - requires 'candle' feature";
    error_msg = "CandleNER requires the 'candle' feature";
    methods {
        /// Load from pretrained (requires candle feature).
        pub fn from_pretrained(_model_id: &str) -> crate::Result<Self> {
            Self::new("")
        }

        /// Get model name.
        pub fn model_name(&self) -> &str {
            "candle-disabled"
        }
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
        assert!(CONLL_LABELS.contains(&"B-PER"));
    }
}
