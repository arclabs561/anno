//! Candle-based NER implementation using HuggingFace transformers.
//!
//! Uses Candle ML framework to run transformer-based NER models from HuggingFace.
//! Supports models like BERT, RoBERTa, and other transformer architectures fine-tuned for NER.
//!
//! This provides state-of-the-art NER without requiring Python or external services.

#[cfg(test)]
#[path = "ner_candle/tests.rs"]
mod tests;

#[cfg(feature = "ml-ner-candle")]
mod candle_impl;

use crate::EntityType;
use crate::{Result};
use crate::{Entity};

/// Candle-based NER model wrapper.
///
/// Uses Candle to load and run transformer NER models from HuggingFace.
/// Models are downloaded automatically via `hf-hub` and cached locally.
///
/// # Example Models
/// - `dslim/bert-base-NER`: BERT-based NER (4 entity types: PER, ORG, LOC, MISC)
/// - `dbmdz/bert-large-cased-finetuned-conll03-english`: CoNLL-03 fine-tuned BERT
/// - `Jean-Baptiste/roberta-large-ner-english`: RoBERTa-based NER
#[cfg(feature = "ml-ner-candle")]
pub struct CandleNER {
    model: candle_impl::CandleModel,
    model_name: String,
}

#[cfg(feature = "ml-ner-candle")]
impl CandleNER {
    /// Create a new Candle NER model.
    ///
    /// # Arguments
    /// * `model_name` - HuggingFace model identifier (e.g., "dslim/bert-base-NER")
    ///
    /// # Returns
    /// Candle NER model instance
    ///
    /// # Errors
    /// Returns error if model download or loading fails
    pub fn new(model_name: &str) -> Result<Self> {
        let model = candle_impl::CandleModel::new(model_name)?;
        Ok(Self {
            model,
            model_name: model_name.to_string(),
        })
    }

    /// Extract entities from text using the loaded transformer model.
    ///
    /// # Arguments
    /// * `text` - Text to extract entities from
    /// * `language` - Optional language hint (currently unused, model determines language)
    ///
    /// # Returns
    /// Vector of NER entities with positions, types, and confidence scores
    pub fn extract_entities(&self, text: &str, _language: Option<&str>) -> Result<Vec<Entity>> {
        self.model.extract_entities(text)
    }

    /// Get the model name.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[cfg(not(feature = "ml-ner-candle"))]
pub struct CandleNER;

#[cfg(not(feature = "ml-ner-candle"))]
impl CandleNER {
    pub fn new(_model_name: &str) -> Result<Self> {
        Err(crate::Error::Parse(
            "Candle NER support requires 'ml-ner-candle' feature. Add to Cargo.toml: ml-ner-candle = [\"dep:candle-core\", \"dep:candle-nn\", \"dep:candle-transformers\", \"dep:tokenizers\", \"dep:hf-hub\"]".to_string()
        ))
    }

    pub fn extract_entities(&self, _text: &str, _language: Option<&str>) -> Result<Vec<Entity>> {
        Err(crate::Error::Parse(
            "Candle NER support requires 'ml-ner-candle' feature".to_string(),
        ))
    }

    pub fn model_name(&self) -> &str {
        "candle-not-enabled"
    }
}
