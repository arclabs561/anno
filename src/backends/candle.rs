//! Candle-based NER implementation using HuggingFace transformers.
//!
//! Uses Candle ML framework to run transformer-based NER models from HuggingFace.
//! Supports models like BERT, RoBERTa, and other transformer architectures fine-tuned for NER.
//!
//! This provides state-of-the-art NER without requiring Python or external services.

#![allow(missing_docs)] // Stub implementation

use crate::{Entity, Result};

/// Candle-based NER model wrapper (stub, candle support not yet implemented).
pub struct CandleNER;

impl CandleNER {
    /// Create a new Candle NER model.
    pub fn new(_model_name: &str) -> Result<Self> {
        Err(crate::Error::FeatureNotAvailable(
            "Candle NER support not yet implemented".to_string(),
        ))
    }

    /// Extract entities from text.
    pub fn extract_entities(&self, _text: &str, _language: Option<&str>) -> Result<Vec<Entity>> {
        Err(crate::Error::FeatureNotAvailable(
            "Candle NER support not yet implemented".to_string(),
        ))
    }

    /// Get the model name.
    pub fn model_name(&self) -> &str {
        "candle-not-enabled"
    }
}

impl crate::Model for CandleNER {
    fn extract_entities(&self, text: &str, language: Option<&str>) -> Result<Vec<Entity>> {
        self.extract_entities(text, language)
    }

    fn supported_types(&self) -> Vec<crate::EntityType> {
        vec![
            crate::EntityType::Person,
            crate::EntityType::Organization,
            crate::EntityType::Location,
            crate::EntityType::Other("MISC".to_string()),
        ]
    }

    fn is_available(&self) -> bool {
        false // Stub, not yet implemented
    }
}
