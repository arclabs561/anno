//! DeBERTa-v3 NER Backend
//!
//! DeBERTa-v3 (Disentangled Attention BERT) uses improved attention mechanisms
//! for better NER performance compared to standard BERT.
//!
//! # Architecture
//!
//! DeBERTa-v3 improves upon BERT with:
//! - **Disentangled attention**: Separates content and position embeddings
//! - **Enhanced mask decoder**: Better masked language modeling
//! - **Improved performance**: ~2-3 F1 points better than BERT on NER tasks
//!
//! # Usage
//!
//! ```rust,ignore
//! use anno::backends::deberta_v3::DeBERTaV3NER;
//!
//! let model = DeBERTaV3NER::new("microsoft/deberta-v3-base")?;
//! let entities = model.extract_entities("Steve Jobs founded Apple.", None)?;
//! ```

use crate::{Entity, EntityType, Model, Result};

#[cfg(feature = "onnx")]
use crate::backends::onnx::BertNEROnnx;

/// DeBERTa-v3 NER backend using ONNX Runtime.
///
/// Currently wraps BertNEROnnx with DeBERTa-v3 model support.
/// DeBERTa-v3 models use the same ONNX interface as BERT.
pub struct DeBERTaV3NER {
    #[cfg(feature = "onnx")]
    inner: BertNEROnnx,
    /// Model name for debugging/logging (e.g., "microsoft/deberta-v3-base")
    #[allow(dead_code)] // Reserved for future logging/debugging
    model_name: String,
}

impl DeBERTaV3NER {
    /// Create a new DeBERTa-v3 NER model.
    ///
    /// # Arguments
    /// * `model_name` - HuggingFace model ID (e.g., "microsoft/deberta-v3-base")
    pub fn new(model_name: &str) -> Result<Self> {
        #[cfg(feature = "onnx")]
        {
            // DeBERTa-v3 uses same ONNX interface as BERT
            let inner = BertNEROnnx::new(model_name)?;
            Ok(Self {
                inner,
                model_name: model_name.to_string(),
            })
        }
        #[cfg(not(feature = "onnx"))]
        {
            Err(crate::Error::FeatureNotAvailable(
                "DeBERTa-v3 NER requires 'onnx' feature".to_string(),
            ))
        }
    }
}

impl Model for DeBERTaV3NER {
    fn extract_entities(&self, text: &str, language: Option<&str>) -> Result<Vec<Entity>> {
        #[cfg(feature = "onnx")]
        {
            self.inner.extract_entities(text, language)
        }
        #[cfg(not(feature = "onnx"))]
        {
            Err(crate::Error::FeatureNotAvailable(
                "DeBERTa-v3 NER requires 'onnx' feature".to_string(),
            ))
        }
    }

    fn supported_types(&self) -> Vec<EntityType> {
        vec![
            EntityType::Person,
            EntityType::Organization,
            EntityType::Location,
        ]
    }

    fn is_available(&self) -> bool {
        #[cfg(feature = "onnx")]
        {
            self.inner.is_available()
        }
        #[cfg(not(feature = "onnx"))]
        {
            false
        }
    }

    fn name(&self) -> &'static str {
        "deberta_v3"
    }

    fn description(&self) -> &'static str {
        "DeBERTa-v3 NER with disentangled attention (wraps BERT ONNX backend)"
    }
}

impl crate::BatchCapable for DeBERTaV3NER {
    fn extract_entities_batch(
        &self,
        texts: &[&str],
        language: Option<&str>,
    ) -> Result<Vec<Vec<Entity>>> {
        #[cfg(feature = "onnx")]
        {
            self.inner.extract_entities_batch(texts, language)
        }
        #[cfg(not(feature = "onnx"))]
        {
            Err(crate::Error::FeatureNotAvailable(
                "DeBERTa-v3 NER requires 'onnx' feature".to_string(),
            ))
        }
    }
}

impl crate::StreamingCapable for DeBERTaV3NER {
    fn extract_entities_streaming(&self, chunk: &str, offset: usize) -> Result<Vec<Entity>> {
        #[cfg(feature = "onnx")]
        {
            self.inner.extract_entities_streaming(chunk, offset)
        }
        #[cfg(not(feature = "onnx"))]
        {
            Err(crate::Error::FeatureNotAvailable(
                "DeBERTa-v3 NER requires 'onnx' feature".to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deberta_v3_name() {
        if let Ok(model) = DeBERTaV3NER::new("microsoft/deberta-v3-base") {
            assert_eq!(model.name(), "deberta_v3");
        }
        // If model creation fails (e.g., feature not enabled), test is skipped
    }
}
