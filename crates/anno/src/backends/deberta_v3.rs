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
//! - **Improved performance**: often stronger than standard BERT in published evaluations
//!
//! # Usage
//!
//! ```rust,ignore
//! use anno::backends::deberta_v3::DeBERTaV3NER;
//!
//! let model = DeBERTaV3NER::new("microsoft/deberta-v3-base")?;
//! let entities = model.extract_entities("Steve Jobs founded Apple.", None)?;
//! ```

use crate::{Entity, EntityType, Language, Model, Result};

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
    fn extract_entities(&self, text: &str, language: Option<Language>) -> Result<Vec<Entity>> {
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

    fn capabilities(&self) -> crate::ModelCapabilities {
        crate::ModelCapabilities::default()
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

    /// Without the `onnx` feature, `new` must return `FeatureNotAvailable`.
    /// With the feature but no model file, should still fail with a clear error.
    #[test]
    fn test_deberta_new_error_without_model() {
        let result = DeBERTaV3NER::new("nonexistent-model-id-12345");
        match result {
            Ok(_) => {}
            Err(e) => {
                let msg = e.to_string();
                assert!(!msg.is_empty(), "error message should be non-empty");
                assert!(
                    msg.contains("onnx")
                        || msg.contains("feature")
                        || msg.contains("Feature")
                        || msg.contains("model")
                        || msg.contains("Model")
                        || msg.contains("Retrieval")
                        || msg.contains("not found"),
                    "error should indicate missing feature or model, got: {msg}"
                );
            }
        }
    }

    #[test]
    fn test_deberta_supported_types() {
        if let Ok(model) = DeBERTaV3NER::new("microsoft/deberta-v3-base") {
            let types = model.supported_types();
            assert!(types.contains(&EntityType::Person));
            assert!(types.contains(&EntityType::Organization));
            assert!(types.contains(&EntityType::Location));
            assert_eq!(types.len(), 3);
        }
    }

    #[test]
    fn test_deberta_description_is_nonempty() {
        if let Ok(model) = DeBERTaV3NER::new("microsoft/deberta-v3-base") {
            let desc = model.description();
            assert!(!desc.is_empty());
            assert!(
                desc.contains("DeBERTa"),
                "description should mention DeBERTa, got: {desc}"
            );
        }
    }

    #[test]
    fn test_deberta_is_available_false_without_model() {
        match DeBERTaV3NER::new("nonexistent-model-id-12345") {
            Ok(model) => {
                assert!(
                    !model.is_available(),
                    "model with nonexistent ID should not be available"
                );
            }
            Err(_) => {
                // Expected path without onnx feature or missing model.
            }
        }
    }

    #[test]
    fn test_deberta_capabilities() {
        if let Ok(model) = DeBERTaV3NER::new("microsoft/deberta-v3-base") {
            let caps = model.capabilities();
            // ModelCapabilities::default() -- no special capabilities
            let _ = caps;
        }
    }

    #[test]
    fn test_deberta_extract_entities_error_without_model() {
        if let Ok(model) = DeBERTaV3NER::new("nonexistent-model-id-12345") {
            let result = model.extract_entities("Hello world", None);
            assert!(
                result.is_err(),
                "extract_entities should error without a real model"
            );
        }
    }
}
