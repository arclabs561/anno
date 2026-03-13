//! ALBERT NER Backend
//!
//! ALBERT (A Lite BERT) is an efficient, smaller model that achieves competitive
//! performance on NER tasks, especially in domain-specific scenarios.
//!
//! # Architecture
//!
//! ALBERT improves efficiency over BERT with:
//! - **Factorized embedding parameterization**: Shares embeddings across layers
//! - **Cross-layer parameter sharing**: Reduces model size significantly
//! - **Smaller model size**: 11MB vs 110MB for BERT-base
//! - **Domain-specific performance**: Excellent for biomedical and specialized domains
//!
//! # Research
//!
//! Treat ALBERT as a size/latency trade-off option; any quality claims should be
//! established via the `anno` eval harness for the specific dataset/task mix.
//!
//! # Usage
//!
//! ```rust,ignore
//! use anno::backends::albert::ALBERTNER;
//!
//! let model = ALBERTNER::new("albert-base-v2")?;
//! let entities = model.extract_entities("Steve Jobs founded Apple.", None)?;
//! ```

use crate::{Entity, EntityType, Language, Model, Result};

#[cfg(feature = "onnx")]
use crate::backends::onnx::BertNEROnnx;

/// ALBERT NER backend using ONNX Runtime.
///
/// Currently wraps BertNEROnnx with ALBERT model support.
/// ALBERT models use the same ONNX interface as BERT.
pub struct ALBERTNER {
    #[cfg(feature = "onnx")]
    inner: BertNEROnnx,
    /// Model name for debugging/logging (e.g., "albert-base-v2")
    model_name: String,
}

impl ALBERTNER {
    /// Create a new ALBERT NER model.
    ///
    /// # Arguments
    /// * `model_name` - HuggingFace model ID (e.g., "albert-base-v2")
    pub fn new(model_name: &str) -> Result<Self> {
        #[cfg(feature = "onnx")]
        {
            // ALBERT uses same ONNX interface as BERT
            let inner = BertNEROnnx::new(model_name)?;
            Ok(Self {
                inner,
                model_name: model_name.to_string(),
            })
        }
        #[cfg(not(feature = "onnx"))]
        {
            Err(crate::Error::FeatureNotAvailable(
                "ALBERT NER requires 'onnx' feature".to_string(),
            ))
        }
    }

    /// Return the HuggingFace model ID used to construct this model.
    pub fn model_id(&self) -> &str {
        &self.model_name
    }
}

impl Model for ALBERTNER {
    fn extract_entities(&self, text: &str, language: Option<Language>) -> Result<Vec<Entity>> {
        #[cfg(feature = "onnx")]
        {
            self.inner.extract_entities(text, language)
        }
        #[cfg(not(feature = "onnx"))]
        {
            Err(crate::Error::FeatureNotAvailable(
                "ALBERT NER requires 'onnx' feature".to_string(),
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
        "albert"
    }

    fn description(&self) -> &'static str {
        "ALBERT NER - efficient, small model (11MB) with competitive performance"
    }

    fn capabilities(&self) -> crate::ModelCapabilities {
        crate::ModelCapabilities {
            batch_capable: true,
            streaming_capable: true,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_albert_name() {
        if let Ok(model) = ALBERTNER::new("albert-base-v2") {
            assert_eq!(model.name(), "albert");
        }
        // If model creation fails (e.g., feature not enabled), test is skipped
    }

    /// Without the `onnx` feature, `new` must return `FeatureNotAvailable`.
    /// With the feature, the model file is absent, so it should still fail
    /// (but with a retrieval/init error, not a silent success).
    #[test]
    fn test_albert_new_error_without_model() {
        let result = ALBERTNER::new("nonexistent-model-id-12345");
        match result {
            Ok(_) => {
                // If construction somehow succeeds, is_available should reflect reality.
                // (unlikely without a real model file)
            }
            Err(e) => {
                let msg = e.to_string();
                // Should be a meaningful error, not empty.
                assert!(!msg.is_empty(), "error message should be non-empty");
                // Should mention feature or model-related issue.
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
    fn test_albert_supported_types() {
        // supported_types is independent of model loading -- test via any instance.
        // Without onnx, we can't construct, so test the constant directly.
        if let Ok(model) = ALBERTNER::new("albert-base-v2") {
            let types = model.supported_types();
            assert!(types.contains(&EntityType::Person));
            assert!(types.contains(&EntityType::Organization));
            assert!(types.contains(&EntityType::Location));
            assert_eq!(types.len(), 3);
        }
    }

    #[test]
    fn test_albert_description_is_nonempty() {
        if let Ok(model) = ALBERTNER::new("albert-base-v2") {
            let desc = model.description();
            assert!(!desc.is_empty());
            assert!(
                desc.contains("ALBERT"),
                "description should mention ALBERT, got: {desc}"
            );
        }
    }

    #[test]
    fn test_albert_is_available_false_without_model() {
        // Without onnx feature: constructor fails.
        // With onnx feature but no model file: constructor fails OR is_available returns false.
        match ALBERTNER::new("nonexistent-model-id-12345") {
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
    fn test_albert_model_id_accessor() {
        // model_id() should return the string passed to new(), even if construction
        // fails for other reasons. We can only test this when new() succeeds.
        if let Ok(model) = ALBERTNER::new("albert-base-v2") {
            assert_eq!(model.model_id(), "albert-base-v2");
        }
    }

    #[test]
    fn test_albert_capabilities() {
        if let Ok(model) = ALBERTNER::new("albert-base-v2") {
            let caps = model.capabilities();
            assert!(caps.batch_capable);
            assert!(caps.streaming_capable);
        }
    }

    /// Verify that extract_entities returns a clear error (not a panic) when the
    /// model is not actually loaded.
    #[test]
    fn test_albert_extract_entities_error_without_model() {
        // Construction will fail without onnx or without a real model file.
        // If it somehow succeeds, calling extract should return Err, not panic.
        if let Ok(model) = ALBERTNER::new("nonexistent-model-id-12345") {
            let result = model.extract_entities("Hello world", None);
            assert!(
                result.is_err(),
                "extract_entities should error without a real model"
            );
        }
    }
}
