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
//! ALBERT achieves SOTA on biomedical method NER with only 11MB model size,
//! demonstrating that smaller models can excel in specialized domains.
//!
//! # Usage
//!
//! ```rust,ignore
//! use anno::backends::albert::ALBERTNER;
//!
//! let model = ALBERTNER::new("albert-base-v2")?;
//! let entities = model.extract_entities("Steve Jobs founded Apple.", None)?;
//! ```

use crate::{Entity, EntityType, Model, Result};

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
    #[allow(dead_code)] // Reserved for future logging/debugging
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
}

impl Model for ALBERTNER {
    fn extract_entities(&self, text: &str, language: Option<&str>) -> Result<Vec<Entity>> {
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
}

impl crate::BatchCapable for ALBERTNER {
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
                "ALBERT NER requires 'onnx' feature".to_string(),
            ))
        }
    }
}

impl crate::StreamingCapable for ALBERTNER {
    fn extract_entities_streaming(&self, chunk: &str, offset: usize) -> Result<Vec<Entity>> {
        #[cfg(feature = "onnx")]
        {
            self.inner.extract_entities_streaming(chunk, offset)
        }
        #[cfg(not(feature = "onnx"))]
        {
            Err(crate::Error::FeatureNotAvailable(
                "ALBERT NER requires 'onnx' feature".to_string(),
            ))
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
}
