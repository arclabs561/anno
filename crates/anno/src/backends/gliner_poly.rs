//! Poly-Encoder GLiNER: Advanced GLiNER Architecture with Post-Fusion
//!
//! Poly-Encoder GLiNER extends the bi-encoder architecture with a post-fusion step
//! that enables inter-label interactions, improving performance on complex NER tasks.
//!
//! # Architecture Difference
//!
//! **Bi-Encoder (current GLiNER)**:
//! - Text and labels encoded separately
//! - Direct cosine similarity matching
//! - Fast but limited inter-label understanding
//!
//! **Poly-Encoder (this backend)**:
//! - Text and labels encoded separately (same as bi-encoder)
//! - **Post-fusion step**: Labels interact with each other and with text
//! - Better disambiguation of semantically similar entities
//! - Slightly slower but more accurate for many entity types
//!
//! # Research
//!
//! - **Paper**: [GLiNER Evolution](https://blog.knowledgator.com/meet-the-new-zero-shot-ner-architecture-30ffc2cb1ee0)
//! - **Key Insight**: Poly-encoder enables bidirectional communication between labels
//!   and between labels and text, addressing bi-encoder limitations
//!
//! # Usage
//!
//! ```rust,ignore
//! use anno::backends::gliner_poly::GLiNERPoly;
//!
//! let model = GLiNERPoly::new("knowledgator/modern-gliner-poly-large-v1.0")?;
//! let entities = model.extract("Steve Jobs founded Apple.", &["person", "organization"], 0.5)?;
//! ```

use crate::backends::inference::ZeroShotNER;
use crate::{Entity, EntityType, Error, Model, Result};

/// Poly-Encoder GLiNER backend for zero-shot NER with inter-label interactions.
///
/// **Status:** Scaffolding / not implemented yet.
///
/// Poly-encoder fusion requires a different ONNX export than the bi-encoder GLiNER models.
/// We keep the type for forward compatibility, but it returns a clear error if used.
#[derive(Debug)]
pub struct GLiNERPoly {
    _private: (),
}

impl GLiNERPoly {
    /// Create a new Poly-Encoder GLiNER model.
    ///
    /// # Arguments
    /// * `model_name` - HuggingFace model ID (e.g., "knowledgator/modern-gliner-poly-large-v1.0")
    ///
    /// # Note
    /// This backend is not implemented yet. Use `GLiNEROnnx` for zero-shot NER today.
    pub fn new(model_name: &str) -> Result<Self> {
        let _ = model_name;
        Err(Error::FeatureNotAvailable(
            "GLiNERPoly is not implemented yet (poly-encoder fusion ONNX export not supported). \
             Use GLiNEROnnx instead."
                .to_string(),
        ))
    }

    // NOTE: Poly-encoder fusion would live here once we have compatible exports.
    // This would allow labels to interact with each other and with text context.
}

impl Model for GLiNERPoly {
    fn extract_entities(&self, text: &str, language: Option<&str>) -> Result<Vec<Entity>> {
        let _ = (text, language);
        Err(Error::FeatureNotAvailable(
            "GLiNERPoly is not implemented yet. Use GLiNEROnnx.".to_string(),
        ))
    }

    fn supported_types(&self) -> Vec<EntityType> {
        vec![]
    }

    fn is_available(&self) -> bool {
        false
    }

    fn name(&self) -> &'static str {
        "gliner_poly"
    }

    fn description(&self) -> &'static str {
        "Poly-Encoder GLiNER (scaffolding only; poly-encoder fusion not implemented yet)"
    }
}

impl ZeroShotNER for GLiNERPoly {
    fn extract_with_types(
        &self,
        text: &str,
        entity_types: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        let _ = (text, entity_types, threshold);
        Err(Error::FeatureNotAvailable(
            "GLiNERPoly is not implemented yet. Use GLiNEROnnx.".to_string(),
        ))
    }

    fn extract_with_descriptions(
        &self,
        text: &str,
        descriptions: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        let _ = (text, descriptions, threshold);
        Err(Error::FeatureNotAvailable(
            "GLiNERPoly is not implemented yet. Use GLiNEROnnx.".to_string(),
        ))
    }

    fn default_types(&self) -> &[&'static str] {
        &[]
    }
}

// Implement BatchCapable and StreamingCapable for consistency
impl crate::BatchCapable for GLiNERPoly {
    fn extract_entities_batch(
        &self,
        texts: &[&str],
        language: Option<&str>,
    ) -> Result<Vec<Vec<Entity>>> {
        let _ = (texts, language);
        Err(Error::FeatureNotAvailable(
            "GLiNERPoly is not implemented yet. Use GLiNEROnnx.".to_string(),
        ))
    }
}

impl crate::StreamingCapable for GLiNERPoly {
    fn extract_entities_streaming(&self, chunk: &str, offset: usize) -> Result<Vec<Entity>> {
        let _ = (chunk, offset);
        Err(Error::FeatureNotAvailable(
            "GLiNERPoly is not implemented yet. Use GLiNEROnnx.".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gliner_poly_creation() {
        let err = GLiNERPoly::new("knowledgator/modern-gliner-poly-large-v1.0").unwrap_err();
        assert!(matches!(err, Error::FeatureNotAvailable(_)));
    }

    #[test]
    fn test_gliner_poly_name() {
        // Name is stable even when backend is unavailable.
        let model = GLiNERPoly { _private: () };
        assert_eq!(model.name(), "gliner_poly");
    }
}
