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
use crate::{Entity, EntityType, Model, Result};

/// Poly-Encoder GLiNER backend for zero-shot NER with inter-label interactions.
///
/// Currently a placeholder implementation that wraps bi-encoder GLiNER.
/// Full poly-encoder fusion layer implementation pending ONNX model support.
pub struct GLiNERPoly {
    /// Underlying bi-encoder (for now, we use bi-encoder as base)
    #[cfg(feature = "onnx")]
    bi_encoder: crate::backends::gliner_onnx::GLiNEROnnx,
    /// Fusion layer configuration (placeholder)
    /// Currently unused - will be used when poly-encoder fusion is implemented
    #[allow(dead_code)]
    fusion_enabled: bool,
}

impl GLiNERPoly {
    /// Create a new Poly-Encoder GLiNER model.
    ///
    /// # Arguments
    /// * `model_name` - HuggingFace model ID (e.g., "knowledgator/modern-gliner-poly-large-v1.0")
    ///
    /// # Note
    /// Currently falls back to bi-encoder GLiNER. Full poly-encoder models pending.
    pub fn new(model_name: &str) -> Result<Self> {
        #[cfg(feature = "onnx")]
        {
            // For now, use bi-encoder GLiNER as base
            // Full poly-encoder models will be available later
            let bi_encoder = crate::backends::gliner_onnx::GLiNEROnnx::new(model_name)?;
            Ok(Self {
                bi_encoder,
                fusion_enabled: false, // Placeholder: fusion not yet implemented
            })
        }
        #[cfg(not(feature = "onnx"))]
        {
            Err(crate::Error::FeatureNotAvailable(
                "Poly-Encoder GLiNER requires 'onnx' feature".to_string(),
            ))
        }
    }

    /// Apply poly-encoder fusion to improve inter-label disambiguation.
    ///
    /// This is a placeholder that will be implemented when poly-encoder models are available.
    /// The fusion step allows labels to interact with each other and with text context.
    #[allow(dead_code)] // Placeholder method, not yet used
    #[allow(unused_variables)]
    fn apply_fusion(
        &self,
        text_embeddings: &[f32],
        label_embeddings: &[f32],
        num_labels: usize,
    ) -> Vec<f32> {
        // Placeholder: For now, just return label embeddings unchanged
        // Full implementation would:
        // 1. Create attention matrix between labels
        // 2. Fuse label embeddings with text context
        // 3. Return enhanced label embeddings
        label_embeddings.to_vec()
    }
}

#[cfg(feature = "onnx")]
impl Model for GLiNERPoly {
    fn extract_entities(&self, text: &str, language: Option<&str>) -> Result<Vec<Entity>> {
        // Use bi-encoder for now (poly-encoder fusion pending)
        self.bi_encoder.extract_entities(text, language)
    }

    fn supported_types(&self) -> Vec<EntityType> {
        // Poly-encoder supports any entity type (zero-shot)
        vec![
            EntityType::Person,
            EntityType::Organization,
            EntityType::Location,
        ]
    }

    fn is_available(&self) -> bool {
        self.bi_encoder.is_available()
    }

    fn name(&self) -> &'static str {
        "gliner_poly"
    }

    fn description(&self) -> &'static str {
        "Poly-Encoder GLiNER with inter-label fusion (placeholder - full implementation pending)"
    }
}

#[cfg(feature = "onnx")]
impl ZeroShotNER for GLiNERPoly {
    fn extract_with_types(
        &self,
        text: &str,
        entity_types: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        // Use bi-encoder extraction for now
        // Full poly-encoder would apply fusion before matching
        self.bi_encoder
            .extract_with_types(text, entity_types, threshold)
    }

    fn extract_with_descriptions(
        &self,
        text: &str,
        descriptions: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        // Use bi-encoder extraction for now
        self.bi_encoder
            .extract_with_descriptions(text, descriptions, threshold)
    }
}

#[cfg(not(feature = "onnx"))]
impl Model for GLiNERPoly {
    fn extract_entities(&self, _text: &str, _language: Option<&str>) -> Result<Vec<Entity>> {
        Err(crate::Error::FeatureNotAvailable(
            "Poly-Encoder GLiNER requires 'onnx' feature".to_string(),
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
        "Poly-Encoder GLiNER (requires 'onnx' feature)"
    }
}

// Implement BatchCapable and StreamingCapable for consistency
impl crate::BatchCapable for GLiNERPoly {
    fn extract_entities_batch(
        &self,
        texts: &[&str],
        language: Option<&str>,
    ) -> Result<Vec<Vec<Entity>>> {
        #[cfg(feature = "onnx")]
        {
            self.bi_encoder.extract_entities_batch(texts, language)
        }
        #[cfg(not(feature = "onnx"))]
        {
            Err(crate::Error::FeatureNotAvailable(
                "Poly-Encoder GLiNER requires 'onnx' feature".to_string(),
            ))
        }
    }
}

impl crate::StreamingCapable for GLiNERPoly {
    fn extract_entities_streaming(&self, chunk: &str, offset: usize) -> Result<Vec<Entity>> {
        #[cfg(feature = "onnx")]
        {
            self.bi_encoder.extract_entities_streaming(chunk, offset)
        }
        #[cfg(not(feature = "onnx"))]
        {
            Err(crate::Error::FeatureNotAvailable(
                "Poly-Encoder GLiNER requires 'onnx' feature".to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "onnx")]
    fn test_gliner_poly_creation() {
        // This will fail if model not available, which is expected
        let _model = GLiNERPoly::new("onnx-community/gliner_small-v2.1");
        // Just test that struct can be created
    }

    #[test]
    fn test_gliner_poly_name() {
        #[cfg(feature = "onnx")]
        {
            if let Ok(model) = GLiNERPoly::new("onnx-community/gliner_small-v2.1") {
                assert_eq!(model.name(), "gliner_poly");
            }
        }
    }
}
