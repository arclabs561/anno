//! Encoder abstraction for GLiNER and span matching models.
//!
//! # Design
//!
//! GLiNER separates the encoder (BERT/RoBERTa/ModernBERT) from the span matching head.
//! This module provides abstractions for:
//!
//! 1. **Encoder**: Transforms text to embeddings
//! 2. **SpanMatcher**: Takes embeddings + entity type embeddings and computes similarity
//!
//! # Available Encoders
//!
//! | Model | Context | Speed | Notes |
//! |-------|---------|-------|-------|
//! | BERT | 512 | Fast | Classic, well-tested |
//! | RoBERTa | 512 | Fast | Improved pre-training |
//! | DeBERTa | 512 | Medium | Better than BERT |
//! | ModernBERT | 8192 | Fast | SOTA, recommended |
//!
//! # GLiNER Models by Encoder
//!
//! | Model ID | Base Encoder | Mode |
//! |----------|--------------|------|
//! | `onnx-community/gliner_small-v2.1` | DeBERTa-v3-small | Span |
//! | `onnx-community/gliner_medium-v2.1` | DeBERTa-v3-base | Span |
//! | `onnx-community/gliner_large-v2.1` | DeBERTa-v3-large | Span |
//! | `knowledgator/modern-gliner-bi-large-v1.0` | ModernBERT-large | Span |
//! | `knowledgator/gliner-multitask-v1.0` | DeBERTa-v3-base | Token |

use std::fmt;

/// Known encoder architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EncoderType {
    /// Classic BERT (512 tokens)
    Bert,
    /// RoBERTa (512 tokens)
    Roberta,
    /// DeBERTa (512 tokens, improved attention)
    Deberta,
    /// DeBERTa v3 (512 tokens, latest)
    DebertaV3,
    /// ModernBERT (8192 tokens, SOTA)
    ModernBert,
    /// Unknown/custom encoder
    Unknown,
}

impl EncoderType {
    /// Maximum context length for this encoder.
    #[must_use]
    pub const fn max_context_length(&self) -> usize {
        match self {
            EncoderType::Bert => 512,
            EncoderType::Roberta => 512,
            EncoderType::Deberta => 512,
            EncoderType::DebertaV3 => 512,
            EncoderType::ModernBert => 8192,
            EncoderType::Unknown => 512,
        }
    }

    /// Whether this encoder uses RoPE (rotary position embeddings).
    #[must_use]
    pub const fn uses_rope(&self) -> bool {
        matches!(self, EncoderType::ModernBert)
    }

    /// Relative speed (higher = faster).
    #[must_use]
    pub const fn relative_speed(&self) -> u8 {
        match self {
            EncoderType::Bert => 5,
            EncoderType::Roberta => 5,
            EncoderType::Deberta => 4,
            EncoderType::DebertaV3 => 4,
            EncoderType::ModernBert => 6, // Unpadding makes it faster
            EncoderType::Unknown => 3,
        }
    }
}

impl fmt::Display for EncoderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EncoderType::Bert => write!(f, "BERT"),
            EncoderType::Roberta => write!(f, "RoBERTa"),
            EncoderType::Deberta => write!(f, "DeBERTa"),
            EncoderType::DebertaV3 => write!(f, "DeBERTa-v3"),
            EncoderType::ModernBert => write!(f, "ModernBERT"),
            EncoderType::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Known GLiNER model variants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GLiNERModel {
    /// HuggingFace model ID.
    pub model_id: &'static str,
    /// Base encoder type.
    pub encoder: EncoderType,
    /// Model size (parameters).
    pub size: ModelSize,
    /// Whether this model supports relation extraction.
    pub supports_relations: bool,
    /// Notes about this model.
    pub notes: &'static str,
}

/// Model size category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelSize {
    /// ~50M parameters
    Small,
    /// ~110M parameters
    Medium,
    /// ~330M parameters
    Large,
    /// ~1B+ parameters
    XLarge,
}

impl fmt::Display for ModelSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelSize::Small => write!(f, "S"),
            ModelSize::Medium => write!(f, "M"),
            ModelSize::Large => write!(f, "L"),
            ModelSize::XLarge => write!(f, "XL"),
        }
    }
}

/// Catalog of known GLiNER models.
pub static GLINER_MODELS: &[GLiNERModel] = &[
    // DeBERTa-based (standard)
    GLiNERModel {
        model_id: "onnx-community/gliner_small-v2.1",
        encoder: EncoderType::DebertaV3,
        size: ModelSize::Small,
        supports_relations: false,
        notes: "Fast, good accuracy, recommended for CPU",
    },
    GLiNERModel {
        model_id: "onnx-community/gliner_medium-v2.1",
        encoder: EncoderType::DebertaV3,
        size: ModelSize::Medium,
        supports_relations: false,
        notes: "Balanced speed/accuracy",
    },
    GLiNERModel {
        model_id: "onnx-community/gliner_large-v2.1",
        encoder: EncoderType::DebertaV3,
        size: ModelSize::Large,
        supports_relations: false,
        notes: "Higher accuracy, recommended for GPU",
    },
    // ModernBERT-based (SOTA)
    GLiNERModel {
        model_id: "knowledgator/modern-gliner-bi-large-v1.0",
        encoder: EncoderType::ModernBert,
        size: ModelSize::Large,
        supports_relations: false,
        notes: "SOTA accuracy, 8K context, ~3% better than DeBERTa",
    },
    // Multitask (relations)
    GLiNERModel {
        model_id: "knowledgator/gliner-multitask-v1.0",
        encoder: EncoderType::DebertaV3,
        size: ModelSize::Medium,
        supports_relations: true,
        notes: "Supports relation extraction",
    },
    GLiNERModel {
        model_id: "onnx-community/gliner-multitask-large-v0.5",
        encoder: EncoderType::DebertaV3,
        size: ModelSize::Large,
        supports_relations: true,
        notes: "Large multitask, higher accuracy relations",
    },
];

impl GLiNERModel {
    /// Find a model by ID.
    #[must_use]
    pub fn by_id(model_id: &str) -> Option<&'static GLiNERModel> {
        GLINER_MODELS.iter().find(|m| m.model_id == model_id)
    }

    /// Get all models with a specific encoder.
    #[must_use]
    pub fn by_encoder(encoder: EncoderType) -> Vec<&'static GLiNERModel> {
        GLINER_MODELS
            .iter()
            .filter(|m| m.encoder == encoder)
            .collect()
    }

    /// Get models that support relations.
    #[must_use]
    pub fn with_relations() -> Vec<&'static GLiNERModel> {
        GLINER_MODELS
            .iter()
            .filter(|m| m.supports_relations)
            .collect()
    }

    /// Get the fastest model.
    #[must_use]
    pub fn fastest() -> &'static GLiNERModel {
        &GLINER_MODELS[0] // Small is fastest
    }

    /// Get the most accurate model.
    #[must_use]
    pub fn most_accurate() -> &'static GLiNERModel {
        // ModernBERT-large is SOTA
        GLINER_MODELS
            .iter()
            .find(|m| m.encoder == EncoderType::ModernBert)
            .unwrap_or(&GLINER_MODELS[2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_type_display() {
        assert_eq!(EncoderType::ModernBert.to_string(), "ModernBERT");
        assert_eq!(EncoderType::DebertaV3.to_string(), "DeBERTa-v3");
    }

    #[test]
    fn test_model_lookup() {
        let model = GLiNERModel::by_id("onnx-community/gliner_small-v2.1");
        assert!(model.is_some());
        assert_eq!(model.unwrap().encoder, EncoderType::DebertaV3);
    }

    #[test]
    fn test_models_by_encoder() {
        let modern_models = GLiNERModel::by_encoder(EncoderType::ModernBert);
        assert!(!modern_models.is_empty());
        assert!(modern_models
            .iter()
            .all(|m| m.encoder == EncoderType::ModernBert));
    }

    #[test]
    fn test_fastest_model() {
        let fastest = GLiNERModel::fastest();
        assert_eq!(fastest.size, ModelSize::Small);
    }

    #[test]
    fn test_most_accurate() {
        let best = GLiNERModel::most_accurate();
        assert_eq!(best.encoder, EncoderType::ModernBert);
    }

    #[test]
    fn test_context_length() {
        assert_eq!(EncoderType::Bert.max_context_length(), 512);
        assert_eq!(EncoderType::ModernBert.max_context_length(), 8192);
    }
}
