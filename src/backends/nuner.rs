//! NuNER - Token-based zero-shot NER from NuMind.
//!
//! NuNER is a family of zero-shot NER models built on the GLiNER architecture
//! with a token classifier design (vs span classifier). Key advantages:
//!
//! - **Arbitrary-length entities**: No hard limit on entity span length
//! - **Efficient training**: Trained on NuNER v2.0 dataset (Pile + C4)
//! - **MIT Licensed**: Open weights from NuMind
//!
//! # Architecture
//!
//! NuNER uses the same bi-encoder architecture as GLiNER:
//!
//! ```text
//! Input: "James Bond works at MI6"
//!        Labels: ["person", "organization"]
//!
//!        ┌──────────────────────┐
//!        │   Shared Encoder     │
//!        │  (DeBERTa/BERT)      │
//!        └──────────────────────┘
//!               │         │
//!        ┌──────┴──┐   ┌──┴─────┐
//!        │  Token  │   │ Label  │
//!        │  Embeds │   │ Embeds │
//!        └─────────┘   └────────┘
//!               │         │
//!        ┌──────┴─────────┴──────┐
//!        │   Token Classification │  (BIO tags per token)
//!        └───────────────────────┘
//! ```
//!
//! # Model Variants
//!
//! | Model | Context | Size | Notes |
//! |-------|---------|------|-------|
//! | NuNER Zero | 512 | 340M | General zero-shot |
//! | NuNER Zero 4k | 4096 | 340M | Long context variant |
//! | NuNER Zero-span | 512 | 340M | Span-based (like GLiNER) |
//!
//! # Usage
//!
//! NuNER models are GLiNER-compatible and can be loaded via `gline-rs`:
//!
//! ```rust,ignore
//! use anno::NuNER;
//!
//! // Load NuNER Zero model (requires ONNX export)
//! let ner = NuNER::new("path/to/nuner-zero")?;
//!
//! // Custom entity types at inference time
//! let entities = ner.extract_entities(
//!     "Apple CEO Tim Cook announced...",
//!     Some(&["person", "organization", "product"])
//! )?;
//! ```
//!
//! # Differences from GLiNER
//!
//! | Aspect | GLiNER | NuNER |
//! |--------|--------|-------|
//! | Output | Span classification | Token classification (BIO) |
//! | Entity length | Limited by span window | Arbitrary |
//! | Training data | Academic NER datasets | Pile + C4 (1M examples) |
//! | License | Apache 2.0 | MIT |
//!
//! # References
//!
//! - [NuNER Zero on HuggingFace](https://huggingface.co/numind/NuNER_Zero)
//! - [NuMind Foundation Models](https://www.numind.ai/models)
//! - [gline-rs](https://github.com/fbilhaut/gline-rs) - Rust inference engine

use crate::{Entity, EntityType, Model, Result};

/// NuNER Zero-shot NER model.
///
/// This is a wrapper around the GLiNER architecture specialized for
/// NuNER's token classification approach. NuNER models use BIO tagging
/// at the token level rather than span classification.
///
/// # Model Loading
///
/// NuNER models must be converted to ONNX format. The recommended
/// models are:
///
/// - `numind/NuNER_Zero` - General zero-shot NER
/// - `numind/NuNER_Zero_4k` - Long context (4096 tokens)
///
/// # Example
///
/// ```rust,ignore
/// let ner = NuNER::default(); // Uses NuNER Zero
///
/// // NuNER excels at arbitrary entity types
/// let entities = ner.extract_entities(
///     "The CRISPR-Cas9 system was developed by Jennifer Doudna",
///     Some(&["technology", "scientist"])
/// )?;
/// ```
pub struct NuNER {
    /// Model path or identifier
    model_id: String,
    /// Confidence threshold (0.0-1.0)
    threshold: f64,
    /// Default entity labels for zero-shot
    default_labels: Vec<String>,
}

impl NuNER {
    /// Create NuNER with default configuration.
    ///
    /// Uses the `numind/NuNER_Zero` model with standard NER labels.
    #[must_use]
    pub fn new() -> Self {
        Self {
            model_id: "numind/NuNER_Zero".to_string(),
            threshold: 0.5,
            default_labels: vec![
                "person".to_string(),
                "organization".to_string(),
                "location".to_string(),
                "date".to_string(),
                "product".to_string(),
                "event".to_string(),
            ],
        }
    }

    /// Create with custom model path.
    #[must_use]
    pub fn with_model(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            ..Self::new()
        }
    }

    /// Set confidence threshold.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set default entity labels for zero-shot inference.
    #[must_use]
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        self.default_labels = labels;
        self
    }

    /// Get the model identifier.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Get the confidence threshold.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Map label string to EntityType.
    fn map_label_to_entity_type(label: &str) -> EntityType {
        match label.to_lowercase().as_str() {
            "person" | "per" => EntityType::Person,
            "organization" | "org" | "company" => EntityType::Organization,
            "location" | "loc" | "place" | "gpe" => EntityType::Location,
            "date" => EntityType::Date,
            "time" => EntityType::Time,
            "money" | "currency" => EntityType::Money,
            "percent" | "percentage" => EntityType::Percent,
            _ => EntityType::Other(label.to_string()),
        }
    }
}

impl Default for NuNER {
    fn default() -> Self {
        Self::new()
    }
}

impl Model for NuNER {
    fn extract_entities(&self, text: &str, _language: Option<&str>) -> Result<Vec<Entity>> {
        // NuNER is a placeholder - actual inference requires gline-rs with
        // the model ONNX files. This demonstrates the API design.
        //
        // In production, this would:
        // 1. Load the NuNER ONNX model
        // 2. Tokenize input with the model's tokenizer
        // 3. Run token classification
        // 4. Decode BIO tags to entities
        
        if text.is_empty() {
            return Ok(Vec::new());
        }

        // Placeholder: Return empty until actual model integration
        // The real implementation would use gline-rs in TokenMode:
        //
        // let model = GLiNER::<TokenMode>::new(params, runtime, tokenizer, onnx)?;
        // let input = TextInput::from_str(&[text], &self.default_labels)?;
        // let output = model.inference(input)?;
        
        Ok(Vec::new())
    }

    fn supported_types(&self) -> Vec<EntityType> {
        self.default_labels
            .iter()
            .map(|l| Self::map_label_to_entity_type(l))
            .collect()
    }

    fn is_available(&self) -> bool {
        // NuNER requires ONNX models to be downloaded
        // This would check if model files exist
        false
    }

    fn name(&self) -> &'static str {
        "nuner"
    }

    fn description(&self) -> &'static str {
        "NuNER Zero: Token-based zero-shot NER from NuMind (MIT licensed)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nuner_creation() {
        let ner = NuNER::new();
        assert_eq!(ner.model_id(), "numind/NuNER_Zero");
        assert!((ner.threshold() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_nuner_with_custom_model() {
        let ner = NuNER::with_model("custom/model")
            .with_threshold(0.7)
            .with_labels(vec!["technology".to_string()]);
        
        assert_eq!(ner.model_id(), "custom/model");
        assert!((ner.threshold() - 0.7).abs() < f64::EPSILON);
        assert_eq!(ner.default_labels.len(), 1);
    }

    #[test]
    fn test_label_mapping() {
        assert_eq!(NuNER::map_label_to_entity_type("person"), EntityType::Person);
        assert_eq!(NuNER::map_label_to_entity_type("PER"), EntityType::Person);
        assert_eq!(NuNER::map_label_to_entity_type("organization"), EntityType::Organization);
        assert_eq!(NuNER::map_label_to_entity_type("custom"), EntityType::Other("custom".to_string()));
    }

    #[test]
    fn test_supported_types() {
        let ner = NuNER::new();
        let types = ner.supported_types();
        assert!(types.contains(&EntityType::Person));
        assert!(types.contains(&EntityType::Organization));
        assert!(types.contains(&EntityType::Location));
    }

    #[test]
    fn test_empty_input() {
        let ner = NuNER::new();
        let entities = ner.extract_entities("", None).unwrap();
        assert!(entities.is_empty());
    }
}

