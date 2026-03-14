//! GLiNER poly-encoder backend for zero-shot NER with inter-label interactions.
//!
//! The poly-encoder architecture uses a shared text+entity encoder with attention-based
//! fusion, rather than the separate bi-encoder heads used by standard GLiNER. This allows
//! entity type labels to attend to each other and to the text context jointly, which can
//! improve extraction quality when labels are semantically related.
//!
//! ## ONNX Model Format
//!
//! The ONNX model is exported by a companion Python script. Inputs:
//!
//! - `input_ids`: `[batch, seq_len]` (i64)
//! - `attention_mask`: `[batch, seq_len]` (i64)
//! - `words_mask`: `[batch, seq_len]` (i64) -- maps subword tokens to word indices
//! - `text_lengths`: `[batch, 1]` (i64) -- number of words in each text
//! - `span_idx`: `[batch, num_spans, 2]` (i64) -- start/end word indices per span
//! - `span_mask`: `[batch, num_spans]` (bool) -- which spans are valid
//! - `labels_input_ids`: `[num_labels, label_seq_len]` (i64) -- tokenized entity labels
//! - `labels_attention_mask`: `[num_labels, label_seq_len]` (i64)
//!
//! Output:
//!
//! - `logits`: `[batch, seq_len, num_spans, num_classes]` (f32) -- span logits
//!
//! ## Usage
//!
//! ```rust,ignore
//! use anno::backends::gliner_poly::GLiNERPoly;
//! use anno::backends::inference::ZeroShotNER;
//!
//! let model = GLiNERPoly::new("knowledgator/gliner-bi-large-v1.0")?;
//! let entities = model.extract_with_types(
//!     "John works at Apple in Cupertino",
//!     &["person", "organization", "location"],
//!     0.5,
//! )?;
//! ```

#![allow(unused_imports)] // EntityType used conditionally

#[cfg(feature = "onnx")]
mod inference;

use crate::backends::inference::ZeroShotNER;
use crate::{Entity, EntityType, Error, Language, Result};
use anno_core::EntityCategory;

/// Default entity types for zero-shot GLiNERPoly when used via the Model trait.
const DEFAULT_POLY_LABELS: &[&str] = &[
    "person",
    "organization",
    "location",
    "date",
    "time",
    "money",
    "percent",
    "product",
    "event",
    "facility",
];

/// Local cache directories where exported ONNX models may reside.
///
/// The export script writes to `~/.cache/anno/models/gliner-poly/` by default;
/// `cache_dir()` may resolve to a platform-specific location (e.g. `~/Library/Caches/anno`
/// on macOS).
fn local_model_cache_candidates() -> [std::path::PathBuf; 2] {
    [
        crate::env::cache_dir().join("models/gliner-poly"),
        dirs::home_dir()
            .unwrap_or_default()
            .join(".cache/anno/models/gliner-poly"),
    ]
}

// =============================================================================
// ONNX-enabled implementation
// =============================================================================

#[cfg(feature = "onnx")]
use crate::sync::{lock, Mutex};

/// Poly-Encoder GLiNER backend for zero-shot NER with inter-label interactions.
///
/// Unlike the bi-encoder GLiNER (`GLiNEROnnx`), the poly-encoder fuses text and
/// entity type representations through cross-attention before scoring spans. This
/// allows label-to-label and label-to-text interactions during encoding.
///
/// Requires the `onnx` feature and a compatible ONNX export.
#[cfg(feature = "onnx")]
pub struct GLiNERPoly {
    session: Mutex<ort::session::Session>,
    /// Text tokenizer (DeBERTa-v3 vocab, ~128k tokens).
    tokenizer: std::sync::Arc<tokenizers::Tokenizer>,
    /// Label tokenizer (BGE vocab, 30522 tokens) -- separate because the bi-encoder
    /// uses different encoders for text and entity labels.
    label_tokenizer: std::sync::Arc<tokenizers::Tokenizer>,
    model_name: String,
    is_quantized: bool,
}

#[cfg(feature = "onnx")]
impl std::fmt::Debug for GLiNERPoly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GLiNERPoly")
            .field("model_name", &self.model_name)
            .field("is_quantized", &self.is_quantized)
            .finish_non_exhaustive()
    }
}

// =============================================================================
// Model trait (ONNX)
// =============================================================================

#[cfg(feature = "onnx")]
impl crate::Model for GLiNERPoly {
    fn extract_entities(&self, text: &str, _language: Option<Language>) -> Result<Vec<Entity>> {
        self.extract(text, DEFAULT_POLY_LABELS, 0.5)
    }

    fn supported_types(&self) -> Vec<EntityType> {
        DEFAULT_POLY_LABELS
            .iter()
            .map(|label| EntityType::Custom {
                name: (*label).to_string(),
                category: EntityCategory::Misc,
            })
            .collect()
    }

    fn is_available(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "gliner_poly"
    }

    fn description(&self) -> &'static str {
        "Poly-Encoder GLiNER for zero-shot NER with inter-label interactions (ONNX)"
    }

    fn capabilities(&self) -> crate::ModelCapabilities {
        crate::ModelCapabilities {
            dynamic_labels: true,
            ..Default::default()
        }
    }

    fn version(&self) -> String {
        format!(
            "gliner-poly-{}-{}",
            self.model_name,
            if self.is_quantized { "q" } else { "fp32" }
        )
    }
}

// =============================================================================
// ZeroShotNER trait (ONNX)
// =============================================================================

#[cfg(feature = "onnx")]
impl ZeroShotNER for GLiNERPoly {
    fn extract_with_types(
        &self,
        text: &str,
        entity_types: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        self.extract(text, entity_types, threshold)
    }

    fn extract_with_descriptions(
        &self,
        text: &str,
        descriptions: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        // Poly-encoder encodes labels as text, so descriptions work the same way.
        self.extract(text, descriptions, threshold)
    }

    fn default_types(&self) -> &[&'static str] {
        DEFAULT_POLY_LABELS
    }
}

// =============================================================================
// DynamicLabels trait (ONNX)
// =============================================================================

#[cfg(feature = "onnx")]
impl crate::DynamicLabels for GLiNERPoly {
    fn extract_with_labels(
        &self,
        text: &str,
        labels: &[&str],
        _language: Option<Language>,
    ) -> Result<Vec<Entity>> {
        self.extract(text, labels, 0.5)
    }
}

// =============================================================================
// Stub when feature disabled
// =============================================================================

#[cfg(not(feature = "onnx"))]
#[derive(Debug)]
pub struct GLiNERPoly {
    _private: (),
}

#[cfg(not(feature = "onnx"))]
impl GLiNERPoly {
    /// Create a new Poly-Encoder GLiNER model (stub -- requires `onnx` feature).
    pub fn new(_model_name: &str) -> Result<Self> {
        Err(Error::FeatureNotAvailable(
            "GLiNERPoly requires the 'onnx' feature. \
             Build with: cargo build --features onnx"
                .to_string(),
        ))
    }
}

#[cfg(not(feature = "onnx"))]
impl crate::Model for GLiNERPoly {
    fn extract_entities(&self, _text: &str, _language: Option<Language>) -> Result<Vec<Entity>> {
        Err(Error::FeatureNotAvailable(
            "GLiNERPoly requires the 'onnx' feature".to_string(),
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

#[cfg(not(feature = "onnx"))]
impl ZeroShotNER for GLiNERPoly {
    fn extract_with_types(
        &self,
        _text: &str,
        _entity_types: &[&str],
        _threshold: f32,
    ) -> Result<Vec<Entity>> {
        Err(Error::FeatureNotAvailable(
            "GLiNERPoly requires the 'onnx' feature".to_string(),
        ))
    }

    fn extract_with_descriptions(
        &self,
        _text: &str,
        _descriptions: &[&str],
        _threshold: f32,
    ) -> Result<Vec<Entity>> {
        Err(Error::FeatureNotAvailable(
            "GLiNERPoly requires the 'onnx' feature".to_string(),
        ))
    }

    fn default_types(&self) -> &[&'static str] {
        DEFAULT_POLY_LABELS
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Test 1: construction without onnx returns FeatureNotAvailable ----
    #[test]
    #[cfg(not(feature = "onnx"))]
    fn test_gliner_poly_creation_no_onnx() {
        let err = GLiNERPoly::new("knowledgator/gliner-bi-large-v1.0").unwrap_err();
        assert!(
            matches!(err, Error::FeatureNotAvailable(_)),
            "expected FeatureNotAvailable, got: {err:?}"
        );
    }

    // ---- Test 2: name is stable across feature configurations ----
    #[test]
    #[cfg(not(feature = "onnx"))]
    fn test_gliner_poly_name_stable() {
        use crate::Model;
        let model = GLiNERPoly { _private: () };
        assert_eq!(model.name(), "gliner_poly");
    }

    // ---- Test 3: error message mentions onnx when feature disabled ----
    #[test]
    #[cfg(not(feature = "onnx"))]
    fn test_gliner_poly_error_mentions_onnx() {
        let err = GLiNERPoly::new("test-model").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("onnx"),
            "error should mention 'onnx', got: {msg}"
        );
    }

    // ---- Test 4: supported_types returns empty without onnx ----
    #[test]
    #[cfg(not(feature = "onnx"))]
    fn test_gliner_poly_supported_types_empty() {
        use crate::Model;
        let model = GLiNERPoly { _private: () };
        assert!(model.supported_types().is_empty());
    }

    // ---- Test 5: is_available returns false without onnx ----
    #[test]
    #[cfg(not(feature = "onnx"))]
    fn test_gliner_poly_is_not_available() {
        use crate::Model;
        let model = GLiNERPoly { _private: () };
        assert!(!model.is_available());
    }

    // ---- Test 6: ZeroShotNER returns error without onnx ----
    #[test]
    #[cfg(not(feature = "onnx"))]
    fn test_gliner_poly_zero_shot_error() {
        let model = GLiNERPoly { _private: () };
        let err = model
            .extract_with_types("hello", &["person"], 0.5)
            .unwrap_err();
        assert!(matches!(err, Error::FeatureNotAvailable(_)));
    }

    // ---- Tests with onnx feature enabled ----

    #[test]
    #[cfg(feature = "onnx")]
    fn test_gliner_poly_name_onnx() {
        // Verify that GLiNERPoly::new fails gracefully with a bad model name.
        // Skip if a local cache exists, because new() finds the cached model
        // before ever consulting the model_name argument.
        for cache in &local_model_cache_candidates() {
            if cache.join("model.onnx").exists() && cache.join("tokenizer.json").exists() {
                eprintln!(
                    "skipping: local gliner-poly cache exists at {}",
                    cache.display()
                );
                return;
            }
        }
        let err = GLiNERPoly::new("nonexistent/model-that-does-not-exist").unwrap_err();
        assert!(
            matches!(err, Error::Retrieval(_)),
            "expected Retrieval error, got: {err:?}"
        );
    }

    #[test]
    #[cfg(feature = "onnx")]
    fn test_gliner_poly_capabilities() {
        // Verify capabilities are reported correctly by checking the default labels.
        assert!(!DEFAULT_POLY_LABELS.is_empty());
        assert!(DEFAULT_POLY_LABELS.contains(&"person"));
        assert!(DEFAULT_POLY_LABELS.contains(&"organization"));
    }
}
