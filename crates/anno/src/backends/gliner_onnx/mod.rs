//! GLiNER-based NER implementation using ONNX Runtime.
//!
//! GLiNER (Generalist and Lightweight Model for Named Entity Recognition) is
//! a popular approach to “open/zero-shot” NER. This implementation follows the GLiNER prompt format
//! and common community conventions.
//!
//! ## Prompt Format
//!
//! GLiNER uses a special prompt format:
//!
//! ```text
//! [START] <<ENT>> type1 <<ENT>> type2 <<SEP>> word1 word2 ... [END]
//! ```
//!
//! Token IDs (for GLiNER tokenizer):
//! - START = 1
//! - END = 2
//! - `<<ENT>>` = 128002
//! - `<<SEP>>` = 128003
//!
//! ## Key Insight
//!
//! Each word is encoded SEPARATELY, preserving word boundaries.
//! Output shape: [batch, num_words, max_width, num_entity_types]

#![allow(missing_docs)]
#![allow(clippy::type_complexity)] // Complex return tuples
#![allow(clippy::manual_contains)] // Shape check style
#![allow(unused_variables)] // Feature-gated code
#![allow(clippy::items_after_test_module)] // Large file; keep local tests near helpers
#![allow(unused_imports)] // EntityType used conditionally

use crate::{Entity, Error, Language, Result};
use anno_core::{EntityCategory, EntityType};
#[cfg(feature = "onnx")]
use std::collections::HashMap;
#[cfg(feature = "onnx")]
use std::sync::Mutex;

/// Special token IDs for GLiNER models (defaults, overridden from gliner_config.json).
const TOKEN_START: u32 = 1;
const TOKEN_END: u32 = 2;
const DEFAULT_TOKEN_ENT: u32 = 128002;
const DEFAULT_TOKEN_SEP: u32 = 128003;

/// Default max span width from GLiNER config
const MAX_SPAN_WIDTH: usize = 12;

/// Configuration for GLiNER model loading.
#[cfg(feature = "onnx")]
pub mod config;
pub use config::*;

pub struct GLiNEROnnx {
    session: Mutex<ort::session::Session>,
    /// Arc-wrapped tokenizer for cheap cloning across threads.
    tokenizer: std::sync::Arc<tokenizers::Tokenizer>,
    /// HuggingFace model identifier (e.g., "onnx-community/gliner_small-v2.1").
    model_name: String,
    /// Whether a quantized model was loaded.
    is_quantized: bool,
    /// LRU cache for prompt encodings (keyed by text + entity types).
    prompt_cache: Option<Mutex<lru::LruCache<PromptCacheKey, PromptCacheValue>>>,
    /// Whether this model uses bi-encoder architecture (Stepanov et al., 2026).
    ///
    /// Bi-encoder models encode text and labels independently, enabling label
    /// embedding caching for higher throughput when many label types are used.
    encoder_mode: config::EncoderMode,
    /// Cached label embeddings for bi-encoder mode. Key: label string, value: embedding.
    ///
    /// Populated by `precompute_labels()`. In bi-encoder mode, if a label is not
    /// in this cache it will be encoded on-the-fly and cached for subsequent calls.
    label_cache: Mutex<HashMap<String, config::LabelEmbedding>>,
    /// Entity class token ID (`<<ENT>>` marker). Loaded from gliner_config.json.
    token_ent: u32,
    /// Separator token ID (`<<SEP>>` marker). Loaded from gliner_config.json.
    token_sep: u32,
    /// Whether the ONNX model expects span_idx/span_mask inputs.
    ///
    /// Token-level classifiers (e.g., gliner-pii-edge) don't have span inputs.
    /// Detected at load time from the ONNX session input names.
    has_span_inputs: bool,
    /// Separate ONNX session for the label encoder (bi-encoder models only).
    ///
    /// Loaded from `label_encoder.onnx` alongside the main `model.onnx`.
    /// Takes `labels_input_ids` + `labels_attention_mask`, outputs `labels_embeddings`.
    label_encoder_session: Option<Mutex<ort::session::Session>>,
    /// Tokenizer for the label encoder (may differ from text tokenizer in bi-encoder models).
    ///
    /// BGE models use a different tokenizer than DeBERTa. Falls back to the main
    /// tokenizer if no separate label tokenizer is found.
    label_tokenizer: Option<std::sync::Arc<tokenizers::Tokenizer>>,
}

#[cfg(feature = "onnx")]
mod inference;
pub(crate) use inference::expand_ner_label;
#[cfg(feature = "onnx")]
pub(crate) use inference::looks_like_company_name;
use inference::DEFAULT_GLINER_LABELS;
/// Approximate max input chars for GLiNER before chunking kicks in.
/// 512 tokens ~ 2000 chars for typical English text.
#[cfg(feature = "onnx")]
const MAX_INPUT_CHARS: usize = 2000;

impl crate::Model for GLiNEROnnx {
    fn extract_entities(
        &self,
        text: &str,
        _language: Option<Language>,
    ) -> crate::Result<Vec<Entity>> {
        // Use default labels for the Model trait interface
        // For custom labels, use the extract(text, labels, threshold) method directly
        #[cfg(feature = "onnx")]
        {
            if text.chars().count() > MAX_INPUT_CHARS {
                use crate::backends::chunking::{extract_chunked_parallel, ChunkConfig};
                let config = ChunkConfig {
                    chunk_size: MAX_INPUT_CHARS,
                    overlap: 200,
                    respect_sentences: true,
                    buffer_size: 1000,
                };
                return extract_chunked_parallel(text, &config, |chunk_text, char_offset| {
                    let mut entities = self.extract(chunk_text, DEFAULT_GLINER_LABELS, 0.5)?;
                    for e in &mut entities {
                        e.set_start(e.start() + char_offset);
                        e.set_end(e.end() + char_offset);
                    }
                    Ok(entities)
                });
            }
        }
        self.extract(text, DEFAULT_GLINER_LABELS, 0.5)
    }

    fn supported_types(&self) -> Vec<anno_core::EntityType> {
        // GLiNER supports any type via zero-shot - return the defaults
        DEFAULT_GLINER_LABELS
            .iter()
            .map(|label| anno_core::EntityType::Custom {
                name: (*label).to_string(),
                category: EntityCategory::Misc,
            })
            .collect()
    }

    fn is_available(&self) -> bool {
        true // If we got this far, it's available
    }

    fn name(&self) -> &'static str {
        "GLiNER-ONNX"
    }

    fn description(&self) -> &'static str {
        "Zero-shot NER using GLiNER with ONNX Runtime backend"
    }

    fn version(&self) -> String {
        // Version depends on the model weights, quantization, and encoder mode
        let quant = if self.is_quantized { "q" } else { "fp32" };
        let enc = match self.encoder_mode {
            config::EncoderMode::Uni => "uni",
            config::EncoderMode::Bi => "bi",
        };
        format!("gliner-onnx-{}-{}-{}", self.model_name, quant, enc)
    }

    fn as_zero_shot(&self) -> Option<&dyn crate::backends::inference::ZeroShotNER> {
        Some(self)
    }
}

#[cfg(feature = "onnx")]
impl crate::backends::inference::ZeroShotNER for GLiNEROnnx {
    fn extract_with_types(
        &self,
        text: &str,
        entity_types: &[&str],
        threshold: f32,
    ) -> crate::Result<Vec<Entity>> {
        self.extract(text, entity_types, threshold)
    }

    fn extract_with_descriptions(
        &self,
        text: &str,
        descriptions: &[&str],
        threshold: f32,
    ) -> crate::Result<Vec<Entity>> {
        // GLiNER encodes labels as text, so descriptions work the same way
        self.extract(text, descriptions, threshold)
    }

    fn default_types(&self) -> &[&'static str] {
        DEFAULT_GLINER_LABELS
    }
}

// =============================================================================
// Stub when feature disabled
// =============================================================================

crate::backends::macros::define_feature_stub! {
    struct GLiNEROnnx;
    feature = "onnx";
    name = "GLiNER-ONNX (unavailable)";
    description = "GLiNER with ONNX Runtime backend - requires 'onnx' feature";
    error_msg = "GLiNER-ONNX requires the 'onnx' feature";
    methods {
        /// Get the model name (stub).
        pub fn model_name(&self) -> &str {
            "gliner-not-enabled"
        }

        /// Extract entities (stub - requires onnx feature).
        pub fn extract(
            &self,
            _text: &str,
            _entity_types: &[&str],
            _threshold: f32,
        ) -> crate::Result<Vec<crate::Entity>> {
            Err(crate::Error::FeatureNotAvailable(
                "GLiNER-ONNX requires the 'onnx' feature".to_string(),
            ))
        }
    }
    impls {
        ZeroShotNER,
    }
}

// Overlap Removal
// =============================================================================

/// Remove overlapping entity spans intelligently.
///
/// Strategy:
/// 1. Prefer shorter spans when they have similar or higher confidence
///    (e.g., prefer "Department of Defense" over "The Department of Defense")
/// 2. For truly overlapping spans of similar length, keep highest confidence
/// 3. Handle comma-separated entities (e.g., "IBM, NASA" should become "IBM" + "NASA")
fn remove_overlapping_spans(mut entities: Vec<Entity>) -> Vec<Entity> {
    super::chunking::deduplicate_overlapping(
        &mut entities,
        super::chunking::OverlapStrategy::KeepShortest,
    );
    entities
}

#[cfg(test)]
mod postprocess_tests;
