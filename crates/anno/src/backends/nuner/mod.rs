//! NuNER - Token-based zero-shot NER from NuMind.
//!
//! The Zero and Zero-4k checkpoints score each prompted type at each word;
//! adjacent words with the same accepted type form an entity. Zero-span
//! scores each type on a grid of contiguous word spans. The exported
//! `gliner_config.json` records the width of that grid.
//!
//! | Upstream checkpoint | Encoder | Span width |
//! |---|---|---|
//! | `numind/NuNER_Zero` | DeBERTa-v3-large | 1 |
//! | `numind/NuNER_Zero-4k` | Longformer-large-4096 | 1 |
//! | `numind/NuNER_Zero-span` | DeBERTa-v3-large | 12 |
//!
//! # Usage
//!
//! ```rust,ignore
//! use anno::NuNER;
//!
//! // Load NuNER model (requires `onnx` feature)
//! let ner = NuNER::from_pretrained("deepanwa/NuNerZero_onnx")?;
//!
//! // Zero-shot extraction with custom labels
//! let entities = ner.extract("Apple CEO Tim Cook announced...",
//!                            &["person", "organization", "product"], 0.5)?;
//! ```
//!
//! # References
//!
//! - [NuNER Zero on HuggingFace](https://huggingface.co/numind/NuNER_Zero)
//! - [NuNER ONNX](https://huggingface.co/deepanwa/NuNerZero_onnx)
//! - GLiNER paper (for span-based prompting inspiration)

use crate::{Entity, EntityCategory, EntityType, Language, Model, Result};

use crate::Error;

/// Encoded prompt result: (input_ids, attention_mask, word_mask, num_entity_types)
#[cfg(feature = "onnx")]
type EncodedPrompt = (Vec<i64>, Vec<i64>, Vec<i64>, i64);

/// Special token IDs for GLiNER/NuNER models (shared architecture)
#[cfg(feature = "onnx")]
const TOKEN_START: u32 = 1;
#[cfg(feature = "onnx")]
const TOKEN_END: u32 = 2;
#[cfg(feature = "onnx")]
const TOKEN_ENT: u32 = 128002;
#[cfg(feature = "onnx")]
const TOKEN_SEP: u32 = 128003;

/// NuNER Zero-shot NER model.
///
/// Runs label-conditioned word or span scoring from a NuNER Zero ONNX export.
///
/// # Model Variants
///
/// - `NuNER::new()` / `NuNER::from_pretrained("deepanwa/NuNerZero_onnx")` -- 512-token context,
///   chunks at ~2000 chars.
/// - `NuNER::new_4k()` / `NuNER::from_pretrained("/path/to/exported-4k")` -- 4096-token
///   context, chunks at ~16000 chars. Useful for long documents without heavy chunking.
///
/// `from_pretrained` auto-detects the 4k variant from the model ID and sets the chunk
/// threshold accordingly.
///
/// # Feature Requirements
///
/// Requires the `onnx` feature for actual inference. Without it, configuration
/// methods work and inference returns a feature-unavailable error.
///
/// # Example
///
/// ```rust,ignore
/// use anno::NuNER;
///
/// let ner = NuNER::from_pretrained("deepanwa/NuNerZero_onnx")?;
/// let entities = ner.extract(
///     "The CRISPR-Cas9 system was developed by Jennifer Doudna",
///     &["technology", "scientist"],
///     0.5
/// )?;
/// ```
pub struct NuNER {
    /// Model path or identifier
    model_id: String,
    /// Confidence threshold (0.0-1.0)
    threshold: f64,
    /// Whether model requires span tensors (detected on load)
    #[cfg(feature = "onnx")]
    requires_span_tensors: bool,
    #[cfg(feature = "onnx")]
    span_width: usize,
    /// Default entity labels for Model trait
    default_labels: Vec<String>,
    /// Max input chars before chunking. 512-token models use ~2000,
    /// 4096-token models use ~16000.
    max_input_chars: usize,
    /// ONNX session (when feature enabled)
    #[cfg(feature = "onnx")]
    session: Option<std::sync::Mutex<ort::session::Session>>,
    /// Tokenizer (when feature enabled)
    #[cfg(feature = "onnx")]
    tokenizer: Option<tokenizers::Tokenizer>,
}

mod inference;
// NuNER ONNX inference: see inference.rs
impl Default for NuNER {
    fn default() -> Self {
        Self::new()
    }
}

/// Default max input chars for 512-token NuNER models (~4 chars/token).
const MAX_INPUT_CHARS_512: usize = 2000;

/// Max input chars for 4096-token NuNER models (~4 chars/token).
const MAX_INPUT_CHARS_4K: usize = 16000;

impl Model for NuNER {
    fn extract_entities(&self, text: &str, _language: Option<Language>) -> Result<Vec<Entity>> {
        if text.trim().is_empty() {
            return Ok(vec![]);
        }

        #[cfg(feature = "onnx")]
        {
            if self.session.is_some() {
                let labels: Vec<&str> = self.default_labels.iter().map(|s| s.as_str()).collect();
                let threshold = self.threshold as f32;

                let max_chars = self.max_input_chars;
                if text.chars().count() > max_chars {
                    use crate::backends::chunking::{extract_chunked_parallel, ChunkConfig};
                    let config = ChunkConfig {
                        chunk_size: max_chars,
                        overlap: 200,
                        respect_sentences: true,
                        buffer_size: 1000,
                    };
                    return extract_chunked_parallel(text, &config, |chunk_text, char_offset| {
                        let mut entities = self.extract(chunk_text, &labels, threshold)?;
                        for e in &mut entities {
                            e.shift_by(char_offset); // atomic; avoids transient span inversion
                        }
                        Ok(entities)
                    });
                }

                return self.extract(text, &labels, threshold);
            }

            Err(Error::ModelInit(
                "NuNER model not loaded. Call `NuNER::from_pretrained(...)` (requires `onnx` feature) before calling `extract_entities`.".to_string(),
            ))
        }

        #[cfg(not(feature = "onnx"))]
        {
            Err(Error::FeatureNotAvailable(
                "NuNER requires the 'onnx' feature. Build with: cargo build --features onnx"
                    .to_string(),
            ))
        }
    }

    fn supported_types(&self) -> Vec<EntityType> {
        self.default_labels
            .iter()
            .map(|l| Self::map_label_to_entity_type(l))
            .collect()
    }

    fn is_available(&self) -> bool {
        #[cfg(feature = "onnx")]
        {
            self.session.is_some()
        }
        #[cfg(not(feature = "onnx"))]
        {
            false
        }
    }

    fn name(&self) -> &'static str {
        "nuner"
    }

    fn description(&self) -> &'static str {
        "NuNER Zero: Token-based zero-shot NER from NuMind (MIT licensed)"
    }

    fn version(&self) -> String {
        format!("nuner-zero-{}", self.model_id)
    }

    fn as_zero_shot(&self) -> Option<&dyn crate::backends::inference::ZeroShotNER> {
        // The `ZeroShotNER` impl below is gated on `onnx`; the default `None`
        // applies when `onnx` is off.
        #[cfg(feature = "onnx")]
        {
            Some(self)
        }
        #[cfg(not(feature = "onnx"))]
        {
            None
        }
    }
}

#[cfg(feature = "onnx")]
impl crate::backends::inference::ZeroShotNER for NuNER {
    fn extract_with_types(
        &self,
        text: &str,
        entity_types: &[&str],
        threshold: f32,
    ) -> crate::Result<Vec<crate::Entity>> {
        self.extract(text, entity_types, threshold)
    }

    fn extract_with_descriptions(
        &self,
        text: &str,
        descriptions: &[&str],
        threshold: f32,
    ) -> crate::Result<Vec<crate::Entity>> {
        // NuNER uses token-level bi-encoder, so descriptions work the same as type labels.
        self.extract(text, descriptions, threshold)
    }

    fn default_types(&self) -> &[&'static str] {
        &["person", "organization", "location", "date", "event"]
    }
}

#[cfg(test)]
mod tests;
