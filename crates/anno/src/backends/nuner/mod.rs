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
//! NuNER uses the same bi-encoder architecture as GLiNER but with token classification:
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
//!               │
//!               ▼
//!        B-PER I-PER  O    O   B-ORG
//!        James Bond works at  MI6
//! ```
//!
//! # Differences from GLiNER (Span Mode)
//!
//! | Aspect | GLiNER (Span) | NuNER (Token) |
//! |--------|---------------|---------------|
//! | Output | Span classification | Token classification (BIO) |
//! | Entity length | Limited by span window (12) | Arbitrary |
//! | ONNX inputs | 6 tensors (incl span_idx) | 4 tensors (no span tensors) |
//! | Decoding | Span scores → entities | BIO tags → entities |
//!
//! # Model Variants
//!
//! | Model | Context | Notes |
//! |-------|---------|-------|
//! | `numind/NuNER_Zero` | 512 | General zero-shot |
//! | `numind/NuNER_Zero_4k` | 4096 | Long context variant |
//! | `deepanwa/NuNerZero_onnx` | 512 | Pre-converted ONNX |
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

/// Maximum span width for span-based inference.
/// NuNER uses max_width=1 (single-word spans only) per its gliner_config.json.
/// This matches the Python GLiNER implementation's prepare_span_idx function.
#[cfg(feature = "onnx")]
const MAX_SPAN_WIDTH: usize = 1;

/// NuNER Zero-shot NER model.
///
/// Token-based variant of GLiNER that uses BIO tagging instead of span classification.
/// This enables arbitrary-length entity extraction without the span window limitation.
///
/// # Feature Requirements
///
/// Requires the `onnx` feature for actual inference. Without it, configuration
/// methods work but extraction returns empty results.
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
    requires_span_tensors: std::sync::atomic::AtomicBool,
    /// Default entity labels for Model trait
    default_labels: Vec<String>,
    /// ONNX session (when feature enabled)
    #[cfg(feature = "onnx")]
    session: Option<crate::sync::Mutex<ort::session::Session>>,
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

/// Approximate max input chars for NuNER before chunking kicks in.
/// 512 tokens ~ 2000 chars for typical English text.
#[cfg(feature = "onnx")]
const MAX_INPUT_CHARS: usize = 2000;

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

                if text.chars().count() > MAX_INPUT_CHARS {
                    use crate::backends::streaming::{extract_chunked_parallel, ChunkConfig};
                    let config = ChunkConfig {
                        chunk_size: MAX_INPUT_CHARS,
                        overlap: 200,
                        respect_sentences: true,
                        buffer_size: 1000,
                    };
                    return extract_chunked_parallel(text, &config, |chunk_text, char_offset| {
                        let mut entities = self.extract(chunk_text, &labels, threshold)?;
                        for e in &mut entities {
                            e.start += char_offset;
                            e.end += char_offset;
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

    fn capabilities(&self) -> crate::ModelCapabilities {
        crate::ModelCapabilities {
            batch_capable: true,
            streaming_capable: true,
            dynamic_labels: true,
            ..Default::default()
        }
    }
}


#[cfg(feature = "onnx")]
impl crate::DynamicLabels for NuNER {
    fn extract_with_labels(
        &self,
        text: &str,
        labels: &[&str],
        _language: Option<Language>,
    ) -> crate::Result<Vec<crate::Entity>> {
        self.extract(text, labels, self.threshold as f32)
    }
}


#[cfg(test)]
mod tests;
