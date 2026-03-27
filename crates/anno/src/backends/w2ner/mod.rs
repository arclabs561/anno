//! W2NER - Unified NER via Word-Word Relation Classification.
//!
//! W2NER (Word-to-Word NER) models NER as classifying relations between
//! every pair of words in a sentence. This handles:
//!
//! - **Nested entities**: "The \[University of \[California\]\]"
//! - **Discontinuous entities**: "severe \[pain\] ... in \[abdomen\]" *(see limitation below)*
//! - **Overlapping entities**: Same span, different types
//!
//! # Discontinuous Entities
//!
//! True discontinuous entity decoding is implemented via [`W2NER::decode_discontinuous_from_matrix`].
//! The algorithm follows arXiv:2112.10070 §3.3: THW cells identify entity boundaries and NNW cells
//! identify adjacent-word connections within the same entity; gaps in the NNW chain produce disjoint
//! sub-spans.
//!
//! Use `extract_discontinuous()` (the [`DiscontinuousNER`] trait) when you need
//! non-contiguous span support. The standard `extract_entities()` returns only
//! contiguous spans for backwards-compatibility with the flat `Entity` type.
//!
//! # Language Support (Important Limitation)
//!
//! **This implementation uses whitespace tokenization** (`split_whitespace()`),
//! which works correctly for:
//!
//! - **Latin-script languages**: English, German, French, Spanish, etc.
//! - **Cyrillic**: Russian, Ukrainian, etc.
//! - **Languages with explicit word boundaries**
//!
//! It does **NOT** work correctly for:
//!
//! - **CJK languages** (Chinese, Japanese, Korean): No whitespace between words
//! - **Thai, Khmer, Lao**: Scriptio continua (no word boundaries)
//! - **Languages requiring morphological analysis**
//!
//! If you need CJK/Thai support, consider:
//! 1. Pre-tokenizing with a proper segmenter (e.g., jieba, mecab, pythainlp)
//! 2. Using a different backend (e.g., GLiNER with subword tokenization)
//!
//! The `language` parameter to [`Model::extract_entities`] is currently ignored,
//! but a warning is logged if a non-whitespace language is detected.
//!
//! # Architecture
//!
//! ```text
//! Input: "New York City is great"
//!
//!        ┌─────────────────────────────┐
//!        │      Encoder (BERT)          │
//!        └─────────────────────────────┘
//!                     │
//!        ┌─────────────────────────────┐
//!        │    Biaffine Attention        │
//!        │    (word-word scoring)       │
//!        └─────────────────────────────┘
//!                     │
//!        ┌───────────────────────────────┐
//!        │     Word-Word Grid (N×N×L)    │
//!        │  ┌───┬───┬───┬───┬───┐       │
//!        │  │   │New│York│City│...│      │
//!        │  ├───┼───┼───┼───┼───┤       │
//!        │  │New│ B │NNW│THW│   │       │
//!        │  ├───┼───┼───┼───┼───┤       │
//!        │  │Yrk│   │ B │NNW│   │       │
//!        │  ├───┼───┼───┼───┼───┤       │
//!        │  │Cty│   │   │ B │   │       │
//!        │  └───┴───┴───┴───┴───┘       │
//!        └───────────────────────────────┘
//!
//! Legend:
//!   B   = Begin entity
//!   NNW = Next-Neighboring-Word (same entity)
//!   THW = Tail-Head-Word (entity boundary)
//! ```
//!
//! # Grid Labels
//!
//! W2NER uses three relation types for each entity label:
//!
//! - **NNW (Next-Neighboring-Word)**: Token i and j are adjacent in same entity
//! - **THW (Tail-Head-Word)**: Token i is tail, token j is head of entity
//! - **None**: No relation
//!
//! # Usage
//!
//! ```rust,ignore
//! use anno::W2NER;
//!
//! // Load W2NER model (requires `onnx` feature)
//! let w2ner = W2NER::from_pretrained("path/to/w2ner-model")?;
//!
//! let text = "The University of California Berkeley";
//! let entities = w2ner.extract_entities(text, None)?;
//! // Returns nested entities: ORG + nested LOC
//! ```
//!
//! # References
//!
//! - [W2NER Paper](https://arxiv.org/abs/2112.10070) (AAAI 2022)
//! - [TPLinker](https://aclanthology.org/2020.coling-main.138/) (related approach)

pub mod decode;
pub use decode::{map_label_to_entity_type, DiscontinuousDecodeRow, W2NERRelation};

use crate::backends::inference::{DiscontinuousEntity, DiscontinuousNER, HandshakingMatrix};
use crate::{Confidence, Entity, EntityType, Language, Model, Result};

#[cfg(feature = "onnx")]
use crate::Error;

/// Configuration for W2NER decoding.
///
/// # Tokenization
///
/// W2NER uses **whitespace tokenization** (`split_whitespace()`), which works
/// for Latin-script languages but fails for CJK/Thai/Lao. See module-level
/// docs for details and workarounds.
#[derive(Debug, Clone)]
pub struct W2NERConfig {
    /// Confidence threshold for grid predictions
    pub threshold: Confidence,
    /// Entity type labels (maps grid channels to types)
    pub entity_labels: Vec<String>,
    /// Whether to extract nested entities
    pub allow_nested: bool,
    /// Whether to extract discontinuous entities.
    ///
    /// **Note**: Currently, discontinuous decoding is not fully implemented.
    /// This flag exists for forward-compatibility; setting it to `true` does
    /// not yet produce true discontinuous spans. See `backend-02` in docs.
    pub allow_discontinuous: bool,
    /// Model identifier for loading
    pub model_id: String,
}

impl Default for W2NERConfig {
    fn default() -> Self {
        Self {
            threshold: Confidence::new(0.5),
            entity_labels: vec!["PER".to_string(), "ORG".to_string(), "LOC".to_string()],
            allow_nested: true,
            allow_discontinuous: true,
            model_id: String::new(),
        }
    }
}

/// W2NER model for unified named entity recognition.
///
/// Uses word-word relation classification to handle complex entity
/// structures (nested, overlapping, discontinuous).
///
/// # Feature Requirements
///
/// Requires the `onnx` feature for actual inference. Without it, only the
/// [`decode_from_matrix`](Self::decode_from_matrix) method works with
/// pre-computed grids.
///
/// # Example
///
/// ```rust,ignore
/// let w2ner = W2NER::from_pretrained("ljynlp/w2ner-bert-base")?;
///
/// // Handles nested entities naturally
/// let text = "The University of California Berkeley";
/// let entities = w2ner.extract_entities(text, None)?;
/// ```
pub struct W2NER {
    config: W2NERConfig,
    #[cfg(feature = "onnx")]
    session: Option<std::sync::Mutex<ort::session::Session>>,
    #[cfg(feature = "onnx")]
    tokenizer: Option<tokenizers::Tokenizer>,
}

impl W2NER {
    /// Create W2NER with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: W2NERConfig::default(),
            #[cfg(feature = "onnx")]
            session: None,
            #[cfg(feature = "onnx")]
            tokenizer: None,
        }
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(config: W2NERConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "onnx")]
            session: None,
            #[cfg(feature = "onnx")]
            tokenizer: None,
        }
    }

    /// Load W2NER model from path or HuggingFace.
    ///
    /// Automatically loads `.env` for HF_TOKEN if present.
    ///
    /// # Arguments
    /// * `model_path` - Local path or HuggingFace model ID
    #[cfg(feature = "onnx")]
    pub fn from_pretrained(model_path: &str) -> Result<Self> {
        use crate::backends::hf_loader;
        use std::path::Path;
        use std::process::Command;

        let (model_file, tokenizer_file) = if Path::new(model_path).exists() {
            // Local path
            let model_file = Path::new(model_path).join("model.onnx");
            let tokenizer_file = Path::new(model_path).join("tokenizer.json");
            (model_file, tokenizer_file)
        } else {
            // HuggingFace download
            let api = hf_loader::hf_api()?;
            let repo = api.model(model_path.to_string());

            let (model_file, tokenizer_file) = match repo
                .get("model.onnx")
                .or_else(|_| repo.get("onnx/model.onnx"))
            {
                Ok(p) => {
                    let tok = repo.get("tokenizer.json").map_err(|e| {
                        Error::Retrieval(format!("Failed to download tokenizer: {}", e))
                    })?;
                    (p, tok)
                }
                Err(e) => {
                    let error_msg = format!("{e}");
                    // Check if it's an authentication error (401) or gated model
                    if error_msg.contains("401") || error_msg.contains("Unauthorized") {
                        return Err(Error::Retrieval(format!(
                            "W2NER model '{}' requires HuggingFace authentication.\n\
                             \n\
                             To fix this:\n\
                             1. Get a HuggingFace token from https://huggingface.co/settings/tokens\n\
                             2. Request access to the model on HuggingFace (if it's gated)\n\
                             3. Set the token: export HF_TOKEN=your_token_here (or HF_API_TOKEN)\n\
                             \n\
                             Alternative: set W2NER_MODEL_PATH to a local export (see scripts/export_w2ner_to_onnx.py).",
                            model_path
                        )));
                    }

                    // 404 / missing ONNX is common: HF repos typically don't ship `model.onnx`.
                    // We can auto-export a local ONNX model (bounded by env + CI) and proceed.
                    //
                    // IMPORTANT: many dev shells set `CI=1`, which should not disable auto-export
                    // when running locally. Only treat GitHub Actions as “CI” for this purpose.
                    let in_github_actions = std::env::var("GITHUB_ACTIONS").is_ok();
                    let auto_export = match std::env::var("ANNO_W2NER_AUTO_EXPORT").ok() {
                        None => !in_github_actions,
                        Some(v) => {
                            let t = v.trim().to_lowercase();
                            t == "1" || t == "true" || t == "yes" || t == "y" || t == "on"
                        }
                    };

                    if auto_export {
                        let Some(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR").ok() else {
                            return Err(Error::Retrieval(format!(
                                "W2NER model '{}' is missing ONNX files, and auto-export is enabled, but CARGO_MANIFEST_DIR is not set.\n\
                                 \n\
                                 Fix:\n\
                                 - Run from the repo via cargo (so CARGO_MANIFEST_DIR is present), or\n\
                                 - Export manually and set W2NER_MODEL_PATH to the export directory.\n\
                                 \n\
                                 Original error: {e}",
                                model_path
                            )));
                        };

                        // Export location under the cache dir.
                        //
                        // IMPORTANT: `anno::eval` is feature-gated, so backends must not depend on
                        // it. Mirror the cache-root logic in a lightweight way here.
                        let cache_dir = std::env::var("ANNO_CACHE_DIR")
                            .ok()
                            .filter(|v| !v.trim().is_empty())
                            .map(std::path::PathBuf::from)
                            .unwrap_or_else(|| {
                                dirs::cache_dir()
                                    .unwrap_or_else(|| std::path::PathBuf::from("."))
                                    .join("anno")
                            });
                        // Export model choice: default to a public BERT id so auto-export works
                        // even when the configured W2NER HF repo is gated.
                        let export_bert_model = std::env::var("W2NER_EXPORT_BERT_MODEL")
                            .ok()
                            .filter(|v| !v.trim().is_empty())
                            .unwrap_or_else(|| "bert-base-cased".to_string());
                        let safe_id = export_bert_model
                            .chars()
                            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
                            .collect::<String>();
                        let out_dir = cache_dir.join("models").join("w2ner").join(safe_id);
                        std::fs::create_dir_all(&out_dir).map_err(|ioe| {
                            Error::Retrieval(format!(
                                "Failed to create W2NER export dir {:?}: {}",
                                out_dir, ioe
                            ))
                        })?;

                        let script_path = std::path::PathBuf::from(manifest_dir)
                            .join("../../scripts/export_w2ner_to_onnx.py");
                        let out_onnx = out_dir.join("model.onnx");

                        // Run export via `uv`, which is expected in dev environments.
                        let mut cmd = Command::new("uv");
                        cmd.arg("run")
                            .arg(script_path)
                            .arg("--bert-model")
                            .arg(&export_bert_model)
                            .arg("--output")
                            .arg(&out_onnx);

                        let output = cmd.output().map_err(|ioe| {
                            Error::Retrieval(format!(
                                "Failed to spawn W2NER auto-export (uv): {}",
                                ioe
                            ))
                        })?;
                        if !output.status.success() {
                            let stderr = String::from_utf8_lossy(&output.stderr);
                            let stdout = String::from_utf8_lossy(&output.stdout);
                            return Err(Error::Retrieval(format!(
                                "W2NER auto-export failed (exit={}).\n\
                                 \n\
                                 stdout:\n{}\n\
                                 \n\
                                 stderr:\n{}\n\
                                 \n\
                                 Original HF error: {e}",
                                output.status.code().unwrap_or(-1),
                                stdout,
                                stderr
                            )));
                        }

                        // Tokenizer is saved alongside the ONNX by the export script.
                        let tok = out_dir.join("tokenizer.json");
                        if !out_onnx.exists() || !tok.exists() {
                            return Err(Error::Retrieval(format!(
                                "W2NER auto-export succeeded but expected files are missing.\n\
                                 expected: {:?} and {:?}",
                                out_onnx, tok
                            )));
                        }

                        (out_onnx, tok)
                    } else {
                        return Err(Error::Retrieval(format!(
                            "W2NER model '{}' not found or missing ONNX files.\n\
                             \n\
                             The model may be:\n\
                             - A gated model requiring access approval at https://huggingface.co/{}\n\
                             - Missing pre-exported ONNX files (model.onnx or onnx/model.onnx)\n\
                             - Removed or renamed on HuggingFace\n\
                             \n\
                             Fix options:\n\
                             - Set ANNO_W2NER_AUTO_EXPORT=1 (dev) to auto-export to ONNX\n\
                             - Or export manually and set W2NER_MODEL_PATH to the export directory\n\
                             \n\
                             If you have HF_TOKEN set, ensure you've requested and received access to this model.\n\
                             Alternative: Use nuner, gliner2, or other available NER backends.\n\
                             \n\
                             Original error: {e}",
                            model_path, model_path
                        )));
                    }
                }
            };

            (model_file, tokenizer_file)
        };

        let session =
            hf_loader::create_onnx_session(&model_file, hf_loader::OnnxSessionConfig::default())?;
        let tokenizer = hf_loader::load_tokenizer(&tokenizer_file)?;

        log::debug!("[W2NER] Loaded model");

        Ok(Self {
            config: W2NERConfig {
                model_id: model_path.to_string(),
                ..Default::default()
            },
            session: Some(std::sync::Mutex::new(session)),
            tokenizer: Some(tokenizer),
        })
    }

    /// Set confidence threshold.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.config.threshold = Confidence::new(threshold);
        self
    }

    /// Set entity type labels.
    #[must_use]
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        self.config.entity_labels = labels;
        self
    }

    /// Enable/disable nested entity extraction.
    #[must_use]
    pub fn with_nested(mut self, allow: bool) -> Self {
        self.config.allow_nested = allow;
        self
    }

    /// Decode entities from a handshaking matrix.
    ///
    /// Delegates to [`decode::decode_from_matrix`] with this model's threshold
    /// and `allow_nested` settings. See that function for the full algorithm.
    pub fn decode_from_matrix(
        &self,
        matrix: &HandshakingMatrix,
        tokens: &[&str],
        entity_type_idx: usize,
    ) -> Vec<(usize, usize, f64)> {
        decode::decode_from_matrix(
            matrix,
            tokens,
            entity_type_idx,
            self.config.threshold.value() as f32,
            self.config.allow_nested,
        )
    }

    /// Decode discontinuous entities from a handshaking matrix.
    ///
    /// Delegates to [`decode::decode_discontinuous_from_matrix`].
    pub fn decode_discontinuous_from_matrix(
        &self,
        matrix: &HandshakingMatrix,
        tokens: &[&str],
        threshold: f32,
    ) -> Vec<DiscontinuousDecodeRow> {
        let first_label = self
            .config
            .entity_labels
            .first()
            .map(|s| s.as_str())
            .unwrap_or("");
        decode::decode_discontinuous_from_matrix(matrix, tokens, threshold, first_label)
    }

    /// Decode dense grid output to HandshakingMatrix.
    ///
    /// Delegates to [`decode::grid_to_matrix`].
    pub fn grid_to_matrix(
        grid: &[f32],
        seq_len: usize,
        num_relations: usize,
        threshold: f32,
    ) -> HandshakingMatrix {
        decode::grid_to_matrix(grid, seq_len, num_relations, threshold)
    }

    /// Run inference with ONNX model.
    #[cfg(feature = "onnx")]
    pub fn extract_with_grid(&self, text: &str, threshold: f32) -> Result<Vec<Entity>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        let session = self.session.as_ref().ok_or_else(|| {
            Error::Retrieval("Model not loaded. Call from_pretrained() first.".to_string())
        })?;

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::Retrieval("Tokenizer not loaded.".to_string()))?;

        // Tokenize via whitespace splitting.
        //
        // LIMITATION: This only works for languages with explicit word boundaries
        // (Latin, Cyrillic, etc.). CJK/Thai/Khmer/Lao will produce single "words"
        // for entire sentences, breaking entity extraction. See module docs.
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Ok(vec![]);
        }

        let encoding = tokenizer
            .encode(text.to_string(), true)
            .map_err(|e| Error::Parse(format!("Tokenization failed: {}", e)))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect();
        let seq_len = input_ids.len();

        // Build tensors
        use ndarray::Array2;

        let input_ids_arr = Array2::from_shape_vec((1, seq_len), input_ids)
            .map_err(|e| Error::Parse(format!("Array error: {}", e)))?;
        let attention_arr = Array2::from_shape_vec((1, seq_len), attention_mask)
            .map_err(|e| Error::Parse(format!("Array error: {}", e)))?;

        let input_ids_t = super::ort_compat::tensor_from_ndarray(input_ids_arr)
            .map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;
        let attention_t = super::ort_compat::tensor_from_ndarray(attention_arr)
            .map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;

        // Run inference with blocking lock for thread-safe parallel access
        let mut session_guard = session.lock().unwrap_or_else(|e| e.into_inner());

        let outputs = session_guard
            .run(ort::inputs![
                "input_ids" => input_ids_t.into_dyn(),
                "attention_mask" => attention_t.into_dyn(),
            ])
            .map_err(|e| Error::Parse(format!("Inference failed: {}", e)))?;

        // Decode grid output
        let output = outputs
            .iter()
            .next()
            .map(|(_, v)| v)
            .ok_or_else(|| Error::Parse("No output".to_string()))?;

        let (_, data) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Parse(format!("Extract failed: {}", e)))?;
        let grid: Vec<f32> = data.to_vec();

        // Convert grid to matrix and decode
        let num_relations = 3; // None, NNW, THW
        let matrix = Self::grid_to_matrix(&grid, seq_len, num_relations, threshold);

        // Calculate word positions
        // Note: This assumes words appear in order and don't overlap.
        // If a word appears multiple times, this will find the first occurrence
        // after the previous word. This is correct for tokenized input where
        // words are in sequence, but may fail if words are out of order.
        let word_positions: Vec<(usize, usize)> = {
            // Performance: Pre-allocate positions vec with known size
            let mut positions = Vec::with_capacity(words.len());
            let mut pos = 0;
            for (idx, word) in words.iter().enumerate() {
                if let Some(start) = text[pos..].find(word) {
                    let abs_start = pos + start;
                    let abs_end = abs_start + word.len();
                    // Validate position is after previous word (words should be in order)
                    if !positions.is_empty() {
                        let (_prev_start, prev_end) = positions[positions.len() - 1];
                        if abs_start < prev_end {
                            log::warn!(
                                "Word '{}' (index {}) at position {} overlaps with previous word ending at {}",
                                word,
                                idx,
                                abs_start,
                                prev_end
                            );
                        }
                    }
                    positions.push((abs_start, abs_end));
                    pos = abs_end;
                } else {
                    // Word not found - return error to prevent silent entity skipping
                    return Err(Error::Parse(format!(
                        "Word '{}' (index {}) not found in text starting at position {}",
                        word, idx, pos
                    )));
                }
            }
            positions
        };

        // Validate that we found positions for all words
        if word_positions.len() != words.len() {
            return Err(Error::Parse(format!(
                "Word position mismatch: found {} positions for {} words",
                word_positions.len(),
                words.len()
            )));
        }

        // Word positions are byte offsets; `Entity` requires character offsets.
        let span_converter = crate::offset::SpanConverter::new(text);

        // Performance: Pre-allocate entities vec with estimated capacity
        // Decode entities for each type
        let mut entities = Vec::with_capacity(16);
        for (type_idx, label) in self.config.entity_labels.iter().enumerate() {
            let spans = self.decode_from_matrix(&matrix, &words.to_vec(), type_idx);

            for (start_word, end_word, score) in spans {
                if let (Some(&(start_pos, _)), Some(&(_, end_pos))) = (
                    word_positions.get(start_word),
                    word_positions.get(end_word.saturating_sub(1)),
                ) {
                    if let Some(entity_text) = text.get(start_pos..end_pos) {
                        entities.push(Entity::new(
                            entity_text,
                            decode::map_label_to_entity_type(label),
                            span_converter.byte_to_char(start_pos),
                            span_converter.byte_to_char_ceil(end_pos),
                            score,
                        ));
                    }
                }
            }
        }

        Ok(entities)
    }

    /// Full discontinuous-NER extraction using the NNW+THW grid decoding algorithm.
    ///
    /// Called by `extract_discontinuous` when an ONNX session is loaded.
    #[cfg(feature = "onnx")]
    fn extract_discontinuous_with_nnw(
        &self,
        text: &str,
        threshold: f32,
    ) -> Result<Vec<DiscontinuousEntity>> {
        use ndarray::Array2;

        let session = self
            .session
            .as_ref()
            .ok_or_else(|| Error::Retrieval("Model not loaded.".to_string()))?;
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::Retrieval("Tokenizer not loaded.".to_string()))?;

        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Ok(vec![]);
        }

        let encoding = tokenizer
            .encode(text.to_string(), true)
            .map_err(|e| Error::Parse(format!("Tokenization failed: {}", e)))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect();
        let seq_len = input_ids.len();

        let input_ids_arr = Array2::from_shape_vec((1, seq_len), input_ids)
            .map_err(|e| Error::Parse(format!("Array error: {}", e)))?;
        let attention_arr = Array2::from_shape_vec((1, seq_len), attention_mask)
            .map_err(|e| Error::Parse(format!("Array error: {}", e)))?;
        let input_ids_t = super::ort_compat::tensor_from_ndarray(input_ids_arr)
            .map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;
        let attention_t = super::ort_compat::tensor_from_ndarray(attention_arr)
            .map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;

        let grid: Vec<f32> = {
            let mut session_guard = session.lock().unwrap_or_else(|e| e.into_inner());
            let outputs = session_guard
                .run(ort::inputs![
                    "input_ids" => input_ids_t.into_dyn(),
                    "attention_mask" => attention_t.into_dyn(),
                ])
                .map_err(|e| Error::Parse(format!("Inference failed: {}", e)))?;
            let output = outputs
                .iter()
                .next()
                .map(|(_, v)| v)
                .ok_or_else(|| Error::Parse("No output".to_string()))?;
            let (_, data) = output
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Parse(format!("Extract failed: {}", e)))?;
            data.to_vec()
        }; // session_guard + outputs dropped here

        let num_relations = 3; // None, NNW, THW
        let matrix = Self::grid_to_matrix(&grid, seq_len, num_relations, threshold);

        // Compute word → byte position map
        let word_positions: Vec<(usize, usize)> = {
            let mut positions = Vec::with_capacity(words.len());
            let mut pos = 0;
            for word in &words {
                if let Some(start) = text[pos..].find(word) {
                    let abs_start = pos + start;
                    let abs_end = abs_start + word.len();
                    positions.push((abs_start, abs_end));
                    pos = abs_end;
                } else {
                    return Err(Error::Parse(format!("Word '{}' not found", word)));
                }
            }
            positions
        };

        let span_converter = crate::offset::SpanConverter::new(text);

        // Decode with NNW-aware discontinuous algorithm
        let decoded = self.decode_discontinuous_from_matrix(&matrix, &words, threshold);
        let mut entities = Vec::new();
        for (type_label, word_spans, score) in decoded {
            // Convert word-index spans to character-offset spans
            let mut char_spans: Vec<(usize, usize)> = Vec::new();
            let mut valid = true;
            for (ws, we) in &word_spans {
                let word_start = *ws;
                let word_end = we.saturating_sub(1);
                if let (Some(&(byte_start, _)), Some(&(_, byte_end))) =
                    (word_positions.get(word_start), word_positions.get(word_end))
                {
                    char_spans.push((
                        span_converter.byte_to_char(byte_start),
                        span_converter.byte_to_char_ceil(byte_end),
                    ));
                } else {
                    valid = false;
                    break;
                }
            }
            if !valid || char_spans.is_empty() {
                continue;
            }

            // Reconstruct entity text from all spans
            let entity_text: String = word_spans
                .iter()
                .filter_map(|(ws, we)| {
                    let last = we.saturating_sub(1);
                    let byte_start = word_positions.get(*ws)?.0;
                    let byte_end = word_positions.get(last)?.1;
                    text.get(byte_start..byte_end)
                })
                .collect::<Vec<_>>()
                .join(" ");

            entities.push(DiscontinuousEntity {
                spans: char_spans,
                text: entity_text,
                entity_type: type_label,
                confidence: Confidence::new(score),
            });
        }

        Ok(entities)
    }
}

impl Default for W2NER {
    fn default() -> Self {
        Self::new()
    }
}

impl Model for W2NER {
    fn extract_entities(&self, text: &str, language: Option<Language>) -> Result<Vec<Entity>> {
        if text.trim().is_empty() {
            return Ok(vec![]);
        }

        // Warn if the language hint suggests a non-whitespace-tokenized language.
        // W2NER uses `split_whitespace()`, which doesn't work for CJK/Thai/etc.
        if let Some(lang) = language {
            if lang.is_cjk() {
                log::warn!(
                    "[W2NER] Language '{}' detected, but W2NER uses whitespace tokenization \
                     which does not work correctly for CJK languages. \
                     Consider pre-tokenizing or using a different backend (e.g., GLiNER).",
                    lang
                );
            }
        }

        #[cfg(feature = "onnx")]
        {
            if self.session.is_some() {
                return self.extract_with_grid(text, self.config.threshold.value() as f32);
            }

            Err(crate::Error::ModelInit(
                "W2NER model not loaded. Call `W2NER::from_pretrained(...)` (requires `onnx` feature) before calling `extract_entities`.".to_string(),
            ))
        }

        #[cfg(not(feature = "onnx"))]
        {
            Err(crate::Error::FeatureNotAvailable(
                "W2NER requires the 'onnx' feature. Build with: cargo build --features onnx"
                    .to_string(),
            ))
        }
    }

    fn supported_types(&self) -> Vec<EntityType> {
        self.config
            .entity_labels
            .iter()
            .map(|l| decode::map_label_to_entity_type(l))
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
        "w2ner"
    }

    fn description(&self) -> &'static str {
        "W2NER: Unified NER via Word-Word Relation Classification (nested/discontinuous support)"
    }

    fn version(&self) -> String {
        format!("w2ner-{}", self.config.model_id)
    }

    fn capabilities(&self) -> crate::ModelCapabilities {
        crate::ModelCapabilities {
            discontinuous_capable: true,
            ..Default::default()
        }
    }
}

// =============================================================================
// DiscontinuousNER Trait Implementation
// =============================================================================

impl DiscontinuousNER for W2NER {
    /// Extract entities with discontinuous span support via the full NNW+THW decoding algorithm.
    ///
    /// Uses `decode_discontinuous_from_matrix` (arXiv:2112.10070 §3.3):
    /// THW cells identify entity boundaries; NNW cells identify adjacent word pairs within
    /// the same entity.  Gaps in the NNW chain produce disjoint sub-spans, yielding true
    /// discontinuous entities (e.g. "severe … pain" → two spans).
    fn extract_discontinuous(
        &self,
        text: &str,
        entity_types: &[&str],
        threshold: f32,
    ) -> Result<Vec<DiscontinuousEntity>> {
        if text.trim().is_empty() {
            return Ok(vec![]);
        }

        #[cfg(feature = "onnx")]
        {
            if self.session.is_some() {
                return self.extract_discontinuous_with_nnw(text, threshold);
            }
        }

        let _ = (entity_types, threshold);

        #[cfg(feature = "onnx")]
        {
            Err(crate::Error::ModelInit(
                "W2NER model not loaded. Call `W2NER::from_pretrained(...)` (requires `onnx` feature) before calling `extract_discontinuous`.".to_string(),
            ))
        }

        #[cfg(not(feature = "onnx"))]
        {
            Err(crate::Error::FeatureNotAvailable(
                "W2NER requires the 'onnx' feature. Build with: cargo build --features onnx"
                    .to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests;
