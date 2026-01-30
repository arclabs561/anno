//! W2NER - Unified NER via Word-Word Relation Classification.
//!
//! W2NER (Word-to-Word NER) models NER as classifying relations between
//! every pair of words in a sentence. This elegantly handles:
//!
//! - **Nested entities**: "The \[University of \[California\]\]"
//! - **Discontinuous entities**: "severe \[pain\] ... in \[abdomen\]" *(see limitation below)*
//! - **Overlapping entities**: Same span, different types
//!
//! # Discontinuous Entities (Important Limitation)
//!
//! **True discontinuous entity decoding is not yet implemented.** The W2NER
//! paper describes a grid-based algorithm for linking non-adjacent spans, but
//! this implementation currently returns only contiguous spans.
//!
//! The [`DiscontinuousNER`] trait is implemented for API compatibility, but
//! `extract_discontinuous()` wraps each contiguous entity into a single-segment
//! result. The `W2NERConfig.allow_discontinuous` flag exists for forward-compatibility
//! but does not change behavior today.
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

use crate::backends::inference::{
    DiscontinuousEntity, DiscontinuousNER, HandshakingCell, HandshakingMatrix,
};
use crate::{Entity, EntityType, Model, Result};

#[cfg(feature = "onnx")]
use crate::Error;

/// W2NER relation types for word-word classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum W2NERRelation {
    /// Next-Neighboring-Word: tokens are adjacent in same entity
    NNW,
    /// Tail-Head-Word: marks entity boundary (tail -> head)
    THW,
    /// No relation between tokens
    None,
}

impl W2NERRelation {
    /// Convert from label index.
    #[must_use]
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::None,
            1 => Self::NNW,
            2 => Self::THW,
            _ => Self::None,
        }
    }

    /// Convert to label index.
    #[must_use]
    pub fn to_index(self) -> usize {
        match self {
            Self::None => 0,
            Self::NNW => 1,
            Self::THW => 2,
        }
    }
}

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
    pub threshold: f64,
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
            threshold: 0.5,
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
    session: Option<crate::sync::Mutex<ort::session::Session>>,
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
        use hf_hub::api::sync::{Api, ApiBuilder};
        use ort::execution_providers::CPUExecutionProvider;
        use ort::session::Session;
        use std::path::Path;
        use std::process::Command;

        // Load .env if present (for HF_TOKEN)
        crate::env::load_dotenv();

        let (model_file, tokenizer_file) = if Path::new(model_path).exists() {
            // Local path
            let model_file = Path::new(model_path).join("model.onnx");
            let tokenizer_file = Path::new(model_path).join("tokenizer.json");
            (model_file, tokenizer_file)
        } else {
            // HuggingFace download - explicitly use token if available
            let api = if let Some(token) = crate::env::hf_token() {
                ApiBuilder::new()
                    .with_token(Some(token))
                    .build()
                    .map_err(|e| {
                        Error::Retrieval(format!(
                            "Failed to initialize HuggingFace API with token: {}",
                            e
                        ))
                    })?
            } else {
                Api::new().map_err(|e| {
                    Error::Retrieval(format!("Failed to initialize HuggingFace API: {}", e))
                })?
            };
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

        let session = Session::builder()
            .map_err(|e| Error::Retrieval(format!("Failed to create session: {}", e)))?
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .map_err(|e| Error::Retrieval(format!("Failed to set providers: {}", e)))?
            .commit_from_file(&model_file)
            .map_err(|e| Error::Retrieval(format!("Failed to load model: {}", e)))?;

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_file)
            .map_err(|e| Error::Retrieval(format!("Failed to load tokenizer: {}", e)))?;

        log::debug!("[W2NER] Loaded model");

        Ok(Self {
            config: W2NERConfig {
                model_id: model_path.to_string(),
                ..Default::default()
            },
            session: Some(crate::sync::Mutex::new(session)),
            tokenizer: Some(tokenizer),
        })
    }

    /// Set confidence threshold.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.config.threshold = threshold.clamp(0.0, 1.0);
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
    /// This is the core W2NER decoding algorithm that can be used with
    /// pre-computed grid predictions (e.g., from external inference).
    ///
    /// # Algorithm
    ///
    /// 1. Find all THW cells (entity boundaries)
    /// 2. For each THW(i,j), the entity spans from word j (head) to word i (tail)
    /// 3. Handle nested/overlapping entities based on config
    ///
    /// # Arguments
    ///
    /// * `matrix` - The predicted word-word relation grid
    /// * `tokens` - Original tokens for text reconstruction
    /// * `entity_type_idx` - Which entity type channel this is
    pub fn decode_from_matrix(
        &self,
        matrix: &HandshakingMatrix,
        tokens: &[&str],
        entity_type_idx: usize,
    ) -> Vec<(usize, usize, f64)> {
        // Performance: Pre-allocate entities vec with estimated capacity
        let mut entities = Vec::with_capacity(16);

        // Find all THW (Tail-Head-Word) markers
        // THW at (i,j) means: token i is tail, token j is head
        // Entity spans from j (head/start) to i (tail/end)
        for cell in &matrix.cells {
            let relation = W2NERRelation::from_index(cell.label_idx as usize);
            if relation == W2NERRelation::THW && cell.score >= self.config.threshold as f32 {
                let tail = cell.i as usize;
                let head = cell.j as usize;

                // Validate: head <= tail (head is start, tail is end)
                if head <= tail && head < tokens.len() && tail < tokens.len() {
                    entities.push((head, tail + 1, cell.score as f64));
                }
            }
        }

        // Performance: Use unstable sort (we don't need stable sort here)
        // Sort by start position, then by length (longer first for nested)
        entities.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| (b.1 - b.0).cmp(&(a.1 - a.0))));

        // Remove nested entities if not allowed
        if !self.config.allow_nested {
            entities = Self::remove_nested(&entities);
        }

        let _ = entity_type_idx; // May be used for multi-type grids
        entities
    }

    /// Decode dense grid output to HandshakingMatrix.
    ///
    /// # Arguments
    /// * `grid` - Dense grid of shape [seq_len, seq_len, num_relations]
    /// * `seq_len` - Sequence length
    /// * `threshold` - Score threshold for sparse representation
    pub fn grid_to_matrix(
        grid: &[f32],
        seq_len: usize,
        num_relations: usize,
        threshold: f32,
    ) -> HandshakingMatrix {
        let mut cells = Vec::new();

        for i in 0..seq_len {
            for j in 0..seq_len {
                for rel in 0..num_relations {
                    let idx = i * seq_len * num_relations + j * num_relations + rel;
                    if let Some(&score) = grid.get(idx) {
                        if score >= threshold && rel > 0 {
                            // rel > 0 excludes "None"
                            cells.push(HandshakingCell {
                                i: i as u32,
                                j: j as u32,
                                label_idx: rel as u16,
                                score,
                            });
                        }
                    }
                }
            }
        }

        HandshakingMatrix {
            cells,
            seq_len,
            num_labels: num_relations,
        }
    }

    /// Remove nested entities (keep outermost only).
    fn remove_nested(entities: &[(usize, usize, f64)]) -> Vec<(usize, usize, f64)> {
        let mut result = Vec::new();
        let mut last_end = 0;

        for &(start, end, score) in entities {
            if start >= last_end {
                result.push((start, end, score));
                last_end = end;
            }
        }

        result
    }

    /// Map label string to EntityType.
    fn map_label(label: &str) -> EntityType {
        match label.to_uppercase().as_str() {
            "PER" | "PERSON" => EntityType::Person,
            "ORG" | "ORGANIZATION" => EntityType::Organization,
            "LOC" | "LOCATION" | "GPE" => EntityType::Location,
            "DATE" => EntityType::Date,
            "TIME" => EntityType::Time,
            "MONEY" => EntityType::Money,
            "PERCENT" => EntityType::Percent,
            "MISC" => EntityType::Other("MISC".to_string()),
            _ => EntityType::Other(label.to_string()),
        }
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
        let mut session_guard = crate::sync::lock(session);

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
                            Self::map_label(label),
                            span_converter.byte_to_char(start_pos),
                            span_converter.byte_to_char(end_pos),
                            score,
                        ));
                    }
                }
            }
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
    fn extract_entities(&self, text: &str, language: Option<&str>) -> Result<Vec<Entity>> {
        if text.trim().is_empty() {
            return Ok(vec![]);
        }

        // Warn if the language hint suggests a non-whitespace-tokenized language.
        // W2NER uses `split_whitespace()`, which doesn't work for CJK/Thai/etc.
        if let Some(lang) = language {
            let lang_lower = lang.to_lowercase();
            let is_non_whitespace_lang = matches!(
                lang_lower.as_str(),
                "zh" | "zh-cn"
                    | "zh-tw"
                    | "chinese"
                    | "mandarin"
                    | "cantonese"
                    | "ja"
                    | "jp"
                    | "japanese"
                    | "ko"
                    | "kr"
                    | "korean"
                    | "th"
                    | "thai"
                    | "km"
                    | "khmer"
                    | "lo"
                    | "lao"
                    | "my"
                    | "burmese"
                    | "myanmar"
            );
            if is_non_whitespace_lang {
                log::warn!(
                    "[W2NER] Language '{}' detected, but W2NER uses whitespace tokenization \
                     which does not work correctly for CJK/Thai/Khmer/Lao. \
                     Consider pre-tokenizing or using a different backend (e.g., GLiNER).",
                    lang
                );
            }
        }

        #[cfg(feature = "onnx")]
        {
            if self.session.is_some() {
                return self.extract_with_grid(text, self.config.threshold as f32);
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
            .map(|l| Self::map_label(l))
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
}

// =============================================================================
// BatchCapable Trait Implementation
// =============================================================================

impl crate::BatchCapable for W2NER {
    fn optimal_batch_size(&self) -> Option<usize> {
        Some(4) // W2NER is more memory-intensive due to grid computation
    }
}

// =============================================================================
// StreamingCapable Trait Implementation
// =============================================================================

impl crate::StreamingCapable for W2NER {
    fn recommended_chunk_size(&self) -> usize {
        2048 // Smaller chunks due to grid memory requirements
    }
}

// =============================================================================
// DiscontinuousNER Trait Implementation
// =============================================================================

impl DiscontinuousNER for W2NER {
    /// Extract entities with discontinuous span support.
    ///
    /// # Current Limitation
    ///
    /// **True discontinuous decoding is not yet implemented.** This method
    /// currently wraps each contiguous entity into a single-segment
    /// `DiscontinuousEntity`. The W2NER paper describes a grid-based decoding
    /// algorithm for discontinuous entities, but this implementation does not
    /// yet decode those relations.
    ///
    /// If you need true discontinuous entity support, consider:
    /// 1. Post-processing with heuristics (e.g., linking "severe" to "pain")
    /// 2. Using a specialized discontinuous NER model
    ///
    /// This trait implementation exists for API compatibility and will be
    /// upgraded when true discontinuous decoding is implemented.
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
                // TODO(discontinuous): Implement true discontinuous decoding.
                //
                // The W2NER grid contains relation information that could be
                // used to link non-adjacent spans into discontinuous entities.
                // For now, we wrap each contiguous entity into a single-segment
                // DiscontinuousEntity for API compatibility.
                //
                // See: https://arxiv.org/abs/2112.10070 (Section 3.3)
                let entities = self.extract_with_grid(text, threshold)?;

                return Ok(entities
                    .into_iter()
                    .map(|e| DiscontinuousEntity {
                        spans: vec![(e.start, e.end)],
                        text: e.text,
                        entity_type: e.entity_type.as_label().to_string(),
                        confidence: e.confidence as f32,
                    })
                    .collect());
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
mod tests {
    use super::*;

    #[test]
    fn test_w2ner_relation_conversion() {
        assert_eq!(W2NERRelation::from_index(0), W2NERRelation::None);
        assert_eq!(W2NERRelation::from_index(1), W2NERRelation::NNW);
        assert_eq!(W2NERRelation::from_index(2), W2NERRelation::THW);

        assert_eq!(W2NERRelation::None.to_index(), 0);
        assert_eq!(W2NERRelation::NNW.to_index(), 1);
        assert_eq!(W2NERRelation::THW.to_index(), 2);
    }

    #[test]
    fn test_w2ner_config_defaults() {
        let config = W2NERConfig::default();
        assert!((config.threshold - 0.5).abs() < f64::EPSILON);
        assert!(config.allow_nested);
        assert!(config.allow_discontinuous);
        assert_eq!(config.entity_labels.len(), 3);
    }

    #[test]
    fn test_decode_simple_entity() {
        let w2ner = W2NER::new();
        let tokens = ["New", "York", "City"];

        // THW marker: tail=2, head=0 (entity spans all 3 tokens)
        let matrix = HandshakingMatrix {
            cells: vec![HandshakingCell {
                i: 2, // tail
                j: 0, // head
                label_idx: W2NERRelation::THW.to_index() as u16,
                score: 0.9,
            }],
            seq_len: 3,
            num_labels: 3,
        };

        let entities = w2ner.decode_from_matrix(&matrix, &tokens, 0);
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].0, 0); // start
        assert_eq!(entities[0].1, 3); // end
    }

    #[test]
    fn test_decode_nested_entities() {
        let w2ner = W2NER::with_config(W2NERConfig {
            allow_nested: true,
            ..Default::default()
        });

        let tokens = ["University", "of", "California", "Berkeley"];

        let matrix = HandshakingMatrix {
            cells: vec![
                // Full entity: tail=3, head=0
                HandshakingCell {
                    i: 3,
                    j: 0,
                    label_idx: W2NERRelation::THW.to_index() as u16,
                    score: 0.95,
                },
                // Nested: tail=2, head=2 (just "California")
                HandshakingCell {
                    i: 2,
                    j: 2,
                    label_idx: W2NERRelation::THW.to_index() as u16,
                    score: 0.85,
                },
            ],
            seq_len: 4,
            num_labels: 3,
        };

        let entities = w2ner.decode_from_matrix(&matrix, &tokens, 0);
        assert_eq!(entities.len(), 2);
    }

    #[test]
    fn test_remove_nested() {
        let entities = vec![
            (0, 4, 0.9), // outer
            (2, 3, 0.8), // nested
        ];

        let filtered = W2NER::remove_nested(&entities);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0], (0, 4, 0.9));
    }

    #[test]
    fn test_grid_to_matrix() {
        // 3x3 grid with 3 relations (None, NNW, THW)
        let seq_len = 3;
        let num_rels = 3;
        let mut grid = vec![0.0f32; seq_len * seq_len * num_rels];

        // Set THW at (2, 0) with score 0.9
        // Index formula: i * seq_len * num_rels + j * num_rels + rel_idx
        let i = 2;
        let j = 0;
        let rel_thw = 2;
        let idx = i * seq_len * num_rels + j * num_rels + rel_thw;
        grid[idx] = 0.9;

        let matrix = W2NER::grid_to_matrix(&grid, seq_len, num_rels, 0.5);
        assert_eq!(matrix.cells.len(), 1);
        assert_eq!(matrix.cells[0].i, 2);
        assert_eq!(matrix.cells[0].j, 0);
    }

    #[test]
    fn test_label_mapping() {
        assert_eq!(W2NER::map_label("PER"), EntityType::Person);
        assert_eq!(W2NER::map_label("org"), EntityType::Organization);
        assert_eq!(W2NER::map_label("GPE"), EntityType::Location);
        assert_eq!(
            W2NER::map_label("CUSTOM"),
            EntityType::Other("CUSTOM".to_string())
        );
    }

    #[test]
    fn test_empty_input() {
        let w2ner = W2NER::new();
        let entities = w2ner.extract_entities("", None).unwrap();
        assert!(entities.is_empty());
    }

    #[test]
    fn test_not_available_without_model() {
        let w2ner = W2NER::new();
        // Without model loaded, should not be available
        assert!(!w2ner.is_available());
    }

    #[test]
    fn test_errors_without_model() {
        let w2ner = W2NER::new();
        // Without model, should return an explicit error (no silent empty fallback).
        let err = w2ner
            .extract_entities("Steve Jobs founded Apple", None)
            .unwrap_err();
        assert!(
            matches!(
                err,
                crate::Error::ModelInit(_) | crate::Error::FeatureNotAvailable(_)
            ),
            "unexpected error: {:?}",
            err
        );
    }
}
