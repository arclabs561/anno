//! GLiREL ONNX backend — relation extraction via DeBERTa + scoring head.
//!
//! Requires `--features onnx`. Loads a GLiREL model exported by
//! `scripts/export_glirel_onnx.py`.

use crate::backends::hf_loader;
use crate::backends::inference::RelationTriple;
use crate::sync::{lock, Mutex};
use crate::{Confidence, Entity, Error, Result};
use ndarray::Array2;
use std::path::{Path, PathBuf};

/// Default cache directory for GLiREL models.
fn default_model_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("anno")
        .join("models")
        .join("glirel")
}

/// GLiREL zero-shot relation extraction model (ONNX backend).
///
/// Takes pre-extracted entity spans and relation type labels, then scores
/// each `(head, tail, relation_type)` triple via neural scoring.
#[derive(Debug)]
pub struct GLiREL {
    session: Mutex<ort::session::Session>,
    tokenizer: tokenizers::Tokenizer,
    #[allow(dead_code)]
    config: GLiRELConfig,
}

/// Configuration loaded from `glirel_config.json`.
#[derive(Debug, Clone, serde::Deserialize)]
#[allow(dead_code)]
pub struct GLiRELConfig {
    /// HuggingFace model name.
    #[serde(default)]
    pub model_name: String,
    /// Hidden dimension of the encoder.
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    /// Maximum span width for entity candidates.
    #[serde(default = "default_max_width")]
    pub max_width: usize,
}

fn default_hidden_size() -> usize {
    1024
}
fn default_max_width() -> usize {
    12
}

impl Default for GLiRELConfig {
    fn default() -> Self {
        Self {
            model_name: "jackboyla/glirel-large-v0".to_string(),
            hidden_size: 1024,
            max_width: 12,
        }
    }
}

/// A scored relation between two entity spans.
#[derive(Debug, Clone)]
pub struct ScoredRelation {
    /// Index of head entity in the input spans.
    pub head_idx: usize,
    /// Index of tail entity in the input spans.
    pub tail_idx: usize,
    /// Relation type label.
    pub relation_type: String,
    /// Confidence score (sigmoid of raw score).
    pub confidence: Confidence,
}

impl GLiREL {
    /// Load GLiREL from a HuggingFace model ID.
    ///
    /// This expects the model to be pre-exported to ONNX format.
    /// If the ONNX model is not available on HuggingFace, use
    /// `scripts/export_glirel_onnx.py` to export it locally, then
    /// call [`GLiREL::from_local`].
    pub fn from_pretrained(model_id: &str) -> Result<Self> {
        let api = hf_loader::hf_api()?;
        let repo = api.model(model_id.to_string());

        let model_path = hf_loader::download_model_file(&repo, &["onnx/model.onnx", "model.onnx"])?;
        let tokenizer_path = hf_loader::download_model_file(&repo, &["tokenizer.json"])?;

        let config = match repo.get("glirel_config.json") {
            Ok(config_path) => {
                let data = std::fs::read_to_string(&config_path)
                    .map_err(|e| Error::Retrieval(format!("glirel config read: {e}")))?;
                serde_json::from_str(&data)
                    .map_err(|e| Error::Parse(format!("glirel config parse: {e}")))?
            }
            Err(_) => GLiRELConfig {
                model_name: model_id.to_string(),
                ..GLiRELConfig::default()
            },
        };

        let tokenizer = hf_loader::load_tokenizer(&tokenizer_path)?;
        let session =
            hf_loader::create_onnx_session(&model_path, hf_loader::OnnxSessionConfig::default())?;

        log::info!(
            "[GLiREL] Loaded {} (hidden={})",
            model_id,
            config.hidden_size
        );

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            config,
        })
    }

    /// Load GLiREL from a local directory (exported by `export_glirel_onnx.py`).
    ///
    /// The directory must contain `model.onnx`, `tokenizer.json`, and
    /// optionally `glirel_config.json`.
    pub fn from_local(dir: &Path) -> Result<Self> {
        let model_path = dir.join("model.onnx");
        if !model_path.exists() {
            // Try default cache location
            let default_dir = default_model_dir();
            let alt_path = default_dir.join("model.onnx");
            if alt_path.exists() {
                return Self::from_local(&default_dir);
            }
            return Err(Error::Retrieval(format!(
                "GLiREL model not found at {}. Export it with: uv run scripts/export_glirel_onnx.py",
                model_path.display()
            )));
        }

        let tokenizer_path = dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(Error::Retrieval(format!(
                "Tokenizer not found at {}",
                tokenizer_path.display()
            )));
        }

        let config = {
            let config_path = dir.join("glirel_config.json");
            if config_path.exists() {
                let data = std::fs::read_to_string(&config_path)
                    .map_err(|e| Error::Retrieval(format!("glirel config read: {e}")))?;
                serde_json::from_str(&data)
                    .map_err(|e| Error::Parse(format!("glirel config parse: {e}")))?
            } else {
                GLiRELConfig::default()
            }
        };

        let tokenizer = hf_loader::load_tokenizer(&tokenizer_path)?;
        let session =
            hf_loader::create_onnx_session(&model_path, hf_loader::OnnxSessionConfig::default())?;

        log::info!("[GLiREL] Loaded from {}", dir.display());

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            config,
        })
    }

    /// Extract relations between entity spans.
    ///
    /// # Arguments
    ///
    /// * `text` - Input text
    /// * `entities` - Pre-extracted entities (with character offsets)
    /// * `relation_types` - Relation type labels to score (zero-shot)
    /// * `threshold` - Minimum confidence to keep a relation
    ///
    /// # Returns
    ///
    /// `RelationTriple`s indexed into the input `entities` slice.
    pub fn extract_relations(
        &self,
        text: &str,
        entities: &[Entity],
        relation_types: &[&str],
        threshold: f32,
    ) -> Result<Vec<RelationTriple>> {
        if entities.len() < 2 || relation_types.is_empty() || text.is_empty() {
            return Ok(Vec::new());
        }

        // Tokenize text
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Ok(Vec::new());
        }

        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| Error::Inference(format!("GLiREL tokenize: {e}")))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();
        let seq_len = input_ids.len();

        // Build words_mask: map each subword token to its word index (1-indexed, 0=special)
        let words_mask = self.build_words_mask(&encoding, &words);

        let text_lengths = vec![words.len() as i64];

        // Build span indices from entities: map character offsets to word indices
        let span_idx = self.entities_to_word_spans(text, &words, entities);
        let num_spans = span_idx.len();
        let span_mask: Vec<bool> = vec![true; num_spans];

        // Tokenize relation type labels
        let (rel_input_ids, rel_attention_mask, rel_seq_len) =
            self.encode_relation_labels(relation_types)?;
        let num_relations = relation_types.len();

        // Build ONNX tensors
        let input_ids_arr = Array2::from_shape_vec((1, seq_len), input_ids)
            .map_err(|e| Error::Parse(format!("input_ids array: {e}")))?;
        let attention_mask_arr = Array2::from_shape_vec((1, seq_len), attention_mask)
            .map_err(|e| Error::Parse(format!("attention_mask array: {e}")))?;
        let words_mask_arr = Array2::from_shape_vec((1, seq_len), words_mask)
            .map_err(|e| Error::Parse(format!("words_mask array: {e}")))?;
        let text_lengths_arr = Array2::from_shape_vec((1, 1), text_lengths)
            .map_err(|e| Error::Parse(format!("text_lengths array: {e}")))?;

        // span_idx: [1, num_spans, 2]
        let span_flat: Vec<i64> = span_idx.iter().flat_map(|&(s, e)| [s, e]).collect();
        let span_idx_arr = ndarray::Array3::from_shape_vec((1, num_spans, 2), span_flat)
            .map_err(|e| Error::Parse(format!("span_idx array: {e}")))?;

        // span_mask: [1, num_spans] as bool -> i64
        let span_mask_i64: Vec<i64> = span_mask.iter().map(|&b| if b { 1 } else { 0 }).collect();
        let span_mask_arr = Array2::from_shape_vec((1, num_spans), span_mask_i64)
            .map_err(|e| Error::Parse(format!("span_mask array: {e}")))?;

        // rel_label_input_ids: [num_relations, rel_seq_len]
        let rel_ids_arr = Array2::from_shape_vec((num_relations, rel_seq_len), rel_input_ids)
            .map_err(|e| Error::Parse(format!("rel_input_ids array: {e}")))?;
        let rel_mask_arr = Array2::from_shape_vec((num_relations, rel_seq_len), rel_attention_mask)
            .map_err(|e| Error::Parse(format!("rel_attention_mask array: {e}")))?;

        // Convert to ONNX tensors
        use super::super::ort_compat::tensor_from_ndarray;

        let t_input_ids = tensor_from_ndarray(input_ids_arr)
            .map_err(|e| Error::Inference(format!("tensor input_ids: {e}")))?;
        let t_attention_mask = tensor_from_ndarray(attention_mask_arr)
            .map_err(|e| Error::Inference(format!("tensor attention_mask: {e}")))?;
        let t_words_mask = tensor_from_ndarray(words_mask_arr)
            .map_err(|e| Error::Inference(format!("tensor words_mask: {e}")))?;
        let t_text_lengths = tensor_from_ndarray(text_lengths_arr)
            .map_err(|e| Error::Inference(format!("tensor text_lengths: {e}")))?;
        let t_span_idx = tensor_from_ndarray(span_idx_arr)
            .map_err(|e| Error::Inference(format!("tensor span_idx: {e}")))?;
        let t_span_mask = tensor_from_ndarray(span_mask_arr)
            .map_err(|e| Error::Inference(format!("tensor span_mask: {e}")))?;
        let t_rel_ids = tensor_from_ndarray(rel_ids_arr)
            .map_err(|e| Error::Inference(format!("tensor rel_input_ids: {e}")))?;
        let t_rel_mask = tensor_from_ndarray(rel_mask_arr)
            .map_err(|e| Error::Inference(format!("tensor rel_attention_mask: {e}")))?;

        // Run ONNX inference
        let mut session = lock(&self.session);
        let outputs = session
            .run(ort::inputs![
                "input_ids" => t_input_ids.into_dyn(),
                "attention_mask" => t_attention_mask.into_dyn(),
                "words_mask" => t_words_mask.into_dyn(),
                "text_lengths" => t_text_lengths.into_dyn(),
                "span_idx" => t_span_idx.into_dyn(),
                "span_mask" => t_span_mask.into_dyn(),
                "rel_label_input_ids" => t_rel_ids.into_dyn(),
                "rel_label_attention_mask" => t_rel_mask.into_dyn(),
            ])
            .map_err(|e| Error::Inference(format!("GLiREL ONNX run: {e}")))?;

        // Decode output: relation_scores [1, num_spans, num_spans, num_relations]
        let scores_output = outputs
            .get("relation_scores")
            .ok_or_else(|| Error::Inference("Missing relation_scores output".to_string()))?;

        let (shape, scores_data) = scores_output
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Inference(format!("extract relation_scores: {e}")))?;

        // Validate shape: [1, num_spans, num_spans, num_relations]
        if shape.len() != 4 {
            return Err(Error::Inference(format!(
                "Unexpected relation_scores shape: {:?}",
                shape
            )));
        }

        // Flat index: [batch, head, tail, rel] = head * (num_spans * num_relations) + tail * num_relations + rel
        let stride_head = num_spans * num_relations;

        // Decode: for each (head, tail, relation) triple above threshold
        let mut relations = Vec::new();
        for head_idx in 0..num_spans {
            for tail_idx in 0..num_spans {
                if head_idx == tail_idx {
                    continue;
                }
                for (rel_idx, rel_type) in relation_types.iter().enumerate() {
                    let flat_idx = head_idx * stride_head + tail_idx * num_relations + rel_idx;
                    let raw_score = scores_data[flat_idx];
                    let conf_f32 = sigmoid(raw_score);
                    if conf_f32 >= threshold {
                        relations.push(RelationTriple {
                            head_idx,
                            tail_idx,
                            relation_type: rel_type.to_string(),
                            confidence: Confidence::new(conf_f32 as f64),
                        });
                    }
                }
            }
        }

        // Sort by confidence descending
        relations.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Deduplicate: keep top relation per directed pair
        let mut seen = std::collections::HashSet::new();
        relations.retain(|r| seen.insert((r.head_idx, r.tail_idx)));

        Ok(relations)
    }

    /// Build words_mask mapping subword tokens to word indices.
    ///
    /// Returns a vec of length `seq_len` where:
    /// - 0 = special token (CLS, SEP, PAD)
    /// - k = token belongs to word k (1-indexed)
    fn build_words_mask(&self, encoding: &tokenizers::Encoding, words: &[&str]) -> Vec<i64> {
        let seq_len = encoding.get_ids().len();
        let mut mask = vec![0i64; seq_len];

        // Use tokenizers word_ids to map tokens to words
        for (token_idx, word_id) in encoding.get_word_ids().iter().enumerate() {
            if let Some(wid) = word_id {
                // word_ids are 0-indexed, we need 1-indexed for the model
                if (*wid as usize) < words.len() {
                    mask[token_idx] = (*wid as i64) + 1;
                }
            }
        }

        mask
    }

    /// Map entity character offsets to word-level span indices.
    fn entities_to_word_spans(
        &self,
        text: &str,
        words: &[&str],
        entities: &[Entity],
    ) -> Vec<(i64, i64)> {
        // Build word -> char offset mapping
        let mut word_starts = Vec::with_capacity(words.len());
        let mut byte_pos = 0;
        let chars: Vec<char> = text.chars().collect();

        for word in words {
            // Find the word in text starting from byte_pos
            if let Some(pos) = text[byte_pos..].find(word) {
                let abs_byte = byte_pos + pos;
                // Convert byte offset to char offset
                let char_offset = text[..abs_byte].chars().count();
                let char_end = char_offset + word.chars().count();
                word_starts.push((char_offset, char_end));
                byte_pos = abs_byte + word.len();
            } else {
                // Fallback: approximate
                let char_offset = if word_starts.is_empty() {
                    0
                } else {
                    word_starts.last().map(|&(_, e)| e).unwrap_or(0)
                };
                word_starts.push((char_offset, char_offset + word.chars().count()));
            }
        }

        let _ = chars; // suppress unused warning

        // Map each entity to (start_word, end_word)
        entities
            .iter()
            .map(|ent| {
                let mut best_start = 0i64;
                let mut best_end = 0i64;
                let mut found = false;

                for (word_idx, &(ws, we)) in word_starts.iter().enumerate() {
                    // Check overlap between entity span and word span
                    if we > ent.start() && ws < ent.end() {
                        if !found {
                            best_start = word_idx as i64;
                            found = true;
                        }
                        best_end = word_idx as i64;
                    }
                }

                (best_start, best_end)
            })
            .collect()
    }

    /// Tokenize relation type labels into padded sequences.
    ///
    /// Returns `(flat_input_ids, flat_attention_mask, max_seq_len)`.
    fn encode_relation_labels(&self, labels: &[&str]) -> Result<(Vec<i64>, Vec<i64>, usize)> {
        let encodings: Vec<_> = labels
            .iter()
            .map(|label| {
                self.tokenizer
                    .encode(*label, true)
                    .map_err(|e| Error::Inference(format!("GLiREL encode label '{label}': {e}")))
            })
            .collect::<Result<Vec<_>>>()?;

        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(1);

        let mut all_ids = Vec::with_capacity(labels.len() * max_len);
        let mut all_masks = Vec::with_capacity(labels.len() * max_len);

        for enc in &encodings {
            let ids = enc.get_ids();
            let masks = enc.get_attention_mask();

            for &id in ids {
                all_ids.push(id as i64);
            }
            for &m in masks {
                all_masks.push(m as i64);
            }
            // Pad to max_len
            let pad = max_len - ids.len();
            all_ids.extend(std::iter::repeat_n(0i64, pad));
            all_masks.extend(std::iter::repeat_n(0i64, pad));
        }

        Ok((all_ids, all_masks, max_len))
    }

    /// Convert `ScoredRelation` results to `RelationTriple` for the standard interface.
    pub fn scored_to_triples(scored: Vec<ScoredRelation>) -> Vec<RelationTriple> {
        scored
            .into_iter()
            .map(|sr| RelationTriple {
                head_idx: sr.head_idx,
                tail_idx: sr.tail_idx,
                relation_type: sr.relation_type,
                confidence: sr.confidence,
            })
            .collect()
    }
}

/// Sigmoid activation.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_config_defaults() {
        let config = GLiRELConfig::default();
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.max_width, 12);
    }
}
