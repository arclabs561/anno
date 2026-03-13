//! BERT-based NER using ONNX Runtime.
//!
//! This module provides a reliable ONNX-based NER backend using standard
//! BERT models fine-tuned for token classification (BIO tags).
//!
//! Unlike GLiNER which has ONNX export issues, this uses properly exported
//! BERT NER models like `protectai/bert-base-NER-onnx`.
//!
//! ## Default Model
//!
//! Uses `protectai/bert-base-NER-onnx` which recognizes:
//! - PER (Person)
//! - ORG (Organization)
//! - LOC (Location)
//! - MISC (Miscellaneous)

#![allow(missing_docs)] // BIO decoding internals; public API is documented
#![allow(clippy::manual_strip)] // Complex BIO tag parsing

use crate::{Entity, Error, Language, Result};
#[cfg(feature = "onnx")]
use anno_core::{EntityCategory, EntityType};

#[cfg(feature = "onnx")]
use {
    crate::sync::lock, ndarray::Array2, ort::session::Session, std::collections::HashMap,
    tokenizers::Tokenizer,
};

/// Configuration for BERT NER model loading.
#[cfg(feature = "onnx")]
#[derive(Debug, Clone)]
pub struct BertNERConfig {
    /// Prefer quantized models (INT8) for faster CPU inference.
    pub prefer_quantized: bool,
    /// ONNX optimization level (1-3, default 3).
    pub optimization_level: u8,
    /// Number of threads for inference (0 = auto).
    pub num_threads: usize,
}

#[cfg(feature = "onnx")]
impl Default for BertNERConfig {
    fn default() -> Self {
        Self {
            prefer_quantized: true,
            optimization_level: 3,
            num_threads: 4,
        }
    }
}

/// BERT-based NER using ONNX Runtime.
///
/// Uses standard BERT models fine-tuned for NER with BIO tagging scheme.
/// Thread-safe with `Arc<Tokenizer>` for efficient sharing.
#[cfg(feature = "onnx")]
pub struct BertNEROnnx {
    session: crate::sync::Mutex<Session>,
    /// Arc-wrapped tokenizer for cheap cloning across threads.
    tokenizer: std::sync::Arc<Tokenizer>,
    id_to_label: HashMap<usize, String>,
    label_to_entity_type: HashMap<String, EntityType>,
    model_name: String,
    /// Whether a quantized model was loaded.
    is_quantized: bool,
}

#[cfg(feature = "onnx")]
impl BertNEROnnx {
    /// Create a new BERT NER ONNX model with default config.
    ///
    /// # Arguments
    /// * `model_name` - HuggingFace model identifier (e.g., "protectai/bert-base-NER-onnx")
    ///
    /// # Returns
    /// BERT NER ONNX model instance
    pub fn new(model_name: &str) -> Result<Self> {
        Self::with_config(model_name, BertNERConfig::default())
    }

    /// Create a new BERT NER ONNX model with custom configuration.
    ///
    /// # Arguments
    /// * `model_name` - HuggingFace model identifier
    /// * `config` - Configuration for model loading
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = BertNERConfig {
    ///     prefer_quantized: true,  // Use INT8 model for 2-4x speedup
    ///     optimization_level: 3,
    ///     num_threads: 8,
    /// };
    /// let model = BertNEROnnx::with_config("protectai/bert-base-NER-onnx", config)?;
    /// ```
    pub fn with_config(model_name: &str, config: BertNERConfig) -> Result<Self> {
        use crate::backends::hf_loader;

        let api = hf_loader::hf_api()?;
        let repo = api.model(model_name.to_string());

        // Download model - try quantized first if preferred
        let (model_path, is_quantized) =
            hf_loader::download_onnx_model(&repo, config.prefer_quantized)?;

        let tokenizer_path = hf_loader::download_model_file(&repo, &["tokenizer.json"])?;
        let config_path = hf_loader::download_model_file(&repo, &["config.json"])?;

        let tokenizer = hf_loader::load_tokenizer(&tokenizer_path)?;

        // Load config and extract id2label mapping
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| Error::Retrieval(format!("Failed to read config.json: {}", e)))?;
        let config_json: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| Error::Parse(format!("Failed to parse config.json: {}", e)))?;

        // Build label mappings
        let id_to_label = Self::build_id_to_label(&config_json);
        let label_to_entity_type = Self::build_label_to_entity_type();

        let session = hf_loader::create_onnx_session(
            &model_path,
            hf_loader::OnnxSessionConfig {
                optimization_level: config.optimization_level,
                num_threads: config.num_threads,
                use_cpu_provider: false,
            },
        )?;

        Ok(Self {
            session: crate::sync::Mutex::new(session),
            tokenizer: std::sync::Arc::new(tokenizer),
            id_to_label,
            label_to_entity_type,
            model_name: model_name.to_string(),
            is_quantized,
        })
    }

    /// Check if a quantized model was loaded.
    #[must_use]
    pub fn is_quantized(&self) -> bool {
        self.is_quantized
    }

    /// Get a clone of the tokenizer Arc (cheap).
    #[must_use]
    pub fn tokenizer(&self) -> std::sync::Arc<Tokenizer> {
        std::sync::Arc::clone(&self.tokenizer)
    }

    /// Build id_to_label mapping from config.
    fn build_id_to_label(config_json: &serde_json::Value) -> HashMap<usize, String> {
        let mut map = HashMap::new();
        if let Some(id2label) = config_json.get("id2label") {
            if let Some(obj) = id2label.as_object() {
                for (id_str, label_value) in obj {
                    if let (Ok(id), Some(label)) = (id_str.parse::<usize>(), label_value.as_str()) {
                        map.insert(id, label.to_string());
                    }
                }
            }
        }
        // Fallback for CoNLL-03 format
        if map.is_empty() {
            map.insert(0, "O".to_string());
            map.insert(1, "B-MISC".to_string());
            map.insert(2, "I-MISC".to_string());
            map.insert(3, "B-PER".to_string());
            map.insert(4, "I-PER".to_string());
            map.insert(5, "B-ORG".to_string());
            map.insert(6, "I-ORG".to_string());
            map.insert(7, "B-LOC".to_string());
            map.insert(8, "I-LOC".to_string());
        }
        map
    }

    /// Build label_to_entity_type mapping for common NER labels.
    fn build_label_to_entity_type() -> HashMap<String, EntityType> {
        let mut map = HashMap::new();
        // Standard CoNLL-03 labels
        map.insert("B-PER".to_string(), EntityType::Person);
        map.insert("I-PER".to_string(), EntityType::Person);
        map.insert("B-ORG".to_string(), EntityType::Organization);
        map.insert("I-ORG".to_string(), EntityType::Organization);
        map.insert("B-LOC".to_string(), EntityType::Location);
        map.insert("I-LOC".to_string(), EntityType::Location);
        map.insert(
            "B-MISC".to_string(),
            EntityType::custom("misc", EntityCategory::Misc),
        );
        map.insert(
            "I-MISC".to_string(),
            EntityType::custom("misc", EntityCategory::Misc),
        );
        // Alternative formats
        map.insert("PER".to_string(), EntityType::Person);
        map.insert("ORG".to_string(), EntityType::Organization);
        map.insert("LOC".to_string(), EntityType::Location);
        map.insert(
            "MISC".to_string(),
            EntityType::custom("misc", EntityCategory::Misc),
        );
        map
    }

    /// Extract entities from text using BERT NER.
    ///
    /// # Arguments
    /// * `text` - Text to extract entities from
    /// * `_language` - Optional language hint (unused, model handles multiple languages)
    ///
    /// # Returns
    /// Vector of NER entities with positions, types, and confidence scores
    /// Maximum tokens per BERT chunk (512 model limit minus [CLS] and [SEP]).
    const MAX_TOKENS: usize = 510;

    pub fn extract_entities(&self, text: &str, _language: Option<Language>) -> Result<Vec<Entity>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        // Check if text exceeds BERT's 512 token limit; if so, split into
        // sentence-boundary chunks and merge results.
        let probe = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| Error::Parse(format!("Failed to tokenize input: {}", e)))?;
        if probe.get_ids().len() > Self::MAX_TOKENS + 2 {
            return self.extract_entities_chunked(text);
        }

        self.extract_entities_single(text, &probe)
    }

    /// Split long text into sentence-boundary chunks that fit within BERT's
    /// 512-token window, run each chunk, and merge entity results.
    ///
    /// Uses 1-sentence overlap between consecutive chunks so entities near
    /// chunk boundaries get a second chance with surrounding context. Entities
    /// from the overlap region are deduplicated by (start, end, type).
    fn extract_entities_chunked(&self, text: &str) -> Result<Vec<Entity>> {
        // First, find all sentence boundaries (byte offsets after '.', '!', '?', '\n').
        let mut sentence_ends: Vec<usize> = Vec::new();
        for (i, c) in text.char_indices() {
            if matches!(c, '.' | '!' | '?' | '\n') {
                sentence_ends.push(i + c.len_utf8());
            }
        }
        if sentence_ends.is_empty() || *sentence_ends.last().unwrap() < text.len() {
            sentence_ends.push(text.len());
        }

        // Build chunks: each chunk is (byte_start, byte_end). We greedily pack
        // sentences until the next one would exceed MAX_TOKENS, then start a new
        // chunk. The new chunk begins 1 sentence *before* the split point
        // (overlap) so cross-boundary entities get full context on both sides.
        let mut chunks: Vec<(usize, usize)> = Vec::new();
        let mut chunk_start_byte = 0usize;
        let mut prev_boundary_idx: Option<usize> = None; // index into sentence_ends

        for (sent_idx, &sent_end) in sentence_ends.iter().enumerate() {
            let candidate = &text[chunk_start_byte..sent_end];
            let tok = self
                .tokenizer
                .encode(candidate, true)
                .map_err(|e| Error::Parse(format!("Chunking tokenization failed: {}", e)))?;

            if tok.get_ids().len() > Self::MAX_TOKENS + 2 {
                // This sentence pushes over the limit. Flush up to the previous boundary.
                if let Some(prev_idx) = prev_boundary_idx {
                    let flush_end = sentence_ends[prev_idx];
                    chunks.push((chunk_start_byte, flush_end));
                    // Overlap: start new chunk 1 sentence back (if possible)
                    chunk_start_byte = if prev_idx > 0 {
                        sentence_ends[prev_idx - 1]
                    } else {
                        flush_end
                    };
                } else {
                    // Single sentence exceeds limit -- flush it anyway
                    chunks.push((chunk_start_byte, sent_end));
                    chunk_start_byte = sent_end;
                }
            }
            prev_boundary_idx = Some(sent_idx);
        }
        // Flush remaining text
        if chunk_start_byte < text.len() {
            chunks.push((chunk_start_byte, text.len()));
        }
        if chunks.is_empty() {
            chunks.push((0, text.len()));
        }

        // Run each chunk and collect entities with global char offsets.
        let mut all_entities = Vec::new();
        for (byte_start, byte_end) in &chunks {
            let chunk = &text[*byte_start..*byte_end];
            let encoding = self
                .tokenizer
                .encode(chunk, true)
                .map_err(|e| Error::Parse(format!("Chunk tokenization failed: {}", e)))?;
            let mut chunk_entities = self.extract_entities_single(chunk, &encoding)?;
            // Shift character offsets to global positions
            if *byte_start > 0 {
                let char_offset = text[..*byte_start].chars().count();
                for e in &mut chunk_entities {
                    e.start += char_offset;
                    e.end += char_offset;
                }
            }
            all_entities.extend(chunk_entities);
        }

        // Deduplicate entities from overlapping regions.
        // Keep the higher-confidence version when (start, end, type) collide.
        all_entities.sort_by(|a, b| {
            a.start
                .cmp(&b.start)
                .then(a.end.cmp(&b.end))
                .then(a.entity_type.to_string().cmp(&b.entity_type.to_string()))
        });
        all_entities.dedup_by(|b, a| {
            a.start == b.start && a.end == b.end && a.entity_type == b.entity_type
        });
        // Also remove entities fully contained within a larger entity (from overlap)
        let mut keep = vec![true; all_entities.len()];
        for i in 0..all_entities.len() {
            for j in 0..all_entities.len() {
                if i != j
                    && keep[j]
                    && all_entities[i].start >= all_entities[j].start
                    && all_entities[i].end <= all_entities[j].end
                    && (all_entities[i].start != all_entities[j].start
                        || all_entities[i].end != all_entities[j].end)
                {
                    keep[i] = false;
                    break;
                }
            }
        }
        let all_entities: Vec<Entity> = all_entities
            .into_iter()
            .enumerate()
            .filter(|(i, _)| keep[*i])
            .map(|(_, e)| e)
            .collect();

        Ok(all_entities)
    }

    fn extract_entities_single(
        &self,
        text: &str,
        encoding: &tokenizers::Encoding,
    ) -> Result<Vec<Entity>> {
        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&mask| mask as i64)
            .collect();
        // Performance: Pre-allocate token_type_ids with known size
        // token_type_ids: all zeros for single-sequence NER
        let token_type_ids: Vec<i64> = vec![0i64; input_ids.len()];

        let batch_size = 1;
        let seq_len = input_ids.len();

        // Create input tensors
        let input_ids_array: Array2<i64> =
            Array2::from_shape_vec((batch_size, seq_len), input_ids.clone())
                .map_err(|e| Error::Parse(format!("Failed to create input_ids array: {}", e)))?;

        let attention_mask_array: Array2<i64> =
            Array2::from_shape_vec((batch_size, seq_len), attention_mask.clone()).map_err(|e| {
                Error::Parse(format!("Failed to create attention_mask array: {}", e))
            })?;

        let token_type_ids_array: Array2<i64> =
            Array2::from_shape_vec((batch_size, seq_len), token_type_ids).map_err(|e| {
                Error::Parse(format!("Failed to create token_type_ids array: {}", e))
            })?;

        let input_ids_tensor = super::ort_compat::tensor_from_ndarray(input_ids_array)
            .map_err(|e| Error::Parse(format!("Failed to create input_ids tensor: {}", e)))?;

        let attention_mask_tensor = super::ort_compat::tensor_from_ndarray(attention_mask_array)
            .map_err(|e| Error::Parse(format!("Failed to create attention_mask tensor: {}", e)))?;

        let token_type_ids_tensor = super::ort_compat::tensor_from_ndarray(token_type_ids_array)
            .map_err(|e| Error::Parse(format!("Failed to create token_type_ids tensor: {}", e)))?;

        // Run inference with blocking lock for thread-safe parallel access
        let mut session = lock(&self.session);

        let outputs = session
            .run(ort::inputs![
                "input_ids" => input_ids_tensor.into_dyn(),
                "attention_mask" => attention_mask_tensor.into_dyn(),
                "token_type_ids" => token_type_ids_tensor.into_dyn(),
            ])
            .map_err(|e| Error::Parse(format!("ONNX inference failed: {}", e)))?;

        // Get logits output - BERT NER models have "logits" as output
        let logits = outputs.get("logits").ok_or_else(|| {
            Error::Parse("ONNX model output does not contain 'logits' key".to_string())
        })?;

        // Decode logits to entities
        self.decode_output(logits, text, encoding)
    }

    /// Decode model output logits to NER entities.
    fn decode_output(
        &self,
        output: &ort::value::DynValue,
        text: &str,
        encoding: &tokenizers::Encoding,
    ) -> Result<Vec<Entity>> {
        // Extract logits as f32 array - ort returns (Shape, &[f32])
        let (shape, logits_data) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Parse(format!("Failed to extract logits tensor: {}", e)))?;

        // Expected shape: [batch_size, seq_len, num_labels]
        if shape.len() != 3 || shape[0] != 1 {
            return Err(Error::Parse(format!(
                "Unexpected logits shape: {:?}",
                shape
            )));
        }

        let seq_len = shape[1] as usize;
        let num_labels = shape[2] as usize;

        // Get token offsets for mapping back to character positions
        let offsets = encoding.get_offsets();

        // Use word_ids for robust subword continuation detection.
        // Tokens belonging to the same word share a word_id; this is far more
        // reliable than checking byte_start == prev_end for subword grouping.
        let word_ids = encoding.get_word_ids();

        // `tokenizers::Encoding::get_offsets()` uses byte offsets in Rust. `Entity` uses character
        // offsets, so convert via SpanConverter when constructing entities.
        let span_converter = crate::offset::SpanConverter::new(text);

        // Helper to access logits[0, token_idx, label_idx] in flattened array
        let get_logit = |token_idx: usize, label_idx: usize| -> f32 {
            logits_data[token_idx * num_labels + label_idx]
        };

        // Helper: finalize an entity from byte offsets, trimming whitespace and
        // adjusting character offsets to match the trimmed text.
        let finalize_entity = |start: usize,
                               end: usize,
                               entity_type: EntityType,
                               conf: f64,
                               entities: &mut Vec<Entity>| {
            if start >= end || end > text.len() {
                return;
            }
            if let Some(slice) = text.get(start..end) {
                let trimmed = slice.trim();
                if trimmed.is_empty() {
                    return;
                }
                // Adjust byte offsets to account for trimmed whitespace.
                let leading = slice.len() - slice.trim_start().len();
                let trailing = slice.len() - slice.trim_end().len();
                let mut adj_start = start + leading;
                let mut adj_end = end - trailing;

                // Heal word-boundary misalignment from BPE subword
                // mislabeling: extend to the enclosing word boundary when the
                // entity starts or ends mid-word.
                while adj_start > 0
                    && text
                        .get(adj_start.saturating_sub(1)..adj_start)
                        .and_then(|s| s.chars().next())
                        .is_some_and(|c| c.is_alphanumeric())
                {
                    adj_start -= text[..adj_start]
                        .chars()
                        .next_back()
                        .map_or(1, |c| c.len_utf8());
                }
                while adj_end < text.len()
                    && text
                        .get(adj_end..adj_end + 1)
                        .and_then(|s| s.chars().next())
                        .is_some_and(|c| c.is_alphanumeric())
                {
                    adj_end += text[adj_end..].chars().next().map_or(1, |c| c.len_utf8());
                }
                let healed = text.get(adj_start..adj_end).unwrap_or(trimmed).trim();
                if healed.is_empty() {
                    return;
                }
                entities.push(Entity::new(
                    healed.to_string(),
                    entity_type,
                    span_converter.byte_to_char(adj_start),
                    span_converter.byte_to_char_ceil(adj_end),
                    conf,
                ));
            }
        };

        // Track the word_id of the last token added to the current entity,
        // so we can detect subword continuation even when labels disagree.
        let mut last_entity_word_id: Option<u32> = None;

        // Performance: Pre-allocate entities vec with estimated capacity
        // Most texts have 0-20 entities, but we'll start with a reasonable default
        let mut entities = Vec::with_capacity(16);
        let mut current_entity: Option<(usize, usize, EntityType, f64)> = None; // (start_byte, end_byte, type, confidence)

        for token_idx in 0..seq_len {
            // Skip special tokens (no offset)
            if token_idx >= offsets.len() {
                continue;
            }
            let (byte_start, byte_end) = offsets[token_idx];
            let cur_word_id = word_ids.get(token_idx).copied().flatten();
            if byte_start == byte_end {
                // Special token, finalize current entity
                if let Some((start, end, entity_type, conf)) = current_entity.take() {
                    finalize_entity(start, end, entity_type, conf, &mut entities);
                    last_entity_word_id = None;
                }
                continue;
            }

            // Get logits for this token and find argmax
            let mut max_idx = 0;
            let mut max_val = f32::NEG_INFINITY;
            for label_idx in 0..num_labels {
                let val = get_logit(token_idx, label_idx);
                if val > max_val {
                    max_val = val;
                    max_idx = label_idx;
                }
            }

            // Convert to probability using softmax
            let exp_sum: f32 = (0..num_labels)
                .map(|i| (get_logit(token_idx, i) - max_val).exp())
                .sum();
            // Handle division by zero: if exp_sum == 0.0 or num_labels == 0, use fallback
            let confidence = if exp_sum > 0.0 && num_labels > 0 {
                (1.0_f32 / exp_sum) as f64 // exp(0) / exp_sum = 1/exp_sum
            } else {
                0.0 // Fallback for edge cases
            };

            let label = self
                .id_to_label
                .get(&max_idx)
                .cloned()
                .unwrap_or_else(|| format!("LABEL_{}", max_idx));

            // Detect subword continuation: either same word_id as the previous
            // entity token, or byte-adjacent and starting with alphanumeric.
            let starts_alnum = text
                .get(byte_start..)
                .and_then(|s| s.chars().next())
                .is_some_and(|c| c.is_alphanumeric());
            let same_word =
                starts_alnum && cur_word_id.is_some() && cur_word_id == last_entity_word_id;
            let byte_adjacent = starts_alnum
                && current_entity
                    .as_ref()
                    .map(|(_, prev_end, _, _)| byte_start == *prev_end)
                    .unwrap_or(false);
            let is_subword_continuation = current_entity.is_some() && (same_word || byte_adjacent);

            // Skip "O" (outside) labels -- but if this token is a subword
            // continuation (same word_id OR no whitespace gap AND starts with
            // an alphanumeric character), extend the current entity span rather
            // than closing it.  This is the "first-token aggregation" strategy:
            // the label of the first subword token determines the entity
            // boundary, and subsequent subwords always extend it.  We require
            // alphanumeric to avoid absorbing trailing punctuation (e.g.
            // "Strasbourg.").
            if label == "O" {
                if is_subword_continuation {
                    // Extend: keep the entity open with the new byte_end.
                    if let Some((start, _, etype, conf)) = current_entity.take() {
                        current_entity = Some((start, byte_end, etype, conf));
                        last_entity_word_id = cur_word_id;
                    }
                } else if current_entity
                    .as_ref()
                    .is_some_and(|(_, prev_end, prev_type, _)| {
                        // Name completion: BERT often tags surnames as O when
                        // they follow a recognized given name.  Absorb adjacent
                        // capitalized words (proper-noun pattern: Uppercase
                        // followed by lowercase) into PER entities.
                        //
                        // Also absorb across hyphens (e.g. "Jean-Claude") where
                        // the gap is exactly a '-' character.
                        matches!(prev_type, EntityType::Person)
                            && byte_start <= *prev_end + 2
                            && {
                                let gap_is_connector = byte_start == *prev_end + 1
                                    || text
                                        .get(*prev_end..byte_start)
                                        .is_some_and(|g| g == " " || g == "-");
                                gap_is_connector
                            }
                            && text
                                .get(byte_start..)
                                .map(|s| {
                                    let mut chars = s.chars();
                                    match (chars.next(), chars.next()) {
                                        (Some(c1), Some(c2)) => {
                                            c1.is_uppercase() && c2.is_lowercase()
                                        }
                                        _ => false,
                                    }
                                })
                                .unwrap_or(false)
                    })
                {
                    if let Some((start, _, etype, conf)) = current_entity.take() {
                        current_entity = Some((start, byte_end, etype, conf));
                        last_entity_word_id = cur_word_id;
                    }
                } else if current_entity
                    .as_ref()
                    .is_some_and(|(_, prev_end, prev_type, _)| {
                        // Keep PER entity open across a hyphen token so the
                        // next B-PER token can merge (e.g. "Jean-Claude").
                        matches!(prev_type, EntityType::Person)
                            && byte_start == *prev_end
                            && text.get(byte_start..byte_end).is_some_and(|t| t == "-")
                    })
                {
                    // Don't finalize -- leave current_entity as-is,
                    // the next token's B-tag merge will pick it up.
                } else if let Some((start, end, entity_type, conf)) = current_entity.take() {
                    finalize_entity(start, end, entity_type, conf, &mut entities);
                    last_entity_word_id = None;
                }
                continue;
            }

            // Parse BIO tag
            let (bio, entity_label) = if label.starts_with("B-") {
                ("B", label[2..].to_string())
            } else if label.starts_with("I-") {
                ("I", label[2..].to_string())
            } else {
                ("B", label.clone())
            };

            let entity_type = self
                .label_to_entity_type
                .get(&format!("B-{}", entity_label))
                .or_else(|| self.label_to_entity_type.get(&entity_label))
                .cloned()
                .unwrap_or_else(|| EntityType::custom(entity_label.clone(), EntityCategory::Misc));

            match bio {
                "B" => {
                    // Check if this B- tag should merge with previous entity.
                    // Three reasons to merge:
                    //   1. Same word_id (subword of same word).
                    //   2. Byte-adjacent AND alphanumeric start (subword tokenization).
                    //   3. Same type AND within 1 byte gap (adjacent words in same entity).
                    let should_merge = if let Some((_, prev_end, ref prev_type, _)) = current_entity
                    {
                        is_subword_continuation
                            || (std::mem::discriminant(prev_type)
                                == std::mem::discriminant(&entity_type)
                                && byte_start <= prev_end + 2
                                && (byte_start == prev_end + 1
                                    || text
                                        .get(prev_end..byte_start)
                                        .is_some_and(|g| g == " " || g == "-")))
                    } else {
                        false
                    };

                    if should_merge {
                        // Extend the current entity instead of starting new
                        if let Some((start, _, prev_type, conf)) = current_entity.take() {
                            current_entity = Some((start, byte_end, prev_type, conf));
                            last_entity_word_id = cur_word_id;
                        }
                    } else {
                        // Finalize previous entity and start new
                        if let Some((start, end, prev_type, conf)) = current_entity.take() {
                            finalize_entity(start, end, prev_type, conf, &mut entities);
                        }
                        // Start new entity
                        current_entity = Some((byte_start, byte_end, entity_type, confidence));
                        last_entity_word_id = cur_word_id;
                    }
                }
                "I" => {
                    // Continue current entity if same type
                    if let Some((start, _end, ref prev_type, conf)) = current_entity {
                        if std::mem::discriminant(prev_type) == std::mem::discriminant(&entity_type)
                        {
                            current_entity = Some((start, byte_end, entity_type, conf));
                            last_entity_word_id = cur_word_id;
                        } else {
                            // Different type - finalize and start new
                            let prev_type = prev_type.clone();
                            let (start, end) = (start, _end);
                            current_entity.take();
                            finalize_entity(start, end, prev_type, conf, &mut entities);
                            current_entity = Some((byte_start, byte_end, entity_type, confidence));
                            last_entity_word_id = cur_word_id;
                        }
                    } else {
                        // No current entity, treat I- as B-
                        current_entity = Some((byte_start, byte_end, entity_type, confidence));
                        last_entity_word_id = cur_word_id;
                    }
                }
                _ => {}
            }
        }

        // Finalize last entity
        if let Some((start, end, entity_type, conf)) = current_entity {
            finalize_entity(start, end, entity_type, conf, &mut entities);
        }

        Ok(entities)
    }

    /// Get the model name.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[cfg(feature = "onnx")]
impl crate::Model for BertNEROnnx {
    fn extract_entities(&self, text: &str, language: Option<Language>) -> Result<Vec<Entity>> {
        self.extract_entities(text, language)
    }

    fn supported_types(&self) -> Vec<EntityType> {
        vec![
            EntityType::Person,
            EntityType::Organization,
            EntityType::Location,
            EntityType::custom("MISC", EntityCategory::Misc),
        ]
    }

    fn is_available(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "bert-onnx"
    }

    fn description(&self) -> &'static str {
        "BERT-based NER using ONNX Runtime (PER/ORG/LOC/MISC)"
    }

    fn version(&self) -> String {
        format!(
            "bert-onnx-{}-{}",
            self.model_name,
            if self.is_quantized { "q" } else { "fp32" }
        )
    }

    fn capabilities(&self) -> crate::ModelCapabilities {
        crate::ModelCapabilities {
            batch_capable: true,
            streaming_capable: true,
            ..Default::default()
        }
    }
}

// Stub implementation when feature is disabled
crate::backends::macros::define_feature_stub! {
    struct BertNEROnnx;
    feature = "onnx";
    name = "bert-onnx (unavailable)";
    description = "BERT-based NER using ONNX Runtime - requires 'onnx' feature";
    error_msg = "BERT NER ONNX requires the 'onnx' feature";
    methods {
        pub fn model_name(&self) -> &str {
            "onnx-not-enabled"
        }
    }
}
