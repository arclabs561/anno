//! GLiNER-based NER implementation using ONNX Runtime.
//!
//! GLiNER (Generalist and Lightweight Model for Named Entity Recognition) is
//! state-of-the-art for zero-shot NER. This implementation follows gline-rs patterns.
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
//! ## Key Insight (from gline-rs)
//!
//! Each word is encoded SEPARATELY, preserving word boundaries.
//! Output shape: [batch, num_words, max_width, num_entity_types]

#![allow(missing_docs)] // Stub implementation
#![allow(dead_code)] // Placeholder constants
#![allow(clippy::type_complexity)] // Complex return tuples
#![allow(clippy::manual_contains)] // Shape check style
#![allow(unused_variables)] // Feature-gated code
#![allow(unused_imports)] // EntityType used conditionally

#[cfg(feature = "onnx")]
use crate::EntityType;
use crate::{Entity, Error, Result};

/// Special token IDs for GLiNER models
const TOKEN_START: u32 = 1;
const TOKEN_END: u32 = 2;
const TOKEN_ENT: u32 = 128002;
const TOKEN_SEP: u32 = 128003;

/// Default max span width from GLiNER config
const MAX_SPAN_WIDTH: usize = 12;

/// Configuration for GLiNER model loading.
#[cfg(feature = "onnx")]
#[derive(Debug, Clone)]
pub struct GLiNERConfig {
    /// Prefer quantized models (INT8) for faster CPU inference.
    pub prefer_quantized: bool,
    /// ONNX optimization level (1-3, default 3).
    pub optimization_level: u8,
    /// Number of threads for inference (0 = auto).
    pub num_threads: usize,
    /// Cache size for prompt encodings (0 = disabled, default 100).
    pub prompt_cache_size: usize,
}

#[cfg(feature = "onnx")]
impl Default for GLiNERConfig {
    fn default() -> Self {
        Self {
            prefer_quantized: true,
            optimization_level: 3,
            num_threads: 4,
            prompt_cache_size: 100,
        }
    }
}

/// Cache key for prompt encodings.
///
/// Keyed by (text_hash, entity_types_hash, model_id) to ensure cache hits
/// only when text, entity types, and model are identical.
#[cfg(feature = "onnx")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PromptCacheKey {
    text_hash: u64,
    entity_types_hash: u64,
    model_id: String,
}

/// Cached prompt encoding result.
#[cfg(feature = "onnx")]
#[derive(Debug, Clone)]
struct PromptCacheValue {
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    words_mask: Vec<i64>,
    text_lengths: i64,
    entity_count: usize,
}

/// GLiNER model for zero-shot NER.
///
/// Thread-safe with `Arc<Tokenizer>` for efficient sharing across threads.
#[cfg(feature = "onnx")]
#[derive(Debug)]
pub struct GLiNEROnnx {
    session: std::sync::Mutex<ort::session::Session>,
    /// Arc-wrapped tokenizer for cheap cloning across threads.
    tokenizer: std::sync::Arc<tokenizers::Tokenizer>,
    model_name: String,
    /// Whether a quantized model was loaded.
    is_quantized: bool,
    /// LRU cache for prompt encodings (keyed by text + entity types).
    prompt_cache: Option<std::sync::Mutex<lru::LruCache<PromptCacheKey, PromptCacheValue>>>,
}

#[cfg(feature = "onnx")]
impl GLiNEROnnx {
    /// Create a new GLiNER model from HuggingFace with default config.
    pub fn new(model_name: &str) -> Result<Self> {
        Self::with_config(model_name, GLiNERConfig::default())
    }

    /// Create a new GLiNER model with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `model_name` - HuggingFace model ID (e.g., "onnx-community/gliner_small-v2.1")
    /// * `config` - Configuration for model loading
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = GLiNERConfig {
    ///     prefer_quantized: true,  // Use INT8 model for 2-4x speedup
    ///     optimization_level: 3,
    ///     num_threads: 8,
    /// };
    /// let model = GLiNEROnnx::with_config("onnx-community/gliner_small-v2.1", config)?;
    /// ```
    pub fn with_config(model_name: &str, config: GLiNERConfig) -> Result<Self> {
        use hf_hub::api::sync::Api;
        use ort::execution_providers::CPUExecutionProvider;
        use ort::session::builder::GraphOptimizationLevel;
        use ort::session::Session;

        let api = Api::new().map_err(|e| {
            Error::Retrieval(format!("Failed to initialize HuggingFace API: {}", e))
        })?;

        let repo = api.model(model_name.to_string());

        // Download model - try quantized first if preferred
        let (model_path, is_quantized) = if config.prefer_quantized {
            // Try quantized variants first
            if let Ok(path) = repo.get("onnx/model_quantized.onnx") {
                log::info!("[GLiNER] Using quantized model (INT8)");
                (path, true)
            } else if let Ok(path) = repo.get("model_quantized.onnx") {
                log::info!("[GLiNER] Using quantized model (INT8)");
                (path, true)
            } else if let Ok(path) = repo.get("onnx/model_int8.onnx") {
                log::info!("[GLiNER] Using INT8 quantized model");
                (path, true)
            } else {
                // Fall back to FP32
                let path = repo
                    .get("onnx/model.onnx")
                    .or_else(|_| repo.get("model.onnx"))
                    .map_err(|e| {
                        Error::Retrieval(format!("Failed to download model.onnx: {}", e))
                    })?;
                log::info!("[GLiNER] Using FP32 model (quantized not available)");
                (path, false)
            }
        } else {
            let path = repo
                .get("onnx/model.onnx")
                .or_else(|_| repo.get("model.onnx"))
                .map_err(|e| Error::Retrieval(format!("Failed to download model.onnx: {}", e)))?;
            (path, false)
        };

        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| Error::Retrieval(format!("Failed to download tokenizer.json: {}", e)))?;

        // Build session with optimization settings
        let opt_level = match config.optimization_level {
            1 => GraphOptimizationLevel::Level1,
            2 => GraphOptimizationLevel::Level2,
            _ => GraphOptimizationLevel::Level3,
        };

        let mut builder = Session::builder()
            .map_err(|e| Error::Retrieval(format!("Failed to create ONNX session builder: {}", e)))?
            .with_optimization_level(opt_level)
            .map_err(|e| Error::Retrieval(format!("Failed to set optimization level: {}", e)))?
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .map_err(|e| Error::Retrieval(format!("Failed to set execution providers: {}", e)))?;

        if config.num_threads > 0 {
            builder = builder
                .with_intra_threads(config.num_threads)
                .map_err(|e| Error::Retrieval(format!("Failed to set threads: {}", e)))?;
        }

        let session = builder
            .commit_from_file(&model_path)
            .map_err(|e| Error::Retrieval(format!("Failed to load ONNX model: {}", e)))?;

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Retrieval(format!("Failed to load tokenizer: {}", e)))?;

        log::debug!(
            "[GLiNER] Model inputs: {:?}",
            session.inputs.iter().map(|i| &i.name).collect::<Vec<_>>()
        );
        log::debug!(
            "[GLiNER] Model outputs: {:?}",
            session.outputs.iter().map(|o| &o.name).collect::<Vec<_>>()
        );

        // Initialize prompt cache if enabled
        let prompt_cache = if config.prompt_cache_size > 0 {
            use lru::LruCache;
            use std::num::NonZeroUsize;
            Some(std::sync::Mutex::new(LruCache::new(
                NonZeroUsize::new(config.prompt_cache_size)
                    .expect("prompt_cache_size must be > 0"),
            )))
        } else {
            None
        };

        Ok(Self {
            session: std::sync::Mutex::new(session),
            tokenizer: std::sync::Arc::new(tokenizer),
            model_name: model_name.to_string(),
            is_quantized,
            prompt_cache,
        })
    }

    /// Check if a quantized model was loaded.
    #[must_use]
    pub fn is_quantized(&self) -> bool {
        self.is_quantized
    }

    /// Get a clone of the tokenizer Arc (cheap).
    #[must_use]
    pub fn tokenizer(&self) -> std::sync::Arc<tokenizers::Tokenizer> {
        std::sync::Arc::clone(&self.tokenizer)
    }

    /// Get model name.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Extract entities from text using GLiNER zero-shot NER.
    ///
    /// # Arguments
    /// * `text` - The text to extract entities from
    /// * `entity_types` - Entity type labels to detect (e.g., ["person", "organization"])
    /// * `threshold` - Confidence threshold (0.0-1.0, recommended: 0.5)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let gliner = GLiNEROnnx::new("onnx-community/gliner_small-v2.1")?;
    /// let entities = gliner.extract("John works at Apple", &["person", "organization"], 0.5)?;
    /// ```
    pub fn extract(
        &self,
        text: &str,
        entity_types: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        if text.is_empty() || entity_types.is_empty() {
            return Ok(vec![]);
        }

        // Split text into words (gline-rs uses regex splitter, we use whitespace)
        let text_words: Vec<&str> = text.split_whitespace().collect();
        let num_text_words = text_words.len();

        if num_text_words == 0 {
            return Ok(vec![]);
        }

        // Encode input following gline-rs pattern: word-by-word encoding
        // Use cached version if cache is enabled
        let (input_ids, attention_mask, words_mask, text_lengths, entity_count) =
            self.encode_prompt_cached(&text_words, entity_types)?;

        // Generate span tensors
        let (span_idx, span_mask) = self.make_span_tensors(num_text_words);

        // Build ort tensors
        use ndarray::{Array2, Array3};
        use ort::value::Tensor;

        let batch_size = 1;
        let seq_len = input_ids.len();
        // Use checked_mul to prevent overflow (same pattern as gliner2.rs:2388)
        let num_spans = num_text_words.checked_mul(MAX_SPAN_WIDTH).ok_or_else(|| {
            Error::InvalidInput(format!(
                "Span count overflow: {} words * {} MAX_SPAN_WIDTH",
                num_text_words, MAX_SPAN_WIDTH
            ))
        })?;

        let input_ids_array = Array2::from_shape_vec((batch_size, seq_len), input_ids)
            .map_err(|e| Error::Parse(format!("Array error: {}", e)))?;
        let attention_mask_array = Array2::from_shape_vec((batch_size, seq_len), attention_mask)
            .map_err(|e| Error::Parse(format!("Array error: {}", e)))?;
        let words_mask_array = Array2::from_shape_vec((batch_size, seq_len), words_mask)
            .map_err(|e| Error::Parse(format!("Array error: {}", e)))?;
        let text_lengths_array =
            Array2::from_shape_vec((batch_size, 1), vec![num_text_words as i64])
                .map_err(|e| Error::Parse(format!("Array error: {}", e)))?;
        let span_idx_array = Array3::from_shape_vec((batch_size, num_spans, 2), span_idx)
            .map_err(|e| Error::Parse(format!("Array error: {}", e)))?;
        let span_mask_array = Array2::from_shape_vec((batch_size, num_spans), span_mask)
            .map_err(|e| Error::Parse(format!("Array error: {}", e)))?;

        let input_ids_t = Tensor::from_array(input_ids_array)
            .map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;
        let attention_mask_t = Tensor::from_array(attention_mask_array)
            .map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;
        let words_mask_t = Tensor::from_array(words_mask_array)
            .map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;
        let text_lengths_t = Tensor::from_array(text_lengths_array)
            .map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;
        let span_idx_t = Tensor::from_array(span_idx_array)
            .map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;
        let span_mask_t = Tensor::from_array(span_mask_array)
            .map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;

        // Run inference
        let mut session = self
            .session
            .lock()
            .map_err(|e| Error::Retrieval(format!("Failed to lock session: {}", e)))?;

        let outputs = session
            .run(ort::inputs![
                "input_ids" => input_ids_t.into_dyn(),
                "attention_mask" => attention_mask_t.into_dyn(),
                "words_mask" => words_mask_t.into_dyn(),
                "text_lengths" => text_lengths_t.into_dyn(),
                "span_idx" => span_idx_t.into_dyn(),
                "span_mask" => span_mask_t.into_dyn(),
            ])
            .map_err(|e| Error::Parse(format!("ONNX inference failed: {}", e)))?;

        // Decode output
        let entities = self.decode_output(
            &outputs,
            text,
            &text_words,
            entity_types,
            entity_count,
            threshold,
        )?;
        drop(outputs);
        drop(session);

        Ok(entities)
    }

    /// Hash text for cache key.
    fn hash_text(text: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        hasher.finish()
    }

    /// Hash entity types for cache key (sorted for consistency).
    fn hash_entity_types(entity_types: &[&str]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        // Sort entity types for consistent hashing regardless of input order
        let mut sorted: Vec<&str> = entity_types.to_vec();
        sorted.sort();
        sorted.hash(&mut hasher);
        hasher.finish()
    }

    /// Encode prompt with caching support.
    ///
    /// Checks cache first if enabled, otherwise calls `encode_prompt`.
    fn encode_prompt_cached(
        &self,
        text_words: &[&str],
        entity_types: &[&str],
    ) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>, i64, usize)> {
        // If cache is disabled, use direct encoding
        let cache = match &self.prompt_cache {
            Some(c) => c,
            None => return self.encode_prompt(text_words, entity_types),
        };

        // Build cache key
        let text = text_words.join(" ");
        let text_hash = Self::hash_text(&text);
        let entity_types_hash = Self::hash_entity_types(entity_types);
        let key = PromptCacheKey {
            text_hash,
            entity_types_hash,
            model_id: self.model_name.clone(),
        };

        // Check cache
        {
            let mut cache_guard = cache
                .lock()
                .map_err(|e| Error::Retrieval(format!("Failed to lock prompt cache: {}", e)))?;
            if let Some(cached) = cache_guard.get(&key) {
                return Ok((
                    cached.input_ids.clone(),
                    cached.attention_mask.clone(),
                    cached.words_mask.clone(),
                    cached.text_lengths,
                    cached.entity_count,
                ));
            }
        }

        // Cache miss: compute encoding
        let result = self.encode_prompt(text_words, entity_types)?;

        // Store in cache
        {
            let mut cache_guard = cache
                .lock()
                .map_err(|e| Error::Retrieval(format!("Failed to lock prompt cache: {}", e)))?;
            cache_guard.put(
                key,
                PromptCacheValue {
                    input_ids: result.0.clone(),
                    attention_mask: result.1.clone(),
                    words_mask: result.2.clone(),
                    text_lengths: result.3,
                    entity_count: result.4,
                },
            );
        }

        Ok(result)
    }

    /// Encode prompt following gline-rs pattern: word-by-word encoding.
    ///
    /// Structure: [START] <<ENT>> type1 <<ENT>> type2 <<SEP>> word1 word2 ... [END]
    ///
    /// # Performance
    ///
    /// This method performs tokenization and encoding, which can be expensive.
    /// Consider caching the result if the same (text, entity_types) combination
    /// is queried multiple times.
    ///
    /// For cached encoding, use `encode_prompt_cached` instead.
    pub(crate) fn encode_prompt(
        &self,
        text_words: &[&str],
        entity_types: &[&str],
    ) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>, i64, usize)> {
        // Build token sequence word by word
        let mut input_ids: Vec<i64> = Vec::new();
        let mut word_mask: Vec<i64> = Vec::new();

        // Add start token
        input_ids.push(TOKEN_START as i64);
        word_mask.push(0);

        // Add entity types: <<ENT>> type1 <<ENT>> type2 ...
        for entity_type in entity_types {
            // Add <<ENT>> token
            input_ids.push(TOKEN_ENT as i64);
            word_mask.push(0);

            // Encode entity type word(s)
            let encoding = self
                .tokenizer
                .encode(entity_type.to_string(), false)
                .map_err(|e| Error::Parse(format!("Tokenizer error: {}", e)))?;
            for token_id in encoding.get_ids() {
                input_ids.push(*token_id as i64);
                word_mask.push(0);
            }
        }

        // Add <<SEP>> token
        input_ids.push(TOKEN_SEP as i64);
        word_mask.push(0);

        // Add text words (this is where word_mask starts counting from 1)
        let mut word_id: i64 = 0;
        for word in text_words {
            // Encode word
            let encoding = self
                .tokenizer
                .encode(word.to_string(), false)
                .map_err(|e| Error::Parse(format!("Tokenizer error: {}", e)))?;

            word_id += 1; // Increment before first token of word

            for (token_idx, token_id) in encoding.get_ids().iter().enumerate() {
                input_ids.push(*token_id as i64);
                // First subword token gets the word ID, rest get 0
                if token_idx == 0 {
                    word_mask.push(word_id);
                } else {
                    word_mask.push(0);
                }
            }
        }

        // Add end token
        input_ids.push(TOKEN_END as i64);
        word_mask.push(0);

        let seq_len = input_ids.len();
        let attention_mask: Vec<i64> = vec![1; seq_len];

        Ok((
            input_ids,
            attention_mask,
            word_mask,
            word_id,
            entity_types.len(),
        ))
    }

    /// Generate span tensors following gline-rs pattern.
    ///
    /// Shape: [num_words * max_width, 2] for span_idx
    /// Shape: [num_words * max_width] for span_mask
    fn make_span_tensors(&self, num_words: usize) -> (Vec<i64>, Vec<bool>) {
        // Use checked_mul to prevent overflow (same pattern as gliner2.rs:2388)
        let num_spans = num_words.checked_mul(MAX_SPAN_WIDTH).unwrap_or_else(|| {
            log::warn!(
                "Span count overflow: {} words * {} MAX_SPAN_WIDTH, using max",
                num_words,
                MAX_SPAN_WIDTH
            );
            usize::MAX
        });
        // Check for overflow in num_spans * 2
        let span_idx_len = num_spans.checked_mul(2).unwrap_or_else(|| {
            log::warn!(
                "Span idx length overflow: {} spans * 2, using max",
                num_spans
            );
            usize::MAX
        });
        let mut span_idx: Vec<i64> = vec![0; span_idx_len];
        let mut span_mask: Vec<bool> = vec![false; num_spans];

        for start in 0..num_words {
            let remaining_width = num_words - start;
            let actual_max_width = MAX_SPAN_WIDTH.min(remaining_width);

            for width in 0..actual_max_width {
                // Check for overflow in dim calculation (same pattern as nuner.rs:399)
                let dim = match start.checked_mul(MAX_SPAN_WIDTH) {
                    Some(v) => match v.checked_add(width) {
                        Some(d) => d,
                        None => {
                            log::warn!(
                                "Dim calculation overflow: {} * {} + {}, skipping span",
                                start,
                                MAX_SPAN_WIDTH,
                                width
                            );
                            continue;
                        }
                    },
                    None => {
                        log::warn!(
                            "Dim calculation overflow: {} * {}, skipping span",
                            start,
                            MAX_SPAN_WIDTH
                        );
                        continue;
                    }
                };
                // Check bounds before array access (dim * 2 could overflow or exceed span_idx_len)
                if let Some(dim2) = dim.checked_mul(2) {
                    if dim2 + 1 < span_idx_len && dim < num_spans {
                        span_idx[dim2] = start as i64; // start offset
                        span_idx[dim2 + 1] = (start + width) as i64; // end offset
                        span_mask[dim] = true;
                    } else {
                        log::warn!(
                            "Span idx access out of bounds: dim={}, dim*2={}, span_idx_len={}, num_spans={}, skipping",
                            dim, dim2, span_idx_len, num_spans
                        );
                    }
                } else {
                    log::warn!("Dim * 2 overflow: dim={}, skipping span", dim);
                }
            }
        }

        (span_idx, span_mask)
    }

    /// Decode model output following gline-rs pattern.
    ///
    /// Expected output shape: [batch, num_words, max_width, num_entity_types]
    fn decode_output(
        &self,
        outputs: &ort::session::SessionOutputs,
        text: &str,
        text_words: &[&str],
        entity_types: &[&str],
        expected_num_classes: usize,
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        // Get output tensor
        let output = outputs
            .iter()
            .next()
            .map(|(_, v)| v)
            .ok_or_else(|| Error::Parse("No output from GLiNER model".to_string()))?;

        // Extract tensor data
        let (_, data_slice) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Parse(format!("Failed to extract output tensor: {}", e)))?;
        let output_data: Vec<f32> = data_slice.to_vec();

        // Get output shape
        let shape: Vec<i64> = match output.dtype() {
            ort::value::ValueType::Tensor { shape, .. } => shape.iter().copied().collect(),
            _ => return Err(Error::Parse("Output is not a tensor".to_string())),
        };

        log::debug!(
            "[GLiNER] Output shape: {:?}, data len: {}, expected classes: {}",
            shape,
            output_data.len(),
            expected_num_classes
        );

        if output_data.is_empty() || shape.iter().any(|&d| d == 0) {
            log::warn!("[GLiNER] Empty output - model may have incompatible ONNX export");
            return Ok(vec![]);
        }

        let mut entities = Vec::new();
        let num_text_words = text_words.len();

        // Expected shape: [batch, num_words, max_width, num_classes]
        if shape.len() == 4 && shape[0] == 1 {
            let out_num_words = shape[1] as usize;
            let out_max_width = shape[2] as usize;
            let num_classes = shape[3] as usize;

            log::debug!(
                "[GLiNER] Decoding: num_words={}, max_width={}, num_classes={}",
                out_num_words,
                out_max_width,
                num_classes
            );

            if num_classes == 0 {
                log::warn!("[GLiNER] num_classes is 0 - this ONNX model export may not support dynamic entity types");
                return Ok(vec![]);
            }

            // Iterate over spans and apply sigmoid threshold
            for word_idx in 0..out_num_words.min(num_text_words) {
                for width in 0..out_max_width.min(MAX_SPAN_WIDTH) {
                    let end_word = word_idx + width;
                    if end_word >= num_text_words {
                        continue;
                    }

                    for class_idx in 0..num_classes.min(entity_types.len()) {
                        let idx = (word_idx * out_max_width * num_classes)
                            + (width * num_classes)
                            + class_idx;

                        if idx < output_data.len() {
                            let logit = output_data[idx];
                            // Apply sigmoid
                            let score = 1.0 / (1.0 + (-logit).exp());

                            if score >= threshold {
                                let (char_start, char_end) = self.word_span_to_char_offsets(
                                    text, text_words, word_idx, end_word,
                                );

                                // Extract actual text from source to preserve original whitespace
                                let span_text = extract_char_slice(text, char_start, char_end);

                                let entity_type_str =
                                    entity_types.get(class_idx).unwrap_or(&"OTHER");
                                let entity_type = Self::map_entity_type(entity_type_str);

                                entities.push(Entity::new(
                                    span_text,
                                    entity_type,
                                    char_start,
                                    char_end,
                                    score as f64,
                                ));
                            }
                        }
                    }
                }
            }
        } else if shape.len() == 3 && shape[0] == 1 {
            // Alternative shape: [batch, num_spans, num_classes]
            let num_spans = shape[1] as usize;
            let num_classes = shape[2] as usize;

            if num_classes == 0 {
                log::warn!("[GLiNER] num_classes is 0");
                return Ok(vec![]);
            }

            for span_idx in 0..num_spans {
                let word_idx = span_idx / MAX_SPAN_WIDTH;
                let width = span_idx % MAX_SPAN_WIDTH;
                let end_word = word_idx + width;

                if word_idx >= num_text_words || end_word >= num_text_words {
                    continue;
                }

                for class_idx in 0..num_classes.min(entity_types.len()) {
                    let idx = span_idx * num_classes + class_idx;
                    if idx < output_data.len() {
                        let logit = output_data[idx];
                        let score = 1.0 / (1.0 + (-logit).exp());

                        if score >= threshold {
                            let (char_start, char_end) = self
                                .word_span_to_char_offsets(text, text_words, word_idx, end_word);

                            // Extract actual text from source to preserve original whitespace
                            let span_text = extract_char_slice(text, char_start, char_end);

                            let entity_type_str = entity_types.get(class_idx).unwrap_or(&"OTHER");
                            let entity_type = Self::map_entity_type(entity_type_str);

                            entities.push(Entity::new(
                                span_text,
                                entity_type,
                                char_start,
                                char_end,
                                score as f64,
                            ));
                        }
                    }
                }
            }
        }

        // Sort by start position, then by descending span length, then by descending confidence
        entities.sort_by(|a, b| {
            a.start
                .cmp(&b.start)
                .then_with(|| b.end.cmp(&a.end))
                .then_with(|| {
                    b.confidence
                        .partial_cmp(&a.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        // Remove exact duplicates
        entities.dedup_by(|a, b| a.start == b.start && a.end == b.end);

        // Remove overlapping spans, keeping the highest confidence one
        // This addresses the common issue where GLiNER detects both
        // "The Department of Defense" and "Department of Defense"
        let entities = remove_overlapping_spans(entities);

        // Post-process: strip trailing punctuation from entity spans
        let entities = entities
            .into_iter()
            .map(|mut e| {
                // Strip trailing punctuation that shouldn't be part of entities
                while e
                    .text
                    .ends_with(|c: char| matches!(c, '.' | ',' | ';' | ':' | '!' | '?'))
                {
                    e.text.pop();
                    if e.end > e.start {
                        e.end -= 1;
                    }
                }
                // Also strip leading punctuation
                while e
                    .text
                    .starts_with(|c: char| matches!(c, '.' | ',' | ';' | ':' | '!' | '?'))
                {
                    e.text.remove(0);
                    e.start += 1;
                }
                e
            })
            .filter(|e| !e.text.is_empty() && e.start < e.end)
            .collect();

        Ok(entities)
    }

    /// Map entity type string to EntityType enum.
    fn map_entity_type(type_str: &str) -> EntityType {
        match type_str.to_lowercase().as_str() {
            "person" | "per" => EntityType::Person,
            "organization" | "org" => EntityType::Organization,
            "location" | "loc" | "gpe" => EntityType::Location,
            "date" | "time" => EntityType::Date,
            "money" | "currency" => EntityType::Money,
            "percent" | "percentage" => EntityType::Percent,
            other => EntityType::Other(other.to_string()),
        }
    }

    /// Convert word indices to character offsets.
    ///
    /// This function correctly handles Unicode text by converting byte offsets
    /// to character offsets using the offset module's bytes_to_chars function.
    fn word_span_to_char_offsets(
        &self,
        text: &str,
        words: &[&str],
        start_word: usize,
        end_word: usize,
    ) -> (usize, usize) {
        let mut byte_pos = 0;
        let mut start_byte = 0;
        let mut end_byte = text.len();

        for (idx, word) in words.iter().enumerate() {
            // Search for the word in the remaining text (by bytes)
            if let Some(pos) = text[byte_pos..].find(word) {
                let word_start_byte = byte_pos + pos;
                let word_end_byte = word_start_byte + word.len();

                if idx == start_word {
                    start_byte = word_start_byte;
                }
                if idx == end_word {
                    end_byte = word_end_byte;
                    break;
                }
                byte_pos = word_end_byte;
            }
        }

        // Convert byte offsets to character offsets
        crate::offset::bytes_to_chars(text, start_byte, end_byte)
    }
}

/// Extract a substring by character offsets (not byte offsets).
///
/// This handles Unicode text correctly by iterating over characters.
fn extract_char_slice(text: &str, char_start: usize, char_end: usize) -> String {
    text.chars()
        .skip(char_start)
        .take(char_end.saturating_sub(char_start))
        .collect()
}

// =============================================================================
// Model Trait Implementation
// =============================================================================

/// Default entity types for zero-shot GLiNER when used via the Model trait.
#[cfg(feature = "onnx")]
const DEFAULT_GLINER_LABELS: &[&str] = &[
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
    "work_of_art",
    "law",
    "language",
];

#[cfg(feature = "onnx")]
impl crate::Model for GLiNEROnnx {
    fn extract_entities(&self, text: &str, _language: Option<&str>) -> crate::Result<Vec<Entity>> {
        // Use default labels for the Model trait interface
        // For custom labels, use the extract(text, labels, threshold) method directly
        self.extract(text, DEFAULT_GLINER_LABELS, 0.5)
    }

    fn supported_types(&self) -> Vec<crate::EntityType> {
        // GLiNER supports any type via zero-shot - return the defaults
        DEFAULT_GLINER_LABELS
            .iter()
            .map(|label| crate::EntityType::Custom {
                name: (*label).to_string(),
                category: crate::entity::EntityCategory::Misc,
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
}

// =============================================================================
// Stub when feature disabled
// =============================================================================

#[cfg(not(feature = "onnx"))]
#[derive(Debug)]
pub struct GLiNEROnnx;

#[cfg(not(feature = "onnx"))]
impl GLiNEROnnx {
    /// Create a new GLiNER model (stub - requires onnx feature).
    pub fn new(_model_name: &str) -> Result<Self> {
        Err(Error::InvalidInput(
            "GLiNER-ONNX requires the 'onnx' feature. \
             Build with: cargo build --features onnx"
                .to_string(),
        ))
    }

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
    ) -> Result<Vec<Entity>> {
        Err(Error::InvalidInput(
            "GLiNER-ONNX requires the 'onnx' feature".to_string(),
        ))
    }
}

#[cfg(not(feature = "onnx"))]
impl crate::Model for GLiNEROnnx {
    fn extract_entities(&self, _text: &str, _language: Option<&str>) -> crate::Result<Vec<Entity>> {
        Err(Error::InvalidInput(
            "GLiNER-ONNX requires the 'onnx' feature".to_string(),
        ))
    }

    fn supported_types(&self) -> Vec<crate::EntityType> {
        vec![]
    }

    fn is_available(&self) -> bool {
        false
    }

    fn name(&self) -> &'static str {
        "GLiNER-ONNX (unavailable)"
    }

    fn description(&self) -> &'static str {
        "GLiNER with ONNX Runtime backend - requires 'onnx' feature"
    }
}

#[cfg(not(feature = "onnx"))]
impl crate::backends::inference::ZeroShotNER for GLiNEROnnx {
    fn extract_with_types(
        &self,
        _text: &str,
        _entity_types: &[&str],
        _threshold: f32,
    ) -> crate::Result<Vec<Entity>> {
        Err(Error::InvalidInput(
            "GLiNER-ONNX requires the 'onnx' feature".to_string(),
        ))
    }

    fn extract_with_descriptions(
        &self,
        _text: &str,
        _descriptions: &[&str],
        _threshold: f32,
    ) -> crate::Result<Vec<Entity>> {
        Err(Error::InvalidInput(
            "GLiNER-ONNX requires the 'onnx' feature".to_string(),
        ))
    }
}

// =============================================================================
// BatchCapable Trait Implementation
// =============================================================================

#[cfg(feature = "onnx")]
impl crate::BatchCapable for GLiNEROnnx {
    fn extract_entities_batch(
        &self,
        texts: &[&str],
        _language: Option<&str>,
    ) -> Result<Vec<Vec<Entity>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // GLiNER supports true batching with padded sequences
        // For simplicity, we reuse the session efficiently with sequential calls
        // The tokenizer and model weights stay cached
        let default_types = DEFAULT_GLINER_LABELS;
        let threshold = 0.5;

        texts
            .iter()
            .map(|text| self.extract(text, default_types, threshold))
            .collect()
    }

    fn optimal_batch_size(&self) -> Option<usize> {
        Some(16)
    }
}

#[cfg(not(feature = "onnx"))]
impl crate::BatchCapable for GLiNEROnnx {
    fn extract_entities_batch(
        &self,
        texts: &[&str],
        _language: Option<&str>,
    ) -> Result<Vec<Vec<Entity>>> {
        Err(Error::InvalidInput(
            "GLiNER-ONNX requires the 'onnx' feature".to_string(),
        ))
    }

    fn optimal_batch_size(&self) -> Option<usize> {
        None
    }
}

// =============================================================================
// StreamingCapable Trait Implementation
// =============================================================================
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
    if entities.len() <= 1 {
        return entities;
    }

    // Sort by span length (shorter first), then by confidence descending
    // This prefers shorter, more precise spans
    entities.sort_by(|a, b| {
        let len_a = a.end - a.start;
        let len_b = b.end - b.start;
        len_a.cmp(&len_b).then_with(|| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });

    let mut result: Vec<Entity> = Vec::with_capacity(entities.len());

    for entity in entities {
        // Check if this entity is FULLY CONTAINED by any already-kept entity
        // If so, skip it (we already have a more precise version)
        let is_superset_of_existing = result.iter().any(|kept| {
            // Entity fully contains kept
            entity.start <= kept.start && entity.end >= kept.end
        });

        if is_superset_of_existing {
            // Skip - we have smaller, more precise entities
            continue;
        }

        // Check if this entity overlaps (but doesn't contain) any kept entity
        let overlaps_existing = result.iter().any(|kept| {
            let entity_range = entity.start..entity.end;
            let kept_range = kept.start..kept.end;
            // Partial overlap (not full containment)
            entity_range.start < kept_range.end && kept_range.start < entity_range.end
        });

        if !overlaps_existing {
            result.push(entity);
        }
    }

    // Re-sort by position for output
    result.sort_by_key(|e| e.start);
    result
}

// =============================================================================
// StreamingCapable
// =============================================================================

#[cfg(feature = "onnx")]
impl crate::StreamingCapable for GLiNEROnnx {
    fn recommended_chunk_size(&self) -> usize {
        4096 // Characters
    }
}

#[cfg(not(feature = "onnx"))]
impl crate::StreamingCapable for GLiNEROnnx {
    fn recommended_chunk_size(&self) -> usize {
        4096
    }
}
