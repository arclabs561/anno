//! GLiNER ONNX inference engine: extraction, tokenization, span scoring.

use super::config::*;
use super::*;

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
    ///
    /// Automatically loads `.env` for HF_TOKEN if present.
    pub fn with_config(model_name: &str, config: GLiNERConfig) -> Result<Self> {
        use crate::backends::hf_loader;

        let api = hf_loader::hf_api()?;
        let repo = api.model(model_name.to_string());

        // Download model - try quantized first if preferred
        let (model_path, is_quantized) =
            hf_loader::download_onnx_model(&repo, config.prefer_quantized)?;

        let tokenizer_path = hf_loader::download_model_file(&repo, &["tokenizer.json"])?;

        let session = hf_loader::create_onnx_session(
            &model_path,
            hf_loader::OnnxSessionConfig {
                optimization_level: config.optimization_level,
                num_threads: config.num_threads,
                use_cpu_provider: true,
            },
        )?;

        let tokenizer = hf_loader::load_tokenizer(&tokenizer_path)?;

        log::debug!("[GLiNER] Model loaded");

        // Initialize prompt cache if enabled
        let prompt_cache = if config.prompt_cache_size > 0 {
            use lru::LruCache;
            use std::num::NonZeroUsize;
            Some(Mutex::new(LruCache::new(
                NonZeroUsize::new(config.prompt_cache_size).expect("prompt_cache_size must be > 0"),
            )))
        } else {
            None
        };

        // Detect encoder mode: check ONNX input names for bi-encoder signature
        let encoder_mode = match config.bi_encoder {
            Some(true) => config::EncoderMode::Bi,
            Some(false) => config::EncoderMode::Uni,
            None => detect_encoder_mode(&session),
        };

        if encoder_mode == config::EncoderMode::Bi {
            log::info!("[GLiNER] Bi-encoder model detected");
        }

        Ok(Self {
            session: Mutex::new(session),
            tokenizer: std::sync::Arc::new(tokenizer),
            model_name: model_name.to_string(),
            is_quantized,
            prompt_cache,
            encoder_mode,
            label_cache: Mutex::new(HashMap::new()),
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

    /// Whether this model uses bi-encoder architecture.
    #[must_use]
    pub fn is_bi_encoder(&self) -> bool {
        self.encoder_mode == config::EncoderMode::Bi
    }

    /// Create a GLiNER model that forces bi-encoder mode.
    ///
    /// If the ONNX model does not actually have bi-encoder inputs
    /// (i.e., no `label_input_ids` input), this falls back to uni-encoder
    /// mode and logs a warning.
    pub fn new_bi_encoder(model_name: &str) -> Result<Self> {
        let config = GLiNERConfig {
            bi_encoder: Some(true),
            ..GLiNERConfig::default()
        };
        let mut model = Self::with_config(model_name, config)?;

        // Verify the ONNX model actually supports bi-encoder inputs.
        // If not, fall back to uni-encoder rather than failing at inference time.
        if model.encoder_mode == config::EncoderMode::Bi {
            let session = model.session.lock().unwrap_or_else(|e| e.into_inner());
            let has_label_input = session
                .inputs()
                .iter()
                .any(|input| input.name() == "label_input_ids");
            drop(session);

            if !has_label_input {
                log::warn!(
                    "[GLiNER] Model '{}' was requested as bi-encoder but lacks \
                     'label_input_ids' input -- falling back to uni-encoder",
                    model_name
                );
                model.encoder_mode = config::EncoderMode::Uni;
            }
        }
        Ok(model)
    }

    /// Pre-compute and cache label embeddings for bi-encoder mode.
    ///
    /// In bi-encoder architecture, label embeddings are independent of the input
    /// text and can be computed once. This method encodes each label and stores the
    /// result in an internal cache. Subsequent `extract` calls reuse cached embeddings
    /// instead of re-encoding labels on every call.
    ///
    /// No-op in uni-encoder mode (labels are part of the concatenated prompt).
    ///
    /// # Errors
    ///
    /// Returns an error if the ONNX session fails during label encoding.
    /// Currently a no-op placeholder: actual label encoding requires a bi-encoder
    /// ONNX model with a label encoder head. When such models become available on
    /// HuggingFace, this method will run the label encoder session.
    pub fn precompute_labels(&self, labels: &[&str]) -> Result<()> {
        if self.encoder_mode != config::EncoderMode::Bi {
            log::debug!("[GLiNER] precompute_labels is a no-op in uni-encoder mode");
            return Ok(());
        }

        // TODO: When bi-encoder ONNX models are available, run the label encoder
        // session here. The expected flow:
        //
        // 1. For each label, tokenize: [CLS] label_text [SEP]
        // 2. Run the label encoder ONNX session (may be the same session with
        //    different input names, or a separate label_encoder.onnx file)
        // 3. Extract the [CLS] embedding (or pooled output)
        // 4. Store in label_cache
        //
        // For now, log and return success so callers can wire up the API.
        let mut cache = self
            .label_cache
            .lock()
            .map_err(|e| Error::Retrieval(format!("label cache lock poisoned: {e}")))?;

        for &label in labels {
            if cache.contains_key(label) {
                continue;
            }
            log::debug!(
                "[GLiNER] Bi-encoder label encoding not yet implemented for '{}'",
                label
            );
            // Placeholder: insert empty embedding so the cache key exists.
            // Real implementation will populate this with the encoder output.
            cache.insert(
                label.to_string(),
                config::LabelEmbedding {
                    embedding: Vec::new(),
                },
            );
        }

        Ok(())
    }

    /// Clear all cached label embeddings.
    pub fn clear_label_cache(&self) {
        if let Ok(mut cache) = self.label_cache.lock() {
            cache.clear();
        }
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

        // Split text into words (this implementation uses whitespace splitting)
        let text_words: Vec<&str> = text.split_whitespace().collect();
        let num_text_words = text_words.len();

        if num_text_words == 0 {
            return Ok(vec![]);
        }

        // Encode input following the GLiNER prompt format: word-by-word encoding
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

        let input_ids_t = crate::backends::ort_compat::tensor_from_ndarray(input_ids_array)
            .map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;
        let attention_mask_t =
            crate::backends::ort_compat::tensor_from_ndarray(attention_mask_array)
                .map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;
        let words_mask_t = crate::backends::ort_compat::tensor_from_ndarray(words_mask_array)
            .map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;
        let text_lengths_t = crate::backends::ort_compat::tensor_from_ndarray(text_lengths_array)
            .map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;
        let span_idx_t = crate::backends::ort_compat::tensor_from_ndarray(span_idx_array)
            .map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;
        let span_mask_t = crate::backends::ort_compat::tensor_from_ndarray(span_mask_array)
            .map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;

        // Run inference with blocking lock for thread-safe parallel access
        let mut session = self.session.lock().unwrap_or_else(|e| e.into_inner());

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

    /// Encode prompt with LRU caching for performance.
    ///
    /// Caches the result of `encode_prompt` keyed by (text_hash, entity_types_hash, model_id).
    /// This provides significant speedup when the same text is queried with different entity types
    /// (common in evaluation loops).
    ///
    /// # Lock Strategy
    ///
    /// The lock is dropped before the expensive `encode_prompt` operation to avoid blocking
    /// other threads. This allows concurrent cache lookups while encoding proceeds.
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

        // Check cache (lock scope minimized)
        let cached_result = {
            let mut cache_guard = cache
                .lock()
                .map_err(|e| crate::Error::Retrieval(format!("cache lock poisoned: {e}")))?;
            cache_guard.get(&key).cloned()
        };

        // Cache hit: return immediately
        if let Some(cached) = cached_result {
            return Ok((
                cached.input_ids,
                cached.attention_mask,
                cached.words_mask,
                cached.text_lengths,
                cached.entity_count,
            ));
        }

        // Cache miss: compute encoding (lock is dropped, allowing other threads to proceed)
        let result = self.encode_prompt(text_words, entity_types)?;

        // Store in cache (re-acquire lock)
        {
            let mut cache_guard = cache
                .lock()
                .map_err(|e| crate::Error::Retrieval(format!("cache lock poisoned: {e}")))?;
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

    /// Encode prompt following the GLiNER prompt format: word-by-word encoding.
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

            // Expand common NER abbreviations to full words the model was trained on
            let expanded = expand_ner_label(entity_type);

            // Encode entity type word(s)
            // Note: tokenizers::Tokenizer::encode requires String, not &str
            let encoding = self
                .tokenizer
                .encode(expanded, false)
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
            // Note: tokenizers::Tokenizer::encode requires String, not &str
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
        // Performance: Pre-allocate attention_mask with known size
        let mut attention_mask = Vec::with_capacity(seq_len);
        attention_mask.resize(seq_len, 1);

        Ok((
            input_ids,
            attention_mask,
            word_mask,
            word_id,
            entity_types.len(),
        ))
    }

    /// Generate span tensors following the GLiNER span layout.
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

    /// Decode model output following the GLiNER output layout.
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
        // Performance: Cache text length once (used in extract_char_slice calls)
        // ROI: High - called once, saves O(n) per entity in decode loops
        let text_char_count = text.chars().count();
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
            return Err(Error::Inference(
                "GLiNER ONNX returned empty/degenerate output tensor. This usually indicates an incompatible ONNX export for this implementation (shape mismatch or missing dynamic axes).".to_string(),
            ));
        }

        // Performance: Pre-allocate entities vec with estimated capacity
        // Most texts have 0-50 entities, but we'll start with a reasonable default
        let mut entities = Vec::with_capacity(32);
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
                return Err(Error::Inference(
                    "GLiNER ONNX model produced num_classes=0. This export likely does not support dynamic entity types for the requested schema.".to_string(),
                ));
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
                                // Performance: Use optimized extraction with cached length
                                let span_text = extract_char_slice_with_len(
                                    text,
                                    char_start,
                                    char_end,
                                    text_char_count,
                                );

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
                return Err(Error::Inference(
                    "GLiNER ONNX model produced num_classes=0. This export likely does not support dynamic entity types for the requested schema.".to_string(),
                ));
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
                            // Performance: Use optimized extraction with cached length
                            let span_text = extract_char_slice_with_len(
                                text,
                                char_start,
                                char_end,
                                text_char_count,
                            );

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

        // Performance: Use unstable sort (we don't need stable sort here)
        // Performance: Use unstable sort (we don't need stable sort here)
        // Sort by start position, then by descending span length, then by descending confidence
        entities.sort_unstable_by(|a, b| {
            a.start()
                .cmp(&b.start())
                .then_with(|| b.end().cmp(&a.end()))
                .then_with(|| {
                    b.confidence
                        .partial_cmp(&a.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        // Remove exact duplicates
        entities.dedup_by(|a, b| a.start() == b.start() && a.end() == b.end());

        // Remove overlapping spans, keeping the highest confidence one
        // This addresses the common issue where GLiNER detects both
        // "The Department of Defense" and "Department of Defense"
        let entities = remove_overlapping_spans(entities);

        // Post-process: strip trailing/leading punctuation from entity spans
        let entities = entities
            .into_iter()
            .filter_map(|mut e| {
                let (cleaned, head, tail) = textprep::spans::clean_span_boundary(&e.text);
                if cleaned.is_empty() {
                    return None;
                }
                let new_start = e.start() + head;
                let new_end = e.end() - tail;
                let cleaned_text = cleaned.to_string();
                e.set_start(new_start);
                e.set_end(new_end);
                e.text = cleaned_text;

                // Post-process: GLiNER sometimes tags obvious companies as PRODUCT.
                // If the surface form has strong company markers, remap PRODUCT → ORG.
                //
                // Keep this conservative: only remap when the mention itself looks like a company
                // ("Inc", "Ltd", "LLC", "株式会社", etc.) to avoid collapsing real products.
                if e.entity_type.as_label().eq_ignore_ascii_case("PRODUCT")
                    && looks_like_company_name(&e.text)
                {
                    e.entity_type = EntityType::Organization;
                }

                (e.start() < e.end()).then_some(e)
            })
            .collect();

        Ok(entities)
    }

    /// Map entity type string to EntityType enum.
    fn map_entity_type(type_str: &str) -> EntityType {
        match type_str.to_lowercase().as_str() {
            "person" | "per" => EntityType::Person,
            "organization" | "org" | "company" => EntityType::Organization,
            "location" | "loc" | "gpe" | "geo-loc" => EntityType::Location,
            "facility" | "fac" => EntityType::custom("FACILITY", anno_core::EntityCategory::Place),
            "product" | "prod" => EntityType::custom("PRODUCT", anno_core::EntityCategory::Misc),
            "misc" | "other" => EntityType::custom("MISC", anno_core::EntityCategory::Misc),
            "date" | "time" => EntityType::Date,
            "money" | "currency" => EntityType::Money,
            "percent" | "percentage" => EntityType::Percent,
            other => EntityType::custom(other, anno_core::EntityCategory::Misc),
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
        // Defensive: Validate bounds
        if words.is_empty()
            || start_word >= words.len()
            || end_word >= words.len()
            || start_word > end_word
        {
            // Return safe defaults: empty span (0, 0)
            return (0, 0);
        }

        let mut byte_pos = 0;
        let mut start_byte = 0;
        let mut end_byte = text.len();
        let mut found_start = false;
        let mut found_end = false;

        for (idx, word) in words.iter().enumerate() {
            // Search for the word in the remaining text (by bytes)
            if let Some(pos) = text[byte_pos..].find(word) {
                let word_start_byte = byte_pos + pos;
                let word_end_byte = word_start_byte + word.len();

                if idx == start_word {
                    start_byte = word_start_byte;
                    found_start = true;
                }
                if idx == end_word {
                    end_byte = word_end_byte;
                    found_end = true;
                    break;
                }
                byte_pos = word_end_byte;
            } else {
                // Word not found - this shouldn't happen in normal operation,
                // but if it does, we can't reliably compute offsets
            }
        }

        // If we didn't find the words, return safe defaults
        if !found_start || !found_end {
            // Return empty span to avoid incorrect entity extraction
            (0, 0)
        } else {
            // Convert byte offsets to character offsets
            crate::offset::bytes_to_chars(text, start_byte, end_byte)
        }
    }
}

pub(crate) fn looks_like_company_name(text: &str) -> bool {
    // Keep the logic cheap and conservative (no regex): normalize and check suffix markers.
    let t = text.trim();
    if t.is_empty() {
        return false;
    }

    let lower = t.to_lowercase();

    // Western-ish suffixes
    let suffixes = [
        " inc",
        " inc.",
        " ltd",
        " ltd.",
        " llc",
        " llp",
        " plc",
        " co",
        " co.",
        " company",
        " corp",
        " corp.",
        " corporation",
        " gmbh",
        " s.a.",
        " sa",
    ];
    if suffixes.iter().any(|s| lower.ends_with(s)) {
        return true;
    }

    // CJK org markers
    if t.contains("株式会社") || t.contains("有限会社") || t.contains("公司") || t.contains("集团")
    {
        return true;
    }

    // Arabic "company" marker
    if t.contains("شركة") {
        return true;
    }

    false
}

/// Expand common NER label abbreviations to the full-word form GLiNER was trained on.
///
/// GLiNER's ONNX model was trained with lowercase full-word labels in its prompt
/// (e.g. "person", "organization"). Users often pass uppercase abbreviations like
/// "PER", "ORG", "LOC" which the tokenizer encodes differently, producing zero
/// entities. This function maps known abbreviations to their training-time forms
/// and lowercases unknown labels.
pub(crate) fn expand_ner_label(label: &str) -> String {
    match label.to_uppercase().as_str() {
        "PER" | "PERSON" => "person".to_string(),
        "ORG" | "ORGANIZATION" => "organization".to_string(),
        "LOC" | "LOCATION" | "GPE" => "location".to_string(),
        "MISC" | "MISCELLANEOUS" => "miscellaneous".to_string(),
        "DATE" => "date".to_string(),
        "MONEY" => "money".to_string(),
        "TIME" => "time".to_string(),
        "PRODUCT" => "product".to_string(),
        "EVENT" => "event".to_string(),
        _ => label.to_lowercase(),
    }
}

/// Extract a substring by character offsets with pre-computed text length.
///
/// This is a performance optimization for batch operations where you've already
/// computed `text.chars().count()`.
fn extract_char_slice_with_len(
    text: &str,
    char_start: usize,
    char_end: usize,
    text_char_count: usize,
) -> String {
    if char_start >= text_char_count || char_end > text_char_count || char_start >= char_end {
        return String::new();
    }
    text.chars()
        .skip(char_start)
        .take(char_end.saturating_sub(char_start))
        .collect()
}

// =============================================================================
// Bi-encoder detection
// =============================================================================

/// Detect whether an ONNX session is a bi-encoder GLiNER model.
///
/// Checks for the presence of `label_input_ids` among the session inputs.
/// Bi-encoder exports (Stepanov et al., 2026) have separate text and label
/// encoder inputs, while the standard uni-encoder concatenates everything
/// into a single `input_ids` tensor.
fn detect_encoder_mode(session: &ort::session::Session) -> config::EncoderMode {
    let has_label_input = session
        .inputs()
        .iter()
        .any(|input| input.name() == "label_input_ids");

    if has_label_input {
        config::EncoderMode::Bi
    } else {
        config::EncoderMode::Uni
    }
}

// =============================================================================
// Model Trait Implementation
// =============================================================================

/// Default entity types for zero-shot GLiNER when used via the Model trait.
#[cfg(feature = "onnx")]
pub(super) const DEFAULT_GLINER_LABELS: &[&str] = &[
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
