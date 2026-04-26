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

        // Download model - try ONNX first, fall back to auto-export from PyTorch
        let (model_path, is_quantized) =
            hf_loader::download_onnx_model(&repo, config.prefer_quantized).or_else(|_| {
                log::info!(
                    "[GLiNER] No ONNX model in repo '{}', attempting auto-export from PyTorch...",
                    model_name
                );
                export_pytorch_to_onnx(model_name, &repo, config.prefer_quantized)
            })?;

        let tokenizer_path = hf_loader::download_model_file(&repo, &["tokenizer.json"])?;

        let session = hf_loader::create_onnx_session(
            &model_path,
            hf_loader::OnnxSessionConfig {
                optimization_level: config.optimization_level,
                num_threads: config.num_threads,
                use_cpu_provider: true,
                prefer_coreml: false,
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

        // Read class_token_index from gliner_config.json (if present).
        let (token_ent, token_sep) =
            match hf_loader::download_model_file(&repo, &["gliner_config.json"]) {
                Ok(config_path) => {
                    let config_str = std::fs::read_to_string(&config_path).unwrap_or_default();
                    let config_json: serde_json::Value =
                        serde_json::from_str(&config_str).unwrap_or_default();
                    let ent = config_json
                        .get("class_token_index")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as u32)
                        .unwrap_or(DEFAULT_TOKEN_ENT);
                    // sep is typically class_token_index + 1
                    let sep = ent.saturating_add(1);
                    if ent != DEFAULT_TOKEN_ENT {
                        log::info!(
                        "[GLiNER] Using class_token_index={} from gliner_config.json (default={})",
                        ent,
                        DEFAULT_TOKEN_ENT
                    );
                    }
                    (ent, sep)
                }
                Err(_) => (DEFAULT_TOKEN_ENT, DEFAULT_TOKEN_SEP),
            };

        // Detect whether the model expects span_idx/span_mask inputs.
        // Token-level classifiers (e.g., gliner-pii-edge) don't have these.
        let has_span_inputs = session.inputs().iter().any(|i| i.name() == "span_idx");

        if !has_span_inputs {
            log::info!("[GLiNER] Token-level model detected (no span_idx input)");
        }

        // For bi-encoder models, try to load the separate label encoder ONNX session.
        // The export script produces `label_encoder.onnx` alongside `model.onnx`.
        let (label_encoder_session, label_tokenizer) = if encoder_mode == config::EncoderMode::Bi {
            log::info!("[GLiNER] Bi-encoder model detected, loading label encoder...");

            // Try HF repo first, then check the local cache dir (auto-export writes there)
            let le_path = hf_loader::download_model_file(
                &repo,
                &["label_encoder.onnx", "label_encoder_quantized.onnx"],
            )
            .ok()
            .or_else(|| {
                // Check the same directory as the main model (auto-export output location)
                let cache_dir = model_path.parent()?;
                let local = cache_dir.join("label_encoder.onnx");
                if local.exists() {
                    Some(local)
                } else {
                    None
                }
            });

            let le_session = le_path.and_then(|path| {
                hf_loader::create_onnx_session(
                    &path,
                    hf_loader::OnnxSessionConfig {
                        optimization_level: config.optimization_level,
                        num_threads: config.num_threads,
                        use_cpu_provider: true,
                        prefer_coreml: false,
                    },
                )
                .ok()
            });

            if le_session.is_some() {
                log::info!("[GLiNER] Label encoder loaded");
            } else {
                log::warn!(
                    "[GLiNER] Bi-encoder model detected but label_encoder.onnx not found. \
                     Label embeddings must be pre-computed externally."
                );
            }

            // Try loading a separate label tokenizer (BGE models use different tokenizer).
            // Check HF repo first, then local cache dir.
            let le_tokenizer = hf_loader::download_model_file(&repo, &["label_tokenizer.json"])
                .ok()
                .or_else(|| {
                    let cache_dir = model_path.parent()?;
                    let local = cache_dir.join("label_tokenizer.json");
                    if local.exists() {
                        Some(local)
                    } else {
                        None
                    }
                })
                .and_then(|path| hf_loader::load_tokenizer(&path).ok());

            (
                le_session.map(Mutex::new),
                le_tokenizer.map(std::sync::Arc::new),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            session: Mutex::new(session),
            tokenizer: std::sync::Arc::new(tokenizer),
            model_name: model_name.to_string(),
            is_quantized,
            prompt_cache,
            encoder_mode,
            token_ent,
            token_sep,
            has_span_inputs,
            label_cache: Mutex::new(HashMap::new()),
            label_encoder_session,
            label_tokenizer,
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

        let le_session_mutex = self.label_encoder_session.as_ref().ok_or_else(|| {
            Error::FeatureNotAvailable(
                "Bi-encoder mode requires label_encoder.onnx (not found in model repo)".into(),
            )
        })?;

        let tokenizer = self.label_tokenizer.as_ref().unwrap_or(&self.tokenizer);

        let mut cache = self
            .label_cache
            .lock()
            .map_err(|e| Error::Retrieval(format!("label cache lock poisoned: {e}")))?;

        // Collect labels that need encoding
        let to_encode: Vec<&str> = labels
            .iter()
            .filter(|&&l| !cache.contains_key(l))
            .copied()
            .collect();

        if to_encode.is_empty() {
            return Ok(());
        }

        // Tokenize all labels, padding to max length in the batch
        let expanded: Vec<String> = to_encode
            .iter()
            .map(|l| expand_ner_label(l).to_string())
            .collect();
        let encodings: Vec<_> = expanded
            .iter()
            .map(|label| {
                tokenizer
                    .encode(label.as_str(), true)
                    .map_err(|e| Error::Parse(format!("Label tokenizer error: {e}")))
            })
            .collect::<Result<Vec<_>>>()?;

        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);
        let num_labels = encodings.len();

        // Build padded input tensors [num_labels, max_len]
        let mut input_ids_flat = vec![0i64; num_labels * max_len];
        let mut attention_mask_flat = vec![0i64; num_labels * max_len];

        for (i, enc) in encodings.iter().enumerate() {
            for (j, &id) in enc.get_ids().iter().enumerate() {
                input_ids_flat[i * max_len + j] = id as i64;
                attention_mask_flat[i * max_len + j] = 1;
            }
        }

        // Run the label encoder session
        use ndarray::Array2;
        let ids_array = Array2::from_shape_vec((num_labels, max_len), input_ids_flat)
            .map_err(|e| Error::Parse(format!("Array error: {e}")))?;
        let mask_array = Array2::from_shape_vec((num_labels, max_len), attention_mask_flat)
            .map_err(|e| Error::Parse(format!("Array error: {e}")))?;

        let ids_tensor = crate::backends::ort_compat::tensor_from_ndarray(ids_array)
            .map_err(|e| Error::Parse(format!("Tensor error: {e}")))?;
        let mask_tensor = crate::backends::ort_compat::tensor_from_ndarray(mask_array)
            .map_err(|e| Error::Parse(format!("Tensor error: {e}")))?;

        let mut le_session = le_session_mutex.lock().unwrap_or_else(|e| e.into_inner());

        let outputs = le_session
            .run(ort::inputs![
                "labels_input_ids" => ids_tensor.into_dyn(),
                "labels_attention_mask" => mask_tensor.into_dyn(),
            ])
            .map_err(|e| Error::Parse(format!("Label encoder inference failed: {e}")))?;

        // Extract embeddings: output shape [num_labels, hidden_dim]
        let (_, data_slice) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Parse(format!("Failed to extract label embeddings: {e}")))?;
        let data: Vec<f32> = data_slice.to_vec();

        // Infer hidden_dim from total data length / num_labels
        let hidden_dim = data.len().checked_div(num_labels).unwrap_or(0);

        for (i, &label) in to_encode.iter().enumerate() {
            let start = i * hidden_dim;
            let end = start + hidden_dim;
            let embedding = if end <= data.len() {
                data[start..end].to_vec()
            } else {
                vec![0.0; hidden_dim]
            };

            cache.insert(label.to_string(), config::LabelEmbedding { embedding });
        }

        log::info!(
            "[GLiNER] Pre-computed {} label embeddings (dim={})",
            to_encode.len(),
            hidden_dim
        );

        Ok(())
    }

    /// Clear all cached label embeddings.
    pub fn clear_label_cache(&self) {
        if let Ok(mut cache) = self.label_cache.lock() {
            cache.clear();
        }
    }

    /// Bi-encoder extraction: encode text separately, use cached label embeddings.
    ///
    /// The main model takes `labels_embeddings` as a pre-computed input tensor
    /// instead of embedding labels in the prompt. This allows label embeddings
    /// to be computed once and reused across many texts.
    fn extract_bi_encoder(
        &self,
        text: &str,
        text_words: &[&str],
        entity_types: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        use ndarray::{Array2, Array3};

        let num_text_words = text_words.len();

        // Ensure labels are pre-computed
        self.precompute_labels(entity_types)?;

        // Build label embeddings tensor from cache [num_labels, hidden_dim]
        let cache = self
            .label_cache
            .lock()
            .map_err(|e| Error::Retrieval(format!("label cache lock: {e}")))?;

        let entity_count = entity_types.len();
        let hidden_dim = entity_types
            .iter()
            .filter_map(|&l| cache.get(l))
            .map(|e| e.embedding.len())
            .next()
            .unwrap_or(0);

        if hidden_dim == 0 {
            return Err(Error::InvalidInput(
                "Label embeddings have zero dimension -- label encoder may have failed".into(),
            ));
        }

        let mut labels_flat = Vec::with_capacity(entity_count * hidden_dim);
        for &label in entity_types {
            match cache.get(label) {
                Some(emb) if emb.embedding.len() == hidden_dim => {
                    labels_flat.extend_from_slice(&emb.embedding);
                }
                _ => {
                    return Err(Error::InvalidInput(format!(
                        "Missing or mismatched label embedding for '{label}'"
                    )));
                }
            }
        }
        drop(cache);

        // Encode text only (no entity type prefix)
        let (input_ids, attention_mask, words_mask) = self.encode_text_only(text_words)?;

        let (span_idx, span_mask) = self.make_span_tensors(num_text_words);

        let batch_size = 1;
        let seq_len = input_ids.len();
        let num_spans = num_text_words.checked_mul(MAX_SPAN_WIDTH).ok_or_else(|| {
            Error::InvalidInput(format!(
                "Span count overflow: {} * {}",
                num_text_words, MAX_SPAN_WIDTH
            ))
        })?;

        let ids_arr = Array2::from_shape_vec((batch_size, seq_len), input_ids)
            .map_err(|e| Error::Parse(format!("Array: {e}")))?;
        let mask_arr = Array2::from_shape_vec((batch_size, seq_len), attention_mask)
            .map_err(|e| Error::Parse(format!("Array: {e}")))?;
        let wmask_arr = Array2::from_shape_vec((batch_size, seq_len), words_mask)
            .map_err(|e| Error::Parse(format!("Array: {e}")))?;
        let tlen_arr = Array2::from_shape_vec((batch_size, 1), vec![num_text_words as i64])
            .map_err(|e| Error::Parse(format!("Array: {e}")))?;
        let span_arr = Array3::from_shape_vec((batch_size, num_spans, 2), span_idx)
            .map_err(|e| Error::Parse(format!("Array: {e}")))?;
        let smask_arr = Array2::from_shape_vec((batch_size, num_spans), span_mask)
            .map_err(|e| Error::Parse(format!("Array: {e}")))?;
        let labels_arr = Array2::from_shape_vec((entity_count, hidden_dim), labels_flat)
            .map_err(|e| Error::Parse(format!("Array: {e}")))?;

        let ids_t = crate::backends::ort_compat::tensor_from_ndarray(ids_arr)
            .map_err(|e| Error::Parse(format!("Tensor: {e}")))?;
        let mask_t = crate::backends::ort_compat::tensor_from_ndarray(mask_arr)
            .map_err(|e| Error::Parse(format!("Tensor: {e}")))?;
        let wmask_t = crate::backends::ort_compat::tensor_from_ndarray(wmask_arr)
            .map_err(|e| Error::Parse(format!("Tensor: {e}")))?;
        let tlen_t = crate::backends::ort_compat::tensor_from_ndarray(tlen_arr)
            .map_err(|e| Error::Parse(format!("Tensor: {e}")))?;
        let span_t = crate::backends::ort_compat::tensor_from_ndarray(span_arr)
            .map_err(|e| Error::Parse(format!("Tensor: {e}")))?;
        let smask_t = crate::backends::ort_compat::tensor_from_ndarray(smask_arr)
            .map_err(|e| Error::Parse(format!("Tensor: {e}")))?;
        let labels_t = crate::backends::ort_compat::tensor_from_ndarray(labels_arr)
            .map_err(|e| Error::Parse(format!("Tensor: {e}")))?;

        let mut session = self.session.lock().unwrap_or_else(|e| e.into_inner());

        let outputs = session
            .run(ort::inputs![
                "input_ids" => ids_t.into_dyn(),
                "attention_mask" => mask_t.into_dyn(),
                "words_mask" => wmask_t.into_dyn(),
                "text_lengths" => tlen_t.into_dyn(),
                "span_idx" => span_t.into_dyn(),
                "span_mask" => smask_t.into_dyn(),
                "labels_embeddings" => labels_t.into_dyn(),
            ])
            .map_err(|e| Error::Parse(format!("Bi-encoder ONNX inference failed: {e}")))?;

        let entities = self.decode_output(
            &outputs,
            text,
            text_words,
            entity_types,
            entity_count,
            threshold,
        )?;
        drop(outputs);
        drop(session);

        Ok(entities)
    }

    /// Encode text words only (no entity type prefix). Used by bi-encoder path.
    fn encode_text_only(&self, text_words: &[&str]) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>)> {
        let mut input_ids: Vec<i64> = Vec::new();
        let mut word_mask: Vec<i64> = Vec::new();

        // [START]
        input_ids.push(TOKEN_START as i64);
        word_mask.push(0);

        // Text words with word IDs (1-indexed, 0 = non-first subword)
        for (word_id, word) in (1_i64..).zip(text_words.iter()) {
            let encoding = self
                .tokenizer
                .encode(word.to_string(), false)
                .map_err(|e| Error::Parse(format!("Tokenizer error: {e}")))?;

            for (idx, &token_id) in encoding.get_ids().iter().enumerate() {
                input_ids.push(token_id as i64);
                word_mask.push(if idx == 0 { word_id } else { 0 });
            }
        }

        // [END]
        input_ids.push(TOKEN_END as i64);
        word_mask.push(0);

        let seq_len = input_ids.len();
        let attention_mask = vec![1i64; seq_len];

        Ok((input_ids, attention_mask, word_mask))
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

        // Branch: bi-encoder uses pre-computed label embeddings + text-only prompt;
        // uni-encoder concatenates labels and text into a single prompt.
        if self.encoder_mode == config::EncoderMode::Bi && self.label_encoder_session.is_some() {
            return self.extract_bi_encoder(text, &text_words, entity_types, threshold);
        }

        // --- Uni-encoder path (existing) ---

        // Encode input following the GLiNER prompt format: word-by-word encoding
        // Use cached version if cache is enabled
        let (input_ids, attention_mask, words_mask, text_lengths, entity_count) =
            self.encode_prompt_cached(&text_words, entity_types)?;

        // Generate span tensors
        let (span_idx, span_mask) = self.make_span_tensors(num_text_words);

        // Build ort tensors
        use ndarray::{Array2, Array3};

        let batch_size = 1;
        let seq_len = input_ids.len();
        // Use checked_mul to prevent overflow (same pattern as gliner_multitask/onnx.rs)
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
        // Run inference with blocking lock for thread-safe parallel access
        let mut session = self.session.lock().unwrap_or_else(|e| e.into_inner());

        let outputs = if self.has_span_inputs {
            let span_idx_t = crate::backends::ort_compat::tensor_from_ndarray(span_idx_array)
                .map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;
            let span_mask_t = crate::backends::ort_compat::tensor_from_ndarray(span_mask_array)
                .map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;
            session.run(ort::inputs![
                "input_ids" => input_ids_t.into_dyn(),
                "attention_mask" => attention_mask_t.into_dyn(),
                "words_mask" => words_mask_t.into_dyn(),
                "text_lengths" => text_lengths_t.into_dyn(),
                "span_idx" => span_idx_t.into_dyn(),
                "span_mask" => span_mask_t.into_dyn(),
            ])
        } else {
            // Token-level classifier (e.g., gliner-pii-edge): no span inputs
            session.run(ort::inputs![
                "input_ids" => input_ids_t.into_dyn(),
                "attention_mask" => attention_mask_t.into_dyn(),
                "words_mask" => words_mask_t.into_dyn(),
                "text_lengths" => text_lengths_t.into_dyn(),
            ])
        }
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
            input_ids.push(self.token_ent as i64);
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
        input_ids.push(self.token_sep as i64);
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
        // Use checked_mul to prevent overflow (same pattern as gliner_multitask/onnx.rs)
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
            "facility" | "fac" => EntityType::custom("FACILITY", crate::EntityCategory::Place),
            "product" | "prod" => EntityType::custom("PRODUCT", crate::EntityCategory::Misc),
            "misc" | "other" => EntityType::custom("MISC", crate::EntityCategory::Misc),
            "date" | "time" => EntityType::Date,
            "money" | "currency" => EntityType::Money,
            "percent" | "percentage" => EntityType::Percent,
            other => EntityType::custom(other, crate::EntityCategory::Misc),
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
    let input_names: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();

    // Bi-encoder exports have either:
    // - `labels_embeddings` input on the main model (pre-computed label vectors)
    // - `label_input_ids` input (inline label encoding)
    let is_bi = input_names
        .iter()
        .any(|&name| name == "labels_embeddings" || name == "label_input_ids");

    if is_bi {
        config::EncoderMode::Bi
    } else {
        config::EncoderMode::Uni
    }
}

// =============================================================================
// Model Trait Implementation
// =============================================================================

/// Auto-export a PyTorch GLiNER model to ONNX format.
///
/// Runs `scripts/export_gliner_poly_onnx.py` via `uv run` (or `python3` fallback).
/// The export produces `model.onnx` (+ optionally `label_encoder.onnx` for bi-encoder
/// models) in the HF cache directory alongside the PyTorch weights.
///
/// This follows the same pattern as `convert_pytorch_to_safetensors` in the Candle backend:
/// automatic on-demand conversion at load time, cached for subsequent loads.
#[cfg(feature = "onnx")]
fn export_pytorch_to_onnx(
    model_name: &str,
    repo: &hf_hub::api::sync::ApiRepo,
    prefer_quantized: bool,
) -> Result<(std::path::PathBuf, bool)> {
    use std::path::Path;

    // Verify PyTorch weights exist (otherwise this isn't an exportable model)
    let _pytorch_path = crate::backends::hf_loader::download_model_file(
        repo,
        &["pytorch_model.bin", "model.safetensors"],
    )
    .map_err(|_| {
        Error::Retrieval(format!(
            "Model '{}' has neither ONNX nor PyTorch weights -- cannot load or export",
            model_name
        ))
    })?;

    // Determine output directory (same as HF cache for this model)
    let cache_dir = _pytorch_path
        .parent()
        .ok_or_else(|| Error::Retrieval("Invalid model path".into()))?;

    let onnx_path = cache_dir.join("model.onnx");
    let quantized_path = cache_dir.join("model_quantized.onnx");

    // Check if already exported
    if prefer_quantized && quantized_path.exists() {
        log::info!(
            "[GLiNER] Using cached ONNX export (quantized): {:?}",
            quantized_path
        );
        return Ok((quantized_path, true));
    }
    if onnx_path.exists() {
        log::info!("[GLiNER] Using cached ONNX export: {:?}", onnx_path);
        return Ok((onnx_path, false));
    }

    log::info!(
        "[GLiNER] Exporting '{}' to ONNX (this may take a few minutes on first run)...",
        model_name
    );

    // Find the export script
    let script_candidates = [
        // Workspace root (when running from the anno repo)
        std::env::var("CARGO_MANIFEST_DIR")
            .map(|d| Path::new(&d).join("../scripts/export_gliner_poly_onnx.py"))
            .unwrap_or_default(),
        // Workspace scripts/ relative to CWD
        Path::new("scripts/export_gliner_poly_onnx.py").to_path_buf(),
    ];

    let script_path = script_candidates
        .iter()
        .find(|p| p.exists())
        .cloned()
        .unwrap_or_else(|| Path::new("scripts/export_gliner_poly_onnx.py").to_path_buf());

    // Build args
    let mut args = vec![
        "run".to_string(),
        "--script".to_string(),
        script_path.display().to_string(),
        "--model".to_string(),
        model_name.to_string(),
        "--output".to_string(),
        cache_dir.display().to_string(),
    ];
    if prefer_quantized {
        args.push("--quantize".to_string());
    }

    // Try uv first, fall back to python3
    let output = std::process::Command::new("uv")
        .args(&args)
        .output()
        .or_else(|_| {
            // python3 fallback: skip "run --script" prefix
            std::process::Command::new("python3")
                .arg(&script_path)
                .arg("--model")
                .arg(model_name)
                .arg("--output")
                .arg(cache_dir)
                .args(if prefer_quantized {
                    &["--quantize"][..]
                } else {
                    &[]
                })
                .output()
        })
        .map_err(|e| {
            Error::Retrieval(format!(
                "Failed to run ONNX export script (uv or python3 not found?): {e}"
            ))
        })?;

    if output.status.success() {
        // Check which output file exists
        if prefer_quantized && quantized_path.exists() {
            log::info!(
                "[GLiNER] Auto-export succeeded (quantized): {:?}",
                quantized_path
            );
            return Ok((quantized_path, true));
        }
        if onnx_path.exists() {
            log::info!("[GLiNER] Auto-export succeeded: {:?}", onnx_path);
            return Ok((onnx_path, false));
        }
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    Err(Error::Retrieval(format!(
        "ONNX export of '{}' failed.\n\
         Script: {}\n\
         Stderr: {}\n\
         Stdout: {}\n\n\
         Manual export: uv run scripts/export_gliner_poly_onnx.py --model {} --output <dir>",
        model_name,
        script_path.display(),
        stderr,
        stdout,
        model_name,
    )))
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_char_slice_with_len_basic() {
        assert_eq!(
            extract_char_slice_with_len("hello world", 0, 5, 11),
            "hello"
        );
        assert_eq!(
            extract_char_slice_with_len("hello world", 6, 11, 11),
            "world"
        );
        assert_eq!(extract_char_slice_with_len("abc", 0, 3, 3), "abc");
    }

    #[test]
    fn test_extract_char_slice_with_len_unicode() {
        let text = "北京 Beijing";
        let len = text.chars().count();
        assert_eq!(extract_char_slice_with_len(text, 0, 2, len), "北京");
        assert_eq!(extract_char_slice_with_len(text, 3, len, len), "Beijing");
    }

    #[test]
    fn test_extract_char_slice_with_len_bounds() {
        assert_eq!(extract_char_slice_with_len("hello", 10, 15, 5), "");
        assert_eq!(extract_char_slice_with_len("hello", 3, 1, 5), "");
        assert_eq!(extract_char_slice_with_len("hello", 2, 2, 5), "");
        assert_eq!(extract_char_slice_with_len("hello", 5, 6, 5), "");
        assert_eq!(extract_char_slice_with_len("", 0, 0, 0), "");
    }

    #[test]
    fn test_span_tensor_math() {
        // Verify the span generation algorithm used by make_span_tensors.
        // We replicate the algorithm here since make_span_tensors requires &self.
        let num_words: usize = 5;
        let max_width: usize = MAX_SPAN_WIDTH;
        let num_spans = num_words.checked_mul(max_width).unwrap();

        assert_eq!(num_spans, 5 * max_width);

        // Count valid spans: sum(min(max_width, num_words - start)) for start 0..num_words
        // = min(12,5) + min(12,4) + min(12,3) + min(12,2) + min(12,1) = 5+4+3+2+1 = 15
        let mut valid = 0;
        for start in 0..num_words {
            valid += max_width.min(num_words - start);
        }
        assert_eq!(valid, 15);
    }

    #[test]
    fn test_span_tensor_math_zero_words() {
        let num_words = 0;
        let num_spans = num_words * MAX_SPAN_WIDTH;
        assert_eq!(num_spans, 0);
    }

    #[test]
    fn test_span_tensor_math_single_word() {
        let num_words = 1;
        let max_width = MAX_SPAN_WIDTH;
        // Only one valid span: (0, 0)
        let valid = max_width.min(num_words); // min(12, 1) = 1
        assert_eq!(valid, 1);
    }

    #[test]
    fn test_bytes_to_chars_ascii() {
        let text = "New York City is great";
        let (start, end) = crate::offset::bytes_to_chars(text, 0, 13);
        let extracted: String = text.chars().skip(start).take(end - start).collect();
        assert_eq!(extracted, "New York City");
    }

    #[test]
    fn test_bytes_to_chars_unicode() {
        let text = "Visit 北京 for tourism";
        // "北京" starts at byte 6, ends at byte 12
        let (start, end) = crate::offset::bytes_to_chars(text, 6, 12);
        let extracted: String = text.chars().skip(start).take(end - start).collect();
        assert_eq!(extracted, "北京");
    }

    #[test]
    fn test_bytes_to_chars_emoji() {
        let text = "Hello 🌍 world";
        // "🌍" is 4 bytes at byte offset 6
        let (start, end) = crate::offset::bytes_to_chars(text, 6, 10);
        let extracted: String = text.chars().skip(start).take(end - start).collect();
        assert_eq!(extracted, "🌍");
    }
}
