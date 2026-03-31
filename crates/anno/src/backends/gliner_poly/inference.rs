//! GLiNER poly-encoder ONNX inference engine.
//!
//! Handles model loading from HuggingFace Hub, tokenization, tensor construction,
//! and output decoding. Follows the same patterns as `gliner_onnx::inference`.

use super::*;

/// Special token IDs for GLiNER poly-encoder models.
///
/// These match the standard GLiNER tokenizer vocabulary. If the poly-encoder
/// export uses a different tokenizer, these values may need adjustment.
const TOKEN_START: u32 = 1;
const TOKEN_END: u32 = 2;
const TOKEN_ENT: u32 = 128002;
const TOKEN_SEP: u32 = 128003;

/// Default max span width from GLiNER config.
pub(super) const MAX_SPAN_WIDTH: usize = 12;

#[cfg(feature = "onnx")]
impl GLiNERPoly {
    /// Create a new poly-encoder GLiNER model.
    ///
    /// Checks the local anno cache (`~/.cache/anno/models/gliner-poly/`) first.
    /// If not found, downloads from HuggingFace Hub.
    ///
    /// # Arguments
    ///
    /// * `model_name` - HuggingFace model ID
    ///   (e.g., `"knowledgator/gliner-bi-large-v1.0"`)
    pub fn new(model_name: &str) -> Result<Self> {
        // Check local caches first (exported ONNX models live here).
        for dir in &super::local_model_cache_candidates() {
            if dir.join("model.onnx").exists() && dir.join("tokenizer.json").exists() {
                log::info!("[GLiNERPoly] Found local model in {}", dir.display());
                return Self::from_dir(dir);
            }
        }
        // Fallback: try HuggingFace Hub (only works if repo has ONNX files).
        Self::with_options(model_name, true, 3, 4)
    }

    /// Load from a local directory containing `model.onnx` and `tokenizer.json`.
    pub fn from_dir(dir: &std::path::Path) -> Result<Self> {
        Self::from_dir_with_options(dir, 3, 4)
    }

    /// Load from a local directory with explicit ONNX session options.
    pub fn from_dir_with_options(
        dir: &std::path::Path,
        optimization_level: u8,
        num_threads: usize,
    ) -> Result<Self> {
        use ort::execution_providers::CPUExecutionProvider;
        use ort::session::builder::GraphOptimizationLevel;
        use ort::session::Session;

        let model_path = dir.join("model.onnx");
        let tokenizer_path = dir.join("tokenizer.json");

        if !model_path.exists() {
            return Err(Error::Retrieval(format!(
                "model.onnx not found in {}",
                dir.display()
            )));
        }
        if !tokenizer_path.exists() {
            return Err(Error::Retrieval(format!(
                "tokenizer.json not found in {}",
                dir.display()
            )));
        }

        let opt_level = match optimization_level {
            1 => GraphOptimizationLevel::Level1,
            2 => GraphOptimizationLevel::Level2,
            _ => GraphOptimizationLevel::Level3,
        };

        let mut builder = Session::builder()
            .map_err(|e| Error::Retrieval(format!("ONNX session builder: {}", e)))?
            .with_optimization_level(opt_level)
            .map_err(|e| Error::Retrieval(format!("ONNX optimization level: {}", e)))?
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .map_err(|e| Error::Retrieval(format!("ONNX execution providers: {}", e)))?;

        if num_threads > 0 {
            builder = builder
                .with_intra_threads(num_threads)
                .map_err(|e| Error::Retrieval(format!("ONNX thread config: {}", e)))?;
        }

        let session = builder
            .commit_from_file(&model_path)
            .map_err(|e| Error::Retrieval(format!("ONNX model load: {}", e)))?;

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Retrieval(format!("Tokenizer load: {}", e)))?;

        // Load label tokenizer from gliner_config.json -> labels_encoder field.
        let label_tokenizer = Self::load_label_tokenizer(dir)?;

        let is_quantized = model_path
            .file_name()
            .is_some_and(|n| n.to_string_lossy().contains("quantized"));

        log::info!(
            "[GLiNERPoly] Loaded from {} (quantized={})",
            dir.display(),
            is_quantized
        );

        Ok(Self {
            session: Mutex::new(session),
            tokenizer: std::sync::Arc::new(tokenizer),
            label_tokenizer: std::sync::Arc::new(label_tokenizer),
            model_name: dir.display().to_string(),
            is_quantized,
        })
    }

    /// Load the label encoder's tokenizer.
    ///
    /// Reads `gliner_config.json` to find `labels_encoder` (e.g. `BAAI/bge-base-en-v1.5`),
    /// then downloads that model's `tokenizer.json` from HuggingFace Hub.
    fn load_label_tokenizer(dir: &std::path::Path) -> Result<tokenizers::Tokenizer> {
        use hf_hub::api::sync::{Api, ApiBuilder};

        let config_path = dir.join("gliner_config.json");
        let labels_encoder_name = if config_path.exists() {
            let config_str = std::fs::read_to_string(&config_path)
                .map_err(|e| Error::Retrieval(format!("Read gliner_config.json: {}", e)))?;
            let config: serde_json::Value = serde_json::from_str(&config_str)
                .map_err(|e| Error::Retrieval(format!("Parse gliner_config.json: {}", e)))?;
            config
                .get("labels_encoder")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        } else {
            None
        };

        let labels_encoder_name =
            labels_encoder_name.unwrap_or_else(|| "BAAI/bge-base-en-v1.5".to_string());

        // Check if label tokenizer is cached locally.
        let local_label_tok = dir.join("label_tokenizer.json");
        if local_label_tok.exists() {
            return tokenizers::Tokenizer::from_file(&local_label_tok)
                .map_err(|e| Error::Retrieval(format!("Label tokenizer load: {}", e)));
        }

        log::info!(
            "[GLiNERPoly] Downloading label tokenizer from {}",
            labels_encoder_name
        );

        let api = crate::backends::hf_loader::hf_api()?;
        let repo = api.model(labels_encoder_name);
        let tok_path = crate::backends::hf_loader::download_model_file(&repo, &["tokenizer.json"])?;
        crate::backends::hf_loader::load_tokenizer(&tok_path)
    }

    /// Create a new poly-encoder GLiNER model with explicit options.
    ///
    /// # Arguments
    ///
    /// * `model_name` - HuggingFace model ID
    /// * `prefer_quantized` - Try quantized (INT8) model first for faster CPU inference
    /// * `optimization_level` - ONNX graph optimization (1-3)
    /// * `num_threads` - Intra-op thread count (0 = auto)
    pub fn with_options(
        model_name: &str,
        prefer_quantized: bool,
        optimization_level: u8,
        num_threads: usize,
    ) -> Result<Self> {
        use crate::backends::hf_loader;

        let api = hf_loader::hf_api()?;
        let repo = api.model(model_name.to_string());

        // Download model -- try quantized variants first if preferred.
        let (model_path, is_quantized) = hf_loader::download_onnx_model(&repo, prefer_quantized)?;

        // Download tokenizer.
        let tokenizer_path = hf_loader::download_model_file(&repo, &["tokenizer.json"])?;

        // Build ONNX session.
        let session = hf_loader::create_onnx_session(
            &model_path,
            hf_loader::OnnxSessionConfig {
                optimization_level,
                num_threads,
                use_cpu_provider: true,
            },
        )?;

        let tokenizer = hf_loader::load_tokenizer(&tokenizer_path)?;

        // Download label encoder tokenizer from gliner_config.json.
        let config_path = repo.get("gliner_config.json").ok();
        let labels_encoder_name = config_path
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|s: String| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|c: serde_json::Value| {
                c.get("labels_encoder")
                    .and_then(|v| v.as_str().map(|s| s.to_string()))
            })
            .unwrap_or_else(|| "BAAI/bge-base-en-v1.5".to_string());

        let label_repo = api.model(labels_encoder_name.clone());
        let label_tok_path = hf_loader::download_model_file(&label_repo, &["tokenizer.json"])?;
        let label_tokenizer = hf_loader::load_tokenizer(&label_tok_path)?;

        log::info!(
            "[GLiNERPoly] Loaded {} (quantized={}, labels_encoder={})",
            model_name,
            is_quantized,
            labels_encoder_name
        );

        Ok(Self {
            session: Mutex::new(session),
            tokenizer: std::sync::Arc::new(tokenizer),
            label_tokenizer: std::sync::Arc::new(label_tokenizer),
            model_name: model_name.to_string(),
            is_quantized,
        })
    }

    /// Check if a quantized model was loaded.
    #[must_use]
    pub fn is_quantized(&self) -> bool {
        self.is_quantized
    }

    /// Get the model name.
    #[must_use]
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Extract entities from text using poly-encoder GLiNER zero-shot NER.
    ///
    /// # Arguments
    ///
    /// * `text` - Input text
    /// * `entity_types` - Entity type labels to detect (e.g., `["person", "organization"]`)
    /// * `threshold` - Confidence threshold (0.0-1.0, recommended: 0.5)
    pub fn extract(
        &self,
        text: &str,
        entity_types: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        if text.is_empty() || entity_types.is_empty() {
            return Ok(vec![]);
        }

        let text_words: Vec<&str> = text.split_whitespace().collect();
        let num_text_words = text_words.len();
        if num_text_words == 0 {
            return Ok(vec![]);
        }

        // Encode input following the GLiNER prompt format.
        let (input_ids, attention_mask, words_mask) =
            self.encode_prompt(&text_words, entity_types)?;

        // Generate span tensors (same layout as GLiNEROnnx).
        let (span_idx, span_mask) = Self::make_span_tensors(num_text_words);
        let num_spans = num_text_words * MAX_SPAN_WIDTH;

        // Tokenize entity type labels for the label encoder.
        let (labels_input_ids, labels_attention_mask) = self.encode_labels(entity_types)?;
        let num_labels = entity_types.len();
        let label_seq_len = labels_input_ids.len() / num_labels;

        // Build ONNX tensors.
        use ndarray::{Array2, Array3};

        let seq_len = input_ids.len();

        let input_ids_arr = Array2::from_shape_vec((1, seq_len), input_ids)
            .map_err(|e| Error::Parse(format!("Array: {}", e)))?;
        let attention_mask_arr = Array2::from_shape_vec((1, seq_len), attention_mask)
            .map_err(|e| Error::Parse(format!("Array: {}", e)))?;
        let words_mask_arr = Array2::from_shape_vec((1, seq_len), words_mask)
            .map_err(|e| Error::Parse(format!("Array: {}", e)))?;
        let text_lengths_arr = Array2::from_shape_vec((1, 1), vec![num_text_words as i64])
            .map_err(|e| Error::Parse(format!("Array: {}", e)))?;
        let span_idx_arr = Array3::from_shape_vec((1, num_spans, 2), span_idx)
            .map_err(|e| Error::Parse(format!("Array: {}", e)))?;
        let span_mask_arr = Array2::from_shape_vec((1, num_spans), span_mask)
            .map_err(|e| Error::Parse(format!("Array: {}", e)))?;
        let labels_ids_arr = Array2::from_shape_vec((num_labels, label_seq_len), labels_input_ids)
            .map_err(|e| Error::Parse(format!("Array: {}", e)))?;
        let labels_mask_arr =
            Array2::from_shape_vec((num_labels, label_seq_len), labels_attention_mask)
                .map_err(|e| Error::Parse(format!("Array: {}", e)))?;

        let input_ids_t = crate::backends::ort_compat::tensor_from_ndarray(input_ids_arr)
            .map_err(|e| Error::Parse(format!("Tensor: {}", e)))?;
        let attention_mask_t = crate::backends::ort_compat::tensor_from_ndarray(attention_mask_arr)
            .map_err(|e| Error::Parse(format!("Tensor: {}", e)))?;
        let words_mask_t = crate::backends::ort_compat::tensor_from_ndarray(words_mask_arr)
            .map_err(|e| Error::Parse(format!("Tensor: {}", e)))?;
        let text_lengths_t = crate::backends::ort_compat::tensor_from_ndarray(text_lengths_arr)
            .map_err(|e| Error::Parse(format!("Tensor: {}", e)))?;
        let span_idx_t = crate::backends::ort_compat::tensor_from_ndarray(span_idx_arr)
            .map_err(|e| Error::Parse(format!("Tensor: {}", e)))?;
        let span_mask_t = crate::backends::ort_compat::tensor_from_ndarray(span_mask_arr)
            .map_err(|e| Error::Parse(format!("Tensor: {}", e)))?;
        let labels_ids_t = crate::backends::ort_compat::tensor_from_ndarray(labels_ids_arr)
            .map_err(|e| Error::Parse(format!("Tensor: {}", e)))?;
        let labels_mask_t = crate::backends::ort_compat::tensor_from_ndarray(labels_mask_arr)
            .map_err(|e| Error::Parse(format!("Tensor: {}", e)))?;

        // Run inference.
        let mut session = self.session.lock().unwrap_or_else(|e| e.into_inner());

        let outputs = session
            .run(ort::inputs![
                "input_ids" => input_ids_t.into_dyn(),
                "attention_mask" => attention_mask_t.into_dyn(),
                "words_mask" => words_mask_t.into_dyn(),
                "text_lengths" => text_lengths_t.into_dyn(),
                "span_idx" => span_idx_t.into_dyn(),
                "span_mask" => span_mask_t.into_dyn(),
                "labels_input_ids" => labels_ids_t.into_dyn(),
                "labels_attention_mask" => labels_mask_t.into_dyn(),
            ])
            .map_err(|e| Error::Inference(format!("ONNX inference failed: {}", e)))?;

        // Decode output.
        let entities = self.decode_output(&outputs, text, &text_words, entity_types, threshold)?;
        drop(outputs);
        drop(session);

        Ok(entities)
    }

    /// Generate span tensors following the GLiNER span layout.
    ///
    /// Returns `(span_idx, span_mask)` where:
    /// - `span_idx`: `[num_words * max_width, 2]` flattened — start/end word indices per span
    /// - `span_mask`: `[num_words * max_width]` — which spans are valid
    pub(super) fn make_span_tensors(num_words: usize) -> (Vec<i64>, Vec<bool>) {
        let num_spans = num_words * MAX_SPAN_WIDTH;
        let mut span_idx: Vec<i64> = vec![0; num_spans * 2];
        let mut span_mask: Vec<bool> = vec![false; num_spans];

        for start in 0..num_words {
            let remaining = num_words - start;
            let actual_max = MAX_SPAN_WIDTH.min(remaining);
            for width in 0..actual_max {
                let dim = start * MAX_SPAN_WIDTH + width;
                if dim < num_spans {
                    span_idx[dim * 2] = start as i64;
                    span_idx[dim * 2 + 1] = (start + width) as i64;
                    span_mask[dim] = true;
                }
            }
        }

        (span_idx, span_mask)
    }

    /// Tokenize entity type labels for the bi-encoder's label encoder input.
    ///
    /// Returns `(labels_input_ids, labels_attention_mask)` where each label is
    /// tokenized and padded to the same length.
    fn encode_labels(&self, entity_types: &[&str]) -> Result<(Vec<i64>, Vec<i64>)> {
        let mut all_ids: Vec<Vec<i64>> = Vec::with_capacity(entity_types.len());
        let mut max_len = 0;

        for label in entity_types {
            let encoding = self
                .label_tokenizer
                .encode(label.to_string(), true)
                .map_err(|e| Error::Parse(format!("Label tokenizer error: {}", e)))?;
            let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
            max_len = max_len.max(ids.len());
            all_ids.push(ids);
        }

        // Pad all labels to the same length.
        let mut labels_input_ids = Vec::with_capacity(entity_types.len() * max_len);
        let mut labels_attention_mask = Vec::with_capacity(entity_types.len() * max_len);

        for ids in &all_ids {
            labels_input_ids.extend_from_slice(ids);
            labels_attention_mask.extend(vec![1i64; ids.len()]);
            let pad = max_len - ids.len();
            labels_input_ids.extend(vec![0i64; pad]);
            labels_attention_mask.extend(vec![0i64; pad]);
        }

        Ok((labels_input_ids, labels_attention_mask))
    }

    /// Encode prompt following the GLiNER format: word-by-word encoding.
    ///
    /// Structure: `[START] <<ENT>> type1 <<ENT>> type2 <<SEP>> word1 word2 ... [END]`
    fn encode_prompt(
        &self,
        text_words: &[&str],
        entity_types: &[&str],
    ) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>)> {
        let mut input_ids: Vec<i64> = Vec::new();
        let mut word_mask: Vec<i64> = Vec::new();

        // Start token.
        input_ids.push(TOKEN_START as i64);
        word_mask.push(0);

        // Entity types: <<ENT>> type1 <<ENT>> type2 ...
        for entity_type in entity_types {
            input_ids.push(TOKEN_ENT as i64);
            word_mask.push(0);

            let encoding = self
                .tokenizer
                .encode(entity_type.to_string(), false)
                .map_err(|e| Error::Parse(format!("Tokenizer error: {}", e)))?;
            for token_id in encoding.get_ids() {
                input_ids.push(*token_id as i64);
                word_mask.push(0);
            }
        }

        // <<SEP>> token.
        input_ids.push(TOKEN_SEP as i64);
        word_mask.push(0);

        // Text words (word_mask starts counting from 1).
        for (word_idx, word) in text_words.iter().enumerate() {
            let encoding = self
                .tokenizer
                .encode(word.to_string(), false)
                .map_err(|e| Error::Parse(format!("Tokenizer error: {}", e)))?;

            let word_id = (word_idx + 1) as i64;
            for (token_idx, token_id) in encoding.get_ids().iter().enumerate() {
                input_ids.push(*token_id as i64);
                // First subword token gets the word ID, rest get 0.
                word_mask.push(if token_idx == 0 { word_id } else { 0 });
            }
        }

        // End token.
        input_ids.push(TOKEN_END as i64);
        word_mask.push(0);

        let seq_len = input_ids.len();
        let attention_mask: Vec<i64> = vec![1; seq_len];

        Ok((input_ids, attention_mask, word_mask))
    }

    /// Decode model output into entities.
    ///
    /// Supports two output shapes:
    /// - `[batch, num_words, max_width, num_entity_types]` (span-level)
    /// - `[3, batch, num_words, num_entity_types]` (BIO token-level)
    fn decode_output(
        &self,
        outputs: &ort::session::SessionOutputs,
        text: &str,
        text_words: &[&str],
        entity_types: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        let output = outputs
            .iter()
            .next()
            .map(|(_, v)| v)
            .ok_or_else(|| Error::Parse("No output from poly-encoder model".to_string()))?;

        let (_, data_slice) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Parse(format!("Failed to extract output tensor: {}", e)))?;
        let output_data: Vec<f32> = data_slice.to_vec();

        let shape: Vec<i64> = match output.dtype() {
            ort::value::ValueType::Tensor { shape, .. } => shape.iter().copied().collect(),
            _ => return Err(Error::Parse("Output is not a tensor".to_string())),
        };

        log::debug!(
            "[GLiNERPoly] Output shape: {:?}, data len: {}",
            shape,
            output_data.len()
        );

        if output_data.is_empty() || shape.contains(&0) {
            return Err(Error::Inference(
                "GLiNERPoly returned empty/degenerate output tensor. \
                 Check ONNX export compatibility."
                    .to_string(),
            ));
        }

        let num_words = text_words.len();
        let mut entities = Vec::with_capacity(32);

        // Span-level output: [batch, num_words, max_width, num_entity_types]
        if shape.len() == 4 && shape[0] == 1 && shape[1] > 0 && shape[2] > 1 {
            let out_num_words = shape[1] as usize;
            let out_max_width = shape[2] as usize;
            let num_classes = shape[3] as usize;

            if num_classes == 0 {
                return Err(Error::Inference(
                    "GLiNERPoly model produced num_classes=0.".to_string(),
                ));
            }

            for word_idx in 0..out_num_words.min(num_words) {
                for width in 0..out_max_width.min(MAX_SPAN_WIDTH) {
                    let end_word = word_idx + width;
                    if end_word >= num_words {
                        continue;
                    }

                    for class_idx in 0..num_classes.min(entity_types.len()) {
                        let idx = (word_idx * out_max_width * num_classes)
                            + (width * num_classes)
                            + class_idx;

                        if idx < output_data.len() {
                            let logit = output_data[idx];
                            let score = 1.0 / (1.0 + (-logit).exp());

                            if score >= threshold {
                                let (char_start, char_end) = Self::word_span_to_char_offsets(
                                    text, text_words, word_idx, end_word,
                                );

                                if char_start == 0 && char_end == 0 && word_idx > 0 {
                                    // Offset lookup failed; skip this span.
                                    continue;
                                }

                                let span_text: String = text
                                    .chars()
                                    .skip(char_start)
                                    .take(char_end.saturating_sub(char_start))
                                    .collect();

                                if span_text.is_empty() {
                                    continue;
                                }

                                let entity_type_str =
                                    entity_types.get(class_idx).unwrap_or(&"OTHER");
                                let entity_type =
                                    crate::schema::map_to_canonical(entity_type_str, None);

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
        }
        // BIO token-level output: [3, batch, num_words, num_entity_types]
        else if shape.len() == 4 && shape[0] == 3 && shape[1] == 1 {
            let out_num_words = shape[2] as usize;
            let num_classes = shape[3] as usize;
            let word_class_size = out_num_words * num_classes;

            let b_offset = 0_usize;
            let i_offset = word_class_size;

            for class_idx in 0..num_classes.min(entity_types.len()) {
                let mut current_start: Option<(usize, f32)> = None;

                for word_idx in 0..out_num_words.min(num_words) {
                    let b_idx = b_offset + word_idx * num_classes + class_idx;
                    let i_idx = i_offset + word_idx * num_classes + class_idx;

                    let b_logit = output_data.get(b_idx).copied().unwrap_or(-100.0);
                    let i_logit = output_data.get(i_idx).copied().unwrap_or(-100.0);

                    let b_score = 1.0 / (1.0 + (-b_logit).exp());
                    let i_score = 1.0 / (1.0 + (-i_logit).exp());

                    if b_score >= threshold {
                        // Emit any in-progress entity.
                        if let Some((start_word, avg_score)) = current_start.take() {
                            let end_word = word_idx.saturating_sub(1);
                            let label = entity_types.get(class_idx).unwrap_or(&"OTHER");
                            if let Some(e) = Self::build_bio_entity(
                                text, text_words, label, start_word, end_word, avg_score,
                            ) {
                                entities.push(e);
                            }
                        }
                        current_start = Some((word_idx, b_score));
                    } else if i_score >= threshold && current_start.is_some() {
                        if let Some((sw, score)) = current_start {
                            current_start = Some((sw, (score + i_score) / 2.0));
                        }
                    } else if let Some((start_word, avg_score)) = current_start.take() {
                        let end_word = word_idx.saturating_sub(1);
                        let label = entity_types.get(class_idx).unwrap_or(&"OTHER");
                        if let Some(e) = Self::build_bio_entity(
                            text, text_words, label, start_word, end_word, avg_score,
                        ) {
                            entities.push(e);
                        }
                    }
                }

                // Handle entity at end of text.
                if let Some((start_word, avg_score)) = current_start.take() {
                    let end_word = out_num_words.min(num_words).saturating_sub(1);
                    let label = entity_types.get(class_idx).unwrap_or(&"OTHER");
                    if let Some(e) = Self::build_bio_entity(
                        text, text_words, label, start_word, end_word, avg_score,
                    ) {
                        entities.push(e);
                    }
                }
            }
        } else {
            // TODO: Verify the actual poly-encoder output shape against the exported model.
            // If the shape doesn't match either known layout, return a descriptive error.
            return Err(Error::Inference(format!(
                "Unsupported GLiNERPoly output shape: {:?}. \
                 Expected [1,words,width,classes] (span) or [3,1,words,classes] (BIO).",
                shape
            )));
        }

        // Sort by position, deduplicate.
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
        entities.dedup_by(|a, b| a.start() == b.start() && a.end() == b.end());

        // Strip trailing/leading punctuation.
        let entities: Vec<Entity> = entities
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
                (e.start() < e.end()).then_some(e)
            })
            .collect();

        Ok(entities)
    }

    /// Build a BIO-decoded entity if the span is valid, or return `None`.
    #[allow(clippy::manual_map)] // clarity over brevity for offset validation
    fn build_bio_entity(
        text: &str,
        text_words: &[&str],
        entity_type_str: &str,
        start_word: usize,
        end_word: usize,
        score: f32,
    ) -> Option<Entity> {
        let num_words = text_words.len();
        if start_word > end_word || end_word >= num_words {
            return None;
        }

        let raw_span = text_words[start_word..=end_word].join(" ");
        let (start, end) = Self::word_span_to_char_offsets(text, text_words, start_word, end_word);

        if start == 0 && end == 0 && start_word > 0 {
            return None; // Offset lookup failed.
        }

        // Trim trailing punctuation that leaks from word-boundary tokenization.
        let (span_text, trimmed_chars) = textprep::spans::clean_span_tail(&raw_span);
        if span_text.is_empty() {
            return None;
        }
        let adj_end = end - trimmed_chars;

        let entity_type = crate::schema::map_to_canonical(entity_type_str, None);
        Some(Entity::new(
            span_text,
            entity_type,
            start,
            adj_end,
            score as f64,
        ))
    }

    /// Convert word indices to character offsets.
    ///
    /// Correctly handles Unicode text by converting byte offsets to character
    /// offsets using the offset module.
    pub(super) fn word_span_to_char_offsets(
        text: &str,
        words: &[&str],
        start_word: usize,
        end_word: usize,
    ) -> (usize, usize) {
        if words.is_empty()
            || start_word >= words.len()
            || end_word >= words.len()
            || start_word > end_word
        {
            return (0, 0);
        }

        let mut byte_pos = 0;
        let mut start_byte = 0;
        let mut end_byte = text.len();
        let mut found_start = false;
        let mut found_end = false;

        for (idx, word) in words.iter().enumerate() {
            if let Some(pos) = text.get(byte_pos..).and_then(|s| s.find(word)) {
                let abs_pos = byte_pos + pos;

                if idx == start_word {
                    start_byte = abs_pos;
                    found_start = true;
                }
                if idx == end_word {
                    end_byte = abs_pos + word.len();
                    found_end = true;
                    break;
                }

                byte_pos = abs_pos + word.len();
            }
        }

        if !found_start || !found_end {
            (0, 0)
        } else {
            crate::offset::bytes_to_chars(text, start_byte, end_byte)
        }
    }
}
