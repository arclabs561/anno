#![allow(unused_imports)]
//! GLiNER Candle inference engine.

use super::layers::*;
use super::*;

/// GLiNER inference engine backed by the Candle framework.
pub struct GLiNERCandle {
    /// Text encoder (BERT/ModernBERT/DeBERTa)
    encoder: CandleEncoder,
    /// Tokenizer
    tokenizer: Tokenizer,
    /// Span representation layer
    span_rep: SpanRepLayer,
    /// Label encoder
    label_encoder: LabelEncoder,
    /// Span-label matcher
    matcher: SpanLabelMatcher,
    /// Model name
    model_name: String,
    /// Hidden size
    hidden_size: usize,
    /// Device
    pub(super) device: Device,
}

#[cfg(feature = "candle")]
impl std::fmt::Debug for GLiNERCandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GLiNERCandle")
            .field("model_name", &self.model_name)
            .field("hidden_size", &self.hidden_size)
            .field("device", &format!("{:?}", self.device))
            .finish_non_exhaustive()
    }
}

/// Helper function to convert pytorch_model.bin to safetensors format
///
/// # Implementation Options
///
/// 1. **Python subprocess** (pragmatic): Calls Python's safetensors library
/// 2. **Pure Rust** (complex): Requires parsing PyTorch pickle format manually
///
/// PyTorch state dicts use Python pickle format with `torch._utils._rebuild_tensor_v2`
/// which requires parsing complex nested structures. The `tch` crate can load models
/// but doesn't provide direct state dict -> safetensors conversion.
#[cfg(feature = "candle")]
pub(crate) fn convert_pytorch_to_safetensors(pytorch_path: &Path) -> Result<PathBuf> {
    let cache_dir = pytorch_path
        .parent()
        .ok_or_else(|| Error::Retrieval("Invalid pytorch model path".to_string()))?;

    let safetensors_path = cache_dir.join("model_converted.safetensors");

    // Check if already converted
    if safetensors_path.exists() {
        log::debug!("Using cached safetensors conversion");
        return Ok(safetensors_path);
    }

    log::info!(
        "Converting PyTorch model to safetensors: {:?}",
        pytorch_path
    );

    // Find the conversion script (in scripts/ directory relative to crate root)
    let script_path = if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        Path::new(&manifest_dir).join("scripts/convert_pytorch_to_safetensors.py")
    } else {
        // Fallback: try to find script relative to current executable
        Path::new("scripts/convert_pytorch_to_safetensors.py").to_path_buf()
    };

    // Try uv run first (PEP 723 script with inline dependencies)
    let output = std::process::Command::new("uv")
        .arg("run")
        .arg("--script")
        .arg(&script_path)
        .arg(pytorch_path)
        .arg(&safetensors_path)
        .output()
        .or_else(|_| {
            // Fallback to python3 if uv is not available
            std::process::Command::new("python3")
                .arg(&script_path)
                .arg(pytorch_path)
                .arg(&safetensors_path)
                .output()
        })
        .map_err(|e| {
            Error::Retrieval(format!(
                "Failed to run conversion script (uv or python3 not found?): {}",
                e
            ))
        })?;

    if output.status.success() && safetensors_path.exists() {
        log::info!(
            "Successfully converted to safetensors: {:?}",
            safetensors_path
        );
        return Ok(safetensors_path);
    }

    // If Python conversion failed, provide helpful error
    let error_msg = String::from_utf8_lossy(&output.stderr);
    let stdout_msg = String::from_utf8_lossy(&output.stdout);

    Err(Error::Retrieval(format!(
        "PyTorch to safetensors conversion failed. \
         \
         Script: {:?} \
         Error: {} \
         Output: {} \
         \
         Recommended solutions (in order of preference): \
         1. Use GLiNEROnnx (ONNX backend) - works with all GLiNER models, no conversion needed \
         2. Use a model that already has safetensors format (e.g., knowledgator/modern-gliner-bi-large-v1.0) \
         3. Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh \
         4. Manual conversion: uv run --script scripts/convert_pytorch_to_safetensors.py \"{}\" \"{}\" \
         \
         Note: Pure Rust conversion would require parsing PyTorch pickle format (torch._utils._rebuild_tensor_v2) \
         which is complex. Python's torch.load handles this automatically.",
        script_path,
        error_msg,
        stdout_msg,
        pytorch_path.display(),
        safetensors_path.display()
    )))
}

#[cfg(feature = "candle")]
impl GLiNERCandle {
    /// Load GLiNER from HuggingFace.
    ///
    /// Automatically loads `.env` for HF_TOKEN if present.
    ///
    /// # Arguments
    /// * `model_id` - HuggingFace model ID (e.g., "urchade/gliner_small-v2.1")
    pub fn from_pretrained(model_id: &str) -> Result<Self> {
        let device = best_device()?;

        let api = crate::backends::hf_loader::hf_api()?;
        let repo = api.model(model_id.to_string());

        // Download files
        // Try knowledgator models first (they have safetensors + tokenizer.json)
        // knowledgator/modern-gliner-bi-large-v1.0 has safetensors available
        // Fall back to urchade models if needed
        let tokenizer_path = repo.get("tokenizer.json").map_err(|e| {
            Error::Retrieval(format!(
                "tokenizer.json not found. GLiNER Candle requires tokenizer.json. \
                 Try using knowledgator/modern-gliner-bi-large-v1.0 (has safetensors) \
                 or GLiNEROnnx instead. Original error: {}",
                e
            ))
        })?;
        // GLiNER Candle requires safetensors format
        // Most GLiNER models only have pytorch_model.bin, which Candle cannot load directly
        // Workaround: Try to convert pytorch_model.bin to safetensors on-the-fly
        let weights_path = repo
            .get("model.safetensors")
            .or_else(|_| repo.get("gliner_model.safetensors"))
            .or_else(|_| {
                // Workaround: Try to convert pytorch_model.bin to safetensors
                // Now that we have From<ApiError>, we can use ? directly
                let pytorch_path = repo.get("pytorch_model.bin")?;
                convert_pytorch_to_safetensors(&pytorch_path)
            })
            .map_err(|e| Error::Retrieval(format!(
                "safetensors weights not found and conversion failed. GLiNER Candle requires safetensors format. \
                 Most GLiNER models (urchade/, knowledgator/) only provide pytorch_model.bin. \
                 Attempted automatic conversion but it failed. \
                 Please use GLiNEROnnx (ONNX version) instead, which works with all GLiNER models. \
                 Original error: {}",
                e
            )))?;
        // GLiNER models use gliner_config.json instead of standard config.json
        let config_path = repo
            .get("config.json")
            .or_else(|_| repo.get("gliner_config.json"))
            .map_err(|e| {
                Error::Retrieval(format!(
                    "config (tried config.json and gliner_config.json): {}",
                    e
                ))
            })?;

        // Load tokenizer (only if tokenizer.json, not tokenizer_config.json)
        let tokenizer = if tokenizer_path.ends_with("tokenizer.json") {
            Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| Error::Retrieval(format!("tokenizer: {}", e)))?
        } else {
            return Err(Error::Retrieval(format!(
                "GLiNER Candle requires tokenizer.json, but only found {}. \
                 The model may not be in Candle-compatible format. \
                 Consider using GLiNEROnnx instead.",
                tokenizer_path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
            )));
        };

        // Parse config - GLiNER config has encoder_config nested inside
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| Error::Retrieval(format!("config: {}", e)))?;
        let config: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| Error::Parse(format!("config JSON: {}", e)))?;

        // GLiNER has encoder config nested inside encoder_config key
        let encoder_config_json = if config.get("encoder_config").is_some() {
            config["encoder_config"].clone()
        } else {
            // Fallback to top-level for non-GLiNER models
            config.clone()
        };

        let hidden_size = encoder_config_json["hidden_size"].as_u64().unwrap_or(768) as usize;

        // Load weights
        // SAFETY: VarBuilder::from_mmaped_safetensors uses unsafe internally for memory mapping.
        // The weights_path is validated to exist before this call, and the safetensors format
        // is validated by the library. This is a safe FFI boundary.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                .map_err(|e| Error::Retrieval(format!("safetensors: {}", e)))?
        };

        // Build encoder from the GLiNER-specific path
        // GLiNER stores BERT weights under token_rep_layer.bert_layer.model.*
        let bert_vb = vb.pp("token_rep_layer").pp("bert_layer").pp("model");

        // Build encoder config from the encoder_config section
        let encoder_config_str = serde_json::to_string(&encoder_config_json)
            .map_err(|e| Error::Parse(format!("encoder config JSON: {}", e)))?;
        let encoder_config = CandleEncoder::parse_config(&encoder_config_str)?;
        let encoder =
            CandleEncoder::from_vb(encoder_config, bert_vb, tokenizer.clone(), device.clone())?;

        // Build GLiNER-specific components
        // GLiNER uses span_rep_layer.span_rep_layer.* and prompt_rep_layer.* paths
        let span_rep = SpanRepLayer::new(
            hidden_size,
            MAX_SPAN_WIDTH,
            vb.pp("span_rep_layer").pp("span_rep_layer"),
        )?;
        let label_encoder = LabelEncoder::new(hidden_size, vb.pp("prompt_rep_layer"))?;
        let matcher = SpanLabelMatcher::new(1.0);

        log::info!(
            "[GLiNER-Candle] Loaded {} (hidden={}) on {:?}",
            model_id,
            hidden_size,
            device
        );

        Ok(Self {
            encoder,
            tokenizer,
            span_rep,
            label_encoder,
            matcher,
            model_name: model_id.to_string(),
            hidden_size,
            device,
        })
    }

    /// Simplified constructor that creates with random weights (for testing).
    pub fn new(model_name: &str) -> Result<Self> {
        Self::from_pretrained(model_name)
    }

    /// Extract entities with custom labels (zero-shot).
    ///
    /// # Arguments
    /// * `text` - Input text
    /// * `labels` - Entity types to detect (e.g., ["person", "organization"])
    /// * `threshold` - Confidence threshold (0.0-1.0)
    pub fn extract(&self, text: &str, labels: &[&str], threshold: f32) -> Result<Vec<Entity>> {
        if text.trim().is_empty() || labels.is_empty() {
            return Ok(vec![]);
        }

        // Tokenize text word-by-word (GLiNER pattern)
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Ok(vec![]);
        }

        // Build prompt: [START] <<ENT>> label1 <<ENT>> label2 <<SEP>> word1 word2 ... [END]
        let (text_embeddings, word_positions) = self.encode_text(text, &words)?;
        let label_embeddings = self.encode_labels(labels)?;

        // Generate span candidates
        let span_indices = self.generate_spans(words.len())?;

        // Compute span embeddings
        let span_embs = self.span_rep.forward(&text_embeddings, &span_indices)?;

        // Compute label embeddings
        let label_embs = self.label_encoder.forward(&label_embeddings)?;

        // Match spans to labels
        let scores = self.matcher.forward(&span_embs, &label_embs)?;

        // Debug: Log score statistics (only when debug logging is enabled)
        if log::log_enabled!(log::Level::Debug) {
            if let Ok(scores_vec) = scores.flatten_all()?.to_vec1::<f32>() {
                if !scores_vec.is_empty() {
                    let max_score = scores_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let min_score = scores_vec.iter().cloned().fold(f32::INFINITY, f32::min);
                    let mean_score: f32 = scores_vec.iter().sum::<f32>() / scores_vec.len() as f32;
                    log::debug!(
                        "[GLiNER-Candle] Score stats: min={:.4}, max={:.4}, mean={:.4}, threshold={:.4}, n={}",
                        min_score, max_score, mean_score, threshold, scores_vec.len()
                    );
                }
            }
        }

        // Decode to entities
        let entities =
            self.decode_entities(text, &words, &word_positions, &scores, labels, threshold)?;

        Ok(entities)
    }

    fn encode_text(&self, text: &str, words: &[&str]) -> Result<(Tensor, Vec<(usize, usize)>)> {
        // GLiNER span extraction operates over *word* indices. The encoder produces *token*
        // embeddings (wordpieces), so we must aggregate token embeddings into per-word embeddings.
        //
        // This fixes a major correctness issue where span indices (word-based) were being applied
        // to token embeddings (token-based), producing incorrect spans.

        let (token_embeddings, seq_len, token_offsets) = self.encoder.encode_with_offsets(text)?;
        if seq_len == 0 {
            return Ok((
                Tensor::zeros((1, 0, self.hidden_size), DType::F32, &self.device)
                    .map_err(|e| Error::Parse(format!("empty text tensor: {}", e)))?,
                vec![],
            ));
        }

        // Build word byte positions in the ORIGINAL text (not a re-joined version).
        // This preserves correct offsets even when the input contains multiple spaces/newlines.
        let word_positions: Vec<(usize, usize)> = {
            let mut positions = Vec::with_capacity(words.len());
            let mut byte_pos = 0usize;
            for (idx, word) in words.iter().enumerate() {
                if let Some(rel_pos) = text[byte_pos..].find(word) {
                    let start = byte_pos + rel_pos;
                    let end = start + word.len();
                    positions.push((start, end));
                    byte_pos = end;
                } else {
                    return Err(Error::Parse(format!(
                        "Word '{}' (index {}) not found in text starting at byte {}",
                        word, idx, byte_pos
                    )));
                }
            }
            positions
        };

        // Aggregate token embeddings into per-word embeddings by offset overlap.
        // token_embeddings: flattened [seq_len, hidden]
        let mut word_embeddings = Vec::with_capacity(words.len().saturating_mul(self.hidden_size));

        // Token offsets are in bytes (tokenizers crate). Special tokens often have (0, 0).
        let mut tok = 0usize;
        for &(w_start, w_end) in &word_positions {
            // Advance to first token that could overlap this word.
            while tok < seq_len && token_offsets[tok].1 <= w_start {
                tok += 1;
            }

            let mut acc = vec![0.0f32; self.hidden_size];
            let mut count = 0usize;

            let mut t = tok;
            while t < seq_len && token_offsets[t].0 < w_end {
                let (t_start, t_end) = token_offsets[t];
                // Skip special tokens / empty offsets.
                if t_end > t_start && t_start >= w_start && t_end <= w_end {
                    let base = t * self.hidden_size;
                    for h in 0..self.hidden_size {
                        acc[h] += token_embeddings[base + h];
                    }
                    count += 1;
                }
                t += 1;
            }

            // Keep tok monotonic for the next word to avoid quadratic behavior.
            tok = t;

            if count == 0 {
                // If we couldn't align any token to this word (can happen with truncation),
                // emit a zero vector rather than failing hard.
                log::debug!(
                    "[GLiNER-Candle] No tokens aligned to word span {}..{}, emitting zeros",
                    w_start,
                    w_end
                );
                word_embeddings.extend(std::iter::repeat_n(0.0f32, self.hidden_size));
            } else {
                let denom = count as f32;
                for val in acc.iter_mut().take(self.hidden_size) {
                    *val /= denom;
                }
                word_embeddings.extend(acc);
            }
        }

        // Reshape to [1, num_words, hidden]
        let tensor = Tensor::from_vec(
            word_embeddings,
            (1, words.len(), self.hidden_size),
            &self.device,
        )
        .map_err(|e| Error::Parse(format!("word text tensor: {}", e)))?;

        Ok((tensor, word_positions))
    }

    fn encode_labels(&self, labels: &[&str]) -> Result<Tensor> {
        // Encode each label
        // Performance: Pre-allocate all_embeddings with estimated capacity
        // Each label produces hidden_size embeddings
        let mut all_embeddings = Vec::with_capacity(labels.len().saturating_mul(self.hidden_size));

        for label in labels {
            let (embeddings, seq_len) = self.encoder.encode(label)?;
            // Average pool to get single embedding - handle empty sequences
            let avg: Vec<f32> = if seq_len == 0 {
                // Return zero vector for empty sequences
                vec![0.0f32; self.hidden_size]
            } else {
                (0..self.hidden_size)
                    .map(|i| {
                        embeddings
                            .iter()
                            .skip(i)
                            .step_by(self.hidden_size)
                            .take(seq_len)
                            .sum::<f32>()
                            / seq_len as f32
                    })
                    .collect()
            };
            all_embeddings.extend(avg);
        }

        Tensor::from_vec(
            all_embeddings,
            (labels.len(), self.hidden_size),
            &self.device,
        )
        .map_err(|e| Error::Parse(format!("label tensor: {}", e)))
    }

    fn generate_spans(&self, num_words: usize) -> Result<Tensor> {
        // Performance: Pre-allocate spans vec with estimated capacity
        // num_words * MAX_SPAN_WIDTH * 2 (for start/end pairs)
        let estimated_capacity = num_words.saturating_mul(MAX_SPAN_WIDTH).saturating_mul(2);
        let mut spans = Vec::with_capacity(estimated_capacity.min(1000));

        for start in 0..num_words {
            for width in 0..MAX_SPAN_WIDTH.min(num_words - start) {
                let end = start + width;
                spans.push(start as i64);
                spans.push(end as i64);
            }
        }

        let num_spans = spans.len() / 2;
        Tensor::from_vec(spans, (1, num_spans, 2), &self.device)
            .map_err(|e| Error::Parse(format!("span tensor: {}", e)))
    }

    fn decode_entities(
        &self,
        text: &str,
        words: &[&str],
        word_positions: &[(usize, usize)],
        scores: &Tensor,
        labels: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        // scores: [1, num_spans, num_labels]
        let scores_vec = scores
            .flatten_all()
            .map_err(|e| Error::Parse(format!("flatten scores: {}", e)))?
            .to_vec1::<f32>()
            .map_err(|e| Error::Parse(format!("scores to vec: {}", e)))?;

        let num_labels = labels.len();
        let num_spans = scores_vec.len() / num_labels;

        // Performance: Pre-allocate entities vec with estimated capacity
        let mut entities = Vec::with_capacity(num_spans.min(32));
        let mut span_idx = 0;
        // Word positions are byte offsets; `Entity` requires character offsets.
        let span_converter = crate::offset::SpanConverter::new(text);

        for start in 0..words.len() {
            for width in 0..MAX_SPAN_WIDTH.min(words.len() - start) {
                if span_idx >= num_spans {
                    break;
                }

                // Note: generate_spans uses end = start + width (inclusive end word index).
                // For word_positions indexing, we need the last word index (inclusive).
                // Since word_positions[i] corresponds to word i, we use end_inclusive directly.
                // Loop bounds ensure: width < words.len() - start, so end_inclusive < words.len().
                let end_inclusive = start + width; // Last word index (inclusive), matches generate_spans

                // Find best label for this span
                let base = span_idx * num_labels;
                let mut best_label = 0;
                let mut best_score = 0.0f32;

                for (label_idx, _) in labels.iter().enumerate() {
                    let score = scores_vec.get(base + label_idx).copied().unwrap_or(0.0);
                    if score > best_score {
                        best_score = score;
                        best_label = label_idx;
                    }
                }

                if best_score >= threshold {
                    // Validate bounds: end_inclusive must be < word_positions.len()
                    // (Loop bounds ensure this, but defensive check for safety)
                    if start < word_positions.len() && end_inclusive < word_positions.len() {
                        if let (Some(&(start_pos, _)), Some(&(_, end_pos))) =
                            (word_positions.get(start), word_positions.get(end_inclusive))
                        {
                            if let Some(entity_text) = text.get(start_pos..end_pos) {
                                let label = labels[best_label];
                                let entity_type = Self::map_label(label);
                                entities.push(Entity::new(
                                    entity_text,
                                    entity_type,
                                    span_converter.byte_to_char(start_pos),
                                    span_converter.byte_to_char_ceil(end_pos),
                                    best_score as f64,
                                ));
                            }
                        }
                    }
                }

                span_idx += 1;
            }
        }

        // Remove overlapping spans, keeping highest confidence
        crate::backends::chunking::deduplicate_overlapping(
            &mut entities,
            crate::backends::chunking::OverlapStrategy::KeepHighestConfidence,
        );
        Ok(entities)
    }

    pub(super) fn map_label(label: &str) -> EntityType {
        match label.to_lowercase().as_str() {
            "person" | "per" => EntityType::Person,
            "organization" | "org" | "company" => EntityType::Organization,
            "location" | "loc" | "place" | "gpe" => EntityType::Location,
            "date" => EntityType::Date,
            "time" => EntityType::Time,
            "money" | "currency" => EntityType::Money,
            "percent" | "percentage" => EntityType::Percent,
            other => EntityType::custom(other, anno_core::EntityCategory::Misc),
        }
    }

    /// Get device as a string.
    pub fn device(&self) -> String {
        match &self.device {
            Device::Cpu => "cpu".to_string(),
            Device::Metal(_) => "metal".to_string(),
            Device::Cuda(_) => "cuda".to_string(),
        }
    }

    /// Get model name.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

// =============================================================================
// Model Trait Implementation
// =============================================================================
