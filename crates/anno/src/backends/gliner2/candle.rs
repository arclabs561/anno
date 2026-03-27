#![allow(unused_imports)]
//! GLiNER2 Candle backend — inference engine and struct definition.
//!
//! Requires `--features candle`. Trait implementations live in `super` (mod.rs).

#[cfg(feature = "candle")]
use super::schema::MAX_COUNT;
use super::schema::{
    ClassificationResult, ClassificationTask, EntityTask, ExtractedStructure, ExtractionResult,
    FieldType, LabelCache, StructureTask, StructureValue, TaskSchema, MAX_SPAN_WIDTH,
};
use super::{map_entity_type, word_span_to_char_offsets};
use crate::backends::inference::{ExtractionWithRelations, RelationExtractor, ZeroShotNER};
#[cfg(feature = "candle")]
use std::sync::RwLock;
use crate::{Entity, EntityType, Error, Result};
use anno_core::EntityCategory;
use std::collections::HashMap;

// Candle Backend
// =============================================================================

#[cfg(feature = "candle")]
use crate::backends::encoder_candle::TextEncoder;
#[cfg(feature = "candle")]
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
#[cfg(feature = "candle")]
use candle_nn::{Linear, VarBuilder};

/// GLiNER2 Candle implementation.
#[cfg(feature = "candle")]
#[derive(Debug)]
pub struct GLiNER2Candle {
    /// Text encoder
    encoder: crate::backends::encoder_candle::CandleEncoder,
    /// Span representation layer
    span_rep: SpanRepLayer,
    /// Label projection
    label_proj: Linear,
    /// Classification head for [L] tokens
    class_head: ClassificationHead,
    /// Structure count predictor for [P] tokens
    count_predictor: CountPredictor,
    /// Device
    pub(super) device: Device,
    #[allow(dead_code)]
    model_name: String,
    hidden_size: usize,
    /// Label embedding cache
    label_cache: LabelCache,
}

/// Span representation layer (from GLiNER).
#[cfg(feature = "candle")]
pub struct SpanRepLayer {
    /// Width embeddings for spans of different sizes
    width_embeddings: candle_nn::Embedding,
    /// Max span width
    max_width: usize,
}

#[cfg(feature = "candle")]
impl std::fmt::Debug for SpanRepLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpanRepLayer")
            .field("max_width", &self.max_width)
            .finish()
    }
}

/// Classification head for text classification tasks.
#[cfg(feature = "candle")]
pub struct ClassificationHead {
    /// MLP that projects [L] token embeddings to logits
    mlp: Linear,
}

#[cfg(feature = "candle")]
impl std::fmt::Debug for ClassificationHead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClassificationHead").finish()
    }
}

/// Count predictor for hierarchical structure extraction.
#[cfg(feature = "candle")]
pub struct CountPredictor {
    /// MLP that predicts instance count (0-19)
    mlp: Linear,
}

#[cfg(feature = "candle")]
impl std::fmt::Debug for CountPredictor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CountPredictor").finish()
    }
}

#[cfg(feature = "candle")]
impl SpanRepLayer {
    fn new(hidden_size: usize, max_width: usize, vb: VarBuilder) -> Result<Self> {
        let width_embeddings =
            candle_nn::embedding(max_width, hidden_size, vb.pp("width_embeddings"))
                .map_err(|e| Error::Retrieval(format!("width_embeddings: {}", e)))?;
        Ok(Self {
            width_embeddings,
            max_width,
        })
    }

    fn forward(&self, token_embeddings: &Tensor, span_indices: &Tensor) -> Result<Tensor> {
        let device = token_embeddings.device();
        let batch_size = token_embeddings.dims()[0];
        let _seq_len = token_embeddings.dims()[1];
        let hidden_size = token_embeddings.dims()[2];
        let num_spans = span_indices.dims()[1];

        let mut all_span_embs = Vec::new();

        for b in 0..batch_size {
            let batch_tokens = token_embeddings
                .i(b)
                .map_err(|e| Error::Inference(format!("batch index: {}", e)))?;
            let batch_spans = span_indices
                .i(b)
                .map_err(|e| Error::Inference(format!("span index: {}", e)))?;

            let spans_data = batch_spans
                .to_vec2::<i64>()
                .map_err(|e| Error::Inference(format!("spans to vec: {}", e)))?;

            let mut span_embs = Vec::new();

            for span in spans_data {
                let start = span[0] as usize;
                let end = span[1] as usize;
                // Validate span: end must be > start to prevent underflow
                if end <= start {
                    log::warn!("Invalid span: end ({}) <= start ({})", end, start);
                    continue;
                }
                let width = end - start;

                // Get start token embedding
                let start_emb = batch_tokens
                    .i(start.min(batch_tokens.dims()[0] - 1))
                    .map_err(|e| Error::Inference(format!("start emb: {}", e)))?;

                // Get width embedding
                let width_idx = width.min(self.max_width - 1);
                let width_emb = self
                    .width_embeddings
                    .forward(
                        &Tensor::new(&[width_idx as u32], device)
                            .map_err(|e| Error::Inference(format!("width idx: {}", e)))?,
                    )
                    .map_err(|e| Error::Inference(format!("width emb: {}", e)))?
                    .squeeze(0)
                    .map_err(|e| Error::Inference(format!("squeeze: {}", e)))?;

                // Combine: start + width (could also use end and pool)
                let combined = start_emb
                    .add(&width_emb)
                    .map_err(|e| Error::Inference(format!("add: {}", e)))?;

                let emb_vec = combined
                    .to_vec1::<f32>()
                    .map_err(|e| Error::Inference(format!("to vec: {}", e)))?;
                span_embs.extend(emb_vec);
            }

            all_span_embs.extend(span_embs);
        }

        Tensor::from_vec(all_span_embs, (batch_size, num_spans, hidden_size), device)
            .map_err(|e| Error::Inference(format!("span tensor: {}", e)))
    }
}

#[cfg(feature = "candle")]
impl ClassificationHead {
    fn new(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let mlp = candle_nn::linear(hidden_size, 1, vb.pp("mlp"))
            .map_err(|e| Error::Retrieval(format!("classification mlp: {}", e)))?;
        Ok(Self { mlp })
    }

    /// Forward pass: project label embeddings to logits.
    fn forward(&self, label_embeddings: &Tensor) -> Result<Tensor> {
        self.mlp
            .forward(label_embeddings)
            .map_err(|e| Error::Inference(format!("class head forward: {}", e)))
    }
}

#[cfg(feature = "candle")]
impl CountPredictor {
    fn new(hidden_size: usize, max_count: usize, vb: VarBuilder) -> Result<Self> {
        let mlp = candle_nn::linear(hidden_size, max_count, vb.pp("mlp"))
            .map_err(|e| Error::Retrieval(format!("count mlp: {}", e)))?;
        Ok(Self { mlp })
    }

    /// Predict number of structure instances from [P] token embedding.
    fn forward(&self, prompt_embedding: &Tensor) -> Result<usize> {
        let logits = self
            .mlp
            .forward(prompt_embedding)
            .map_err(|e| Error::Inference(format!("count forward: {}", e)))?;

        // Argmax to get predicted count
        let logits_vec = logits
            .flatten_all()
            .map_err(|e| Error::Inference(format!("flatten: {}", e)))?
            .to_vec1::<f32>()
            .map_err(|e| Error::Inference(format!("to vec: {}", e)))?;

        let (max_idx, _) = logits_vec
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((1, &0.0));

        Ok(max_idx.max(1)) // At least 1 instance
    }
}

#[cfg(feature = "candle")]
impl GLiNER2Candle {
    /// Load model from HuggingFace Hub.
    pub fn from_pretrained(model_id: &str) -> Result<Self> {
        use crate::backends::encoder_candle::CandleEncoder;

        let api = crate::backends::hf_loader::hf_api()?;
        let repo = api.model(model_id.to_string());

        // Load config
        let config_path = repo
            .get("config.json")
            .map_err(|e| Error::Retrieval(format!("config.json: {}", e)))?;
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| Error::Retrieval(format!("read config: {}", e)))?;
        let config: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| Error::Parse(format!("parse config: {}", e)))?;
        let hidden_size = config["hidden_size"].as_u64().unwrap_or(768) as usize;

        // Determine device
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

        // Load weights - try safetensors first, then convert pytorch if needed
        let weights_path = repo
            .get("model.safetensors")
            .or_else(|_| repo.get("gliner_model.safetensors"))
            .or_else(|_| {
                // Try to convert pytorch_model.bin to safetensors
                let pytorch_path = repo.get("pytorch_model.bin")?;
                crate::backends::gliner_candle::convert_pytorch_to_safetensors(&pytorch_path)
            })
            .map_err(|e| {
                Error::Retrieval(format!("weights not found and conversion failed: {}", e))
            })?;

        // SAFETY: VarBuilder::from_mmaped_safetensors uses unsafe internally for memory mapping.
        // The weights_path is validated to exist before this call, and the safetensors format
        // is validated by the library. This is a safe FFI boundary.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                .map_err(|e| Error::Retrieval(format!("varbuilder: {}", e)))?
        };

        // Build components
        let encoder = CandleEncoder::from_pretrained(model_id)?;
        let span_rep = SpanRepLayer::new(hidden_size, MAX_SPAN_WIDTH, vb.pp("span_rep"))?;
        let label_proj = candle_nn::linear(hidden_size, hidden_size, vb.pp("label_projection"))
            .map_err(|e| Error::Retrieval(format!("label_projection: {}", e)))?;
        let class_head = ClassificationHead::new(hidden_size, vb.pp("classification"))?;
        let count_predictor =
            CountPredictor::new(hidden_size, MAX_COUNT, vb.pp("count_predictor"))?;

        log::info!(
            "[GLiNER2-Candle] Loaded {} (hidden={}) on {:?}",
            model_id,
            hidden_size,
            device
        );

        Ok(Self {
            encoder,
            span_rep,
            label_proj,
            class_head,
            count_predictor,
            device,
            model_name: model_id.to_string(),
            hidden_size,
            label_cache: LabelCache::new(),
        })
    }

    /// Extract entities, classifications, and structures according to schema.
    pub fn extract(&self, text: &str, schema: &TaskSchema) -> Result<ExtractionResult> {
        let mut result = ExtractionResult::default();

        // NER extraction
        if let Some(ref ent_task) = schema.entities {
            let entities = self.extract_entities(text, &ent_task.types, 0.5)?;
            result.entities = entities;
        }

        // Classification
        for class_task in &schema.classifications {
            let class_result = self.classify(text, &class_task.labels, class_task.multi_label)?;
            result
                .classifications
                .insert(class_task.name.clone(), class_result);
        }

        // Structure extraction with count prediction
        for struct_task in &schema.structures {
            let structures = self.extract_structure_with_count(text, struct_task)?;
            result.structures.extend(structures);
        }

        Ok(result)
    }

    /// Extract named entities with zero-shot labels.
    pub(super) fn extract_entities(
        &self,
        text: &str,
        types: &[String],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        if text.is_empty() || types.is_empty() {
            return Ok(Vec::new());
        }

        let labels: Vec<&str> = types.iter().map(|s| s.as_str()).collect();

        // Tokenize and get words
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Ok(Vec::new());
        }

        // Encode text
        let (text_embeddings, word_positions) = self.encode_text(&words)?;

        // Encode labels (with caching)
        let label_embeddings = self.encode_labels_cached(&labels)?;

        // Generate span candidates
        let span_indices = self.generate_spans(words.len())?;

        // Compute span embeddings
        let span_embs = self.span_rep.forward(&text_embeddings, &span_indices)?;

        // Project labels
        let label_embs = self
            .label_proj
            .forward(&label_embeddings)
            .map_err(|e| Error::Inference(format!("label projection: {}", e)))?;

        // Match spans to labels via cosine similarity
        let scores = self.match_spans_labels(&span_embs, &label_embs)?;

        // Decode to entities
        self.decode_entities(text, &words, &word_positions, &scores, &labels, threshold)
    }

    /// Classify text using the ClassificationHead.
    fn classify(
        &self,
        text: &str,
        labels: &[String],
        multi_label: bool,
    ) -> Result<ClassificationResult> {
        if text.is_empty() || labels.is_empty() {
            return Ok(ClassificationResult::default());
        }

        // Encode text and get [CLS] embedding
        let (text_emb, _seq_len) = self.encoder.encode(text)?;
        let cls_emb = Tensor::from_vec(
            text_emb[..self.hidden_size].to_vec(),
            (1, self.hidden_size),
            &self.device,
        )
        .map_err(|e| Error::Inference(format!("cls tensor: {}", e)))?;

        // Encode labels
        let labels_str: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
        let label_embs = self.encode_labels_cached(&labels_str)?;

        // Use classification head to get logits
        let label_logits = self.class_head.forward(&label_embs)?;
        let label_logits_vec = label_logits
            .flatten_all()
            .map_err(|e| Error::Inference(format!("flatten: {}", e)))?
            .to_vec1::<f32>()
            .map_err(|e| Error::Inference(format!("to vec: {}", e)))?;

        // Also compute similarity for ranking
        let cls_norm = l2_normalize(&cls_emb, D::Minus1)?;
        let label_norm = l2_normalize(&label_embs, D::Minus1)?;

        let sim_scores = cls_norm
            .matmul(
                &label_norm
                    .t()
                    .map_err(|e| Error::Inference(format!("transpose: {}", e)))?,
            )
            .map_err(|e| Error::Inference(format!("matmul: {}", e)))?;

        let sim_vec = sim_scores
            .flatten_all()
            .map_err(|e| Error::Inference(format!("flatten: {}", e)))?
            .to_vec1::<f32>()
            .map_err(|e| Error::Inference(format!("to vec: {}", e)))?;

        // Combine head logits with similarity (weighted)
        let combined: Vec<f32> = sim_vec
            .iter()
            .zip(label_logits_vec.iter().cycle())
            .map(|(s, l)| 0.7 * s + 0.3 * l)
            .collect();

        // Apply softmax (single-label) or sigmoid (multi-label)
        let probs = if multi_label {
            combined.iter().map(|&s| 1.0 / (1.0 + (-s).exp())).collect()
        } else {
            let max_score = combined.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Vec<f32> = combined.iter().map(|&s| (s - max_score).exp()).collect();
            let sum: f32 = exp_scores.iter().sum();
            // Handle division by zero: if sum is 0 (all logits are -inf), return uniform distribution
            if sum > 0.0 {
                exp_scores.iter().map(|&e| e / sum).collect::<Vec<_>>()
            } else if combined.is_empty() {
                // Edge case: empty scores, return empty probabilities
                vec![]
            } else {
                // All scores are -inf, return uniform distribution
                let uniform = 1.0 / combined.len() as f32;
                vec![uniform; combined.len()]
            }
        };

        let mut scores_map = HashMap::new();
        let mut result_labels = Vec::new();

        for (i, label) in labels.iter().enumerate() {
            let prob = probs.get(i).copied().unwrap_or(0.0);
            scores_map.insert(label.clone(), prob);

            if multi_label && prob > 0.5 {
                result_labels.push(label.clone());
            }
        }

        if !multi_label {
            if let Some((idx, _)) = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                if let Some(label) = labels.get(idx) {
                    result_labels.push(label.clone());
                }
            }
        }

        Ok(ClassificationResult {
            labels: result_labels,
            scores: scores_map,
        })
    }

    /// Extract hierarchical structures using count predictor.
    fn extract_structure_with_count(
        &self,
        text: &str,
        task: &StructureTask,
    ) -> Result<Vec<ExtractedStructure>> {
        if text.is_empty() || task.fields.is_empty() {
            return Ok(Vec::new());
        }

        // Encode text to get [P] token embedding for count prediction
        let (text_emb, _) = self.encoder.encode(text)?;
        let prompt_emb = Tensor::from_vec(
            text_emb[..self.hidden_size].to_vec(),
            (self.hidden_size,),
            &self.device,
        )
        .map_err(|e| Error::Inference(format!("prompt tensor: {}", e)))?;

        // Predict number of instances
        let num_instances = self.count_predictor.forward(&prompt_emb)?;

        log::debug!(
            "[GLiNER2] Count predictor: {} instances for {}",
            num_instances,
            task.name
        );

        let mut structures = Vec::new();

        // Extract fields for each predicted instance
        for instance_idx in 0..num_instances {
            let mut structure = ExtractedStructure {
                structure_type: task.name.clone(),
                fields: HashMap::new(),
            };

            for field in &task.fields {
                let field_label = field.description.as_ref().unwrap_or(&field.name);

                // Extract values for this field
                let labels_vec: Vec<String> = vec![field_label.to_string()];
                let entities = self.extract_entities(text, &labels_vec, 0.3)?;

                // For multi-instance, try to get the nth entity
                let entity_for_instance = entities.get(instance_idx);

                if let Some(entity) = entity_for_instance {
                    let value = match field.field_type {
                        FieldType::List => {
                            // For list type, get all matching entities
                            let values: Vec<String> =
                                entities.iter().map(|e| e.text.clone()).collect();
                            StructureValue::List(values)
                        }
                        FieldType::Choice => {
                            if let Some(ref choices) = field.choices {
                                let extracted = &entity.text;
                                let best_choice = choices
                                    .iter()
                                    .find(|c| extracted.to_lowercase().contains(&c.to_lowercase()))
                                    .cloned()
                                    .unwrap_or_else(|| extracted.clone());
                                StructureValue::Single(best_choice)
                            } else {
                                StructureValue::Single(entity.text.clone())
                            }
                        }
                        FieldType::String => StructureValue::Single(entity.text.clone()),
                    };

                    structure.fields.insert(field.name.clone(), value);
                }
            }

            if !structure.fields.is_empty() {
                structures.push(structure);
            }
        }

        Ok(structures)
    }

    // =========================================================================
    // Helper methods
    // =========================================================================

    fn encode_text(&self, words: &[&str]) -> Result<(Tensor, Vec<(usize, usize)>)> {
        let text = words.join(" ");
        let (embeddings, seq_len) = self.encoder.encode(&text)?;

        // Reshape to [1, seq_len, hidden]
        let tensor = Tensor::from_vec(embeddings, (1, seq_len, self.hidden_size), &self.device)
            .map_err(|e| Error::Inference(format!("text tensor: {}", e)))?;

        // Build word positions using character offsets
        let full_text = words.join(" ");
        let word_positions: Vec<(usize, usize)> = {
            let mut positions = Vec::new();
            let mut pos = 0;
            for (idx, word) in words.iter().enumerate() {
                if let Some(start) = full_text[pos..].find(word) {
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
                    return Err(Error::Inference(format!(
                        "Word '{}' (index {}) not found in text starting at position {}",
                        word, idx, pos
                    )));
                }
            }
            positions
        };

        // Validate that we found positions for all words
        if word_positions.len() != words.len() {
            return Err(Error::Inference(format!(
                "Word position mismatch: found {} positions for {} words",
                word_positions.len(),
                words.len()
            )));
        }

        Ok((tensor, word_positions))
    }

    pub(super) fn encode_labels_cached(&self, labels: &[&str]) -> Result<Tensor> {
        let mut all_embeddings = Vec::new();

        for label in labels {
            // Check cache first
            if let Some(cached) = self.label_cache.get(label) {
                all_embeddings.extend(cached);
            } else {
                let (embeddings, seq_len) = self.encoder.encode(label)?;
                // Average pool - handle empty sequences
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

                // Cache it
                self.label_cache.insert(label.to_string(), avg.clone());
                all_embeddings.extend(avg);
            }
        }

        Tensor::from_vec(
            all_embeddings,
            (labels.len(), self.hidden_size),
            &self.device,
        )
        .map_err(|e| Error::Inference(format!("label tensor: {}", e)))
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
            .map_err(|e| Error::Inference(format!("span tensor: {}", e)))
    }

    fn match_spans_labels(&self, span_embs: &Tensor, label_embs: &Tensor) -> Result<Tensor> {
        let span_norm = l2_normalize(span_embs, D::Minus1)?;
        let label_norm = l2_normalize(label_embs, D::Minus1)?;

        let batch_size = span_norm.dims()[0];
        let label_t = label_norm
            .t()
            .map_err(|e| Error::Inference(format!("transpose: {}", e)))?;
        let label_t = label_t
            .unsqueeze(0)
            .map_err(|e| Error::Inference(format!("unsqueeze: {}", e)))?
            .broadcast_as((batch_size, label_t.dims()[0], label_t.dims()[1]))
            .map_err(|e| Error::Inference(format!("broadcast: {}", e)))?;

        let scores = span_norm
            .matmul(&label_t)
            .map_err(|e| Error::Inference(format!("matmul: {}", e)))?;

        candle_nn::ops::sigmoid(&scores).map_err(|e| Error::Inference(format!("sigmoid: {}", e)))
    }

    fn decode_entities(
        &self,
        text: &str,
        words: &[&str],
        _word_positions: &[(usize, usize)],
        scores: &Tensor,
        labels: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        let scores_vec = scores
            .flatten_all()
            .map_err(|e| Error::Inference(format!("flatten scores: {}", e)))?
            .to_vec1::<f32>()
            .map_err(|e| Error::Inference(format!("scores to vec: {}", e)))?;

        let num_labels = labels.len();
        let num_spans = scores_vec.len() / num_labels;

        // Performance: Pre-allocate entities vec with estimated capacity
        let mut entities = Vec::with_capacity(num_spans.min(32));
        let mut span_idx = 0;

        for start in 0..words.len() {
            for width in 0..MAX_SPAN_WIDTH.min(words.len() - start) {
                if span_idx >= num_spans {
                    break;
                }

                let end = start + width;

                for (label_idx, label) in labels.iter().enumerate() {
                    let score = scores_vec[span_idx * num_labels + label_idx];

                    if score >= threshold {
                        let span_text = words[start..=end].join(" ");
                        let (char_start, char_end) =
                            word_span_to_char_offsets(text, words, start, end);

                        let entity_type = map_entity_type(label);

                        entities.push(Entity::new(
                            span_text,
                            entity_type,
                            char_start,
                            char_end,
                            score as f64,
                        ));
                    }
                }

                span_idx += 1;
            }
        }

        // Deduplicate
        entities.sort_by(|a, b| {
            a.start()
                .cmp(&b.start())
                .then_with(|| b.end().cmp(&a.end()))
        });
        entities.dedup_by(|a, b| a.start() == b.start() && a.end() == b.end());

        Ok(entities)
    }
}

/// L2 normalize tensor along dimension.
#[cfg(feature = "candle")]
fn l2_normalize(tensor: &Tensor, dim: D) -> Result<Tensor> {
    let norm = tensor
        .sqr()
        .map_err(|e| Error::Inference(format!("sqr: {}", e)))?
        .sum(dim)
        .map_err(|e| Error::Inference(format!("sum: {}", e)))?
        .sqrt()
        .map_err(|e| Error::Inference(format!("sqrt: {}", e)))?
        .unsqueeze(D::Minus1)
        .map_err(|e| Error::Inference(format!("unsqueeze: {}", e)))?;

    let norm_clamped = norm
        .clamp(1e-12, f32::MAX)
        .map_err(|e| Error::Inference(format!("clamp: {}", e)))?;

    tensor
        .broadcast_div(&norm_clamped)
        .map_err(|e| Error::Inference(format!("div: {}", e)))
}

// =============================================================================
