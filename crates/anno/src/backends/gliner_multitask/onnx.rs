#![allow(unused_imports)]
//! GLiNER multi-task ONNX backend — inference engine and struct definition.
//!
//! Requires `--features onnx`. Trait implementations live in `super` (mod.rs).

use super::schema::{
    ClassificationResult, ClassificationTask, EntityTask, ExtractedStructure, ExtractionResult,
    FieldType, LabelCache, StructureTask, StructureValue, TaskSchema, MAX_SPAN_WIDTH, TOKEN_END,
    TOKEN_ENT, TOKEN_SEP, TOKEN_START,
};
use super::{map_entity_type, word_span_to_char_offsets};
use crate::backends::inference::{ExtractionWithRelations, RelationExtractor, ZeroShotNER};
use crate::{Entity, EntityType, Error, Result};
use anno_core::EntityCategory;
use std::collections::HashMap;
#[cfg(feature = "onnx")]
use std::sync::Mutex;

// ONNX Backend
// =============================================================================

/// GLiNER multi-task ONNX implementation.
/// GLiNER multi-task ONNX implementation.
#[cfg(feature = "onnx")]
#[derive(Debug)]
pub struct GLiNERMultitaskOnnx {
    pub(super) session: Mutex<ort::session::Session>,
    tokenizer: tokenizers::Tokenizer,
    #[allow(dead_code)]
    model_name: String,
    #[allow(dead_code)]
    hidden_size: usize,
}

#[cfg(feature = "onnx")]
impl GLiNERMultitaskOnnx {
    /// Load model from HuggingFace Hub.
    pub fn from_pretrained(model_id: &str) -> Result<Self> {
        use crate::backends::hf_loader;

        let api = hf_loader::hf_api()?;
        let repo = api.model(model_id.to_string());

        let model_path = hf_loader::download_model_file(&repo, &["onnx/model.onnx", "model.onnx"])?;
        let tokenizer_path = hf_loader::download_model_file(&repo, &["tokenizer.json"])?;
        let config_path = hf_loader::download_model_file(&repo, &["config.json"])?;

        let tokenizer = hf_loader::load_tokenizer(&tokenizer_path)?;

        // Parse config
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| Error::Retrieval(format!("config read: {}", e)))?;
        let config: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| Error::Parse(format!("config parse: {}", e)))?;
        let hidden_size = config["hidden_size"].as_u64().unwrap_or(768) as usize;

        let session =
            hf_loader::create_onnx_session(&model_path, hf_loader::OnnxSessionConfig::default())?;

        log::info!(
            "[GLiNERMultitask-ONNX] Loaded {} (hidden={})",
            model_id,
            hidden_size
        );
        log::debug!("[GLiNERMultitask-ONNX] Model loaded");

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            model_name: model_id.to_string(),
            hidden_size,
        })
    }

    /// Extract entities, classifications, and structures according to schema.
    pub fn extract(&self, text: &str, schema: &TaskSchema) -> Result<ExtractionResult> {
        let mut result = ExtractionResult::default();

        // NER extraction
        if let Some(ref ent_task) = schema.entities {
            let labels: Vec<&str> = ent_task.types.iter().map(|s| s.as_str()).collect();
            let entities = self.extract_ner(text, &labels, 0.5)?;
            result.entities = entities;
        }

        // Classification
        for class_task in &schema.classifications {
            let labels: Vec<&str> = class_task.labels.iter().map(|s| s.as_str()).collect();
            let class_result = self.classify(text, &labels, class_task.multi_label)?;
            result
                .classifications
                .insert(class_task.name.clone(), class_result);
        }

        // Structure extraction
        for struct_task in &schema.structures {
            let structures = self.extract_structure(text, struct_task)?;
            result.structures.extend(structures);
        }

        Ok(result)
    }

    /// Extract named entities using GLiNER multi-task NER format.
    pub(super) fn extract_ner(
        &self,
        text: &str,
        entity_types: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        if text.is_empty() || entity_types.is_empty() {
            return Ok(Vec::new());
        }

        // Split into words
        let text_words: Vec<&str> = text.split_whitespace().collect();
        if text_words.is_empty() {
            return Ok(Vec::new());
        }

        // Encode following GLiNER multi-task format: [P] entities ([E]type1 [E]type2 ...) [SEP] text
        let (input_ids, attention_mask, words_mask) =
            self.encode_ner_prompt(&text_words, entity_types)?;

        // Build tensors - GLiNER multi-task ONNX model only needs 4 inputs:
        // input_ids, attention_mask, words_mask, text_lengths
        // (NOT span_idx/span_mask - those were for older model variants)
        use ndarray::Array2;

        let batch_size = 1;
        let seq_len = input_ids.len();

        let input_ids_arr = Array2::from_shape_vec((batch_size, seq_len), input_ids)
            .map_err(|e| Error::Parse(format!("Array: {}", e)))?;
        let attention_mask_arr = Array2::from_shape_vec((batch_size, seq_len), attention_mask)
            .map_err(|e| Error::Parse(format!("Array: {}", e)))?;
        let words_mask_arr = Array2::from_shape_vec((batch_size, seq_len), words_mask)
            .map_err(|e| Error::Parse(format!("Array: {}", e)))?;
        let text_lengths_arr =
            Array2::from_shape_vec((batch_size, 1), vec![text_words.len() as i64])
                .map_err(|e| Error::Parse(format!("Array: {}", e)))?;

        let input_ids_t = crate::backends::ort_compat::tensor_from_ndarray(input_ids_arr)
            .map_err(|e| Error::Parse(format!("Tensor: {}", e)))?;
        let attention_mask_t = crate::backends::ort_compat::tensor_from_ndarray(attention_mask_arr)
            .map_err(|e| Error::Parse(format!("Tensor: {}", e)))?;
        let words_mask_t = crate::backends::ort_compat::tensor_from_ndarray(words_mask_arr)
            .map_err(|e| Error::Parse(format!("Tensor: {}", e)))?;
        let text_lengths_t = crate::backends::ort_compat::tensor_from_ndarray(text_lengths_arr)
            .map_err(|e| Error::Parse(format!("Tensor: {}", e)))?;

        // Run inference with blocking lock for thread-safe parallel access
        let mut session = self.session.lock().unwrap_or_else(|e| e.into_inner());

        let outputs = session
            .run(ort::inputs![
                "input_ids" => input_ids_t.into_dyn(),
                "attention_mask" => attention_mask_t.into_dyn(),
                "words_mask" => words_mask_t.into_dyn(),
                "text_lengths" => text_lengths_t.into_dyn(),
            ])
            .map_err(|e| Error::Inference(format!("ONNX run: {}", e)))?;

        // Decode output
        self.decode_ner_output(&outputs, text, &text_words, entity_types, threshold)
    }

    /// Encode NER prompt: [START] [P] entities ([E]type1 ...) [SEP] word1 word2 ... [END]
    pub(super) fn encode_ner_prompt(
        &self,
        text_words: &[&str],
        entity_types: &[&str],
    ) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>)> {
        let mut input_ids: Vec<i64> = Vec::new();
        let mut word_mask: Vec<i64> = Vec::new();

        // Start token [CLS]
        input_ids.push(TOKEN_START as i64);
        word_mask.push(0);

        // Entity types: <<ENT>>type1 <<ENT>>type2 ...
        // Format for token-level GLiNER: [CLS] <<ENT>>type1 <<ENT>>type2 ... <<SEP>> text [SEP]
        for entity_type in entity_types {
            input_ids.push(TOKEN_ENT as i64);
            word_mask.push(0);

            let type_enc = self
                .tokenizer
                .encode(*entity_type, false)
                .map_err(|e| Error::Parse(format!("Tokenize: {}", e)))?;
            for token_id in type_enc.get_ids() {
                input_ids.push(*token_id as i64);
                word_mask.push(0);
            }
        }

        // [SEP] token
        input_ids.push(TOKEN_SEP as i64);
        word_mask.push(0);

        // Text words with word_mask tracking
        for (word_idx, word) in text_words.iter().enumerate() {
            let word_enc = self
                .tokenizer
                .encode(*word, false)
                .map_err(|e| Error::Parse(format!("Tokenize: {}", e)))?;

            let word_id = (word_idx + 1) as i64; // 1-indexed
            for (token_idx, token_id) in word_enc.get_ids().iter().enumerate() {
                input_ids.push(*token_id as i64);
                // First subword gets word ID, rest get 0
                word_mask.push(if token_idx == 0 { word_id } else { 0 });
            }
        }

        // End token
        input_ids.push(TOKEN_END as i64);
        word_mask.push(0);

        let seq_len = input_ids.len();
        let attention_mask: Vec<i64> = vec![1; seq_len];

        Ok((input_ids, attention_mask, word_mask))
    }

    /// Decode NER output.
    fn decode_ner_output(
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
            .ok_or_else(|| Error::Parse("No output".into()))?;

        let (_, data_slice) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Parse(format!("Extract tensor: {}", e)))?;
        let output_data: Vec<f32> = data_slice.to_vec();

        let shape: Vec<i64> = match output.dtype() {
            ort::value::ValueType::Tensor { shape, .. } => shape.iter().copied().collect(),
            _ => return Err(Error::Parse("Not a tensor".into())),
        };

        if output_data.is_empty() || shape.contains(&0) {
            return Err(Error::Inference(
                "GLiNER multi-task ONNX returned empty/degenerate output tensor. This usually indicates an incompatible ONNX export (shape mismatch or missing dynamic axes).".to_string(),
            ));
        }

        let mut entities = Vec::new();
        let num_words = text_words.len();

        // Token-level model: shape [position, batch, num_words, num_classes]
        // where position = 3 for BIO tagging (B=0, I=1, O=2)
        if shape.len() == 4 && shape[0] == 3 && shape[1] == 1 {
            let out_num_words = shape[2] as usize;
            let num_classes = shape[3] as usize;
            let word_class_size = out_num_words * num_classes;

            // BIO decoding: find B-type starts, extend with I-type
            // Output shape [BIO=3, batch=1, words, classes] flattened to [BIO * batch * words * classes]
            // BIO dimension: 0=Begin, 1=Inside, 2=Outside
            let b_offset = 0_usize; // Begin logits start at offset 0
            let i_offset = word_class_size; // Inside logits start after B (1 * word_class_size)

            #[allow(clippy::needless_range_loop)] // class_idx used for multiple array accesses
            for class_idx in 0..num_classes.min(entity_types.len()) {
                let mut current_start: Option<(usize, f32)> = None; // (word_idx, score)

                for word_idx in 0..out_num_words.min(num_words) {
                    // B logit at BIO dimension 0
                    let b_idx = b_offset + word_idx * num_classes + class_idx;
                    // I logit at BIO dimension 1
                    let i_idx = i_offset + word_idx * num_classes + class_idx;

                    let b_logit = if b_idx < output_data.len() {
                        output_data[b_idx]
                    } else {
                        -100.0
                    };
                    let i_logit = if i_idx < output_data.len() {
                        output_data[i_idx]
                    } else {
                        -100.0
                    };

                    let b_score = 1.0 / (1.0 + (-b_logit).exp());
                    let i_score = 1.0 / (1.0 + (-i_logit).exp());

                    if b_score >= threshold {
                        // End any existing entity
                        if let Some((start_word, avg_score)) = current_start.take() {
                            let end_word = word_idx - 1;
                            if start_word <= end_word && end_word < num_words {
                                let span_text = text_words[start_word..=end_word].join(" ");
                                let (start, end) = word_span_to_char_offsets(
                                    text, text_words, start_word, end_word,
                                );
                                let entity_type = map_entity_type(entity_types[class_idx]);
                                entities.push(Entity::new(
                                    span_text,
                                    entity_type,
                                    start,
                                    end,
                                    avg_score as f64,
                                ));
                            }
                        }
                        // Start new entity
                        current_start = Some((word_idx, b_score));
                    } else if i_score >= threshold && current_start.is_some() {
                        // Continue entity - update score
                        if let Some((start_word, score)) = current_start {
                            current_start = Some((start_word, (score + i_score) / 2.0));
                        }
                    } else if current_start.is_some() {
                        // End entity
                        if let Some((start_word, avg_score)) = current_start.take() {
                            let end_word = word_idx - 1;
                            if start_word <= end_word && end_word < num_words {
                                let span_text = text_words[start_word..=end_word].join(" ");
                                let (start, end) = word_span_to_char_offsets(
                                    text, text_words, start_word, end_word,
                                );
                                let entity_type = map_entity_type(entity_types[class_idx]);
                                entities.push(Entity::new(
                                    span_text,
                                    entity_type,
                                    start,
                                    end,
                                    avg_score as f64,
                                ));
                            }
                        }
                    }
                }

                // Handle entity at end of text
                if let Some((start_word, avg_score)) = current_start.take() {
                    let end_word = out_num_words.min(num_words) - 1;
                    if start_word <= end_word {
                        let span_text = text_words[start_word..=end_word].join(" ");
                        let (start, end) =
                            word_span_to_char_offsets(text, text_words, start_word, end_word);
                        let entity_type = map_entity_type(entity_types[class_idx]);
                        entities.push(Entity::new(
                            span_text,
                            entity_type,
                            start,
                            end,
                            avg_score as f64,
                        ));
                    }
                }
            }
        }
        // Span-level model: shape [batch, num_words, max_width, num_classes]
        else if shape.len() == 4 && shape[0] == 1 {
            let out_num_words = shape[1] as usize;
            let out_max_width = shape[2] as usize;
            let num_classes = shape[3] as usize;

            for word_idx in 0..out_num_words.min(num_words) {
                for width in 0..out_max_width.min(MAX_SPAN_WIDTH) {
                    let end_word = word_idx + width;
                    if end_word >= num_words {
                        continue;
                    }

                    #[allow(clippy::needless_range_loop)] // class_idx used for index math
                    for class_idx in 0..num_classes.min(entity_types.len()) {
                        let idx = (word_idx * out_max_width * num_classes)
                            + (width * num_classes)
                            + class_idx;

                        if idx < output_data.len() {
                            let logit = output_data[idx];
                            let score = 1.0 / (1.0 + (-logit).exp());

                            if score >= threshold {
                                let span_text = text_words[word_idx..=end_word].join(" ");
                                let (start, end) =
                                    word_span_to_char_offsets(text, text_words, word_idx, end_word);

                                let entity_type = map_entity_type(entity_types[class_idx]);

                                entities.push(Entity::new(
                                    span_text,
                                    entity_type,
                                    start,
                                    end,
                                    score as f64,
                                ));
                            }
                        }
                    }
                }
            }
        }

        // Deduplicate
        entities.sort_by(|a, b| {
            a.start()
                .cmp(&b.start())
                .then_with(|| b.end().cmp(&a.end()))
        });
        entities.dedup_by(|a, b| a.start() == b.start() && a.end() == b.end());

        // Trim trailing punctuation from entity spans (e.g., "Seattle." -> "Seattle")
        for entity in &mut entities {
            let (trimmed, chars_removed) = textprep::spans::clean_span_tail(&entity.text);
            if chars_removed > 0 {
                let new_end = entity.end().saturating_sub(chars_removed);
                let trimmed_text = trimmed.to_string();
                entity.set_end(new_end);
                entity.text = trimmed_text;
            }
        }

        Ok(entities)
    }

    /// Classify text.
    fn classify(
        &self,
        text: &str,
        labels: &[&str],
        multi_label: bool,
    ) -> Result<ClassificationResult> {
        if text.is_empty() || labels.is_empty() {
            return Ok(ClassificationResult::default());
        }

        // For classification, encode <<ENT>>label1 <<ENT>>label2 ... <<SEP>> text
        // Using same format as NER since this model uses shared token markers

        // Encode input
        let mut input_ids: Vec<i64> = Vec::new();

        input_ids.push(TOKEN_START as i64);

        // Labels: <<ENT>>label1 <<ENT>>label2 ...
        for label in labels {
            input_ids.push(TOKEN_ENT as i64);
            let label_enc = self
                .tokenizer
                .encode(*label, false)
                .map_err(|e| Error::Parse(format!("Tokenize: {}", e)))?;
            for id in label_enc.get_ids() {
                input_ids.push(*id as i64);
            }
        }

        input_ids.push(TOKEN_SEP as i64);

        // Text
        let text_enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| Error::Parse(format!("Tokenize: {}", e)))?;
        for id in text_enc.get_ids() {
            input_ids.push(*id as i64);
        }

        input_ids.push(TOKEN_END as i64);

        let seq_len = input_ids.len();
        let attention_mask: Vec<i64> = vec![1; seq_len];

        use ndarray::Array2;

        let input_arr = Array2::from_shape_vec((1, seq_len), input_ids)
            .map_err(|e| Error::Parse(format!("Array: {}", e)))?;
        let attn_arr = Array2::from_shape_vec((1, seq_len), attention_mask)
            .map_err(|e| Error::Parse(format!("Array: {}", e)))?;

        let input_t = crate::backends::ort_compat::tensor_from_ndarray(input_arr)
            .map_err(|e| Error::Parse(format!("Tensor: {}", e)))?;
        let attn_t = crate::backends::ort_compat::tensor_from_ndarray(attn_arr)
            .map_err(|e| Error::Parse(format!("Tensor: {}", e)))?;

        // For classification models, we typically need just input_ids and attention_mask
        // The model should output classification logits
        let mut session = self.session.lock().unwrap_or_else(|e| e.into_inner());

        // Try running with standard classification inputs
        let outputs = session
            .run(ort::inputs![
                "input_ids" => input_t.into_dyn(),
                "attention_mask" => attn_t.into_dyn(),
            ])
            .map_err(|e| Error::Inference(format!("ONNX run: {}", e)))?;

        // Decode classification output
        let output = outputs
            .iter()
            .next()
            .map(|(_, v)| v)
            .ok_or_else(|| Error::Parse("No output".into()))?;

        let (_, data_slice) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Parse(format!("Extract: {}", e)))?;
        let logits: Vec<f32> = data_slice.to_vec();

        // Apply softmax or sigmoid
        let probs = if multi_label {
            logits.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
        } else {
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
            let sum: f32 = exp_logits.iter().sum();
            // Handle division by zero: if sum is 0 (all logits are -inf), return uniform distribution
            if sum > 0.0 {
                exp_logits.iter().map(|&x| x / sum).collect::<Vec<_>>()
            } else if logits.is_empty() {
                // Edge case: empty logits, return empty probabilities
                vec![]
            } else {
                // All logits are -inf, return uniform distribution
                let uniform = 1.0 / logits.len() as f32;
                vec![uniform; logits.len()]
            }
        };

        let mut scores = HashMap::new();
        let mut selected_labels: Vec<String> = Vec::new();

        for (i, label) in labels.iter().enumerate() {
            let prob = probs.get(i).copied().unwrap_or(0.0);
            scores.insert((*label).to_string(), prob);

            if multi_label && prob > 0.5 {
                selected_labels.push((*label).to_string());
            }
        }

        if !multi_label {
            if let Some((idx, _)) = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                if let Some(label) = labels.get(idx) {
                    selected_labels.push((*label).to_string());
                }
            }
        }

        Ok(ClassificationResult {
            labels: selected_labels,
            scores,
        })
    }

    /// Extract hierarchical structures.
    fn extract_structure(
        &self,
        text: &str,
        task: &StructureTask,
    ) -> Result<Vec<ExtractedStructure>> {
        if text.is_empty() || task.fields.is_empty() {
            return Ok(Vec::new());
        }

        // For structure extraction, first predict count of instances
        // Then extract fields for each instance
        // For simplicity, we'll use NER-style extraction for each field

        let mut structures = Vec::new();

        // Extract each field as a span
        let field_names: Vec<&str> = task.fields.iter().map(|f| f.name.as_str()).collect();
        let field_entities = self.extract_ner(text, &field_names, 0.3)?;

        // Group by field type and build structure
        let mut structure = ExtractedStructure {
            structure_type: task.name.clone(),
            fields: HashMap::new(),
        };

        for field in &task.fields {
            let matching: Vec<_> = field_entities
                .iter()
                .filter(|e| e.entity_type.as_label().eq_ignore_ascii_case(&field.name))
                .collect();

            if !matching.is_empty() {
                let value = match field.field_type {
                    FieldType::List => {
                        let values: Vec<String> = matching.iter().map(|e| e.text.clone()).collect();
                        StructureValue::List(values)
                    }
                    FieldType::Choice => {
                        if let Some(ref choices) = field.choices {
                            let extracted = matching.first().map(|e| e.text.as_str()).unwrap_or("");
                            let best = choices
                                .iter()
                                .find(|c| extracted.to_lowercase().contains(&c.to_lowercase()))
                                .cloned()
                                .unwrap_or_else(|| extracted.to_string());
                            StructureValue::Single(best)
                        } else {
                            StructureValue::Single(
                                matching.first().map(|e| e.text.clone()).unwrap_or_default(),
                            )
                        }
                    }
                    FieldType::String => StructureValue::Single(
                        matching.first().map(|e| e.text.clone()).unwrap_or_default(),
                    ),
                };
                structure.fields.insert(field.name.clone(), value);
            }
        }

        if !structure.fields.is_empty() {
            structures.push(structure);
        }

        Ok(structures)
    }
}

// =============================================================================
