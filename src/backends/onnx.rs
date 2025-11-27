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
//!
//! ## Performance
//!
//! - F1 Score: ~86% on CoNLL-03
//! - Speed: ~50ms/doc on CPU

#![allow(missing_docs)] // Stub implementation
#![allow(dead_code)] // Placeholder constants
#![allow(clippy::manual_strip)] // Complex BIO tag parsing

#[cfg(feature = "onnx")]
use crate::EntityType;
use crate::{Entity, Error, Result};

#[cfg(feature = "onnx")]
use {
    hf_hub::api::sync::Api,
    ndarray::Array2,
    ort::{session::builder::GraphOptimizationLevel, session::Session, value::Tensor},
    std::collections::HashMap,
    std::sync::Mutex,
    tokenizers::Tokenizer,
};

/// Default BERT NER ONNX model (properly exported, reliable).
pub const DEFAULT_BERT_NER_MODEL: &str = "protectai/bert-base-NER-onnx";

/// BERT-based NER using ONNX Runtime.
///
/// Uses standard BERT models fine-tuned for NER with BIO tagging scheme.
/// This is more reliable than GLiNER ONNX since the models are properly exported.
#[cfg(feature = "onnx")]
pub struct BertNEROnnx {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    id_to_label: HashMap<usize, String>,
    label_to_entity_type: HashMap<String, EntityType>,
    model_name: String,
}

#[cfg(feature = "onnx")]
impl BertNEROnnx {
    /// Create a new BERT NER ONNX model.
    ///
    /// # Arguments
    /// * `model_name` - HuggingFace model identifier (e.g., "protectai/bert-base-NER-onnx")
    ///
    /// # Returns
    /// BERT NER ONNX model instance
    pub fn new(model_name: &str) -> Result<Self> {
        let api = Api::new().map_err(|e| {
            Error::Retrieval(format!("Failed to initialize HuggingFace API: {}", e))
        })?;

        // Download model.onnx
        let model_path = api
            .model(model_name.to_string())
            .get("model.onnx")
            .or_else(|_| api.model(model_name.to_string()).get("onnx/model.onnx"))
            .map_err(|e| Error::Retrieval(format!("Failed to download model.onnx: {}", e)))?;

        // Download tokenizer.json
        let tokenizer_path = api
            .model(model_name.to_string())
            .get("tokenizer.json")
            .map_err(|e| Error::Retrieval(format!("Failed to download tokenizer.json: {}", e)))?;

        // Download config.json for label mapping
        let config_path = api
            .model(model_name.to_string())
            .get("config.json")
            .map_err(|e| Error::Retrieval(format!("Failed to download config.json: {}", e)))?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Retrieval(format!("Failed to load tokenizer: {}", e)))?;

        // Load config and extract id2label mapping
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| Error::Retrieval(format!("Failed to read config.json: {}", e)))?;
        let config_json: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| Error::Parse(format!("Failed to parse config.json: {}", e)))?;

        // Build label mappings
        let id_to_label = Self::build_id_to_label(&config_json);
        let label_to_entity_type = Self::build_label_to_entity_type();

        // Create ONNX session
        let session = Session::builder()
            .map_err(|e| Error::Retrieval(format!("Failed to create session builder: {}", e)))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| Error::Retrieval(format!("Failed to set optimization level: {}", e)))?
            .with_intra_threads(4)
            .map_err(|e| Error::Retrieval(format!("Failed to set threads: {}", e)))?
            .commit_from_file(&model_path)
            .map_err(|e| Error::Retrieval(format!("Failed to load ONNX model: {}", e)))?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            id_to_label,
            label_to_entity_type,
            model_name: model_name.to_string(),
        })
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
        map.insert("B-MISC".to_string(), EntityType::Other("misc".to_string()));
        map.insert("I-MISC".to_string(), EntityType::Other("misc".to_string()));
        // Alternative formats
        map.insert("PER".to_string(), EntityType::Person);
        map.insert("ORG".to_string(), EntityType::Organization);
        map.insert("LOC".to_string(), EntityType::Location);
        map.insert("MISC".to_string(), EntityType::Other("misc".to_string()));
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
    pub fn extract_entities(&self, text: &str, _language: Option<&str>) -> Result<Vec<Entity>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        // Tokenize input text
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| Error::Parse(format!("Failed to tokenize input: {}", e)))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&mask| mask as i64)
            .collect();
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

        let input_ids_tensor = Tensor::from_array(input_ids_array)
            .map_err(|e| Error::Parse(format!("Failed to create input_ids tensor: {}", e)))?;

        let attention_mask_tensor = Tensor::from_array(attention_mask_array)
            .map_err(|e| Error::Parse(format!("Failed to create attention_mask tensor: {}", e)))?;

        let token_type_ids_tensor = Tensor::from_array(token_type_ids_array)
            .map_err(|e| Error::Parse(format!("Failed to create token_type_ids tensor: {}", e)))?;

        // Run inference
        let mut session = self
            .session
            .lock()
            .map_err(|e| Error::Retrieval(format!("Failed to lock session: {}", e)))?;

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
        self.decode_output(logits, text, &encoding)
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

        // Helper to access logits[0, token_idx, label_idx] in flattened array
        let get_logit = |token_idx: usize, label_idx: usize| -> f32 {
            logits_data[token_idx * num_labels + label_idx]
        };

        // Decode BIO tags and extract entities
        let mut entities = Vec::new();
        let mut current_entity: Option<(usize, usize, EntityType, f64)> = None; // (start_char, end_char, type, confidence)

        for token_idx in 0..seq_len {
            // Skip special tokens (no offset)
            if token_idx >= offsets.len() {
                continue;
            }
            let (char_start, char_end) = offsets[token_idx];
            if char_start == char_end {
                // Special token, finalize current entity
                if let Some((start, end, entity_type, conf)) = current_entity.take() {
                    let entity_text: String = text.chars().skip(start).take(end - start).collect();
                    if !entity_text.trim().is_empty() {
                        entities.push(Entity::new(
                            entity_text.trim().to_string(),
                            entity_type,
                            start,
                            end,
                            conf,
                        ));
                    }
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
            let confidence = (1.0_f32 / exp_sum) as f64; // exp(0) / exp_sum = 1/exp_sum

            let label = self
                .id_to_label
                .get(&max_idx)
                .cloned()
                .unwrap_or_else(|| format!("LABEL_{}", max_idx));

            // Skip "O" (outside) labels
            if label == "O" {
                if let Some((start, end, entity_type, conf)) = current_entity.take() {
                    let entity_text: String = text.chars().skip(start).take(end - start).collect();
                    if !entity_text.trim().is_empty() {
                        entities.push(Entity::new(
                            entity_text.trim().to_string(),
                            entity_type,
                            start,
                            end,
                            conf,
                        ));
                    }
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
                .unwrap_or_else(|| EntityType::Other(entity_label.clone()));

            match bio {
                "B" => {
                    // Finalize previous entity
                    if let Some((start, end, prev_type, conf)) = current_entity.take() {
                        let entity_text: String =
                            text.chars().skip(start).take(end - start).collect();
                        if !entity_text.trim().is_empty() {
                            entities.push(Entity::new(
                                entity_text.trim().to_string(),
                                prev_type,
                                start,
                                end,
                                conf,
                            ));
                        }
                    }
                    // Start new entity
                    current_entity = Some((char_start, char_end, entity_type, confidence));
                }
                "I" => {
                    // Continue current entity if same type
                    if let Some((start, _end, ref prev_type, conf)) = current_entity {
                        if std::mem::discriminant(prev_type) == std::mem::discriminant(&entity_type)
                        {
                            current_entity = Some((start, char_end, entity_type, conf));
                        } else {
                            // Different type - finalize and start new
                            let entity_text: String =
                                text.chars().skip(start).take(_end - start).collect();
                            if !entity_text.trim().is_empty() {
                                entities.push(Entity::new(
                                    entity_text.trim().to_string(),
                                    prev_type.clone(),
                                    start,
                                    _end,
                                    conf,
                                ));
                            }
                            current_entity = Some((char_start, char_end, entity_type, confidence));
                        }
                    } else {
                        // No current entity, treat I- as B-
                        current_entity = Some((char_start, char_end, entity_type, confidence));
                    }
                }
                _ => {}
            }
        }

        // Finalize last entity
        if let Some((start, end, entity_type, conf)) = current_entity {
            let entity_text: String = text.chars().skip(start).take(end - start).collect();
            if !entity_text.trim().is_empty() {
                entities.push(Entity::new(
                    entity_text.trim().to_string(),
                    entity_type,
                    start,
                    end,
                    conf,
                ));
            }
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
    fn extract_entities(&self, text: &str, language: Option<&str>) -> Result<Vec<Entity>> {
        self.extract_entities(text, language)
    }

    fn supported_types(&self) -> Vec<EntityType> {
        vec![
            EntityType::Person,
            EntityType::Organization,
            EntityType::Location,
            EntityType::Other("MISC".to_string()),
        ]
    }

    fn is_available(&self) -> bool {
        true
    }
}

// Stub implementation when feature is disabled
#[cfg(not(feature = "onnx"))]
pub struct BertNEROnnx;

#[cfg(not(feature = "onnx"))]
impl BertNEROnnx {
    pub fn new(_model_name: &str) -> Result<Self> {
        Err(Error::Parse(
            "BERT NER ONNX support requires 'onnx' feature".to_string(),
        ))
    }

    pub fn extract_entities(&self, _text: &str, _language: Option<&str>) -> Result<Vec<Entity>> {
        Err(Error::Parse(
            "BERT NER ONNX support requires 'onnx' feature".to_string(),
        ))
    }

    pub fn model_name(&self) -> &str {
        "onnx-not-enabled"
    }
}

#[cfg(not(feature = "onnx"))]
impl crate::Model for BertNEROnnx {
    fn extract_entities(&self, _text: &str, _language: Option<&str>) -> Result<Vec<Entity>> {
        Err(Error::Parse(
            "BERT NER ONNX support requires 'onnx' feature".to_string(),
        ))
    }

    fn supported_types(&self) -> Vec<crate::EntityType> {
        vec![]
    }

    fn is_available(&self) -> bool {
        false
    }
}
