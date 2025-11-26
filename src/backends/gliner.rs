//! GLiNER-based NER implementation using ONNX Runtime.
//!
//! GLiNER (Generalist and Lightweight Model for Named Entity Recognition) is
//! state-of-the-art for zero-shot NER. This implementation follows gline-rs patterns.
//!
//! ## Prompt Format
//!
//! GLiNER uses a special prompt format:
//! ```text
//! [START] <<ENT>> type1 <<ENT>> type2 <<SEP>> word1 word2 ... [END]
//! ```
//!
//! Token IDs (for GLiNER tokenizer):
//! - START = 1
//! - END = 2
//! - <<ENT>> = 128002
//! - <<SEP>> = 128003
//!
//! ## Key Insight (from gline-rs)
//!
//! Each word is encoded SEPARATELY, preserving word boundaries.
//! Output shape: [batch, num_words, max_width, num_entity_types]

#[cfg(test)]
#[path = "ner_gliner/tests.rs"]
mod tests;

use crate::EntityType;
use crate::{{Error, Result}};
use crate::{Entity};

/// Special token IDs for GLiNER models
const TOKEN_START: u32 = 1;
const TOKEN_END: u32 = 2;
const TOKEN_ENT: u32 = 128002;
const TOKEN_SEP: u32 = 128003;

/// Default max span width from GLiNER config
const MAX_SPAN_WIDTH: usize = 12;

/// GLiNER model for zero-shot NER.
#[cfg(feature = "ml-ner-onnx")]
pub struct GLiNERNER {
    session: std::sync::Mutex<ort::session::Session>,
    tokenizer: tokenizers::Tokenizer,
    model_name: String,
}

#[cfg(feature = "ml-ner-onnx")]
impl GLiNERNER {
    /// Create a new GLiNER model from HuggingFace.
    pub fn new(model_name: &str) -> Result<Self> {
        use hf_hub::api::sync::Api;
        use ort::execution_providers::CPUExecutionProvider;
        use ort::session::Session;

        let api = Api::new()
            .map_err(|e| Error::Retrieval(format!("Failed to initialize HuggingFace API: {}", e)))?;

        let repo = api.model(model_name.to_string());

        // Download model and tokenizer
        let model_path = repo
            .get("onnx/model.onnx")
            .or_else(|_| repo.get("model.onnx"))
            .map_err(|e| Error::Retrieval(format!("Failed to download model.onnx: {}", e)))?;

        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| Error::Retrieval(format!("Failed to download tokenizer.json: {}", e)))?;

        let session = Session::builder()
            .map_err(|e| Error::Retrieval(format!("Failed to create ONNX session builder: {}", e)))?
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .map_err(|e| Error::Retrieval(format!("Failed to set execution providers: {}", e)))?
            .commit_from_file(&model_path)
            .map_err(|e| Error::Retrieval(format!("Failed to load ONNX model: {}", e)))?;

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Retrieval(format!("Failed to load tokenizer: {}", e)))?;

        log::debug!("[GLiNER] Model inputs: {:?}", session.inputs.iter().map(|i| &i.name).collect::<Vec<_>>());
        log::debug!("[GLiNER] Model outputs: {:?}", session.outputs.iter().map(|o| &o.name).collect::<Vec<_>>());

        Ok(Self {
            session: std::sync::Mutex::new(session),
            tokenizer,
            model_name: model_name.to_string(),
        })
    }

    /// Get model name.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Extract entities from text using GLiNER zero-shot NER.
    pub fn extract_entities(
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
        let (input_ids, attention_mask, words_mask, text_lengths, entity_count) = 
            self.encode_prompt(&text_words, entity_types)?;

        // Generate span tensors
        let (span_idx, span_mask) = self.make_span_tensors(num_text_words);

        // Build ort tensors
        use ndarray::{Array2, Array3};
        use ort::value::Tensor;

        let batch_size = 1;
        let seq_len = input_ids.len();
        let num_spans = num_text_words * MAX_SPAN_WIDTH;

        let input_ids_array = Array2::from_shape_vec((batch_size, seq_len), input_ids)
            .map_err(|e| Error::Parse(format!("Array error: {}", e)))?;
        let attention_mask_array = Array2::from_shape_vec((batch_size, seq_len), attention_mask)
            .map_err(|e| Error::Parse(format!("Array error: {}", e)))?;
        let words_mask_array = Array2::from_shape_vec((batch_size, seq_len), words_mask)
            .map_err(|e| Error::Parse(format!("Array error: {}", e)))?;
        let text_lengths_array = Array2::from_shape_vec((batch_size, 1), vec![num_text_words as i64])
            .map_err(|e| Error::Parse(format!("Array error: {}", e)))?;
        let span_idx_array = Array3::from_shape_vec((batch_size, num_spans, 2), span_idx)
            .map_err(|e| Error::Parse(format!("Array error: {}", e)))?;
        let span_mask_array = Array2::from_shape_vec((batch_size, num_spans), span_mask)
            .map_err(|e| Error::Parse(format!("Array error: {}", e)))?;

        let input_ids_t = Tensor::from_array(input_ids_array).map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;
        let attention_mask_t = Tensor::from_array(attention_mask_array).map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;
        let words_mask_t = Tensor::from_array(words_mask_array).map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;
        let text_lengths_t = Tensor::from_array(text_lengths_array).map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;
        let span_idx_t = Tensor::from_array(span_idx_array).map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;
        let span_mask_t = Tensor::from_array(span_mask_array).map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;

        // Run inference
        let mut session = self.session.lock()
            .map_err(|e| Error::Retrieval(format!("Failed to lock session: {}", e)))?;
        
        let outputs = session.run(ort::inputs![
            "input_ids" => input_ids_t.into_dyn(),
            "attention_mask" => attention_mask_t.into_dyn(),
            "words_mask" => words_mask_t.into_dyn(),
            "text_lengths" => text_lengths_t.into_dyn(),
            "span_idx" => span_idx_t.into_dyn(),
            "span_mask" => span_mask_t.into_dyn(),
        ]).map_err(|e| Error::Parse(format!("ONNX inference failed: {}", e)))?;

        // Decode output
        let entities = self.decode_output(&outputs, text, &text_words, entity_types, entity_count, threshold)?;
        drop(outputs);
        drop(session);
        
        Ok(entities)
    }

    /// Encode prompt following gline-rs pattern: word-by-word encoding.
    ///
    /// Structure: [START] <<ENT>> type1 <<ENT>> type2 <<SEP>> word1 word2 ... [END]
    fn encode_prompt(
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
            let encoding = self.tokenizer.encode(entity_type.to_string(), false)
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
            let encoding = self.tokenizer.encode(word.to_string(), false)
                .map_err(|e| Error::Parse(format!("Tokenizer error: {}", e)))?;
            
            word_id += 1;  // Increment before first token of word
            
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

        Ok((input_ids, attention_mask, word_mask, word_id, entity_types.len()))
    }

    /// Generate span tensors following gline-rs pattern.
    ///
    /// Shape: [num_words * max_width, 2] for span_idx
    /// Shape: [num_words * max_width] for span_mask
    fn make_span_tensors(&self, num_words: usize) -> (Vec<i64>, Vec<bool>) {
        let num_spans = num_words * MAX_SPAN_WIDTH;
        let mut span_idx: Vec<i64> = vec![0; num_spans * 2];
        let mut span_mask: Vec<bool> = vec![false; num_spans];

        for start in 0..num_words {
            let remaining_width = num_words - start;
            let actual_max_width = MAX_SPAN_WIDTH.min(remaining_width);

            for width in 0..actual_max_width {
                let dim = start * MAX_SPAN_WIDTH + width;
                span_idx[dim * 2] = start as i64;           // start offset
                span_idx[dim * 2 + 1] = (start + width) as i64;  // end offset
                span_mask[dim] = true;
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
        let output = outputs.iter().next()
            .map(|(_, v)| v)
            .ok_or_else(|| Error::Parse("No output from GLiNER model".to_string()))?;

        // Extract tensor data
        let (_, data_slice) = output.try_extract_tensor::<f32>()
            .map_err(|e| Error::Parse(format!("Failed to extract output tensor: {}", e)))?;
        let output_data: Vec<f32> = data_slice.to_vec();

        // Get output shape
        let shape: Vec<i64> = match output.dtype() {
            ort::value::ValueType::Tensor { shape, .. } => shape.iter().copied().collect(),
            _ => return Err(Error::Parse("Output is not a tensor".to_string())),
        };

        log::debug!("[GLiNER] Output shape: {:?}, data len: {}, expected classes: {}", 
                   shape, output_data.len(), expected_num_classes);

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

            log::debug!("[GLiNER] Decoding: num_words={}, max_width={}, num_classes={}", 
                       out_num_words, out_max_width, num_classes);

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
                                let span_words: Vec<&str> = text_words[word_idx..=end_word].to_vec();
                                let span_text = span_words.join(" ");
                                
                                let (char_start, char_end) = self.word_span_to_char_offsets(
                                    text, text_words, word_idx, end_word
                                );

                                let entity_type_str = entity_types.get(class_idx).unwrap_or(&"OTHER");
                                let entity_type = Self::map_entity_type(entity_type_str);

                                entities.push(Entity {
                                    text: span_text,
                                    entity_type,
                                    start: char_start,
                                    end: char_end,
                                    confidence: score as f64,
                                });
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
                            let span_words: Vec<&str> = text_words[word_idx..=end_word].to_vec();
                            let span_text = span_words.join(" ");
                            
                            let (char_start, char_end) = self.word_span_to_char_offsets(
                                text, text_words, word_idx, end_word
                            );

                            let entity_type_str = entity_types.get(class_idx).unwrap_or(&"OTHER");
                            let entity_type = Self::map_entity_type(entity_type_str);

                            entities.push(Entity {
                                text: span_text,
                                entity_type,
                                start: char_start,
                                end: char_end,
                                confidence: score as f64,
                            });
                        }
                    }
                }
            }
        }

        // Sort and deduplicate
        entities.sort_by(|a, b| {
            a.start.cmp(&b.start)
                .then_with(|| b.end.cmp(&a.end))
                .then_with(|| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal))
        });
        entities.dedup_by(|a, b| a.start == b.start && a.end == b.end);

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
    fn word_span_to_char_offsets(
        &self,
        text: &str,
        words: &[&str],
        start_word: usize,
        end_word: usize,
    ) -> (usize, usize) {
        let mut char_pos = 0;
        let mut start_char = 0;
        let mut end_char = text.len();

        for (idx, word) in words.iter().enumerate() {
            if let Some(pos) = text[char_pos..].find(word) {
                let word_start = char_pos + pos;
                let word_end = word_start + word.len();

                if idx == start_word {
                    start_char = word_start;
                }
                if idx == end_word {
                    end_char = word_end;
                    break;
                }
                char_pos = word_end;
            }
        }

        (start_char, end_char)
    }
}

// Stub when feature disabled
#[cfg(not(feature = "ml-ner-onnx"))]
pub struct GLiNERNER;

#[cfg(not(feature = "ml-ner-onnx"))]
impl GLiNERNER {
    pub fn new(_model_name: &str) -> Result<Self> {
        Err(Error::Parse("GLiNER requires 'ml-ner-onnx' feature".to_string()))
    }

    pub fn model_name(&self) -> &str {
        "gliner-not-enabled"
    }

    pub fn extract_entities(
        &self,
        _text: &str,
        _entity_types: &[&str],
        _threshold: f32,
    ) -> Result<Vec<Entity>> {
        Err(Error::Parse("GLiNER requires 'ml-ner-onnx' feature".to_string()))
    }
}
