//! NuNER - Token-based zero-shot NER from NuMind.
//!
//! NuNER is a family of zero-shot NER models built on the GLiNER architecture
//! with a token classifier design (vs span classifier). Key advantages:
//!
//! - **Arbitrary-length entities**: No hard limit on entity span length
//! - **Efficient training**: Trained on NuNER v2.0 dataset (Pile + C4)
//! - **MIT Licensed**: Open weights from NuMind
//!
//! # Architecture
//!
//! NuNER uses the same bi-encoder architecture as GLiNER but with token classification:
//!
//! ```text
//! Input: "James Bond works at MI6"
//!        Labels: ["person", "organization"]
//!
//!        ┌──────────────────────┐
//!        │   Shared Encoder     │
//!        │  (DeBERTa/BERT)      │
//!        └──────────────────────┘
//!               │         │
//!        ┌──────┴──┐   ┌──┴─────┐
//!        │  Token  │   │ Label  │
//!        │  Embeds │   │ Embeds │
//!        └─────────┘   └────────┘
//!               │         │
//!        ┌──────┴─────────┴──────┐
//!        │   Token Classification │  (BIO tags per token)
//!        └───────────────────────┘
//!               │
//!               ▼
//!        B-PER I-PER  O    O   B-ORG
//!        James Bond works at  MI6
//! ```
//!
//! # Differences from GLiNER (Span Mode)
//!
//! | Aspect | GLiNER (Span) | NuNER (Token) |
//! |--------|---------------|---------------|
//! | Output | Span classification | Token classification (BIO) |
//! | Entity length | Limited by span window (12) | Arbitrary |
//! | ONNX inputs | 6 tensors (incl span_idx) | 4 tensors (no span tensors) |
//! | Decoding | Span scores → entities | BIO tags → entities |
//!
//! # Model Variants
//!
//! | Model | Context | Notes |
//! |-------|---------|-------|
//! | `numind/NuNER_Zero` | 512 | General zero-shot |
//! | `numind/NuNER_Zero_4k` | 4096 | Long context variant |
//! | `deepanwa/NuNerZero_onnx` | 512 | Pre-converted ONNX |
//!
//! # Usage
//!
//! ```rust,ignore
//! use anno::NuNER;
//!
//! // Load NuNER model (requires `onnx` feature)
//! let ner = NuNER::from_pretrained("deepanwa/NuNerZero_onnx")?;
//!
//! // Zero-shot extraction with custom labels
//! let entities = ner.extract("Apple CEO Tim Cook announced...", 
//!                            &["person", "organization", "product"], 0.5)?;
//! ```
//!
//! # References
//!
//! - [NuNER Zero on HuggingFace](https://huggingface.co/numind/NuNER_Zero)
//! - [NuNER ONNX](https://huggingface.co/deepanwa/NuNerZero_onnx)
//! - [gline-rs](https://github.com/fbilhaut/gline-rs) - Rust GLiNER inference

use crate::{Entity, EntityType, Model, Result};

#[cfg(feature = "onnx")]
use crate::Error;

/// Special token IDs for GLiNER/NuNER models (shared architecture)
#[cfg(feature = "onnx")]
const TOKEN_START: u32 = 1;
#[cfg(feature = "onnx")]
const TOKEN_END: u32 = 2;
#[cfg(feature = "onnx")]
const TOKEN_ENT: u32 = 128002;
#[cfg(feature = "onnx")]
const TOKEN_SEP: u32 = 128003;

/// NuNER Zero-shot NER model.
///
/// Token-based variant of GLiNER that uses BIO tagging instead of span classification.
/// This enables arbitrary-length entity extraction without the span window limitation.
///
/// # Feature Requirements
///
/// Requires the `onnx` feature for actual inference. Without it, configuration
/// methods work but extraction returns empty results.
///
/// # Example
///
/// ```rust,ignore
/// use anno::NuNER;
///
/// let ner = NuNER::from_pretrained("deepanwa/NuNerZero_onnx")?;
/// let entities = ner.extract(
///     "The CRISPR-Cas9 system was developed by Jennifer Doudna",
///     &["technology", "scientist"],
///     0.5
/// )?;
/// ```
pub struct NuNER {
    /// Model path or identifier
    model_id: String,
    /// Confidence threshold (0.0-1.0)
    threshold: f64,
    /// Default entity labels for Model trait
    default_labels: Vec<String>,
    /// ONNX session (when feature enabled)
    #[cfg(feature = "onnx")]
    session: Option<std::sync::Mutex<ort::session::Session>>,
    /// Tokenizer (when feature enabled)
    #[cfg(feature = "onnx")]
    tokenizer: Option<tokenizers::Tokenizer>,
}

impl NuNER {
    /// Create NuNER with default configuration.
    ///
    /// Uses standard NER labels. Call [`from_pretrained`](Self::from_pretrained)
    /// to load actual model weights.
    #[must_use]
    pub fn new() -> Self {
        Self {
            model_id: "numind/NuNER_Zero".to_string(),
            threshold: 0.5,
            default_labels: vec![
                "person".to_string(),
                "organization".to_string(),
                "location".to_string(),
                "date".to_string(),
                "product".to_string(),
                "event".to_string(),
            ],
            #[cfg(feature = "onnx")]
            session: None,
            #[cfg(feature = "onnx")]
            tokenizer: None,
        }
    }

    /// Load NuNER model from HuggingFace.
    ///
    /// # Arguments
    /// * `model_id` - HuggingFace model ID (e.g., "deepanwa/NuNerZero_onnx")
    ///
    /// # Example
    /// ```rust,ignore
    /// let ner = NuNER::from_pretrained("deepanwa/NuNerZero_onnx")?;
    /// ```
    #[cfg(feature = "onnx")]
    pub fn from_pretrained(model_id: &str) -> Result<Self> {
        use hf_hub::api::sync::Api;
        use ort::execution_providers::CPUExecutionProvider;
        use ort::session::Session;

        let api = Api::new().map_err(|e| {
            Error::Retrieval(format!("Failed to initialize HuggingFace API: {}", e))
        })?;

        let repo = api.model(model_id.to_string());

        // Download model and tokenizer
        let model_path = repo
            .get("onnx/model.onnx")
            .or_else(|_| repo.get("model.onnx"))
            .map_err(|e| Error::Retrieval(format!("Failed to download model.onnx: {}", e)))?;

        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| Error::Retrieval(format!("Failed to download tokenizer.json: {}", e)))?;

        let session = Session::builder()
            .map_err(|e| Error::Retrieval(format!("Failed to create ONNX session: {}", e)))?
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .map_err(|e| Error::Retrieval(format!("Failed to set execution providers: {}", e)))?
            .commit_from_file(&model_path)
            .map_err(|e| Error::Retrieval(format!("Failed to load ONNX model: {}", e)))?;

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Retrieval(format!("Failed to load tokenizer: {}", e)))?;

        log::debug!(
            "[NuNER] Loaded model: {} with inputs: {:?}",
            model_id,
            session.inputs.iter().map(|i| &i.name).collect::<Vec<_>>()
        );

        Ok(Self {
            model_id: model_id.to_string(),
            threshold: 0.5,
            default_labels: vec![
                "person".to_string(),
                "organization".to_string(),
                "location".to_string(),
            ],
            session: Some(std::sync::Mutex::new(session)),
            tokenizer: Some(tokenizer),
        })
    }

    /// Create with custom model identifier (for configuration only).
    #[must_use]
    pub fn with_model(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            ..Self::new()
        }
    }

    /// Set confidence threshold.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set default entity labels for Model trait.
    #[must_use]
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        self.default_labels = labels;
        self
    }

    /// Get the model identifier.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Get the confidence threshold.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Extract entities with custom labels.
    ///
    /// Unlike the `Model` trait which uses default labels, this method
    /// allows specifying arbitrary entity types at runtime.
    ///
    /// # Arguments
    /// * `text` - Text to extract from
    /// * `entity_types` - Entity type labels (e.g., ["person", "company"])
    /// * `threshold` - Confidence threshold (0.0-1.0)
    #[cfg(feature = "onnx")]
    pub fn extract(
        &self,
        text: &str,
        entity_types: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        if text.is_empty() || entity_types.is_empty() {
            return Ok(vec![]);
        }

        let session = self.session.as_ref().ok_or_else(|| {
            Error::Retrieval("Model not loaded. Call from_pretrained() first.".to_string())
        })?;

        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            Error::Retrieval("Tokenizer not loaded.".to_string())
        })?;

        // Split text into words
        let text_words: Vec<&str> = text.split_whitespace().collect();
        if text_words.is_empty() {
            return Ok(vec![]);
        }

        // Encode input (token mode - no span tensors)
        let (input_ids, attention_mask, words_mask, text_lengths) =
            self.encode_prompt(tokenizer, &text_words, entity_types)?;

        // Build ONNX tensors
        use ndarray::Array2;
        use ort::value::Tensor;

        let batch_size = 1;
        let seq_len = input_ids.len();

        let input_ids_array = Array2::from_shape_vec((batch_size, seq_len), input_ids)
            .map_err(|e| Error::Parse(format!("Array error: {}", e)))?;
        let attention_mask_array = Array2::from_shape_vec((batch_size, seq_len), attention_mask)
            .map_err(|e| Error::Parse(format!("Array error: {}", e)))?;
        let words_mask_array = Array2::from_shape_vec((batch_size, seq_len), words_mask)
            .map_err(|e| Error::Parse(format!("Array error: {}", e)))?;
        let text_lengths_array = Array2::from_shape_vec((batch_size, 1), vec![text_lengths])
            .map_err(|e| Error::Parse(format!("Array error: {}", e)))?;

        let input_ids_t = Tensor::from_array(input_ids_array)
            .map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;
        let attention_mask_t = Tensor::from_array(attention_mask_array)
            .map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;
        let words_mask_t = Tensor::from_array(words_mask_array)
            .map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;
        let text_lengths_t = Tensor::from_array(text_lengths_array)
            .map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;

        // Run inference (token mode - only 4 inputs)
        let mut session_guard = session
            .lock()
            .map_err(|e| Error::Retrieval(format!("Failed to lock session: {}", e)))?;

        let outputs = session_guard
            .run(ort::inputs![
                "input_ids" => input_ids_t.into_dyn(),
                "attention_mask" => attention_mask_t.into_dyn(),
                "words_mask" => words_mask_t.into_dyn(),
                "text_lengths" => text_lengths_t.into_dyn(),
            ])
            .map_err(|e| Error::Parse(format!("ONNX inference failed: {}", e)))?;

        // Decode BIO output to entities
        let entities = self.decode_token_output(
            &outputs,
            text,
            &text_words,
            entity_types,
            threshold,
        )?;

        Ok(entities)
    }

    /// Encode prompt for token mode (no span tensors).
    #[cfg(feature = "onnx")]
    fn encode_prompt(
        &self,
        tokenizer: &tokenizers::Tokenizer,
        text_words: &[&str],
        entity_types: &[&str],
    ) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>, i64)> {
        let mut input_ids: Vec<i64> = Vec::new();
        let mut word_mask: Vec<i64> = Vec::new();

        // [START]
        input_ids.push(TOKEN_START as i64);
        word_mask.push(0);

        // <<ENT>> type1 <<ENT>> type2 ...
        for entity_type in entity_types {
            input_ids.push(TOKEN_ENT as i64);
            word_mask.push(0);

            let encoding = tokenizer
                .encode(entity_type.to_string(), false)
                .map_err(|e| Error::Parse(format!("Tokenizer error: {}", e)))?;
            for token_id in encoding.get_ids() {
                input_ids.push(*token_id as i64);
                word_mask.push(0);
            }
        }

        // <<SEP>>
        input_ids.push(TOKEN_SEP as i64);
        word_mask.push(0);

        // Text words (word_mask starts from 1)
        let mut word_id: i64 = 0;
        for word in text_words {
            let encoding = tokenizer
                .encode(word.to_string(), false)
                .map_err(|e| Error::Parse(format!("Tokenizer error: {}", e)))?;

            word_id += 1;
            for (token_idx, token_id) in encoding.get_ids().iter().enumerate() {
                input_ids.push(*token_id as i64);
                word_mask.push(if token_idx == 0 { word_id } else { 0 });
            }
        }

        // [END]
        input_ids.push(TOKEN_END as i64);
        word_mask.push(0);

        let seq_len = input_ids.len();
        let attention_mask: Vec<i64> = vec![1; seq_len];

        Ok((input_ids, attention_mask, word_mask, word_id))
    }

    /// Decode token classification output to entities.
    ///
    /// Token mode output shape: [batch, seq_len, num_entity_types]
    /// Each position has scores for each entity type (BIO-style).
    #[cfg(feature = "onnx")]
    fn decode_token_output(
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
            .ok_or_else(|| Error::Parse("No output from NuNER model".to_string()))?;

        let (_, data_slice) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Parse(format!("Failed to extract output tensor: {}", e)))?;
        let output_data: Vec<f32> = data_slice.to_vec();

        // Get shape: [batch, num_words, num_classes]
        let shape: Vec<i64> = match output.dtype() {
            ort::value::ValueType::Tensor { shape, .. } => shape.iter().copied().collect(),
            _ => return Err(Error::Parse("Expected tensor output".to_string())),
        };

        if shape.len() < 3 {
            return Err(Error::Parse(format!("Unexpected output shape: {:?}", shape)));
        }

        let num_words = shape[1] as usize;
        let num_classes = shape[2] as usize;

        // Calculate word positions in original text
        let word_positions: Vec<(usize, usize)> = {
            let mut positions = Vec::new();
            let mut pos = 0;
            for word in text_words {
                if let Some(start) = text[pos..].find(word) {
                    let abs_start = pos + start;
                    let abs_end = abs_start + word.len();
                    positions.push((abs_start, abs_end));
                    pos = abs_end;
                }
            }
            positions
        };

        let mut entities = Vec::new();
        let mut current_entity: Option<(usize, usize, usize, f32)> = None; // (start_word, end_word, type_idx, score)

        // Process each word position
        for word_idx in 0..num_words.min(text_words.len()) {
            let base_idx = word_idx * num_classes;
            
            // Find best class for this word
            let mut best_class = 0;
            let mut best_score = 0.0f32;
            
            for class_idx in 0..num_classes {
                let score = output_data.get(base_idx + class_idx).copied().unwrap_or(0.0);
                if score > best_score {
                    best_score = score;
                    best_class = class_idx;
                }
            }

            // BIO decoding: class 0 = O, odd = B-type, even = I-type
            let is_begin = best_class > 0 && best_class % 2 == 1;
            let is_inside = best_class > 0 && best_class % 2 == 0;
            let type_idx = if best_class > 0 { (best_class - 1) / 2 } else { 0 };

            if best_score >= threshold {
                if is_begin {
                    // Flush previous entity
                    if let Some((start, end, etype, score)) = current_entity.take() {
                        if let Some(e) = self.create_entity(text, &word_positions, start, end, etype, score, entity_types) {
                            entities.push(e);
                        }
                    }
                    // Start new entity
                    current_entity = Some((word_idx, word_idx + 1, type_idx, best_score));
                } else if is_inside {
                    // Extend current entity if same type
                    if let Some((start, _end, etype, score)) = current_entity.as_mut() {
                        if *etype == type_idx {
                            *_end = word_idx + 1;
                            *score = (*score + best_score) / 2.0; // Average confidence
                        }
                    }
                }
            } else {
                // Low confidence or O tag - flush current entity
                if let Some((start, end, etype, score)) = current_entity.take() {
                    if let Some(e) = self.create_entity(text, &word_positions, start, end, etype, score, entity_types) {
                        entities.push(e);
                    }
                }
            }
        }

        // Flush final entity
        if let Some((start, end, etype, score)) = current_entity.take() {
            if let Some(e) = self.create_entity(text, &word_positions, start, end, etype, score, entity_types) {
                entities.push(e);
            }
        }

        Ok(entities)
    }

    #[cfg(feature = "onnx")]
    fn create_entity(
        &self,
        text: &str,
        word_positions: &[(usize, usize)],
        start_word: usize,
        end_word: usize,
        type_idx: usize,
        score: f32,
        entity_types: &[&str],
    ) -> Option<Entity> {
        let start_pos = word_positions.get(start_word)?.0;
        let end_pos = word_positions.get(end_word - 1)?.1;
        
        let entity_text = text.get(start_pos..end_pos)?;
        let label = entity_types.get(type_idx)?;
        let entity_type = Self::map_label_to_entity_type(label);

        Some(Entity::new(
            entity_text,
            entity_type,
            start_pos,
            end_pos,
            score as f64,
        ))
    }

    /// Map label string to EntityType.
    fn map_label_to_entity_type(label: &str) -> EntityType {
        match label.to_lowercase().as_str() {
            "person" | "per" => EntityType::Person,
            "organization" | "org" | "company" => EntityType::Organization,
            "location" | "loc" | "place" | "gpe" => EntityType::Location,
            "date" => EntityType::Date,
            "time" => EntityType::Time,
            "money" | "currency" => EntityType::Money,
            "percent" | "percentage" => EntityType::Percent,
            _ => EntityType::Other(label.to_string()),
        }
    }
}

impl Default for NuNER {
    fn default() -> Self {
        Self::new()
    }
}

impl Model for NuNER {
    fn extract_entities(&self, text: &str, _language: Option<&str>) -> Result<Vec<Entity>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        #[cfg(feature = "onnx")]
        {
            if self.session.is_some() {
                let labels: Vec<&str> = self.default_labels.iter().map(|s| s.as_str()).collect();
                return self.extract(text, &labels, self.threshold as f32);
            }
        }

        // Without ONNX or without loaded model, return empty
        Ok(Vec::new())
    }

    fn supported_types(&self) -> Vec<EntityType> {
        self.default_labels
            .iter()
            .map(|l| Self::map_label_to_entity_type(l))
            .collect()
    }

    fn is_available(&self) -> bool {
        #[cfg(feature = "onnx")]
        {
            return self.session.is_some();
        }
        #[cfg(not(feature = "onnx"))]
        {
            false
        }
    }

    fn name(&self) -> &'static str {
        "nuner"
    }

    fn description(&self) -> &'static str {
        "NuNER Zero: Token-based zero-shot NER from NuMind (MIT licensed)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nuner_creation() {
        let ner = NuNER::new();
        assert_eq!(ner.model_id(), "numind/NuNER_Zero");
        assert!((ner.threshold() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_nuner_with_custom_model() {
        let ner = NuNER::with_model("custom/model")
            .with_threshold(0.7)
            .with_labels(vec!["technology".to_string()]);
        
        assert_eq!(ner.model_id(), "custom/model");
        assert!((ner.threshold() - 0.7).abs() < f64::EPSILON);
        assert_eq!(ner.default_labels.len(), 1);
    }

    #[test]
    fn test_label_mapping() {
        assert_eq!(NuNER::map_label_to_entity_type("person"), EntityType::Person);
        assert_eq!(NuNER::map_label_to_entity_type("PER"), EntityType::Person);
        assert_eq!(NuNER::map_label_to_entity_type("organization"), EntityType::Organization);
        assert_eq!(NuNER::map_label_to_entity_type("custom"), EntityType::Other("custom".to_string()));
    }

    #[test]
    fn test_supported_types() {
        let ner = NuNER::new();
        let types = ner.supported_types();
        assert!(types.contains(&EntityType::Person));
        assert!(types.contains(&EntityType::Organization));
        assert!(types.contains(&EntityType::Location));
    }

    #[test]
    fn test_empty_input() {
        let ner = NuNER::new();
        let entities = ner.extract_entities("", None).unwrap();
        assert!(entities.is_empty());
    }

    #[test]
    fn test_not_available_without_model() {
        let ner = NuNER::new();
        assert!(!ner.is_available());
    }
}
