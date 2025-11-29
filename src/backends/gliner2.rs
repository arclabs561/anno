//! GLiNER2: Multi-task Information Extraction.
//!
//! GLiNER2 extends GLiNER to support:
//! - Named Entity Recognition (with label descriptions)
//! - Text Classification (single/multi-label)
//! - Hierarchical Structure Extraction
//! - Task Composition (multiple tasks in one pass)
//!
//! # Architecture
//!
//! Based on arXiv:2507.18546 (July 2025):
//!
//! ```text
//! Input: [Task Prompt] ⊕ [SEP] ⊕ [Input Text]
//!
//! Task Prompts:
//!   NER:    [P] entities ([E]type1 [E]type2 ...) [SEP] text
//!   Class:  [P] task ([L]label1 [L]label2 ...) [SEP] text
//!   Struct: [P] parent ([C]field1 [C]field2 ...) [SEP] text
//! ```
//!
//! # Special Tokens
//!
//! - `[P]` - Prompt marker (task specification)
//! - `[E]` - Entity type marker
//! - `[C]` - Child/Component marker (for hierarchical)
//! - `[L]` - Label marker (for classification)
//! - `[SEP]` - Separator between task and text
//!
//! # Trait Integration
//!
//! GLiNER2 implements the standard `anno` traits:
//! - `Model` - Core entity extraction interface
//! - `ZeroShotNER` - Open-domain entity types
//! - `RelationExtractor` - Joint entity-relation extraction (via GLiREL)
//! - `BatchCapable` - Efficient batch processing
//!
//! # Usage
//!
//! ```rust,ignore
//! use anno::{Model, ZeroShotNER, DEFAULT_GLINER2_MODEL};
//! use anno::backends::gliner2::{GLiNER2, TaskSchema};
//!
//! // Use the official Fastino Labs GLiNER2 model
//! let model = GLiNER2::from_pretrained(DEFAULT_GLINER2_MODEL)?;
//! // Or: GLiNER2::from_pretrained("fastino/gliner2-base-v1")?;
//!
//! // Standard Model trait
//! let entities = model.extract_entities("Apple announced iPhone 15", None)?;
//!
//! // Zero-shot with custom types
//! let types = &["company", "product", "event"];
//! let entities = model.extract_with_types(text, types, 0.5)?;
//!
//! // Multi-task extraction with schema
//! let schema = TaskSchema::new()
//!     .with_entities(&["person", "organization", "product"])
//!     .with_classification("sentiment", &["positive", "negative", "neutral"]);
//!
//! let result = model.extract_with_schema("Apple announced iPhone 15", &schema)?;
//! ```
//!
//! # Backends
//!
//! - **ONNX** (recommended): `cargo build --features onnx`
//! - **Candle** (native): `cargo build --features candle`

use crate::entity::EntityCategory;
use crate::{Entity, EntityType, Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
#[cfg(feature = "candle")]
use std::sync::RwLock;

// Import trait definitions for implementations
use crate::backends::inference::{ExtractionWithRelations, RelationExtractor, ZeroShotNER};

// =============================================================================
// Special Token IDs (GLiNER2 vocabulary)
// =============================================================================

/// Prompt marker token [P]
#[cfg(feature = "onnx")]
const TOKEN_P: u32 = 128004;
/// Entity type marker token [E]
#[cfg(feature = "onnx")]
const TOKEN_E: u32 = 128002;
/// Child/component marker token [C] (used for structure extraction)
#[allow(dead_code)]
const TOKEN_C: u32 = 128005;
/// Label marker token [L]
#[cfg(feature = "onnx")]
const TOKEN_L: u32 = 128006;
/// Separator token [SEP]
#[cfg(feature = "onnx")]
const TOKEN_SEP: u32 = 128003;
/// Start token
#[cfg(feature = "onnx")]
const TOKEN_START: u32 = 1;
/// End token
#[cfg(feature = "onnx")]
const TOKEN_END: u32 = 2;

/// Default max span width
const MAX_SPAN_WIDTH: usize = 12;
/// Max count for structure instances (0-19)
#[cfg(feature = "candle")]
const MAX_COUNT: usize = 20;

// =============================================================================
// Label Embedding Cache
// =============================================================================

/// Cache for label embeddings to avoid recomputation
#[derive(Debug, Default)]
pub struct LabelCache {
    #[cfg(feature = "candle")]
    cache: RwLock<HashMap<String, Vec<f32>>>,
    #[cfg(not(feature = "candle"))]
    _phantom: std::marker::PhantomData<()>,
}

#[cfg(feature = "candle")]
impl LabelCache {
    fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
        }
    }

    fn get(&self, label: &str) -> Option<Vec<f32>> {
        self.cache.read().ok()?.get(label).cloned()
    }

    fn insert(&self, label: String, embedding: Vec<f32>) {
        if let Ok(mut cache) = self.cache.write() {
            cache.insert(label, embedding);
        }
    }
}

#[cfg(not(feature = "candle"))]
impl LabelCache {
    #[allow(dead_code)]
    fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

// =============================================================================
// Task Schema
// =============================================================================

/// Schema defining what to extract.
///
/// Use builder methods to construct complex schemas:
///
/// ```rust,ignore
/// let schema = TaskSchema::new()
///     .with_entities(&["person", "organization"])
///     .with_classification("sentiment", &["positive", "negative"], false)
///     .with_structure(
///         StructureTask::new("product")
///             .with_field("name", FieldType::String)
///             .with_field("price", FieldType::String)
///     );
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TaskSchema {
    /// Entity types to extract
    pub entities: Option<EntityTask>,
    /// Classification tasks
    pub classifications: Vec<ClassificationTask>,
    /// Structure extraction tasks
    pub structures: Vec<StructureTask>,
}

impl TaskSchema {
    /// Create empty schema.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add entity types to extract.
    pub fn with_entities(mut self, types: &[&str]) -> Self {
        self.entities = Some(EntityTask {
            types: types.iter().map(|s| s.to_string()).collect(),
            descriptions: HashMap::new(),
        });
        self
    }

    /// Add entity types with descriptions for better zero-shot.
    pub fn with_entities_described(mut self, types_with_desc: HashMap<String, String>) -> Self {
        let types: Vec<String> = types_with_desc.keys().cloned().collect();
        self.entities = Some(EntityTask {
            types,
            descriptions: types_with_desc,
        });
        self
    }

    /// Add a classification task.
    pub fn with_classification(mut self, name: &str, labels: &[&str], multi_label: bool) -> Self {
        self.classifications.push(ClassificationTask {
            name: name.to_string(),
            labels: labels.iter().map(|s| s.to_string()).collect(),
            multi_label,
            descriptions: HashMap::new(),
        });
        self
    }

    /// Add a structure extraction task.
    pub fn with_structure(mut self, task: StructureTask) -> Self {
        self.structures.push(task);
        self
    }
}

/// Entity extraction task configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EntityTask {
    /// Entity type labels
    pub types: Vec<String>,
    /// Optional descriptions for each type
    pub descriptions: HashMap<String, String>,
}

/// Classification task configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClassificationTask {
    /// Task name (e.g., "sentiment")
    pub name: String,
    /// Class labels
    pub labels: Vec<String>,
    /// Whether multiple labels can be selected
    pub multi_label: bool,
    /// Optional descriptions for labels
    pub descriptions: HashMap<String, String>,
}

/// Hierarchical structure extraction task.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StructureTask {
    /// Structure type name (parent entity)
    pub name: String,
    /// Internal alias for compatibility
    #[serde(skip)]
    pub structure_type: String,
    /// Child fields to extract
    pub fields: Vec<StructureField>,
}

impl StructureTask {
    /// Create new structure task.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            structure_type: name.to_string(),
            fields: Vec::new(),
        }
    }

    /// Add a field to extract.
    pub fn with_field(mut self, name: &str, field_type: FieldType) -> Self {
        self.fields.push(StructureField {
            name: name.to_string(),
            field_type,
            description: None,
            choices: None,
        });
        self
    }

    /// Add a field with description.
    pub fn with_field_described(
        mut self,
        name: &str,
        field_type: FieldType,
        description: &str,
    ) -> Self {
        self.fields.push(StructureField {
            name: name.to_string(),
            field_type,
            description: Some(description.to_string()),
            choices: None,
        });
        self
    }

    /// Add a choice field with constrained options.
    pub fn with_choice_field(mut self, name: &str, choices: &[&str]) -> Self {
        self.fields.push(StructureField {
            name: name.to_string(),
            field_type: FieldType::Choice,
            description: None,
            choices: Some(choices.iter().map(|s| s.to_string()).collect()),
        });
        self
    }
}

/// Structure field configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructureField {
    /// Field name
    pub name: String,
    /// Field type
    pub field_type: FieldType,
    /// Optional description
    pub description: Option<String>,
    /// For Choice type: allowed values
    pub choices: Option<Vec<String>>,
}

/// Field type for structure extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldType {
    /// Single string value
    String,
    /// List of values
    List,
    /// Choice from constrained options
    Choice,
}

// =============================================================================
// Extraction Results
// =============================================================================

/// Combined extraction result.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExtractionResult {
    /// Extracted entities
    pub entities: Vec<Entity>,
    /// Classification results by task name
    pub classifications: HashMap<String, ClassificationResult>,
    /// Extracted structures
    pub structures: Vec<ExtractedStructure>,
}

/// Classification result.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClassificationResult {
    /// Selected label(s)
    pub labels: Vec<String>,
    /// Score for each label
    pub scores: HashMap<String, f32>,
}

/// Extracted structure instance.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExtractedStructure {
    /// Structure type
    pub structure_type: String,
    /// Extracted field values
    pub fields: HashMap<String, StructureValue>,
}

/// Value for a structure field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StructureValue {
    /// Single value
    Single(String),
    /// List of values
    List(Vec<String>),
}

// =============================================================================
// ONNX Backend
// =============================================================================

/// GLiNER2 ONNX implementation.
/// GLiNER2 ONNX implementation.
#[cfg(feature = "onnx")]
#[derive(Debug)]
pub struct GLiNER2Onnx {
    session: std::sync::Mutex<ort::session::Session>,
    tokenizer: tokenizers::Tokenizer,
    #[allow(dead_code)]
    model_name: String,
    #[allow(dead_code)]
    hidden_size: usize,
}

#[cfg(feature = "onnx")]
impl GLiNER2Onnx {
    /// Load model from HuggingFace Hub.
    pub fn from_pretrained(model_id: &str) -> Result<Self> {
        use hf_hub::api::sync::Api;
        use ort::execution_providers::CPUExecutionProvider;
        use ort::session::Session;

        let api = Api::new().map_err(|e| Error::Retrieval(format!("HF API: {}", e)))?;
        let repo = api.model(model_id.to_string());

        // Try different model file names
        let model_path = repo
            .get("onnx/model.onnx")
            .or_else(|_| repo.get("model.onnx"))
            .map_err(|e| Error::Retrieval(format!("model.onnx: {}", e)))?;

        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| Error::Retrieval(format!("tokenizer.json: {}", e)))?;

        let config_path = repo
            .get("config.json")
            .map_err(|e| Error::Retrieval(format!("config.json: {}", e)))?;

        // Load tokenizer
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Retrieval(format!("tokenizer: {}", e)))?;

        // Parse config
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| Error::Retrieval(format!("config read: {}", e)))?;
        let config: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| Error::Parse(format!("config parse: {}", e)))?;
        let hidden_size = config["hidden_size"].as_u64().unwrap_or(768) as usize;

        // Create ONNX session
        let session = Session::builder()
            .map_err(|e| Error::Retrieval(format!("ONNX builder: {}", e)))?
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .map_err(|e| Error::Retrieval(format!("ONNX providers: {}", e)))?
            .commit_from_file(&model_path)
            .map_err(|e| Error::Retrieval(format!("ONNX load: {}", e)))?;

        log::info!(
            "[GLiNER2-ONNX] Loaded {} (hidden={})",
            model_id,
            hidden_size
        );
        log::debug!(
            "[GLiNER2-ONNX] Inputs: {:?}",
            session.inputs.iter().map(|i| &i.name).collect::<Vec<_>>()
        );
        log::debug!(
            "[GLiNER2-ONNX] Outputs: {:?}",
            session.outputs.iter().map(|o| &o.name).collect::<Vec<_>>()
        );

        Ok(Self {
            session: std::sync::Mutex::new(session),
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

    /// Extract named entities using GLiNER2 NER format.
    fn extract_ner(
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

        // Encode following GLiNER2 format: [P] entities ([E]type1 [E]type2 ...) [SEP] text
        let (input_ids, attention_mask, words_mask) =
            self.encode_ner_prompt(&text_words, entity_types)?;

        // Generate span tensors
        let (span_idx, span_mask) = self.make_span_tensors(text_words.len());

        // Build tensors
        use ndarray::{Array2, Array3};
        use ort::value::Tensor;

        let batch_size = 1;
        let seq_len = input_ids.len();
        let num_spans = text_words.len() * MAX_SPAN_WIDTH;

        let input_ids_arr = Array2::from_shape_vec((batch_size, seq_len), input_ids)
            .map_err(|e| Error::Parse(format!("Array: {}", e)))?;
        let attention_mask_arr = Array2::from_shape_vec((batch_size, seq_len), attention_mask)
            .map_err(|e| Error::Parse(format!("Array: {}", e)))?;
        let words_mask_arr = Array2::from_shape_vec((batch_size, seq_len), words_mask)
            .map_err(|e| Error::Parse(format!("Array: {}", e)))?;
        let text_lengths_arr =
            Array2::from_shape_vec((batch_size, 1), vec![text_words.len() as i64])
                .map_err(|e| Error::Parse(format!("Array: {}", e)))?;
        let span_idx_arr = Array3::from_shape_vec((batch_size, num_spans, 2), span_idx)
            .map_err(|e| Error::Parse(format!("Array: {}", e)))?;
        let span_mask_arr = Array2::from_shape_vec((batch_size, num_spans), span_mask)
            .map_err(|e| Error::Parse(format!("Array: {}", e)))?;

        let input_ids_t = Tensor::from_array(input_ids_arr)
            .map_err(|e| Error::Parse(format!("Tensor: {}", e)))?;
        let attention_mask_t = Tensor::from_array(attention_mask_arr)
            .map_err(|e| Error::Parse(format!("Tensor: {}", e)))?;
        let words_mask_t = Tensor::from_array(words_mask_arr)
            .map_err(|e| Error::Parse(format!("Tensor: {}", e)))?;
        let text_lengths_t = Tensor::from_array(text_lengths_arr)
            .map_err(|e| Error::Parse(format!("Tensor: {}", e)))?;
        let span_idx_t =
            Tensor::from_array(span_idx_arr).map_err(|e| Error::Parse(format!("Tensor: {}", e)))?;
        let span_mask_t = Tensor::from_array(span_mask_arr)
            .map_err(|e| Error::Parse(format!("Tensor: {}", e)))?;

        // Run inference
        let mut session = self
            .session
            .lock()
            .map_err(|_| Error::Inference("Session lock failed".into()))?;

        let outputs = session
            .run(ort::inputs![
                "input_ids" => input_ids_t.into_dyn(),
                "attention_mask" => attention_mask_t.into_dyn(),
                "words_mask" => words_mask_t.into_dyn(),
                "text_lengths" => text_lengths_t.into_dyn(),
                "span_idx" => span_idx_t.into_dyn(),
                "span_mask" => span_mask_t.into_dyn(),
            ])
            .map_err(|e| Error::Inference(format!("ONNX run: {}", e)))?;

        // Decode output
        self.decode_ner_output(&outputs, text, &text_words, entity_types, threshold)
    }

    /// Encode NER prompt: [START] [P] entities ([E]type1 ...) [SEP] word1 word2 ... [END]
    fn encode_ner_prompt(
        &self,
        text_words: &[&str],
        entity_types: &[&str],
    ) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>)> {
        let mut input_ids: Vec<i64> = Vec::new();
        let mut word_mask: Vec<i64> = Vec::new();

        // Start token
        input_ids.push(TOKEN_START as i64);
        word_mask.push(0);

        // [P] token for prompt marker
        input_ids.push(TOKEN_P as i64);
        word_mask.push(0);

        // "entities" keyword tokens (optional, some models skip this)
        let entities_enc = self
            .tokenizer
            .encode("entities", false)
            .map_err(|e| Error::Parse(format!("Tokenize: {}", e)))?;
        for token_id in entities_enc.get_ids() {
            input_ids.push(*token_id as i64);
            word_mask.push(0);
        }

        // Entity types: [E]type1 [E]type2 ...
        for entity_type in entity_types {
            input_ids.push(TOKEN_E as i64);
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
        let mut word_id: i64 = 0;
        for word in text_words {
            let word_enc = self
                .tokenizer
                .encode(*word, false)
                .map_err(|e| Error::Parse(format!("Tokenize: {}", e)))?;

            word_id += 1;
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

    /// Generate span tensors.
    fn make_span_tensors(&self, num_words: usize) -> (Vec<i64>, Vec<bool>) {
        let num_spans = num_words * MAX_SPAN_WIDTH;
        let mut span_idx: Vec<i64> = vec![0; num_spans * 2];
        let mut span_mask: Vec<bool> = vec![false; num_spans];

        for start in 0..num_words {
            let remaining = num_words - start;
            let max_width = MAX_SPAN_WIDTH.min(remaining);

            for width in 0..max_width {
                let dim = start * MAX_SPAN_WIDTH + width;
                span_idx[dim * 2] = start as i64;
                span_idx[dim * 2 + 1] = (start + width) as i64;
                span_mask[dim] = true;
            }
        }

        (span_idx, span_mask)
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

        if output_data.is_empty() {
            return Ok(Vec::new());
        }

        let mut entities = Vec::new();
        let num_words = text_words.len();

        // Shape: [batch, num_words, max_width, num_classes]
        if shape.len() == 4 && shape[0] == 1 {
            let out_num_words = shape[1] as usize;
            let out_max_width = shape[2] as usize;
            let num_classes = shape[3] as usize;

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
        entities.sort_by(|a, b| a.start.cmp(&b.start).then_with(|| b.end.cmp(&a.end)));
        entities.dedup_by(|a, b| a.start == b.start && a.end == b.end);

        Ok(entities)
    }

    /// Decode batch NER output into per-text entity vectors.
    fn decode_ner_batch_output(
        &self,
        outputs: &ort::session::SessionOutputs,
        texts: &[&str],
        text_words_batch: &[Vec<&str>],
        entity_types: &[&str],
        threshold: f32,
    ) -> Result<Vec<Vec<Entity>>> {
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

        if output_data.is_empty() {
            return Ok(texts.iter().map(|_| Vec::new()).collect());
        }

        let mut results = Vec::with_capacity(texts.len());

        // Shape: [batch, num_words, max_width, num_classes]
        if shape.len() == 4 {
            let batch_size = shape[0] as usize;
            let out_num_words = shape[1] as usize;
            let out_max_width = shape[2] as usize;
            let num_classes = shape[3] as usize;
            let stride_per_batch = out_num_words * out_max_width * num_classes;

            for batch_idx in 0..batch_size.min(texts.len()) {
                let text = texts[batch_idx];
                let text_words = &text_words_batch[batch_idx];
                let num_words = text_words.len();
                let batch_offset = batch_idx * stride_per_batch;
                let mut entities = Vec::new();

                for word_idx in 0..out_num_words.min(num_words) {
                    for width in 0..out_max_width.min(MAX_SPAN_WIDTH) {
                        let end_word = word_idx + width;
                        if end_word >= num_words {
                            continue;
                        }

                        for class_idx in 0..num_classes.min(entity_types.len()) {
                            let idx = batch_offset
                                + (word_idx * out_max_width * num_classes)
                                + (width * num_classes)
                                + class_idx;

                            if idx < output_data.len() {
                                let logit = output_data[idx];
                                let score = 1.0 / (1.0 + (-logit).exp());

                                if score >= threshold {
                                    let span_text = text_words[word_idx..=end_word].join(" ");
                                    let (start, end) = word_span_to_char_offsets(
                                        text, text_words, word_idx, end_word,
                                    );

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

                // Deduplicate per batch item
                entities.sort_by(|a, b| a.start.cmp(&b.start).then_with(|| b.end.cmp(&a.end)));
                entities.dedup_by(|a, b| a.start == b.start && a.end == b.end);
                results.push(entities);
            }
        } else {
            // Fallback: return empty results
            results = texts.iter().map(|_| Vec::new()).collect();
        }

        Ok(results)
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

        // For classification, we use a simpler approach: encode [P] task ([L]label1 ...) [SEP] text
        // Then use [CLS] or mean-pooled representation

        // Encode input
        let mut input_ids: Vec<i64> = Vec::new();

        input_ids.push(TOKEN_START as i64);
        input_ids.push(TOKEN_P as i64);

        // Classification task marker
        let class_enc = self
            .tokenizer
            .encode("classification", false)
            .map_err(|e| Error::Parse(format!("Tokenize: {}", e)))?;
        for id in class_enc.get_ids() {
            input_ids.push(*id as i64);
        }

        // Labels: [L]label1 [L]label2 ...
        for label in labels {
            input_ids.push(TOKEN_L as i64);
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
        use ort::value::Tensor;

        let input_arr = Array2::from_shape_vec((1, seq_len), input_ids)
            .map_err(|e| Error::Parse(format!("Array: {}", e)))?;
        let attn_arr = Array2::from_shape_vec((1, seq_len), attention_mask)
            .map_err(|e| Error::Parse(format!("Array: {}", e)))?;

        let input_t =
            Tensor::from_array(input_arr).map_err(|e| Error::Parse(format!("Tensor: {}", e)))?;
        let attn_t =
            Tensor::from_array(attn_arr).map_err(|e| Error::Parse(format!("Tensor: {}", e)))?;

        // For classification models, we typically need just input_ids and attention_mask
        // The model should output classification logits
        let mut session = self
            .session
            .lock()
            .map_err(|_| Error::Inference("Lock failed".into()))?;

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
            exp_logits.iter().map(|&x| x / sum).collect::<Vec<_>>()
        };

        let mut scores = HashMap::new();
        let mut selected_labels = Vec::new();

        for (i, label) in labels.iter().enumerate() {
            let prob = probs.get(i).copied().unwrap_or(0.0);
            scores.insert(label.to_string(), prob);

            if multi_label && prob > 0.5 {
                selected_labels.push(label.to_string());
            }
        }

        if !multi_label {
            if let Some((idx, _)) = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                if let Some(label) = labels.get(idx) {
                    selected_labels.push(label.to_string());
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

    /// Build prompt string for logging.
    #[allow(dead_code)]
    fn build_prompt(&self, schema: &TaskSchema) -> String {
        let mut parts = Vec::new();

        if let Some(ref ent_task) = schema.entities {
            let types: Vec<String> = ent_task
                .types
                .iter()
                .map(|t| {
                    if let Some(desc) = ent_task.descriptions.get(t) {
                        format!("[E] {}: {}", t, desc)
                    } else {
                        format!("[E] {}", t)
                    }
                })
                .collect();
            parts.push(format!("[P] entities ({})", types.join(" ")));
        }

        for class_task in &schema.classifications {
            let labels: Vec<String> = class_task
                .labels
                .iter()
                .map(|l| format!("[L] {}", l))
                .collect();
            parts.push(format!("[P] {} ({})", class_task.name, labels.join(" ")));
        }

        for struct_task in &schema.structures {
            let fields: Vec<String> = struct_task
                .fields
                .iter()
                .map(|f| format!("[C] {}", f.name))
                .collect();
            parts.push(format!("[P] {} ({})", struct_task.name, fields.join(" ")));
        }

        parts.join(" [SEP] ")
    }
}

// =============================================================================
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
    device: Device,
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
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap_or((1, &0.0));

        Ok(max_idx.max(1)) // At least 1 instance
    }
}

#[cfg(feature = "candle")]
impl GLiNER2Candle {
    /// Load model from HuggingFace Hub.
    pub fn from_pretrained(model_id: &str) -> Result<Self> {
        use crate::backends::encoder_candle::CandleEncoder;
        use hf_hub::api::sync::Api;

        let api = Api::new().map_err(|e| Error::Retrieval(format!("HF API: {}", e)))?;
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

        // Load weights
        let weights_path = repo
            .get("model.safetensors")
            .or_else(|_| repo.get("gliner_model.safetensors"))
            .or_else(|_| repo.get("pytorch_model.bin"))
            .map_err(|e| Error::Retrieval(format!("weights: {}", e)))?;

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
    fn extract_entities(
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
            exp_scores.iter().map(|&e| e / sum).collect::<Vec<_>>()
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
            for word in words {
                if let Some(start) = full_text[pos..].find(word) {
                    let abs_start = pos + start;
                    positions.push((abs_start, abs_start + word.len()));
                    pos = abs_start + word.len();
                }
            }
            positions
        };

        Ok((tensor, word_positions))
    }

    fn encode_labels_cached(&self, labels: &[&str]) -> Result<Tensor> {
        let mut all_embeddings = Vec::new();

        for label in labels {
            // Check cache first
            if let Some(cached) = self.label_cache.get(label) {
                all_embeddings.extend(cached);
            } else {
                let (embeddings, seq_len) = self.encoder.encode(label)?;
                // Average pool
                let avg: Vec<f32> = (0..self.hidden_size)
                    .map(|i| {
                        embeddings
                            .iter()
                            .skip(i)
                            .step_by(self.hidden_size)
                            .take(seq_len)
                            .sum::<f32>()
                            / seq_len as f32
                    })
                    .collect();

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
        let mut spans = Vec::new();

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

        let mut entities = Vec::new();
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
        entities.sort_by(|a, b| a.start.cmp(&b.start).then_with(|| b.end.cmp(&a.end)));
        entities.dedup_by(|a, b| a.start == b.start && a.end == b.end);

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
// Stub implementations (no feature)
// =============================================================================

/// GLiNER2 stub (requires onnx or candle feature).
#[cfg(not(any(feature = "onnx", feature = "candle")))]
#[derive(Debug)]
pub struct GLiNER2 {
    _private: (),
}

#[cfg(not(any(feature = "onnx", feature = "candle")))]
impl GLiNER2 {
    /// Load model (requires feature).
    pub fn from_pretrained(_model_id: &str) -> Result<Self> {
        Err(Error::FeatureNotAvailable(
            "GLiNER2 requires 'onnx' or 'candle' feature. \
             Build with: cargo build --features candle"
                .to_string(),
        ))
    }

    /// Extract (requires feature).
    pub fn extract(&self, _text: &str, _schema: &TaskSchema) -> Result<ExtractionResult> {
        Err(Error::FeatureNotAvailable(
            "GLiNER2 requires features".to_string(),
        ))
    }
}

// =============================================================================
// Unified GLiNER2 type
// =============================================================================

/// GLiNER2 model - automatically selects best available backend.
#[cfg(feature = "candle")]
pub type GLiNER2 = GLiNER2Candle;

/// GLiNER2 model - ONNX backend (when candle not enabled).
#[cfg(all(feature = "onnx", not(feature = "candle")))]
pub type GLiNER2 = GLiNER2Onnx;

// =============================================================================
// Helper functions
// =============================================================================

/// Convert word span indices to character offsets.
fn word_span_to_char_offsets(
    text: &str,
    words: &[&str],
    start_word: usize,
    end_word: usize,
) -> (usize, usize) {
    let mut char_pos = 0;
    let mut char_start = 0;
    let mut char_end = text.len();

    for (i, word) in words.iter().enumerate() {
        if let Some(pos) = text[char_pos..].find(word) {
            let abs_pos = char_pos + pos;

            if i == start_word {
                char_start = abs_pos;
            }
            if i == end_word {
                char_end = abs_pos + word.len();
            }

            char_pos = abs_pos + word.len();
        }
    }

    (char_start, char_end)
}

/// Map entity type string to EntityType.
///
/// Uses the canonical schema mapper for consistent semantics across all backends.
fn map_entity_type(type_str: &str) -> EntityType {
    crate::schema::map_to_canonical(type_str, None)
}

// =============================================================================
// Model Trait Implementation (ONNX)
// =============================================================================

#[cfg(feature = "onnx")]
impl crate::Model for GLiNER2Onnx {
    fn extract_entities(&self, text: &str, _language: Option<&str>) -> Result<Vec<Entity>> {
        let schema = TaskSchema::new().with_entities(&[
            "person",
            "organization",
            "location",
            "date",
            "event",
        ]);

        let result = self.extract(text, &schema)?;
        Ok(result.entities)
    }

    fn supported_types(&self) -> Vec<EntityType> {
        vec![
            EntityType::Person,
            EntityType::Organization,
            EntityType::Location,
            EntityType::Date,
            EntityType::Custom {
                name: "event".to_string(),
                category: EntityCategory::Creative,
            },
            EntityType::Custom {
                name: "product".to_string(),
                category: EntityCategory::Creative,
            },
            EntityType::Other("misc".to_string()),
        ]
    }

    fn is_available(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "GLiNER2-ONNX"
    }

    fn description(&self) -> &'static str {
        "Multi-task information extraction via GLiNER2 (ONNX backend)"
    }
}

// =============================================================================
// Model Trait Implementation (Candle)
// =============================================================================

#[cfg(feature = "candle")]
impl crate::Model for GLiNER2Candle {
    fn extract_entities(&self, text: &str, _language: Option<&str>) -> Result<Vec<Entity>> {
        let schema = TaskSchema::new().with_entities(&[
            "person",
            "organization",
            "location",
            "date",
            "event",
        ]);

        let result = self.extract(text, &schema)?;
        Ok(result.entities)
    }

    fn supported_types(&self) -> Vec<EntityType> {
        vec![
            EntityType::Person,
            EntityType::Organization,
            EntityType::Location,
            EntityType::Date,
            EntityType::Custom {
                name: "event".to_string(),
                category: EntityCategory::Creative,
            },
            EntityType::Custom {
                name: "product".to_string(),
                category: EntityCategory::Creative,
            },
            EntityType::Other("misc".to_string()),
        ]
    }

    fn is_available(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "GLiNER2-Candle"
    }

    fn description(&self) -> &'static str {
        "Multi-task information extraction via GLiNER2 (native Rust/Candle)"
    }
}

// =============================================================================
// ZeroShotNER Trait Implementation
// =============================================================================

#[cfg(feature = "onnx")]
impl ZeroShotNER for GLiNER2Onnx {
    fn extract_with_types(
        &self,
        text: &str,
        types: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        self.extract_ner(text, types, threshold)
    }

    fn extract_with_descriptions(
        &self,
        text: &str,
        descriptions: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        // Use descriptions as entity types directly (GLiNER2 supports this)
        self.extract_ner(text, descriptions, threshold)
    }
}

#[cfg(feature = "candle")]
impl ZeroShotNER for GLiNER2Candle {
    fn extract_with_types(
        &self,
        text: &str,
        types: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        let type_strings: Vec<String> = types.iter().map(|s| s.to_string()).collect();
        self.extract_entities(text, &type_strings, threshold)
    }

    fn extract_with_descriptions(
        &self,
        text: &str,
        descriptions: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        // Use descriptions as entity types directly (GLiNER2 supports this)
        let type_strings: Vec<String> = descriptions.iter().map(|s| s.to_string()).collect();
        self.extract_entities(text, &type_strings, threshold)
    }
}

// =============================================================================
// RelationExtractor Trait Implementation
// =============================================================================

/// Relation extraction patterns for common entity type pairs.
/// Maps (head_type, tail_type) -> likely relation types.
#[cfg(any(feature = "onnx", feature = "candle"))]
fn get_likely_relations(head_type: &str, tail_type: &str) -> Vec<(&'static str, f32)> {
    let head = head_type.to_uppercase();
    let tail = tail_type.to_uppercase();

    match (head.as_str(), tail.as_str()) {
        // Person-Organization relations
        ("PERSON", "ORGANIZATION") | ("PER", "ORG") => vec![
            ("WORKS_FOR", 0.7),
            ("FOUNDED", 0.5),
            ("CEO_OF", 0.4),
            ("MEMBER_OF", 0.6),
        ],
        ("ORGANIZATION", "PERSON") | ("ORG", "PER") => {
            vec![("EMPLOYS", 0.7), ("FOUNDED_BY", 0.5), ("LED_BY", 0.4)]
        }
        // Person-Location relations
        ("PERSON", "LOCATION") | ("PER", "LOC") | ("PERSON", "GPE") | ("PER", "GPE") => {
            vec![("LIVES_IN", 0.6), ("BORN_IN", 0.5), ("VISITED", 0.4)]
        }
        // Organization-Location relations
        ("ORGANIZATION", "LOCATION")
        | ("ORG", "LOC")
        | ("ORGANIZATION", "GPE")
        | ("ORG", "GPE") => vec![
            ("HEADQUARTERED_IN", 0.7),
            ("LOCATED_IN", 0.8),
            ("OPERATES_IN", 0.5),
        ],
        // Product-Organization relations
        ("PRODUCT", "ORGANIZATION") | ("PRODUCT", "ORG") => {
            vec![("MADE_BY", 0.8), ("PRODUCED_BY", 0.7)]
        }
        ("ORGANIZATION", "PRODUCT") | ("ORG", "PRODUCT") => {
            vec![("MAKES", 0.8), ("PRODUCES", 0.7), ("ANNOUNCED", 0.5)]
        }
        // Date relations
        (_, "DATE") | (_, "TIME") => vec![("OCCURRED_ON", 0.5), ("FOUNDED_ON", 0.4)],
        // Default: no strong relation signal
        _ => vec![],
    }
}

/// Extract relations using proximity and type-based heuristics.
/// This is a lightweight approach that doesn't require a separate relation model.
#[cfg(any(feature = "onnx", feature = "candle"))]
fn extract_relations_heuristic(
    entities: &[Entity],
    text: &str,
    relation_types: &[&str],
    threshold: f32,
) -> Vec<crate::backends::inference::RelationTriple> {
    use crate::backends::inference::RelationTriple;

    let mut relations = Vec::new();
    let words: Vec<&str> = text.split_whitespace().collect();
    let _text_len = words.len().max(1) as f32;

    // Relation trigger patterns
    let trigger_patterns: Vec<(&str, &str)> = vec![
        ("CEO", "CEO_OF"),
        ("founder", "FOUNDED"),
        ("founded", "FOUNDED"),
        ("works at", "WORKS_FOR"),
        ("works for", "WORKS_FOR"),
        ("employee", "WORKS_FOR"),
        ("headquartered", "HEADQUARTERED_IN"),
        ("based in", "LOCATED_IN"),
        ("located in", "LOCATED_IN"),
        ("born in", "BORN_IN"),
        ("lives in", "LIVES_IN"),
        ("announced", "ANNOUNCED"),
        ("released", "RELEASED"),
        ("acquired", "ACQUIRED"),
        ("bought", "ACQUIRED"),
        ("merged", "MERGED_WITH"),
    ];

    let text_lower = text.to_lowercase();

    for (i, head) in entities.iter().enumerate() {
        for (j, tail) in entities.iter().enumerate() {
            if i == j {
                continue;
            }

            // Distance-based scoring: closer entities are more likely related
            let head_center = (head.start + head.end) as f32 / 2.0;
            let tail_center = (tail.start + tail.end) as f32 / 2.0;
            let distance = (head_center - tail_center).abs() / text.len().max(1) as f32;
            let proximity_score = 1.0 - distance.min(1.0);

            // Type-based relation candidates
            let head_type = head.entity_type.as_label();
            let tail_type = tail.entity_type.as_label();
            let type_relations = get_likely_relations(head_type, tail_type);

            // Check for trigger patterns in text between entities
            let (span_start, span_end) = if head.end < tail.start {
                (head.end, tail.start)
            } else if tail.end < head.start {
                (tail.end, head.start)
            } else {
                // Overlapping entities - use surrounding context
                let min_start = head.start.min(tail.start);
                let max_end = head.end.max(tail.end);
                (min_start.saturating_sub(20), (max_end + 20).min(text.len()))
            };

            let between_text = if span_end > span_start && span_end <= text.len() {
                &text_lower[span_start..span_end]
            } else {
                ""
            };

            // Check trigger patterns
            for (trigger, rel_type) in &trigger_patterns {
                if between_text.contains(trigger) {
                    // Filter by requested relation types if specified
                    if !relation_types.is_empty()
                        && !relation_types
                            .iter()
                            .any(|r| r.eq_ignore_ascii_case(rel_type))
                    {
                        continue;
                    }

                    let confidence = (proximity_score * 0.6 + 0.4)
                        * (head.confidence + tail.confidence) as f32
                        / 2.0;
                    if confidence >= threshold {
                        relations.push(RelationTriple {
                            head_idx: i,
                            tail_idx: j,
                            relation_type: rel_type.to_string(),
                            confidence,
                        });
                    }
                }
            }

            // Type-based relations (if no explicit trigger found)
            let has_trigger_relation = relations.iter().any(|r| r.head_idx == i && r.tail_idx == j);
            if !has_trigger_relation && proximity_score > 0.3 {
                for (rel_type, base_score) in type_relations {
                    if !relation_types.is_empty()
                        && !relation_types
                            .iter()
                            .any(|r| r.eq_ignore_ascii_case(rel_type))
                    {
                        continue;
                    }

                    let confidence =
                        proximity_score * base_score * (head.confidence + tail.confidence) as f32
                            / 2.0;
                    if confidence >= threshold {
                        relations.push(RelationTriple {
                            head_idx: i,
                            tail_idx: j,
                            relation_type: rel_type.to_string(),
                            confidence,
                        });
                        break; // Only add one type-based relation per pair
                    }
                }
            }
        }
    }

    // Sort by confidence and deduplicate
    relations.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Keep only top relation per entity pair
    let mut seen_pairs = std::collections::HashSet::new();
    relations.retain(|r| seen_pairs.insert((r.head_idx, r.tail_idx)));

    relations
}

#[cfg(feature = "onnx")]
impl RelationExtractor for GLiNER2Onnx {
    fn extract_with_relations(
        &self,
        text: &str,
        types: &[&str],
        relation_types: &[&str],
        threshold: f32,
    ) -> Result<ExtractionWithRelations> {
        // Extract entities first
        let entities = self.extract_ner(text, types, threshold)?;

        // Extract relations using heuristics
        let relations = extract_relations_heuristic(&entities, text, relation_types, threshold);

        Ok(ExtractionWithRelations {
            entities,
            relations,
        })
    }
}

#[cfg(feature = "candle")]
impl RelationExtractor for GLiNER2Candle {
    fn extract_with_relations(
        &self,
        text: &str,
        types: &[&str],
        relation_types: &[&str],
        threshold: f32,
    ) -> Result<ExtractionWithRelations> {
        let type_strings: Vec<String> = types.iter().map(|s| s.to_string()).collect();
        let entities = self.extract_entities(text, &type_strings, threshold)?;

        // Extract relations using heuristics
        let relations = extract_relations_heuristic(&entities, text, relation_types, threshold);

        Ok(ExtractionWithRelations {
            entities,
            relations,
        })
    }
}

// =============================================================================
// BatchCapable Trait Implementation
// =============================================================================

#[cfg(feature = "onnx")]
impl crate::BatchCapable for GLiNER2Onnx {
    fn extract_entities_batch(
        &self,
        texts: &[&str],
        _language: Option<&str>,
    ) -> Result<Vec<Vec<Entity>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let default_types = &["person", "organization", "location", "date", "event"];

        // For true batching, we need to:
        // 1. Tokenize all texts
        // 2. Pad to max length
        // 3. Run as single batch
        // 4. Split results back

        // Collect word-level tokenizations
        let text_words: Vec<Vec<&str>> = texts
            .iter()
            .map(|t| t.split_whitespace().collect())
            .collect();

        // Find max word count
        let max_words = text_words.iter().map(|w| w.len()).max().unwrap_or(0);
        if max_words == 0 {
            return Ok(texts.iter().map(|_| Vec::new()).collect());
        }

        // Encode all prompts
        let mut all_input_ids = Vec::new();
        let mut all_attention_masks = Vec::new();
        let mut all_words_masks = Vec::new();
        let mut all_text_lengths = Vec::new();
        let mut all_span_idx = Vec::new();
        let mut all_span_masks = Vec::new();
        let mut seq_lens = Vec::new();

        for words in &text_words {
            if words.is_empty() {
                // Handle empty text
                seq_lens.push(0);
                continue;
            }

            let (input_ids, attention_mask, words_mask) =
                self.encode_ner_prompt(words, default_types)?;
            seq_lens.push(input_ids.len());
            all_input_ids.push(input_ids);
            all_attention_masks.push(attention_mask);
            all_words_masks.push(words_mask);
            all_text_lengths.push(words.len() as i64);

            let (span_idx, span_mask) = self.make_span_tensors(words.len());
            all_span_idx.push(span_idx);
            all_span_masks.push(span_mask);
        }

        // If all texts were empty, return empty results
        if seq_lens.iter().all(|&l| l == 0) {
            return Ok(texts.iter().map(|_| Vec::new()).collect());
        }

        // Pad sequences to max length
        let max_seq_len = seq_lens.iter().copied().max().unwrap_or(0);
        let max_span_count = max_words * MAX_SPAN_WIDTH;

        for i in 0..all_input_ids.len() {
            let pad_len = max_seq_len - all_input_ids[i].len();
            all_input_ids[i].extend(std::iter::repeat(0i64).take(pad_len));
            all_attention_masks[i].extend(std::iter::repeat(0i64).take(pad_len));
            all_words_masks[i].extend(std::iter::repeat(0i64).take(pad_len));

            // Pad span tensors
            let span_pad = max_span_count * 2 - all_span_idx[i].len();
            all_span_idx[i].extend(std::iter::repeat(0i64).take(span_pad));
            let mask_pad = max_span_count - all_span_masks[i].len();
            all_span_masks[i].extend(std::iter::repeat(false).take(mask_pad));
        }

        // Build batched tensors
        use ndarray::{Array2, Array3};
        use ort::value::Tensor;

        let batch_size = all_input_ids.len();

        let input_ids_flat: Vec<i64> = all_input_ids.into_iter().flatten().collect();
        let attention_mask_flat: Vec<i64> = all_attention_masks.into_iter().flatten().collect();
        let words_mask_flat: Vec<i64> = all_words_masks.into_iter().flatten().collect();
        let span_idx_flat: Vec<i64> = all_span_idx.into_iter().flatten().collect();
        let span_mask_flat: Vec<bool> = all_span_masks.into_iter().flatten().collect();

        let input_ids_arr = Array2::from_shape_vec((batch_size, max_seq_len), input_ids_flat)
            .map_err(|e| Error::Parse(format!("Array: {}", e)))?;
        let attention_mask_arr =
            Array2::from_shape_vec((batch_size, max_seq_len), attention_mask_flat)
                .map_err(|e| Error::Parse(format!("Array: {}", e)))?;
        let words_mask_arr = Array2::from_shape_vec((batch_size, max_seq_len), words_mask_flat)
            .map_err(|e| Error::Parse(format!("Array: {}", e)))?;
        let text_lengths_arr = Array2::from_shape_vec((batch_size, 1), all_text_lengths)
            .map_err(|e| Error::Parse(format!("Array: {}", e)))?;
        let span_idx_arr = Array3::from_shape_vec((batch_size, max_span_count, 2), span_idx_flat)
            .map_err(|e| Error::Parse(format!("Array: {}", e)))?;
        let span_mask_arr = Array2::from_shape_vec((batch_size, max_span_count), span_mask_flat)
            .map_err(|e| Error::Parse(format!("Array: {}", e)))?;

        let input_ids_t = Tensor::from_array(input_ids_arr)
            .map_err(|e| Error::Parse(format!("Tensor: {}", e)))?;
        let attention_mask_t = Tensor::from_array(attention_mask_arr)
            .map_err(|e| Error::Parse(format!("Tensor: {}", e)))?;
        let words_mask_t = Tensor::from_array(words_mask_arr)
            .map_err(|e| Error::Parse(format!("Tensor: {}", e)))?;
        let text_lengths_t = Tensor::from_array(text_lengths_arr)
            .map_err(|e| Error::Parse(format!("Tensor: {}", e)))?;
        let span_idx_t =
            Tensor::from_array(span_idx_arr).map_err(|e| Error::Parse(format!("Tensor: {}", e)))?;
        let span_mask_t = Tensor::from_array(span_mask_arr)
            .map_err(|e| Error::Parse(format!("Tensor: {}", e)))?;

        // Run batched inference
        let mut session = self
            .session
            .lock()
            .map_err(|_| Error::Inference("Session lock failed".into()))?;

        let outputs = session
            .run(ort::inputs![
                "input_ids" => input_ids_t.into_dyn(),
                "attention_mask" => attention_mask_t.into_dyn(),
                "words_mask" => words_mask_t.into_dyn(),
                "text_lengths" => text_lengths_t.into_dyn(),
                "span_idx" => span_idx_t.into_dyn(),
                "span_mask" => span_mask_t.into_dyn(),
            ])
            .map_err(|e| Error::Inference(format!("ONNX batch run: {}", e)))?;

        // Decode batch results
        self.decode_ner_batch_output(&outputs, texts, &text_words, default_types, 0.5)
    }

    fn optimal_batch_size(&self) -> Option<usize> {
        Some(16)
    }
}

#[cfg(feature = "candle")]
impl crate::BatchCapable for GLiNER2Candle {
    fn extract_entities_batch(
        &self,
        texts: &[&str],
        _language: Option<&str>,
    ) -> Result<Vec<Vec<Entity>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let default_types = vec![
            "person".to_string(),
            "organization".to_string(),
            "location".to_string(),
            "date".to_string(),
            "event".to_string(),
        ];

        // Pre-compute label embeddings once for all texts
        let label_refs: Vec<&str> = default_types.iter().map(|s| s.as_str()).collect();
        let _ = self.encode_labels_cached(&label_refs)?;

        // Process texts - labels are now cached for efficiency
        let mut results = Vec::with_capacity(texts.len());

        for text in texts {
            let entities = self.extract_entities(text, &default_types, 0.5)?;
            results.push(entities);
        }

        Ok(results)
    }

    fn optimal_batch_size(&self) -> Option<usize> {
        Some(8)
    }
}

// =============================================================================
// StreamingCapable Trait Implementation
// =============================================================================

#[cfg(feature = "onnx")]
impl crate::StreamingCapable for GLiNER2Onnx {
    // Uses default extract_entities_streaming implementation which adjusts offsets

    fn recommended_chunk_size(&self) -> usize {
        4096 // Characters - translates to ~500 words
    }
}

#[cfg(feature = "candle")]
impl crate::StreamingCapable for GLiNER2Candle {
    // Uses default extract_entities_streaming implementation which adjusts offsets

    fn recommended_chunk_size(&self) -> usize {
        4096
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_schema_builder() {
        let schema = TaskSchema::new()
            .with_entities(&["person", "organization"])
            .with_classification("sentiment", &["positive", "negative"], false);

        assert!(schema.entities.is_some());
        assert_eq!(schema.entities.as_ref().unwrap().types.len(), 2);
        assert_eq!(schema.classifications.len(), 1);
    }

    #[test]
    fn test_structure_task_builder() {
        let task = StructureTask::new("product")
            .with_field("name", FieldType::String)
            .with_field_described("price", FieldType::String, "Product price in USD")
            .with_choice_field("category", &["electronics", "clothing"]);

        assert_eq!(task.fields.len(), 3);
        assert_eq!(task.fields[2].choices.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_word_span_to_char_offsets() {
        let text = "John works at Apple";
        let words: Vec<&str> = text.split_whitespace().collect();

        let (start, end) = word_span_to_char_offsets(text, &words, 0, 0);
        assert_eq!(&text[start..end], "John");

        let (start, end) = word_span_to_char_offsets(text, &words, 3, 3);
        assert_eq!(&text[start..end], "Apple");

        let (start, end) = word_span_to_char_offsets(text, &words, 0, 2);
        assert_eq!(&text[start..end], "John works at");
    }

    #[test]
    fn test_map_entity_type() {
        assert!(matches!(map_entity_type("person"), EntityType::Person));
        assert!(matches!(
            map_entity_type("ORGANIZATION"),
            EntityType::Organization
        ));
        assert!(matches!(map_entity_type("loc"), EntityType::Location));
        // Unknown types map to Other with the uppercase version (due to schema normalization)
        assert!(
            matches!(map_entity_type("custom_type"), EntityType::Other(ref s) if s == "CUSTOM_TYPE")
        );
        // Known special types map to Custom
        assert!(matches!(
            map_entity_type("product"),
            EntityType::Custom { .. }
        ));
        assert!(matches!(
            map_entity_type("event"),
            EntityType::Custom { .. }
        ));
    }
}
