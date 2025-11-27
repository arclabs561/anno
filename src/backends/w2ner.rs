//! W2NER - Unified NER via Word-Word Relation Classification.
//!
//! W2NER (Word-to-Word NER) models NER as classifying relations between
//! every pair of words in a sentence. This elegantly handles:
//!
//! - **Nested entities**: "The \[University of \[California\]\]"
//! - **Discontinuous entities**: "severe \[pain\] ... in \[abdomen\]"
//! - **Overlapping entities**: Same span, different types
//!
//! # Architecture
//!
//! ```text
//! Input: "New York City is great"
//!
//!        ┌─────────────────────────────┐
//!        │      Encoder (BERT)          │
//!        └─────────────────────────────┘
//!                     │
//!        ┌─────────────────────────────┐
//!        │    Biaffine Attention        │
//!        │    (word-word scoring)       │
//!        └─────────────────────────────┘
//!                     │
//!        ┌───────────────────────────────┐
//!        │     Word-Word Grid (N×N×L)    │
//!        │  ┌───┬───┬───┬───┬───┐       │
//!        │  │   │New│York│City│...│      │
//!        │  ├───┼───┼───┼───┼───┤       │
//!        │  │New│ B │NNW│THW│   │       │
//!        │  ├───┼───┼───┼───┼───┤       │
//!        │  │Yrk│   │ B │NNW│   │       │
//!        │  ├───┼───┼───┼───┼───┤       │
//!        │  │Cty│   │   │ B │   │       │
//!        │  └───┴───┴───┴───┴───┘       │
//!        └───────────────────────────────┘
//!
//! Legend:
//!   B   = Begin entity
//!   NNW = Next-Neighboring-Word (same entity)
//!   THW = Tail-Head-Word (entity boundary)
//! ```
//!
//! # Grid Labels
//!
//! W2NER uses three relation types for each entity label:
//!
//! - **NNW (Next-Neighboring-Word)**: Token i and j are adjacent in same entity
//! - **THW (Tail-Head-Word)**: Token i is tail, token j is head of entity
//! - **None**: No relation
//!
//! # Usage
//!
//! ```rust,ignore
//! use anno::W2NER;
//!
//! // Load W2NER model (requires `onnx` feature)
//! let w2ner = W2NER::from_pretrained("path/to/w2ner-model")?;
//!
//! let text = "The University of California Berkeley";
//! let entities = w2ner.extract_entities(text, None)?;
//! // Returns nested entities: ORG + nested LOC
//! ```
//!
//! # References
//!
//! - [W2NER Paper](https://arxiv.org/abs/2112.10070) (AAAI 2022)
//! - [TPLinker](https://aclanthology.org/2020.coling-main.138/) (related approach)

use crate::{Entity, EntityType, Model, Result};
use crate::backends::inference::{DiscontinuousEntity, DiscontinuousNER, HandshakingMatrix, HandshakingCell};

#[cfg(feature = "onnx")]
use crate::Error;

/// W2NER relation types for word-word classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum W2NERRelation {
    /// Next-Neighboring-Word: tokens are adjacent in same entity
    NNW,
    /// Tail-Head-Word: marks entity boundary (tail -> head)
    THW,
    /// No relation between tokens
    None,
}

impl W2NERRelation {
    /// Convert from label index.
    #[must_use]
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::None,
            1 => Self::NNW,
            2 => Self::THW,
            _ => Self::None,
        }
    }

    /// Convert to label index.
    #[must_use]
    pub fn to_index(self) -> usize {
        match self {
            Self::None => 0,
            Self::NNW => 1,
            Self::THW => 2,
        }
    }
}

/// Configuration for W2NER decoding.
#[derive(Debug, Clone)]
pub struct W2NERConfig {
    /// Confidence threshold for grid predictions
    pub threshold: f64,
    /// Entity type labels (maps grid channels to types)
    pub entity_labels: Vec<String>,
    /// Whether to extract nested entities
    pub allow_nested: bool,
    /// Whether to extract discontinuous entities
    pub allow_discontinuous: bool,
    /// Model identifier for loading
    pub model_id: String,
}

impl Default for W2NERConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            entity_labels: vec![
                "PER".to_string(),
                "ORG".to_string(),
                "LOC".to_string(),
            ],
            allow_nested: true,
            allow_discontinuous: true,
            model_id: String::new(),
        }
    }
}

/// W2NER model for unified named entity recognition.
///
/// Uses word-word relation classification to handle complex entity
/// structures (nested, overlapping, discontinuous).
///
/// # Feature Requirements
///
/// Requires the `onnx` feature for actual inference. Without it, only the
/// [`decode_from_matrix`](Self::decode_from_matrix) method works with
/// pre-computed grids.
///
/// # Example
///
/// ```rust,ignore
/// let w2ner = W2NER::from_pretrained("ljynlp/w2ner-bert-base")?;
///
/// // Handles nested entities naturally
/// let text = "The University of California Berkeley";
/// let entities = w2ner.extract_entities(text, None)?;
/// ```
pub struct W2NER {
    config: W2NERConfig,
    #[cfg(feature = "onnx")]
    session: Option<std::sync::Mutex<ort::session::Session>>,
    #[cfg(feature = "onnx")]
    tokenizer: Option<tokenizers::Tokenizer>,
}

impl W2NER {
    /// Create W2NER with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: W2NERConfig::default(),
            #[cfg(feature = "onnx")]
            session: None,
            #[cfg(feature = "onnx")]
            tokenizer: None,
        }
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(config: W2NERConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "onnx")]
            session: None,
            #[cfg(feature = "onnx")]
            tokenizer: None,
        }
    }

    /// Load W2NER model from path or HuggingFace.
    ///
    /// # Arguments
    /// * `model_path` - Local path or HuggingFace model ID
    #[cfg(feature = "onnx")]
    pub fn from_pretrained(model_path: &str) -> Result<Self> {
        use hf_hub::api::sync::Api;
        use ort::execution_providers::CPUExecutionProvider;
        use ort::session::Session;
        use std::path::Path;

        let (model_file, tokenizer_file) = if Path::new(model_path).exists() {
            // Local path
            let model_file = Path::new(model_path).join("model.onnx");
            let tokenizer_file = Path::new(model_path).join("tokenizer.json");
            (model_file, tokenizer_file)
        } else {
            // HuggingFace download
            let api = Api::new().map_err(|e| {
                Error::Retrieval(format!("Failed to initialize HuggingFace API: {}", e))
            })?;
            let repo = api.model(model_path.to_string());

            let model_file = repo
                .get("model.onnx")
                .or_else(|_| repo.get("onnx/model.onnx"))
                .map_err(|e| Error::Retrieval(format!("Failed to download model: {}", e)))?;

            let tokenizer_file = repo
                .get("tokenizer.json")
                .map_err(|e| Error::Retrieval(format!("Failed to download tokenizer: {}", e)))?;

            (model_file, tokenizer_file)
        };

        let session = Session::builder()
            .map_err(|e| Error::Retrieval(format!("Failed to create session: {}", e)))?
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .map_err(|e| Error::Retrieval(format!("Failed to set providers: {}", e)))?
            .commit_from_file(&model_file)
            .map_err(|e| Error::Retrieval(format!("Failed to load model: {}", e)))?;

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_file)
            .map_err(|e| Error::Retrieval(format!("Failed to load tokenizer: {}", e)))?;

        log::debug!(
            "[W2NER] Loaded model with inputs: {:?}",
            session.inputs.iter().map(|i| &i.name).collect::<Vec<_>>()
        );

        Ok(Self {
            config: W2NERConfig {
                model_id: model_path.to_string(),
                ..Default::default()
            },
            session: Some(std::sync::Mutex::new(session)),
            tokenizer: Some(tokenizer),
        })
    }

    /// Set confidence threshold.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.config.threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set entity type labels.
    #[must_use]
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        self.config.entity_labels = labels;
        self
    }

    /// Enable/disable nested entity extraction.
    #[must_use]
    pub fn with_nested(mut self, allow: bool) -> Self {
        self.config.allow_nested = allow;
        self
    }

    /// Decode entities from a handshaking matrix.
    ///
    /// This is the core W2NER decoding algorithm that can be used with
    /// pre-computed grid predictions (e.g., from external inference).
    ///
    /// # Algorithm
    ///
    /// 1. Find all THW cells (entity boundaries)
    /// 2. For each THW(i,j), the entity spans from word j (head) to word i (tail)
    /// 3. Handle nested/overlapping entities based on config
    ///
    /// # Arguments
    ///
    /// * `matrix` - The predicted word-word relation grid
    /// * `tokens` - Original tokens for text reconstruction
    /// * `entity_type_idx` - Which entity type channel this is
    pub fn decode_from_matrix(
        &self,
        matrix: &HandshakingMatrix,
        tokens: &[&str],
        entity_type_idx: usize,
    ) -> Vec<(usize, usize, f64)> {
        let mut entities = Vec::new();

        // Find all THW (Tail-Head-Word) markers
        // THW at (i,j) means: token i is tail, token j is head
        // Entity spans from j (head/start) to i (tail/end)
        for cell in &matrix.cells {
            let relation = W2NERRelation::from_index(cell.label_idx as usize);
            if relation == W2NERRelation::THW && cell.score >= self.config.threshold as f32 {
                let tail = cell.i as usize;
                let head = cell.j as usize;
                
                // Validate: head <= tail (head is start, tail is end)
                if head <= tail && head < tokens.len() && tail < tokens.len() {
                    entities.push((head, tail + 1, cell.score as f64));
                }
            }
        }

        // Sort by start position, then by length (longer first for nested)
        entities.sort_by(|a, b| {
            a.0.cmp(&b.0).then_with(|| (b.1 - b.0).cmp(&(a.1 - a.0)))
        });

        // Remove nested entities if not allowed
        if !self.config.allow_nested {
            entities = Self::remove_nested(&entities);
        }

        let _ = entity_type_idx; // May be used for multi-type grids
        entities
    }

    /// Decode dense grid output to HandshakingMatrix.
    ///
    /// # Arguments
    /// * `grid` - Dense grid of shape [seq_len, seq_len, num_relations]
    /// * `seq_len` - Sequence length
    /// * `threshold` - Score threshold for sparse representation
    pub fn grid_to_matrix(
        grid: &[f32],
        seq_len: usize,
        num_relations: usize,
        threshold: f32,
    ) -> HandshakingMatrix {
        let mut cells = Vec::new();

        for i in 0..seq_len {
            for j in 0..seq_len {
                for rel in 0..num_relations {
                    let idx = i * seq_len * num_relations + j * num_relations + rel;
                    if let Some(&score) = grid.get(idx) {
                        if score >= threshold && rel > 0 { // rel > 0 excludes "None"
                            cells.push(HandshakingCell {
                                i: i as u32,
                                j: j as u32,
                                label_idx: rel as u16,
                                score,
                            });
                        }
                    }
                }
            }
        }

        HandshakingMatrix {
            cells,
            seq_len,
            num_labels: num_relations,
        }
    }

    /// Remove nested entities (keep outermost only).
    fn remove_nested(entities: &[(usize, usize, f64)]) -> Vec<(usize, usize, f64)> {
        let mut result = Vec::new();
        let mut last_end = 0;

        for &(start, end, score) in entities {
            if start >= last_end {
                result.push((start, end, score));
                last_end = end;
            }
        }

        result
    }

    /// Map label string to EntityType.
    fn map_label(label: &str) -> EntityType {
        match label.to_uppercase().as_str() {
            "PER" | "PERSON" => EntityType::Person,
            "ORG" | "ORGANIZATION" => EntityType::Organization,
            "LOC" | "LOCATION" | "GPE" => EntityType::Location,
            "DATE" => EntityType::Date,
            "TIME" => EntityType::Time,
            "MONEY" => EntityType::Money,
            "PERCENT" => EntityType::Percent,
            "MISC" => EntityType::Other("MISC".to_string()),
            _ => EntityType::Other(label.to_string()),
        }
    }

    /// Run inference with ONNX model.
    #[cfg(feature = "onnx")]
    pub fn extract_with_grid(
        &self,
        text: &str,
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        let session = self.session.as_ref().ok_or_else(|| {
            Error::Retrieval("Model not loaded. Call from_pretrained() first.".to_string())
        })?;

        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            Error::Retrieval("Tokenizer not loaded.".to_string())
        })?;

        // Tokenize
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Ok(vec![]);
        }

        let encoding = tokenizer
            .encode(text.to_string(), true)
            .map_err(|e| Error::Parse(format!("Tokenization failed: {}", e)))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
        let seq_len = input_ids.len();

        // Build tensors
        use ndarray::Array2;
        use ort::value::Tensor;

        let input_ids_arr = Array2::from_shape_vec((1, seq_len), input_ids)
            .map_err(|e| Error::Parse(format!("Array error: {}", e)))?;
        let attention_arr = Array2::from_shape_vec((1, seq_len), attention_mask)
            .map_err(|e| Error::Parse(format!("Array error: {}", e)))?;

        let input_ids_t = Tensor::from_array(input_ids_arr)
            .map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;
        let attention_t = Tensor::from_array(attention_arr)
            .map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;

        // Run inference
        let mut session_guard = session.lock()
            .map_err(|e| Error::Retrieval(format!("Session lock failed: {}", e)))?;

        let outputs = session_guard
            .run(ort::inputs![
                "input_ids" => input_ids_t.into_dyn(),
                "attention_mask" => attention_t.into_dyn(),
            ])
            .map_err(|e| Error::Parse(format!("Inference failed: {}", e)))?;

        // Decode grid output
        let output = outputs.iter().next().map(|(_, v)| v)
            .ok_or_else(|| Error::Parse("No output".to_string()))?;

        let (_, data) = output.try_extract_tensor::<f32>()
            .map_err(|e| Error::Parse(format!("Extract failed: {}", e)))?;
        let grid: Vec<f32> = data.to_vec();

        // Convert grid to matrix and decode
        let num_relations = 3; // None, NNW, THW
        let matrix = Self::grid_to_matrix(&grid, seq_len, num_relations, threshold);

        // Calculate word positions
        let word_positions: Vec<(usize, usize)> = {
            let mut positions = Vec::new();
            let mut pos = 0;
            for word in &words {
                if let Some(start) = text[pos..].find(word) {
                    let abs_start = pos + start;
                    positions.push((abs_start, abs_start + word.len()));
                    pos = abs_start + word.len();
                }
            }
            positions
        };

        // Decode entities for each type
        let mut entities = Vec::new();
        for (type_idx, label) in self.config.entity_labels.iter().enumerate() {
            let spans = self.decode_from_matrix(&matrix, &words.iter().map(|s| *s).collect::<Vec<_>>(), type_idx);
            
            for (start_word, end_word, score) in spans {
                if let (Some(&(start_pos, _)), Some(&(_, end_pos))) = 
                    (word_positions.get(start_word), word_positions.get(end_word.saturating_sub(1))) 
                {
                    if let Some(entity_text) = text.get(start_pos..end_pos) {
                        entities.push(Entity::new(
                            entity_text,
                            Self::map_label(label),
                            start_pos,
                            end_pos,
                            score,
                        ));
                    }
                }
            }
        }

        Ok(entities)
    }
}

impl Default for W2NER {
    fn default() -> Self {
        Self::new()
    }
}

impl Model for W2NER {
    fn extract_entities(&self, text: &str, _language: Option<&str>) -> Result<Vec<Entity>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        #[cfg(feature = "onnx")]
        {
            if self.session.is_some() {
                return self.extract_with_grid(text, self.config.threshold as f32);
            }
        }

        // Without model, return empty
        Ok(Vec::new())
    }

    fn supported_types(&self) -> Vec<EntityType> {
        self.config.entity_labels
            .iter()
            .map(|l| Self::map_label(l))
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
        "w2ner"
    }

    fn description(&self) -> &'static str {
        "W2NER: Unified NER via Word-Word Relation Classification (nested/discontinuous support)"
    }
}

impl DiscontinuousNER for W2NER {
    fn extract_discontinuous(
        &self,
        text: &str,
        entity_types: &[&str],
        threshold: f32,
    ) -> Result<Vec<DiscontinuousEntity>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        #[cfg(feature = "onnx")]
        {
            if self.session.is_some() {
                // W2NER naturally handles discontinuous entities
                // For now, convert regular entities to discontinuous format
                let entities = self.extract_with_grid(text, threshold)?;
                
                return Ok(entities.into_iter().map(|e| {
                    DiscontinuousEntity {
                        fragments: vec![(e.start, e.end)],
                        text: e.text,
                        entity_type: e.entity_type,
                        confidence: e.confidence,
                    }
                }).collect());
            }
        }

        let _ = (entity_types, threshold);
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_w2ner_relation_conversion() {
        assert_eq!(W2NERRelation::from_index(0), W2NERRelation::None);
        assert_eq!(W2NERRelation::from_index(1), W2NERRelation::NNW);
        assert_eq!(W2NERRelation::from_index(2), W2NERRelation::THW);
        
        assert_eq!(W2NERRelation::None.to_index(), 0);
        assert_eq!(W2NERRelation::NNW.to_index(), 1);
        assert_eq!(W2NERRelation::THW.to_index(), 2);
    }

    #[test]
    fn test_w2ner_config_defaults() {
        let config = W2NERConfig::default();
        assert!((config.threshold - 0.5).abs() < f64::EPSILON);
        assert!(config.allow_nested);
        assert!(config.allow_discontinuous);
        assert_eq!(config.entity_labels.len(), 3);
    }

    #[test]
    fn test_decode_simple_entity() {
        let w2ner = W2NER::new();
        let tokens = ["New", "York", "City"];
        
        // THW marker: tail=2, head=0 (entity spans all 3 tokens)
        let matrix = HandshakingMatrix {
            cells: vec![
                HandshakingCell {
                    i: 2, // tail
                    j: 0, // head
                    label_idx: W2NERRelation::THW.to_index() as u16,
                    score: 0.9,
                },
            ],
            seq_len: 3,
            num_labels: 3,
        };

        let entities = w2ner.decode_from_matrix(&matrix, &tokens, 0);
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].0, 0); // start
        assert_eq!(entities[0].1, 3); // end
    }

    #[test]
    fn test_decode_nested_entities() {
        let w2ner = W2NER::with_config(W2NERConfig {
            allow_nested: true,
            ..Default::default()
        });
        
        let tokens = ["University", "of", "California", "Berkeley"];
        
        let matrix = HandshakingMatrix {
            cells: vec![
                // Full entity: tail=3, head=0
                HandshakingCell {
                    i: 3, j: 0,
                    label_idx: W2NERRelation::THW.to_index() as u16,
                    score: 0.95,
                },
                // Nested: tail=2, head=2 (just "California")
                HandshakingCell {
                    i: 2, j: 2,
                    label_idx: W2NERRelation::THW.to_index() as u16,
                    score: 0.85,
                },
            ],
            seq_len: 4,
            num_labels: 3,
        };

        let entities = w2ner.decode_from_matrix(&matrix, &tokens, 0);
        assert_eq!(entities.len(), 2);
    }

    #[test]
    fn test_remove_nested() {
        let entities = vec![
            (0, 4, 0.9),  // outer
            (2, 3, 0.8),  // nested
        ];
        
        let filtered = W2NER::remove_nested(&entities);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0], (0, 4, 0.9));
    }

    #[test]
    fn test_grid_to_matrix() {
        // 3x3 grid with 3 relations (None, NNW, THW)
        let seq_len = 3;
        let num_rels = 3;
        let mut grid = vec![0.0f32; seq_len * seq_len * num_rels];
        
        // Set THW at (2, 0) with score 0.9
        let idx = 2 * seq_len * num_rels + 0 * num_rels + 2; // i=2, j=0, rel=THW
        grid[idx] = 0.9;
        
        let matrix = W2NER::grid_to_matrix(&grid, seq_len, num_rels, 0.5);
        assert_eq!(matrix.cells.len(), 1);
        assert_eq!(matrix.cells[0].i, 2);
        assert_eq!(matrix.cells[0].j, 0);
    }

    #[test]
    fn test_label_mapping() {
        assert_eq!(W2NER::map_label("PER"), EntityType::Person);
        assert_eq!(W2NER::map_label("org"), EntityType::Organization);
        assert_eq!(W2NER::map_label("GPE"), EntityType::Location);
        assert_eq!(W2NER::map_label("CUSTOM"), EntityType::Other("CUSTOM".to_string()));
    }

    #[test]
    fn test_empty_input() {
        let w2ner = W2NER::new();
        let entities = w2ner.extract_entities("", None).unwrap();
        assert!(entities.is_empty());
    }

    #[test]
    fn test_not_available_without_model() {
        let w2ner = W2NER::new();
        assert!(!w2ner.is_available());
    }
}
