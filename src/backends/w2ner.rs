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
//! This maps to our `HandshakingMatrix` (from TPLinker).
//!
//! # Implementation Status
//!
//! This module provides the W2NER data structures and decoding logic.
//! The actual model inference requires:
//!
//! 1. BERT/DeBERTa encoder (via ONNX or Candle)
//! 2. Biaffine attention layer for grid prediction
//! 3. Grid decoding to extract entity spans
//!
//! # References
//!
//! - [W2NER: Unified Named Entity Recognition as Word-Word Relation Classification](https://arxiv.org/abs/2112.10070)
//! - [TPLinker: Single-stage Joint Extraction of Entities and Relations](https://aclanthology.org/2020.coling-main.138/)

use crate::{Entity, EntityType, Model, Result};
use crate::backends::inference::{DiscontinuousEntity, DiscontinuousNER, HandshakingMatrix};

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
        }
    }
}

/// W2NER model for unified named entity recognition.
///
/// Uses word-word relation classification to handle complex entity
/// structures (nested, overlapping, discontinuous).
///
/// # Example
///
/// ```rust,ignore
/// let w2ner = W2NER::default();
///
/// // Handles nested entities naturally
/// let text = "The University of California Berkeley";
/// let entities = w2ner.extract_entities(text, None)?;
/// // Returns: "University of California Berkeley" (ORG)
/// //          "California" (LOC) - nested
/// ```
pub struct W2NER {
    config: W2NERConfig,
}

impl W2NER {
    /// Create W2NER with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: W2NERConfig::default(),
        }
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(config: W2NERConfig) -> Self {
        Self { config }
    }

    /// Set confidence threshold.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.config.threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Decode entities from a handshaking matrix.
    ///
    /// This is the core W2NER decoding algorithm:
    /// 1. Find all THW cells (entity boundaries)
    /// 2. For each THW(i,j), trace back via NNW to find complete span
    /// 3. Handle nested/overlapping entities if configured
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
        let _entity_label = self.config.entity_labels
            .get(entity_type_idx)
            .cloned()
            .unwrap_or_default();

        // Find all THW (Tail-Head-Word) markers - these indicate entity boundaries
        // THW at (i,j) means: token i is tail, token j is head of an entity
        for cell in &matrix.cells {
            let relation = W2NERRelation::from_index(cell.label_idx as usize);
            if relation == W2NERRelation::THW && cell.score >= self.config.threshold as f32 {
                let tail = cell.i as usize;
                let head = cell.j as usize;
                
                // Validate: tail should be >= head (head is start, tail is end)
                if head <= tail && head < tokens.len() && tail < tokens.len() {
                    entities.push((head, tail + 1, cell.score as f64));
                }
            }
        }

        // Sort by start position, then by length (longer first for nested)
        entities.sort_by(|a, b| {
            a.0.cmp(&b.0).then_with(|| (b.1 - b.0).cmp(&(a.1 - a.0)))
        });

        // Remove duplicates and handle nesting
        if !self.config.allow_nested {
            entities = Self::remove_nested(&entities);
        }

        entities
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
            _ => EntityType::Other(label.to_string()),
        }
    }
}

impl Default for W2NER {
    fn default() -> Self {
        Self::new()
    }
}

impl Model for W2NER {
    fn extract_entities(&self, text: &str, _language: Option<&str>) -> Result<Vec<Entity>> {
        // W2NER requires:
        // 1. Tokenization
        // 2. BERT encoding
        // 3. Biaffine attention for grid prediction
        // 4. Grid decoding (implemented above)
        //
        // This is a placeholder - actual inference requires model files.
        
        if text.is_empty() {
            return Ok(Vec::new());
        }

        // Placeholder: The actual implementation would:
        // 1. tokenize(text) -> tokens
        // 2. encode(tokens) -> embeddings
        // 3. biaffine(embeddings) -> grid scores
        // 4. HandshakingMatrix::from_dense(&scores, seq_len, num_labels, threshold)
        // 5. decode_from_matrix(&matrix, &tokens, entity_type_idx)

        Ok(Vec::new())
    }

    fn supported_types(&self) -> Vec<EntityType> {
        self.config.entity_labels
            .iter()
            .map(|l| Self::map_label(l))
            .collect()
    }

    fn is_available(&self) -> bool {
        // W2NER requires model files
        false
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

        // W2NER naturally handles discontinuous entities through the word-word grid.
        // A discontinuous entity like "severe [pain] ... in [abdomen]" would have:
        // - THW markers for each fragment
        // - NNW links connecting fragments that belong to the same entity
        //
        // Full implementation would:
        // 1. Run model to get the word-word relation grid
        // 2. Find entity fragments via THW markers
        // 3. Group fragments via cross-fragment NNW/THW links
        // 4. Return DiscontinuousEntity for multi-fragment entities

        // Placeholder: return empty since we need actual model inference
        // to produce real discontinuous entities
        let _ = (entity_types, threshold); // Suppress unused warnings

        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::inference::HandshakingCell;

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
        
        // Simulate THW marker: tail=2, head=0 (entity spans all 3 tokens)
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
        
        // Two entities: full span and nested "California"
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
}

