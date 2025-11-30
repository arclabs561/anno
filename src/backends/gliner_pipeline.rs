//! Unified GLiNER pipeline with pluggable encoders.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │                    GLiNERPipeline                                 │
//! │                                                                   │
//! │  ┌─────────────┐     ┌──────────────┐     ┌────────────────┐    │
//! │  │   Encoder   │     │  SpanRep     │     │ LateInteraction│    │
//! │  │ (pluggable) │ ──▶ │  Layer       │ ──▶ │ (dot product)  │    │
//! │  │ BERT/Modern │     │              │     │                │    │
//! │  └─────────────┘     └──────────────┘     └────────────────┘    │
//! │        │                    │                     │              │
//! │        │                    │                     ▼              │
//! │  ┌─────▼─────┐        ┌─────▼─────┐        ┌──────────────┐     │
//! │  │ Text      │        │ Span      │        │ Match        │     │
//! │  │ Embeddings│        │ Candidates│        │ Scores       │     │
//! │  └───────────┘        └───────────┘        └──────────────┘     │
//! │                                                   │              │
//! │                             ┌─────────────────────┘              │
//! │                             ▼                                    │
//! │  ┌──────────────────────────────────────────────────────────┐   │
//! │  │ SemanticRegistry (pre-computed label embeddings)         │   │
//! │  │ ["person", "organization", "location", ...]              │   │
//! │  └──────────────────────────────────────────────────────────┘   │
//! │                                                                   │
//! └──────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use anno::backends::gliner_pipeline::{GLiNERPipeline, PipelineConfig};
//!
//! // Create with ModernBERT encoder
//! let pipeline = GLiNERPipeline::builder()
//!     .encoder("answerdotai/ModernBERT-base")
//!     .entity_types(&["person", "organization", "location"])
//!     .threshold(0.5)
//!     .build()?;
//!
//! let entities = pipeline.extract("Steve Jobs founded Apple")?;
//! ```

#![allow(dead_code)]
#![allow(unused_variables)]

use crate::backends::inference::{
    DotProductInteraction, LateInteraction, SemanticRegistry, SpanRepConfig,
    SpanRepresentationLayer,
};
use crate::entity::{generate_span_candidates, RaggedBatch, SpanCandidate};
use crate::{Entity, EntityType, Result};

#[cfg(feature = "candle")]
use crate::backends::encoder_candle::{CandleEncoder, TextEncoder};

use std::sync::Arc;

// =============================================================================
// Pipeline Configuration
// =============================================================================

/// Configuration for GLiNER pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Maximum span width for entity candidates
    pub max_span_width: usize,
    /// Confidence threshold for entity detection
    pub threshold: f32,
    /// Entity types to detect
    pub entity_types: Vec<String>,
    /// Entity type descriptions (for embedding)
    pub entity_descriptions: Vec<String>,
    /// Hidden dimension (from encoder)
    pub hidden_dim: usize,
    /// Whether to use GPU acceleration
    pub use_gpu: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_span_width: 12,
            threshold: 0.5,
            entity_types: vec!["person".into(), "organization".into(), "location".into()],
            entity_descriptions: vec![
                "A named individual human being".into(),
                "A company, institution, or group".into(),
                "A geographical place or region".into(),
            ],
            hidden_dim: 768,
            use_gpu: true,
        }
    }
}

impl PipelineConfig {
    /// Standard NER types (PER, ORG, LOC, MISC)
    pub fn standard_ner() -> Self {
        Self {
            entity_types: vec![
                "person".into(),
                "organization".into(),
                "location".into(),
                "miscellaneous".into(),
            ],
            entity_descriptions: vec![
                "A named individual human being".into(),
                "A company, institution, agency, or group".into(),
                "A geographical place, city, country, or region".into(),
                "Other named entities like events, products, works of art".into(),
            ],
            ..Self::default()
        }
    }

    /// Movie domain entity types
    pub fn movie_domain() -> Self {
        Self {
            entity_types: vec![
                "actor".into(),
                "director".into(),
                "movie_title".into(),
                "genre".into(),
                "year".into(),
            ],
            entity_descriptions: vec![
                "A person who acts in films".into(),
                "A person who directs films".into(),
                "The name of a movie or film".into(),
                "A category of film (action, comedy, drama, etc.)".into(),
                "A year when a film was released".into(),
            ],
            ..Self::default()
        }
    }

    /// Legal domain entity types
    pub fn legal_domain() -> Self {
        Self {
            entity_types: vec![
                "law".into(),
                "court".into(),
                "judge".into(),
                "plaintiff".into(),
                "defendant".into(),
                "case_number".into(),
            ],
            entity_descriptions: vec![
                "A statute, regulation, or legal code".into(),
                "A court of law or judicial body".into(),
                "A judge or justice".into(),
                "The party bringing a lawsuit".into(),
                "The party being sued".into(),
                "A case or docket number".into(),
            ],
            threshold: 0.4, // Lower for legal domain
            ..Self::default()
        }
    }

    /// Biomedical domain entity types
    pub fn biomedical_domain() -> Self {
        Self {
            entity_types: vec![
                "drug".into(),
                "disease".into(),
                "gene".into(),
                "protein".into(),
                "chemical".into(),
            ],
            entity_descriptions: vec![
                "A medication or pharmaceutical compound".into(),
                "A medical condition or illness".into(),
                "A gene or genetic element".into(),
                "A protein or enzyme".into(),
                "A chemical compound or element".into(),
            ],
            threshold: 0.4,
            ..Self::default()
        }
    }
}

// =============================================================================
// Pipeline Builder
// =============================================================================

/// Builder for GLiNER pipeline.
pub struct PipelineBuilder {
    config: PipelineConfig,
    encoder_model: Option<String>,
}

impl PipelineBuilder {
    /// Create a new builder with default config.
    pub fn new() -> Self {
        Self {
            config: PipelineConfig::default(),
            encoder_model: None,
        }
    }

    /// Set the encoder model (HuggingFace ID).
    pub fn encoder(mut self, model_id: &str) -> Self {
        self.encoder_model = Some(model_id.to_string());
        self
    }

    /// Set entity types.
    pub fn entity_types(mut self, types: &[&str]) -> Self {
        self.config.entity_types = types.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Set entity descriptions.
    pub fn entity_descriptions(mut self, descriptions: &[&str]) -> Self {
        self.config.entity_descriptions = descriptions.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Set confidence threshold.
    pub fn threshold(mut self, threshold: f32) -> Self {
        self.config.threshold = threshold;
        self
    }

    /// Set max span width.
    pub fn max_span_width(mut self, width: usize) -> Self {
        self.config.max_span_width = width;
        self
    }

    /// Use a predefined config.
    #[must_use]
    pub fn with_config(mut self, config: PipelineConfig) -> Self {
        self.config = config;
        self
    }

    /// Build the pipeline.
    #[cfg(feature = "candle")]
    pub fn build(self) -> Result<GLiNERPipeline<CandleEncoder>> {
        let encoder_id = self
            .encoder_model
            .unwrap_or_else(|| "answerdotai/ModernBERT-base".to_string());

        // Load encoder
        let encoder = CandleEncoder::from_pretrained(&encoder_id)?;
        let hidden_dim = encoder.hidden_dim();

        // Build registry from entity types
        let mut registry_builder = SemanticRegistry::builder();
        for (type_name, desc) in self
            .config
            .entity_types
            .iter()
            .zip(self.config.entity_descriptions.iter())
        {
            registry_builder = registry_builder.add_entity(type_name, desc);
        }
        let registry = registry_builder.build_placeholder(hidden_dim);

        // Create span representation layer
        let span_config = SpanRepConfig {
            hidden_dim,
            max_width: self.config.max_span_width,
            use_width_embeddings: true,
            width_emb_dim: hidden_dim / 4,
        };
        let span_layer = SpanRepresentationLayer::new(span_config);

        // Create interaction scorer
        let interaction = DotProductInteraction::with_temperature(1.0);

        Ok(GLiNERPipeline {
            encoder: Arc::new(encoder),
            registry,
            span_layer,
            interaction: Box::new(interaction),
            config: PipelineConfig {
                hidden_dim,
                ..self.config
            },
        })
    }

    #[cfg(not(feature = "candle"))]
    pub fn build(self) -> Result<GLiNERPipeline<()>> {
        Err(Error::FeatureNotAvailable(
            "GLiNERPipeline requires 'candle' feature".into(),
        ))
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Pipeline
// =============================================================================

/// GLiNER-style NER pipeline with pluggable encoder.
pub struct GLiNERPipeline<E> {
    /// The encoder (BERT/ModernBERT/etc)
    encoder: Arc<E>,
    /// Pre-computed label embeddings
    registry: SemanticRegistry,
    /// Span representation layer
    span_layer: SpanRepresentationLayer,
    /// Late interaction scorer
    interaction: Box<dyn LateInteraction>,
    /// Pipeline configuration
    config: PipelineConfig,
}

#[cfg(feature = "candle")]
impl<E: TextEncoder> GLiNERPipeline<E> {
    /// Create a builder.
    pub fn builder() -> PipelineBuilder {
        PipelineBuilder::new()
    }

    /// Extract entities from text.
    pub fn extract(&self, text: &str) -> Result<Vec<Entity>> {
        if text.trim().is_empty() {
            return Ok(vec![]);
        }

        // Step 1: Encode text
        let (token_embeddings, seq_len) = self.encoder.encode(text)?;
        let hidden_dim = self.config.hidden_dim;

        // Step 2: Create ragged batch (single document)
        // Generate dummy token IDs for the batch structure
        let dummy_tokens: Vec<u32> = (0..seq_len as u32).collect();
        let batch = RaggedBatch::from_sequences(&[dummy_tokens]);

        // Step 3: Generate span candidates
        let candidates = generate_span_candidates(&batch, self.config.max_span_width);

        if candidates.is_empty() {
            return Ok(vec![]);
        }

        // Step 4: Compute span embeddings
        let span_embeddings = self
            .span_layer
            .forward(&token_embeddings, &candidates, &batch);

        // Step 5: Compute similarity scores via late interaction
        let scores = self.interaction.compute_similarity(
            &span_embeddings,
            candidates.len(),
            &self.registry.embeddings,
            self.registry.len(),
            hidden_dim,
        );

        // Step 6: Apply sigmoid
        let mut scores = scores;
        self.interaction.apply_sigmoid(&mut scores);

        // Step 7: Decode entities
        let entities = self.decode_entities(text, &candidates, &scores, seq_len);

        Ok(entities)
    }

    /// Extract entities with custom types (zero-shot).
    pub fn extract_with_types(
        &self,
        text: &str,
        entity_types: &[&str],
        threshold: Option<f32>,
    ) -> Result<Vec<Entity>> {
        // For true zero-shot, we'd need to encode the new labels
        // For now, filter to configured types
        let threshold = threshold.unwrap_or(self.config.threshold);
        let entities = self.extract(text)?;

        Ok(entities
            .into_iter()
            .filter(|e| {
                let type_name = match &e.entity_type {
                    EntityType::Person => "person",
                    EntityType::Organization => "organization",
                    EntityType::Location => "location",
                    EntityType::Date => "date",
                    EntityType::Money => "money",
                    EntityType::Other(s) => s.as_str(),
                    _ => return false,
                };
                entity_types
                    .iter()
                    .any(|&t| t.eq_ignore_ascii_case(type_name))
            })
            .filter(|e| e.confidence >= threshold as f64)
            .collect())
    }

    /// Get the encoder architecture name.
    pub fn encoder_name(&self) -> &str {
        self.encoder.architecture()
    }

    /// Get the hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        self.config.hidden_dim
    }

    /// Get configured entity types.
    pub fn entity_types(&self) -> &[String] {
        &self.config.entity_types
    }

    /// Decode entities from scores.
    fn decode_entities(
        &self,
        text: &str,
        candidates: &[SpanCandidate],
        scores: &[f32],
        seq_len: usize,
    ) -> Vec<Entity> {
        let num_labels = self.registry.len();
        let mut entities = Vec::new();

        // Split text into words for offset calculation
        let words: Vec<&str> = text.split_whitespace().collect();

        for (span_idx, candidate) in candidates.iter().enumerate() {
            for label_idx in 0..num_labels {
                let score = scores[span_idx * num_labels + label_idx];

                if score >= self.config.threshold {
                    // Get label info
                    let label = &self.registry.labels[label_idx];

                    // Calculate character offsets from word indices
                    let start_word = candidate.start as usize;
                    let end_word = (candidate.end as usize).min(words.len());

                    if start_word >= words.len() || end_word > words.len() {
                        continue;
                    }

                    let (char_start, char_end) = word_indices_to_char_offsets(
                        text,
                        &words,
                        start_word,
                        end_word.saturating_sub(1),
                    );

                    // Extract text using character offsets (not byte offsets)
                    let entity_text: String = text
                        .chars()
                        .skip(char_start)
                        .take(char_end.saturating_sub(char_start))
                        .collect();

                    if entity_text.trim().is_empty() {
                        continue;
                    }

                    let entity_type = slug_to_entity_type(&label.slug);

                    entities.push(Entity::new(
                        entity_text.trim(),
                        entity_type,
                        char_start,
                        char_end,
                        score as f64,
                    ));
                }
            }
        }

        // Sort by position
        entities.sort_by(|a, b| {
            a.start
                .cmp(&b.start)
                .then_with(|| b.end.cmp(&a.end))
                .then_with(|| {
                    b.confidence
                        .partial_cmp(&a.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        // Remove duplicates (keep highest confidence)
        let mut kept = Vec::new();
        for entity in entities {
            let overlaps = kept
                .iter()
                .any(|e: &Entity| !(entity.end <= e.start || e.end <= entity.start));
            if !overlaps {
                kept.push(entity);
            }
        }

        kept
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Convert word indices to character offsets.
///
/// This function correctly handles Unicode text by converting byte offsets
/// to character offsets using the offset module's bytes_to_chars function.
fn word_indices_to_char_offsets(
    text: &str,
    words: &[&str],
    start_word: usize,
    end_word: usize,
) -> (usize, usize) {
    let mut byte_pos = 0;
    let mut start_byte = 0;
    let mut end_byte = text.len();

    for (idx, word) in words.iter().enumerate() {
        // Search for the word in the remaining text (by bytes)
        if let Some(pos) = text[byte_pos..].find(word) {
            let word_start_byte = byte_pos + pos;
            let word_end_byte = word_start_byte + word.len();

            if idx == start_word {
                start_byte = word_start_byte;
            }
            if idx == end_word {
                end_byte = word_end_byte;
                break;
            }
            byte_pos = word_end_byte;
        }
    }

    // Convert byte offsets to character offsets
    crate::offset::bytes_to_chars(text, start_byte, end_byte)
}

/// Convert label slug to EntityType.
fn slug_to_entity_type(slug: &str) -> EntityType {
    match slug.to_lowercase().as_str() {
        "person" | "per" => EntityType::Person,
        "organization" | "org" | "company" => EntityType::Organization,
        "location" | "loc" | "gpe" | "place" => EntityType::Location,
        "date" | "time" => EntityType::Date,
        "money" | "currency" | "amount" => EntityType::Money,
        "percent" | "percentage" => EntityType::Percent,
        other => EntityType::Other(other.to_string()),
    }
}

// =============================================================================
// Model Trait Implementation
// =============================================================================

#[cfg(feature = "candle")]
impl<E: TextEncoder + 'static> crate::Model for GLiNERPipeline<E> {
    fn extract_entities(&self, text: &str, _language: Option<&str>) -> Result<Vec<Entity>> {
        self.extract(text)
    }

    fn supported_types(&self) -> Vec<EntityType> {
        self.config
            .entity_types
            .iter()
            .map(|s| slug_to_entity_type(s))
            .collect()
    }

    fn is_available(&self) -> bool {
        true
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_domains() {
        let standard = PipelineConfig::standard_ner();
        assert!(standard.entity_types.contains(&"person".to_string()));

        let movie = PipelineConfig::movie_domain();
        assert!(movie.entity_types.contains(&"actor".to_string()));
        assert!(movie.entity_types.contains(&"director".to_string()));

        let legal = PipelineConfig::legal_domain();
        assert!(legal.entity_types.contains(&"court".to_string()));
        assert_eq!(legal.threshold, 0.4);

        let bio = PipelineConfig::biomedical_domain();
        assert!(bio.entity_types.contains(&"drug".to_string()));
    }

    #[test]
    fn test_word_to_char_offsets() {
        let text = "Hello world test";
        let words: Vec<&str> = text.split_whitespace().collect();

        let (start, end) = word_indices_to_char_offsets(text, &words, 1, 1);
        assert_eq!(&text[start..end], "world");

        let (start, end) = word_indices_to_char_offsets(text, &words, 0, 2);
        assert_eq!(&text[start..end], "Hello world test");
    }

    #[test]
    fn test_slug_to_entity_type() {
        assert_eq!(slug_to_entity_type("person"), EntityType::Person);
        assert_eq!(slug_to_entity_type("PER"), EntityType::Person);
        assert_eq!(
            slug_to_entity_type("organization"),
            EntityType::Organization
        );
        assert_eq!(slug_to_entity_type("LOC"), EntityType::Location);
        assert!(matches!(slug_to_entity_type("actor"), EntityType::Other(_)));
    }

    #[test]
    fn test_builder_pattern() {
        let builder = PipelineBuilder::new()
            .entity_types(&["person", "org"])
            .threshold(0.6)
            .max_span_width(8);

        assert_eq!(builder.config.entity_types.len(), 2);
        assert_eq!(builder.config.threshold, 0.6);
        assert_eq!(builder.config.max_span_width, 8);
    }
}
