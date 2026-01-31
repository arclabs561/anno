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
use crate::{Entity, EntityType, Result};
use anno_core::{generate_span_candidates, RaggedBatch, SpanCandidate};

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

        // Build registry from entity types using the SAME encoder (bi-encoder baseline).
        //
        // This replaces the previous placeholder (all-zero) label embeddings, which produced
        // meaningless similarity scores.
        let entity_types: Vec<&str> = self
            .config
            .entity_types
            .iter()
            .map(|s| s.as_str())
            .collect();
        let entity_descs: Vec<&str> = self
            .config
            .entity_descriptions
            .iter()
            .map(|s| s.as_str())
            .collect();
        let registry = build_registry_from_encoder(
            &encoder,
            &entity_types,
            &entity_descs,
            self.config.threshold,
        )?;

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
/// `TextEncoder` that can also return token byte offsets.
///
/// This is used by `GLiNERPipeline` to decode token spans into character offsets safely.
pub trait TextEncoderWithOffsets: TextEncoder {
    /// Encode text into token embeddings and return byte offsets for each token.
    ///
    /// # Returns
    /// - Token embeddings: `[seq_len, hidden_dim]` (flattened)
    /// - Sequence length
    /// - Token byte offsets: `(byte_start, byte_end)` for each token
    fn encode_with_offsets(&self, text: &str) -> Result<(Vec<f32>, usize, Vec<(usize, usize)>)>;
}

#[cfg(feature = "candle")]
impl TextEncoderWithOffsets for CandleEncoder {
    fn encode_with_offsets(&self, text: &str) -> Result<(Vec<f32>, usize, Vec<(usize, usize)>)> {
        CandleEncoder::encode_with_offsets(self, text)
    }
}

#[cfg(feature = "candle")]
impl<E: TextEncoderWithOffsets> GLiNERPipeline<E> {
    /// Create a builder.
    pub fn builder() -> PipelineBuilder {
        PipelineBuilder::new()
    }

    /// Extract entities from text.
    pub fn extract(&self, text: &str) -> Result<Vec<Entity>> {
        self.extract_with_registry(text, &self.registry, self.config.threshold)
    }

    /// Extract entities with custom types (zero-shot).
    pub fn extract_with_types(
        &self,
        text: &str,
        entity_types: &[&str],
        threshold: Option<f32>,
    ) -> Result<Vec<Entity>> {
        let threshold = threshold.unwrap_or(self.config.threshold);
        let registry =
            build_registry_from_encoder(self.encoder.as_ref(), entity_types, &[], threshold)?;
        self.extract_with_registry(text, &registry, threshold)
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

    fn extract_with_registry(
        &self,
        text: &str,
        registry: &SemanticRegistry,
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        if text.trim().is_empty() {
            return Ok(vec![]);
        }

        if registry.is_empty() {
            return Err(crate::Error::InvalidInput(
                "GLiNERPipeline requires at least one entity type label".to_string(),
            ));
        }

        // Step 1: Encode text (token embeddings + byte offsets)
        let (token_embeddings, seq_len, token_offsets) = self.encoder.encode_with_offsets(text)?;
        let hidden_dim = self.config.hidden_dim;

        if seq_len == 0 || token_offsets.is_empty() {
            return Ok(vec![]);
        }

        // Step 2: Create ragged batch (single document)
        // Generate dummy token IDs for the batch structure (only lengths matter).
        let dummy_tokens: Vec<u32> = (0..seq_len as u32).collect();
        let batch = RaggedBatch::from_sequences(&[dummy_tokens]);

        // Step 3: Generate span candidates (token-level, end-exclusive)
        let candidates = generate_span_candidates(&batch, self.config.max_span_width);

        if candidates.is_empty() {
            return Err(crate::Error::Inference(
                "GLiNERPipeline produced no span candidates for non-empty text. This suggests an encoder/tokenization failure (seq_len=0) or an invalid max_span_width configuration.".to_string(),
            ));
        }

        // Step 4: Compute span embeddings
        let span_embeddings = self
            .span_layer
            .forward(&token_embeddings, &candidates, &batch);

        if span_embeddings.is_empty() {
            return Ok(vec![]);
        }

        // Step 5: Compute similarity scores via late interaction
        let scores = self.interaction.compute_similarity(
            &span_embeddings,
            candidates.len(),
            &registry.embeddings,
            registry.len(),
            hidden_dim,
        );

        // Step 6: Apply sigmoid
        let mut scores = scores;
        self.interaction.apply_sigmoid(&mut scores);

        // Step 7: Decode entities (token offsets -> character offsets)
        let entities = self.decode_entities(
            text,
            &candidates,
            &scores,
            &token_offsets,
            registry,
            threshold,
        );

        Ok(entities)
    }

    /// Decode entities from scores.
    fn decode_entities(
        &self,
        text: &str,
        candidates: &[SpanCandidate],
        scores: &[f32],
        token_offsets: &[(usize, usize)],
        registry: &SemanticRegistry,
        threshold: f32,
    ) -> Vec<Entity> {
        let num_labels = registry.len();
        let mut entities = Vec::new();

        // Convert byte offsets to character offsets safely.
        let converter = crate::offset::SpanConverter::new(text);

        for (span_idx, candidate) in candidates.iter().enumerate() {
            for label_idx in 0..num_labels {
                let score = scores[span_idx * num_labels + label_idx];

                // Get label info
                let label = &registry.labels[label_idx];
                let min_threshold = threshold.max(label.threshold);

                if score >= min_threshold {
                    // Get label info
                    let start_tok = candidate.start as usize;
                    let end_tok = candidate.end as usize;
                    if end_tok <= start_tok {
                        continue;
                    }

                    if start_tok >= token_offsets.len() || end_tok > token_offsets.len() {
                        continue;
                    }

                    // Map token span -> byte span.
                    let (byte_start, _) = token_offsets[start_tok];
                    let (_, byte_end) = token_offsets[end_tok - 1];

                    // Skip special tokens / empty offsets.
                    if byte_end <= byte_start {
                        continue;
                    }

                    let char_start = converter.byte_to_char(byte_start);
                    let char_end = converter.byte_to_char(byte_end);

                    // Extract surface text (byte slice is safe here; offsets come from tokenizers).
                    let entity_text = match text.get(byte_start..byte_end) {
                        Some(s) => s.trim(),
                        None => continue,
                    };

                    if entity_text.is_empty() {
                        continue;
                    }

                    let entity_type = slug_to_entity_type(&label.slug);

                    entities.push(Entity::new(
                        entity_text,
                        entity_type,
                        char_start,
                        char_end,
                        score as f64,
                    ));
                }
            }
        }

        // Performance: Use unstable sort (we don't need stable sort here)
        // Sort by position
        entities.sort_unstable_by(|a, b| {
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
        // Two entities overlap if: entity.start < e.end && e.start < entity.end
        let mut kept = Vec::new();
        for entity in entities {
            let has_overlap = kept
                .iter()
                .any(|e: &Entity| entity.start < e.end && e.start < entity.end);
            if !has_overlap {
                kept.push(entity);
            }
        }

        kept
    }
}

// =============================================================================
// Helpers
// =============================================================================

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

#[cfg(feature = "candle")]
fn build_registry_from_encoder<E: TextEncoder>(
    encoder: &E,
    entity_types: &[&str],
    entity_descriptions: &[&str],
    threshold: f32,
) -> Result<SemanticRegistry> {
    let hidden_dim = encoder.hidden_dim();
    if entity_types.is_empty() {
        return Ok(SemanticRegistry {
            embeddings: Vec::new(),
            hidden_dim,
            labels: Vec::new(),
            label_index: std::collections::HashMap::new(),
        });
    }

    let mut labels: Vec<crate::backends::inference::LabelDefinition> =
        Vec::with_capacity(entity_types.len());
    for (i, slug) in entity_types.iter().enumerate() {
        let desc = entity_descriptions.get(i).copied().unwrap_or(*slug);
        labels.push(crate::backends::inference::LabelDefinition {
            slug: (*slug).to_string(),
            description: desc.to_string(),
            category: crate::backends::inference::LabelCategory::Entity,
            modality: crate::backends::inference::ModalityHint::TextOnly,
            threshold,
        });
    }

    let label_index: std::collections::HashMap<String, usize> = labels
        .iter()
        .enumerate()
        .map(|(i, l)| (l.slug.clone(), i))
        .collect();

    // Encode each label description and mean-pool token embeddings.
    let mut embeddings = Vec::with_capacity(labels.len() * hidden_dim);
    for label in &labels {
        let (emb, seq_len) = encoder.encode(&label.description)?;
        embeddings.extend(mean_pool_embeddings(&emb, seq_len, hidden_dim));
    }

    Ok(SemanticRegistry {
        embeddings,
        hidden_dim,
        labels,
        label_index,
    })
}

fn mean_pool_embeddings(embeddings: &[f32], seq_len: usize, hidden_dim: usize) -> Vec<f32> {
    // Defensive: handle unexpected shapes.
    if seq_len == 0 || embeddings.len() < seq_len.saturating_mul(hidden_dim) {
        return vec![0.0f32; hidden_dim];
    }

    // Heuristic: tokenizers typically include [CLS] ... [SEP] when `encode(..., true)` is used.
    // Exclude those when present.
    let start = if seq_len > 2 { 1 } else { 0 };
    let end = if seq_len > 2 { seq_len - 1 } else { seq_len };
    let count = (end - start).max(1);

    let mut pooled = vec![0.0f32; hidden_dim];
    for t in start..end {
        let base = t * hidden_dim;
        for h in 0..hidden_dim {
            pooled[h] += embeddings[base + h];
        }
    }

    let denom = count as f32;
    for h in 0..hidden_dim {
        pooled[h] /= denom;
    }
    pooled
}

// =============================================================================
// Model Trait Implementation
// =============================================================================

#[cfg(feature = "candle")]
impl<E: TextEncoderWithOffsets + 'static> crate::Model for GLiNERPipeline<E> {
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
    fn test_mean_pool_embeddings_excludes_special_tokens_when_present() {
        // 4 tokens, hidden_dim=2:
        // tok0=[CLS], tok1=a, tok2=b, tok3=[SEP]
        let hidden_dim = 2;
        let seq_len = 4;
        let embeddings = vec![
            // tok0
            100.0, 100.0, // tok1
            1.0, 2.0, // tok2
            3.0, 4.0, // tok3
            200.0, 200.0,
        ];
        let pooled = mean_pool_embeddings(&embeddings, seq_len, hidden_dim);
        assert_eq!(pooled, vec![2.0, 3.0]); // mean of tok1 and tok2
    }

    #[test]
    fn test_mean_pool_embeddings_short_sequences_include_all_tokens() {
        let hidden_dim = 2;
        let seq_len = 2;
        let embeddings = vec![
            // tok0
            1.0, 3.0, // tok1
            5.0, 7.0,
        ];
        let pooled = mean_pool_embeddings(&embeddings, seq_len, hidden_dim);
        assert_eq!(pooled, vec![3.0, 5.0]);
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
