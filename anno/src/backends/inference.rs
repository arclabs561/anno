//! Core inference abstractions aligned with bleeding-edge NER research.
//!
//! # Why Traditional NER Fails at New Entity Types
//!
//! Traditional NER uses a **classifier head**: for each token, output one of N fixed labels.
//!
//! ```text
//! TRADITIONAL NER (Token Classification)
//! ──────────────────────────────────────
//!
//! Input: "Steve Jobs founded Apple"
//!
//!         [Steve] [Jobs] [founded] [Apple]
//!            │       │       │        │
//!            ▼       ▼       ▼        ▼
//!      ┌──────────────────────────────────┐
//!      │        BERT / Transformer        │
//!      └──────────────────────────────────┘
//!            │       │       │        │
//!            ▼       ▼       ▼        ▼
//!      ┌──────────────────────────────────┐
//!      │   Classification Head (fixed!)   │
//!      │                                  │
//!      │   Weights: [W_PER, W_ORG, W_LOC] │ ← Trained once, frozen forever
//!      └──────────────────────────────────┘
//!            │       │       │        │
//!            ▼       ▼       ▼        ▼
//!         B-PER   I-PER     O      B-ORG
//!
//! PROBLEM: Want to add DISEASE type? Must retrain the entire model.
//!          The weights W_DISEASE don't exist!
//! ```
//!
//! # How Bi-Encoder Enables Zero-Shot
//!
//! Modern NER (GLiNER, UniversalNER) uses a **matching architecture**:
//!
//! ```text
//! BI-ENCODER NER (Matching, not Classification)
//! ─────────────────────────────────────────────
//!
//! Key insight: Don't classify into fixed labels.
//!              Instead, MATCH text spans to label descriptions.
//!
//! ┌──────────────────────────────┐     ┌──────────────────────────────┐
//! │       TEXT ENCODER           │     │       LABEL ENCODER          │
//! │       (ModernBERT)           │     │       (BGE-small)            │
//! │                              │     │                              │
//! │  "Steve Jobs" ──► [768 dims] │     │ "person" ──► [768 dims]      │
//! │                              │     │ "company" ──► [768 dims]     │
//! │  "Apple" ──► [768 dims]      │     │ "disease" ──► [768 dims]     │
//! │                              │     │                              │
//! └───────────────┬──────────────┘     └───────────────┬──────────────┘
//!                 │                                    │
//!                 │         COSINE SIMILARITY          │
//!                 └────────────►◄───────────────────────┘
//!                              │
//!                              ▼
//!                    ┌───────────────────┐
//!                    │  Similarity Matrix │
//!                    │                    │
//!                    │         person company disease │
//!                    │ "Steve"   0.91   0.12   0.05  │
//!                    │ "Apple"   0.08   0.89   0.03  │
//!                    └───────────────────┘
//!                              │
//!                              ▼
//!                       Threshold (0.5)
//!                              │
//!                              ▼
//!                 Steve Jobs → PERSON ✓
//!                 Apple → COMPANY ✓
//!
//! WHY THIS ENABLES ZERO-SHOT:
//! ───────────────────────────
//!
//! The label encoder never saw "disease" during training!
//! It just learned to embed ANY text description.
//!
//! At inference time, you can add:
//!   • "disease" ──► [768 dims]
//!   • "pharmaceutical compound" ──► [768 dims]
//!   • "19th century French philosopher" ──► [768 dims]
//!
//! The model matches text spans to these NEW descriptions
//! using general semantic similarity. No retraining needed.
//! ```
//!
//! # The Trade-off
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │              BI-ENCODER vs CLASSIFIER HEAD                          │
//! ├───────────────────┬─────────────────────┬───────────────────────────┤
//! │ Aspect            │ Classifier Head     │ Bi-Encoder                │
//! ├───────────────────┼─────────────────────┼───────────────────────────┤
//! │ Entity types      │ Fixed at training   │ Unlimited (zero-shot!)    │
//! │ Inference speed   │ Fast (single pass)  │ Slower (encode all labels)│
//! │ Type confusion    │ Low (learned)       │ Higher (semantic overlap) │
//! │ Generalization    │ Poor to new domains │ Excellent                 │
//! │ Memory            │ Low                 │ Higher (label embeddings) │
//! └───────────────────┴─────────────────────┴───────────────────────────┘
//!
//! WHEN TO USE EACH:
//!
//! • Fixed types, high throughput → Traditional classifier
//! • Custom types, flexibility → Bi-encoder (GLiNER)
//! ```
//!
//! # Overview
//!
//! This module provides the foundational abstractions for building modern NER
//! systems that can:
//!
//! - Extract **any entity type** without retraining (zero-shot)
//! - Handle **nested and discontinuous** entities
//! - Extract **relations** between entities for knowledge graphs
//! - Process **visual documents** as well as plain text
//!
//! This module implements these key insights:
//!
//! | Concept | Description | Research |
//! |---------|-------------|----------|
//! | Late Interaction | Bi-encoder span-label matching | GLiNER |
//! | Semantic Registry | Pre-computed label embeddings | GLiNER bi-encoder |
//! | Unpadded Batches | Memory-efficient ragged tensors | ModernBERT |
//! | Handshaking Matrix | Word-word relation classification | W2NER, TPLinker |
//! | Modality Agnostic | Text and visual in same space | ColPali |
//!
//! # Core Traits
//!
//! | Trait | Purpose | Example Implementation |
//! |-------|---------|------------------------|
//! | [`TextEncoder`] | Encode text → embeddings | ModernBERT, DeBERTaV3 |
//! | [`LabelEncoder`] | Encode labels → embeddings | BGE-small, Sentence-T5 |
//! | [`BiEncoder`] | Combine text + label encoding | GLiNER |
//! | [`LateInteraction`] | Compute span-label similarity | DotProduct, MaxSim |
//! | [`ZeroShotNER`] | Extract arbitrary entity types | GLiNER, UniversalNER |
//! | [`RelationExtractor`] | Joint entity + relation | W2NER, TPLinker |
//! | [`DiscontinuousNER`] | Handle non-contiguous spans | W2NER |
//!
//! # Architecture
//!
//! ```text
//! Input (Text/Image)
//!       │
//!       ▼
//! ┌─────────────────┐
//! │  TextEncoder    │  ModernBERT / DeBERTaV3 / ColPali
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │  RaggedBatch    │  Unpadded token embeddings
//! │  \[total_tokens\] │  + cumulative sequence offsets
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────────────────────────────────────┐
//! │              LateInteraction                     │
//! │                                                  │
//! │  SpanCandidates  ←─────→  SemanticRegistry      │
//! │  [num_spans, H]   cosine  [num_labels, H]       │
//! │                    sim                          │
//! │        scores = σ(spans @ labels.T / τ)         │
//! └────────────────────┬────────────────────────────┘
//!                      │
//!                      ▼
//!                 Entities + Relations
//! ```
//!
//! # Example Usage
//!
//! ```ignore
//! use anno::{SemanticRegistry, DotProductInteraction, ZeroShotNER};
//!
//! // 1. Build a semantic registry with custom entity types
//! let registry = SemanticRegistry::builder()
//!     .add_entity("drug", "A pharmaceutical compound or medication")
//!     .add_entity("disease", "A medical condition or illness")
//!     .add_relation("TREATS", "Drug is used to treat disease")
//!     .build(&label_encoder)?;
//!
//! // 2. Create the zero-shot NER model
//! let ner = GLiNER::new("knowledgator/modern-gliner-bi-base-v1.0")?;
//!
//! // 3. Extract entities using the registry
//! let text = "Aspirin is commonly used to treat headaches.";
//! let entities = ner.extract_with_registry(text, &registry, 0.5)?;
//!
//! // 4. Build knowledge graph from extractions
//! for e in entities {
//!     println!("{}: {}", e.entity_type, e.text);
//! }
//! ```
//!
//! # Training Insights (GLiNER Ablations)
//!
//! From the GLiNER paper ablations:
//! - **Negative entity sampling**: 50% negative entities is optimal. 0% causes
//!   excessive false positives; 75% causes missed entities.
//! - **Entity type dropping**: Randomly varying prompt count improves
//!   out-of-domain generalization by ~1.4 F1 points.
//! - **Max span width K=12**: Keeps O(N) complexity without harming recall.
//!
//! # Research References
//!
//! ## Core Architecture Papers
//! - **GLiNER**: arXiv:2311.08526 - "GLiNER: Generalist Model for NER" (NAACL 2024)
//! - **ModernBERT**: arXiv:2412.13663 - "Smarter, Better, Faster, Longer" (Dec 2024)
//! - **W2NER**: arXiv:2112.10070 - "Word-Word Relation Classification" (AAAI 2022)
//! - **UniversalNER**: arXiv:2308.03279 - "Universal NER" (ICLR 2024)
//! - **ColPali**: arXiv:2407.01449 - "Efficient Document Retrieval"
//!
//! ## Bleeding Edge (2025)
//! - **ReasoningNER**: arXiv:2511.11978 - Chain-of-thought NER with GRPO, F1=85.2
//! - **CMAS**: arXiv:2502.18702 - Multi-agent zero-shot NER
//! - **NER Retriever**: arXiv:2509.04011 - Type-aware retrieval for ad-hoc NER
//! - **BioClinical ModernBERT**: Domain-adapted encoder for medical NER (SOTA)
//!
//! ## Evaluation
//! - **TMR Metrics**: arXiv:2103.12312 - "Tough Mentions Recall for NER"
//! - **SeqScore**: arXiv:2107.14154 - "Reproducible NER Evaluation"
//! - **Familiarity**: arXiv:2412.10121 - Label overlap bias in zero-shot eval

use std::borrow::Cow;
use std::collections::HashMap;

use crate::{Entity, EntityType};
use anno_core::{RaggedBatch, Relation, SpanCandidate};

// =============================================================================
// Modality Types
// =============================================================================

/// Input modality for the encoder.
///
/// Supports text, images, and hybrid (OCR + visual) inputs.
/// This enables ColPali-style visual document understanding.
#[derive(Debug, Clone)]
pub enum ModalityInput<'a> {
    /// Plain text input
    Text(Cow<'a, str>),
    /// Image bytes (PNG/JPEG)
    Image {
        /// Raw image bytes
        data: Cow<'a, [u8]>,
        /// Image format hint
        format: ImageFormat,
    },
    /// Hybrid: text with visual location (e.g., OCR result)
    Hybrid {
        /// Extracted text
        text: Cow<'a, str>,
        /// Visual bounding boxes for each token/word
        visual_positions: Vec<VisualPosition>,
    },
}

/// Image format hint for decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageFormat {
    /// PNG format
    #[default]
    Png,
    /// JPEG format
    Jpeg,
    /// WebP format
    Webp,
    /// Unknown/auto-detect
    Unknown,
}

/// Visual position of a text token in an image.
#[derive(Debug, Clone, Copy)]
pub struct VisualPosition {
    /// Token/word index
    pub token_idx: u32,
    /// Normalized x coordinate (0.0-1.0)
    pub x: f32,
    /// Normalized y coordinate (0.0-1.0)
    pub y: f32,
    /// Normalized width (0.0-1.0)
    pub width: f32,
    /// Normalized height (0.0-1.0)
    pub height: f32,
    /// Page number (for multi-page documents)
    pub page: u32,
}

// =============================================================================
// Semantic Registry (Pre-computed Label Embeddings)
// =============================================================================

/// A frozen, pre-computed registry of entity and relation types.
///
/// # Motivation
///
/// The `SemanticRegistry` is the "knowledge base" of a bi-encoder NER system.
/// It stores pre-computed embeddings for all entity/relation types, enabling:
///
/// - **Zero-shot**: Add new types without retraining
/// - **Speed**: Encode labels once, reuse forever
/// - **Semantics**: Rich descriptions enable better matching
///
/// # Architecture
///
/// ```text
/// ┌────────────────────────────────────────────────────────────────┐
/// │                     SemanticRegistry                           │
/// ├────────────────────────────────────────────────────────────────┤
/// │  labels: [                                                     │
/// │    { slug: "person", description: "named individual human" }   │
/// │    { slug: "organization", description: "company or group" }   │
/// │    { slug: "CEO_OF", description: "leads organization" }       │
/// │  ]                                                             │
/// │                                                                │
/// │  embeddings: [768 floats] [768 floats] [768 floats]            │
/// │              └────┬────┘  └────┬────┘  └────┬────┘             │
/// │                   ▲            ▲            ▲                  │
/// │              person        organization   CEO_OF               │
/// │                                                                │
/// │  label_index: { "person" → 0, "organization" → 1, ... }        │
/// └────────────────────────────────────────────────────────────────┘
/// ```
///
/// # Bi-Encoder Efficiency
///
/// The key insight from GLiNER is that label embeddings can be computed once
/// and reused across all inference requests:
///
/// | Approach | Cost per query | Benefit |
/// |----------|----------------|---------|
/// | Cross-encoder | O(N × L) | Better accuracy |
/// | Bi-encoder | O(N) + O(L) | Much faster, labels cached |
///
/// # Example
///
/// ```ignore
/// use anno::SemanticRegistry;
///
/// // Build registry (expensive, do once at startup)
/// let registry = SemanticRegistry::builder()
///     .add_entity("person", "A named individual human being")
///     .add_entity("organization", "A company, institution, or organized group")
///     .add_relation("CEO_OF", "Chief executive officer of an organization")
///     .build(&label_encoder)?;
///
/// // Use registry for all inference (cheap, cached embeddings)
/// for document in documents {
///     let entities = engine.extract(&document, &registry)?;
/// }
/// ```
///
/// # Adding Custom Types
///
/// ```ignore
/// // Domain-specific medical entities
/// let medical_registry = SemanticRegistry::builder()
///     .add_entity("drug", "A pharmaceutical compound or medication")
///     .add_entity("disease", "A medical condition or illness")
///     .add_entity("gene", "A genetic sequence encoding a protein")
///     .add_relation("TREATS", "Drug is used to treat disease")
///     .add_relation("CAUSES", "Factor causes or leads to condition")
///     .build(&label_encoder)?;
/// ```
#[derive(Debug, Clone)]
pub struct SemanticRegistry {
    /// Pre-computed embeddings for all labels.
    /// Shape: [num_labels, hidden_dim]
    /// Stored as flattened f32 for simplicity without tensor deps.
    pub embeddings: Vec<f32>,
    /// Hidden dimension of embeddings
    pub hidden_dim: usize,
    /// Metadata for each label (index corresponds to embedding row)
    pub labels: Vec<LabelDefinition>,
    /// Index mapping from label slug to embedding row
    pub label_index: HashMap<String, usize>,
}

/// Definition of a semantic label (entity type or relation type).
#[derive(Debug, Clone)]
pub struct LabelDefinition {
    /// Unique identifier (e.g., "person", "CEO_OF")
    pub slug: String,
    /// Human-readable description (used for encoding)
    pub description: String,
    /// Category: Entity or Relation
    pub category: LabelCategory,
    /// Expected source modality
    pub modality: ModalityHint,
    /// Minimum confidence threshold for this label
    pub threshold: f32,
}

/// Category of semantic label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LabelCategory {
    /// Named entity (Person, Organization, Location, etc.)
    Entity,
    /// Relation between entities (CEO_OF, LOCATED_IN, etc.)
    Relation,
    /// Attribute of an entity (date of birth, revenue, etc.)
    Attribute,
}

/// Hint for which modality this label applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ModalityHint {
    /// Text-only (most entity types)
    #[default]
    TextOnly,
    /// Visual-only (e.g., logos, signatures)
    VisualOnly,
    /// Works with both text and visual
    Any,
}

impl SemanticRegistry {
    /// Create a builder for constructing a registry.
    pub fn builder() -> SemanticRegistryBuilder {
        SemanticRegistryBuilder::new()
    }

    /// Get number of labels in the registry.
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Get embedding for a label by slug.
    pub fn get_embedding(&self, slug: &str) -> Option<&[f32]> {
        let idx = self.label_index.get(slug)?;
        let start = idx * self.hidden_dim;
        let end = start + self.hidden_dim;
        if end <= self.embeddings.len() {
            Some(&self.embeddings[start..end])
        } else {
            None
        }
    }

    /// Get all entity labels (excluding relations).
    pub fn entity_labels(&self) -> impl Iterator<Item = &LabelDefinition> {
        self.labels
            .iter()
            .filter(|l| l.category == LabelCategory::Entity)
    }

    /// Get all relation labels.
    pub fn relation_labels(&self) -> impl Iterator<Item = &LabelDefinition> {
        self.labels
            .iter()
            .filter(|l| l.category == LabelCategory::Relation)
    }

    /// Create a standard NER registry with common entity types.
    pub fn standard_ner(hidden_dim: usize) -> Self {
        // Placeholder embeddings - in real use, these would be encoder outputs
        let labels = vec![
            LabelDefinition {
                slug: "person".into(),
                description: "A named individual human being".into(),
                category: LabelCategory::Entity,
                modality: ModalityHint::TextOnly,
                threshold: 0.5,
            },
            LabelDefinition {
                slug: "organization".into(),
                description: "A company, institution, agency, or other group".into(),
                category: LabelCategory::Entity,
                modality: ModalityHint::TextOnly,
                threshold: 0.5,
            },
            LabelDefinition {
                slug: "location".into(),
                description: "A geographical place, city, country, or region".into(),
                category: LabelCategory::Entity,
                modality: ModalityHint::TextOnly,
                threshold: 0.5,
            },
            LabelDefinition {
                slug: "date".into(),
                description: "A calendar date or time expression".into(),
                category: LabelCategory::Entity,
                modality: ModalityHint::TextOnly,
                threshold: 0.5,
            },
            LabelDefinition {
                slug: "money".into(),
                description: "A monetary amount with currency".into(),
                category: LabelCategory::Entity,
                modality: ModalityHint::TextOnly,
                threshold: 0.5,
            },
        ];

        let num_labels = labels.len();
        let label_index: HashMap<String, usize> = labels
            .iter()
            .enumerate()
            .map(|(i, l)| (l.slug.clone(), i))
            .collect();

        // Initialize with zeros (placeholder)
        let embeddings = vec![0.0f32; num_labels * hidden_dim];

        Self {
            embeddings,
            hidden_dim,
            labels,
            label_index,
        }
    }
}

/// Builder for SemanticRegistry.
#[derive(Debug, Default)]
pub struct SemanticRegistryBuilder {
    labels: Vec<LabelDefinition>,
}

impl SemanticRegistryBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an entity type.
    pub fn add_entity(mut self, slug: &str, description: &str) -> Self {
        self.labels.push(LabelDefinition {
            slug: slug.into(),
            description: description.into(),
            category: LabelCategory::Entity,
            modality: ModalityHint::TextOnly,
            threshold: 0.5,
        });
        self
    }

    /// Add a relation type.
    pub fn add_relation(mut self, slug: &str, description: &str) -> Self {
        self.labels.push(LabelDefinition {
            slug: slug.into(),
            description: description.into(),
            category: LabelCategory::Relation,
            modality: ModalityHint::TextOnly,
            threshold: 0.5,
        });
        self
    }

    /// Add a label with full configuration.
    pub fn add_label(mut self, label: LabelDefinition) -> Self {
        self.labels.push(label);
        self
    }

    /// Build the registry (placeholder - real impl needs encoder).
    pub fn build_placeholder(self, hidden_dim: usize) -> SemanticRegistry {
        let num_labels = self.labels.len();
        let label_index: HashMap<String, usize> = self
            .labels
            .iter()
            .enumerate()
            .map(|(i, l)| (l.slug.clone(), i))
            .collect();

        SemanticRegistry {
            embeddings: vec![0.0f32; num_labels * hidden_dim],
            hidden_dim,
            labels: self.labels,
            label_index,
        }
    }
}

// =============================================================================
// Core Encoder Traits (GLiNER/ModernBERT Alignment)
// =============================================================================

/// Text encoder trait for transformer-based encoders.
///
/// # Motivation
///
/// Modern NER systems require converting raw text into dense vector representations
/// that capture semantic meaning. This trait abstracts the encoding step, allowing
/// different transformer architectures to be used interchangeably.
///
/// # Supported Architectures
///
/// | Architecture | Context | Key Features | Speed |
/// |--------------|---------|--------------|-------|
/// | ModernBERT   | 8,192   | RoPE, GeGLU, unpadded inference | 3x faster |
/// | DeBERTaV3    | 512     | Disentangled attention | Baseline |
/// | BERT/RoBERTa | 512     | Classic, widely available | Baseline |
///
/// # Research Alignment (ModernBERT, Dec 2024)
///
/// From ModernBERT paper (arXiv:2412.13663):
/// > "Pareto improvements to BERT... encoder-only models offer great
/// > performance-size tradeoff for retrieval and classification."
///
/// Key innovations:
/// - **Alternating Attention**: Global attention every 3 layers, local (128-token
///   window) elsewhere. Reduces complexity for long sequences.
/// - **Unpadding**: "ModernBERT unpads inputs *before* the token embedding layer
///   and optionally repads model outputs leading to a 10-to-20 percent
///   performance improvement over previous methods."
/// - **RoPE**: Rotary positional embeddings enable extrapolation to longer sequences.
/// - **GeGLU**: Gated activation function improves over GELU.
///
/// # Example
///
/// ```ignore
/// use anno::TextEncoder;
///
/// fn process_document(encoder: &dyn TextEncoder, text: &str) {
///     let output = encoder.encode(text).unwrap();
///     println!("Encoded {} tokens into {} dimensions",
///              output.num_tokens, output.hidden_dim);
///     
///     // Token offsets map back to character positions
///     for (i, (start, end)) in output.token_offsets.iter().enumerate() {
///         println!("Token {}: chars {}..{}", i, start, end);
///     }
/// }
/// ```
pub trait TextEncoder: Send + Sync {
    /// Encode text into token embeddings.
    ///
    /// # Arguments
    /// * `text` - Input text to encode
    ///
    /// # Returns
    /// * Token embeddings as flattened [num_tokens, hidden_dim]
    /// * Attention mask indicating valid tokens
    fn encode(&self, text: &str) -> crate::Result<EncoderOutput>;

    /// Encode a batch of texts.
    ///
    /// # Arguments
    /// * `texts` - Batch of input texts
    ///
    /// # Returns
    /// * RaggedBatch containing all embeddings with document boundaries
    fn encode_batch(&self, texts: &[&str]) -> crate::Result<(Vec<f32>, RaggedBatch)>;

    /// Get the hidden dimension of the encoder.
    fn hidden_dim(&self) -> usize;

    /// Get the maximum sequence length.
    fn max_length(&self) -> usize;

    /// Get the encoder architecture name.
    fn architecture(&self) -> &'static str;
}

/// Output from text encoding.
#[derive(Debug, Clone)]
pub struct EncoderOutput {
    /// Token embeddings: [num_tokens, hidden_dim]
    pub embeddings: Vec<f32>,
    /// Number of tokens
    pub num_tokens: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Token-to-character mapping (for span recovery)
    pub token_offsets: Vec<(usize, usize)>,
}

/// Label encoder trait for encoding entity type descriptions.
///
/// # Motivation
///
/// Zero-shot NER works by encoding entity type *descriptions* into the same
/// vector space as text spans. Instead of training separate classifiers for
/// each entity type, we compute similarity between spans and label embeddings.
///
/// This enables:
/// - **Unlimited entity types** at inference (no retraining needed)
/// - **Faster inference** when labels are pre-computed
/// - **Better generalization** to unseen entity types via semantic similarity
///
/// # Research Alignment
///
/// From GLiNER bi-encoder (knowledgator/modern-gliner-bi-base-v1.0):
/// > "textual encoder is ModernBERT-base and entity label encoder is
/// > sentence transformer - BGE-small-en."
///
/// # Example
///
/// ```ignore
/// use anno::LabelEncoder;
///
/// fn setup_custom_types(encoder: &dyn LabelEncoder) {
///     // Encode rich descriptions for better matching
///     let labels = &[
///         "a named individual human being",
///         "a company, institution, or organized group",
///         "a geographical location, city, country, or region",
///     ];
///     
///     let embeddings = encoder.encode_labels(labels).unwrap();
///     // Store embeddings in SemanticRegistry for fast lookup
/// }
/// ```
pub trait LabelEncoder: Send + Sync {
    /// Encode a single label description.
    ///
    /// # Arguments
    /// * `label` - Label description (e.g., "a named individual human being")
    fn encode_label(&self, label: &str) -> crate::Result<Vec<f32>>;

    /// Encode multiple labels.
    ///
    /// # Arguments
    /// * `labels` - Label descriptions
    ///
    /// # Returns
    /// Flattened embeddings: [num_labels, hidden_dim]
    fn encode_labels(&self, labels: &[&str]) -> crate::Result<Vec<f32>>;

    /// Get the hidden dimension.
    fn hidden_dim(&self) -> usize;
}

/// Bi-encoder architecture combining text and label encoders.
///
/// # Motivation
///
/// The bi-encoder architecture treats NER as a **matching problem** rather than
/// a classification problem. It encodes text spans and entity labels separately,
/// then computes similarity scores to determine matches.
///
/// ```text
/// ┌─────────────────┐         ┌─────────────────┐
/// │   Text Input    │         │  Label Desc.    │
/// │ "Steve Jobs"    │         │ "person name"   │
/// └────────┬────────┘         └────────┬────────┘
///          │                           │
///          ▼                           ▼
/// ┌─────────────────┐         ┌─────────────────┐
/// │  TextEncoder    │         │  LabelEncoder   │
/// │  (ModernBERT)   │         │  (BGE-small)    │
/// └────────┬────────┘         └────────┬────────┘
///          │                           │
///          ▼                           ▼
/// ┌─────────────────┐         ┌─────────────────┐
/// │ Span Embedding  │◄───────►│ Label Embedding │
/// │   [768]         │ cosine  │   [768]         │
/// └─────────────────┘ sim     └─────────────────┘
///                      │
///                      ▼
///               Score: 0.92
/// ```
///
/// # Trade-offs
///
/// | Aspect | Bi-Encoder | Uni-Encoder |
/// |--------|------------|-------------|
/// | Entity types | Unlimited | Fixed at training |
/// | Inference speed | Faster (pre-compute labels) | Slower |
/// | Disambiguation | Harder (no label interaction) | Better |
/// | Generalization | Better to new types | Limited |
///
/// # Research Alignment
///
/// From GLiNER: "GLiNER frames NER as a matching problem, comparing candidate
/// spans with entity type embeddings."
///
/// From knowledgator: "Bi-encoder architecture brings several advantages...
/// unlimited entities, faster inference, better generalization."
///
/// Drawback: "Lack of inter-label interactions that make it hard to
/// disambiguate semantically similar but contextually different entities."
///
/// # Example
///
/// ```ignore
/// use anno::BiEncoder;
///
/// fn extract_custom_entities(bi_enc: &dyn BiEncoder, text: &str) {
///     let labels = &["software company", "hardware manufacturer", "person"];
///     let scores = bi_enc.encode_and_match(text, labels, 8).unwrap();
///     
///     for s in scores.iter().filter(|s| s.score > 0.5) {
///         println!("Found '{}' as type {} (score: {:.2})",
///                  &text[s.start..s.end], labels[s.label_idx], s.score);
///     }
/// }
/// ```
pub trait BiEncoder: Send + Sync {
    /// Get the text encoder.
    fn text_encoder(&self) -> &dyn TextEncoder;

    /// Get the label encoder.
    fn label_encoder(&self) -> &dyn LabelEncoder;

    /// Encode text and labels, compute span-label similarities.
    ///
    /// # Arguments
    /// * `text` - Input text
    /// * `labels` - Entity type descriptions
    /// * `max_span_width` - Maximum span width to consider
    ///
    /// # Returns
    /// Similarity scores for each (span, label) pair
    fn encode_and_match(
        &self,
        text: &str,
        labels: &[&str],
        max_span_width: usize,
    ) -> crate::Result<Vec<SpanLabelScore>>;
}

/// Score for a (span, label) match.
#[derive(Debug, Clone)]
pub struct SpanLabelScore {
    /// Span start (character offset)
    pub start: usize,
    /// Span end (character offset, exclusive)
    pub end: usize,
    /// Label index
    pub label_idx: usize,
    /// Similarity score (0.0 - 1.0)
    pub score: f32,
}

// =============================================================================
// Zero-Shot NER Trait
// =============================================================================

/// Zero-shot NER for open entity types.
///
/// # Motivation
///
/// Traditional NER models are trained on fixed taxonomies (PER, ORG, LOC, etc.)
/// and cannot extract new entity types without retraining. Zero-shot NER solves
/// this by allowing **arbitrary entity types at inference time**.
///
/// Instead of asking "is this a PERSON?", zero-shot NER asks "does this text
/// span match the description 'a named individual human being'?"
///
/// # Use Cases
///
/// - **Domain adaptation**: Extract "gene names" or "legal citations" without
///   training data
/// - **Custom taxonomies**: Use your own entity hierarchy
/// - **Rapid prototyping**: Test new entity types before investing in annotation
///
/// # Research Alignment
///
/// From GLiNER (arXiv:2311.08526):
/// > "NER model capable of identifying any entity type using a bidirectional
/// > transformer encoder... provides a practical alternative to traditional
/// > NER models, which are limited to predefined entity types."
///
/// From UniversalNER (arXiv:2308.03279):
/// > "Large language models demonstrate remarkable generalizability, such as
/// > understanding arbitrary entities and relations."
///
/// # Example
///
/// ```ignore
/// use anno::ZeroShotNER;
///
/// fn extract_medical_entities(ner: &dyn ZeroShotNER, clinical_note: &str) {
///     // Define custom medical entity types at runtime
///     let types = &["drug name", "disease", "symptom", "dosage"];
///     
///     let entities = ner.extract_with_types(clinical_note, types, 0.5).unwrap();
///     for e in entities {
///         println!("{}: {} (conf: {:.2})", e.entity_type, e.text, e.confidence);
///     }
/// }
///
/// fn extract_with_descriptions(ner: &dyn ZeroShotNER, text: &str) {
///     // Even richer: use natural language descriptions
///     let descriptions = &[
///         "a medication or pharmaceutical compound",
///         "a medical condition or illness",
///         "a physical sensation indicating illness",
///     ];
///     
///     let entities = ner.extract_with_descriptions(text, descriptions, 0.5).unwrap();
/// }
/// ```
pub trait ZeroShotNER: Send + Sync {
    /// Extract entities with custom types.
    ///
    /// # Arguments
    /// * `text` - Input text
    /// * `entity_types` - Entity type descriptions (arbitrary text, not fixed vocabulary)
    ///   - Encoded as text embeddings via bi-encoder (semantic matching, not exact string match)
    ///   - Any string works: `"disease"`, `"pharmaceutical compound"`, `"19th century French philosopher"`
    ///   - **Replaces default types completely** - model only extracts the specified types
    ///   - To include defaults, pass them explicitly: `&["person", "organization", "disease"]`
    /// * `threshold` - Confidence threshold (0.0 - 1.0)
    ///
    /// # Returns
    /// Entities with their matched types
    ///
    /// # Behavior
    ///
    /// - **Arbitrary text**: Type hints are not fixed vocabulary. They're encoded as embeddings,
    ///   so semantic similarity determines matches (not exact string matching).
    /// - **Replace, don't union**: This method completely replaces default entity types.
    ///   The model only extracts the types you specify.
    /// - **Semantic matching**: Uses cosine similarity between text span embeddings and label embeddings.
    fn extract_with_types(
        &self,
        text: &str,
        entity_types: &[&str],
        threshold: f32,
    ) -> crate::Result<Vec<Entity>>;

    /// Extract entities with natural language descriptions.
    ///
    /// # Arguments
    /// * `text` - Input text
    /// * `descriptions` - Natural language descriptions of what to extract
    ///   - Encoded as text embeddings (same as `extract_with_types`)
    ///   - Examples: `"companies headquartered in Europe"`, `"diseases affecting the heart"`
    ///   - **Replaces default types completely** - model only extracts the specified descriptions
    /// * `threshold` - Confidence threshold
    ///
    /// # Behavior
    ///
    /// Same as `extract_with_types`, but accepts natural language descriptions instead of
    /// short type labels. Both methods encode labels as embeddings and use semantic matching.
    fn extract_with_descriptions(
        &self,
        text: &str,
        descriptions: &[&str],
        threshold: f32,
    ) -> crate::Result<Vec<Entity>>;

    /// Get default entity types for this model.
    ///
    /// Returns the entity types used by `extract_entities()` (via `Model` trait).
    /// Useful for extending defaults: combine with custom types and pass to `extract_with_types()`.
    ///
    /// # Example: Extending defaults
    ///
    /// ```ignore
    /// use anno::ZeroShotNER;
    ///
    /// let ner: &dyn ZeroShotNER = ...;
    /// let defaults = ner.default_types();
    ///
    /// // Combine defaults with custom types
    /// let mut types: Vec<&str> = defaults.to_vec();
    /// types.extend(&["disease", "medication"]);
    ///
    /// let entities = ner.extract_with_types(text, &types, 0.5)?;
    /// ```
    fn default_types(&self) -> &[&'static str];
}

// =============================================================================
// Relation Extractor Trait
// =============================================================================

/// Joint entity and relation extraction.
///
/// # Motivation
///
/// Real-world information extraction often requires both entities AND their
/// relationships. For example, extracting "Steve Jobs" and "Apple" is useful,
/// but knowing "Steve Jobs FOUNDED Apple" is far more valuable.
///
/// Joint extraction (vs pipeline) is preferred because:
/// - **Error propagation**: Pipeline errors compound (bad entities → bad relations)
/// - **Shared context**: Entities and relations inform each other
/// - **Efficiency**: Single forward pass instead of two
///
/// # Architecture
///
/// ```text
/// Input: "Steve Jobs founded Apple in 1976."
///                │
///                ▼
/// ┌──────────────────────────────────┐
/// │     Shared Encoder (BERT)        │
/// └──────────────────────────────────┘
///                │
///         ┌──────┴──────┐
///         ▼             ▼
/// ┌───────────────┐  ┌───────────────┐
/// │ Entity Head   │  │ Relation Head │
/// │ (span class.) │  │ (pair class.) │
/// └───────┬───────┘  └───────┬───────┘
///         │                  │
///         ▼                  ▼
/// Entities:              Relations:
/// - Steve Jobs [PER]     - (Steve Jobs, FOUNDED, Apple)
/// - Apple [ORG]          - (Apple, FOUNDED_IN, 1976)
/// - 1976 [DATE]
/// ```
///
/// # Research Alignment
///
/// From GLiNER multi-task (arXiv:2406.12925):
/// > "Generalist Lightweight Model for Various Information Extraction Tasks...
/// > joint entity and relation extraction."
///
/// From W2NER (arXiv:2112.10070):
/// > "Unified Named Entity Recognition as Word-Word Relation Classification...
/// > handles flat, overlapped, and discontinuous NER."
///
/// # Example
///
/// ```ignore
/// use anno::RelationExtractor;
///
/// fn build_knowledge_graph(extractor: &dyn RelationExtractor, text: &str) {
///     let entity_types = &["person", "organization", "date"];
///     let relation_types = &["founded", "works_for", "acquired"];
///     
///     let result = extractor.extract_with_relations(
///         text, entity_types, relation_types, 0.5
///     ).unwrap();
///     
///     // Build graph nodes from entities
///     for e in &result.entities {
///         println!("Node: {} ({})", e.text, e.entity_type);
///     }
///     
///     // Build graph edges from relations
///     for r in &result.relations {
///         let head = &result.entities[r.head_idx];
///         let tail = &result.entities[r.tail_idx];
///         println!("Edge: {} --[{}]--> {}", head.text, r.relation_type, tail.text);
///     }
/// }
/// ```
pub trait RelationExtractor: Send + Sync {
    /// Extract entities and relations jointly.
    ///
    /// # Arguments
    /// * `text` - Input text
    /// * `entity_types` - Entity types to extract
    /// * `relation_types` - Relation types to extract
    /// * `threshold` - Confidence threshold
    ///
    /// # Returns
    /// Entities and relations between them
    fn extract_with_relations(
        &self,
        text: &str,
        entity_types: &[&str],
        relation_types: &[&str],
        threshold: f32,
    ) -> crate::Result<ExtractionWithRelations>;
}

/// Output from joint entity-relation extraction.
#[derive(Debug, Clone, Default)]
pub struct ExtractionWithRelations {
    /// Extracted entities
    pub entities: Vec<Entity>,
    /// Relations between entities (indices into entities vec)
    pub relations: Vec<RelationTriple>,
}

/// A relation triple linking two entities.
#[derive(Debug, Clone)]
pub struct RelationTriple {
    /// Index of head entity in entities vec
    pub head_idx: usize,
    /// Index of tail entity in entities vec
    pub tail_idx: usize,
    /// Relation type
    pub relation_type: String,
    /// Confidence score
    pub confidence: f32,
}

// =============================================================================
// Discontinuous Entity Support (W2NER Research)
// =============================================================================

/// Support for discontinuous entity spans.
///
/// # Motivation
///
/// Not all entities are contiguous text spans. In coordination structures,
/// entities can be **discontinuous** - scattered across non-adjacent positions.
///
/// # Examples of Discontinuous Entities
///
/// ```text
/// "New York and Los Angeles airports"
///  ^^^^^^^^     ^^^^^^^^^^^ ^^^^^^^^
///  └──────────────────────────┘
///     LOCATION: "New York airports" (discontinuous!)
///                ^^^^^^^^^^^ ^^^^^^^^
///                └───────────┘
///                LOCATION: "Los Angeles airports" (contiguous)
///
/// "protein A and B complex"
///  ^^^^^^^^^ ^^^ ^^^^^^^^^
///  └────────────────────┘
///     PROTEIN: "protein A ... complex" (discontinuous!)
/// ```
///
/// # NER Complexity Hierarchy
///
/// | Type | Description | Example |
/// |------|-------------|---------|
/// | Flat | Non-overlapping spans | "John works at Google" |
/// | Nested | Overlapping spans | "\[New \[York\] City\]" |
/// | Discontinuous | Non-contiguous | "New York and LA \[airports\]" |
///
/// # Research Alignment
///
/// From W2NER (arXiv:2112.10070):
/// > "Named entity recognition has been involved with three major types,
/// > including flat, overlapped (aka. nested), and discontinuous NER...
/// > we propose a novel architecture to model NER as word-word relation
/// > classification."
///
/// W2NER achieves this by building a **handshaking matrix** where each cell
/// (i, j) indicates whether tokens i and j are part of the same entity.
///
/// # Example
///
/// ```ignore
/// use anno::DiscontinuousNER;
///
/// fn extract_complex_entities(ner: &dyn DiscontinuousNER, text: &str) {
///     let types = &["location", "protein"];
///     let entities = ner.extract_discontinuous(text, types, 0.5).unwrap();
///     
///     for e in entities {
///         if e.is_contiguous() {
///             println!("Contiguous {}: '{}'", e.entity_type, e.text);
///         } else {
///             println!("Discontinuous {}: '{}' spans: {:?}",
///                      e.entity_type, e.text, e.spans);
///         }
///     }
/// }
/// ```
pub trait DiscontinuousNER: Send + Sync {
    /// Extract entities including discontinuous spans.
    ///
    /// # Arguments
    /// * `text` - Input text
    /// * `entity_types` - Entity types to extract
    /// * `threshold` - Confidence threshold
    ///
    /// # Returns
    /// Entities, potentially with multiple non-contiguous spans
    fn extract_discontinuous(
        &self,
        text: &str,
        entity_types: &[&str],
        threshold: f32,
    ) -> crate::Result<Vec<DiscontinuousEntity>>;
}

/// An entity that may span multiple non-contiguous regions.
#[derive(Debug, Clone)]
pub struct DiscontinuousEntity {
    /// The spans that make up this entity (may be non-contiguous)
    pub spans: Vec<(usize, usize)>,
    /// Concatenated text from all spans
    pub text: String,
    /// Entity type
    pub entity_type: String,
    /// Confidence score
    pub confidence: f32,
}

impl DiscontinuousEntity {
    /// Check if this entity is contiguous (single span).
    pub fn is_contiguous(&self) -> bool {
        self.spans.len() == 1
    }

    /// Convert to a standard Entity if contiguous.
    pub fn to_entity(&self) -> Option<Entity> {
        if self.is_contiguous() {
            let (start, end) = self.spans[0];
            Some(Entity::new(
                self.text.clone(),
                EntityType::from_label(&self.entity_type),
                start,
                end,
                self.confidence as f64,
            ))
        } else {
            None
        }
    }
}

// =============================================================================
// Late Interaction Trait
// =============================================================================

/// The core abstraction for bi-encoder NER scoring.
///
/// # Motivation
///
/// "Late interaction" refers to when the text and label representations
/// interact: at the very end of the pipeline, after both have been
/// independently encoded. This is in contrast to "early fusion" where
/// text and labels are concatenated before encoding.
///
/// ```text
///                      Early Fusion             Late Interaction
///                      ────────────             ────────────────
///
/// Encode:          [text + label]              text    label
///                        │                       │       │
///                        ▼                       ▼       ▼
///                    Encoder                  Enc_T   Enc_L
///                        │                       │       │
///                        ▼                       ▼       ▼
///                    Score                   emb_t   emb_l
///                                                │       │
///                                                └───┬───┘
///                                                    ▼
///                                              dot(emb_t, emb_l)
/// ```
///
/// Late interaction enables:
/// - Pre-computing label embeddings (major speedup)
/// - Adding new labels without re-encoding text
/// - Parallelizing text and label encoding
///
/// # The Math
///
/// ```text
/// Score(span, label) = σ(span_emb · label_emb / τ)
///
/// where:
///   σ = sigmoid activation
///   · = dot product
///   τ = temperature (sharpness parameter)
/// ```
///
/// # Implementations
///
/// | Interaction | Formula | Speed | Accuracy | Use Case |
/// |-------------|---------|-------|----------|----------|
/// | DotProduct  | s·l     | Fast  | Good     | General purpose |
/// | MaxSim      | max(s·l)| Medium| Better   | Multi-token labels |
/// | Bilinear    | s·W·l   | Slow  | Best     | When accuracy critical |
///
/// # Example
///
/// ```ignore
/// use anno::{LateInteraction, DotProductInteraction};
///
/// let interaction = DotProductInteraction::with_temperature(20.0);
///
/// // Span embeddings: 3 spans × 768 dim
/// let span_embs: Vec<f32> = get_span_embeddings(&tokens, &candidates);
///
/// // Label embeddings: 5 labels × 768 dim  
/// let label_embs: Vec<f32> = registry.all_embeddings();
///
/// // Compute 3×5 = 15 similarity scores
/// let mut scores = interaction.compute_similarity(
///     &span_embs, 3, &label_embs, 5, 768
/// );
/// interaction.apply_sigmoid(&mut scores);
///
/// // scores[i*5 + j] = similarity between span i and label j
/// ```
pub trait LateInteraction: Send + Sync {
    /// Compute similarity scores between span and label embeddings.
    ///
    /// # Arguments
    /// * `span_embeddings` - Shape: [num_spans, hidden_dim]
    /// * `label_embeddings` - Shape: [num_labels, hidden_dim]
    ///
    /// # Returns
    /// Similarity matrix of shape: [num_spans, num_labels]
    fn compute_similarity(
        &self,
        span_embeddings: &[f32],
        num_spans: usize,
        label_embeddings: &[f32],
        num_labels: usize,
        hidden_dim: usize,
    ) -> Vec<f32>;

    /// Apply sigmoid activation to scores.
    fn apply_sigmoid(&self, scores: &mut [f32]) {
        for s in scores.iter_mut() {
            *s = 1.0 / (1.0 + (-*s).exp());
        }
    }
}

/// Dot product interaction (default, fast).
#[derive(Debug, Clone, Copy, Default)]
pub struct DotProductInteraction {
    /// Temperature scaling (higher = sharper distribution)
    pub temperature: f32,
}

impl DotProductInteraction {
    /// Create with default temperature (1.0).
    pub fn new() -> Self {
        Self { temperature: 1.0 }
    }

    /// Create with custom temperature.
    #[must_use]
    pub fn with_temperature(temperature: f32) -> Self {
        Self { temperature }
    }
}

impl LateInteraction for DotProductInteraction {
    fn compute_similarity(
        &self,
        span_embeddings: &[f32],
        num_spans: usize,
        label_embeddings: &[f32],
        num_labels: usize,
        hidden_dim: usize,
    ) -> Vec<f32> {
        let mut scores = vec![0.0f32; num_spans * num_labels];

        for s in 0..num_spans {
            let span_start = s * hidden_dim;
            let span_end = span_start + hidden_dim;
            let span_vec = &span_embeddings[span_start..span_end];

            for l in 0..num_labels {
                let label_start = l * hidden_dim;
                let label_end = label_start + hidden_dim;
                let label_vec = &label_embeddings[label_start..label_end];

                // Dot product
                let mut dot: f32 = span_vec
                    .iter()
                    .zip(label_vec.iter())
                    .map(|(a, b)| a * b)
                    .sum();

                // Temperature scaling
                dot *= self.temperature;

                scores[s * num_labels + l] = dot;
            }
        }

        scores
    }
}

/// MaxSim interaction (ColBERT-style, better for phrases).
#[derive(Debug, Clone, Copy, Default)]
pub struct MaxSimInteraction {
    /// Temperature scaling
    pub temperature: f32,
}

impl MaxSimInteraction {
    /// Create with default settings.
    pub fn new() -> Self {
        Self { temperature: 1.0 }
    }
}

impl LateInteraction for MaxSimInteraction {
    fn compute_similarity(
        &self,
        span_embeddings: &[f32],
        num_spans: usize,
        label_embeddings: &[f32],
        num_labels: usize,
        hidden_dim: usize,
    ) -> Vec<f32> {
        // For single-vector embeddings, MaxSim degrades to dot product
        // True MaxSim requires multi-vector representations
        DotProductInteraction::new().compute_similarity(
            span_embeddings,
            num_spans,
            label_embeddings,
            num_labels,
            hidden_dim,
        )
    }
}

// =============================================================================
// Span Representation
// =============================================================================

/// Configuration for span representation.
///
/// # Research Context (Deep Span Representations, arXiv:2210.04182)
///
/// From "Deep Span Representations for NER":
/// > "Existing span-based NER systems **shallowly aggregate** the token
/// > representations to span representations. However, this typically results
/// > in significant ineffectiveness for **long-span entities**."
///
/// Common span representation strategies:
///
/// | Method | Formula | Pros | Cons |
/// |--------|---------|------|------|
/// | Concat | [h_i; h_j] | Simple, fast | Ignores middle tokens |
/// | Pooling | mean(h_i:h_j) | Uses all tokens | Loses boundary info |
/// | Attention | attn(h_i:h_j) | Learnable | Expensive |
/// | GLiNER | FFN([h_i; h_j; w]) | Balanced | Requires width emb |
///
/// # Recommendation (GLiNER Default)
///
/// For most use cases, concatenating first + last token embeddings with
/// a width embedding provides the best tradeoff:
/// - O(N) complexity (vs O(N²) for all-pairs attention)
/// - Captures boundary positions (critical for NER)
/// - Width embedding disambiguates "I" vs "New York City"
#[derive(Debug, Clone)]
pub struct SpanRepConfig {
    /// Hidden dimension of the encoder
    pub hidden_dim: usize,
    /// Maximum span width (in tokens)
    ///
    /// GLiNER uses K=12: "to keep linear complexity without harming recall."
    /// Wider spans rarely contain coherent entities.
    pub max_width: usize,
    /// Whether to include width embeddings
    ///
    /// Critical for distinguishing spans of different lengths
    /// with similar boundary tokens.
    pub use_width_embeddings: bool,
    /// Width embedding dimension (typically hidden_dim / 4)
    pub width_emb_dim: usize,
}

impl Default for SpanRepConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 768,
            max_width: 12,
            use_width_embeddings: true,
            width_emb_dim: 192, // 768 / 4
        }
    }
}

/// Computes span representations from token embeddings.
///
/// # Research Alignment (GLiNER, NAACL 2024)
///
/// From the GLiNER paper (arXiv:2311.08526):
/// > "The representation of a span starting at position i and ending at
/// > position j in the input text, S_ij ∈ R^D, is computed as:
/// > **S_ij = FFN(h_i ⊗ h_j)**
/// > where FFN denotes a two-layer feedforward network, and ⊗ represents
/// > the concatenation operation."
///
/// The paper also notes:
/// > "We set an upper bound to the length (K=12) of the span in order to
/// > keep linear complexity in the size of the input text, without harming recall."
///
/// # Span Representation Formula
///
/// ```text
/// span_emb = FFN(Concat(token[i], token[j], width_emb[j-i]))
///          = W_2 · ReLU(W_1 · [h_i; h_j; w_{j-i}] + b_1) + b_2
/// ```
///
/// where:
/// - h_i = start token embedding
/// - h_j = end token embedding  
/// - w_{j-i} = learned width embedding (captures span length)
///
/// This is the "gnarly bit" from GLiNER that enables zero-shot matching.
///
/// # Alternative: Global Pointer (arXiv:2208.03054)
///
/// Instead of enumerating spans, Global Pointer uses RoPE (rotary position
/// embeddings) to predict (start, end) pairs simultaneously:
///
/// ```text
/// score(i, j) = q_i^T * k_j    (where q, k have RoPE applied)
/// ```
///
/// Advantages:
/// - No explicit span enumeration needed
/// - Naturally handles nested entities
/// - More parameter-efficient
///
/// GLiNER-style enumeration is still preferred for zero-shot because
/// it allows pre-computing label embeddings.
pub struct SpanRepresentationLayer {
    /// Configuration
    pub config: SpanRepConfig,
    /// Projection weights: [input_dim, hidden_dim]
    pub projection_weights: Vec<f32>,
    /// Projection bias: \[hidden_dim\]
    pub projection_bias: Vec<f32>,
    /// Width embeddings: [max_width, width_emb_dim]
    pub width_embeddings: Vec<f32>,
}

impl SpanRepresentationLayer {
    /// Create a new span representation layer with random initialization.
    pub fn new(config: SpanRepConfig) -> Self {
        let input_dim = config.hidden_dim * 2 + config.width_emb_dim;

        Self {
            projection_weights: vec![0.0f32; input_dim * config.hidden_dim],
            projection_bias: vec![0.0f32; config.hidden_dim],
            width_embeddings: vec![0.0f32; config.max_width * config.width_emb_dim],
            config,
        }
    }

    /// Compute span representations from token embeddings.
    ///
    /// # Arguments
    /// * `token_embeddings` - Flattened [num_tokens, hidden_dim]
    /// * `candidates` - Span candidates with start/end indices
    ///
    /// # Returns
    /// Span embeddings: [num_candidates, hidden_dim]
    pub fn forward(
        &self,
        token_embeddings: &[f32],
        candidates: &[SpanCandidate],
        batch: &RaggedBatch,
    ) -> Vec<f32> {
        let hidden_dim = self.config.hidden_dim;
        let width_emb_dim = self.config.width_emb_dim;
        let max_width = self.config.max_width;

        // Check for overflow in allocation
        let total_elements = match candidates.len().checked_mul(hidden_dim) {
            Some(v) => v,
            None => {
                log::warn!(
                    "Span embedding allocation overflow: {} candidates * {} hidden_dim, returning empty",
                    candidates.len(), hidden_dim
                );
                return vec![];
            }
        };
        let mut span_embeddings = vec![0.0f32; total_elements];

        for (span_idx, candidate) in candidates.iter().enumerate() {
            // Get document token range
            let doc_range = match batch.doc_range(candidate.doc_idx as usize) {
                Some(r) => r,
                None => continue,
            };

            // Validate span before computing global indices
            if candidate.end <= candidate.start {
                log::warn!(
                    "Invalid span candidate: end ({}) <= start ({})",
                    candidate.end,
                    candidate.start
                );
                continue;
            }

            // Global token indices
            let start_global = doc_range.start + candidate.start as usize;
            let end_global = doc_range.start + (candidate.end as usize) - 1; // Safe now that we validated

            // Bounds check - must ensure both start and end slices fit
            // Use checked arithmetic to prevent overflow
            let start_byte = match start_global.checked_mul(hidden_dim) {
                Some(v) => v,
                None => {
                    log::warn!(
                        "Token index overflow: start_global={} * hidden_dim={}",
                        start_global,
                        hidden_dim
                    );
                    continue;
                }
            };
            let start_end_byte = match (start_global + 1).checked_mul(hidden_dim) {
                Some(v) => v,
                None => {
                    log::warn!(
                        "Token index overflow: (start_global+1)={} * hidden_dim={}",
                        start_global + 1,
                        hidden_dim
                    );
                    continue;
                }
            };
            let end_byte = match end_global.checked_mul(hidden_dim) {
                Some(v) => v,
                None => {
                    log::warn!(
                        "Token index overflow: end_global={} * hidden_dim={}",
                        end_global,
                        hidden_dim
                    );
                    continue;
                }
            };
            let end_end_byte = match (end_global + 1).checked_mul(hidden_dim) {
                Some(v) => v,
                None => {
                    log::warn!(
                        "Token index overflow: (end_global+1)={} * hidden_dim={}",
                        end_global + 1,
                        hidden_dim
                    );
                    continue;
                }
            };

            if start_byte >= token_embeddings.len()
                || start_end_byte > token_embeddings.len()
                || end_byte >= token_embeddings.len()
                || end_end_byte > token_embeddings.len()
            {
                continue;
            }

            // Get start and end token embeddings
            let start_emb = &token_embeddings[start_byte..start_end_byte];
            let end_emb = &token_embeddings[end_byte..end_end_byte];

            // Get width embedding
            let width = (candidate.width() as usize).min(max_width - 1);
            let width_start = width * width_emb_dim;
            let width_end = (width + 1) * width_emb_dim;
            // Bounds check: ensure width embedding slice is valid
            if width_end > self.width_embeddings.len() {
                continue;
            }
            let width_emb = &self.width_embeddings[width_start..width_end];

            // Concatenate and project (simplified - no actual matmul)
            // In a real implementation, this would be a linear layer
            let output_start = span_idx * hidden_dim;
            for h in 0..hidden_dim {
                // Placeholder: just average start and end
                span_embeddings[output_start + h] = (start_emb[h] + end_emb[h]) / 2.0;
                // Add width signal
                if h < width_emb_dim {
                    span_embeddings[output_start + h] += width_emb[h] * 0.1;
                }
            }
        }

        span_embeddings
    }
}

// =============================================================================
// Handshaking Matrix (TPLinker-style Joint Extraction)
// =============================================================================

/// Result cell in a handshaking matrix.
#[derive(Debug, Clone, Copy)]
pub struct HandshakingCell {
    /// Row index (token i)
    pub i: u32,
    /// Column index (token j)
    pub j: u32,
    /// Predicted label index
    pub label_idx: u16,
    /// Confidence score
    pub score: f32,
}

/// Handshaking matrix for joint entity-relation extraction.
///
/// # Research Alignment (W2NER, AAAI 2022)
///
/// From the W2NER paper (arXiv:2112.10070):
/// > "We present a novel alternative by modeling the unified NER as word-word
/// > relation classification, namely W2NER. The architecture resolves the kernel
/// > bottleneck of unified NER by effectively modeling the neighboring relations
/// > between entity words with **Next-Neighboring-Word (NNW)** and
/// > **Tail-Head-Word-* (THW-*)** relations."
///
/// In TPLinker/W2NER, we don't just tag tokens - we tag token PAIRS.
/// The matrix M\[i,j\] contains the label for the span (i, j).
///
/// # Key Relations
///
/// | Relation | Description | Purpose |
/// |----------|-------------|---------|
/// | NNW | Next-Neighboring-Word | Links adjacent tokens within entity |
/// | THW-* | Tail-Head-Word | Links end of one entity to start of next |
///
/// # Benefits
///
/// - Overlapping entities (same token in multiple spans)
/// - Joint entity-relation extraction in one pass
/// - Explicit boundary modeling
/// - Handles flat, nested, AND discontinuous NER in one model
pub struct HandshakingMatrix {
    /// Non-zero cells (sparse representation)
    pub cells: Vec<HandshakingCell>,
    /// Sequence length
    pub seq_len: usize,
    /// Number of labels
    pub num_labels: usize,
}

impl HandshakingMatrix {
    /// Create from dense scores with thresholding.
    ///
    /// # Arguments
    /// * `scores` - Dense [seq_len, seq_len, num_labels] scores
    /// * `threshold` - Minimum score to keep
    pub fn from_dense(scores: &[f32], seq_len: usize, num_labels: usize, threshold: f32) -> Self {
        // Performance: Pre-allocate cells vec with estimated capacity
        // Most matrices have sparse cells (only high-scoring ones), so we estimate conservatively
        let estimated_capacity = (seq_len * seq_len / 10).min(1000); // ~10% of cells typically pass threshold
        let mut cells = Vec::with_capacity(estimated_capacity);

        for i in 0..seq_len {
            for j in i..seq_len {
                // Upper triangular (i <= j)
                for l in 0..num_labels {
                    let idx = i * seq_len * num_labels + j * num_labels + l;
                    if idx < scores.len() {
                        let score = scores[idx];
                        if score >= threshold {
                            cells.push(HandshakingCell {
                                i: i as u32,
                                j: j as u32,
                                label_idx: l as u16,
                                score,
                            });
                        }
                    }
                }
            }
        }

        Self {
            cells,
            seq_len,
            num_labels,
        }
    }

    /// Decode entities from handshaking matrix.
    ///
    /// In W2NER convention, cell (i, j) represents a span where:
    /// - j is the start token index
    /// - i is the end token index (inclusive, so we add 1 for exclusive end)
    pub fn decode_entities<'a>(
        &self,
        registry: &'a SemanticRegistry,
    ) -> Vec<(SpanCandidate, &'a LabelDefinition, f32)> {
        let mut entities = Vec::new();

        for cell in &self.cells {
            if let Some(label) = registry.labels.get(cell.label_idx as usize) {
                if label.category == LabelCategory::Entity {
                    // W2NER: j=start, i=end (inclusive), so span is [j, i+1)
                    entities.push((SpanCandidate::new(0, cell.j, cell.i + 1), label, cell.score));
                }
            }
        }

        // Performance: Use unstable sort (we don't need stable sort here)
        // Sort by position, then by score (descending)
        entities.sort_unstable_by(|a, b| {
            a.0.start
                .cmp(&b.0.start)
                .then_with(|| a.0.end.cmp(&b.0.end))
                .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
        });

        // Performance: Pre-allocate kept vec with estimated capacity
        // Non-maximum suppression
        let mut kept = Vec::with_capacity(entities.len().min(32));
        for (span, label, score) in entities {
            let overlaps = kept.iter().any(|(s, _, _): &(SpanCandidate, _, _)| {
                !(span.end <= s.start || s.end <= span.start)
            });
            if !overlaps {
                kept.push((span, label, score));
            }
        }

        kept
    }
}

// =============================================================================
// Coreference Resolution
// =============================================================================

/// A coreference cluster (mentions referring to same entity).
#[derive(Debug, Clone)]
pub struct CoreferenceCluster {
    /// Cluster ID
    pub id: u64,
    /// Member entities (indices into entity list)
    pub members: Vec<usize>,
    /// Representative entity index (most informative mention)
    pub representative: usize,
    /// Canonical name (from representative)
    pub canonical_name: String,
}

/// Configuration for coreference resolution.
#[derive(Debug, Clone)]
pub struct CoreferenceConfig {
    /// Minimum cosine similarity to link mentions
    pub similarity_threshold: f32,
    /// Maximum token distance between coreferent mentions
    pub max_distance: Option<usize>,
    /// Whether to use exact string matching as a signal
    pub use_string_match: bool,
}

impl Default for CoreferenceConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.85,
            max_distance: Some(500),
            use_string_match: true,
        }
    }
}

/// Resolve coreferences between entities using embedding similarity.
///
/// # Algorithm
///
/// 1. Compute pairwise cosine similarity between entity embeddings
/// 2. Link entities above threshold (with optional distance constraint)
/// 3. Build clusters via transitive closure
/// 4. Select representative (longest/most informative mention)
///
/// # Example
///
/// Input entities: ["Marie Curie", "She", "The scientist", "Curie"]
/// Output clusters: [{0, 1, 2, 3}] with canonical_name = "Marie Curie"
pub fn resolve_coreferences(
    entities: &[Entity],
    embeddings: &[f32], // [num_entities, hidden_dim]
    hidden_dim: usize,
    config: &CoreferenceConfig,
) -> Vec<CoreferenceCluster> {
    let n = entities.len();
    if n == 0 {
        return vec![];
    }

    // Union-find for clustering
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut [usize], i: usize) -> usize {
        if parent[i] != i {
            parent[i] = find(parent, parent[i]);
        }
        parent[i]
    }

    fn union(parent: &mut [usize], i: usize, j: usize) {
        let pi = find(parent, i);
        let pj = find(parent, j);
        if pi != pj {
            parent[pi] = pj;
        }
    }

    // Check all pairs
    for i in 0..n {
        for j in (i + 1)..n {
            // String match check (fast path)
            if config.use_string_match {
                let text_i = entities[i].text.to_lowercase();
                let text_j = entities[j].text.to_lowercase();
                if text_i == text_j || text_i.contains(&text_j) || text_j.contains(&text_i) {
                    // Same entity type required
                    if entities[i].entity_type == entities[j].entity_type {
                        union(&mut parent, i, j);
                        continue;
                    }
                }
            }

            // Distance check
            if let Some(max_dist) = config.max_distance {
                let dist = if entities[i].end <= entities[j].start {
                    entities[j].start - entities[i].end
                } else {
                    entities[i].start.saturating_sub(entities[j].end)
                };
                if dist > max_dist {
                    continue;
                }
            }

            // Embedding similarity
            if embeddings.len() >= (j + 1) * hidden_dim {
                let emb_i = &embeddings[i * hidden_dim..(i + 1) * hidden_dim];
                let emb_j = &embeddings[j * hidden_dim..(j + 1) * hidden_dim];

                let similarity = cosine_similarity(emb_i, emb_j);

                if similarity >= config.similarity_threshold {
                    // Same entity type required
                    if entities[i].entity_type == entities[j].entity_type {
                        union(&mut parent, i, j);
                    }
                }
            }
        }
    }

    // Build clusters
    let mut cluster_members: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        cluster_members.entry(root).or_default().push(i);
    }

    // Convert to CoreferenceCluster
    let mut clusters = Vec::new();
    let mut cluster_id = 0u64;

    for (_root, members) in cluster_members {
        if members.len() > 1 {
            // Find representative (longest mention)
            let representative = *members
                .iter()
                .max_by_key(|&&i| entities[i].text.len())
                .unwrap_or(&members[0]);

            clusters.push(CoreferenceCluster {
                id: cluster_id,
                members,
                representative,
                canonical_name: entities[representative].text.clone(),
            });
            cluster_id += 1;
        }
    }

    clusters
}

/// Compute cosine similarity between two vectors.
///
/// Returns a value in [-1.0, 1.0] where:
/// - 1.0 = identical direction
/// - 0.0 = orthogonal
/// - -1.0 = opposite direction
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

// =============================================================================
// Relation Extraction
// =============================================================================

/// Configuration for relation extraction.
#[derive(Debug, Clone)]
pub struct RelationExtractionConfig {
    /// Maximum token distance between head and tail
    pub max_span_distance: usize,
    /// Minimum confidence for relation
    pub threshold: f32,
    /// Whether to extract relation triggers
    pub extract_triggers: bool,
}

impl Default for RelationExtractionConfig {
    fn default() -> Self {
        Self {
            max_span_distance: 50,
            threshold: 0.5,
            extract_triggers: true,
        }
    }
}

/// Extract relations between entities.
///
/// # Algorithm (Two-Pass)
///
/// 1. Run entity NER to find all entity mentions
/// 2. For each entity pair within distance threshold:
///    - Encode the span between them
///    - Match against relation type embeddings
///    - Optionally identify trigger span
///
/// # Returns
///
/// Relations with head/tail entities and optional trigger spans.
pub fn extract_relations(
    entities: &[Entity],
    text: &str,
    registry: &SemanticRegistry,
    _config: &RelationExtractionConfig,
) -> Vec<Relation> {
    let mut relations = Vec::new();

    // Get relation labels
    let relation_labels: Vec<_> = registry.relation_labels().collect();
    if relation_labels.is_empty() {
        return relations;
    }

    // Check all entity pairs
    for (i, head) in entities.iter().enumerate() {
        for (j, tail) in entities.iter().enumerate() {
            if i == j {
                continue;
            }

            // Check distance
            let distance = if head.end <= tail.start {
                tail.start - head.end
            } else {
                head.start.saturating_sub(tail.end)
            };

            if distance > _config.max_span_distance {
                continue;
            }

            // Look for relation triggers in the text between entities
            let (span_start, span_end) = if head.end <= tail.start {
                (head.end, tail.start)
            } else {
                (tail.end, head.start)
            };

            let between_text = text.get(span_start..span_end).unwrap_or("");

            // Simple heuristic: check for common relation indicators
            let relation_type = detect_relation_type(head, tail, between_text, &relation_labels);

            if let Some((rel_type, confidence, trigger)) = relation_type {
                let trigger_span = trigger.map(|(s, e)| (span_start + s, span_start + e));

                relations.push(Relation {
                    head: head.clone(),
                    tail: tail.clone(),
                    relation_type: rel_type.to_string(),
                    trigger_span,
                    confidence,
                });
            }
        }
    }

    relations
}

/// Result of relation detection: (label, confidence, optional span).
type RelationMatch<'a> = (&'a str, f64, Option<(usize, usize)>);

/// Detect relation type from context (heuristic fallback).
fn detect_relation_type<'a>(
    head: &Entity,
    tail: &Entity,
    between_text: &str,
    relation_labels: &[&'a LabelDefinition],
) -> Option<RelationMatch<'a>> {
    let between_lower = between_text.to_lowercase();

    // Common patterns: (relation_slug, triggers, confidence)
    struct RelPattern {
        slug: &'static str,
        triggers: &'static [&'static str],
        confidence: f64,
    }

    let patterns: &[RelPattern] = &[
        // Employment relations
        RelPattern {
            slug: "CEO_OF",
            triggers: &["ceo of", "chief executive", "leads", "founded"],
            confidence: 0.8,
        },
        RelPattern {
            slug: "WORKS_FOR",
            triggers: &["works for", "works at", "employed by", "employee of"],
            confidence: 0.7,
        },
        RelPattern {
            slug: "FOUNDED",
            triggers: &["founded", "co-founded", "started", "established"],
            confidence: 0.8,
        },
        // Location relations
        RelPattern {
            slug: "LOCATED_IN",
            triggers: &["in", "at", "based in", "located in", "headquartered in"],
            confidence: 0.6,
        },
        RelPattern {
            slug: "BORN_IN",
            triggers: &["born in", "native of", "from"],
            confidence: 0.7,
        },
        // Other
        RelPattern {
            slug: "PART_OF",
            triggers: &["part of", "member of", "belongs to", "subsidiary of"],
            confidence: 0.7,
        },
    ];

    for pattern in patterns {
        // Check if relation type is in registry
        let in_registry = relation_labels.iter().any(|l| l.slug == pattern.slug);
        if !in_registry {
            continue;
        }

        for trigger in pattern.triggers {
            if let Some(pos) = between_lower.find(trigger) {
                // Validate entity types make sense
                let valid = match pattern.slug {
                    "CEO_OF" | "WORKS_FOR" | "FOUNDED" => {
                        matches!(head.entity_type, EntityType::Person)
                            && matches!(tail.entity_type, EntityType::Organization)
                    }
                    "LOCATED_IN" | "BORN_IN" => {
                        matches!(tail.entity_type, EntityType::Location)
                    }
                    _ => true,
                };

                if valid {
                    return Some((
                        pattern.slug,
                        pattern.confidence,
                        Some((pos, pos + trigger.len())),
                    ));
                }
            }
        }
    }

    None
}

// =============================================================================
// Binary Embeddings for Fast Blocking (Research: Hamming Distance)
// =============================================================================

/// Binary hash for fast approximate nearest neighbor search.
///
/// # Research Background
///
/// Binary embeddings enable sub-linear search via Hamming distance. Key insight
/// from our research synthesis: **binary embeddings are for blocking, not primary
/// retrieval**. The sign-rank limitation means they cannot represent all similarity
/// relationships, but they excel at fast candidate filtering.
///
/// # Two-Stage Retrieval Pattern
///
/// ```text
/// Query → [Binary Hash] → Hamming Filter (fast) → Candidates
///                                                      ↓
///                                              [Dense Similarity]
///                                                      ↓
///                                               Final Results
/// ```
///
/// # Example
///
/// ```rust
/// use anno::backends::inference::BinaryHash;
///
/// // Create hashes from embeddings
/// let hash1 = BinaryHash::from_embedding(&[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]);
/// let hash2 = BinaryHash::from_embedding(&[0.15, -0.25, 0.35, -0.45, 0.55, -0.65, 0.75, -0.85]);
///
/// // Similar embeddings → low Hamming distance
/// assert!(hash1.hamming_distance(&hash2) < 2);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BinaryHash {
    /// Packed bits (each u64 holds 64 bits)
    pub bits: Vec<u64>,
    /// Original dimension (number of bits)
    pub dim: usize,
}

impl BinaryHash {
    /// Create from a dense embedding using sign function.
    ///
    /// Each positive value → 1, each negative/zero value → 0.
    #[must_use]
    pub fn from_embedding(embedding: &[f32]) -> Self {
        let dim = embedding.len();
        let num_u64s = dim.div_ceil(64);
        let mut bits = vec![0u64; num_u64s];

        for (i, &val) in embedding.iter().enumerate() {
            if val > 0.0 {
                let word_idx = i / 64;
                let bit_idx = i % 64;
                bits[word_idx] |= 1u64 << bit_idx;
            }
        }

        Self { bits, dim }
    }

    /// Create from a dense f64 embedding.
    #[must_use]
    pub fn from_embedding_f64(embedding: &[f64]) -> Self {
        let dim = embedding.len();
        let num_u64s = dim.div_ceil(64);
        let mut bits = vec![0u64; num_u64s];

        for (i, &val) in embedding.iter().enumerate() {
            if val > 0.0 {
                let word_idx = i / 64;
                let bit_idx = i % 64;
                bits[word_idx] |= 1u64 << bit_idx;
            }
        }

        Self { bits, dim }
    }

    /// Compute Hamming distance (number of differing bits).
    ///
    /// Uses POPCNT instruction when available for hardware acceleration.
    #[must_use]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        self.bits
            .iter()
            .zip(other.bits.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }

    /// Compute normalized Hamming distance (0.0 to 1.0).
    #[must_use]
    pub fn hamming_distance_normalized(&self, other: &Self) -> f64 {
        if self.dim == 0 {
            return 0.0;
        }
        self.hamming_distance(other) as f64 / self.dim as f64
    }

    /// Convert Hamming distance to approximate cosine similarity.
    ///
    /// Based on the relationship: cos(θ) ≈ 1 - 2 * (hamming_distance / dim)
    /// This is an approximation valid for random hyperplane hashing.
    #[must_use]
    pub fn approximate_cosine(&self, other: &Self) -> f64 {
        1.0 - 2.0 * self.hamming_distance_normalized(other)
    }
}

/// Blocker using binary embeddings for fast candidate filtering.
///
/// # Usage Pattern
///
/// 1. Pre-compute binary hashes for all entities in your KB
/// 2. At query time, hash the query embedding
/// 3. Find candidates within Hamming distance threshold
/// 4. Run dense similarity only on candidates
///
/// # Example
///
/// ```rust
/// use anno::backends::inference::{BinaryBlocker, BinaryHash};
///
/// let mut blocker = BinaryBlocker::new(8); // 8-bit Hamming threshold
///
/// // Add entities to the index
/// let hash1 = BinaryHash::from_embedding(&vec![0.1; 768]);
/// let hash2 = BinaryHash::from_embedding(&vec![-0.1; 768]);
/// blocker.add(0, hash1);
/// blocker.add(1, hash2);
///
/// // Query
/// let query = BinaryHash::from_embedding(&vec![0.1; 768]);
/// let candidates = blocker.query(&query);
/// assert!(candidates.contains(&0)); // Similar to hash1
/// ```
#[derive(Debug, Clone)]
pub struct BinaryBlocker {
    /// Hamming distance threshold for candidates
    pub threshold: u32,
    /// Index of hashes by ID
    index: Vec<(usize, BinaryHash)>,
}

impl BinaryBlocker {
    /// Create a new blocker with the given threshold.
    #[must_use]
    pub fn new(threshold: u32) -> Self {
        Self {
            threshold,
            index: Vec::new(),
        }
    }

    /// Add an entity to the index.
    pub fn add(&mut self, id: usize, hash: BinaryHash) {
        self.index.push((id, hash));
    }

    /// Add multiple entities.
    pub fn add_batch(&mut self, entries: impl IntoIterator<Item = (usize, BinaryHash)>) {
        self.index.extend(entries);
    }

    /// Find candidate IDs within Hamming distance threshold.
    #[must_use]
    pub fn query(&self, query: &BinaryHash) -> Vec<usize> {
        self.index
            .iter()
            .filter(|(_, hash)| hash.hamming_distance(query) <= self.threshold)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Find candidates with their distances.
    #[must_use]
    pub fn query_with_distance(&self, query: &BinaryHash) -> Vec<(usize, u32)> {
        self.index
            .iter()
            .map(|(id, hash)| (*id, hash.hamming_distance(query)))
            .filter(|(_, dist)| *dist <= self.threshold)
            .collect()
    }

    /// Number of entries in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Check if index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Clear the index.
    pub fn clear(&mut self) {
        self.index.clear();
    }
}

/// Recommended two-stage retrieval using binary blocking + dense reranking.
///
/// # Research Context
///
/// This implements the pattern identified in our research synthesis:
/// - Stage 1: Binary blocking for O(n) candidate filtering
/// - Stage 2: Dense similarity for accurate ranking
///
/// The key insight is that binary embeddings have fundamental limitations
/// (sign-rank theorem) but excel at fast filtering.
///
/// # Arguments
///
/// * `query_embedding` - Dense query embedding
/// * `candidate_embeddings` - Dense embeddings of all candidates
/// * `binary_threshold` - Hamming distance threshold for blocking
/// * `top_k` - Number of final results to return
///
/// # Returns
///
/// Vector of (candidate_index, similarity_score) pairs, sorted by score descending.
#[must_use]
pub fn two_stage_retrieval(
    query_embedding: &[f32],
    candidate_embeddings: &[Vec<f32>],
    binary_threshold: u32,
    top_k: usize,
) -> Vec<(usize, f32)> {
    // Stage 1: Binary blocking
    let query_hash = BinaryHash::from_embedding(query_embedding);

    let candidate_hashes: Vec<BinaryHash> = candidate_embeddings
        .iter()
        .map(|e| BinaryHash::from_embedding(e))
        .collect();

    let mut blocker = BinaryBlocker::new(binary_threshold);
    for (i, hash) in candidate_hashes.into_iter().enumerate() {
        blocker.add(i, hash);
    }

    let candidates = blocker.query(&query_hash);

    // Stage 2: Dense similarity on candidates only
    // Performance: Pre-allocate scored vec with known size
    let mut scored: Vec<(usize, f32)> = Vec::with_capacity(candidates.len());
    scored.extend(candidates.into_iter().map(|idx| {
        let sim = cosine_similarity_f32(query_embedding, &candidate_embeddings[idx]);
        (idx, sim)
    }));

    // Performance: Use unstable sort (we don't need stable sort here)
    // Sort by similarity descending
    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_k);
    scored
}

/// Compute cosine similarity between two f32 vectors.
#[must_use]
pub fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_registry_builder() {
        let registry = SemanticRegistry::builder()
            .add_entity("person", "A human being")
            .add_entity("organization", "A company or group")
            .add_relation("WORKS_FOR", "Employment relationship")
            .build_placeholder(768);

        assert_eq!(registry.len(), 3);
        assert_eq!(registry.entity_labels().count(), 2);
        assert_eq!(registry.relation_labels().count(), 1);
    }

    #[test]
    fn test_standard_ner_registry() {
        let registry = SemanticRegistry::standard_ner(768);
        assert!(registry.len() >= 5);
        assert!(registry.label_index.contains_key("person"));
        assert!(registry.label_index.contains_key("organization"));
    }

    #[test]
    fn test_dot_product_interaction() {
        let interaction = DotProductInteraction::new();

        // 2 spans, 3 labels, hidden_dim=4
        let span_embs = vec![
            1.0, 0.0, 0.0, 0.0, // span 0
            0.0, 1.0, 0.0, 0.0, // span 1
        ];
        let label_embs = vec![
            1.0, 0.0, 0.0, 0.0, // label 0 (matches span 0)
            0.0, 1.0, 0.0, 0.0, // label 1 (matches span 1)
            0.5, 0.5, 0.0, 0.0, // label 2 (partial match both)
        ];

        let scores = interaction.compute_similarity(&span_embs, 2, &label_embs, 3, 4);

        assert_eq!(scores.len(), 6); // 2 * 3
        assert!((scores[0] - 1.0).abs() < 0.01); // span0 vs label0
        assert!((scores[4] - 1.0).abs() < 0.01); // span1 vs label1
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.001);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_coreference_string_match() {
        let entities = vec![
            Entity::new("Marie Curie", EntityType::Person, 0, 11, 0.95),
            Entity::new("Curie", EntityType::Person, 50, 55, 0.90),
        ];

        let embeddings = vec![0.0f32; 2 * 768]; // Placeholder
        let clusters =
            resolve_coreferences(&entities, &embeddings, 768, &CoreferenceConfig::default());

        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].members.len(), 2);
        assert_eq!(clusters[0].canonical_name, "Marie Curie");
    }

    #[test]
    fn test_handshaking_matrix() {
        // 3 tokens, 2 labels, threshold 0.5
        let scores = vec![
            // token 0 with tokens 0,1,2 for labels 0,1
            0.9, 0.1, // (0,0)
            0.2, 0.8, // (0,1)
            0.1, 0.1, // (0,2)
            // token 1 with tokens 0,1,2
            0.0, 0.0, // (1,0) - skipped (lower triangle)
            0.7, 0.2, // (1,1)
            0.3, 0.6, // (1,2)
            // token 2
            0.0, 0.0, // (2,0)
            0.0, 0.0, // (2,1)
            0.1, 0.1, // (2,2)
        ];

        let matrix = HandshakingMatrix::from_dense(&scores, 3, 2, 0.5);

        // Should have cells for scores >= 0.5
        assert!(matrix.cells.len() >= 4);
    }

    #[test]
    fn test_relation_extraction() {
        let entities = vec![
            Entity::new("Steve Jobs", EntityType::Person, 0, 10, 0.95),
            Entity::new("Apple", EntityType::Organization, 20, 25, 0.90),
        ];

        let text = "Steve Jobs founded Apple Inc in 1976";

        let registry = SemanticRegistry::builder()
            .add_relation("FOUNDED", "Founded an organization")
            .build_placeholder(768);

        let config = RelationExtractionConfig::default();
        let relations = extract_relations(&entities, text, &registry, &config);

        assert!(!relations.is_empty());
        assert_eq!(relations[0].relation_type, "FOUNDED");
    }

    // =========================================================================
    // Binary Embedding Tests
    // =========================================================================

    #[test]
    fn test_binary_hash_creation() {
        let embedding = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let hash = BinaryHash::from_embedding(&embedding);

        assert_eq!(hash.dim, 8);
        // Positive values at indices 0, 2, 4, 6 should be set
        // bits[0] should have bits 0, 2, 4, 6 set = 0b01010101 = 85
        assert_eq!(hash.bits[0], 85);
    }

    #[test]
    fn test_hamming_distance_identical() {
        let embedding = vec![0.1; 64];
        let hash1 = BinaryHash::from_embedding(&embedding);
        let hash2 = BinaryHash::from_embedding(&embedding);

        assert_eq!(hash1.hamming_distance(&hash2), 0);
    }

    #[test]
    fn test_hamming_distance_opposite() {
        let embedding1 = vec![0.1; 64];
        let embedding2 = vec![-0.1; 64];
        let hash1 = BinaryHash::from_embedding(&embedding1);
        let hash2 = BinaryHash::from_embedding(&embedding2);

        assert_eq!(hash1.hamming_distance(&hash2), 64);
    }

    #[test]
    fn test_hamming_distance_half() {
        let embedding1 = vec![0.1; 64];
        let mut embedding2 = vec![0.1; 64];
        // Flip second half
        for i in 32..64 {
            embedding2[i] = -0.1;
        }

        let hash1 = BinaryHash::from_embedding(&embedding1);
        let hash2 = BinaryHash::from_embedding(&embedding2);

        assert_eq!(hash1.hamming_distance(&hash2), 32);
    }

    #[test]
    fn test_binary_blocker() {
        let mut blocker = BinaryBlocker::new(5);

        // Add some hashes
        let base_embedding = vec![0.1; 64];
        let similar_embedding = {
            let mut e = vec![0.1; 64];
            e[0] = -0.1; // Flip 1 bit
            e[1] = -0.1; // Flip 2 bits
            e
        };
        let different_embedding = vec![-0.1; 64];

        blocker.add(0, BinaryHash::from_embedding(&base_embedding));
        blocker.add(1, BinaryHash::from_embedding(&similar_embedding));
        blocker.add(2, BinaryHash::from_embedding(&different_embedding));

        // Query with base
        let query = BinaryHash::from_embedding(&base_embedding);
        let candidates = blocker.query(&query);

        assert!(candidates.contains(&0), "Should find exact match");
        assert!(
            candidates.contains(&1),
            "Should find similar (2 bits different)"
        );
        assert!(
            !candidates.contains(&2),
            "Should NOT find opposite (64 bits different)"
        );
    }

    #[test]
    fn test_two_stage_retrieval() {
        // Create embeddings
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let candidates = vec![
            vec![1.0, 0.0, 0.0, 0.0],  // Identical
            vec![0.9, 0.1, 0.0, 0.0],  // Similar
            vec![-1.0, 0.0, 0.0, 0.0], // Opposite
            vec![0.0, 1.0, 0.0, 0.0],  // Orthogonal
        ];

        // Generous threshold to get candidates
        let results = two_stage_retrieval(&query, &candidates, 4, 2);

        assert!(!results.is_empty());
        // First result should be exact match
        assert_eq!(results[0].0, 0);
        assert!((results[0].1 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_approximate_cosine() {
        let embedding1 = vec![0.1; 768];
        let embedding2 = vec![0.1; 768];
        let hash1 = BinaryHash::from_embedding(&embedding1);
        let hash2 = BinaryHash::from_embedding(&embedding2);

        // Identical → approximate cosine should be ~1.0
        let approx = hash1.approximate_cosine(&hash2);
        assert!((approx - 1.0).abs() < 0.001);

        // Opposite → approximate cosine should be ~-1.0
        let embedding3 = vec![-0.1; 768];
        let hash3 = BinaryHash::from_embedding(&embedding3);
        let approx_opp = hash1.approximate_cosine(&hash3);
        assert!((approx_opp - (-1.0)).abs() < 0.001);
    }
}
