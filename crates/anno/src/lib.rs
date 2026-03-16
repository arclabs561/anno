//! # anno
//!
//! Information extraction for unstructured text: named entity recognition (NER),
//! coreference resolution, entity linking, relation extraction, and zero-shot entity types.
//!
//! - **NER output**: variable-length spans with **character offsets** (Unicode scalar values), not
//!   byte offsets.
//! - **Coreference output**: clusters (“tracks”) of mentions within one document.
//!
//! This crate focuses on inference-time extraction. Dataset loaders, benchmarking, and matrix
//! evaluation tooling live in `anno-eval` (and the `anno` CLI lives in `anno-cli`).
//!
//! ## Quickstart
//!
//! ```rust
//! use anno::{Model, StackedNER};
//!
//! let m = StackedNER::default();
//! let ents = m.extract_entities("Lynn Conway worked at IBM and Xerox PARC.", None)?;
//! for e in &ents {
//!     println!("{} [{}] ({},{}) {:.2}", e.text, e.entity_type, e.start(), e.end(), e.confidence);
//! }
//! // Lynn Conway [PER] (0,12) 0.95
//! // IBM [ORG] (27,30) 0.95
//! // Xerox PARC [ORG] (35,45) 0.95
//! # Ok::<(), anno::Error>(())
//! ```
//!
//! ## Zero-shot custom entity types
//!
//! Zero-shot custom entity types are provided by GLiNER backends when the `onnx` feature is
//! enabled. See the repo docs for the CLI flag (`--extract-types`) and the library API.
//!
//! ## Offline / downloads
//!
//! By default, ML weights may download on first use. To force cached-only behavior, set
//! `ANNO_NO_DOWNLOADS=1` (after prefetching models).

#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

// Allow unit tests (and included CI test modules) to refer to this crate as `anno::...`,
// matching integration-test style imports.
extern crate self as anno;

// Module declarations (standard Cargo layout under `src/`)
pub mod backends;
/// Edit distance algorithms.
pub mod edit_distance;
pub mod env;
pub mod error;
/// Evaluation/analysis primitives (coref metrics, cluster encoders, etc.).
///
/// This module is only available when the legacy `eval` feature (or the preferred `analysis`
/// alias) is enabled.
#[cfg(feature = "analysis")]
#[cfg_attr(docsrs, doc(cfg(feature = "analysis")))]
pub mod eval;
/// Export entity results to annotation and interchange formats (brat, CoNLL, JSONL, RDF, JSON-LD, CSV).
pub mod export;
/// Entity feature extraction for downstream ML and analysis.
pub mod features;
/// Small, dependency-light heuristics (negation, quantifiers, etc.).
pub mod heuristics;
/// Lightweight URL/file ingestion helpers (not a crawling/pipeline product).
pub mod ingest;
/// Joint inference experiments (optional; not the primary API surface).
#[cfg(feature = "joint")]
pub mod joint;
/// Keyword and keyphrase extraction (TF-IDF, YAKE, TextRank).
#[cfg(feature = "graph")]
#[cfg_attr(docsrs, doc(cfg(feature = "graph")))]
pub mod keywords;
pub mod lang;
/// Knowledge-base linking helpers (experimental).
pub mod linking;
pub mod offset;
/// Shared PageRank algorithm for graph-based ranking.
#[cfg(feature = "graph")]
#[cfg_attr(docsrs, doc(cfg(feature = "graph")))]
pub mod pagerank;
/// Preprocessing for mention detection.
pub(crate) mod preprocess;
/// Coreference preprocessing for RAG: rewrite pronouns for self-contained chunks.
///
/// See [`rag::resolve_for_rag`] for the main entry point.
#[cfg(feature = "analysis")]
#[cfg_attr(docsrs, doc(cfg(feature = "analysis")))]
pub mod rag;
/// Entity salience and importance ranking.
#[cfg(feature = "graph")]
#[cfg_attr(docsrs, doc(cfg(feature = "graph")))]
pub mod salience;
pub mod schema;
pub mod similarity;
pub mod sync;
/// Temporal entity tracking, parsing, and diachronic NER.
pub mod temporal;
/// Language-specific tokenization for multilingual NLP.
pub mod tokenizer;
pub mod types;

// Note: research-only geometry experiments were archived out of `anno` to keep the public
// surface grounded. Prefer `docs/` for repo-local design notes and experiments.

/// Discourse-level analysis for coreference resolution.
///
/// Provides infrastructure for handling phenomena that span sentence boundaries:
///
/// - **Centering theory**: Track discourse focus through forward/backward-looking centers
/// - **Uncertain reference**: Deferred resolution using epsilon-term semantics
/// - **Abstract anaphora**: Pronouns referring to events, propositions, facts
/// - **Shell nouns**: Abstract nouns like "problem", "issue", "fact"
///
/// Enable with the `discourse` feature.
///
/// See `discourse::centering` for salience-based pronoun resolution and
/// `discourse::uncertain_reference` for handling ambiguous references.
#[cfg(feature = "discourse")]
#[cfg_attr(docsrs, doc(cfg(feature = "discourse")))]
pub mod discourse;

// Re-export error types
pub use error::{Error, Result};

// =============================================================================
// Core types live in `anno-core`
// =============================================================================

// Re-export core types at the crate root (the `anno` public API surface).
pub use anno_core::{
    generate_span_candidates, Animacy, CanonicalId, Confidence, CorefChain, CorefDocument,
    CoreferenceResolver, Corpus, DiscontinuousSpan, Entity, EntityBuilder, EntityCategory,
    EntityType, ExtractionMethod, Gender, GroundedDocument, HashMapLexicon, HierarchicalConfidence,
    Identity, IdentityId, IdentitySource, Lexicon, Location, Mention, MentionType, Modality,
    Number, Person, PhiFeatures, Provenance, Quantifier, RaggedBatch, Relation, Signal, SignalId,
    SignalRef, Span, SpanCandidate, Track, TrackId, TrackRef, TrackStats, TypeLabel, TypeMapper,
    ValidationIssue,
};

/// `anno-core`’s stable types under a namespaced module.
///
/// This exists for readability in downstream codebases (e.g. `anno::core::Entity`)
/// and mirrors the structure of the internal `anno-core` crate.
pub mod core {
    pub use anno_core::core::{coref, grounded};
}

// Re-export commonly used types
pub use lang::{detect_language, Language};
pub use offset::{
    bytes_to_chars, chars_to_bytes, is_ascii, OffsetMapping, SpanConverter, TextSpan, TokenSpan,
};
pub use schema::{
    map_to_canonical, CanonicalType, CoarseType, DatasetSchema, InformationLoss, SchemaMapper,
};
pub use similarity::{jaccard_word_similarity, jaccard_word_similarity_f32, string_similarity};
pub use sync::{lock, read_lock, try_lock, write_lock, Mutex, RwLock};
pub use types::{EntitySliceExt, Score};

// =============================================================================
// Sealed Trait Pattern
// =============================================================================
//
// The `Model` trait is sealed to:
// 1. Maintain invariants (entities have valid offsets, confidence in [0,1])
// 2. Allow adding methods without breaking external implementations
// 3. Ensure all backends share consistent behavior
//
// For external/plugin backends, use the `AnyModel` wrapper (see below).
// =============================================================================

mod sealed {
    pub trait Sealed {}

    impl Sealed for super::RegexNER {}
    impl Sealed for super::HeuristicNER {}
    impl Sealed for super::StackedNER {}
    impl Sealed for super::EnsembleNER {}
    impl Sealed for super::CrfNER {}
    impl Sealed for super::NuNER {}
    impl Sealed for super::W2NER {}

    #[cfg(feature = "onnx")]
    impl Sealed for super::BertNEROnnx {}

    #[cfg(feature = "onnx")]
    impl Sealed for super::GLiNEROnnx {}

    #[cfg(feature = "onnx")]
    impl Sealed for super::backends::albert::ALBERTNER {}

    #[cfg(feature = "onnx")]
    impl Sealed for super::backends::deberta_v3::DeBERTaV3NER {}

    impl Sealed for super::backends::gliner_poly::GLiNERPoly {}

    #[cfg(feature = "onnx")]
    impl Sealed for super::backends::gliner2::GLiNER2Onnx {}

    #[cfg(feature = "candle")]
    impl Sealed for super::CandleNER {}

    #[cfg(feature = "candle")]
    impl Sealed for super::backends::gliner_candle::GLiNERCandle {}

    #[cfg(feature = "candle")]
    impl Sealed for super::backends::gliner2::GLiNER2Candle {}

    impl Sealed for super::backends::tplinker::TPLinker {}
    impl Sealed for super::backends::universal_ner::UniversalNER {}
    impl Sealed for super::backends::lexicon::LexiconNER {}

    impl Sealed for super::backends::hmm::HmmNER {}
    impl Sealed for super::backends::bilstm_crf::BiLstmCrfNER {}
    impl Sealed for super::backends::router::AutoNER {}

    impl Sealed for super::MockModel {}
    #[cfg(feature = "joint")]
    impl Sealed for super::joint::JointModel {}
}

/// Trait for NER model backends.
///
/// # Sealed Trait
///
/// `Model` is intentionally sealed (cannot be implemented outside this crate) to:
///
/// 1. **Maintain invariants**: All backends must produce entities with valid character
///    offsets, confidence in `[0, 1]`, and non-empty text.
/// 2. **Allow evolution**: New methods can be added with default implementations
///    without breaking external code.
/// 3. **Ensure consistency**: All backends share standardized behavior for
///    `is_available()`, `supported_types()`, etc.
///
/// # For External Backends
///
/// If you need to integrate an external NER backend (e.g., a REST API, Python model
/// via PyO3, or custom implementation), use the [`AnyModel`] wrapper:
///
/// ```rust,ignore
/// use anno::{AnyModel, Entity, EntityType, Result};
///
/// struct MyExternalNER { /* ... */ }
///
/// impl MyExternalNER {
///     fn extract(&self, text: &str) -> Vec<Entity> {
///         // Your implementation
///         vec![]
///     }
/// }
///
/// // Wrap in AnyModel to use with anno's infrastructure
/// let model = AnyModel::new(
///     "my-ner",
///     vec![EntityType::Person, EntityType::Organization],
///     move |text, _lang| Ok(my_ner.extract(text)),
/// );
///
/// // Now usable wherever Box<dyn Model> is expected
/// let entities = model.extract_entities("Hello world", None)?;
/// ```
///
/// [`AnyModel`]: crate::AnyModel
pub trait Model: sealed::Sealed + Send + Sync {
    /// Extract entities from text.
    fn extract_entities(&self, text: &str, language: Option<Language>) -> Result<Vec<Entity>>;

    /// Get supported entity types.
    fn supported_types(&self) -> Vec<EntityType>;

    /// Check if model is available and ready.
    fn is_available(&self) -> bool;

    /// Get the model name/identifier.
    fn name(&self) -> &'static str {
        "unknown"
    }

    /// Get a description of the model.
    fn description(&self) -> &'static str {
        "Unknown NER model"
    }

    /// Get capability summary for this model.
    ///
    /// Override this in implementations that support additional capabilities
    /// (relations, dynamic labels, discontinuous entities) to enable runtime discovery.
    ///
    /// # Default
    ///
    /// Returns a [`ModelCapabilities`] with all fields set to `false`/`None`.
    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities::default()
    }

    /// Get a version identifier for the model configuration/weights.
    ///
    /// Used for cache invalidation. Default implementation returns "1".
    fn version(&self) -> String {
        "1".to_string()
    }
}

// =============================================================================
// AnyModel: Adapter for External Backends
// =============================================================================

/// A wrapper that allows external code to implement NER backends without
/// directly implementing the sealed `Model` trait.
///
/// `AnyModel` acts as an adapter: you provide a closure that does the actual
/// entity extraction, and `AnyModel` implements `Model` on your behalf.
///
/// # Example
///
/// ```rust
/// use anno::{AnyModel, Entity, EntityType, Language, Model, Result};
///
/// // Define extraction logic as a closure or function
/// let my_extractor = |text: &str, _lang: Option<Language>| -> Result<Vec<Entity>> {
///     // Your custom NER logic here
///     Ok(vec![])
/// };
///
/// // Wrap in AnyModel
/// let model = AnyModel::new(
///     "my-custom-ner",
///     "Custom NER backend using external API",
///     vec![EntityType::Person, EntityType::Organization],
///     my_extractor,
/// );
///
/// // Use like any other Model
/// assert!(model.is_available());
/// let entities = model.extract_entities("Hello world", None).unwrap();
/// ```
///
/// # Thread Safety
///
/// The extractor closure must be `Send + Sync`. For interior mutability
/// (e.g., caching, connection pooling), use `Arc<Mutex<...>>` or similar.
/// Type alias for the `AnyModel` extractor closure.
type AnyModelExtractor = dyn Fn(&str, Option<Language>) -> Result<Vec<Entity>> + Send + Sync;

/// Type alias for the `AnyModel` label-extraction closure (`DynamicLabels`).
type AnyModelLabelExtractor =
    dyn Fn(&str, &[&str], Option<Language>) -> Result<Vec<Entity>> + Send + Sync;

/// Type alias for the `AnyModel` relation-extraction closure (`RelationCapable`).
type AnyModelRelationExtractor =
    dyn Fn(&str, Option<Language>) -> Result<(Vec<Entity>, Vec<Relation>)> + Send + Sync;

/// A wrapper that turns an extractor closure into a `Model`.
///
/// `AnyModel` supports [`DynamicLabels`] and [`RelationCapable`] via closures
/// (see [`with_dynamic_labels`](Self::with_dynamic_labels) and
/// [`with_relations`](Self::with_relations)).
pub struct AnyModel {
    name: &'static str,
    description: &'static str,
    supported_types: Vec<EntityType>,
    extractor: Box<AnyModelExtractor>,
    version: String,
    /// Optional closure backing [`DynamicLabels::extract_with_labels`].
    label_extractor: Option<Box<AnyModelLabelExtractor>>,
    /// Optional closure backing [`RelationCapable::extract_with_relations`].
    relation_extractor: Option<Box<AnyModelRelationExtractor>>,
}

impl AnyModel {
    /// Create a new `AnyModel` wrapper.
    ///
    /// # Arguments
    ///
    /// * `name` - Model identifier (e.g., "my-ner")
    /// * `description` - Human-readable description
    /// * `supported_types` - Entity types this model can extract
    /// * `extractor` - Closure that performs the actual extraction
    pub fn new(
        name: &'static str,
        description: &'static str,
        supported_types: Vec<EntityType>,
        extractor: impl Fn(&str, Option<Language>) -> Result<Vec<Entity>> + Send + Sync + 'static,
    ) -> Self {
        Self {
            name,
            description,
            supported_types,
            extractor: Box::new(extractor),
            version: "1".to_string(),
            label_extractor: None,
            relation_extractor: None,
        }
    }

    /// Set the version string for cache invalidation.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Attach a `DynamicLabels` implementation via closure.
    ///
    /// When set, `AnyModel` will implement [`DynamicLabels`] by delegating to this
    /// closure, and [`Model::capabilities()`] will report `dynamic_labels = true`.
    #[must_use]
    pub fn with_dynamic_labels(
        mut self,
        f: impl Fn(&str, &[&str], Option<Language>) -> Result<Vec<Entity>> + Send + Sync + 'static,
    ) -> Self {
        self.label_extractor = Some(Box::new(f));
        self
    }

    /// Attach a `RelationCapable` implementation via closure.
    ///
    /// When set, `AnyModel` will implement [`RelationCapable`] by delegating to this
    /// closure, and [`Model::capabilities()`] will report `relation_capable = true`.
    #[must_use]
    pub fn with_relations(
        mut self,
        f: impl Fn(&str, Option<Language>) -> Result<(Vec<Entity>, Vec<Relation>)>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        self.relation_extractor = Some(Box::new(f));
        self
    }
}

impl std::fmt::Debug for AnyModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnyModel")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("supported_types", &self.supported_types)
            .finish()
    }
}

// AnyModel gets the Sealed impl so it can implement Model
impl sealed::Sealed for AnyModel {}

impl Model for AnyModel {
    fn extract_entities(&self, text: &str, language: Option<Language>) -> Result<Vec<Entity>> {
        (self.extractor)(text, language)
    }

    fn supported_types(&self) -> Vec<EntityType> {
        self.supported_types.clone()
    }

    fn is_available(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        self.name
    }

    fn description(&self) -> &'static str {
        self.description
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            dynamic_labels: self.label_extractor.is_some(),
            relation_capable: self.relation_extractor.is_some(),
            ..ModelCapabilities::default()
        }
    }

    fn version(&self) -> String {
        self.version.clone()
    }
}

impl DynamicLabels for AnyModel {
    fn extract_with_labels(
        &self,
        text: &str,
        labels: &[&str],
        language: Option<Language>,
    ) -> Result<Vec<Entity>> {
        match &self.label_extractor {
            Some(f) => f(text, labels, language),
            None => Err(Error::FeatureNotAvailable(
                "AnyModel: DynamicLabels closure not configured (use .with_dynamic_labels())"
                    .into(),
            )),
        }
    }
}

impl RelationCapable for AnyModel {
    fn extract_with_relations(
        &self,
        text: &str,
        language: Option<Language>,
    ) -> Result<(Vec<Entity>, Vec<Relation>)> {
        match &self.relation_extractor {
            Some(f) => f(text, language),
            None => Err(Error::FeatureNotAvailable(
                "AnyModel: RelationCapable closure not configured (use .with_relations())".into(),
            )),
        }
    }
}

// =============================================================================
// Capability Discovery for Trait Objects
// =============================================================================

/// Runtime discovery mechanism for model capabilities behind `Box<dyn Model>`.
///
/// Surfaces capability information through [`Model::capabilities()`],
/// making it available for any `&dyn Model` without downcasting.
///
/// # Example
///
/// ```rust,ignore
/// use anno::{Model, ModelCapabilities};
///
/// fn process_with_model(model: &dyn Model) {
///     let caps = model.capabilities();
///
///     if caps.relation_capable {
///         println!("Model supports relation extraction");
///     }
///     if caps.dynamic_labels {
///         println!("Model supports zero-shot entity types");
///     }
/// }
/// ```
#[derive(Debug, Clone, Default)]
pub struct ModelCapabilities {
    /// True if the model supports relation extraction.
    pub relation_capable: bool,
    /// True if the model supports zero-shot, caller-supplied entity types.
    pub dynamic_labels: bool,
    /// True if the model can extract discontinuous entities spanning non-adjacent spans.
    /// Only `W2NER` (when loaded with an ONNX session) sets this today.
    pub discontinuous_capable: bool,
}

/// Trait for models that can extract relations between entities.
///
/// Models implementing this trait can jointly extract entities and their relationships,
/// producing (head, relation_type, tail) triples.
pub trait RelationCapable: Model {
    /// Extract entities and their relations from text.
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to extract from
    /// * `language` - Optional language hint (e.g., "en", "es")
    ///
    /// # Returns
    ///
    /// A tuple of (entities, relations) where relations link entities together.
    fn extract_with_relations(
        &self,
        text: &str,
        language: Option<Language>,
    ) -> Result<(Vec<Entity>, Vec<Relation>)>;
}

/// Trait for models that support dynamic/zero-shot entity type specification.
///
/// Models implementing this trait can extract entities of arbitrary types
/// specified at inference time (e.g., GLiNER, UniversalNER), rather than
/// being limited to a fixed set of pre-trained types.
///
/// # Relationship to `ZeroShotNER`
///
/// `DynamicLabels` is the sealed, `Model`-hierarchy trait for zero-shot extraction.
/// [`ZeroShotNER`](backends::inference::ZeroShotNER) is the open, backend-internal trait
/// with additional parameters (threshold, descriptions). Backends that support zero-shot
/// typically implement both: `DynamicLabels` delegates to `ZeroShotNER` with a default
/// threshold. Use `DynamicLabels` in application code; use `ZeroShotNER` only when you
/// need direct threshold control.
///
/// # Label count warning
///
/// GLiNER-family models degrade significantly when more than ~30 labels are
/// passed in a single call. The label-token cross-attention matrix grows
/// quadratically, causing both latency spikes and accuracy drops (the model
/// was trained with at most ~20-30 types per batch). If you need >30 types,
/// split them into batches and merge results with span-level deduplication.
pub trait DynamicLabels: Model {
    /// Extract entities with custom type labels.
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to extract from
    /// * `labels` - Custom entity type labels to extract (e.g., ["PERSON", "ORGANIZATION"]).
    ///   Passing more than 30 labels may degrade accuracy for GLiNER-family models;
    ///   consider batching labels if the count exceeds this threshold.
    /// * `language` - Optional language hint (e.g., "en", "es")
    ///
    /// # Returns
    ///
    /// Entities of the specified types found in the text.
    fn extract_with_labels(
        &self,
        text: &str,
        labels: &[&str],
        language: Option<Language>,
    ) -> Result<Vec<Entity>>;
}

// Re-export backends
pub use backends::{
    AutoNER, ConflictStrategy, CrfNER, EnsembleNER, HeuristicNER, LexiconNER, NuNER, RegexNER,
    StackedNER, TPLinker, W2NERConfig, W2NERRelation, W2NER,
};

// Mention-ranking coreference (Bourgois & Poibeau 2025)
pub use backends::coref::mention_ranking::{
    ClusteringStrategy, MentionCluster, MentionRankingConfig, MentionRankingCoref, RankedMention,
};

// Unified coref backend trait (open, not sealed)
pub use backends::CorefBackend;

// Re-export MockModel for testing

// Re-export inference traits and types used at the crate root
pub use backends::inference::{
    extract_relation_triples, extract_relation_triples_simple, extract_relations,
    CoreferenceConfig, DiscontinuousEntity, DiscontinuousNER, ExtractionWithRelations,
    RelationExtractionConfig, RelationExtractor, RelationTriple,
};

#[cfg(feature = "onnx")]
#[cfg_attr(docsrs, doc(cfg(feature = "onnx")))]
pub use backends::{BertNEROnnx, GLiNEROnnx};

#[cfg(feature = "onnx")]
#[cfg_attr(docsrs, doc(cfg(feature = "onnx")))]
pub use backends::{FCoref, FCorefConfig};

#[cfg(feature = "candle")]
#[cfg_attr(docsrs, doc(cfg(feature = "candle")))]
pub use backends::CandleNER;

// Constants

/// Default BERT ONNX model identifier (HuggingFace model ID).
pub const DEFAULT_BERT_ONNX_MODEL: &str = "protectai/bert-base-NER-onnx";

/// Default GLiNER ONNX model identifier (HuggingFace model ID).
pub const DEFAULT_GLINER_MODEL: &str = "onnx-community/gliner_small-v2.1";

/// Default GLiNER2 ONNX model identifier (HuggingFace model ID).
pub const DEFAULT_GLINER2_MODEL: &str = "onnx-community/gliner-multitask-large-v0.5";

/// Default Candle model identifier (HuggingFace model ID).
/// Uses dbmdz's model which has both tokenizer.json and safetensors.
pub const DEFAULT_CANDLE_MODEL: &str = "dslim/bert-base-NER";

/// Default GLiNER Candle model identifier (HuggingFace model ID).
/// Uses a model with tokenizer.json and pytorch_model.bin for Candle compatibility.
/// The backend converts pytorch_model.bin to safetensors automatically.
// NeuML/gliner-bert-tiny uses BERT (not DeBERTa) which is compatible with CandleEncoder
// Other GLiNER models use DeBERTa-v3 which has different architecture (relative attention)
pub const DEFAULT_GLINER_CANDLE_MODEL: &str = "NeuML/gliner-bert-tiny";

/// Default NuNER ONNX model identifier (HuggingFace model ID).
pub const DEFAULT_NUNER_MODEL: &str = "deepanwa/NuNerZero_onnx";

/// Default GLiNER Poly-Encoder ONNX model identifier (HuggingFace model ID).
///
/// Uses the bi-encoder large v1.0 (DeBERTa text + BGE label encoder).
pub const DEFAULT_GLINER_POLY_MODEL: &str = "knowledgator/gliner-bi-large-v1.0";

/// Default W2NER ONNX model identifier (HuggingFace model ID).
pub const DEFAULT_W2NER_MODEL: &str = "ljynlp/w2ner-bert-base";

/// Automatically select the best available NER backend.
pub fn auto() -> Result<Box<dyn Model>> {
    #[cfg(feature = "onnx")]
    {
        if let Ok(model) = GLiNEROnnx::new(DEFAULT_GLINER_MODEL) {
            return Ok(Box::new(model));
        }
        if let Ok(model) = BertNEROnnx::new(DEFAULT_BERT_ONNX_MODEL) {
            return Ok(Box::new(model));
        }
    }
    #[cfg(feature = "candle")]
    {
        if let Ok(model) = CandleNER::from_pretrained(DEFAULT_CANDLE_MODEL) {
            return Ok(Box::new(model));
        }
    }
    Ok(Box::new(StackedNER::default()))
}

/// Check which backends are currently available.
///
/// Derives the list from [`backends::catalog::BACKEND_CATALOG`] so every cataloged
/// backend is always shown, with availability determined by compiled feature flags.
pub fn available_backends() -> Vec<(&'static str, bool)> {
    use backends::catalog::BACKEND_CATALOG;

    BACKEND_CATALOG
        .iter()
        .map(|info| {
            let available = match info.feature {
                None => true,
                Some("onnx") => cfg!(feature = "onnx"),
                Some("candle") => cfg!(feature = "candle"),
                Some("llm") => cfg!(feature = "llm"),
                // Unknown/planned feature gates -- not yet in Cargo.toml.
                Some(_) => false,
            };
            (info.name, available)
        })
        .collect()
}

/// A mock NER model for testing purposes.
///
/// This is provided so tests can create custom mock implementations
/// without breaking the sealed trait pattern.
///
/// # Entity Validation
///
/// By default, `extract_entities` validates that entity offsets are within
/// the input text bounds and that `start < end`. Set `validate = false`
/// to disable this (useful for testing error handling).
///
/// # Example
///
/// ```rust
/// use anno::{MockModel, Entity, EntityType, Result};
///
/// let mock = MockModel::new("test-mock")
///     .with_entities(vec![
///         Entity::new("John", EntityType::Person, 0, 4, 0.9),
///     ]);
///
/// // Use mock in tests
/// ```
#[derive(Clone)]
pub struct MockModel {
    /// Model name identifier.
    name: &'static str,
    /// Entities to return when `extract_entities` is called.
    entities: Vec<Entity>,
    /// Supported entity types for this mock model.
    types: Vec<EntityType>,
    /// If true, validate entity offsets against input text (default: true).
    validate: bool,
}

impl MockModel {
    /// Create a new mock model.
    #[must_use]
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            entities: Vec::new(),
            types: Vec::new(),
            validate: true,
        }
    }

    /// Set entities to return on extraction.
    ///
    /// # Panics
    ///
    /// Panics if any entity has `start >= end`.
    #[must_use]
    pub fn with_entities(mut self, entities: Vec<Entity>) -> Self {
        // Basic validation on construction
        for (i, e) in entities.iter().enumerate() {
            assert!(
                e.start() < e.end(),
                "MockModel entity {}: start ({}) must be < end ({})",
                i,
                e.start(),
                e.end()
            );
            assert!(
                e.confidence >= 0.0 && e.confidence <= 1.0,
                "MockModel entity {}: confidence ({}) must be in [0.0, 1.0]",
                i,
                e.confidence
            );
        }
        self.entities = entities;
        self
    }

    /// Set supported entity types.
    #[must_use]
    pub fn with_types(mut self, types: Vec<EntityType>) -> Self {
        self.types = types;
        self
    }

    /// Disable offset validation during extraction (for testing error paths).
    #[must_use]
    pub fn without_validation(mut self) -> Self {
        self.validate = false;
        self
    }

    /// Validate that entity offsets are within text bounds.
    fn validate_entities(&self, text: &str) -> Result<()> {
        // Performance optimization: Cache text length (called once, used for all entities)
        let text_len = text.chars().count();
        for (i, e) in self.entities.iter().enumerate() {
            if e.end() > text_len {
                return Err(Error::InvalidInput(format!(
                    "MockModel entity {} '{}': end offset ({}) exceeds text length ({} chars)",
                    i,
                    e.text,
                    e.end(),
                    text_len
                )));
            }
            // Verify text matches (using char offsets)
            // Use optimized extract_text_with_len to avoid recalculating length
            let actual_text = e.extract_text_with_len(text, text_len);
            if actual_text != e.text {
                return Err(Error::InvalidInput(format!(
                    "MockModel entity {} text mismatch: expected '{}' at [{},{}), found '{}'",
                    i,
                    e.text,
                    e.start(),
                    e.end(),
                    actual_text
                )));
            }
        }
        Ok(())
    }
}

impl Model for MockModel {
    fn extract_entities(&self, text: &str, _language: Option<Language>) -> Result<Vec<Entity>> {
        if self.validate && !self.entities.is_empty() {
            self.validate_entities(text)?;
        }
        Ok(self.entities.clone())
    }

    fn supported_types(&self) -> Vec<EntityType> {
        self.types.clone()
    }

    fn is_available(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        self.name
    }

    fn description(&self) -> &'static str {
        "Mock NER model for testing"
    }
}

// CI matrix harness moved to `anno-eval`.

#[cfg(test)]
mod any_model_tests {
    use super::*;

    fn base_any_model() -> AnyModel {
        AnyModel::new(
            "test-any",
            "test model",
            vec![EntityType::Person],
            |_text, _lang| Ok(vec![]),
        )
    }

    #[test]
    fn any_model_capabilities_default_no_dynamic_no_relations() {
        let m = base_any_model();
        let caps = m.capabilities();
        assert!(
            !caps.dynamic_labels,
            "should not report dynamic_labels without closure"
        );
        assert!(
            !caps.relation_capable,
            "should not report relation_capable without closure"
        );
    }

    #[test]
    fn any_model_dynamic_labels_returns_entities() {
        let m = base_any_model().with_dynamic_labels(|_text, labels, _lang| {
            Ok(labels
                .iter()
                .enumerate()
                .map(|(i, &lbl)| {
                    Entity::new(
                        lbl,
                        EntityType::custom(lbl, EntityCategory::Misc),
                        i,
                        i + 1,
                        0.8,
                    )
                })
                .collect())
        });
        assert!(m.capabilities().dynamic_labels);
        let ents = m
            .extract_with_labels("hello world", &["GREETING", "NOUN"], None)
            .unwrap();
        assert_eq!(ents.len(), 2);
        assert_eq!(ents[0].text, "GREETING");
        assert_eq!(ents[1].text, "NOUN");
    }

    #[test]
    fn any_model_dynamic_labels_missing_returns_feature_not_available() {
        let m = base_any_model();
        let err = m.extract_with_labels("hello", &["X"], None).unwrap_err();
        assert!(
            matches!(err, Error::FeatureNotAvailable(_)),
            "expected FeatureNotAvailable, got: {err:?}"
        );
    }

    #[test]
    fn any_model_relations_returns_entities_and_relations() {
        let m = base_any_model().with_relations(|_text, _lang| {
            let head = Entity::new("Alice", EntityType::Person, 0, 5, 0.9);
            let tail = Entity::new("Acme", EntityType::Organization, 15, 19, 0.85);
            let rel = Relation::new(head.clone(), tail.clone(), "WORKS_AT", 0.8);
            Ok((vec![head, tail], vec![rel]))
        });
        assert!(m.capabilities().relation_capable);
        let (ents, rels) = m
            .extract_with_relations("Alice works at Acme Corp", None)
            .unwrap();
        assert_eq!(ents.len(), 2);
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].relation_type, "WORKS_AT");
    }

    #[test]
    fn any_model_relations_missing_returns_feature_not_available() {
        let m = base_any_model();
        let err = m.extract_with_relations("hello", None).unwrap_err();
        assert!(
            matches!(err, Error::FeatureNotAvailable(_)),
            "expected FeatureNotAvailable, got: {err:?}"
        );
    }
}
