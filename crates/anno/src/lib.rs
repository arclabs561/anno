//! # anno
//!
//! Information extraction for unstructured text: named entity recognition (NER),
//! coreference resolution, relation extraction, PII detection, and zero-shot entity types.
//!
//! - **NER output**: variable-length spans with **character offsets** (Unicode scalar values), not
//!   byte offsets.
//! - **Coreference output**: clusters (“tracks”) of mentions within one document.
//! - **Relation output**: `(head, relation, tail)` triples via [`RelationCapable`] backends.
//! - **PII detection**: [`pii`] module for detecting and redacting personally identifiable information.
//! - **RAG preprocessing**: [`rag::preprocess`] chunks text, extracts entities, and rewrites pronouns
//!   for self-contained retrieval chunks.
//! - **Export**: [`export`] module for brat, CoNLL, JSONL, N-Triples, JSON-LD, and graph CSV.
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
mod annotated;
pub mod backends;
/// Discourse-level analysis: centering theory, abstract anaphora, dialogue acts.
///
/// Enable with the `discourse` feature.
///
/// See `discourse::centering` for salience-based pronoun resolution and
/// `discourse::uncertain_reference` for handling ambiguous references.
#[cfg(feature = "discourse")]
#[cfg_attr(docsrs, doc(cfg(feature = "discourse")))]
pub mod discourse;
/// Edit distance algorithms.
pub mod edit_distance;
pub mod env;
pub mod error;
/// Evaluation/analysis primitives (coref metrics, cluster encoders, etc.).
///
/// This module is only available when the `analysis` feature is enabled.
#[cfg(feature = "analysis")]
#[cfg_attr(docsrs, doc(cfg(feature = "analysis")))]
pub mod eval;
/// Export entity results to annotation and interchange formats (brat, CoNLL, JSONL, RDF, JSON-LD, CSV).
pub mod export;
/// Small, dependency-light heuristics (negation, quantifiers, etc.).
pub mod heuristics;
/// Lightweight URL/file ingestion helpers (not a crawling/pipeline product).
pub mod ingest;
pub mod lang;
pub mod offset;
/// PII detection and redaction (library-level privacy functions).
pub mod pii;
/// Coreference preprocessing for RAG: rewrite pronouns for self-contained chunks.
///
/// See [`rag::resolve_for_rag`] for the main entry point.
pub mod rag;
pub mod schema;
pub mod similarity;
pub mod types;

// Note: research-only geometry experiments were archived out of `anno` to keep the public
// surface grounded. Prefer `docs/` for repo-local design notes and experiments.

// Re-export error types
pub use error::{Error, Result};

// =============================================================================
// Core types live in `anno-core`
// =============================================================================

// Re-export core types at the crate root (the `anno` public API surface).
pub use anno_core::{
    generate_span_candidates, CanonicalId, Confidence, CorefChain, CorefDocument,
    CoreferenceResolver, Corpus, DiscontinuousSpan, Entity, EntityBuilder, EntityCategory,
    EntityType, ExtractionMethod, Gender, GroundedDocument, HashMapLexicon, HierarchicalConfidence,
    Identity, IdentityId, IdentitySource, Lexicon, Location, Mention, MentionType, Modality,
    Number, Person, Provenance, Quantifier, RaggedBatch, Relation, Signal, SignalId, SignalRef,
    Span, SpanCandidate, Track, TrackId, TrackRef, TrackStats, TypeLabel, TypeMapper,
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
pub use types::EntitySliceExt;

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
    impl Sealed for super::backends::heuristic_crf::HeuristicCrfNER {}
    #[cfg(test)]
    impl Sealed for super::MockModel {}
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
///     "Custom NER backend",
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
    /// (relations, zero-shot types, discontinuous entities) to enable runtime discovery.
    ///
    /// # Default
    ///
    /// Returns a [`ModelCapabilities`] with all fields set to `false`/`None`.
    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities::default()
    }

    /// Extract entities from multiple texts.
    ///
    /// The default implementation calls [`extract_entities`](Self::extract_entities)
    /// sequentially. ONNX backends can override this with internal batching for
    /// better throughput.
    ///
    /// Each element in the returned `Vec` is independent: a failure on one text
    /// does not affect the others.
    fn extract_batch(
        &self,
        texts: &[&str],
        language: Option<Language>,
    ) -> Vec<Result<Vec<Entity>>> {
        texts
            .iter()
            .map(|t| self.extract_entities(t, language))
            .collect()
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

/// Type alias for the `AnyModel` zero-shot extraction closure (`ZeroShotNER`).
type AnyModelZeroShotExtractor = dyn Fn(&str, &[&str], f32) -> Result<Vec<Entity>> + Send + Sync;

/// Type alias for the `AnyModel` relation-extraction closure.
type AnyModelRelationExtractor =
    dyn Fn(&str, Option<Language>) -> Result<(Vec<Entity>, Vec<Relation>)> + Send + Sync;

/// A wrapper that turns an extractor closure into a `Model`.
///
/// `AnyModel` supports [`ZeroShotNER`] and
/// relation extraction via closures (see [`with_zero_shot`](Self::with_zero_shot)
/// and [`with_relations`](Self::with_relations)).
pub struct AnyModel {
    name: &'static str,
    description: &'static str,
    supported_types: Vec<EntityType>,
    extractor: Box<AnyModelExtractor>,
    version: String,
    /// Optional closure backing [`ZeroShotNER::extract_with_types`](backends::inference::ZeroShotNER::extract_with_types).
    zero_shot_extractor: Option<Box<AnyModelZeroShotExtractor>>,
    /// Optional closure backing relation extraction (deprecated [`RelationCapable`]).
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
            zero_shot_extractor: None,
            relation_extractor: None,
        }
    }

    /// Set the version string for cache invalidation.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Attach a [`ZeroShotNER`] implementation via closure.
    ///
    /// When set, `AnyModel` will implement `ZeroShotNER` by delegating to this
    /// closure, and [`Model::capabilities()`] will report `zero_shot = true`.
    #[must_use]
    pub fn with_zero_shot(
        mut self,
        f: impl Fn(&str, &[&str], f32) -> Result<Vec<Entity>> + Send + Sync + 'static,
    ) -> Self {
        self.zero_shot_extractor = Some(Box::new(f));
        self
    }

    /// Attach a relation extraction implementation via closure.
    ///
    /// When set, `AnyModel` will support relation extraction by delegating to this
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
            zero_shot: self.zero_shot_extractor.is_some(),
            relation_capable: self.relation_extractor.is_some(),
            ..ModelCapabilities::default()
        }
    }

    fn version(&self) -> String {
        self.version.clone()
    }
}

impl backends::inference::ZeroShotNER for AnyModel {
    fn extract_with_types(
        &self,
        text: &str,
        entity_types: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        match &self.zero_shot_extractor {
            Some(f) => f(text, entity_types, threshold),
            None => Err(Error::FeatureNotAvailable(
                "AnyModel: ZeroShotNER closure not configured (use .with_zero_shot())".into(),
            )),
        }
    }

    fn extract_with_descriptions(
        &self,
        text: &str,
        descriptions: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        // Descriptions are treated the same as types for closure-based backends.
        self.extract_with_types(text, descriptions, threshold)
    }

    fn default_types(&self) -> &[&'static str] {
        &[]
    }
}

#[allow(deprecated)]
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
///     if caps.zero_shot {
///         println!("Model supports zero-shot entity types");
///     }
/// }
/// ```
#[derive(Debug, Clone, Default)]
pub struct ModelCapabilities {
    /// True if the model supports relation extraction.
    pub relation_capable: bool,
    /// True if the model supports zero-shot, caller-supplied entity types
    /// via [`ZeroShotNER`].
    pub zero_shot: bool,
    /// True if the model can extract discontinuous entities spanning non-adjacent spans.
    /// Only `W2NER` (when loaded with an ONNX session) sets this today.
    pub discontinuous_capable: bool,
}

/// Convenience trait for models that extract relations with default types.
///
/// Prefer [`RelationExtractor::extract_relations_default`]
/// for new code. This trait exists for backward compatibility and will be
/// removed in a future version.
#[deprecated(
    since = "0.4.0",
    note = "use RelationExtractor::extract_relations_default instead; will be removed in 0.5.0"
)]
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

// Re-export backends
pub use backends::{
    ConflictStrategy, CrfNER, EnsembleNER, HeuristicNER, LexiconNER, NuNER, RegexNER, StackedNER,
    TPLinker, W2NERConfig, W2NERRelation, W2NER,
};

// Mention-ranking coreference (Bourgois & Poibeau 2025)
pub use backends::coref::mention_ranking::{
    ClusteringStrategy, MentionCluster, MentionRankingConfig, MentionRankingCoref, RankedMention,
};

// Unified coref backend trait (open, not sealed)
pub use backends::CorefBackend;

// Re-export inference traits and types used at the crate root
pub use backends::inference::{
    extract_relation_triples, extract_relation_triples_simple, extract_relations,
    CoreferenceConfig, DiscontinuousEntity, DiscontinuousNER, ExtractionWithRelations,
    RelationExtractionConfig, RelationExtractor, RelationTriple, ZeroShotNER,
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

// =============================================================================
// Convenience API
// =============================================================================

/// Extract entities from text using the best available backend.
///
/// This is a one-liner convenience function. For control over which backend
/// to use, construct a specific model (e.g., [`StackedNER`], [`GLiNEROnnx`]).
///
/// ```rust
/// let entities = anno::extract("Marie Curie won the Nobel Prize.")?;
/// for e in &entities {
///     println!("{} [{}]", e.text, e.entity_type);
/// }
/// # Ok::<(), anno::Error>(())
/// ```
pub fn extract(text: &str) -> Result<Vec<Entity>> {
    let model = StackedNER::default();
    model.extract_entities(text, None)
}

/// Extract entities from multiple texts using the best available backend.
///
/// Batch counterpart to [`extract()`]. Each result is independent: a failure
/// on one text does not prevent others from succeeding.
///
/// ```rust
/// let results = anno::extract_batch(&[
///     "Marie Curie won the Nobel Prize.",
///     "Ada Lovelace wrote the first program.",
/// ]);
/// assert_eq!(results.len(), 2);
/// # Ok::<(), anno::Error>(())
/// ```
pub fn extract_batch(texts: &[&str]) -> Vec<Result<Vec<Entity>>> {
    let model = StackedNER::default();
    model.extract_batch(texts, None)
}

pub use annotated::annotate;
pub use annotated::AnnotatedDoc;

// =============================================================================
// Prelude
// =============================================================================

/// Common imports for working with anno.
///
/// ```rust
/// use anno::prelude::*;
///
/// let m = StackedNER::default();
/// let ents = m.extract_entities("Marie Curie won the Nobel Prize.", None)?;
/// let people: Vec<_> = ents.of_type(&EntityType::Person).collect();
/// let confident: Vec<_> = ents.above_confidence(0.8).collect();
/// # Ok::<(), anno::Error>(())
/// ```
pub mod prelude {
    pub use crate::types::EntitySliceExt;
    pub use crate::{
        AnnotatedDoc, Confidence, Entity, EntityType, Error, Language, Model, Result, StackedNER,
    };
}

// =============================================================================
// Model IDs (backend-internal, re-exported for direct backend construction)
// =============================================================================

/// Default model identifiers for backend construction.
///
/// These are only needed when constructing backends directly (e.g.,
/// `BertNEROnnx::new(models::BERT_ONNX)`). Users of [`StackedNER`] or
/// [`extract()`] do not need these.
pub mod models {
    /// BERT ONNX model (HuggingFace).
    pub const BERT_ONNX: &str = "protectai/bert-base-NER-onnx";
    /// GLiNER ONNX model (HuggingFace).
    pub const GLINER: &str = "onnx-community/gliner_small-v2.1";
    /// GLiNER2 ONNX model (HuggingFace).
    pub const GLINER2: &str = "onnx-community/gliner-multitask-large-v0.5";
    /// BERT Candle model (HuggingFace).
    pub const CANDLE: &str = "dslim/bert-base-NER";
    /// GLiNER Candle model (HuggingFace, BERT-based).
    ///
    /// Uses the same underlying weights as [`GLINER`] for quality parity.
    /// First load may require Python (torch + safetensors) for format conversion
    /// if HuggingFace hasn't auto-generated safetensors for this repo.
    pub const GLINER_CANDLE: &str = "urchade/gliner_small-v2.1";
    /// NuNER ONNX model (HuggingFace).
    pub const NUNER: &str = "deepanwa/NuNerZero_onnx";
    /// GLiNER Poly-Encoder ONNX model (HuggingFace).
    pub const GLINER_POLY: &str = "knowledgator/gliner-bi-large-v1.0";
    /// W2NER ONNX model (HuggingFace).
    pub const W2NER: &str = "ljynlp/w2ner-bert-base";
    /// B2NER model (COLING 2025, trained on 54 unified NER datasets).
    /// Note: only LLM-scale models (7B/20B LoRA) are on HuggingFace as of 2026-03.
    /// Encoder-scale weights pending release.
    pub const B2NER: &str = "Umean/B2NER-Internlm2.5-7B-LoRA";
    /// GLiNER PII Edge (60+ PII categories, zero-shot).
    pub const GLINER_PII: &str = "knowledgator/gliner-pii-edge-v1.0";
    /// GLiNER-RelEx (joint NER + relation extraction, zero-shot).
    pub const GLINER_RELEX: &str = "knowledgator/gliner-relex-large-v1.0";
    /// GLiNER bi-encoder base model (HuggingFace, Feb 2026).
    ///
    /// Bi-encoder architecture pre-computes label embeddings, giving ~130x
    /// speedup at high label counts compared to cross-encoder GLiNER.
    pub const GLINER_BI_BASE: &str = "knowledgator/gliner-bi-base-v2.0";
    /// GLiNER bi-encoder large model (HuggingFace, Feb 2026).
    pub const GLINER_BI_LARGE: &str = "knowledgator/gliner-bi-large-v2.0";
    /// NuNER Zero 4k-context model (HuggingFace).
    pub const NUNER_ZERO_4K: &str = "numind/NuNER_Zero-4k";
    /// NuNER Zero span-level model (HuggingFace).
    pub const NUNER_ZERO_SPAN: &str = "numind/NuNER_Zero-span";
}

// Backward-compat aliases (hidden from docs).
#[doc(hidden)]
pub const DEFAULT_BERT_ONNX_MODEL: &str = models::BERT_ONNX;
#[doc(hidden)]
pub const DEFAULT_GLINER_MODEL: &str = models::GLINER;
#[doc(hidden)]
pub const DEFAULT_GLINER2_MODEL: &str = models::GLINER2;
#[doc(hidden)]
pub const DEFAULT_CANDLE_MODEL: &str = models::CANDLE;
#[doc(hidden)]
pub const DEFAULT_GLINER_CANDLE_MODEL: &str = models::GLINER_CANDLE;
#[doc(hidden)]
pub const DEFAULT_NUNER_MODEL: &str = models::NUNER;
#[doc(hidden)]
pub const DEFAULT_GLINER_POLY_MODEL: &str = models::GLINER_POLY;
#[doc(hidden)]
pub const DEFAULT_W2NER_MODEL: &str = models::W2NER;

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
/// Mock NER model for testing.
///
/// By default, `extract_entities` validates that entity offsets are within
/// the input text bounds and that `start < end`. Set `validate = false`
/// to disable this (useful for testing error handling).
#[cfg(test)]
#[derive(Clone)]
pub(crate) struct MockModel {
    /// Model name identifier.
    name: &'static str,
    /// Entities to return when `extract_entities` is called.
    entities: Vec<Entity>,
    /// Supported entity types for this mock model.
    types: Vec<EntityType>,
    /// If true, validate entity offsets against input text (default: true).
    validate: bool,
}

#[cfg(test)]
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

#[cfg(test)]
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
    fn any_model_capabilities_default_no_zero_shot_no_relations() {
        let m = base_any_model();
        let caps = m.capabilities();
        assert!(
            !caps.zero_shot,
            "should not report zero_shot without closure"
        );
        assert!(
            !caps.relation_capable,
            "should not report relation_capable without closure"
        );
    }

    #[test]
    fn any_model_zero_shot_returns_entities() {
        let m = base_any_model().with_zero_shot(|_text, types, _threshold| {
            Ok(types
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
        assert!(m.capabilities().zero_shot);
        let ents = m
            .extract_with_types("hello world", &["GREETING", "NOUN"], 0.5)
            .unwrap();
        assert_eq!(ents.len(), 2);
        assert_eq!(ents[0].text, "GREETING");
        assert_eq!(ents[1].text, "NOUN");
    }

    #[test]
    fn any_model_zero_shot_missing_returns_feature_not_available() {
        let m = base_any_model();
        let ents: Result<Vec<Entity>> = m.extract_with_types("hello", &["X"], 0.5);
        let err = ents.unwrap_err();
        assert!(
            matches!(err, Error::FeatureNotAvailable(_)),
            "expected FeatureNotAvailable, got: {err:?}"
        );
    }

    #[test]
    #[allow(deprecated)]
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
    #[allow(deprecated)]
    fn any_model_relations_missing_returns_feature_not_available() {
        let m = base_any_model();
        let err = m.extract_with_relations("hello", None).unwrap_err();
        assert!(
            matches!(err, Error::FeatureNotAvailable(_)),
            "expected FeatureNotAvailable, got: {err:?}"
        );
    }
}

#[cfg(test)]
mod convenience_tests {
    use super::*;

    #[test]
    fn extract_finds_entities() {
        let ents = extract("Marie Curie won the Nobel Prize.").unwrap();
        assert!(!ents.is_empty(), "extract() should find entities");
    }

    #[test]
    fn extract_empty_text() {
        let ents = extract("").unwrap();
        assert!(ents.is_empty());
    }

    #[test]
    fn prelude_imports_work() {
        use crate::prelude::*;
        let m = StackedNER::default();
        let ents = m.extract_entities("Test input", None).unwrap();
        let _: Vec<_> = ents.above_confidence(0.5).collect();
    }
}

#[cfg(test)]
mod batch_tests {
    use super::*;

    #[test]
    fn extract_batch_empty_slice() {
        let results = extract_batch(&[]);
        assert!(results.is_empty());
    }

    #[test]
    fn extract_batch_single_text() {
        let results = extract_batch(&["Marie Curie won the Nobel Prize."]);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_ok());
        assert!(!results[0].as_ref().unwrap().is_empty());
    }

    #[test]
    fn extract_batch_multiple_texts() {
        let results = extract_batch(&[
            "Marie Curie won the Nobel Prize.",
            "Ada Lovelace wrote the first program.",
            "No entities here in this plain sentence.",
        ]);
        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(r.is_ok());
        }
    }

    #[test]
    fn trait_method_extract_batch_empty() {
        let m = StackedNER::default();
        let results = m.extract_batch(&[], None);
        assert!(results.is_empty());
    }

    #[test]
    fn trait_method_extract_batch_count() {
        let m = StackedNER::default();
        let texts = ["Alice", "Bob", "Carol"];
        let results = m.extract_batch(&texts, None);
        assert_eq!(results.len(), 3);
    }
}
