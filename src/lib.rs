//! # anno
//!
//! Information extraction for Rust: NER, coreference resolution, and evaluation.
//!
//! - **NER**: Multiple backends (Pattern, BERT, GLiNER, NuNER, W2NER)
//! - **Coreference**: Resolution (rule-based, T5-based) and metrics (MUC, B³, CEAF, LEA, BLANC)
//! - **Evaluation**: Comprehensive benchmarking framework with bias analysis
//!
//! ## Quick Start - Automatic Backend Selection
//!
//! ```rust,ignore
//! use anno::{auto, Model};
//!
//! // Automatically picks GLiNER (ONNX) → BERT (ONNX) → Candle → Pattern
//! let model = auto()?;
//! let entities = model.extract_entities("Steve Jobs founded Apple", None)?;
//! ```
//!
//! Or use the `NERExtractor` for more control:
//!
//! ```rust,ignore
//! use anno::backends::extractor::NERExtractor;
//!
//! // Best available (GLiNER if onnx feature enabled)
//! let extractor = NERExtractor::best_available();
//!
//! // Explicit speed/quality tradeoffs
//! let fast = NERExtractor::fast();           // GLiNER small or patterns
//! let quality = NERExtractor::best_quality(); // GLiNER large
//! ```
//!
//! ## NER Backends
//!
//! | Backend | Feature | CoNLL-03* | Zero-Shot | Speed | Status |
//! |---------|---------|-----------|-----------|-------|--------|
//! | `RegexNER` | always | N/A | No | ~400ns | ✅ Complete |
//! | `HeuristicNER` | always | ~65% F1 | No | ~50μs | ✅ Complete |
//! | `StackedNER` | always | varies | No | varies | ✅ Complete |
//! | `BertNEROnnx` | `onnx` | ~86% F1 | No | ~50ms | ✅ Complete |
//! | `GLiNEROnnx` | `onnx` | ~90% F1 | **Yes** | ~100ms | ✅ Complete |
//! | `NuNER` | `onnx` | ~86% F1 | **Yes** | ~100ms | ✅ Complete |
//! | `W2NER` | `onnx` | ~85% F1 | No | ~150ms | ✅ Complete |
//! | `CandleNER` | `candle` | ~86% F1 | No | varies | ✅ Complete |
//! | `GLiNERCandle` | `candle` | ~90% F1 | **Yes** | varies | ✅ Complete |
//!
//! *CoNLL-03 is the standard supervised NER benchmark. For zero-shot cross-domain
//! benchmarks (CrossNER), expect ~60% F1. See [`DEFAULT_GLINER2_MODEL`] docs for details.
//!
//! ## Feature Flags
//!
//! ### Evaluation Framework (tiered)
//!
//! ```toml
//! anno = "0.2"                                      # Includes eval (P/R/F1)
//! anno = { version = "0.2", features = ["eval-bias"] }      # + bias analysis
//! anno = { version = "0.2", features = ["eval-advanced"] }  # + calibration, robustness
//! anno = { version = "0.2", features = ["eval-full"] }      # Everything in eval
//! ```
//!
//! ### ML Backends
//!
//! ```toml
//! anno = { version = "0.2", features = ["onnx"] }   # BERT, GLiNER, NuNER via ONNX
//! anno = { version = "0.2", features = ["candle"] } # Pure Rust inference
//! anno = { version = "0.2", features = ["candle", "metal"] } # + Apple GPU
//! anno = { version = "0.2", features = ["candle", "cuda"] }  # + NVIDIA GPU
//! ```
//!
//! ### Discourse Analysis
//!
//! ```toml
//! anno = { version = "0.2", features = ["discourse"] }  # Abstract anaphora, events
//! ```
//!
//! Enables:
//! - Event extraction (ACE-style triggers)
//! - Shell noun detection ("this problem", "the fact that...")
//! - Discourse-aware coreference resolution
//! - Abstract anaphora evaluation
//!
//! ### Everything
//!
//! ```toml
//! anno = { version = "0.2", features = ["full"] }   # All features enabled
//! ```
//!
//! ## Zero-Shot NER (Custom Entity Types)
//!
//! ```rust,ignore
//! use anno::GLiNEROnnx;
//!
//! let model = GLiNEROnnx::new("onnx-community/gliner_small-v2.1")?;
//!
//! // Detect ANY entity type at runtime - no retraining needed!
//! let entities = model.extract(
//!     "CRISPR-Cas9 was developed by Jennifer Doudna at UC Berkeley",
//!     &["technology", "scientist", "university"],
//!     0.5
//! )?;
//! ```
//!
//! ## Simple Pattern Extraction
//!
//! ```rust
//! use anno::{Model, RegexNER};
//!
//! let model = RegexNER::new();
//! let entities = model.extract_entities("Meeting on January 15, 2025", None).unwrap();
//! assert!(entities.iter().any(|e| e.text.contains("January")));
//! ```
//!
//! ## Evaluation
//!
//! - **NER**: P/R/F1 (micro/macro), partial match modes
//! - **Coref**: MUC, B³, CEAF, LEA, BLANC
//! - **Datasets**: CoNLL-2003, WikiGold, WNUT-17, MultiNERD, GAP
//!
//! ## Design Philosophy
//!
//! - **ML-first**: BERT ONNX is the recommended default
//! - **No hardcoded gazetteers**: RegexNER only extracts format-based entities
//! - **Trait-based**: All backends implement the `Model` trait
//! - **Graceful degradation**: Falls back to patterns if ML unavailable
//! - **No stubs**: Every backend has a complete implementation

#![warn(missing_docs)]

pub mod backends;
#[cfg(feature = "discourse")]
pub mod discourse;
mod entity;
mod error;
pub mod eval;
pub mod graph;
pub mod grounded;
/// Language detection and classification utilities.
pub mod lang;
pub mod offset;
pub mod schema;
pub mod similarity;
pub mod types;

// =============================================================================
// Sealed Trait Pattern
// =============================================================================
//
// The Model trait is "sealed" - it can only be implemented by types in this
// crate. This allows us to:
// 1. Add methods to Model in the future without breaking changes
// 2. Rely on internal invariants in generic code
// 3. Prevent downstream users from creating broken implementations
//
// Users can still use Model as a trait bound, they just can't implement it.

mod sealed {
    /// Sealed trait marker. Cannot be implemented outside this crate.
    pub trait Sealed {}

    // Implement Sealed for all built-in backends
    impl Sealed for super::RegexNER {}
    impl Sealed for super::HeuristicNER {}
    impl Sealed for super::StackedNER {}
    impl Sealed for super::NuNER {}
    impl Sealed for super::W2NER {}
    impl Sealed for super::backends::tplinker::TPLinker {}
    #[cfg(feature = "onnx")]
    impl Sealed for super::backends::gliner_poly::GLiNERPoly {}
    #[cfg(feature = "onnx")]
    impl Sealed for super::backends::deberta_v3::DeBERTaV3NER {}
    #[cfg(feature = "onnx")]
    impl Sealed for super::backends::albert::ALBERTNER {}
    impl Sealed for super::backends::universal_ner::UniversalNER {}
    impl Sealed for super::NERExtractor {}

    // Deprecated backends (still need Sealed for backwards compat)
    #[allow(deprecated)]
    impl Sealed for super::backends::rule::RuleBasedNER {}

    #[cfg(feature = "onnx")]
    impl Sealed for super::BertNEROnnx {}

    #[cfg(feature = "onnx")]
    impl Sealed for super::GLiNEROnnx {}

    #[cfg(feature = "candle")]
    impl Sealed for super::CandleNER {}

    #[cfg(feature = "candle")]
    impl Sealed for super::backends::gliner_candle::GLiNERCandle {}

    #[cfg(feature = "candle")]
    impl<E: super::backends::encoder_candle::TextEncoder + 'static> Sealed
        for super::backends::gliner_pipeline::GLiNERPipeline<E>
    {
    }

    // GLiNER2 multi-task model
    #[cfg(feature = "onnx")]
    impl Sealed for super::backends::gliner2::GLiNER2Onnx {}

    #[cfg(feature = "candle")]
    impl Sealed for super::backends::gliner2::GLiNER2Candle {}
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
    name: &'static str,
    entities: Vec<Entity>,
    types: Vec<EntityType>,
    /// If true, validate entity offsets against input text (default: true)
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
                e.start < e.end,
                "MockModel entity {}: start ({}) must be < end ({})",
                i,
                e.start,
                e.end
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
        let text_len = text.chars().count();
        for (i, e) in self.entities.iter().enumerate() {
            if e.end > text_len {
                return Err(Error::InvalidInput(format!(
                    "MockModel entity {} '{}': end offset ({}) exceeds text length ({} chars)",
                    i, e.text, e.end, text_len
                )));
            }
            // Verify text matches (using char offsets)
            let actual_text: String = text.chars().skip(e.start).take(e.end - e.start).collect();
            if actual_text != e.text {
                return Err(Error::InvalidInput(format!(
                    "MockModel entity {} text mismatch: expected '{}' at [{},{}), found '{}'",
                    i, e.text, e.start, e.end, actual_text
                )));
            }
        }
        Ok(())
    }
}

impl sealed::Sealed for MockModel {}

impl Model for MockModel {
    fn extract_entities(&self, text: &str, _language: Option<&str>) -> Result<Vec<Entity>> {
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

pub mod prelude {
    //! Commonly used items, re-exported for convenience.
    //!
    //! ```rust
    //! use anno::prelude::*;
    //!
    //! let ner = StackedNER::default();
    //! let entities = ner.extract_entities("Meeting on Jan 15, 2024", None).unwrap();
    //! for e in entities {
    //!     println!("{}: {}", e.entity_type.as_label(), e.text);
    //! }
    //! ```
    pub use crate::entity::{
        Entity, EntityCategory, EntityType, ExtractionMethod, Provenance, TypeMapper,
    };
    pub use crate::error::{Error, Result};
    pub use crate::types::{Confidence, EntitySliceExt, Score};
    pub use crate::Model;
    pub use crate::{MockModel, RegexNER, StackedNER};

    // Schema harmonization (preferred over TypeMapper for multi-dataset work)
    pub use crate::schema::{CanonicalType, CoarseType, DatasetSchema, SchemaMapper};

    // Graph RAG integration
    pub use crate::graph::{GraphDocument, GraphEdge, GraphExportFormat, GraphNode};

    // Evaluation helpers
    pub use crate::eval::datasets::GoldEntity;
    pub use crate::eval::modes::EvalMode;

    #[cfg(feature = "onnx")]
    pub use crate::{BertNEROnnx, GLiNEROnnx};

    #[cfg(feature = "candle")]
    pub use crate::CandleNER;
}

// Re-exports
pub use entity::{
    generate_filtered_candidates, generate_span_candidates, DiscontinuousSpan, Entity,
    EntityBuilder, EntityCategory, EntityType, EntityViewport, ExtractionMethod, HashMapLexicon,
    HierarchicalConfidence, Lexicon, Provenance, RaggedBatch, Relation, Span, SpanCandidate,
    TypeMapper, ValidationIssue,
};
pub use error::{Error, Result};

// Grounded entity hierarchy (Signal → Track → Identity)
// Research-aligned abstractions for unified detection across modalities
pub use grounded::{
    Corpus, GroundedDocument, Identity, IdentityId, IdentitySource, Location, Modality, Quantifier,
    Signal, SignalId, SignalRef, Track, TrackId, TrackRef,
};
pub use lang::{detect_language, Language};
pub use offset::{
    bytes_to_chars, chars_to_bytes, is_ascii, OffsetMapping, SpanConverter, TextSpan, TokenSpan,
};

// Discourse-level entities (abstract anaphora, event coreference)
#[cfg(feature = "discourse")]
pub use discourse::{
    classify_shell_noun,
    DiscourseReferent,
    DiscourseScope,
    EventCluster,
    EventCorefResolver,
    // Event extraction (NEW - fixes abstract anaphora detection)
    EventExtractor,
    EventExtractorConfig,
    EventMention,
    EventTriggerLexicon,
    ReferentType,
    ShellNoun,
    ShellNounClass,
};

// Backend re-exports (always available)
pub use backends::{
    AutoNER, BackendType, ConflictStrategy, HeuristicNER, NERExtractor, NuNER, RegexNER,
    StackedNER, TPLinker, W2NERConfig, W2NERRelation, W2NER,
};

// Backwards compatibility
#[allow(deprecated)]
pub use backends::StatisticalNER;

// Inference abstractions (research-aligned traits)
pub use backends::inference::{
    cosine_similarity_f32,
    two_stage_retrieval,
    // Core encoder traits
    BiEncoder,
    // Binary embeddings for fast blocking (two-stage retrieval)
    BinaryBlocker,
    BinaryHash,
    // Coreference
    CoreferenceCluster,
    CoreferenceConfig,
    // Supporting types
    DiscontinuousEntity,
    DiscontinuousNER,
    // Late interaction
    DotProductInteraction,
    EncoderOutput,
    ExtractionWithRelations,
    // Handshaking matrix (W2NER-style)
    HandshakingMatrix,
    // Modality
    ImageFormat,
    // Semantic registry
    LabelCategory,
    LabelDefinition,
    LabelEncoder,
    LateInteraction,
    MaxSimInteraction,
    ModalityHint,
    ModalityInput,
    RelationExtractor,
    RelationTriple,
    SemanticRegistry,
    SemanticRegistryBuilder,
    SpanLabelScore,
    // Span representation
    SpanRepConfig,
    SpanRepresentationLayer,
    TextEncoder,
    VisualPosition,
    ZeroShotNER,
};

// Backwards compatibility aliases (deprecated)
#[allow(deprecated)]
pub use backends::{CompositeNER, LayeredNER, RuleBasedNER, TieredNER};

#[cfg(feature = "onnx")]
pub use backends::{BertNERConfig, BertNEROnnx, GLiNERConfig, GLiNEROnnx};

#[cfg(feature = "onnx")]
pub use backends::{CorefCluster, T5Coref, T5CorefConfig};

#[cfg(feature = "async-inference")]
pub use backends::{batch_extract, batch_extract_limited, AsyncNER, IntoAsync};

#[cfg(feature = "session-pool")]
pub use backends::{GLiNERPool, PoolConfig, SessionPool};

// Warmup utilities (always available)
pub use backends::{warmup_model, warmup_with_callback, WarmupConfig, WarmupResult};

#[cfg(feature = "candle")]
pub use backends::CandleNER;

/// Default BERT ONNX model (reliable, widely tested).
pub const DEFAULT_BERT_ONNX_MODEL: &str = "protectai/bert-base-NER-onnx";

/// Default GLiNER model (zero-shot NER) - well-tested, reliable.
///
/// GLiNER v2.1 small is the stable, well-tested version for pure NER tasks.
/// For higher accuracy, use `GLINER_MEDIUM_MODEL` or `GLINER_LARGE_MODEL`.
pub const DEFAULT_GLINER_MODEL: &str = "onnx-community/gliner_small-v2.1";

/// GLiNER small model - fastest inference (~1.6x faster than medium).
/// Use when speed is critical and some accuracy loss is acceptable.
pub const GLINER_SMALL_MODEL: &str = "onnx-community/gliner_small-v2.1";

/// GLiNER medium model - balanced speed/accuracy (recommended default).
pub const GLINER_MEDIUM_MODEL: &str = "onnx-community/gliner_medium-v2.1";

/// GLiNER large model - highest accuracy, slower inference.
/// Use when maximum accuracy is required and resources allow.
pub const GLINER_LARGE_MODEL: &str = "onnx-community/gliner_large-v2.1";

/// Default GLiNER2 model (multi-task: NER + classification + relations).
///
/// GLiNER2 from Fastino Labs (EMNLP 2025, arxiv:2507.18546) supports:
/// - Named entity recognition (CrossNER F1: 0.590)
/// - Text classification (multi-label supported)
/// - Structured data extraction (hierarchical JSON)
///
/// **Why this model?**
/// - Same original author as GLiNER (Urchade Zaratiana)
/// - Official EMNLP 2025 publication
/// - 15x more community adoption than alternatives
/// - Dedicated `gliner2` Python library
/// - Apache 2.0 license, CPU-first design
///
/// **Alternatives:**
/// - `knowledgator/gliner-multitask-large-v0.5`: Higher NER F1 (0.6276),
///   but different authors, less maintained
/// - `fastino/gliner2-large-v1`: Higher accuracy, larger (340M params)
///
/// For pure NER, prefer [`DEFAULT_GLINER_MODEL`] instead.
pub const DEFAULT_GLINER2_MODEL: &str = "fastino/gliner2-base-v1";

/// Default Candle model (BERT-based NER).
/// Note: dslim/bert-base-NER only has vocab.txt, not tokenizer.json.
/// For Candle, consider using a model with tokenizer.json or use BertNEROnnx instead.
pub const DEFAULT_CANDLE_MODEL: &str = "dslim/bert-base-NER";

/// Alternative Candle model with tokenizer.json (if available).
/// Falls back to DEFAULT_CANDLE_MODEL if not found.
pub const ALTERNATIVE_CANDLE_MODEL: &str = "dbmdz/bert-large-cased-finetuned-conll03-english";

/// Default GLiNER model for Candle backend (safetensors format).
/// Note: Most GLiNER models don't have safetensors, so GLiNERCandle may not work.
/// Use GLiNEROnnx (ONNX version) instead, which works with all GLiNER models.
/// This constant is kept for compatibility but may not work with most models.
/// Default GLiNER Candle model.
///
/// Tries models with safetensors format first:
/// 1. knowledgator/modern-gliner-bi-large-v1.0 (has safetensors)
/// 2. knowledgator/gliner-x-small (may need conversion)
pub const DEFAULT_GLINER_CANDLE_MODEL: &str = "knowledgator/modern-gliner-bi-large-v1.0";

/// Default NuNER model (token-based zero-shot).
pub const DEFAULT_NUNER_MODEL: &str = "deepanwa/NuNerZero_onnx";

/// Default W2NER model (nested/discontinuous NER).
///
/// **Note**: This model (`ljynlp/w2ner-bert-base`) currently requires authentication
/// and returns 401 errors. See `PROBLEMS.md` for details and alternatives.
/// The backend factory will skip W2NER if this model cannot be loaded.
pub const DEFAULT_W2NER_MODEL: &str = "ljynlp/w2ner-bert-base";

// =============================================================================
// Automatic Backend Selection
// =============================================================================

/// Use case hints for automatic backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UseCase {
    /// Best quality regardless of speed (prefer ML backends).
    BestQuality,
    /// Fast inference, acceptable quality (prefer patterns + heuristic).
    Fast,
    /// Zero-shot NER with custom entity types (GLiNER/NuNER).
    ZeroShot,
    /// Production deployment (stable, well-tested).
    Production,
    /// Nested/discontinuous entities (W2NER).
    NestedEntities,
}

/// Automatically select the best available NER backend.
///
/// Returns the highest-quality available backend based on enabled features.
/// Priority order:
/// 1. GLiNEROnnx (if `onnx` feature) - best quality, zero-shot
/// 2. BertNEROnnx (if `onnx` feature) - high quality, fixed types
/// 3. CandleNER (if `candle` feature) - pure Rust, GPU capable
/// 4. StackedNER (always) - pattern + heuristic fallback
///
/// # Example
///
/// ```rust
/// use anno::{auto, Model};
///
/// let model = anno::auto().expect("At least StackedNER is always available");
/// let entities = model.extract_entities("John works at Apple", None).unwrap();
/// ```
pub fn auto() -> Result<Box<dyn Model>> {
    auto_for(UseCase::BestQuality)
}

/// Select the best backend for a specific use case.
///
/// # Arguments
/// * `use_case` - Hint about intended usage pattern
///
/// # Example
///
/// ```rust,no_run
/// use anno::{auto_for, UseCase, Model};
///
/// fn main() -> anno::Result<()> {
///     // For zero-shot NER with custom types
///     let model = auto_for(UseCase::ZeroShot)?;
///
///     // For fastest inference  
///     let model = auto_for(UseCase::Fast)?;
///     Ok(())
/// }
/// ```
pub fn auto_for(use_case: UseCase) -> Result<Box<dyn Model>> {
    match use_case {
        UseCase::Fast => {
            // Fast: StackedNER (no ML)
            Ok(Box::new(StackedNER::default()))
        }
        UseCase::ZeroShot => {
            // Zero-shot: GLiNER > NuNER > error
            #[cfg(feature = "onnx")]
            {
                if let Ok(model) = backends::gliner_onnx::GLiNEROnnx::new(DEFAULT_GLINER_MODEL) {
                    return Ok(Box::new(model));
                }
            }
            Err(Error::FeatureNotAvailable(
                "Zero-shot NER requires the 'onnx' feature. \
                 Build with: cargo build --features onnx"
                    .to_string(),
            ))
        }
        UseCase::NestedEntities => {
            // Nested: W2NER (placeholder for now)
            Ok(Box::new(W2NER::default()))
        }
        UseCase::Production | UseCase::BestQuality => {
            // Best quality: GLiNER > BERT ONNX > Candle > Stacked
            #[cfg(feature = "onnx")]
            {
                if let Ok(model) = backends::gliner_onnx::GLiNEROnnx::new(DEFAULT_GLINER_MODEL) {
                    return Ok(Box::new(model));
                }
                if let Ok(model) = backends::onnx::BertNEROnnx::new(DEFAULT_BERT_ONNX_MODEL) {
                    return Ok(Box::new(model));
                }
            }
            #[cfg(feature = "candle")]
            {
                if let Ok(model) =
                    backends::candle::CandleNER::from_pretrained(DEFAULT_CANDLE_MODEL)
                {
                    return Ok(Box::new(model));
                }
            }
            // Fallback to StackedNER (always available)
            Ok(Box::new(StackedNER::default()))
        }
    }
}

/// Check which backends are currently available.
///
/// Returns a list of available backend names and their status.
///
/// # Example
///
/// ```rust
/// let backends = anno::available_backends();
/// for (name, available) in backends {
///     println!("{}: {}", name, if available { "✓" } else { "✗" });
/// }
/// ```
pub fn available_backends() -> Vec<(&'static str, bool)> {
    let mut backends = vec![
        ("RegexNER", true),
        ("HeuristicNER", true),
        ("StackedNER", true),
    ];

    #[cfg(feature = "onnx")]
    {
        backends.push(("BertNEROnnx", true));
        backends.push(("GLiNEROnnx", true));
        backends.push(("NuNER", true));
        backends.push(("W2NER", true));
    }
    #[cfg(not(feature = "onnx"))]
    {
        backends.push(("BertNEROnnx", false));
        backends.push(("GLiNEROnnx", false));
        backends.push(("NuNER", false));
        backends.push(("W2NER", false));
    }

    #[cfg(feature = "candle")]
    {
        backends.push(("CandleNER", true));
        backends.push(("GLiNERCandle", true));
    }
    #[cfg(not(feature = "candle"))]
    {
        backends.push(("CandleNER", false));
        backends.push(("GLiNERCandle", false));
    }

    backends
}

/// Trait for NER model backends.
///
/// All NER backends implement this trait for consistent usage.
///
/// # Sealed Trait
///
/// This trait is **sealed** - it can only be implemented by types defined
/// in the `anno` crate. This allows the library to:
///
/// - Add methods in minor versions without breaking changes
/// - Rely on internal invariants (e.g., confidence scores in [0, 1])
/// - Provide a stable, well-tested set of backends
///
/// If you need custom extraction logic, consider:
///
/// - Using [`StackedNER::builder()`] to compose existing backends
/// - Using [`StackedNER`] to combine pattern + ML backends
/// - Opening an issue if you need a backend that doesn't exist
///
/// # For Testing
///
/// If you need a mock implementation for testing, use the
/// [`MockModel`] type provided by this crate (test feature).
pub trait Model: sealed::Sealed + Send + Sync {
    /// Extract entities from text.
    ///
    /// # Arguments
    /// * `text` - Text to extract entities from
    /// * `language` - Optional language hint (ISO 639-1 code, e.g., "en", "de")
    ///
    /// # Returns
    /// Vector of entities with positions, types, and confidence scores
    fn extract_entities(&self, text: &str, language: Option<&str>) -> Result<Vec<Entity>>;

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
}

// =============================================================================
// Capability Marker Traits
// =============================================================================
//
// These traits document backend capabilities at the type level.
// They don't add new methods - they serve as compile-time documentation
// and enable more precise trait bounds in generic code.

/// Marker trait for models that can process batches efficiently.
///
/// Batch-capable models can process multiple texts in a single call,
/// which is significantly more efficient for GPU-based inference.
///
/// # Example
///
/// ```rust,ignore
/// use anno::{Model, BatchCapable};
///
/// fn run_batch<M: Model + BatchCapable>(model: &M, texts: &[&str]) -> Vec<Vec<Entity>> {
///     model.extract_entities_batch(texts, None).unwrap()
/// }
/// ```
pub trait BatchCapable: Model {
    /// Extract entities from multiple texts in a single batch.
    ///
    /// Default implementation falls back to sequential processing.
    /// Implementations should override for true batch processing.
    fn extract_entities_batch(
        &self,
        texts: &[&str],
        language: Option<&str>,
    ) -> Result<Vec<Vec<Entity>>> {
        texts
            .iter()
            .map(|text| self.extract_entities(text, language))
            .collect()
    }

    /// Get optimal batch size for this model.
    ///
    /// Returns None if batch size should be determined dynamically.
    fn optimal_batch_size(&self) -> Option<usize> {
        None
    }
}

/// Marker trait for models that support GPU acceleration.
///
/// GPU-capable models can optionally use GPU resources for inference.
/// This doesn't guarantee GPU usage - the model may fall back to CPU.
pub trait GpuCapable: Model {
    /// Check if GPU acceleration is currently active.
    fn is_gpu_active(&self) -> bool;

    /// Get the device being used (e.g., "cuda:0", "cpu").
    fn device(&self) -> &str;
}

/// Marker trait for models that support streaming/chunked extraction.
///
/// Streaming-capable models can process text in chunks, yielding entities
/// incrementally. This is useful for very large documents that don't fit
/// in memory or when you want early results.
///
/// # Conceptual Example
///
/// ```rust,ignore
/// // For models implementing StreamingCapable:
/// let chunks = document.as_bytes().chunks(10_000);
/// let mut offset = 0;
/// for chunk in chunks {
///     let text = std::str::from_utf8(chunk)?;
///     let entities = model.extract_entities_streaming(text, offset)?;
///     offset += chunk.len();
/// }
/// ```
pub trait StreamingCapable: Model {
    /// Extract entities from a chunk of text with a base offset.
    ///
    /// The `offset` is added to all entity positions to maintain
    /// correct positions within the full document.
    fn extract_entities_streaming(&self, chunk: &str, offset: usize) -> Result<Vec<Entity>> {
        let entities = self.extract_entities(chunk, None)?;
        Ok(entities
            .into_iter()
            .map(|mut e| {
                e.start += offset;
                e.end += offset;
                e
            })
            .collect())
    }

    /// Recommended chunk size for streaming extraction.
    ///
    /// Returns the optimal number of bytes per chunk for this model.
    /// Default is 10KB which balances memory usage and entity boundary handling.
    fn recommended_chunk_size(&self) -> usize {
        10_000
    }
}

/// Marker trait for models that extract named entities (Person, Org, Location).
///
/// Named entities require ML/context for detection, unlike structured
/// entities (dates, emails, etc.) which can be detected via patterns.
pub trait NamedEntityCapable: Model {}

/// Marker trait for models that extract structured entities via patterns.
///
/// Structured entities include dates, times, money, percentages, emails,
/// URLs, and phone numbers - anything detectable via regex patterns.
pub trait StructuredEntityCapable: Model {}

/// Marker trait for models that can extract relations between entities.
///
/// Relation-capable models perform joint entity-relation extraction,
/// producing knowledge graph triples in addition to entity spans.
pub trait RelationCapable: Model {
    /// Extract entities and relations together.
    fn extract_with_relations(
        &self,
        text: &str,
        language: Option<&str>,
    ) -> Result<(Vec<Entity>, Vec<entity::Relation>)>;
}

/// Marker trait for models that support custom entity type labels.
///
/// Dynamic-label models (like GLiNER) can extract arbitrary entity types
/// specified at runtime, not just a fixed set.
pub trait DynamicLabels: Model {
    /// Extract entities using custom labels.
    ///
    /// # Arguments
    /// * `text` - Text to extract from
    /// * `labels` - Custom entity type labels to detect
    /// * `language` - Optional language hint
    fn extract_with_labels(
        &self,
        text: &str,
        labels: &[&str],
        language: Option<&str>,
    ) -> Result<Vec<Entity>>;
}

// NOTE: CalibratedConfidence and VisualCapable traits were removed as they were
// never implemented. They can be re-added in the future if needed:
//
// - CalibratedConfidence: For models that provide calibrated confidence scores.
//   Calibration evaluation exists in src/eval/calibration.rs but is not exposed
//   as a trait. Re-add if we need trait-based calibration queries.
//
// - VisualCapable: For multi-modal models that process images. The infrastructure
//   exists (ModalityInput, ImageFormat in inference.rs) but no backends implement
//   it yet. Re-add when implementing ColPali or similar visual NER models.
