//! # anno
//!
//! NER for Rust.
//!
//! - **NER**: Multiple backends (Pattern, BERT, GLiNER, NuNER, W2NER)
//! - **Coreference**: Metrics (MUC, B³, CEAF, LEA)
//! - **Evaluation**: Benchmarking against standard datasets
//!
//! ## Quick Start - Automatic Backend Selection
//!
//! ```rust,ignore
//! use anno::{auto, Model};
//!
//! // Automatically picks the best available backend
//! let model = auto()?;
//! let entities = model.extract_entities("Steve Jobs founded Apple", None)?;
//! ```
//!
//! ## NER Backends
//!
//! | Backend | Feature | Quality | Zero-Shot | Speed | Status |
//! |---------|---------|---------|-----------|-------|--------|
//! | `PatternNER` | always | N/A | No | ~400ns | ✅ Complete |
//! | `StatisticalNER` | always | ~65% F1 | No | ~50μs | ✅ Complete |
//! | `StackedNER` | always | varies | No | varies | ✅ Complete |
//! | `BertNEROnnx` | `onnx` | ~86% F1 | No | ~50ms | ✅ Complete |
//! | `GLiNEROnnx` | `onnx` | ~86% F1 | **Yes** | ~100ms | ✅ Complete |
//! | `NuNER` | `onnx` | ~86% F1 | **Yes** | ~100ms | ✅ Complete |
//! | `W2NER` | `onnx` | ~85% F1 | No | ~150ms | ✅ Complete |
//! | `CandleNER` | `candle` | ~86% F1 | No | varies | ✅ Complete |
//! | `GLiNERCandle` | `candle` | ~86% F1 | **Yes** | varies | ✅ Complete |
//!
//! ## Feature Flags
//!
//! ```toml
//! [dependencies]
//! anno = "0.2"                           # PatternNER, StatisticalNER only
//! anno = { version = "0.2", features = ["onnx"] }   # + BERT, GLiNER, NuNER
//! anno = { version = "0.2", features = ["candle"] } # + Pure Rust inference
//! anno = { version = "0.2", features = ["candle", "metal"] } # + Apple GPU
//! anno = { version = "0.2", features = ["candle", "cuda"] }  # + NVIDIA GPU
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
//! use anno::{Model, PatternNER};
//!
//! let model = PatternNER::new();
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
//! - **No hardcoded gazetteers**: PatternNER only extracts format-based entities
//! - **Trait-based**: All backends implement the `Model` trait
//! - **Graceful degradation**: Falls back to patterns if ML unavailable
//! - **No stubs**: Every backend has a complete implementation

#![warn(missing_docs)]

pub mod backends;
mod entity;
mod error;
pub mod eval;
pub mod offset;
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
    impl Sealed for super::PatternNER {}
    impl Sealed for super::StatisticalNER {}
    impl Sealed for super::StackedNER {}
    impl Sealed for super::HybridNER {}
    impl Sealed for super::NuNER {}
    impl Sealed for super::W2NER {}
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
        for super::backends::gliner_pipeline::GLiNERPipeline<E> {}
}

/// A mock NER model for testing purposes.
///
/// This is provided so tests can create custom mock implementations
/// without breaking the sealed trait pattern.
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
}

impl MockModel {
    /// Create a new mock model.
    #[must_use]
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            entities: Vec::new(),
            types: Vec::new(),
        }
    }

    /// Set entities to return on extraction.
    #[must_use]
    pub fn with_entities(mut self, entities: Vec<Entity>) -> Self {
        self.entities = entities;
        self
    }

    /// Set supported entity types.
    #[must_use]
    pub fn with_types(mut self, types: Vec<EntityType>) -> Self {
        self.types = types;
        self
    }
}

impl sealed::Sealed for MockModel {}

impl Model for MockModel {
    fn extract_entities(&self, _text: &str, _language: Option<&str>) -> Result<Vec<Entity>> {
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
    pub use crate::entity::{Entity, EntityCategory, EntityType, ExtractionMethod, Provenance, TypeMapper};
    pub use crate::error::{Error, Result};
    pub use crate::types::{Confidence, EntitySliceExt, Score};
    pub use crate::Model;
    pub use crate::{HybridConfig, HybridNER, MergeStrategy, MockModel, PatternNER, StackedNER};

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
    DiscontinuousSpan, Entity, EntityBuilder, EntityCategory, EntityType, ExtractionMethod,
    HashMapLexicon, HierarchicalConfidence, Lexicon, Provenance, RaggedBatch, Relation, Span,
    SpanCandidate, TypeMapper, generate_filtered_candidates, generate_span_candidates,
};
pub use error::{Error, Result};
pub use offset::{
    OffsetMapping, SpanConverter, TextSpan, TokenSpan, bytes_to_chars, chars_to_bytes, is_ascii,
};

// Backend re-exports (always available)
pub use backends::{
    BackendType, ConflictStrategy, HybridConfig, HybridNER, MergeStrategy,
    NERExtractor, NuNER, PatternNER, StackedNER, StatisticalNER,
    W2NER, W2NERConfig, W2NERRelation,
};

// Inference abstractions (research-aligned traits)
pub use backends::inference::{
    // Core encoder traits
    BiEncoder, DiscontinuousNER, LabelEncoder, RelationExtractor, TextEncoder, ZeroShotNER,
    // Supporting types
    DiscontinuousEntity, EncoderOutput, ExtractionWithRelations, RelationTriple, SpanLabelScore,
    // Late interaction
    DotProductInteraction, LateInteraction, MaxSimInteraction,
    // Span representation
    SpanRepConfig, SpanRepresentationLayer,
    // Semantic registry
    LabelCategory, LabelDefinition, ModalityHint, SemanticRegistry, SemanticRegistryBuilder,
    // Handshaking matrix (W2NER-style)
    HandshakingMatrix,
    // Coreference
    CoreferenceCluster, CoreferenceConfig,
    // Modality
    ImageFormat, ModalityInput, VisualPosition,
};

// Backwards compatibility aliases (deprecated)
#[allow(deprecated)]
pub use backends::{CompositeNER, LayeredNER, RuleBasedNER, TieredNER};

#[cfg(feature = "onnx")]
pub use backends::{BertNEROnnx, GLiNEROnnx};

#[cfg(feature = "candle")]
pub use backends::CandleNER;

/// Default BERT ONNX model (reliable, widely tested).
pub const DEFAULT_BERT_ONNX_MODEL: &str = "protectai/bert-base-NER-onnx";

/// Default GLiNER model (zero-shot NER).
pub const DEFAULT_GLINER_MODEL: &str = "onnx-community/gliner_small-v2.1";

/// Default Candle model (BERT-based NER).
pub const DEFAULT_CANDLE_MODEL: &str = "dslim/bert-base-NER";

/// Default NuNER model (token-based zero-shot).
pub const DEFAULT_NUNER_MODEL: &str = "deepanwa/NuNerZero_onnx";

// =============================================================================
// Automatic Backend Selection
// =============================================================================

/// Use case hints for automatic backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UseCase {
    /// Best quality regardless of speed (prefer ML backends).
    BestQuality,
    /// Fast inference, acceptable quality (prefer patterns + statistical).
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
/// 4. StackedNER (always) - pattern + statistical fallback
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
                 Build with: cargo build --features onnx".to_string()
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
                if let Ok(model) = backends::candle::CandleNER::from_pretrained(DEFAULT_CANDLE_MODEL) {
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
        ("PatternNER", true),
        ("StatisticalNER", true),
        ("StackedNER", true),
        ("HybridNER", true),
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
/// - Using [`HybridNER`] to combine pattern + ML backends
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
    fn extract_entities_streaming(
        &self,
        chunk: &str,
        offset: usize,
    ) -> Result<Vec<Entity>> {
        let entities = self.extract_entities(chunk, None)?;
        Ok(entities.into_iter().map(|mut e| {
            e.start += offset;
            e.end += offset;
            e
        }).collect())
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

/// Marker trait for models that provide calibrated confidence scores.
///
/// Calibrated models produce confidence scores that reflect true
/// probability of correctness (e.g., 80% confidence = 80% accuracy).
pub trait CalibratedConfidence: Model {
    /// Get the expected calibration error for this model.
    ///
    /// Lower is better. 0.0 = perfectly calibrated.
    fn expected_calibration_error(&self) -> f64;
}

/// Marker trait for multi-modal models that can process images.
///
/// Visual models (like ColPali) can extract entities from images
/// in addition to or instead of text.
pub trait VisualCapable: Model {
    /// Extract entities from an image.
    ///
    /// Returns entities with bounding box locations.
    fn extract_from_image(&self, image_data: &[u8], format: &str) -> Result<Vec<Entity>>;
}
