//! # anno
//!
//! Information extraction for Rust: NER, coreference resolution, and evaluation.
//!
//! - **NER**: Multiple backends (Regex, BERT, GLiNER, NuNER, W2NER)
//! - **Coreference**: Resolution (rule-based, T5-based) and metrics (MUC, B³, CEAF, LEA, BLANC)
//! - **Evaluation**: Comprehensive benchmarking framework with bias analysis
//!
//! Core types (Entity, GroundedDocument, etc.) are in `anno-core` and re-exported here.

#![warn(missing_docs)]

// Module declarations (core types are in anno-core, not declared here)
pub mod backends;
pub mod error;
pub mod eval;
pub mod ingest;
pub mod lang;
pub mod offset;
pub mod schema;
pub mod similarity;
pub mod sync;
pub mod types;

#[cfg(feature = "cli")]
pub mod cli;

#[cfg(feature = "discourse")]
pub mod discourse;

// Re-export error types
pub use error::{Error, Result};

// Re-export anno-core types for backward compatibility
pub use anno_core::{
    Corpus, DiscontinuousSpan, Entity, EntityBuilder, EntityCategory, EntityType, EntityViewport,
    ExtractionMethod, GraphDocument, GraphEdge, GraphExportFormat, GraphNode, GroundedDocument,
    HashMapLexicon, HierarchicalConfidence, Identity, IdentityId, IdentitySource, Lexicon,
    Location, Modality, Provenance, Quantifier, RaggedBatch, Relation, Signal, SignalId, SignalRef,
    Span, SpanCandidate, Track, TrackId, TrackRef, TypeMapper, ValidationIssue,
};

/// Re-export graph module for backward compatibility (anno::graph::*)
///
/// This module re-exports all graph-related types from `anno-core`:
/// - `GraphNode`, `GraphEdge`, `GraphDocument`
/// - Graph export formats and utilities
pub mod graph {
    pub use anno_core::graph::*;
}

/// Re-export grounded module for backward compatibility (anno::grounded::*)
///
/// This module re-exports all grounded document types from `anno-core`:
/// - `GroundedDocument`, `Signal`, `Track`, `Identity`
/// - Coreference resolution types and utilities
pub mod grounded {
    pub use anno_core::grounded::*;
}

// Re-export commonly used types
pub use lang::{detect_language, Language};
pub use offset::{
    bytes_to_chars, chars_to_bytes, is_ascii, OffsetMapping, SpanConverter, TextSpan, TokenSpan,
};
pub use schema::*;
pub use similarity::*;
pub use sync::*;
pub use types::*;

// =============================================================================
// Sealed Trait Pattern
// =============================================================================

mod sealed {
    pub trait Sealed {}

    impl Sealed for super::RegexNER {}
    impl Sealed for super::HeuristicNER {}
    impl Sealed for super::StackedNER {}
    impl Sealed for super::NuNER {}
    impl Sealed for super::W2NER {}
    impl Sealed for super::NERExtractor {}

    #[cfg(feature = "onnx")]
    impl Sealed for super::BertNEROnnx {}

    #[cfg(feature = "onnx")]
    impl Sealed for super::GLiNEROnnx {}

    #[cfg(feature = "onnx")]
    impl Sealed for super::backends::albert::ALBERTNER {}

    #[cfg(feature = "onnx")]
    impl Sealed for super::backends::deberta_v3::DeBERTaV3NER {}

    #[cfg(feature = "onnx")]
    impl Sealed for super::backends::gliner_poly::GLiNERPoly {}

    #[cfg(feature = "onnx")]
    impl Sealed for super::backends::gliner2::GLiNER2Onnx {}

    #[cfg(feature = "candle")]
    impl Sealed for super::CandleNER {}

    impl Sealed for super::backends::tplinker::TPLinker {}
    impl Sealed for super::backends::universal_ner::UniversalNER {}

    #[allow(deprecated)]
    impl Sealed for super::backends::rule::RuleBasedNER {}

    impl Sealed for super::MockModel {}
}

/// Trait for NER model backends.
pub trait Model: sealed::Sealed + Send + Sync {
    /// Extract entities from text.
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

/// Trait for models that support batch processing.
///
/// Models implementing this trait can process multiple texts efficiently,
/// potentially using parallel processing or optimized batch operations.
pub trait BatchCapable: Model {
    /// Extract entities from multiple texts in a batch.
    ///
    /// # Arguments
    /// * `texts` - Slice of text strings to process
    /// * `language` - Optional language hint for the texts
    ///
    /// # Returns
    /// A vector of entity vectors, one per input text
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

    /// Get the optimal batch size for this model, if applicable.
    ///
    /// Returns `None` if the model doesn't have a specific optimal batch size,
    /// or `Some(n)` if there's a recommended batch size for best performance.
    fn optimal_batch_size(&self) -> Option<usize> {
        None
    }
}

pub trait GpuCapable: Model {
    fn is_gpu_active(&self) -> bool;
    fn device(&self) -> &str;
}

pub trait StreamingCapable: Model {
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

    fn recommended_chunk_size(&self) -> usize {
        10_000
    }
}

pub trait NamedEntityCapable: Model {}
pub trait StructuredEntityCapable: Model {}

pub trait RelationCapable: Model {
    fn extract_with_relations(
        &self,
        text: &str,
        language: Option<&str>,
    ) -> Result<(Vec<Entity>, Vec<Relation>)>;
}

pub trait DynamicLabels: Model {
    fn extract_with_labels(
        &self,
        text: &str,
        labels: &[&str],
        language: Option<&str>,
    ) -> Result<Vec<Entity>>;
}

// Re-export backends
pub use backends::{
    AutoNER, BackendType, ConflictStrategy, HeuristicNER, NERExtractor, NuNER, RegexNER,
    StackedNER, TPLinker, W2NERConfig, W2NERRelation, W2NER,
};

// Re-export MockModel for testing

// Re-export Model trait and related
pub use backends::inference::*;

#[cfg(feature = "onnx")]
pub use backends::{BertNEROnnx, GLiNEROnnx};

#[cfg(feature = "candle")]
pub use backends::CandleNER;

// Constants
pub const DEFAULT_BERT_ONNX_MODEL: &str = "protectai/bert-base-NER-onnx";
pub const DEFAULT_GLINER_MODEL: &str = "onnx-community/gliner_small-v2.1";
pub const DEFAULT_GLINER2_MODEL: &str = "onnx-community/gliner-multitask-large-v0.5";
pub const DEFAULT_CANDLE_MODEL: &str = "dslim/bert-base-NER";
pub const DEFAULT_NUNER_MODEL: &str = "deepanwa/NuNerZero_onnx";
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
pub fn available_backends() -> Vec<(&'static str, bool)> {
    let backends = vec![
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

    #[cfg(feature = "candle")]
    {
        backends.push(("CandleNER", true));
    }

    backends
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
        // Performance optimization: Cache text length (called once, used for all entities)
        let text_len = text.chars().count();
        for (i, e) in self.entities.iter().enumerate() {
            if e.end > text_len {
                return Err(Error::InvalidInput(format!(
                    "MockModel entity {} '{}': end offset ({}) exceeds text length ({} chars)",
                    i, e.text, e.end, text_len
                )));
            }
            // Verify text matches (using char offsets)
            // Use optimized extract_text_with_len to avoid recalculating length
            let actual_text = e.extract_text_with_len(text, text_len);
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
