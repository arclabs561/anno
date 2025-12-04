//! NER backend implementations.
//!
//! Each backend implements the `Model` trait for consistent usage.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │ Layer 3: ML Backends (feature-gated)                │
//! │                                                     │
//! │  Zero-Shot NER (any entity type):                   │
//! │   - GLiNER: Bi-encoder span classification          │
//! │   - NuNER: Token classification (arbitrary length)  │
//! │                                                     │
//! │  Complex Structures (nested/discontinuous):         │
//! │   - W2NER: Word-word relation grids                 │
//! │                                                     │
//! │  Traditional (fixed types):                         │
//! │   - BertNEROnnx: Sequence labeling                  │
//! │                                                     │
//! │  ~85-92% F1, requires features                      │
//! ├─────────────────────────────────────────────────────┤
//! │ Layer 2: HeuristicNER (zero deps)                   │
//! │   Person/Org/Location via heuristics                │
//! │   ~60-70% F1, always available                      │
//! ├─────────────────────────────────────────────────────┤
//! │ Layer 1: RegexNER (zero deps)                     │
//! │   Date/Time/Money/Email/URL/Phone                   │
//! │   ~95%+ precision, always available                 │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! # Backend Comparison
//!
//! | Backend | Feature | Zero-Shot | Nested | Speed | Notes |
//! |---------|---------|-----------|--------|-------|-------|
//! | `StackedNER` | - | No | No | Fast | **Composable with any backend** |
//! | `RegexNER` | - | No | No | ~400ns | Structured only |
//! | `HeuristicNER` | - | No | No | ~50μs | Capitalization + context |
//! | `GLiNER` | `onnx` | Yes | No | ~100ms | Span-based |
//! | `NuNER` | `onnx` | Yes | No | ~100ms | Token-based |
//! | `W2NER` | `onnx` | No | **Yes** | ~150ms | Grid-based |
//! | `BertNEROnnx` | `onnx` | No | No | ~50ms | Traditional |
//!
//! # When to Use What
//!
//! - **Best accuracy**: `NERExtractor::best_available()` - uses GLiNER (~90% F1)
//! - **Zero deps**: `StackedNER::default()` - no ML, good baseline
//! - **Hybrid approach**: `StackedNER` with ML backends - combine ML accuracy with pattern speed
//! - **Custom types**: `GLiNER` or `NuNER` - zero-shot, any entity type
//! - **Nested entities**: `W2NER` - handles overlapping spans
//! - **Structured data**: `RegexNER` - dates, emails, money
//!
//! # Quick Start
//!
//! Zero-dependency default (Pattern + Heuristic):
//!
//! ```rust
//! use anno::{Model, StackedNER};
//!
//! let ner = StackedNER::default();
//! let entities = ner.extract_entities("Dr. Smith charges $100/hr", None).unwrap();
//! ```
//!
//! Custom stack with pattern + heuristic:
//!
//! ```rust
//! use anno::{Model, RegexNER, HeuristicNER, StackedNER};
//! use anno::backends::stacked::ConflictStrategy;
//!
//! let ner = StackedNER::builder()
//!     .layer(RegexNER::new())
//!     .layer(HeuristicNER::new())
//!     .strategy(ConflictStrategy::LongestSpan)
//!     .build();
//! ```
//!
//! **StackedNER is fully composable** - you can combine ML backends with pattern/heuristic layers:
//!
//! ```rust,no_run
//! #[cfg(feature = "onnx")]
//! {
//! use anno::{Model, StackedNER, GLiNEROnnx, RegexNER, HeuristicNER};
//! use anno::backends::stacked::ConflictStrategy;
//!
//! // ML-first: ML runs first, then patterns fill gaps
//! let ner = StackedNER::with_ml_first(
//!     Box::new(GLiNEROnnx::new("onnx-community/gliner_small-v2.1").unwrap())
//! );
//!
//! // ML-fallback: patterns/heuristics first, ML as fallback
//! let ner = StackedNER::with_ml_fallback(
//!     Box::new(GLiNEROnnx::new("onnx-community/gliner_small-v2.1").unwrap())
//! );
//!
//! // Custom stack: any combination of backends
//! let ner = StackedNER::builder()
//!     .layer(RegexNER::new())           // High-precision structured entities
//!     .layer_boxed(Box::new(GLiNEROnnx::new("onnx-community/gliner_small-v2.1").unwrap()))  // ML layer
//!     .layer(HeuristicNER::new())       // Quick named entities
//!     .strategy(ConflictStrategy::HighestConf)  // Resolve conflicts by confidence
//!     .build();
//! }
//! ```

// Always available (zero deps beyond std)
/// Box embeddings for geometric coreference resolution.
pub mod box_embeddings;
/// Training system for box embeddings.
///
/// This is the canonical training implementation. The [matryoshka-box](https://github.com/arclabs561/matryoshka-box)
/// research project extends this with matryoshka-specific features (variable dimensions, etc.).
pub mod box_embeddings_training;
pub mod catalog;
pub mod encoder;
pub mod extractor;
pub mod heuristic;
pub mod inference;
pub mod nuner;
pub mod pattern_config;
pub mod regex;
/// Language-aware routing for automatic backend selection.
pub mod router;
pub mod rule;
pub mod stacked;
pub mod tplinker;
pub mod w2ner;

// Advanced backends
#[cfg(feature = "onnx")]
pub mod albert;
#[cfg(feature = "onnx")]
pub mod deberta_v3;
#[cfg(feature = "onnx")]
pub mod gliner_poly;
pub mod universal_ner;

// LLM-based NER prompting (CodeNER-style)
pub mod llm_prompt;

// Demonstration selection for few-shot NER (CMAS-inspired)
pub mod demonstration;

// GLiNER via ONNX (uses same feature as other ONNX models)
// Note: gline-rs crate not yet published to crates.io

// ONNX implementations
#[cfg(feature = "onnx")]
pub mod gliner_onnx;

#[cfg(feature = "onnx")]
pub mod onnx;

// Pure Rust via Candle
#[cfg(feature = "candle")]
pub mod candle;

#[cfg(feature = "candle")]
pub mod encoder_candle;

#[cfg(feature = "candle")]
pub mod gliner_candle;

#[cfg(feature = "candle")]
pub mod gliner_pipeline;

// GLiNER2 multi-task extraction (ONNX or Candle)
#[cfg(any(feature = "onnx", feature = "candle"))]
pub mod gliner2;

// Production infrastructure
#[cfg(feature = "async-inference")]
pub mod async_adapter;

#[cfg(feature = "session-pool")]
pub mod session_pool;

// Model warmup for cold-start mitigation
pub mod warmup;

// T5-based coreference resolution
#[cfg(feature = "onnx")]
pub mod coref_t5;

// Re-exports (always available)
pub use extractor::{BackendType, NERExtractor};
pub use heuristic::HeuristicNER;
pub use nuner::NuNER;
pub use regex::RegexNER;
pub use router::AutoNER;
pub use stacked::{ConflictStrategy, StackedNER};

/// Backwards compatibility alias for `HeuristicNER`.
///
/// The name "StatisticalNER" was misleading since it doesn't use
/// statistical/probabilistic methods like CRF or HMM - it uses
/// capitalization and context heuristics.
#[deprecated(
    since = "0.3.0",
    note = "Use HeuristicNER instead - StatisticalNER was misleading"
)]
pub type StatisticalNER = HeuristicNER;
pub use tplinker::TPLinker;
pub use w2ner::{W2NERConfig, W2NERRelation, W2NER};

// Advanced backends
#[cfg(feature = "onnx")]
pub use albert::ALBERTNER;
#[cfg(feature = "onnx")]
pub use deberta_v3::DeBERTaV3NER;
#[cfg(feature = "onnx")]
pub use gliner_poly::GLiNERPoly;
pub use universal_ner::UniversalNER;

// Backwards compatibility
#[allow(deprecated)]
pub use stacked::{CompositeNER, LayeredNER, TieredNER};

#[allow(deprecated)]
pub use rule::RuleBasedNER;

// Re-exports (feature-gated)
#[cfg(feature = "onnx")]
pub use gliner_onnx::GLiNEROnnx;

#[cfg(feature = "onnx")]
pub use onnx::BertNEROnnx;

#[cfg(feature = "candle")]
pub use candle::CandleNER;

#[cfg(feature = "candle")]
pub use encoder_candle::{EncoderArchitecture, EncoderConfig};

#[cfg(feature = "candle")]
pub use gliner_candle::GLiNERCandle;

// GLiNER2 multi-task model
#[cfg(any(feature = "onnx", feature = "candle"))]
pub use gliner2::{
    ClassificationResult, ClassificationTask, EntityTask, ExtractedStructure, ExtractionResult,
    FieldType, GLiNER2, StructureTask, StructureValue, TaskSchema,
};

#[cfg(feature = "onnx")]
pub use gliner2::GLiNER2Onnx;

#[cfg(feature = "candle")]
pub use gliner2::GLiNER2Candle;

// Production infrastructure re-exports
#[cfg(feature = "async-inference")]
pub use async_adapter::{batch_extract, batch_extract_limited, AsyncNER, IntoAsync};

#[cfg(feature = "session-pool")]
pub use session_pool::{GLiNERPool, PoolConfig, SessionPool};

// T5 coreference
#[cfg(feature = "onnx")]
pub use coref_t5::{CorefCluster, T5Coref, T5CorefConfig};

// Config re-exports (for quantization control)
#[cfg(feature = "onnx")]
pub use gliner_onnx::GLiNERConfig;

#[cfg(feature = "onnx")]
pub use onnx::BertNERConfig;

// Warmup utilities (always available)
pub use warmup::{warmup_model, warmup_with_callback, WarmupConfig, WarmupResult};

// Box embeddings for geometric coreference
pub use box_embeddings::{
    acquisition_roles, interaction_strength, BoxCorefConfig, BoxEmbedding, BoxVelocity, Conflict,
    GumbelBox, TemporalBox, UncertainBox,
};

// Box embedding training (canonical implementation; matryoshka-box extends with research features)
pub use box_embeddings_training::{
    coref_documents_to_training_examples, split_train_val, BoxEmbeddingTrainer, TrainingConfig,
    TrainingExample,
};
