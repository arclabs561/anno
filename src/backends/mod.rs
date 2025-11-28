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
//! │ Layer 2: StatisticalNER (zero deps)                 │
//! │   Person/Org/Location via heuristics                │
//! │   ~60-70% F1, always available                      │
//! ├─────────────────────────────────────────────────────┤
//! │ Layer 1: PatternNER (zero deps)                     │
//! │   Date/Time/Money/Email/URL/Phone                   │
//! │   ~95%+ precision, always available                 │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! # Backend Comparison
//!
//! | Backend | Feature | Zero-Shot | Nested | Speed | Notes |
//! |---------|---------|-----------|--------|-------|-------|
//! | `StackedNER` | - | No | No | Fast | Composable layers |
//! | `PatternNER` | - | No | No | ~400ns | Structured only |
//! | `StatisticalNER` | - | No | No | ~50μs | Heuristics |
//! | `GLiNER` | `onnx` | Yes | No | ~100ms | Span-based |
//! | `NuNER` | `onnx` | Yes | No | ~100ms | Token-based |
//! | `W2NER` | `onnx` | No | **Yes** | ~150ms | Grid-based |
//! | `BertNEROnnx` | `onnx` | No | No | ~50ms | Traditional |
//!
//! # When to Use What
//!
//! - **Simple NER**: `StackedNER::default()` - zero deps, good baseline
//! - **Custom types**: `GLiNER` or `NuNER` - zero-shot, any entity type
//! - **Nested entities**: `W2NER` - handles overlapping spans
//! - **Structured data**: `PatternNER` - dates, emails, money
//!
//! # Quick Start
//!
//! ```rust
//! use anno::{Model, StackedNER};
//!
//! let ner = StackedNER::default();
//! let entities = ner.extract_entities("Dr. Smith charges $100/hr", None).unwrap();
//! ```
//!
//! Custom stack:
//!
//! ```rust
//! use anno::{Model, PatternNER, StatisticalNER, StackedNER};
//! use anno::backends::stacked::ConflictStrategy;
//!
//! let ner = StackedNER::builder()
//!     .layer(PatternNER::new())
//!     .layer(StatisticalNER::new())
//!     .strategy(ConflictStrategy::LongestSpan)
//!     .build();
//! ```

// Always available (zero deps beyond std)
pub mod catalog;
pub mod encoder;
pub mod extractor;
pub mod hybrid;
pub mod inference;
pub mod nuner;
pub mod pattern;
pub mod pattern_config;
pub mod rule;
pub mod stacked;
pub mod statistical;
pub mod w2ner;

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

// Re-exports (always available)
pub use extractor::{BackendType, NERExtractor};
pub use hybrid::{HybridConfig, HybridNER, MergeStrategy};
pub use nuner::NuNER;
pub use pattern::PatternNER;
pub use stacked::{ConflictStrategy, StackedNER};
pub use statistical::StatisticalNER;
pub use w2ner::{W2NER, W2NERConfig, W2NERRelation};

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
