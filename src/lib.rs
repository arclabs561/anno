//! # anno - Fast Named Entity Recognition
//!
//! `anno` provides Named Entity Recognition (NER) with multiple backends:
//!
//! | Backend | Feature | Quality | Notes |
//! |---------|---------|---------|-------|
//! | Pattern | always | N/A | DATE/MONEY/PERCENT only |
//! | Rule-based | always | ~84% F1* | Gazetteers |
//! | BERT ONNX | `onnx` | ~74% F1 | Recommended default |
//! | Candle | `candle` | ~74% F1 | Rust-native |
//!
//! *Rule-based F1 is inflated on curated tests; real-world is lower.
//!
//! ## Quick Start
//!
//! ```rust
//! use anno::{Model, PatternNER};
//!
//! let model = PatternNER::new();
//! let entities = model.extract_entities("Meeting on January 15, 2025", None).unwrap();
//! assert!(entities.iter().any(|e| e.text.contains("January")));
//! ```
//!
//! ## Design Philosophy
//!
//! - **ML-first**: BERT ONNX is the recommended default (reliable, widely tested)
//! - **No hardcoded gazetteers**: Pattern NER only extracts format-based entities
//! - **Trait-based**: All backends implement the `Model` trait
//! - **Graceful degradation**: Falls back to patterns if ML unavailable

#![warn(missing_docs)]

pub mod backends;
mod entity;
mod error;
pub mod eval;

/// Prelude for common imports.
///
/// ```rust
/// use anno::prelude::*;
/// ```
pub mod prelude {
    pub use crate::entity::{Entity, EntityType};
    pub use crate::error::{Error, Result};
    pub use crate::Model;
    pub use crate::PatternNER;

    #[cfg(feature = "onnx")]
    pub use crate::{BertNEROnnx, GLiNERNER};

    #[cfg(feature = "candle")]
    pub use crate::CandleNER;
}

// Re-exports
pub use entity::{Entity, EntityType};
pub use error::{Error, Result};

// Backend re-exports (always available)
pub use backends::PatternNER;

#[allow(deprecated)]
pub use backends::RuleBasedNER;

#[cfg(feature = "onnx")]
pub use backends::{BertNEROnnx, GLiNERNER};

#[cfg(feature = "candle")]
pub use backends::CandleNER;

/// Default BERT ONNX model (reliable, widely tested).
pub const DEFAULT_BERT_ONNX_MODEL: &str = "protectai/bert-base-NER-onnx";

/// Default GLiNER model (zero-shot NER).
pub const DEFAULT_GLINER_MODEL: &str = "onnx-community/gliner_small-v2.1";

/// Default Candle model (BERT-based NER).
pub const DEFAULT_CANDLE_MODEL: &str = "dslim/bert-base-NER";

/// Trait for NER model backends.
///
/// All NER backends implement this trait for consistent usage.
pub trait Model: Send + Sync {
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
