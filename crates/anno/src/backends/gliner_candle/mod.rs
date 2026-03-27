//! GLiNER implementation using Candle (pure Rust ML) with Metal/CUDA support.
//!
//! Zero-shot NER using bi-encoder architecture: match text spans to entity labels.
//!
//! # Architecture
//!
//! ```text
//! Text Input     Label Input
//!     |              |
//!     v              v
//! [Tokenizer]   [Tokenizer]
//!     |              |
//!     v              v
//! [Transformer Encoder] (shared)
//!     |              |
//!     v              v
//! [SpanRepLayer]  [LabelEncoder]
//!     |              |
//!     +------+-------+
//!            |
//!            v
//!     [SpanLabelMatcher]
//!            |
//!            v
//!       [Entities]
//! ```
//!
//! # GPU Support
//!
//! - **Metal** (Apple Silicon): `cargo build --features candle,metal`
//! - **CUDA** (NVIDIA): `cargo build --features candle,cuda`
//! - **CPU**: Always available as fallback
//!
//! # Example
//!
//! ```rust,ignore
//! use anno::backends::gliner_candle::GLiNERCandle;
//!
//! let model = GLiNERCandle::from_pretrained("urchade/gliner_small-v2.1")?;
//! let entities = model.extract(
//!     "Steve Jobs founded Apple in California.",
//!     &["person", "organization", "location"],
//!     0.5,
//! )?;
//! ```

#![allow(dead_code)] // Token constants for future prompt encoding

use crate::{Entity, EntityType, Error, Language, Result};
use std::path::{Path, PathBuf};

#[cfg(feature = "candle")]
use {
    super::encoder_candle::{CandleEncoder, CandleTextEncoder},
    candle_core::{DType, Device, IndexOp, Module, Tensor, D},
    candle_nn::{linear, Linear, VarBuilder},
    tokenizers::Tokenizer,
};

/// Maximum span width for entity candidates.
const MAX_SPAN_WIDTH: usize = 12;

/// Special tokens for GLiNER models.
#[cfg(feature = "candle")]
const TOKEN_START: u32 = 1;
#[cfg(feature = "candle")]
const TOKEN_END: u32 = 2;
#[cfg(feature = "candle")]
const TOKEN_ENT: u32 = 128002;
#[cfg(feature = "candle")]
const TOKEN_SEP: u32 = 128003;

// =============================================================================
// Device Selection
// =============================================================================

/// Get the best available compute device.
#[cfg(feature = "candle")]
pub fn best_device() -> Result<Device> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        if let Ok(device) = Device::new_metal(0) {
            log::info!("[GLiNER-Candle] Using Metal GPU");
            return Ok(device);
        }
    }

    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = Device::new_cuda(0) {
            log::info!("[GLiNER-Candle] Using CUDA GPU");
            return Ok(device);
        }
    }

    log::info!("[GLiNER-Candle] Using CPU");
    Ok(Device::Cpu)
}

// =============================================================================
// Span Representation Layer (SpanMarker style)
// =============================================================================

/// Span representation using the SpanMarker approach from GLiNER.
/// Projects start and end positions separately and combines them.
#[cfg(feature = "candle")]
pub mod layers;
pub use layers::*;

// =============================================================================
// GLiNER Candle Model
// =============================================================================

/// GLiNER zero-shot NER using pure Rust Candle backend.
///
/// Matches text spans to entity type descriptions using a bi-encoder.
/// Supports Metal (Apple Silicon) and CUDA (NVIDIA) GPU acceleration.
mod inference;
#[cfg(feature = "candle")]
pub(crate) use inference::convert_pytorch_to_safetensors;
#[cfg(feature = "candle")]
pub use inference::GLiNERCandle;

const DEFAULT_GLINER_LABELS: &[&str] = &[
    "person",
    "organization",
    "location",
    "date",
    "time",
    "money",
    "percent",
    "product",
    "event",
    "facility",
    "work_of_art",
    "law",
    "language",
];

#[cfg(feature = "candle")]
impl crate::Model for GLiNERCandle {
    fn extract_entities(&self, text: &str, _language: Option<Language>) -> Result<Vec<Entity>> {
        // Use lower threshold for smaller models (NeuML/gliner-bert-tiny)
        // The threshold may need tuning based on the specific model
        self.extract(text, DEFAULT_GLINER_LABELS, 0.3)
    }

    fn supported_types(&self) -> Vec<EntityType> {
        DEFAULT_GLINER_LABELS
            .iter()
            .map(|label| Self::map_label(label))
            .collect()
    }

    fn is_available(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "GLiNER-Candle"
    }

    fn description(&self) -> &'static str {
        "Zero-shot NER using GLiNER bi-encoder (pure Rust with Metal/CUDA support)"
    }

    fn capabilities(&self) -> crate::ModelCapabilities {
        crate::ModelCapabilities {
            zero_shot: true,
            ..Default::default()
        }
    }
}

#[cfg(feature = "candle")]
impl crate::backends::inference::ZeroShotNER for GLiNERCandle {
    fn extract_with_types(
        &self,
        text: &str,
        entity_types: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        self.extract(text, entity_types, threshold)
    }

    fn extract_with_descriptions(
        &self,
        text: &str,
        descriptions: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        // GLiNER can use descriptions directly as label text
        self.extract(text, descriptions, threshold)
    }

    fn default_types(&self) -> &[&'static str] {
        &["person", "organization", "location", "date", "event"]
    }
}

// =============================================================================
// Non-candle stub (struct + Model + ZeroShotNER)
// =============================================================================

crate::backends::macros::define_feature_stub! {
    struct GLiNERCandle;
    feature = "candle";
    name = "GLiNER-Candle (unavailable)";
    description = "Zero-shot NER with Candle - requires 'candle' feature";
    error_msg = "GLiNER-Candle requires the 'candle' feature";
    methods {
        /// Load from pretrained (requires candle feature).
        pub fn from_pretrained(_model_id: &str) -> crate::Result<Self> {
            Self::new("")
        }
    }
    impls {
        ZeroShotNER,
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
