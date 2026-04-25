//! GLiREL: Zero-shot Relation Extraction via GLiNER
//!
//! GLiREL extends GLiNER to predict typed relations between entity pairs.
//! The model uses a shared DeBERTa-v3 encoder for text and relation labels,
//! then scores (head, tail, relation_type) triples via dot-product scoring.
//!
//! # Architecture
//!
//! ```text
//! Input text + entity spans + relation labels
//!                    │
//!                    ▼
//! ┌──────────────────────────────────────┐
//! │     Shared DeBERTa-v3 Encoder        │
//! └──────────────────────────────────────┘
//!         │                 │
//!         ▼                 ▼
//!  Token/word reps     Relation label reps
//!         │
//!         ▼
//!  Span pooling → Entity pair reps
//!         │
//!         ▼
//!  dot(pair_repr, rel_repr) → relation_scores
//! ```
//!
//! # Model Source
//!
//! Export with: `uv run scripts/export_glirel_onnx.py`
//! Compatible models: `jackboyla/glirel-large-v0`
//!
//! # Usage
//!
//! ```rust,ignore
//! use anno::backends::glirel::GLiREL;
//!
//! let model = GLiREL::from_pretrained("jackboyla/glirel-large-v0")?;
//! let relations = model.extract_relations(
//!     "Steve Jobs founded Apple.",
//!     &[(0, 10, "person"), (19, 24, "organization")],
//!     &["founded", "works_for", "ceo_of"],
//!     0.5,
//! )?;
//! ```

#[cfg(feature = "onnx")]
mod onnx;

#[cfg(feature = "onnx")]
pub use onnx::GLiREL;

// Stub when ONNX feature is not enabled.
#[cfg(not(feature = "onnx"))]
#[derive(Debug)]
/// GLiREL stub (requires `onnx` feature).
pub struct GLiREL {
    _private: (),
}

#[cfg(not(feature = "onnx"))]
impl GLiREL {
    /// Load model (requires `onnx` feature).
    pub fn from_pretrained(_model_id: &str) -> crate::Result<Self> {
        Err(crate::Error::FeatureNotAvailable(
            "GLiREL requires the 'onnx' feature. Build with: cargo build --features onnx"
                .to_string(),
        ))
    }

    /// Load from local directory (requires `onnx` feature).
    pub fn from_local(_dir: &std::path::Path) -> crate::Result<Self> {
        Err(crate::Error::FeatureNotAvailable(
            "GLiREL requires the 'onnx' feature. Build with: cargo build --features onnx"
                .to_string(),
        ))
    }
}
