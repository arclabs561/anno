//! Re-exports of the ONNX backend's decoder + supporting types.
//!
//! The Candle pipeline produces a [`crate::backends::gliner2_fastino::pipeline::ScorerOutput`]
//! (an `Array4<f32>` of shape `[MAX_COUNT, num_words, MAX_WIDTH, num_labels]`)
//! that's identical to the ONNX backend's output. From there, decoding
//! to `Vec<Entity>` / `Vec<ExtractedStructure>` reuses the same logic.

// pub(crate) re-exports — the decoder family is `pub(crate)` in the
// ONNX backend and can't be `pub use`-d to `pub` here.
pub(crate) use crate::backends::gliner2_fastino::pipeline::{
    decode_entities, decode_entities_with_thresholds, decode_structure, ScorerOutput, MAX_COUNT,
    MAX_WIDTH,
};
