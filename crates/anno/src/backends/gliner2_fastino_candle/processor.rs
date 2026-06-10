//! Re-export of the ONNX backend's processor (input prep). Identical
//! tokenization + prompt assembly logic — no Candle-specific changes.

pub use crate::backends::gliner2_fastino::processor::*;
