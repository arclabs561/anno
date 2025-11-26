//! NER backend implementations.
//!
//! Each backend implements the `Model` trait for consistent usage.

pub mod pattern;
pub mod rule;

#[cfg(feature = "onnx")]
pub mod onnx;

#[cfg(feature = "onnx")]
pub mod gliner;

#[cfg(feature = "candle")]
pub mod candle;

// Re-exports
pub use pattern::PatternNER;
pub use rule::RuleBasedNER;

#[cfg(feature = "onnx")]
pub use onnx::BertNEROnnx;

#[cfg(feature = "onnx")]
pub use gliner::GLiNERNER;

#[cfg(feature = "candle")]
pub use candle::CandleNER;

