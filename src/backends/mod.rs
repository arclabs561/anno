//! NER backend implementations.
//!
//! Each backend implements the `Model` trait for consistent usage.
//!
//! # Available Backends
//!
//! | Backend | Feature | Speed | Entities |
//! |---------|---------|-------|----------|
//! | `PatternNER` | always | ~400ns | DATE, MONEY, PERCENT, EMAIL, URL, PHONE |
//! | `HybridNER` | always | varies | All (pattern + optional ML) |
//! | `BertNEROnnx` | `onnx` | ~50ms | PER, ORG, LOC, MISC |
//! | `GLiNERNER` | `onnx` | ~100ms | Zero-shot (any type) |
//! | `CandleNER` | `candle` | ~50ms | PER, ORG, LOC, MISC |

pub mod hybrid;
pub mod pattern;
pub mod rule;

#[cfg(feature = "onnx")]
pub mod onnx;

#[cfg(feature = "onnx")]
pub mod gliner;

#[cfg(feature = "candle")]
pub mod candle;

// Re-exports
pub use hybrid::{HybridConfig, HybridNER, MergeStrategy};
pub use pattern::PatternNER;

#[allow(deprecated)]
pub use rule::RuleBasedNER;

#[cfg(feature = "onnx")]
pub use onnx::BertNEROnnx;

#[cfg(feature = "onnx")]
pub use gliner::GLiNERNER;

#[cfg(feature = "candle")]
pub use candle::CandleNER;
