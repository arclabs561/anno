//! Pure Rust encoder implementations using Candle.
//!
//! # Design Philosophy
//!
//! Pluggable encoder backends sharing a common trait:
//!
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │           TextEncoder Trait                 │
//! │  fn encode(&self, text) -> Embeddings      │
//! │  fn hidden_dim(&self) -> usize              │
//! └──────────────────┬──────────────────────────┘
//!                    │
//!        ┌───────────┴───────────┐
//!        │                       │
//! ┌──────▼──────┐         ┌──────▼──────┐
//! │ BertEncoder │         │ModernBertEnc│
//! │  512 ctx    │         │  8192 ctx   │
//! │  APE        │         │  RoPE       │
//! └─────────────┘         └─────────────┘
//! ```
//!
//! # Key Innovation: ModernBERT
//!
//! ModernBERT (late 2024) combines:
//! - 8192 token context (vs 512 for BERT)
//! - RoPE (Rotary Position Embeddings) for extrapolation
//! - GeGLU activation functions
//! - Unpadding for memory efficiency
//!
//! Reference: <https://arxiv.org/abs/2412.13663>

#![allow(unused_variables)]

use crate::{Error, Result};

#[cfg(feature = "candle")]
use {
    candle_core::{DType, Device, IndexOp, Module, Tensor, D},
    candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, VarBuilder},
};

#[cfg(feature = "candle")]
use tokenizers::Tokenizer;

/// Encoder configuration types and defaults.
pub mod config;
#[allow(unused_imports)]
pub use config::*;
/// Encoder backend implementations (Candle, stubs).
pub mod implementations;
#[cfg(feature = "candle")]
pub use implementations::candle_impl::{best_device, CandleEncoder};

// =============================================================================
// Stub for non-candle builds
// =============================================================================
#[cfg(not(feature = "candle"))]
pub struct CandleEncoder;

#[cfg(not(feature = "candle"))]
impl CandleEncoder {
    pub fn new_random(_config: EncoderConfig, _name: &str) -> Result<Self> {
        Err(Error::FeatureNotAvailable(
            "CandleEncoder requires 'candle' feature".into(),
        ))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
