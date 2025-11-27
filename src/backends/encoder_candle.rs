//! Pure Rust encoder implementations using Candle.
//!
//! # Design Philosophy
//!
//! This module provides pluggable encoder backends that share a common trait:
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

#![allow(dead_code)]
#![allow(unused_variables)]

use crate::{Error, Result};

#[cfg(feature = "candle")]
use {
    candle_core::{DType, Device, IndexOp, Module, Tensor, D},
    candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, VarBuilder},
    std::collections::HashMap,
};

#[cfg(feature = "candle")]
use tokenizers::Tokenizer;

// =============================================================================
// Core Trait
// =============================================================================

/// Trait for text-to-embedding encoders.
///
/// This is the main abstraction that allows swapping BERT/RoBERTa/ModernBERT.
pub trait TextEncoder: Send + Sync {
    /// Encode text into token embeddings.
    ///
    /// # Returns
    /// - Token embeddings: `[seq_len, hidden_dim]` (flattened)
    /// - Sequence length
    fn encode(&self, text: &str) -> Result<(Vec<f32>, usize)>;

    /// Encode multiple texts into a ragged batch.
    ///
    /// # Returns
    /// - Concatenated embeddings: `[total_tokens, hidden_dim]`
    /// - Cumulative sequence lengths (for unpadding)
    fn encode_batch(&self, texts: &[&str]) -> Result<(Vec<f32>, Vec<usize>)> {
        let mut all_embeddings = Vec::new();
        let mut cu_seqlens = vec![0usize];
        let mut total = 0usize;

        for text in texts {
            let (embeddings, seq_len) = self.encode(text)?;
            all_embeddings.extend(embeddings);
            total += seq_len;
            cu_seqlens.push(total);
        }

        Ok((all_embeddings, cu_seqlens))
    }

    /// Hidden dimension of embeddings.
    fn hidden_dim(&self) -> usize;

    /// Maximum context length.
    fn max_length(&self) -> usize;

    /// Encoder architecture name.
    fn architecture(&self) -> &str;
}

// =============================================================================
// Encoder Configuration
// =============================================================================

/// Configuration for transformer encoder.
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of layers
    pub num_hidden_layers: usize,
    /// Intermediate (FFN) dimension
    pub intermediate_size: usize,
    /// Maximum sequence length
    pub max_position_embeddings: usize,
    /// Dropout probability
    pub hidden_dropout_prob: f32,
    /// Layer norm epsilon
    pub layer_norm_eps: f64,
    /// Whether to use RoPE
    pub use_rope: bool,
    /// Whether to use GeGLU activation
    pub use_geglu: bool,
    /// RoPE theta (for position encoding)
    pub rope_theta: f64,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self::bert_base()
    }
}

impl EncoderConfig {
    /// BERT-base configuration (110M params)
    pub fn bert_base() -> Self {
        Self {
            vocab_size: 30522,
            hidden_size: 768,
            num_attention_heads: 12,
            num_hidden_layers: 12,
            intermediate_size: 3072,
            max_position_embeddings: 512,
            hidden_dropout_prob: 0.1,
            layer_norm_eps: 1e-12,
            use_rope: false,
            use_geglu: false,
            rope_theta: 10000.0,
        }
    }

    /// ModernBERT-base configuration (149M params)
    pub fn modernbert_base() -> Self {
        Self {
            vocab_size: 50368,
            hidden_size: 768,
            num_attention_heads: 12,
            num_hidden_layers: 22,
            intermediate_size: 1152, // Narrower with GeGLU
            max_position_embeddings: 8192,
            hidden_dropout_prob: 0.0, // No dropout during inference
            layer_norm_eps: 1e-5,
            use_rope: true,
            use_geglu: true,
            rope_theta: 160000.0, // Higher for long context
        }
    }

    /// ModernBERT-large configuration (395M params)
    pub fn modernbert_large() -> Self {
        Self {
            vocab_size: 50368,
            hidden_size: 1024,
            num_attention_heads: 16,
            num_hidden_layers: 28,
            intermediate_size: 2624,
            max_position_embeddings: 8192,
            hidden_dropout_prob: 0.0,
            layer_norm_eps: 1e-5,
            use_rope: true,
            use_geglu: true,
            rope_theta: 160000.0,
        }
    }

    /// DeBERTa-v3-base configuration
    pub fn deberta_v3_base() -> Self {
        Self {
            vocab_size: 128100,
            hidden_size: 768,
            num_attention_heads: 12,
            num_hidden_layers: 12,
            intermediate_size: 3072,
            max_position_embeddings: 512,
            hidden_dropout_prob: 0.1,
            layer_norm_eps: 1e-7,
            use_rope: false,
            use_geglu: false,
            rope_theta: 10000.0,
        }
    }

    /// DeBERTa-v3-large configuration
    pub fn deberta_v3_large() -> Self {
        Self {
            vocab_size: 128100,
            hidden_size: 1024,
            num_attention_heads: 16,
            num_hidden_layers: 24,
            intermediate_size: 4096,
            max_position_embeddings: 512,
            hidden_dropout_prob: 0.1,
            layer_norm_eps: 1e-7,
            use_rope: false,
            use_geglu: false,
            rope_theta: 10000.0,
        }
    }

    /// Get config from model name
    pub fn from_model_name(name: &str) -> Self {
        let lower = name.to_lowercase();
        if lower.contains("modernbert") {
            if lower.contains("large") {
                Self::modernbert_large()
            } else {
                Self::modernbert_base()
            }
        } else if lower.contains("deberta") {
            if lower.contains("large") {
                Self::deberta_v3_large()
            } else {
                Self::deberta_v3_base()
            }
        } else {
            Self::bert_base()
        }
    }
}

// =============================================================================
// Encoder Type Selection
// =============================================================================

/// Available encoder architectures for GLiNER.
///
/// Each architecture has different tradeoffs:
/// - **BERT**: Fast, proven, 512 context
/// - **DeBERTaV3**: Better accuracy, disentangled attention
/// - **ModernBERT**: Best accuracy, 8K context, RoPE, GeGLU
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EncoderArchitecture {
    /// Classic BERT encoder (512 context, absolute position)
    Bert,
    /// DeBERTa-v3 encoder (512 context, disentangled attention)
    DeBertaV3,
    /// ModernBERT encoder (8192 context, RoPE, GeGLU)
    #[default]
    ModernBert,
}

impl EncoderArchitecture {
    /// Get default configuration for this architecture.
    pub fn default_config(&self) -> EncoderConfig {
        match self {
            Self::Bert => EncoderConfig::bert_base(),
            Self::DeBertaV3 => EncoderConfig::deberta_v3_base(),
            Self::ModernBert => EncoderConfig::modernbert_base(),
        }
    }

    /// Get HuggingFace model ID for this architecture.
    pub fn default_model_id(&self) -> &'static str {
        match self {
            Self::Bert => "google-bert/bert-base-uncased",
            Self::DeBertaV3 => "microsoft/deberta-v3-base",
            Self::ModernBert => "answerdotai/ModernBERT-base",
        }
    }

    /// Get max context length for this architecture.
    pub fn max_length(&self) -> usize {
        match self {
            Self::Bert | Self::DeBertaV3 => 512,
            Self::ModernBert => 8192,
        }
    }

    /// Whether this architecture uses RoPE.
    pub fn uses_rope(&self) -> bool {
        matches!(self, Self::ModernBert)
    }

    /// Architecture name for display.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Bert => "BERT",
            Self::DeBertaV3 => "DeBERTa-v3",
            Self::ModernBert => "ModernBERT",
        }
    }
}

impl std::fmt::Display for EncoderArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// =============================================================================
// Candle Implementations
// =============================================================================

#[cfg(feature = "candle")]
mod candle_impl {
    use super::*;

    /// Get the best available device.
    pub fn best_device() -> Result<Device> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Ok(device) = Device::new_metal(0) {
                log::info!("[Encoder] Using Metal GPU");
                return Ok(device);
            }
        }

        #[cfg(feature = "cuda")]
        {
            if let Ok(device) = Device::new_cuda(0) {
                log::info!("[Encoder] Using CUDA GPU");
                return Ok(device);
            }
        }

        log::info!("[Encoder] Using CPU");
        Ok(Device::Cpu)
    }

    // =========================================================================
    // RoPE (Rotary Position Embeddings)
    // =========================================================================

    /// Compute rotary position embeddings.
    ///
    /// RoPE encodes position by rotating query/key vectors:
    /// ```text
    /// q' = q * cos(θ) + rotate_half(q) * sin(θ)
    /// ```
    ///
    /// This allows extrapolation beyond training length.
    pub struct RotaryEmbedding {
        /// Cosine cache: [max_seq_len, head_dim/2]
        cos_cache: Tensor,
        /// Sine cache: [max_seq_len, head_dim/2]
        sin_cache: Tensor,
        /// Head dimension
        head_dim: usize,
    }

    impl RotaryEmbedding {
        pub fn new(head_dim: usize, max_seq_len: usize, theta: f64, device: &Device) -> Result<Self> {
            // Compute inverse frequencies
            let half_dim = head_dim / 2;
            let inv_freq: Vec<f32> = (0..half_dim)
                .map(|i| 1.0 / (theta.powf(i as f64 * 2.0 / head_dim as f64) as f32))
                .collect();

            // Position indices
            let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();

            // Compute angles: [max_seq_len, half_dim]
            let inv_freq_t = Tensor::from_vec(inv_freq.clone(), (1, half_dim), device)
                .map_err(|e| Error::Parse(format!("RoPE inv_freq: {}", e)))?;
            let positions_t = Tensor::from_vec(positions.clone(), (max_seq_len, 1), device)
                .map_err(|e| Error::Parse(format!("RoPE positions: {}", e)))?;

            let angles = positions_t
                .matmul(&inv_freq_t)
                .map_err(|e| Error::Parse(format!("RoPE angles: {}", e)))?;

            let cos_cache = angles
                .cos()
                .map_err(|e| Error::Parse(format!("RoPE cos: {}", e)))?;
            let sin_cache = angles
                .sin()
                .map_err(|e| Error::Parse(format!("RoPE sin: {}", e)))?;

            Ok(Self {
                cos_cache,
                sin_cache,
                head_dim,
            })
        }

        /// Apply rotary embeddings to query or key tensor.
        ///
        /// Input shape: [batch, seq_len, num_heads, head_dim]
        pub fn apply(&self, x: &Tensor, start_pos: usize) -> Result<Tensor> {
            let (batch, seq_len, num_heads, head_dim) = x.dims4()
                .map_err(|e| Error::Parse(format!("RoPE dims: {}", e)))?;

            // Get position-specific cos/sin
            let cos = self.cos_cache.i((start_pos..start_pos + seq_len, ..))
                .map_err(|e| Error::Parse(format!("RoPE cos slice: {}", e)))?;
            let sin = self.sin_cache.i((start_pos..start_pos + seq_len, ..))
                .map_err(|e| Error::Parse(format!("RoPE sin slice: {}", e)))?;

            // Split x into two halves
            let half_dim = head_dim / 2;
            let x1 = x.i((.., .., .., ..half_dim))
                .map_err(|e| Error::Parse(format!("RoPE x1: {}", e)))?;
            let x2 = x.i((.., .., .., half_dim..))
                .map_err(|e| Error::Parse(format!("RoPE x2: {}", e)))?;

            // Rotate: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
            let cos_exp = cos.unsqueeze(0)
                .map_err(|e| Error::Parse(format!("RoPE cos unsqueeze: {}", e)))?
                .unsqueeze(2)
                .map_err(|e| Error::Parse(format!("RoPE cos unsqueeze2: {}", e)))?;
            let sin_exp = sin.unsqueeze(0)
                .map_err(|e| Error::Parse(format!("RoPE sin unsqueeze: {}", e)))?
                .unsqueeze(2)
                .map_err(|e| Error::Parse(format!("RoPE sin unsqueeze2: {}", e)))?;

            let x1_cos = (&x1 * &cos_exp)
                .map_err(|e| Error::Parse(format!("RoPE x1*cos: {}", e)))?;
            let x2_sin = (&x2 * &sin_exp)
                .map_err(|e| Error::Parse(format!("RoPE x2*sin: {}", e)))?;
            let rotated_x1 = (&x1_cos - &x2_sin)
                .map_err(|e| Error::Parse(format!("RoPE rotated_x1: {}", e)))?;
            
            let x1_sin = (&x1 * &sin_exp)
                .map_err(|e| Error::Parse(format!("RoPE x1*sin: {}", e)))?;
            let x2_cos = (&x2 * &cos_exp)
                .map_err(|e| Error::Parse(format!("RoPE x2*cos: {}", e)))?;
            let rotated_x2 = (&x1_sin + &x2_cos)
                .map_err(|e| Error::Parse(format!("RoPE rotated_x2: {}", e)))?;

            Tensor::cat(&[&rotated_x1, &rotated_x2], D::Minus1)
                .map_err(|e| Error::Parse(format!("RoPE cat: {}", e)))
        }
    }

    // =========================================================================
    // GeGLU Activation
    // =========================================================================

    /// GeGLU activation: gate * GELU(x)
    ///
    /// Splits input in half, applies GELU to one half, multiplies.
    /// Better than standard GELU for language modeling.
    pub fn geglu(x: &Tensor) -> Result<Tensor> {
        let dim = x.dims().last().copied().unwrap_or(0);
        let half = dim / 2;

        let gate = x.i((.., ..half))
            .map_err(|e| Error::Parse(format!("GeGLU gate: {}", e)))?;
        let x_half = x.i((.., half..))
            .map_err(|e| Error::Parse(format!("GeGLU x: {}", e)))?;

        // GELU activation on gate using tensor method
        let gelu_gate = gate.gelu_erf()
            .map_err(|e| Error::Parse(format!("GeGLU gelu: {}", e)))?;

        (&gelu_gate * &x_half).map_err(|e| Error::Parse(format!("GeGLU mul: {}", e)))
    }

    // =========================================================================
    // Transformer Layer
    // =========================================================================

    /// Self-attention layer.
    pub struct Attention {
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        num_heads: usize,
        head_dim: usize,
        rope: Option<RotaryEmbedding>,
    }

    impl Attention {
        pub fn new(config: &EncoderConfig, vb: VarBuilder, device: &Device) -> Result<Self> {
            let hidden = config.hidden_size;
            let num_heads = config.num_attention_heads;
            let head_dim = hidden / num_heads;

            let q_proj = linear(hidden, hidden, vb.pp("q_proj"))
                .map_err(|e| Error::Retrieval(format!("Attention q_proj: {}", e)))?;
            let k_proj = linear(hidden, hidden, vb.pp("k_proj"))
                .map_err(|e| Error::Retrieval(format!("Attention k_proj: {}", e)))?;
            let v_proj = linear(hidden, hidden, vb.pp("v_proj"))
                .map_err(|e| Error::Retrieval(format!("Attention v_proj: {}", e)))?;
            let o_proj = linear(hidden, hidden, vb.pp("o_proj"))
                .map_err(|e| Error::Retrieval(format!("Attention o_proj: {}", e)))?;

            let rope = if config.use_rope {
                Some(RotaryEmbedding::new(
                    head_dim,
                    config.max_position_embeddings,
                    config.rope_theta,
                    device,
                )?)
            } else {
                None
            };

            Ok(Self {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                num_heads,
                head_dim,
                rope,
            })
        }

        pub fn forward(&self, hidden_states: &Tensor, start_pos: usize) -> Result<Tensor> {
            let (batch, seq_len, hidden) = hidden_states.dims3()
                .map_err(|e| Error::Parse(format!("Attention dims: {}", e)))?;

            // Project Q, K, V
            let q = self.q_proj.forward(hidden_states)
                .map_err(|e| Error::Parse(format!("Attention Q: {}", e)))?;
            let k = self.k_proj.forward(hidden_states)
                .map_err(|e| Error::Parse(format!("Attention K: {}", e)))?;
            let v = self.v_proj.forward(hidden_states)
                .map_err(|e| Error::Parse(format!("Attention V: {}", e)))?;

            // Reshape to [batch, seq, num_heads, head_dim]
            let q = q.reshape((batch, seq_len, self.num_heads, self.head_dim))?;
            let k = k.reshape((batch, seq_len, self.num_heads, self.head_dim))?;
            let v = v.reshape((batch, seq_len, self.num_heads, self.head_dim))?;

            // Apply RoPE if configured
            let (q, k) = if let Some(rope) = &self.rope {
                (rope.apply(&q, start_pos)?, rope.apply(&k, start_pos)?)
            } else {
                (q, k)
            };

            // Transpose for attention: [batch, num_heads, seq, head_dim]
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;

            // Scaled dot-product attention
            let scale = (self.head_dim as f64).sqrt();
            let attn_weights = (q.matmul(&k.transpose(2, 3)?)? / scale)?;
            let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)
                .map_err(|e| Error::Parse(format!("Attention softmax: {}", e)))?;
            let attn_output = attn_weights.matmul(&v)?;

            // Transpose back and reshape
            let attn_output = attn_output.transpose(1, 2)?;
            let attn_output = attn_output.reshape((batch, seq_len, hidden))?;

            // Output projection
            self.o_proj.forward(&attn_output)
                .map_err(|e| Error::Parse(format!("Attention output: {}", e)))
        }
    }

    /// Feed-forward network (MLP).
    pub struct FeedForward {
        up_proj: Linear,
        down_proj: Linear,
        use_geglu: bool,
    }

    impl FeedForward {
        pub fn new(config: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
            let hidden = config.hidden_size;
            let intermediate = if config.use_geglu {
                // GeGLU doubles the intermediate size then halves it
                config.intermediate_size * 2
            } else {
                config.intermediate_size
            };

            let up_proj = linear(hidden, intermediate, vb.pp("up_proj"))
                .map_err(|e| Error::Retrieval(format!("FFN up_proj: {}", e)))?;
            let down_proj = linear(config.intermediate_size, hidden, vb.pp("down_proj"))
                .map_err(|e| Error::Retrieval(format!("FFN down_proj: {}", e)))?;

            Ok(Self {
                up_proj,
                down_proj,
                use_geglu: config.use_geglu,
            })
        }

        pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
            let up = self.up_proj.forward(x)
                .map_err(|e| Error::Parse(format!("FFN up: {}", e)))?;

            let activated = if self.use_geglu {
                geglu(&up)?
            } else {
                // Use tensor method for GELU
                up.gelu_erf()
                    .map_err(|e| Error::Parse(format!("FFN gelu: {}", e)))?
            };

            self.down_proj.forward(&activated)
                .map_err(|e| Error::Parse(format!("FFN down: {}", e)))
        }
    }

    /// Transformer layer (attention + FFN).
    pub struct TransformerLayer {
        attention: Attention,
        ffn: FeedForward,
        ln1: LayerNorm,
        ln2: LayerNorm,
    }

    impl TransformerLayer {
        pub fn new(config: &EncoderConfig, vb: VarBuilder, device: &Device) -> Result<Self> {
            let attention = Attention::new(config, vb.pp("attention"), device)?;
            let ffn = FeedForward::new(config, vb.pp("ffn"))?;

            let ln1 = layer_norm(config.hidden_size, config.layer_norm_eps, vb.pp("ln1"))
                .map_err(|e| Error::Retrieval(format!("Layer ln1: {}", e)))?;
            let ln2 = layer_norm(config.hidden_size, config.layer_norm_eps, vb.pp("ln2"))
                .map_err(|e| Error::Retrieval(format!("Layer ln2: {}", e)))?;

            Ok(Self {
                attention,
                ffn,
                ln1,
                ln2,
            })
        }

        pub fn forward(&self, x: &Tensor, start_pos: usize) -> Result<Tensor> {
            // Pre-norm: LN -> Attention -> Residual
            let normed = self.ln1.forward(x)
                .map_err(|e| Error::Parse(format!("Layer ln1: {}", e)))?;
            let attn_out = self.attention.forward(&normed, start_pos)?;
            let x = (x + attn_out)?;

            // Pre-norm: LN -> FFN -> Residual
            let normed = self.ln2.forward(&x)
                .map_err(|e| Error::Parse(format!("Layer ln2: {}", e)))?;
            let ffn_out = self.ffn.forward(&normed)?;
            (&x + ffn_out).map_err(|e| Error::Parse(format!("Layer residual: {}", e)))
        }
    }

    // =========================================================================
    // Full Encoder
    // =========================================================================

    /// Pure Rust transformer encoder.
    pub struct CandleEncoder {
        config: EncoderConfig,
        embeddings: Embedding,
        layers: Vec<TransformerLayer>,
        final_norm: LayerNorm,
        tokenizer: Tokenizer,
        device: Device,
        architecture_name: String,
    }

    impl CandleEncoder {
        /// Create a new encoder with random weights (for testing).
        pub fn new_random(config: EncoderConfig, tokenizer: Tokenizer, name: &str) -> Result<Self> {
            let device = best_device()?;
            let varmap = candle_nn::VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

            let embeddings = embedding(config.vocab_size, config.hidden_size, vb.pp("embeddings"))
                .map_err(|e| Error::Retrieval(format!("Embeddings: {}", e)))?;

            let mut layers = Vec::new();
            for i in 0..config.num_hidden_layers {
                layers.push(TransformerLayer::new(&config, vb.pp(format!("layer_{}", i)), &device)?);
            }

            let final_norm = layer_norm(config.hidden_size, config.layer_norm_eps, vb.pp("final_norm"))
                .map_err(|e| Error::Retrieval(format!("Final norm: {}", e)))?;

            Ok(Self {
                config,
                embeddings,
                layers,
                final_norm,
                tokenizer,
                device,
                architecture_name: name.to_string(),
            })
        }

        /// Load encoder from HuggingFace model (safetensors).
        pub fn from_pretrained(model_id: &str) -> Result<Self> {
            use hf_hub::api::sync::Api;

            let api = Api::new().map_err(|e| {
                Error::Retrieval(format!("HF API: {}", e))
            })?;

            let repo = api.model(model_id.to_string());

            // Download config, weights, tokenizer
            let config_path = repo.get("config.json")
                .map_err(|e| Error::Retrieval(format!("config.json: {}", e)))?;
            let weights_path = repo.get("model.safetensors")
                .or_else(|_| repo.get("pytorch_model.bin"))
                .map_err(|e| Error::Retrieval(format!("weights: {}", e)))?;
            let tokenizer_path = repo.get("tokenizer.json")
                .map_err(|e| Error::Retrieval(format!("tokenizer: {}", e)))?;

            // Parse config
            let config_str = std::fs::read_to_string(&config_path)
                .map_err(|e| Error::Retrieval(format!("read config: {}", e)))?;
            let config = Self::parse_config(&config_str)?;

            // Load tokenizer
            let tokenizer = Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| Error::Retrieval(format!("tokenizer: {}", e)))?;

            // Load weights
            let device = best_device()?;
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                    .map_err(|e| Error::Retrieval(format!("safetensors: {}", e)))?
            };

            let embeddings = embedding(config.vocab_size, config.hidden_size, vb.pp("embeddings.word_embeddings"))
                .map_err(|e| Error::Retrieval(format!("Embeddings: {}", e)))?;

            let mut layers = Vec::new();
            for i in 0..config.num_hidden_layers {
                layers.push(TransformerLayer::new(&config, vb.pp(format!("encoder.layer.{}", i)), &device)?);
            }

            let final_norm = layer_norm(config.hidden_size, config.layer_norm_eps, vb.pp("encoder.final_layer_norm"))
                .map_err(|e| Error::Retrieval(format!("Final norm: {}", e)))?;

            // Detect architecture
            let arch_name = if config.use_rope { "ModernBERT" } else { "BERT" };

            Ok(Self {
                config,
                embeddings,
                layers,
                final_norm,
                tokenizer,
                device,
                architecture_name: arch_name.to_string(),
            })
        }

        fn parse_config(json: &str) -> Result<EncoderConfig> {
            let v: serde_json::Value = serde_json::from_str(json)
                .map_err(|e| Error::Parse(format!("config JSON: {}", e)))?;

            // Detect architecture from model_type
            let model_type = v["model_type"].as_str().unwrap_or("bert");
            let is_modern = model_type.contains("modern") || v.get("rope_theta").is_some();

            Ok(EncoderConfig {
                vocab_size: v["vocab_size"].as_u64().unwrap_or(30522) as usize,
                hidden_size: v["hidden_size"].as_u64().unwrap_or(768) as usize,
                num_attention_heads: v["num_attention_heads"].as_u64().unwrap_or(12) as usize,
                num_hidden_layers: v["num_hidden_layers"].as_u64().unwrap_or(12) as usize,
                intermediate_size: v["intermediate_size"].as_u64().unwrap_or(3072) as usize,
                max_position_embeddings: v["max_position_embeddings"].as_u64().unwrap_or(512) as usize,
                hidden_dropout_prob: v["hidden_dropout_prob"].as_f64().unwrap_or(0.1) as f32,
                layer_norm_eps: v["layer_norm_eps"].as_f64().unwrap_or(1e-12),
                use_rope: is_modern,
                use_geglu: is_modern,
                rope_theta: v["rope_theta"].as_f64().unwrap_or(10000.0),
            })
        }

        fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
            // Get embeddings
            let mut hidden = self.embeddings.forward(input_ids)
                .map_err(|e| Error::Parse(format!("Embeddings forward: {}", e)))?;

            // Pass through layers
            for layer in &self.layers {
                hidden = layer.forward(&hidden, 0)?;
            }

            // Final norm
            self.final_norm.forward(&hidden)
                .map_err(|e| Error::Parse(format!("Final norm: {}", e)))
        }
    }

    impl TextEncoder for CandleEncoder {
        fn encode(&self, text: &str) -> Result<(Vec<f32>, usize)> {
            // Tokenize
            let encoding = self.tokenizer
                .encode(text, true)
                .map_err(|e| Error::Parse(format!("Tokenize: {}", e)))?;

            let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
            let seq_len = input_ids.len().min(self.config.max_position_embeddings);
            let input_ids = &input_ids[..seq_len];

            // Create tensor
            let input_tensor = Tensor::from_vec(input_ids.to_vec(), (1, seq_len), &self.device)
                .map_err(|e| Error::Parse(format!("Input tensor: {}", e)))?;

            // Forward pass
            let output = self.forward(&input_tensor)?;

            // Extract to CPU
            let output_flat = output.flatten_all()
                .map_err(|e| Error::Parse(format!("Flatten: {}", e)))?;
            let embeddings = output_flat.to_vec1::<f32>()
                .map_err(|e| Error::Parse(format!("To vec: {}", e)))?;

            Ok((embeddings, seq_len))
        }

        fn hidden_dim(&self) -> usize {
            self.config.hidden_size
        }

        fn max_length(&self) -> usize {
            self.config.max_position_embeddings
        }

        fn architecture(&self) -> &str {
            &self.architecture_name
        }
    }
}

// Re-export candle implementations
#[cfg(feature = "candle")]
pub use candle_impl::*;

// =============================================================================
// Stub for non-candle builds
// =============================================================================

#[cfg(not(feature = "candle"))]
pub struct CandleEncoder;

#[cfg(not(feature = "candle"))]
impl CandleEncoder {
    pub fn new_random(_config: EncoderConfig, _name: &str) -> Result<Self> {
        Err(Error::FeatureNotAvailable(
            "CandleEncoder requires 'candle' feature".into()
        ))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_config_defaults() {
        let bert = EncoderConfig::bert_base();
        assert_eq!(bert.hidden_size, 768);
        assert_eq!(bert.max_position_embeddings, 512);
        assert!(!bert.use_rope);

        let modern = EncoderConfig::modernbert_base();
        assert_eq!(modern.hidden_size, 768);
        assert_eq!(modern.max_position_embeddings, 8192);
        assert!(modern.use_rope);
        assert!(modern.use_geglu);
    }

    #[test]
    fn test_modernbert_large() {
        let config = EncoderConfig::modernbert_large();
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_hidden_layers, 28);
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_geglu() {
        use candle_core::{Device, Tensor};

        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1., (2, 8), &device).unwrap();
        let result = candle_impl::geglu(&x);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.dims(), &[2, 4]);
    }
}

