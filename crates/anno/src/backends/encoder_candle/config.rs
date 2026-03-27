use super::*;

// =============================================================================
// Core Trait
// =============================================================================

/// Trait for text-to-embedding encoders.
///
/// This is the main abstraction that allows swapping BERT/RoBERTa/ModernBERT.
pub trait CandleTextEncoder: Send + Sync {
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
    /// Whether to use pre-norm (ModernBERT) vs post-norm (classic BERT)
    /// Pre-norm: LN -> Attention -> Residual
    /// Post-norm: Attention -> Residual -> LN (classic BERT)
    pub use_pre_norm: bool,
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
            use_pre_norm: false, // Classic BERT uses post-norm
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
            use_pre_norm: true,   // ModernBERT uses pre-norm
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
            use_pre_norm: true, // ModernBERT uses pre-norm
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
            use_pre_norm: true, // DeBERTa uses pre-norm
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
            use_pre_norm: true, // DeBERTa uses pre-norm
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
