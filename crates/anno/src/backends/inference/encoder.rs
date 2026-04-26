//! Core encoder traits for GLiNER/ModernBERT-style bi-encoder extraction.

use crate::RaggedBatch;
#[allow(unused_imports)]
use crate::{Entity, EntityType};

// Core Encoder Traits (GLiNER/ModernBERT Alignment)
// =============================================================================

/// Text encoder trait for transformer-based encoders.
///
/// # Motivation
///
/// Modern NER systems require converting raw text into dense vector representations
/// that capture semantic meaning. This trait abstracts the encoding step, allowing
/// different transformer architectures to be used interchangeably.
///
/// # Supported Architectures
///
/// | Architecture | Context | Key Features | Speed |
/// |--------------|---------|--------------|-------|
/// | ModernBERT   | 8,192   | RoPE, GeGLU, unpadded inference | 3x faster |
/// | DeBERTaV3    | 512     | Disentangled attention | Baseline |
/// | BERT/RoBERTa | 512     | Classic, widely available | Baseline |
///
/// # Research Alignment (ModernBERT, Dec 2024)
///
/// From ModernBERT paper (arXiv:2412.13663):
/// > "Pareto improvements to BERT... encoder-only models offer great
/// > performance-size tradeoff for retrieval and classification."
///
/// Key innovations:
/// - **Alternating Attention**: Global attention every 3 layers, local (128-token
///   window) elsewhere. Reduces complexity for long sequences.
/// - **Unpadding**: "ModernBERT unpads inputs *before* the token embedding layer
///   and optionally repads model outputs leading to a 10-to-20 percent
///   performance improvement over previous methods."
/// - **RoPE**: Rotary positional embeddings enable extrapolation to longer sequences.
/// - **GeGLU**: Gated activation function improves over GELU.
///
/// # Example
///
/// ```ignore
/// use anno::TextEncoder;
///
/// fn process_document(encoder: &dyn TextEncoder, text: &str) {
///     let output = encoder.encode(text).unwrap();
///     println!("Encoded {} tokens into {} dimensions",
///              output.num_tokens, output.hidden_dim);
///
///     // Token offsets map back to character positions
///     for (i, (start, end)) in output.token_offsets.iter().enumerate() {
///         println!("Token {}: chars {}..{}", i, start, end);
///     }
/// }
/// ```
pub trait TextEncoder: Send + Sync {
    /// Encode text into token embeddings.
    ///
    /// # Arguments
    /// * `text` - Input text to encode
    ///
    /// # Returns
    /// * Token embeddings as flattened [num_tokens, hidden_dim]
    /// * Attention mask indicating valid tokens
    fn encode(&self, text: &str) -> crate::Result<EncoderOutput>;

    /// Encode a batch of texts.
    ///
    /// # Arguments
    /// * `texts` - Batch of input texts
    ///
    /// # Returns
    /// * RaggedBatch containing all embeddings with document boundaries
    fn encode_batch(&self, texts: &[&str]) -> crate::Result<(Vec<f32>, RaggedBatch)>;

    /// Get the hidden dimension of the encoder.
    fn hidden_dim(&self) -> usize;

    /// Get the maximum sequence length.
    fn max_length(&self) -> usize;

    /// Get the encoder architecture name.
    fn architecture(&self) -> &'static str;
}

/// Output from text encoding.
#[derive(Debug, Clone)]
pub struct EncoderOutput {
    /// Token embeddings: [num_tokens, hidden_dim]
    pub embeddings: Vec<f32>,
    /// Number of tokens
    pub num_tokens: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Token-to-character mapping (for span recovery)
    pub token_offsets: Vec<(usize, usize)>,
}

// =============================================================================
