# ModernBERT Integration in Anno

This document explains how anno integrates ModernBERT, a modern bidirectional encoder
that offers significant improvements over classic BERT for NER tasks.

## What is ModernBERT?

ModernBERT (December 2024, arXiv:2412.13663) is a modern redesign of BERT that combines:

| Feature | BERT | ModernBERT |
|---------|------|------------|
| Context length | 512 tokens | **8,192 tokens** |
| Position encoding | Absolute (APE) | **Rotary (RoPE)** |
| Activation | GELU | **GeGLU** |
| Dropout | 0.1 | **0.0** (inference) |
| Memory efficiency | Standard | **Unpadding** |

### Why It Matters for NER

1. **Long documents**: Process entire documents without chunking, preserving cross-sentence context
2. **Better extrapolation**: RoPE enables handling sequences slightly longer than training length
3. **Improved accuracy**: GeGLU activation functions provide better gradient flow

## Architecture Comparison

```
BERT-base (110M params):
  vocab_size: 30,522
  hidden_size: 768
  num_layers: 12
  max_position: 512
  
ModernBERT-base (149M params):
  vocab_size: 50,368
  hidden_size: 768
  num_layers: 22
  max_position: 8,192
  
ModernBERT-large (395M params):
  vocab_size: 50,368
  hidden_size: 1,024
  num_layers: 28
  max_position: 8,192
```

## Usage in Anno

### With Candle Backend (Pure Rust)

```rust,ignore
use anno::backends::encoder_candle::{CandleEncoder, EncoderArchitecture};

// Use ModernBERT encoder
let encoder = CandleEncoder::from_architecture(EncoderArchitecture::ModernBert)?;

// Process long document (up to 8192 tokens)
let long_document = "..." ; // Very long text
let (embeddings, seq_len) = encoder.encode(long_document)?;
```

### Automatic Detection

The encoder configuration is automatically detected from model name:

```rust,ignore
use anno::backends::encoder_candle::EncoderConfig;

// Detects ModernBERT configuration
let config = EncoderConfig::from_model_name("answerdotai/ModernBERT-base");
assert!(config.use_rope);
assert_eq!(config.max_position_embeddings, 8192);
```

## Configuration Details

### RoPE (Rotary Position Embeddings)

```rust
// ModernBERT uses RoPE with higher theta for long context
pub struct EncoderConfig {
    use_rope: true,
    rope_theta: 160000.0,  // vs 10000 for standard RoPE
}
```

Higher `rope_theta` enables better extrapolation to longer sequences. The
relationship between `rope_theta` and effective context is studied in
arXiv:2405.14591 ("Base of RoPE Bounds Context Length").

### GeGLU Activation

GeGLU (Gated Linear Unit with GELU) replaces the standard FFN:

```
Standard FFN:   output = GELU(xW1) @ W2
GeGLU FFN:      output = (GELU(xW1) * (xW3)) @ W2
```

This provides better gradient flow and slightly improves accuracy.

## Supported Models

| Model ID | Architecture | Params | Context |
|----------|--------------|--------|---------|
| `answerdotai/ModernBERT-base` | ModernBERT | 149M | 8,192 |
| `answerdotai/ModernBERT-large` | ModernBERT | 395M | 8,192 |
| `llm-jp/modernbert-ja-130m` | ModernBERT (Japanese) | 130M | 8,192 |

## Research References

1. **ModernBERT**: Warner et al., "Smarter, Better, Faster, Longer" (arXiv:2412.13663)
2. **RoPE**: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
3. **RoPE Bounds**: Men et al., "Base of RoPE Bounds Context Length" (arXiv:2405.14591)
4. **GeGLU**: Shazeer, "GLU Variants Improve Transformer"

## Implementation Notes

### Candle Implementation

The pure Rust implementation in `src/backends/encoder_candle.rs` supports:

- Automatic model detection from HuggingFace model ID
- Metal (Apple Silicon) and CUDA acceleration
- Memory-efficient unpadding for variable-length batches

### ONNX Fallback

For systems without GPU support, the ONNX runtime provides a CPU fallback
with optimized kernels. Note that ONNX models may need re-export for
ModernBERT's GeGLU activation.

## Future Work

- [ ] Flash Attention integration for even longer contexts
- [ ] Quantization (INT8/INT4) for deployment
- [ ] Multi-GPU sharding for ModernBERT-large

