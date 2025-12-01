# Encoder Trait Design

## Overview

There are two `TextEncoder` trait definitions in the codebase, serving different purposes:

1. **`src/backends/inference.rs::TextEncoder`** - Comprehensive API for future ONNX/other backends
2. **`src/backends/encoder_candle.rs::TextEncoder`** - Candle-specific implementation

## Design Rationale

### Why Two Traits?

The traits serve different abstraction levels:

- **`inference.rs` version**: Higher-level API designed for bi-encoder architectures (GLiNER-style). Returns `EncoderOutput` with rich metadata including token offsets and supports `RaggedBatch` for efficient unpadded batching.

- **`encoder_candle.rs` version**: Lower-level API optimized for Candle's tensor operations. Returns simple `(Vec<f32>, usize)` tuples that map directly to Candle's `Tensor` types.

### Current Usage

**Candle backends use `encoder_candle.rs::TextEncoder`**:
- `GLiNERCandle`
- `CandleNER`
- `GLiNER2Candle`
- `GLiNERPipeline`

**`inference.rs::TextEncoder` is defined but not yet implemented**:
- Intended for future ONNX encoder backends
- Provides richer metadata (token offsets, RaggedBatch)
- Better suited for span-based NER architectures

## Future Consolidation

When ONNX encoders are implemented, we have two options:

1. **Adapter pattern**: Implement `inference.rs::TextEncoder` for Candle encoders via an adapter
2. **Unified trait**: Merge both into a single trait with default implementations

For now, keeping them separate is pragmatic:
- Candle backends work efficiently with the simpler API
- Future ONNX backends can use the richer API
- No performance overhead from unnecessary abstractions

## LabelEncoder

Similarly, `LabelEncoder` exists only in `inference.rs` and is used by:
- `GLiNEROnnx` (via `BiEncoder` trait)
- `GLiNERCandle` (implements `LabelEncoder` directly)

The Candle version doesn't need a separate `LabelEncoder` trait because it uses the same `CandleEncoder` for both text and labels (with different prompt formatting).

## Recommendations

1. **Keep both traits** for now - they serve different purposes
2. **Document the distinction** clearly (this document)
3. **Consider adapter** if we need to use Candle encoders with `inference.rs` APIs
4. **Unify later** if the abstraction overhead becomes negligible

