# Backend Architecture

## Backend Taxonomy

```text
┌────────────────────────────────────────────────────────────────────────────┐
│                           NER Backend Architecture                          │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ZERO-SHOT CAPABLE (any entity type at runtime)                            │
│  ──────────────────────────────────────────────                            │
│                                                                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│  │ GLiNEROnnx  │    │ GLiNERCandle│    │   NuNER     │                     │
│  │ (manual)    │    │ (Metal GPU) │    │ (token clf) │                     │
│  │             │    │             │    │             │                     │
│  │ Feature:    │    │ Feature:    │    │ Feature:    │                     │
│  │   onnx      │    │   candle    │    │   onnx      │                     │
│  │             │    │             │    │             │                     │
│  │ Speed: ~100ms    │ Speed: ~80ms│    │ Speed: ~100ms                     │
│  │ Accuracy: SOTA   │ Accuracy: ? │    │ Accuracy: SOTA                    │
│  └─────────────┘    └─────────────┘    └─────────────┘                     │
│         │                  │                  │                            │
│         └──────────────────┼──────────────────┘                            │
│                            │                                               │
│                   Uses bi-encoder architecture:                            │
│                   Text spans + Entity labels → same embedding space        │
│                                                                            │
│  ──────────────────────────────────────────────                            │
│  COMPLEX STRUCTURES (nested/discontinuous)                                 │
│  ──────────────────────────────────────────────                            │
│                                                                            │
│  ┌─────────────┐    ┌─────────────────────┐                                │
│  │   W2NER     │    │ HandshakingMatrix   │                                │
│  │ (word-word) │    │ (TPLinker)          │                                │
│  │             │    │                     │                                │
│  │ Feature:    │    │ Feature: always     │                                │
│  │   onnx      │    │ (data structure)    │                                │
│  │             │    │                     │                                │
│  │ Handles:    │    │ Handles:            │                                │
│  │  - Nested   │    │  - Relations        │                                │
│  │  - Discontin│    │  - Joint extraction │                                │
│  └─────────────┘    └─────────────────────┘                                │
│                                                                            │
│  ──────────────────────────────────────────────                            │
│  TRADITIONAL ML (fixed entity types)                                       │
│  ──────────────────────────────────────────────                            │
│                                                                            │
│  ┌─────────────┐    ┌─────────────┐                                        │
│  │ BertNEROnnx │    │ CandleNER   │                                        │
│  │ (ONNX)      │    │ (Candle)    │                                        │
│  │             │    │             │                                        │
│  │ Feature:    │    │ Feature:    │                                        │
│  │   onnx      │    │   candle    │                                        │
│  │             │    │             │                                        │
│  │ Fixed types:│    │ Fixed types:│                                        │
│  │ PER/ORG/LOC │    │ PER/ORG/LOC │                                        │
│  │ MISC/DATE   │    │ MISC        │                                        │
│  │             │    │             │                                        │
│  │ Speed: ~50ms│    │ Speed: ~50ms│                                        │
│  │ F1: ~86%    │    │ F1: ~74%    │                                        │
│  └─────────────┘    └─────────────┘                                        │
│                                                                            │
│  ──────────────────────────────────────────────                            │
│  ZERO DEPENDENCY (always available)                                        │
│  ──────────────────────────────────────────────                            │
│                                                                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│  │ PatternNER  │    │StatisticalNER│    │ StackedNER │                     │
│  │ (regex)     │    │ (heuristics)│    │ (composite) │                     │
│  │             │    │             │    │             │                     │
│  │ Feature:    │    │ Feature:    │    │ Feature:    │                     │
│  │   none      │    │   none      │    │   none      │                     │
│  │             │    │             │    │             │                     │
│  │ Types:      │    │ Types:      │    │ Combines:   │                     │
│  │  Date/Time  │    │  Person     │    │  Pattern+   │                     │
│  │  Money/%    │    │  Org        │    │  Statistical│                     │
│  │  Email/URL  │    │  Location   │    │  +optional  │                     │
│  │  Phone      │    │             │    │   ML        │                     │
│  │             │    │             │    │             │                     │
│  │ Speed: <1µs │    │ Speed: ~50µs│    │ Speed: varies                     │
│  │ F1: ~95%*   │    │ F1: ~65%    │    │ F1: varies  │                     │
│  └─────────────┘    └─────────────┘    └─────────────┘                     │
│  * High precision on structured entities only                              │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

## Backend Comparison Table

| Backend | Feature | Zero-Shot | Nested | Speed | F1 | Use When |
|---------|---------|-----------|--------|-------|-----|----------|
| `PatternNER` | - | No | No | ~400ns | ~95%* | Structured data extraction |
| `StatisticalNER` | - | No | No | ~50µs | ~65% | Quick PER/ORG/LOC |
| `StackedNER` | - | No | No | varies | varies | Default/baseline |
| `GLiNEROnnx` | `onnx` | **Yes** | No | ~100ms | ~92% | Custom entity types |
| `GLiNERCandle` | `candle` | **Yes** | No | ~80ms | TBD | Apple Silicon GPU |
| `NuNER` | `onnx` | **Yes** | No | ~100ms | ~92% | Arbitrary-length entities |
| `W2NER` | `onnx` | No | **Yes** | ~150ms | ~88% | Medical/nested NER |
| `BertNEROnnx` | `onnx` | No | No | ~50ms | ~86% | Production (fixed types) |
| `CandleNER` | `candle` | No | No | ~50ms | ~74% | Rust-native ML |

*PatternNER only detects structured entities (dates, money, etc.) - not named entities.

## GLiNER Variant Explained

We have multiple GLiNER implementations for different use cases:

### GLiNEROnnx (Manual ONNX)

**File**: `gliner_onnx.rs`
**Feature**: `onnx`
**Dependency**: `ort` (ONNX Runtime), `tokenizers`

Our hand-written ONNX inference implementation. Directly implements the GLiNER prompt format:

```text
[START] <<ENT>> type1 <<ENT>> type2 <<SEP>> word1 word2 ... [END]
```

**Pros**:
- Full control over tokenization
- Transparent error handling
- Custom span decoding

**Cons**:
- Complex code (~500 lines)
- May drift from reference implementation
- Less tested than gline-rs

### GLiNERCandle (Pure Rust)

**File**: `gliner_candle.rs`
**Feature**: `candle`
**Dependency**: `candle-core`, `candle-nn`, `candle-transformers`

Placeholder for pure Rust implementation using Candle ML framework.

**Pros**:
- Metal (Apple GPU) support
- CUDA support
- No C++ dependencies (pure Rust)
- Quantization support

**Cons**:
- Not yet implemented
- Requires porting model weights

### Why Multiple Implementations?

1. **Portability**: ONNX works everywhere, Candle requires GPU setup
2. **Performance**: Candle can leverage Metal/CUDA, ONNX relies on onnxruntime
3. **Dependencies**: ONNX needs onnxruntime (~200MB), Candle is pure Rust
4. **Control**: Manual impl allows debugging, library hides complexity

## NuNER vs GLiNER

Both use bi-encoder architecture but differ in output format:

| Aspect | GLiNER | NuNER |
|--------|--------|-------|
| Output | Span classification | Token classification (BIO) |
| Entity length | Limited by span window | Arbitrary |
| Training data | Academic NER | Pile + C4 (1M examples) |
| License | Apache 2.0 | MIT |

**When to use NuNER**: Long entities, variable-length spans (e.g., "Intergovernmental Panel on Climate Change")

## W2NER for Complex Structures

W2NER models NER as word-word relation classification:

```text
     New  York  City  is  great
New   B   NNW   THW   -    -
York  -    B    NNW   -    -
City  -    -     B    -    -
is    -    -     -    -    -
great -    -     -    -    -
```

**Relations**:
- `B`: Begin entity
- `NNW`: Next-Neighboring-Word (same entity)
- `THW`: Tail-Head-Word (entity boundary)

**Handles**:
- Nested: "University of [California]" (ORG + LOC)
- Discontinuous: "severe [pain] in [abdomen]" → "severe abdominal pain"
- Overlapping: Same span, different types

## Encoder Comparison (2024-2025)

| Encoder | Context | Speed | Quality | Notes |
|---------|---------|-------|---------|-------|
| BERT | 512 | Baseline | Good | Classic, well-tested |
| DeBERTa-v3 | 512 | 0.8x | Better | Improved attention |
| ModernBERT | 8192 | 1.0x | SOTA | Dec 2024, RoPE, unpadding |
| RoBERTa | 512 | 1.0x | Good | Improved pretraining |

**ModernBERT** (Dec 2024) is the new default for GLiNER models:
- 8192 token context (vs 512)
- RoPE positional encoding
- Unpadding for efficiency
- GeGLU activation

## Default Thresholds (Data-Motivated)

Based on benchmarking against CoNLL-2003 and synthetic datasets:

| Backend | Default Threshold | Rationale |
|---------|-------------------|-----------|
| `PatternNER` | Pattern-specific (0.90-0.98) | High precision regexes |
| `StatisticalNER` | 0.5 | Balance precision/recall |
| `GLiNEROnnx` | 0.5 | GLiNER paper default |
| `W2NER` | 0.5 | Standard for grid models |
| `BertNEROnnx` | N/A (argmax) | Sequence labeling |

## Feature Flags

```toml
[features]
default = []
onnx = ["dep:ort", "dep:tokenizers", "dep:hf-hub", "dep:ndarray"]
candle = ["dep:candle-core", "dep:candle-nn", "dep:candle-transformers"]
metal = ["candle", "candle-core/metal", "candle-nn/metal"]
cuda = ["candle", "candle-core/cuda", "candle-nn/cuda"]
network = ["dep:ureq", "dep:dirs"]
full = ["onnx", "candle", "network"]
```

## Recommended Configuration

### Minimal (Zero Dependencies)

```rust
use anno::StackedNER;
let ner = StackedNER::default();
```

### Production (Fixed Types)

```rust
use anno::BertNEROnnx;
let ner = BertNEROnnx::new("protectai/bert-base-NER-onnx")?;
```

### Zero-Shot (Custom Types)

```rust
use anno::GLiNEROnnx;
let ner = GLiNEROnnx::new("onnx-community/gliner_small-v2.1")?;
let entities = ner.extract_entities(text, Some(&["product", "company"]))?;
```

### Apple Silicon (GPU)

```rust
// Requires `candle` + `metal` features
use anno::GLiNERCandle;
let ner = GLiNERCandle::new("gliner-small-v2.1")?;
```

## Timeline

- **Nov 2023**: GLiNER paper published
- **Jan 2024**: gline-rs first release
- **Apr 2024**: NuNER released by NuMind
- **Dec 2024**: ModernBERT released
- **Jun 2025**: gline-rs 1.0.0 (production-ready)
- **Nov 2025**: Anno backend consolidation

## References

1. [GLiNER Paper](https://arxiv.org/abs/2311.08526)
2. [gline-rs](https://github.com/fbilhaut/gline-rs) - Rust GLiNER inference
3. [W2NER Paper](https://arxiv.org/abs/2112.10070)
4. [NuNER](https://huggingface.co/numind/NuNER_Zero) - Token classifier
5. [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base)

