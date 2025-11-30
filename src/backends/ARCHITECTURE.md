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
│  │ PatternNER  │    │HeuristicNER│    │ StackedNER │                     │
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
| `HeuristicNER` | - | No | No | ~50µs | ~65% | Quick PER/ORG/LOC |
| `StackedNER` | - | No | No | varies | varies | Default/baseline |
| `GLiNEROnnx` | `onnx` | **Yes** | No | ~100ms | ~92% | Custom entity types |
| `GLiNERCandle` | `candle` | **Yes** | No | ~80ms | TBD | Apple Silicon GPU |
| `NuNER` | `onnx` | **Yes** | No | ~100ms | ~92% | Arbitrary-length entities |
| `W2NER` | `onnx` | No | **Yes** | ~150ms | ~88% | Medical/nested NER |
| `BertNEROnnx` | `onnx` | No | No | ~50ms | ~86% | Production (fixed types) |
| `CandleNER` | `candle` | No | No | ~50ms | ~74% | Rust-native ML |

*PatternNER only detects structured entities (dates, money, etc.) - not named entities.

## ⭐ Recommended: GLiNER2 (Multi-Task)

**For new projects, use `GLiNER2`** - it's the most capable and efficient option.

| Feature | GLiNER (v1) | GLiNER2 |
|---------|-------------|---------|
| Named Entity Recognition | ✓ | ✓ |
| Text Classification | ✗ | ✓ |
| Hierarchical Extraction | ✗ | ✓ |
| Multi-task in single pass | ✗ | ✓ |
| Parameters | ~300M | ~205M |
| CPU Latency | ~100ms | ~130-200ms |

**Why GLiNER2?**

1. **Unified extraction**: NER + classification + structure in one forward pass
2. **Smaller model**: 205M params vs 7B+ for LLMs
3. **CPU-efficient**: 130-208ms on CPU (2.6× faster than GPT-4o API)
4. **Zero-shot F1**: Competitive with GPT-4o on CrossNER benchmarks

```rust
use anno::backends::gliner2::{GLiNER2, TaskSchema};

let model = GLiNER2::from_pretrained("knowledgator/gliner-multitask-large-v0.5")?;

// Multi-task schema
let schema = TaskSchema::new()
    .with_entities(&["person", "organization", "product"])
    .with_classification("sentiment", &["positive", "negative"]);

let result = model.extract_with_schema("Apple announced iPhone 15", &schema)?;
```

**Reference**: arXiv:2507.18546 (July 2025)

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
| `HeuristicNER` | 0.5 | Balance precision/recall |
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
# eval includes dirs for platform cache directory
# eval-advanced includes ureq for dataset download
full = ["eval-full", "onnx", "candle", "discourse"]
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

## Production Infrastructure

Anno provides several production-ready features for deployment:

### Async Inference (Feature: `async-inference`)

Wrap blocking ONNX inference for async runtimes:

```rust
use anno::{GLiNEROnnx, backends::IntoAsync};

let model = GLiNEROnnx::new("onnx-community/gliner_small-v2.1")?;
let async_model = model.into_async();

// Safe to use in tokio handlers
let entities = async_model.extract_entities("John works at Apple").await?;
```

### Session Pooling (Feature: `session-pool`)

Enable parallel inference with multiple ONNX sessions:

```rust
use anno::backends::{GLiNERPool, PoolConfig};

let pool = GLiNERPool::new(
    "onnx-community/gliner_small-v2.1",
    PoolConfig::with_size(4),  // 4 parallel sessions
)?;

// Each call uses an available session
let entities = pool.extract(text, &["person", "organization"], 0.5)?;
```

### Quantized Models

INT8 quantization for 2-4x CPU speedup:

```rust
use anno::{GLiNEROnnx, backends::GLiNERConfig};

let config = GLiNERConfig {
    prefer_quantized: true,   // Try model_quantized.onnx first
    optimization_level: 3,     // Max ONNX optimization
    num_threads: 8,            // Inference threads
};

let model = GLiNEROnnx::with_config("model-id", config)?;
println!("Using INT8: {}", model.is_quantized());
```

### Model Warmup

Mitigate cold-start latency in serverless environments:

```rust
use anno::backends::{warmup_model, WarmupConfig};

let model = GLiNEROnnx::new("onnx-community/gliner_small-v2.1")?;

let result = warmup_model(&model, WarmupConfig::default())?;
println!("Speedup: {:.2}x", result.speedup);
// First inference: 450ms → Warm inference: ~100ms
```

### Entity Validation

Verify extraction quality before downstream use:

```rust
use anno::{Entity, EntityType};

let text = "John works at Apple";
let entities = model.extract_entities(text, None)?;

// Validate all entities against source text
let issues = Entity::validate_batch(&entities, text);
if !issues.is_empty() {
    for (idx, errs) in &issues {
        for err in errs {
            eprintln!("Entity {}: {}", idx, err);
        }
    }
}
```

### Production Feature Bundle

Enable all production features at once:

```toml
[dependencies]
anno = { version = "0.2", features = ["production"] }
```

This enables:
- `async-inference` - spawn_blocking wrapper
- `session-pool` - parallel ONNX sessions
- `fast-lock` - parking_lot mutexes
- `onnx` - ONNX Runtime backend

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

