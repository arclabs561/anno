# anno Examples

Focused examples demonstrating core capabilities. Each serves a distinct purpose.

## Quick Start

```bash
# Basic evaluation (no ML dependencies)
cargo run --example quickstart --features eval
```

## Examples Overview

| Example | Features | Purpose |
|---------|----------|---------|
| `quickstart` | eval | Entry point - evaluate PatternNER on custom data |
| `bert` | onnx | BERT NER demo - standard named entity recognition |
| `candle` | candle | Candle architecture - pure Rust ML backend |
| `download_models` | network | Utility - pre-download models for offline use |
| `eval` | eval | Evaluation framework - synthetic + real datasets |
| `coref` | eval | Coreference resolution metrics (MUC, B³, CEAF, LEA) |
| `bias` | eval-bias | Bias analysis - gender, demographic, temporal |
| `benchmark` | eval-full | Comprehensive quality metrics + bias + robustness |
| `advanced` | eval | Discontinuous NER, relation extraction, visual NER |
| `models` | onnx | Model showcase - all backends + zero-shot NER |
| `hybrid` | onnx | Hybrid evaluation - ML + pattern combined |

## Running Examples

```bash
# Pattern-only (no downloads)
cargo run --example quickstart --features eval
cargo run --example eval --features eval

# ML backends (requires model download)
cargo run --example bert --features onnx
cargo run --example models --features onnx

# Full benchmark suite
cargo run --example benchmark --features eval-full

# With real datasets
cargo run --example benchmark --features "eval-full,network"
```

## Backend Comparison

| Backend | Speed | Accuracy | Entity Types |
|---------|-------|----------|--------------|
| PatternNER | ~500ns | High* | DATE, MONEY, EMAIL, URL, PHONE |
| StatisticalNER | ~50us | ~65% | PER, ORG, LOC |
| StackedNER | ~60us | ~75% | Combined |
| BertNEROnnx | ~20ms | ~86% | PER, ORG, LOC, MISC |
| GLiNEROnnx | ~80ms | ~86% | Any (zero-shot) |

\* High precision on structured patterns only

## Zero-Shot NER

GLiNER supports custom entity types at inference time:

```rust
let gliner = GLiNEROnnx::new("onnx-community/gliner_small-v2.1")?;
let entities = gliner.extract_with_types(
    "Patient has severe headache, prescribed 400mg ibuprofen",
    &["symptom", "medication", "dosage"],
    0.4,
)?;
```

See `models.rs` for comprehensive zero-shot examples.

