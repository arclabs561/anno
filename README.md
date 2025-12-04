# anno

Information extraction for Rust: NER, coreference resolution, evaluation.

[![CI](https://github.com/arclabs561/anno/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/anno/actions)
[![Crates.io](https://img.shields.io/crates/v/anno.svg)](https://crates.io/crates/anno)
[![Docs](https://docs.rs/anno/badge.svg)](https://docs.rs/anno)

## What

Unified API for named entity recognition. Swap between regex (~400ns), transformers (~50-150ms), zero-shot NER without code changes.

## Workspace

```
anno-core/      # Types: Entity, GroundedDocument, GraphDocument
anno/           # NER backends, evaluation
anno-coalesce/  # Cross-document entity resolution
anno-strata/    # Hierarchical clustering
anno-cli/       # CLI binary
```

Tagline: **Extract. Coalesce. Stratify.**

## Usage

```toml
[dependencies]
anno = "0.2"
```

## Examples

### Basic extraction

```rust
use anno::{Model, RegexNER};

let ner = RegexNER::new();
let entities = ner.extract_entities("Contact alice@acme.com by Jan 15", None)?;

for e in &entities {
    println!("{}: \"{}\" [{}, {})", e.entity_type.as_label(), e.text, e.start, e.end);
}
// EMAIL: "alice@acme.com" [8, 22)
// DATE: "Jan 15" [26, 32)
```

### Named entities

```rust
use anno::StackedNER;

let ner = StackedNER::default();
let entities = ner.extract_entities("Sarah Chen joined Microsoft in Seattle", None)?;
// PER: "Sarah Chen" [0, 10)
// ORG: "Microsoft" [18, 27)
// LOC: "Seattle" [31, 38)
```

### ML backends

```rust
#[cfg(feature = "onnx")]
use anno::GLiNEROnnx;

#[cfg(feature = "onnx")]
let ner = GLiNEROnnx::new("onnx-community/gliner_small-v2.1")?;
#[cfg(feature = "onnx")]
let entities = ner.extract_entities("Marie Curie discovered radium", None)?;
```

### Zero-shot NER

```rust
#[cfg(feature = "onnx")]
let entities = ner.extract(
    "Patient presents with diabetes, prescribed metformin 500mg",
    &["disease", "medication", "dosage"],
    0.5,
)?;
```

## Backends

| Backend | Latency | Accuracy | Feature | Use Case |
|---------|---------|----------|---------|----------|
| `RegexNER` | ~400ns | ~95%* | always | Structured entities (dates, money, emails) |
| `HeuristicNER` | ~50μs | ~65% | always | Person/Org/Location heuristics |
| `StackedNER` | ~100μs | varies | always | Composable layered extraction |
| `BertNEROnnx` | ~50ms | ~86% | `onnx` | Fixed 4-type NER (PER/ORG/LOC/MISC) |
| `GLiNEROnnx` | ~100ms | ~92% | `onnx` | Zero-shot NER (custom types) |
| `GLiNER2` | ~130ms | ~92% | `onnx`/`candle` | Multi-task (NER + classification) |

*Pattern accuracy on structured entities only.

## Features

| Feature | What it enables |
|---------|-----------------|
| *(default)* | `RegexNER`, `HeuristicNER`, `StackedNER`, `GraphDocument` |
| `onnx` | BERT, GLiNER, GLiNER2, NuNER, W2NER via ONNX Runtime |
| `candle` | Pure Rust inference (`CandleNER`, `GLiNERCandle`) |
| `eval` | Core metrics (P/R/F1), datasets, evaluation framework |
| `eval-advanced` | Calibration, robustness, OOD detection |
| `discourse` | Event extraction, abstract anaphora |

## Documentation

- **API docs**: https://docs.rs/anno
- **Architecture**: [docs/TOOLBOX_ARCHITECTURE.md](docs/TOOLBOX_ARCHITECTURE.md)
- **Evaluation**: [docs/EVALUATION.md](docs/EVALUATION.md)

## License

MIT OR Apache-2.0
