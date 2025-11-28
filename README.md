# anno

Named entity recognition for Rust.

[![CI](https://github.com/arclabs561/anno/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/anno/actions)
[![Crates.io](https://img.shields.io/crates/v/anno.svg)](https://crates.io/crates/anno)
[![Docs](https://docs.rs/anno/badge.svg)](https://docs.rs/anno)

Extracts structured information from text: names, companies, places, dates, money, emails.

## The Simplest Thing That Works

```rust
use anno::{PatternNER, Model};

let ner = PatternNER::new();
let entities = ner.extract_entities("Call me at 555-1234 on Jan 15", None)?;

for e in &entities {
    println!("{}: {}", e.entity_type.as_label(), e.text);
}
// PHONE: 555-1234
// DATE: Jan 15
```

This extracts dates, times, money, emails, URLs, and phone numbers. No ML, no model downloads, compiles in 5 seconds.

**But it won't find "John Smith" or "Apple Inc."** — those require context, not patterns.

## If You Need Names and Companies

```rust
use anno::StackedNER;

let ner = StackedNER::default();
let entities = ner.extract_entities("Sarah Chen joined Microsoft", None)?;
// PER: Sarah Chen
// ORG: Microsoft
```

`StackedNER` combines patterns with statistical heuristics. It's fast (~100μs) and dependency-free, but accuracy varies by domain.

**For production accuracy**, add `--features onnx` and use a real model:

```rust
use anno::BertNEROnnx;

let ner = BertNEROnnx::new(anno::DEFAULT_BERT_ONNX_MODEL)?;
let entities = ner.extract_entities("Elon Musk founded SpaceX in 2002", None)?;
```

This downloads a ~400MB model on first run and gets ~86% F1 on standard benchmarks.

## Installation

```bash
cargo add anno                    # Patterns only (~5s compile)
cargo add anno --features onnx    # + BERT/GLiNER models (~50s compile)
cargo add anno --features eval    # + Evaluation framework
```

## Which Backend Do I Use?

| I need to extract... | Use this | Speed |
|---------------------|----------|-------|
| Dates, money, emails, URLs | `PatternNER` | ~400ns |
| Names, companies (good enough) | `StackedNER` | ~100μs |
| Names, companies (production) | `BertNEROnnx` | ~50ms |
| Custom types like "disease" or "gene" | `GLiNEROnnx` | ~100ms |

`GLiNEROnnx` is special — it extracts whatever entity types you specify at runtime:

```rust
use anno::GLiNEROnnx;

let ner = GLiNEROnnx::new("onnx-community/gliner_small-v2.1")?;
let entities = ner.extract(
    "Patient has diabetes, prescribed metformin",
    &["disease", "medication"],  // you define these
    0.5
)?;
```

## Evaluating Accuracy

If you're benchmarking or need metrics:

```bash
cargo add anno --features eval
```

```rust
use anno::eval::{ReportBuilder, TestCase, SimpleGoldEntity};

let report = ReportBuilder::new("MyModel")
    .with_test_data(my_test_cases)
    .build(&model);

println!("{}", report.summary());
// Precision: 85.2%  Recall: 78.1%  F1: 81.5%
```

See [docs/EVALUATION.md](docs/EVALUATION.md) for:
- Evaluation modes (strict, partial, type-only)
- Coreference metrics (MUC, B³, CEAF, LEA)
- Bias analysis (gender, demographic)
- Standard datasets (CoNLL-2003, WikiGold, etc.)

## Feature Flags

| Feature | What it adds | Compile time |
|---------|--------------|--------------|
| *(default)* | `PatternNER`, `StackedNER` | ~5s |
| `onnx` | BERT, GLiNER, NuNER models | +45s |
| `candle` | Pure Rust ML (Metal/CUDA) | +60s |
| `eval` | Metrics, datasets, analysis | +3s |
| `eval-bias` | Gender/demographic bias tests | +2s |
| `eval-advanced` | Calibration, robustness | +2s |
| `full` | Everything | ~90s |

## Performance

Apple M3 Max, single thread:

| Backend | Latency | Throughput |
|---------|---------|------------|
| PatternNER | 0.4μs | 2.5M chars/sec |
| StackedNER | 100μs | 10K chars/sec |
| BertNEROnnx | 50ms | 20 chars/sec |

Pattern extraction is CPU-bound. ML inference is model-bound — batch multiple texts or use GPU for throughput.

## More Documentation

- [API Reference](https://docs.rs/anno) — All types and methods
- [docs/SCOPE.md](docs/SCOPE.md) — Architecture, trait hierarchy, roadmap
- [docs/EVALUATION.md](docs/EVALUATION.md) — Metrics, datasets, bias testing
- [examples/](examples/) — Runnable code for each backend

## License

MIT OR Apache-2.0
