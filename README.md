# anno

Named Entity Recognition for Rust.

[![CI](https://github.com/arclabs561/anno/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/anno/actions)
[![Crates.io](https://img.shields.io/crates/v/anno.svg)](https://crates.io/crates/anno)
[![Docs](https://docs.rs/anno/badge.svg)](https://docs.rs/anno)
[![MSRV](https://img.shields.io/badge/MSRV-1.75-blue)](https://blog.rust-lang.org/2023/12/28/Rust-1.75.0.html)

## The Problem NER Solves

Text is unstructured. You have strings like:

```
"John Smith joined Apple Inc. in San Francisco on January 15, 2024 for $150,000"
```

You need structure:

| Entity | Type | Position |
|--------|------|----------|
| John Smith | PERSON | 0-10 |
| Apple Inc. | ORG | 18-28 |
| San Francisco | LOC | 32-45 |
| January 15, 2024 | DATE | 49-65 |
| $150,000 | MONEY | 70-78 |

Without NER, you're doing regex for every entity type you care about. With NER, you get
structured extraction from arbitrary text.

## Why NER in Rust?

Most NER lives in Python (spaCy, Hugging Face transformers). That works until:

- You need **sub-millisecond latency** in a hot path
- You're building a **Rust service** and don't want Python FFI
- You need **no-std or WASM** deployment
- You want **predictable memory** without GC pauses

anno provides NER with the Rust guarantees you expect.

## Where NER Fits in Pipelines

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Raw Text    │────▶│    NER      │────▶│ Structured  │
│             │     │   (anno)    │     │  Entities   │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   ┌─────────┐       ┌─────────┐       ┌─────────┐
   │ Search  │       │Knowledge│       │   PII   │
   │Filtering│       │  Graph  │       │Detection│
   └─────────┘       └─────────┘       └─────────┘
```

**Search**: Filter results by entity type ("show me documents mentioning Apple")
**Knowledge Graphs**: Extract (subject, predicate, object) triples
**PII Detection**: Find names, dates, SSNs before logging

## Quick Start

```rust
use anno::prelude::*;

let ner = PatternNER::new();
let entities = ner.extract_entities("Meeting on January 15, 2025 for $100", None).unwrap();
// [Entity { text: "January 15, 2025", type: Date }, Entity { text: "$100", type: Money }]
```

## Backends: Accuracy vs Speed Tradeoff

| Backend | Feature | Speed | Accuracy | Entity Types |
|---------|---------|-------|----------|--------------|
| `PatternNER` | — | 1μs | N/A | DATE, MONEY, PERCENT only |
| `BertNEROnnx` | `onnx` | ~10ms | ~74% F1 | PER, ORG, LOC, etc. |
| `GLiNERNER` | `onnx` | ~50ms | varies | Custom (zero-shot) |
| `CandleNER` | `candle` | ~15ms | ~74% F1 | PER, ORG, LOC, etc. |

**Pattern**: Use when you only need structured formats (dates, money, percentages).
Deterministic, fast, zero dependencies.

**BERT ONNX**: Use for general NER. Downloads ~30MB model on first run.
Best balance of speed and accuracy.

**GLiNER**: Use when you need custom entity types at runtime ("extract PRODUCT, COMPETITOR
from this text"). Zero-shot means no training required.

**Candle**: Use when you need pure Rust with no C dependencies. Slower to compile,
but no ONNX runtime needed.

## Hybrid Mode

Combine backends when you need both pattern precision and ML coverage:

```rust
use anno::{HybridNER, HybridConfig, MergeStrategy};

let config = HybridConfig::new()
    .with_pattern(true)      // Always extract DATE/MONEY/PERCENT
    .with_ml(true)           // Also run BERT for PER/ORG/LOC
    .with_merge(MergeStrategy::PreferPattern);  // Pattern wins on overlap

let ner = HybridNER::new(config)?;
```

## Evaluation

anno includes an evaluation framework for measuring precision, recall, and F1:

```rust,ignore
use anno::{Model, EntityType, eval::{GoldEntity, evaluate_ner_model}};

let test_cases = vec![("John works at Google.", vec![
    GoldEntity::new("John", EntityType::Person, 0),
    GoldEntity::new("Google", EntityType::Organization, 15),
])];
let results = evaluate_ner_model(&model, &test_cases).unwrap();
println!("Precision: {:.2}, Recall: {:.2}, F1: {:.2}", 
         results.precision, results.recall, results.f1);
```

## Related

- [`rank-fusion`](https://crates.io/crates/rank-fusion) — Combine ranked lists from multiple retrievers
- [`rank-refine`](https://crates.io/crates/rank-refine) — Reranking algorithms (ColBERT, MRL, cross-encoder)

## License

MIT OR Apache-2.0
