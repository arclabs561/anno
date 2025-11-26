# anno

Named entity recognition.

[![CI](https://github.com/arclabs561/anno/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/anno/actions)
[![Crates.io](https://img.shields.io/crates/v/anno.svg)](https://crates.io/crates/anno)
[![Docs](https://docs.rs/anno/badge.svg)](https://docs.rs/anno)
[![MSRV](https://img.shields.io/badge/MSRV-1.75-blue)](https://blog.rust-lang.org/2023/12/28/Rust-1.75.0.html)

## Why This Library?

NER extracts structured entities (people, dates, organizations) from unstructured text.
You need it for:

- **Query understanding** — "flights to Paris on Friday" → intent + entities
- **Document indexing** — Extract company names, dates, monetary values
- **Content filtering** — Detect PII before logging or storage

This crate provides multiple backends with a unified `Model` trait, so you can
start fast with regex patterns and upgrade to neural models without API changes.

```rust
use anno::prelude::*;

let ner = PatternNER::new();
let entities = ner.extract_entities("Meeting on January 15, 2025 for $100", None).unwrap();
// [Entity { text: "January 15, 2025", type: Date }, Entity { text: "$100", type: Money }]
```

## When to Use Which Backend

| You Need | Use | Tradeoff |
|----------|-----|----------|
| Dates, money, percentages | `PatternNER` | Fast, no deps, limited types |
| General NER (PER/ORG/LOC) | `BertNEROnnx` | ONNX model required, more accurate |
| Custom entity types | `GLiNERNER` | Zero-shot, specify types at runtime |
| No external deps | `CandleNER` | Pure Rust, slower to compile |

## Prelude

Import common types with one line:

```rust
use anno::prelude::*;
// Imports: Entity, EntityType, Error, Result, Model, PatternNER
// Plus feature-gated: BertNEROnnx, GLiNERNER (onnx), CandleNER (candle)
```

## Backends

| Backend | Feature | Entities | Notes |
|---------|---------|----------|-------|
| `PatternNER` | — | DATE, MONEY, PERCENT | Always available |
| `BertNEROnnx` | `onnx` | PER, ORG, LOC, etc | BERT via ONNX |
| `GLiNERNER` | `onnx` | Custom types | Zero-shot |
| `CandleNER` | `candle` | PER, ORG, LOC, etc | Rust-native |

## Evaluation

```rust,ignore
use anno::{Model, EntityType, eval::{GoldEntity, evaluate_ner_model}};

let test_cases = vec![("John works at Google.", vec![
    GoldEntity::new("John", EntityType::Person, 0),
    GoldEntity::new("Google", EntityType::Organization, 15),
])];
let results = evaluate_ner_model(&model, &test_cases).unwrap();
println!("F1: {:.2}", results.f1);
```

## Related

See [`rank-fusion`](https://crates.io/crates/rank-fusion) for combining ranked lists.
See [`rank-refine`](https://crates.io/crates/rank-refine) for reranking.

## License

MIT OR Apache-2.0

