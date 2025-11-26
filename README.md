# anno

Named entity recognition.

[![CI](https://github.com/arclabs561/anno/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/anno/actions)
[![Crates.io](https://img.shields.io/crates/v/anno.svg)](https://crates.io/crates/anno)
[![Docs](https://docs.rs/anno/badge.svg)](https://docs.rs/anno)
[![MSRV](https://img.shields.io/badge/MSRV-1.75-blue)](https://blog.rust-lang.org/2023/12/28/Rust-1.75.0.html)

```rust
use anno::{PatternNER, Model};

let ner = PatternNER::new();
let entities = ner.extract_entities("Meeting on January 15, 2025 for $100", None)?;
// [Entity { text: "January 15, 2025", type: Date }, Entity { text: "$100", type: Money }]
```

## Backends

| Backend | Feature | Entities | Notes |
|---------|---------|----------|-------|
| `PatternNER` | — | DATE, MONEY, PERCENT | Always available |
| `BertNEROnnx` | `onnx` | PER, ORG, LOC, etc | BERT via ONNX |
| `GLiNERNER` | `onnx` | Custom types | Zero-shot |
| `CandleNER` | `candle` | PER, ORG, LOC, etc | Rust-native |

## Evaluation

```rust
use anno::eval::{GoldEntity, evaluate_ner_model};

let test_cases = vec![("John works at Google.", vec![
    GoldEntity::new("John", EntityType::Person, 0),
    GoldEntity::new("Google", EntityType::Organization, 15),
])];
let results = evaluate_ner_model(&model, &test_cases)?;
println!("F1: {:.2}", results.f1);
```

## Related

See [`rank-fusion`](https://crates.io/crates/rank-fusion) for combining ranked lists.
See [`rank-refine`](https://crates.io/crates/rank-refine) for reranking.

## License

MIT OR Apache-2.0

