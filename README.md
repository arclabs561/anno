# anno

Text annotation and knowledge extraction for Rust.

**Primary**: Named Entity Recognition (NER) with multiple backends  
**Also**: Coreference metrics, relation extraction traits, evaluation framework

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

## Quick Start

```rust
use anno::prelude::*;

// Zero-dependency default (Pattern + Statistical heuristics)
let ner = StackedNER::default();
let entities = ner.extract_entities(
    "Dr. Smith charges $100/hr. Email: smith@test.com", 
    None
).unwrap();

for e in entities {
    println!("{}: {} ({:.0}%)", e.entity_type.as_label(), e.text, e.confidence * 100.0);
}
// PER: Dr. Smith (70%)
// MONEY: $100 (95%)
// EMAIL: smith@test.com (99%)
```

## Feature Flags

anno uses feature flags to minimize compile times and dependencies:

| Feature | Description | Dependencies | Compile Time |
|---------|-------------|--------------|--------------|
| *(none)* | `PatternNER` + `StatisticalNER` + `StackedNER` | Zero | ~5s |
| `network` | Dataset downloading and caching | `ureq`, `dirs` | +3s |
| `onnx` | BERT and GLiNER models via ONNX | `ort`, `tokenizers` | +40s |
| `candle` | Pure Rust ML (Metal/CUDA) | `candle-*` | +60s |
| `full` | All features | All | ~90s |

```bash
# Zero dependencies (default)
cargo add anno

# With ML backends
cargo add anno --features onnx

# With dataset downloading
cargo add anno --features network

# Everything
cargo add anno --features full
```

## Backends

| Backend | Feature | Speed | F1 | Entity Types | Best For |
|---------|---------|-------|-----|--------------|----------|
| `PatternNER` | — | ~400ns | N/A* | DATE, MONEY, EMAIL, etc. | Structured entities |
| `StatisticalNER` | — | ~50μs | ~30% | PER, ORG, LOC | Zero-dep heuristics |
| `StackedNER` | — | ~100μs | ~50% | All above | **Recommended default** |
| `BertNEROnnx` | `onnx` | ~50ms | ~90% | PER, ORG, LOC, MISC | Standard NER |
| `GLiNEROnnx` | `onnx` | ~100ms | ~85% | **Custom** | Zero-shot NER |
| `NuNER` | `onnx` | ~80ms | ~86% | **Custom** | Token-level zero-shot |
| `W2NER` | `onnx` | ~120ms | ~82% | **Custom** | Nested/discontinuous |
| `CandleNER` | `candle` | ~80ms | ~88% | PER, ORG, LOC | Pure Rust / Metal |
| `GLiNERCandle` | `candle` | ~100ms | ~84% | **Custom** | Pure Rust zero-shot |

*PatternNER has ~99% precision on structured entities but doesn't detect named entities.

### Choosing a Backend

```
Do you need Person/Org/Location?
├─ No  → PatternNER (dates, money, emails only)
├─ Yes, and...
│  ├─ Zero dependencies OK → StackedNER (default)
│  ├─ Need high accuracy → BertNEROnnx (onnx feature)
│  ├─ Need custom entity types → GLiNEROnnx or NuNER (onnx feature)
│  ├─ Need nested/discontinuous entities → W2NER (onnx feature)
│  └─ Need pure Rust / Metal → CandleNER or GLiNERCandle (candle feature)
```

### Automatic Backend Selection

Use `anno::auto()` to get the best available backend:

```rust
use anno::{auto, Model};

// Automatically selects the highest-quality available backend
let model = auto()?;
let entities = model.extract_entities("John works at Apple", None)?;
```

## Evaluation

anno provides comprehensive evaluation following NER community standards:

### F1 Score Variants

| Metric | Meaning | Use Case |
|--------|---------|----------|
| **Micro F1** | Aggregate TP/FP/FN across types | Overall performance |
| **Macro F1** | Average F1 across types | Fair to rare types |
| **Weighted F1** | F1 weighted by support | Realistic expectation |

### Evaluation Modes (SemEval-2013)

| Mode | Boundary | Type | Use Case |
|------|----------|------|----------|
| **Strict** | Exact | Exact | CoNLL standard (default) |
| **Exact** | Exact | Any | Boundary detection |
| **Partial** | Overlap | Exact | Lenient evaluation |
| **Type** | Any | Exact | Type classification |

```rust
use anno::eval::modes::{EvalMode, MultiModeResults};

let results = MultiModeResults::compute(&predicted, &gold);
println!("Strict F1: {:.1}%", results.strict.f1 * 100.0);
println!("Partial F1: {:.1}%", results.partial.f1 * 100.0);
```

### Coreference Metrics

For coreference resolution tasks:

| Metric | Focus | Notes |
|--------|-------|-------|
| **MUC** | Links | Ignores singletons |
| **B³** | Mentions | Per-mention scores |
| **CEAF** | Entities | Optimal alignment |
| **LEA** | Links+Entities | Entity-aware |
| **BLANC** | Rand index | Best discriminative power |
| **CoNLL F1** | Composite | Average of MUC, B³, CEAF-e |

## Dataset Support

With `--features network`, download standard NER datasets:

```rust
use anno::eval::loader::{DatasetLoader, DatasetId};

let loader = DatasetLoader::new()?;
let dataset = loader.load_or_download(DatasetId::WikiGold)?;
println!("Loaded {} examples", dataset.sentences.len());
```

| Dataset | Domain | Size | Entity Types |
|---------|--------|------|--------------|
| WikiGold | Wikipedia | ~3.5k | PER, LOC, ORG, MISC |
| WNUT-17 | Social Media | ~2k | Emerging entities |
| CoNLL-2003 | News | ~20k | PER, LOC, ORG, MISC |
| MIT Movie | Movies | ~10k | Actor, Director, Genre |
| MIT Restaurant | Food | ~8k | Cuisine, Location, Price |
| BC5CDR | Biomedical | ~28k | Disease, Chemical |
| FewNERD | Cross-domain | ~188k | 8 coarse + 66 fine types |
| CrossNER | Multi-domain | ~5k | Domain-specific types |

### TypeMapper for Domain-Specific Datasets

Domain datasets use different entity schemas. TypeMapper normalizes them:

```rust
use anno::{TypeMapper, eval::{evaluate_ner_model_with_mapper, loader::DatasetId}};

// MIT Movie: "ACTOR" → Person, "DIRECTOR" → Person
let mapper = DatasetId::MitMovie.type_mapper();

// Or use the mapper directly
let mut mapper = TypeMapper::new();
mapper.add("ACTOR", anno::EntityType::Person);
mapper.add("TITLE", anno::EntityType::custom("WORK_OF_ART", anno::EntityCategory::Creative));
```

## Advanced Usage

### Streaming (Large Documents)

For documents too large to fit in memory:

```rust
use anno::{Model, StackedNER};

let ner = StackedNER::default();

// Process in chunks
for chunk in document.chunks(10_000) {
    let entities = ner.extract_entities(chunk, None)?;
    // Process entities...
}
```

### Custom Entity Types (Zero-Shot)

With GLiNER, define entity types at runtime:

```rust
use anno::GLiNEROnnx;

let ner = GLiNEROnnx::new("path/to/model")?;
let entities = ner.extract_entities(
    "Tesla announced the Model Y at $45,000",
    Some(&["COMPANY", "PRODUCT", "PRICE"].map(String::from).to_vec())
)?;
```

### Provenance Tracking

Every entity includes extraction metadata:

```rust
for entity in entities {
    if let Some(prov) = &entity.provenance {
        println!("Source: {} (method: {:?})", prov.source, prov.method);
    }
}
```

## Performance

Benchmarks on Apple M3 Max (single thread):

| Backend | 100 chars | 1K chars | 10K chars |
|---------|-----------|----------|-----------|
| PatternNER | 0.4μs | 2μs | 15μs |
| StackedNER | 50μs | 200μs | 1.5ms |
| BertNEROnnx | 50ms | 55ms | 80ms |

Memory usage is proportional to text length (~1KB overhead per backend).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Run tests:
```bash
cargo test                      # Zero-dep tests
cargo test --features network   # Include download tests
cargo test --features full      # All tests
```

## Research Alignment

anno's architecture draws from recent NER research:

| Paper | Concept | Implementation |
|-------|---------|----------------|
| [GLiNER](https://arxiv.org/abs/2311.08526) (NAACL 2024) | Bi-encoder span-label matching | `BiEncoder`, `ZeroShotNER` traits |
| [UniversalNER](https://arxiv.org/abs/2308.03279) (ICLR 2024) | Cross-domain distillation | 43 datasets, `TypeMapper` |
| [ModernBERT](https://arxiv.org/abs/2412.13663) (2024) | Efficient encoder | `TextEncoder`, `RaggedBatch` |
| [ReasoningNER](https://arxiv.org/abs/2511.11978) (2025) | Chain-of-thought NER | Future: reasoning traits |
| [GEMNET](https://aclanthology.org/2021.naacl-main.118/) (NAACL 2021) | Gated gazetteers | `Lexicon` trait, `ExtractionMethod` |

See `docs/SCOPE.md` for the full roadmap and `src/eval/TAXONOMY.md` for evaluation methodology.

## Related

- [`rank-fusion`](https://crates.io/crates/rank-fusion) — Combine ranked lists from multiple retrievers
- [`rank-refine`](https://crates.io/crates/rank-refine) — Reranking algorithms (ColBERT, MRL, cross-encoder)

## License

MIT OR Apache-2.0
