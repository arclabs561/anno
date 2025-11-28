# anno

NER for Rust.

Named entity recognition with multiple backends. Also: coreference metrics, evaluation.

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

### NER Backends

| Feature | Description | Dependencies | Compile Time |
|---------|-------------|--------------|--------------|
| *(default)* | `PatternNER` + `StatisticalNER` + `StackedNER` | Zero | ~5s |
| `onnx` | BERT and GLiNER models via ONNX | `ort`, `tokenizers` | +40s |
| `candle` | Pure Rust ML (Metal/CUDA) | `candle-*` | +60s |
| `network` | Dataset downloading and caching | `ureq`, `dirs` | +3s |

### Evaluation Framework (tiered, opt-in)

| Feature | Description | Includes |
|---------|-------------|----------|
| `eval` | Core P/R/F1, coreference metrics, BIO adapter | — |
| `eval-bias` | Gender, demographic, temporal bias | + eval |
| `eval-advanced` | Calibration, robustness, active learning | + eval |
| `eval-full` | All evaluation modules | eval + eval-bias + eval-advanced |

### Combined

| Feature | Description |
|---------|-------------|
| `full` | All features (eval-full + onnx + candle + network) |

```bash
# Minimal (PatternNER + StackedNER only, ~5s compile)
cargo add anno

# With evaluation framework
cargo add anno --features eval

# With ML backends
cargo add anno --features onnx

# With bias analysis
cargo add anno --features eval-bias

# Everything
cargo add anno --features full
```

### Feature Matrix

| Use Case | Features | Compile Time |
|----------|----------|--------------|
| Pattern extraction only | `default` | ~5s |
| + NER evaluation | `eval` | ~8s |
| + Bias analysis | `eval-bias` | ~10s |
| + ML backend (BERT) | `eval,onnx` | ~50s |
| Full research setup | `full` | ~90s |

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
use anno::eval::modes::{EvalMode, MultiModeResults, EvalConfig, evaluate_with_config};

let results = MultiModeResults::compute(&predicted, &gold);
println!("Strict F1: {:.1}%", results.strict.f1 * 100.0);
println!("Partial F1: {:.1}%", results.partial.f1 * 100.0);

// Partial mode with minimum overlap threshold (e.g., 50%)
let config = EvalConfig::new().with_min_overlap(0.5);
let strict_partial = evaluate_with_config(&predicted, &gold, EvalMode::Partial, &config);
```

### BIO Tag Adapter

Convert between BIO-tagged sequences and entity spans:

```rust
use anno::eval::bio_adapter::{bio_to_entities, validate_bio_sequence, BioScheme};

let tokens = ["John", "Smith", "works", "at", "Apple"];
let tags = ["B-PER", "I-PER", "O", "O", "B-ORG"];

// Convert to entities
let entities = bio_to_entities(&tokens, &tags, BioScheme::IOB2)?;
assert_eq!(entities[0].text, "John Smith");

// Validate sequence
let errors = validate_bio_sequence(&tags, BioScheme::IOB2);
assert!(errors.is_empty()); // Valid sequence
```

### CLI Tool

Quick evaluation from command line (with `--features eval`):

```bash
# Build CLI
cargo install --path . --features eval

# Quick evaluation
anno-eval quick

# Validate BIO sequence
anno-eval bio validate "B-PER I-PER O B-ORG"

# Repair invalid sequence
anno-eval bio repair "O I-PER I-PER O"

# Calculate span overlap IoU
anno-eval overlap 0 10 5 15
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

### Comprehensive Evaluation Framework

Beyond basic F1 metrics, `anno` provides specialized evaluation tools (requires feature flags):

```rust
// With eval-bias feature:
use anno::eval::{GenderBiasEvaluator, DemographicBiasEvaluator};

// With eval-advanced feature:
use anno::eval::{CalibrationEvaluator, RobustnessEvaluator, ThresholdAnalyzer};
```

| Module | Feature | Purpose |
|--------|---------|---------|
| **Calibration** | `eval-advanced` | Confidence reliability (ECE, Brier) |
| **Gender Bias** | `eval-bias` | WinoBias-style tests |
| **Demographic Bias** | `eval-bias` | Ethnicity/region fairness |
| **Robustness** | `eval-advanced` | Perturbation tolerance |
| **Error Analysis** | `eval-advanced` | Boundary vs type errors |
| **Threshold Analysis** | `eval-advanced` | Precision-recall curves |

## Examples

Examples are organized by complexity:

```bash
# Getting Started
cargo run --example 01_quickstart            # Basic evaluation
cargo run --example 10_bert_onnx -F onnx     # ML backend

# Evaluation
cargo run --example 20_eval_basic            # P/R/F1 metrics
cargo run --example 30_bias_analysis -F eval-bias  # Bias testing

# Benchmarking
cargo run --example 40_benchmark_backend -F onnx   # Backend comparison
cargo run --example 52_quality_bench -F eval-full  # Comprehensive
```

| Range | Category | Key Examples |
|-------|----------|--------------|
| 01-09 | Getting Started | `01_quickstart` |
| 10-19 | ML Backends | `10_bert_onnx`, `11_candle_gliner` |
| 20-29 | Evaluation | `20_eval_basic`, `21_eval_coref` |
| 30-39 | Bias Analysis | `30_bias_analysis` |
| 40-49 | Benchmarking | `40_benchmark_backend`, `42_benchmark_full` |
| 50-59 | Comprehensive | `52_quality_bench` |

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

With GLiNER, define entity types at runtime using the `extract()` method:

```rust
use anno::GLiNEROnnx;

let ner = GLiNEROnnx::new("onnx-community/gliner_small-v2.1")?;

// Use extract() for custom entity types with confidence threshold
let entities = ner.extract(
    "Tesla announced the Model Y at $45,000",
    &["company", "product", "price"],  // custom labels
    0.5  // confidence threshold
)?;

// Or use the Model trait with default labels
use anno::Model;
let entities = ner.extract_entities("John works at Google", None)?;
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

## Research

Based on: [GLiNER](https://arxiv.org/abs/2311.08526), [UniversalNER](https://arxiv.org/abs/2308.03279), [ModernBERT](https://arxiv.org/abs/2412.13663), [W2NER](https://arxiv.org/abs/2112.10070).

See `docs/SCOPE.md` for roadmap.

## Related

- [`rank-fusion`](https://crates.io/crates/rank-fusion) — Combine ranked lists from multiple retrievers
- [`rank-refine`](https://crates.io/crates/rank-refine) — Reranking algorithms (ColBERT, MRL, cross-encoder)

## License

MIT OR Apache-2.0
