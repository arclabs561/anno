# anno

Information extraction for Rust: NER, coreference resolution, and evaluation.

[![CI](https://github.com/arclabs561/anno/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/anno/actions)
[![Crates.io](https://img.shields.io/crates/v/anno.svg)](https://crates.io/crates/anno)
[![Docs](https://docs.rs/anno/badge.svg)](https://docs.rs/anno)

Unified API for named entity recognition, coreference resolution, and evaluation. Swap between regex patterns (~400ns), transformer models (~50-150ms), and zero-shot NER without changing your code.

**Key features:**
- Zero-dependency baselines (`RegexNER`, `HeuristicNER`) for fast iteration
- ML backends (BERT, GLiNER, GLiNER2, NuNER, W2NER) via ONNX Runtime
- Comprehensive evaluation framework with bias analysis and calibration
- Coreference metrics (MUC, B³, CEAF, LEA, BLANC) and resolution
- Graph export for RAG applications (Neo4j, NetworkX)

Dual-licensed under MIT or Apache-2.0.

### Documentation

- **API docs**: https://docs.rs/anno
- **Research contributions**: See [docs/RESEARCH.md](docs/RESEARCH.md) for what's novel vs. implementation

### Usage

```
cargo add anno
```

### Example: basic extraction

Extract entity spans from text. Each entity includes the matched text, its type, and character offsets:

```rust
use anno::{Model, RegexNER};

let ner = RegexNER::new();
let entities = ner.extract_entities("Contact alice@acme.com by Jan 15", None)?;

for e in &entities {
    println!("{}: \"{}\" [{}, {})", e.entity_type.as_label(), e.text, e.start, e.end);
}
// Output:
// EMAIL: "alice@acme.com" [8, 22)
// DATE: "Jan 15" [26, 32)
```

**Note**: All examples use `?` for error handling. In production, handle `Result` types appropriately.

`RegexNER` detects structured entities via regex: dates, times, money, percentages, emails, URLs, phone numbers. It won't find "John Smith" or "Apple Inc." — those require context, not patterns.

### Example: named entity recognition

For person names, organizations, and locations, use `StackedNER` which combines patterns with heuristics. `StackedNER` is composable — you can add ML backends on top for better accuracy:

```rust
use anno::StackedNER;

let ner = StackedNER::default();
let entities = ner.extract_entities("Sarah Chen joined Microsoft in Seattle", None)?;
```

This prints:

```
PER: "Sarah Chen" [0, 10)
ORG: "Microsoft" [18, 27)
LOC: "Seattle" [31, 38)
```

This requires no model downloads and runs in ~100μs, but accuracy varies by domain. 

**StackedNER is composable**: You can add ML backends on top of the default pattern+heuristic layers for better accuracy while keeping fast structured entity extraction:

```rust
#[cfg(feature = "onnx")]
use anno::{StackedNER, GLiNEROnnx};

// ML-first: GLiNER runs first, then patterns fill gaps
let ner = StackedNER::with_ml_first(
    Box::new(GLiNEROnnx::new("onnx-community/gliner_small-v2.1")?)
);

// Or ML-fallback: patterns/heuristics first, ML as fallback
let ner = StackedNER::with_ml_fallback(
    Box::new(GLiNEROnnx::new("onnx-community/gliner_small-v2.1")?)
);

// Or custom stack with builder
let ner = StackedNER::builder()
    .layer(RegexNER::new())           // High-precision structured entities
    .layer(HeuristicNER::new())       // Quick named entities
    .layer_boxed(Box::new(GLiNEROnnx::new("onnx-community/gliner_small-v2.1")?))  // ML fallback
    .build();
```

For standalone ML backends, enable the `onnx` feature:

```rust
#[cfg(feature = "onnx")]
use anno::BertNEROnnx;

#[cfg(feature = "onnx")]
let ner = BertNEROnnx::new(anno::DEFAULT_BERT_ONNX_MODEL)?;
#[cfg(feature = "onnx")]
let entities = ner.extract_entities("Marie Curie discovered radium in 1898", None)?;
```

**Note**: ML backends (BERT, GLiNER, etc.) download **pre-trained models** from HuggingFace on first run:
- BERT: ~400MB
- GLiNER small: ~150MB
- GLiNER medium: ~400MB  
- GLiNER large: ~1.3GB
- GLiNER2: ~400MB

Models are cached locally after download. All NER models are pre-trained by their original authors; we only run inference. 

**Using your own models**: W2NER and T5Coref support local file paths. Other backends use HuggingFace model IDs (you can upload your own models to HuggingFace). For detailed "bring your own model" instructions, see [`docs/MODEL_DOWNLOADS.md`](docs/MODEL_DOWNLOADS.md).

**Box embedding training**: Training code is in `anno` (`src/backends/box_embeddings_training.rs`). The [matryoshka-box](https://github.com/arclabs561/matryoshka-box) research project extends this with matryoshka-specific features (variable dimensions, etc.). See [`docs/MATRYOSHKA_BOX_INTEGRATION.md`](docs/MATRYOSHKA_BOX_INTEGRATION.md) for details.

To download models ahead of time:

```bash
# Download all models (ONNX + Candle)
cargo run --example download_models --features "onnx,candle"

# Download only ONNX models
cargo run --example download_models --features onnx
```

This pre-warms the cache so models are ready for offline use or faster first runs.

### Example: zero-shot NER

Supervised NER models only recognize entity types seen during training. GLiNER uses a bi-encoder architecture that lets you specify entity types at inference time:

```rust
#[cfg(feature = "onnx")]
use anno::GLiNEROnnx;

#[cfg(feature = "onnx")]
let ner = GLiNEROnnx::new("onnx-community/gliner_small-v2.1")?;

// Extract domain-specific entities without retraining
#[cfg(feature = "onnx")]
let entities = ner.extract(
    "Patient presents with diabetes, prescribed metformin 500mg",
    &["disease", "medication", "dosage"],
    0.5,  // confidence threshold
)?;
```

This is slower (~100ms) but supports arbitrary entity schemas.

### Example: multi-task extraction with GLiNER2

GLiNER2 extends GLiNER with multi-task capabilities. Extract entities, classify text, and extract hierarchical structures in a single forward pass:

```rust
#[cfg(any(feature = "onnx", feature = "candle"))]
use anno::backends::gliner2::{GLiNER2, TaskSchema};

#[cfg(any(feature = "onnx", feature = "candle"))]
let model = GLiNER2::from_pretrained(anno::DEFAULT_GLINER2_MODEL)?;
// DEFAULT_GLINER2_MODEL is "onnx-community/gliner-multitask-large-v0.5"
// Alternative: "fastino/gliner2-base-v1" (if available)

#[cfg(any(feature = "onnx", feature = "candle"))]
let schema = TaskSchema::new()
    .with_entities(&["person", "organization", "product"])
    .with_classification("sentiment", &["positive", "negative", "neutral"], false); // false = single-label

#[cfg(any(feature = "onnx", feature = "candle"))]
let result = model.extract("Apple announced iPhone 15", &schema)?;
// result.entities: [Apple/organization, iPhone 15/product]
// result.classifications["sentiment"].labels: ["positive"]
```

GLiNER2 supports zero-shot NER, text classification, and structured extraction. See the GLiNER2 paper (arxiv:2507.18546) for details.

### Example: Graph RAG integration

Extract entities and relations, then export to knowledge graphs for RAG applications:

```rust
use anno::graph::GraphDocument;
use anno::backends::tplinker::TPLinker;
use anno::backends::inference::RelationExtractor;
use anno::StackedNER;

let text = "Steve Jobs founded Apple in 1976. The company is headquartered in Cupertino.";

// Extract entities
let ner = StackedNER::default();
let entities = ner.extract_entities(text, None)?;

// Extract relations between entities
// Note: TPLinker is currently a placeholder implementation using heuristics.
// For production, consider GLiNER2 which supports relation extraction via ONNX.
let rel_extractor = TPLinker::new()?;
let result = rel_extractor.extract_with_relations(
    text,
    &["person", "organization", "location", "date"],
    &["founded", "headquartered_in", "founded_in"],
    0.5,
)?;

// Convert relations to graph format
use anno::entity::Relation;
let relations: Vec<Relation> = result.relations.iter().map(|r| {
    let head = &result.entities[r.head_idx];
    let tail = &result.entities[r.tail_idx];
    Relation::new(
        head.clone(),
        tail.clone(),
        r.relation_type.clone(),
        r.confidence,
    )
}).collect();

// Build graph document (deduplicates via coreference if provided)
let graph = GraphDocument::from_extraction(&result.entities, &relations, None);

// Export to Neo4j Cypher
println!("{}", graph.to_cypher());
// Output: Creates nodes for entities and edges for relations

// Or NetworkX JSON for Python
println!("{}", graph.to_networkx_json());
```

This creates a knowledge graph with:
- **Nodes**: Entities (Steve Jobs, Apple, Cupertino, 1976)
- **Edges**: Relations (founded, headquartered_in, founded_in)

### Example: grounded entity representation

The `grounded` module provides a hierarchy for entity representation that unifies text NER and visual detection:

```rust
use anno::grounded::{GroundedDocument, Signal, Track, Identity, Location};

// Create a document with the Signal → Track → Identity hierarchy
let mut doc = GroundedDocument::new("doc1", "Marie Curie won the Nobel Prize. She was a physicist.");

// Level 1: Signals (raw detections)
let s1 = doc.add_signal(Signal::new(0, Location::text(0, 12), "Marie Curie", "Person", 0.95));
let s2 = doc.add_signal(Signal::new(1, Location::text(38, 41), "She", "Person", 0.88));

// Level 2: Tracks (within-document coreference)
let mut track = Track::new(0, "Marie Curie");
track.add_signal(s1, 0);
track.add_signal(s2, 1);
let track_id = doc.add_track(track);

// Level 3: Identities (knowledge base linking)
let identity = Identity::from_kb(0, "Marie Curie", "wikidata", "Q7186");
let identity_id = doc.add_identity(identity);
doc.link_track_to_identity(track_id, identity_id);

// Traverse the hierarchy
for signal in doc.signals() {
    if let Some(identity) = doc.identity_for_signal(signal.id) {
        println!("{} → {}", signal.surface, identity.canonical_name);
    }
}
```

The same `Location` type works for text spans, bounding boxes, and other modalities. See `examples/grounded.rs` for a complete walkthrough.

### Backend comparison

| Backend | Use Case | Latency | Accuracy | Feature | When to Use |
|---------|----------|---------|----------|---------|-------------|
| `RegexNER` | Structured entities (dates, money, emails) | ~400ns | ~95%* | always | Fast structured data extraction |
| `HeuristicNER` | Person/Org/Location via heuristics | ~50μs | ~65% | always | Quick baseline, no dependencies |
| `StackedNER` | Composable layered extraction | ~100μs | varies | always | Combine patterns + heuristics + ML backends |
| `BertNEROnnx` | High-quality NER (fixed types) | ~50ms | ~86% | `onnx` | Standard 4-type NER (PER/ORG/LOC/MISC) |
| `GLiNEROnnx` | Zero-shot NER (custom types) | ~100ms | ~92% | `onnx` | **Recommended**: Custom entity types without retraining |
| `NuNER` | Zero-shot NER (token-based) | ~100ms | ~86% | `onnx` | Alternative zero-shot approach |
| `W2NER` | Nested/discontinuous NER | ~150ms | ~85% | `onnx` | Overlapping or non-contiguous entities |
| `CandleNER` | Pure Rust BERT NER | varies | ~86% | `candle` | Rust-native, no ONNX dependency |
| `GLiNERCandle` | Pure Rust zero-shot NER | varies | ~90% | `candle` | Rust-native zero-shot (requires model conversion) |
| `GLiNER2` | Multi-task (NER + classification) | ~130ms | ~92% | `onnx`/`candle` | Joint NER + text classification |

*Pattern accuracy on structured entities only

**Quick selection guide:**
- **Fastest**: `RegexNER` for structured entities, `StackedNER` for general use
- **Best accuracy**: `GLiNEROnnx` for zero-shot, `BertNEROnnx` for fixed types
- **Custom types**: `GLiNEROnnx` (zero-shot, no retraining needed)
- **No dependencies**: `StackedNER` (patterns + heuristics)
- **Hybrid approach**: `StackedNER::with_ml_first()` or `with_ml_fallback()` to combine ML accuracy with pattern speed

Known limitations:

- W2NER: The default model (`ljynlp/w2ner-bert-base`) requires HuggingFace authentication. You may need to authenticate with `huggingface-cli login` or use an alternative model.
- GLiNERCandle: Most GLiNER models only provide PyTorch weights. Automatic conversion requires Python dependencies (`torch`, `safetensors`). Prefer `GLiNEROnnx` for production use.

### Evaluation

This library includes an evaluation framework for measuring precision, recall, and F1 with different matching semantics (strict, partial, type-only). It also implements coreference metrics (MUC, B³, CEAF, LEA) for systems that resolve mentions to entities.

```rust
use anno::{Model, RegexNER};
use anno::eval::report::ReportBuilder;

let model = RegexNER::new();
let report = ReportBuilder::new("RegexNER")
    .with_core_metrics(true)
    .with_error_analysis(true)
    .build(&model);
println!("{}", report.summary());
```

See [docs/EVALUATION.md](docs/EVALUATION.md) for details on evaluation modes, bias analysis, and dataset support.

### Related projects

- **[rust-bert](https://github.com/guillaume-be/rust-bert)**: Full transformer implementations via tch-rs (requires libtorch). Covers many NLP tasks beyond NER.
- **[gline-rs](https://github.com/fbilhaut/gline-rs)**: Focused GLiNER inference engine. Use if you only need GLiNER.

**What makes `anno` different:**
- **Unified API**: Swap between regex (~400ns) and ML models (~50-150ms) without code changes
- **Zero-dependency defaults**: `RegexNER` and `StackedNER` work out of the box
- **Evaluation framework**: Comprehensive metrics, bias analysis, and calibration (unique in Rust NER)
- **Coreference support**: Metrics (MUC, B³, CEAF, LEA, BLANC) and resolution
- **Graph export**: Built-in Neo4j/NetworkX export for RAG applications

### Feature flags

| Feature | What it enables |
|---------|-----------------|
| *(default)* | `RegexNER`, `HeuristicNER`, `StackedNER`, `GraphDocument`, `SchemaMapper` |
| `onnx` | BERT, GLiNER, GLiNER2, NuNER, W2NER via ONNX Runtime |
| `candle` | Pure Rust inference (`CandleNER`, `GLiNERCandle`, `GLiNER2Candle`) with optional Metal/CUDA |
| `eval` | Core metrics (P/R/F1), datasets, evaluation framework |
| `eval-bias` | Gender, demographic, temporal, length bias analysis |
| `eval-advanced` | Calibration, robustness, OOD detection, dataset download |
| `discourse` | Event extraction, shell nouns, abstract anaphora |
| `full` | Everything |

### Static Analysis

This project uses comprehensive static analysis tools:

- **cargo-deny** - Dependency security and license checking
- **cargo-machete** - Fast unused dependency detection
- **cargo-geiger** - Unsafe code statistics
- **OpenGrep** - Security pattern detection with custom rules
- **Miri** - Undefined behavior detection
- **cargo-nextest** - Faster test runner
- **cargo-llvm-cov** - Code coverage

Quick commands:
```bash
just static-analysis      # Run all tools
just safety-report-full   # Comprehensive safety report
just validate-setup       # Check tool installation
```

See [docs/STATIC_ANALYSIS_SETUP.md](docs/STATIC_ANALYSIS_SETUP.md) for details.

### Minimum Rust version policy

This crate's minimum supported rustc version is 1.75.0.

### License

MIT OR Apache-2.0
