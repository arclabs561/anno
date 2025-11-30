# anno

Information extraction for Rust: NER, coreference resolution, and evaluation.

[![CI](https://github.com/arclabs561/anno/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/anno/actions)
[![Crates.io](https://img.shields.io/crates/v/anno.svg)](https://crates.io/crates/anno)
[![Docs](https://docs.rs/anno/badge.svg)](https://docs.rs/anno)

Extract entities, resolve coreference, and evaluate models. Supports regex patterns (dates, money, emails), transformer models (BERT, GLiNER), and coreference resolution (rule-based and T5-based).

All backends implement the same `Model` trait. You can swap between a 400ns regex matcher and a 50ms BERT model without changing calling code.

Dual-licensed under MIT or Apache-2.0.

### Documentation

https://docs.rs/anno

### Usage

```
cargo add anno
```

### Example: basic extraction

Extract entity spans from text. Each entity includes the matched text, its type, and character offsets:

```rust
use anno::{Model, PatternNER};

let ner = PatternNER::new();
let entities = ner.extract_entities("Contact alice@acme.com by Jan 15", None)?;

for e in &entities {
    println!("{}: \"{}\" [{}, {})", e.entity_type.as_label(), e.text, e.start, e.end);
}
```

This prints:

```
EMAIL: "alice@acme.com" [8, 22)
DATE: "Jan 15" [26, 32)
```

`PatternNER` detects structured entities via regex: dates, times, money, percentages, emails, URLs, phone numbers. It won't find "John Smith" or "Apple Inc." — those require context, not patterns.

### Example: named entity recognition

For person names, organizations, and locations, use `StackedNER` which combines patterns with heuristics:

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

This requires no model downloads and runs in ~100μs, but accuracy varies by domain. For higher quality, enable the `onnx` feature and use a transformer model:

```rust
use anno::BertNEROnnx;

let ner = BertNEROnnx::new(anno::DEFAULT_BERT_ONNX_MODEL)?;
let entities = ner.extract_entities("Marie Curie discovered radium in 1898", None)?;
```

This downloads a ~400MB model on first run.

### Example: zero-shot NER

Supervised NER models only recognize entity types seen during training. GLiNER uses a bi-encoder architecture that lets you specify entity types at inference time:

```rust
use anno::GLiNEROnnx;

let ner = GLiNEROnnx::new("onnx-community/gliner_small-v2.1")?;

// Extract domain-specific entities without retraining
let entities = ner.extract(
    "Patient presents with diabetes, prescribed metformin 500mg",
    &["disease", "medication", "dosage"],
    0.5,  // confidence threshold
)?;
```

This is slower (~100ms) but handles arbitrary entity schemas.

### Example: multi-task extraction with GLiNER2

GLiNER2 extends GLiNER with multi-task capabilities. Extract entities, classify text, and extract hierarchical structures in a single forward pass:

```rust
use anno::backends::gliner2::{GLiNER2, TaskSchema};

let model = GLiNER2::from_pretrained(anno::DEFAULT_GLINER2_MODEL)?;
// Or use: "fastino/gliner2-base-v1" (default) or "knowledgator/gliner-multitask-large-v0.5"

let schema = TaskSchema::new()
    .with_entities(&["person", "organization", "product"])
    .with_classification("sentiment", &["positive", "negative", "neutral"], false); // false = single-label

let result = model.extract("Apple announced iPhone 15", &schema)?;
// result.entities: [Apple/organization, iPhone 15/product]
// result.classifications["sentiment"].labels: ["positive"]
```

GLiNER2 supports zero-shot NER, text classification, and structured extraction. See the GLiNER2 paper (arxiv:2507.18546) for details.

### Example: Graph RAG integration

Export entities and relations to knowledge graphs for RAG applications:

```rust
use anno::graph::GraphDocument;
use anno::Entity;

// From NER extraction
let entities = ner.extract_entities(text, None)?;
// Relations would come from a relation extraction model
// let relations = relation_model.extract_relations(text, &entities)?;

// Build graph document (deduplicates via coreference if provided)
let graph = GraphDocument::from_extraction(&entities, &[], None);

// Export to Neo4j Cypher
println!("{}", graph.to_cypher());

// Or NetworkX JSON for Python
println!("{}", graph.to_networkx_json());
```

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

| Backend | Use Case | Latency | Accuracy | Feature | Notes |
|---------|----------|---------|----------|---------|-------|
| `PatternNER` | Structured entities (dates, money, emails) | ~400ns | ~95%* | always | |
| `HeuristicNER` | Person/Org/Location via heuristics | ~50μs | ~65% | always | |
| `StackedNER` | Composable layered extraction | ~100μs | varies | always | |
| `BertNEROnnx` | High-quality NER (fixed types) | ~50ms | ~86% | `onnx` | |
| `GLiNEROnnx` | Zero-shot NER (custom types) | ~100ms | ~92% | `onnx` | |
| `NuNER` | Zero-shot NER (token-based) | ~100ms | ~86% | `onnx` | |
| `W2NER` | Nested/discontinuous NER | ~150ms | ~85% | `onnx` | Requires authentication |
| `CandleNER` | Pure Rust BERT NER | varies | ~86% | `candle` | |
| `GLiNERCandle` | Pure Rust zero-shot NER | varies | ~90% | `candle` | Requires PyTorch to safetensors conversion |
| `GLiNER2` | Multi-task (NER + classification) | ~130ms | ~92% | `onnx`/`candle` | |

*Pattern accuracy on structured entities only

Known limitations:

- W2NER: The default model (`ljynlp/w2ner-bert-base`) requires HuggingFace authentication. See `PROBLEMS.md` for alternatives.
- GLiNERCandle: Most GLiNER models only provide PyTorch weights. Automatic conversion requires Python dependencies (`torch`, `safetensors`). Prefer `GLiNEROnnx` for production use.

### Evaluation

This library includes an evaluation framework for measuring precision, recall, and F1 with different matching semantics (strict, partial, type-only). It also implements coreference metrics (MUC, B³, CEAF, LEA) for systems that resolve mentions to entities.

```rust
use anno::{Model, PatternNER};
use anno::eval::report::ReportBuilder;

let model = PatternNER::new();
let report = ReportBuilder::new("PatternNER")
    .with_core_metrics(true)
    .with_error_analysis(true)
    .build(&model);
println!("{}", report.summary());
```

See [docs/EVALUATION.md](docs/EVALUATION.md) for details on evaluation modes, bias analysis, and dataset support.

### Related projects

[rust-bert](https://github.com/guillaume-be/rust-bert) provides full transformer implementations via tch-rs (requires libtorch). It covers many NLP tasks beyond NER.

[gline-rs](https://github.com/fbilhaut/gline-rs) is a focused GLiNER inference engine with excellent documentation. If you only need GLiNER, it may be simpler.

This library provides:

- Unified `Model` trait across regex, heuristics, and ML backends
- Zero-dependency baselines (`PatternNER`, `HeuristicNER`, `StackedNER`) for fast iteration
- Coreference resolution (rule-based and T5-based) with comprehensive metrics
- Evaluation framework with SemEval modes and coreference metrics (MUC, B³, CEAF, LEA, BLANC)
- Multiple ONNX backends (BERT, GLiNER, GLiNER2, NuNER, W2NER) behind one interface
- Pure Rust inference via Candle (optional Metal/CUDA support)

The ONNX backends are integration work — similar inference code to gline-rs and rust-bert's ONNX mode. The evaluation framework and zero-dependency baselines are the parts that don't exist elsewhere.

### Feature flags

| Feature | What it enables |
|---------|-----------------|
| *(default)* | `PatternNER`, `HeuristicNER`, `StackedNER`, `GraphDocument`, `SchemaMapper` |
| `onnx` | BERT, GLiNER, GLiNER2, NuNER, W2NER via ONNX Runtime |
| `candle` | Pure Rust inference (`CandleNER`, `GLiNERCandle`, `GLiNER2Candle`) with optional Metal/CUDA |
| `eval` | Core metrics (P/R/F1), datasets, evaluation framework |
| `eval-bias` | Gender, demographic, temporal, length bias analysis |
| `eval-advanced` | Calibration, robustness, OOD detection, dataset download |
| `discourse` | Event extraction, shell nouns, abstract anaphora |
| `full` | Everything |

### Minimum Rust version policy

This crate's minimum supported rustc version is 1.75.0.

### License

MIT OR Apache-2.0
