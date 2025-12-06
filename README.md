<div align="center">

![anno logo](assets/logo.png)

# anno

Information extraction for Rust: NER and cross-document entity resolution.

[![CI](https://github.com/arclabs561/anno/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/anno/actions)
[![Crates.io](https://img.shields.io/crates/v/anno.svg)](https://crates.io/crates/anno)
[![Docs](https://docs.rs/anno/badge.svg)](https://docs.rs/anno)

</div>

Rust library and CLI for named entity recognition and cross-document entity resolution. Multiple backends: regex (~400ns), transformers (~50-150ms), zero-shot NER.

## Installation

```bash
cargo install anno-cli
# or from source:
git clone https://github.com/arclabs561/anno
cd anno && cargo build --release
```

## Usage

### Extract entities

```bash
$ anno extract --model heuristic "Marie Curie won the Nobel Prize in Paris"

  PER (2):
    [  0, 11) ########..  75% "Marie Curie"
    [ 20, 31) ######....  60% "Nobel Prize"
  LOC (1):
    [ 35, 40) ########..  80% "Paris"
```

JSON output:
```bash
$ anno extract --format json --model heuristic "Marie Curie won the Nobel Prize in Paris"
[{"text":"Marie Curie","type":"PER","start":0,"end":11,"confidence":0.75},...]
```

### Cross-document entity resolution

```bash
# Process directory of text files
$ anno crossdoc --directory ./docs --model heuristic --threshold 0.6 --format tree

# Or import pre-processed JSON files
$ anno extract --file doc1.txt --export doc1.json
$ anno extract --file doc2.txt --export doc2.json
$ anno crossdoc --import doc1.json --import doc2.json --threshold 0.6 --format tree
```

## Common Use Cases

### Ingest directory and coalesce → see entity clusters

```bash
$ anno crossdoc --directory ./docs --model heuristic --threshold 0.7 --format tree
```

Example output:
```
Identity 1: "Marie Curie" (3 mentions)
  ├─ Document: doc1.txt [0, 11) confidence: 0.85
  ├─ Document: doc2.txt [5, 16) confidence: 0.82
  └─ Document: doc3.txt [12, 23) confidence: 0.79

Identity 2: "Paris" (2 mentions)
  ├─ Document: doc1.txt [35, 40) confidence: 0.90
  └─ Document: doc3.txt [45, 50) confidence: 0.88
```

### Ingest URL and extract entities

```bash
$ anno extract --url https://example.com/article --model heuristic
```

Example output:
```
ok: extracted 5 entities in 12.3ms (model: heuristic, avg confidence: 0.78, tracks: 5, identities: 0)

  PER (2):
    [  0, 11) ########.. "Marie Curie"
    [ 20, 31) ######.... "Nobel Prize"
  LOC (1):
    [ 35, 40) ########.. "Paris"
  ORG (1):
    [ 50, 60) #######... "Acme Corp"
  DATE (1):
    [ 70, 78) ########.. "2024-01-15"

  [PER: Marie Curie] won the [PER: Nobel Prize] in [LOC: Paris]. [ORG: Acme Corp] announced on [DATE: 2024-01-15].
```

### Ingest URL and debug (HTML visualization)

```bash
$ anno debug --url https://example.com/article --html --output debug.html
```

Opens interactive HTML with entity highlighting, coreference chains, and metadata.

### Ingest URL and see entities in terminal (with coreference)

```bash
$ anno debug --url https://example.com/article --coref
```

Shows entities with intra-document coreference resolution (pronouns linked to antecedents).

**Note**: `crossdoc` requires `eval-advanced` feature. Use `--model heuristic` for better entity detection. Import pre-processed JSON files with `--import` for best results.

**More examples**: See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for advanced workflows.

## Library

```toml
[dependencies]
anno = "0.2"
```

## Structure

Workspace crates (top-level directories):

```
anno-core/      # Foundation: Entity, GroundedDocument
anno/           # NER backends, evaluation
coalesce/       # Cross-document entity resolution (anno-coalesce)
strata/         # Hierarchical clustering (anno-strata)
anno-cli/       # CLI binary
```

Each crate is independently usable. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for details.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for full architecture.

## Pipeline

Three-level hierarchy: **Signal → Track → Identity**.[^1]

1. **Extract** (Signal): Detect entities in text
2. **Coalesce** (Identity): Merge mentions across documents into canonical entities

[^1]: See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for architecture details.

### Extract

Named entity recognition: detect persons, organizations, locations, dates, etc.

- **Input**: raw text
- **Output**: entities with spans, types, confidence
- **CLI**: `anno extract --model heuristic "text"`

```rust
use anno::{Model, RegexNER};

let ner = RegexNER::new();
let entities = ner.extract_entities("Contact alice@acme.com by Jan 15", None)?;
```

### Coalesce

Cross-document entity resolution: merge mentions across documents into canonical entities.

- **Input**: entities from multiple documents
- **Output**: identity clusters linking mentions across documents
- **CLI**: `anno crossdoc --directory ./docs --model heuristic --threshold 0.6 --format tree`
- **When to use**: Processing multiple documents, need to know "Barack Obama" in doc1 and doc2 refer to the same person
- **Note**: Requires `eval-advanced` feature. Use `--import` for pre-processed GroundedDocument JSON files.

```rust
use anno_coalesce::Resolver;

let resolver = Resolver::new();
let identities = resolver.resolve_inter_doc_coref(&mut corpus, Some(0.7), Some(true))?;
```

## Library Examples

```rust
use anno::{Model, RegexNER};

let ner = RegexNER::new();
let entities = ner.extract_entities("Contact alice@acme.com by Jan 15", None)?;
```

**Zero-shot NER** (custom entity types):
```rust
#[cfg(feature = "onnx")]
use anno::{ZeroShotNER, GLiNEROnnx};

#[cfg(feature = "onnx")]
let ner = GLiNEROnnx::new("onnx-community/gliner_small-v2.1")?;
#[cfg(feature = "onnx")]
let entities = ner.extract_with_types(
    "Patient presents with diabetes, prescribed metformin",
    &["disease", "medication"],
    0.5,
)?;
```

**See [`docs/SCOPE.md`](docs/SCOPE.md) for complete API documentation.**

## Backends

| Backend | Latency | Accuracy | Feature | Use Case |
|---------|---------|----------|---------|----------|
| `RegexNER` | ~400ns | ~95%¹ | always | Structured entities (dates, money, emails) |
| `HeuristicNER` | ~50μs | ~65%² | always | Person/Org/Location heuristics |
| `GLiNEROnnx` | ~100ms | ~92%³ | `onnx` | Zero-shot NER (custom types) |
| `BertNEROnnx` | ~50ms | ~86%⁴ | `onnx` | Fixed 4-type NER (PER/ORG/LOC/MISC) |

¹ Pattern accuracy on structured entities only. ² F1 on Person/Org/Location. ³ Zero-shot F1 varies by entity types. ⁴ F1 on CoNLL-2003.

**See [`docs/TASK_DATASET_MAPPING.md`](docs/TASK_DATASET_MAPPING.md) for complete backend list and task support.**

## Features

- `onnx`: BERT, GLiNER, GLiNER2 via ONNX Runtime
- `eval`: Evaluation framework, datasets, metrics
- `eval-advanced`: Cross-document resolution, advanced evaluation

**See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for complete feature list.**

## Documentation

- **API docs**: https://docs.rs/anno
- **Architecture**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Evaluation**: [docs/EVALUATION.md](docs/EVALUATION.md)

## License

MIT OR Apache-2.0
