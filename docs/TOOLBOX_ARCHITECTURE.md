# Anno Toolbox Architecture

## Structure

```
anno/
├── anno-core/      # Types: Entity, GroundedDocument, GraphDocument
├── anno/           # NER backends, evaluation
├── coalesce/       # Cross-document entity coalescing
├── strata/         # Hierarchical clustering (Leiden, RAPTOR)
└── anno-cli/       # CLI binary
```

## Dependencies

```
anno-core (no workspace deps)
    ↑
    ├── anno
    ├── coalesce
    └── strata
            ↑
            └── anno-cli
```

## Usage

### Library

```rust
// NER
use anno::{Model, GLiNEROnnx};
let ner = GLiNEROnnx::new("onnx-community/gliner_small-v2.1")?;
let entities = ner.extract_entities(text, None)?;

// Cross-doc coalescing
use anno_coalesce::Resolver;
let resolver = Resolver::new();
let identities = resolver.resolve_inter_doc_coref(&mut corpus, Some(0.7), Some(true))?;

// Hierarchical clustering
use anno_strata::HierarchicalLeiden;
let hierarchy = HierarchicalLeiden::cluster(&graph)?;
```

### CLI

```bash
# Extract
anno extract "Marie Curie won the Nobel Prize"

# Coalesce (cross-doc entity resolution)
anno crossdoc --directory ./docs --threshold 0.6
# or: anno coalesce --directory ./docs --threshold 0.6

# Stratify (hierarchical clustering)
anno strata --input graph.json --method leiden --levels 3
```

## Philosophy

Each crate is independent. Use what you need:

- `anno`: NER only
- `coalesce`: Entity resolution without NER
- `strata`: Clustering without NER

Or use together via `anno-cli` or your own code.
