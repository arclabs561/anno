# Architecture

## Structure

```
anno/
├── anno-core/      # Foundation: Entity, GroundedDocument, GraphDocument
├── anno/           # NER backends, evaluation framework
├── coalesce/       # Cross-document entity coalescing
├── strata/         # Hierarchical clustering (Leiden, RAPTOR)
└── anno-cli/       # Unified CLI binary
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

Each crate is independent. Use what you need:

- `anno`: NER only
- `anno-coalesce`: Entity resolution without NER
- `anno-strata`: Clustering without NER

Or use together via `anno-cli`.

## Library

### NER

```rust
use anno::{Model, GLiNEROnnx};

let ner = GLiNEROnnx::new("onnx-community/gliner_small-v2.1")?;
let entities = ner.extract_entities(text, None)?;
```

### Cross-document Coalescing

```rust
use anno_coalesce::Resolver;

let resolver = Resolver::new();
let identities = resolver.resolve_inter_doc_coref(&mut corpus, Some(0.7), Some(true))?;
```

### Hierarchical Clustering

```rust
use anno_strata::HierarchicalLeiden;

let hierarchy = HierarchicalLeiden::cluster(&graph)?;
```

## CLI

```bash
# Extract
anno extract "Marie Curie won the Nobel Prize"

# Coalesce (cross-doc entity resolution)
anno crossdoc --directory ./docs --threshold 0.6
# or: anno coalesce --directory ./docs --threshold 0.6

# Stratify (hierarchical clustering)
anno strata --input graph.json --method leiden --levels 3
```

## Pipeline

**Extract. Coalesce. Stratify.**

1. **Extract**: Detect entities in text (NER)
   - Input: raw text
   - Output: entity mentions (Signal → Track within document)

2. **Coalesce**: Cross-document entity resolution
   - Input: entities from multiple documents (Tracks)
   - Output: canonical entities (Identity) linking mentions across documents
   - Purpose: Identity resolution - "Marie Curie" in doc1 and doc2 → same Identity
   - Algorithm: Similarity-based clustering (embeddings or string similarity)
   - Example: `anno crossdoc --directory ./docs --threshold 0.7`

3. **Stratify**: Hierarchical community detection
   - Input: graph of entities and relations (GraphDocument)
   - Output: hierarchical layers of communities at multiple resolutions
   - Purpose: Reveal abstraction levels (specific → themes → domains)
   - Algorithm: Leiden algorithm at multiple resolutions (modularity optimization)
   - Example: `anno strata --input graph.json --method leiden --levels 3`

**Key Difference**: 
- **Coalesce** = identity resolution (same entity, different documents)
- **Strata** = hierarchical organization (communities, themes, abstraction layers)
