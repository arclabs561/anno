# Anno Toolbox Architecture

## How the Crates Tie Together

The anno workspace is organized as a **modular toolbox** where each crate has a clear purpose, but they work together seamlessly.

### The Toolbox Structure

```
anno/                          # Workspace root
├── anno-core/                 # Foundation: Shared types
│   └── Entity, Span, GroundedDocument, Track, Identity, GraphDocument
│
├── anno/                      # Main NER library
│   └── NER backends, evaluation framework, document processing
│
├── coalesce/                   # Cross-document entity coalescing
│   └── Merge mentions into canonical entities across documents
│
├── strata/                     # Hierarchical clustering
│   └── Reveal strata of abstraction (Leiden, RAPTOR, etc.)
│
└── anno-cli/                  # Unified CLI (ties everything together)
    └── All 17 commands that orchestrate the other crates
```

### Dependency Graph

```
anno-core (no dependencies on other workspace crates)
    ↑
    ├── anno (depends on anno-core)
    ├── coalesce (depends on anno-core)
    └── strata (depends on anno-core)
            ↑
            └── anno-cli (depends on anno, coalesce, strata, anno-core)
```

### How Users Interact with the Toolbox

#### 1. **As a Library User** (Programmatic API)

```rust
// Use just NER (anno crate)
use anno::{Model, GLiNEROnnx};

let model = GLiNEROnnx::new("onnx-community/gliner_small-v2.1")?;
let entities = model.extract_entities("Steve Jobs founded Apple", None)?;

// Use cross-doc entity coalescing (coalesce crate)
use anno_coalesce::Resolver;
use anno_core::{Corpus, GroundedDocument};

let mut corpus = Corpus::new();
corpus.add_document(doc1);
corpus.add_document(doc2);

let resolver = Resolver::new();
let identities = resolver.resolve_inter_doc_coref(&mut corpus, Some(0.7), Some(true))?;

// Use hierarchical clustering (strata crate)
use anno_strata::HierarchicalLeiden;
use anno_core::GraphDocument;

let graph = GraphDocument::from_extraction(&entities, &relations, None);
let hierarchy = HierarchicalLeiden::cluster(&graph)?;
```

#### 2. **As a CLI User** (Unified Tool)

The `anno-cli` crate provides a single `anno` binary that orchestrates all crates:

```bash
# NER extraction (uses anno crate)
anno extract "Marie Curie won the Nobel Prize"

# Cross-doc entity coalescing (uses coalesce crate)
anno coalesce --directory ./docs --threshold 0.7

# Hierarchical clustering (uses strata crate)
anno extract --export-graph neo4j | anno strata --hierarchical

# Full pipeline (uses all crates)
anno pipeline --input ./docs --output ./graph.cypher
```

### The "Toolbox" Philosophy

Each crate can be used **independently** or **together**:

1. **Standalone Usage:**
   - `anno`: Just NER, no cross-doc resolution needed
   - `coalesce`: Entity coalescing without NER (if you have pre-extracted entities)
   - `strata`: Hierarchical clustering without NER (if you have a graph)

2. **Integrated Usage:**
   - `anno-cli`: Uses all crates together for end-to-end workflows
   - Your own code: Import multiple crates and compose them

### Example: Full Workflow

```rust
use anno::{Model, GLiNEROnnx};
use anno_core::{GroundedDocument, Corpus};
use anno_coalesce::Resolver;
use anno_strata::HierarchicalLeiden;
use anno_core::GraphDocument;

// 1. Extract entities (anno crate)
let model = GLiNEROnnx::new("onnx-community/gliner_small-v2.1")?;
let entities1 = model.extract_entities(&text1, None)?;
let entities2 = model.extract_entities(&text2, None)?;

// 2. Build grounded documents (anno-core types)
let mut doc1 = GroundedDocument::new("doc1", &text1);
let mut doc2 = GroundedDocument::new("doc2", &text2);
// ... add signals, tracks ...

// 3. Cross-doc entity resolution (coalesce crate)
let mut corpus = Corpus::new();
corpus.add_document(doc1);
corpus.add_document(doc2);

let resolver = Resolver::new();
let identities = resolver.resolve_inter_doc_coref(&mut corpus, Some(0.7), Some(true))?;

// 4. Build graph (anno-core)
let graph = GraphDocument::from_grounded_document(corpus.get_document("doc1")?);

// 5. Hierarchical clustering (strata crate)
let hierarchy = HierarchicalLeiden::cluster(&graph)?;

// 6. Export
println!("{}", hierarchy.to_cypher());
```

### Benefits of This Architecture

1. **Modularity**: Use only what you need
   - NER only? Just `anno`
   - Entity resolution? Just `anno-coalesce`
   - Hierarchical clustering? Just `anno-strata`

2. **Independent Evolution**: Each crate can evolve at its own pace
   - `coalesce` can add new KB linking without affecting `anno`
   - `strata` can experiment with algorithms without breaking `anno-cli`

3. **Clear Boundaries**: Each crate has a single, well-defined purpose
   - `anno-core`: Types only (no algorithms)
   - `anno`: NER algorithms
   - `coalesce`: Cross-doc entity coalescing algorithms
   - `strata`: Hierarchical clustering algorithms
   - `anno-cli`: User interface

4. **Reusability**: Other projects can use individual crates
   - A graph database tool can use `strata` without pulling in NER
   - An entity linking service can use `coalesce` without ML backends

### Workspace Benefits

Even though they're separate crates, being in a workspace provides:

1. **Coordinated Development**: All crates evolve together
2. **Shared Dependencies**: Common versions via `workspace.dependencies`
3. **Unified Testing**: `cargo test --workspace` tests everything
4. **Single Repository**: Easier to maintain and contribute

### Migration Path

For existing `anno` users, the migration is transparent:

```rust
// Old (still works via re-exports)
use anno::{Entity, GroundedDocument, Corpus};

// New (explicit, but same types)
use anno_core::{Entity, GroundedDocument, Corpus};
use anno::Model;
```

The `anno` crate re-exports types from `anno-core` for backward compatibility.

