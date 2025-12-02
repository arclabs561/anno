# Box Embeddings Implementation Summary

## Overview

Complete implementation of box embeddings for coreference resolution, encoding logical invariants as geometric properties. This addresses limitations of vector-based approaches by providing explicit constraints for transitivity, syntactic rules, temporal evolution, uncertainty, and noise.

**Note**: This implementation is related to the **matryoshka-box** research project (not yet published), which explores combining matryoshka embeddings (variable dimensions) with box embeddings (hierarchical reasoning with uncertainty).

## Implementation Status: ✅ Complete

### Core Infrastructure

- ✅ **BoxEmbedding** (`src/backends/box_embeddings.rs`)
  - Axis-aligned hyperrectangles with min/max bounds
  - Operations: `volume()`, `intersection_volume()`, `conditional_probability()`, `coreference_score()`
  - Helper methods: `from_vector()`, `center()`, `size()`
  - 19 comprehensive tests passing

- ✅ **BoxCorefResolver** (`src/eval/coref_resolver.rs`)
  - Implements `CoreferenceResolver` trait
  - Syntactic constraint enforcement (Principle B/C)
  - Union-find clustering with box-based scoring

### Advanced Features

- ✅ **Temporal Boxes** (BoxTE-style)
  - `TemporalBox` and `BoxVelocity` types
  - `at_time()` for time-slice operations
  - Prevents false coreference across time boundaries

- ✅ **Uncertainty-Aware Boxes** (UKGE-style)
  - `UncertainBox` with confidence derived from volume
  - `Conflict` detection for contradictory claims
  - Source trust modeling

- ✅ **Gumbel Boxes** (Noise Robustness)
  - `GumbelBox` with soft, probabilistic boundaries
  - `membership_probability()` for fuzzy membership
  - `robust_coreference()` with grid sampling

- ✅ **Interaction Modeling**
  - `interaction_strength()` for actor-action-target triples
  - `acquisition_roles()` for asymmetric relations
  - Triple intersection for event modeling

### Integration

- ✅ **Identity Type** (`src/grounded.rs`)
  - `Identity.box_embedding` field added
  - Serialization support (Serialize/Deserialize)

- ✅ **Utility Functions** (`src/eval/coref_resolver.rs`)
  - `vectors_to_boxes()` - Convert vector embeddings to boxes
  - `resolve_with_box_embeddings()` - Convenience wrapper

### Documentation

- ✅ **Design Document** (`docs/BOX_EMBEDDINGS_COREFERENCE.md`)
  - Logical invariants → box geometry mapping
  - Temporal, interaction, misinformation, noise handling
  - Implementation phases and research references

- ✅ **Usage Guide** (`docs/BOX_EMBEDDINGS_USAGE.md`)
  - Quick start examples
  - Integration patterns
  - Performance considerations

- ✅ **Example** (`examples/box_coreference.rs`)
  - Complete workflow demonstration
  - Temporal, uncertainty, Gumbel examples
  - Runs successfully

## Key Features

### 1. Logical Invariants as Geometry

| Invariant | Box Encoding | Implementation |
|-----------|--------------|----------------|
| **Transitivity** | Box containment is transitive | `coreference_score()` uses conditional probability |
| **Syntactic Constraints** | Disjoint boxes for Principle B/C | `check_syntactic_constraints()` |
| **Temporal Evolution** | Time-sliced boxes | `TemporalBox::at_time()` |
| **Uncertainty** | Volume = confidence | `UncertainBox::confidence()` |
| **Noise Robustness** | Soft boundaries | `GumbelBox::membership_probability()` |

### 2. Research Alignment

- **Vilnis et al. (2018)**: Box lattice measures for knowledge graphs
- **BERE (2022)**: Conditional probability for event relations
- **BoxTE (2022)**: Temporal box evolution
- **UKGE (2021)**: Uncertainty-aware embeddings

### 3. Code Organization

```
src/
├── backends/
│   └── box_embeddings.rs      # Core types (BoxEmbedding, TemporalBox, etc.)
├── eval/
│   └── coref_resolver.rs       # BoxCorefResolver + utilities
└── grounded.rs                 # Identity.box_embedding integration

docs/
├── BOX_EMBEDDINGS_COREFERENCE.md  # Design & theory
├── BOX_EMBEDDINGS_USAGE.md        # Usage guide
└── BOX_EMBEDDINGS_SUMMARY.md      # This file

examples/
└── box_coreference.rs            # Complete example
```

## Test Coverage

All 19 tests passing:
- Core box operations (volume, intersection, conditional probability)
- Temporal boxes (at_time, coreference_at_time, velocity)
- Uncertainty boxes (confidence, conflict detection)
- Gumbel boxes (membership, robust coreference, temperature effects)
- Interaction modeling (interaction_strength, acquisition_roles)
- Helper methods (from_vector, center, size)

## Usage Example

```rust
use anno::backends::box_embeddings::{BoxCorefConfig, BoxEmbedding};
use anno::eval::coref_resolver::BoxCorefResolver;
use anno::{Entity, EntityType};

let entities = vec![
    Entity::new("John", EntityType::Person, 0, 4, 0.9),
    Entity::new("he", EntityType::Person, 10, 12, 0.8),
];

let boxes = vec![
    BoxEmbedding::new(vec![0.0, 0.0], vec![1.0, 1.0]),
    BoxEmbedding::new(vec![0.1, 0.1], vec![0.9, 0.9]),
];

let resolver = BoxCorefResolver::new(BoxCorefConfig::default());
let resolved = resolver.resolve_with_boxes(&entities, &boxes);
```

## Next Steps (Research)

1. **Learning Box Embeddings**: How to learn box parameters from coreference annotations?
2. **Evaluation**: Compare box-based vs. vector-based on CoNLL-2012
3. **Hybrid Approach**: When to use boxes vs. vectors?
4. **Performance**: Optimize box operations (currently O(d) per pair)
5. **Full BoxTE**: Temporal training for time-varying entities

## Exports

All types exported via `src/backends/mod.rs`:
- `BoxEmbedding`, `BoxCorefConfig`
- `TemporalBox`, `BoxVelocity`
- `UncertainBox`, `Conflict`
- `GumbelBox`
- `interaction_strength()`, `acquisition_roles()`

Resolver exported via `src/eval/coref_resolver.rs`:
- `BoxCorefResolver`
- `vectors_to_boxes()`, `resolve_with_box_embeddings()`

