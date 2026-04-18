# anno-core

[![crates.io](https://img.shields.io/crates/v/anno-core.svg)](https://crates.io/crates/anno-core)
[![Documentation](https://docs.rs/anno-core/badge.svg)](https://docs.rs/anno-core)
[![CI](https://github.com/arclabs561/anno/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/anno/actions/workflows/ci.yml)

Stable data model and invariants for [anno](https://crates.io/crates/anno-lib): entities, spans, tracks, signals, coreference chains, and corpus types.

This crate intentionally avoids CLI and evaluation dependencies so it can serve as the shared type foundation across the anno workspace.

## Install

```toml
[dependencies]
anno-core = "0.6"
```

## Key types

| Type | Purpose |
|------|---------|
| `Entity` | A named entity with span, type, confidence, and provenance |
| `Span` | Character-offset range (Unicode scalar values, not bytes) |
| `Signal` / `Track` | Layered extraction outputs with provenance tracking |
| `CorefChain` / `CorefDocument` | Within-document coreference clusters |
| `Identity` | Cross-document entity identity |
| `Corpus` | Collection of grounded documents |
| `Relation` | Typed `(head, relation, tail)` triple |

...and ~30 supporting types; see [docs.rs](https://docs.rs/anno-core) for the full API.

## Usage

```rust
use anno_core::{EntityBuilder, EntityType};

let entity = EntityBuilder::new("Sophie Wilson", EntityType::Person)
    .span(0, 14)
    .confidence(0.95)
    .build();

assert_eq!(entity.text, "Sophie Wilson");
assert_eq!(entity.entity_type, EntityType::Person);
```

## Offset invariant

All spans use **character offsets** (Unicode scalar values), not byte offsets. See the [CONTRACT](../../docs/CONTRACT.md) for details.

## License

Dual-licensed under MIT or Apache-2.0.
