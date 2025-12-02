# Box Embeddings Organization

## Current Organization

### Training Code Location

**`anno` has the canonical training implementation**:
- `src/backends/box_embeddings_training.rs` (1680 lines)
- Includes `evaluate_standard_metrics()` for standard coreference evaluation
- Actively used in examples
- Tightly integrated with `anno`'s evaluation framework

**`matryoshka-box` extends with research features**:
- Uses `anno`'s training as base
- Adds matryoshka-specific features (variable dimensions, hierarchical reasoning, etc.)
- Depends on `anno` for box types and training

## Structure

```
anno (production + standard training)
├── src/backends/
│   ├── box_embeddings.rs              # Core types (inference)
│   └── box_embeddings_training.rs     # Standard training (CANONICAL)
├── examples/
│   ├── box_training.rs                # Training examples
│   └── box_training_real_data.rs      # Real data training
└── docs/
    └── Training is in anno, matryoshka-box extends it

matryoshka-box (research extensions)
├── inference/rust/src/
│   ├── matryoshka_training.rs         # Research extensions (if needed)
│   └── trained_resolver.rs            # Resolver with trained boxes
└── Uses anno's training as base, extends with research features
```

## Rationale

1. **Active Development**: We're actively using and improving training in `anno`
2. **Tight Integration**: Training uses `anno`'s evaluation framework extensively
3. **User Experience**: Users expect training in `anno` (it's there, it works)
4. **Research Extensions**: `matryoshka-box` can extend `anno`'s training without duplication

## Key Points

- **Training code is in `anno`** (canonical location)
- **`matryoshka-box` extends** with research features
- **No duplication**: Use `anno`'s training, extend in matryoshka-box
- **Clear ownership**: Standard in `anno`, research in `matryoshka-box`

## Documentation

- `docs/MATRYOSHKA_BOX_INTEGRATION.md` - Detailed integration design
- `docs/MIGRATION_NOTES.md` - Organization details (updated to reflect current state)
- `docs/BOX_ORGANIZATION_SUMMARY.md` - Quick reference (this file consolidates the review/decision/summary docs)

