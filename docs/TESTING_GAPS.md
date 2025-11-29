# Testing & Evaluation Gaps

Audit of areas requiring additional testing and evaluation.

## CI/CD Status

Single workflow `ci.yml` - runs on push, PR, or manual trigger.

### Jobs (Always Run)

| Job | Features | What it tests |
|-----|----------|---------------|
| `check` | - | Cargo check |
| `fmt` | - | rustfmt |
| `clippy` | - | Lints |
| `test-minimal` | default | Zero-dependency baseline |
| `test-eval` | eval-advanced, discourse | Evaluation framework |
| `test-cross-platform` | eval | Ubuntu, macOS, Windows |
| `test-onnx` | onnx | ONNX backend (no models) |
| `test-candle` | candle | Candle backend (no models) |
| `proptest` | eval-advanced | Property tests |
| `regression` | eval | F1 regression checks |
| `examples` | eval, discourse | Example compilation |
| `audit` | - | Security audit |

### Jobs (Manual Trigger Only)

| Job | Purpose |
|-----|---------|
| `comprehensive` | All tests including `--ignored`, model downloads, HTML reports |

To run comprehensive tests: Go to Actions > CI > Run workflow.

---

## Summary

| Area | Coverage | Priority | Blocker |
|------|----------|----------|---------|
| Binary Embeddings | ✅ Unit + Integration | - | - |
| Entropy Filtering | ✅ Unit + Integration | - | - |
| GLiNER2 Multi-task | ⚠️ Schema only | HIGH | Model download |
| CDCR Real Data | ⚠️ Synthetic only | MEDIUM | Dataset access |
| Abstract Anaphora | ⚠️ Examples only | MEDIUM | Gold annotations |
| Event Extraction | ⚠️ Lexicon gaps | LOW | Manual expansion |

---

## Priority 1: Needs Model Download

### GLiNER2 End-to-End

**Current state**: Only `TaskSchema` builder tests exist (4 tests).

**Missing**: Actual model inference with multi-task extraction.

```bash
# To test properly:
cargo test --test gliner2_tests --features candle
# Requires: model download (~500MB)
```

**Recommended tests**:
1. Multi-task schema: NER + classification in single pass
2. Structure extraction: Product/person info extraction
3. Zero-shot entity types: Custom labels at inference time
4. Performance: Throughput on 1K documents

### Neural Event Extraction

**Current state**: Rule-based tests only.

```bash
# To test with GLiNER backend:
cargo run --example discourse_pipeline --features candle
```

---

## Priority 2: Needs Real Datasets

### CDCR Evaluation

**Current state**: Synthetic tech/political/sports/financial datasets (defined in `cdcr.rs`).

**Missing**: Evaluation on standard CDCR benchmarks.

| Dataset | Status | Notes |
|---------|--------|-------|
| ECB+ | ❌ Not tested | Standard CDCR benchmark |
| WEC | ❌ Not tested | Wikipedia Event Coreference |
| CD²CR | ❌ Not tested | Cross-domain CDCR |

**Blocking**: Need dataset loaders for these formats.

### Abstract Anaphora Evaluation

**Current state**: Synthetic `AbstractAnaphoraDataset::standard()` and `::extended()`.

**Missing**: Comparison with human-annotated corpora.

| Dataset | Status | Notes |
|---------|--------|-------|
| ARRAU | ❌ Not tested | Bridging + abstract anaphora |
| SciCo | ❌ Not tested | Scientific coreference |

---

## Priority 3: Known Limitations to Address

### Event Extraction Lexicon Gaps

Documented in `tests/discourse_comprehensive.rs`:

```rust
// TODO: Add these to lexicon:
// - "decision", "action", "meeting" (light verb constructions)
// - "surprised", "shocked", "amazed" (reaction events)
// - "respond", "react", "retaliate" (response events)
```

### Coreference Definition Edge Cases

From research discussion - not fully tested:

1. **Generic NPs**: "Dogs are loyal" - should "Dogs" corefer across sentences?
2. **Bridging anaphora**: "The car... the engine" (part-whole)
3. **Split antecedents**: "John met Mary. They left together."

---

## Test Commands Cheatsheet

### Using Just (recommended)

```bash
just              # List all commands
just check        # Quick check (fmt + clippy + test)
just ci           # Full CI simulation
just test-all     # All tests with features
just eval-quick   # Fast synthetic evaluation
```

### Manual Commands

```bash
# Run all unit tests
cargo test --lib

# Run with eval features
cargo test --lib --features eval-advanced

# Run discourse tests
cargo test --features discourse

# Run GLiNER2 tests (requires model)
cargo test --test gliner2_tests --features candle

# Run CDCR tests
cargo test cdcr

# Run new integration tests
cargo test --test new_features_integration --features eval-advanced

# Run abstract anaphora example
cargo run --example abstract_anaphora_eval

# Run full pipeline example
cargo run --example discourse_pipeline
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANNO_MAX_EXAMPLES` | Max examples per dataset | 50 in CI, 0 (unlimited) locally |
| `PROPTEST_CASES` | Property test iterations | 256 |
| `CI` / `GITHUB_ACTIONS` | Detected automatically | - |

### Dataset Subsets

```rust
use anno::eval::loader::DatasetId;

// For CI smoke tests (3 datasets, ~2MB)
let quick = DatasetId::quick();

// For development (6 datasets, ~5MB)
let medium = DatasetId::medium();

// All datasets (24 datasets, ~100MB)
let all = DatasetId::all();
```

---

## Evaluation Metrics to Report

When evaluating, always report:

### NER
- Strict F1 (exact boundary + type)
- Partial F1 (overlap + type)
- Per-type breakdown
- Unseen entity ratio

### Coreference
- MUC, B³, CEAF, LEA, BLANC (not just CoNLL F1)
- Chain-length stratified metrics
- Singleton handling mode

### CDCR
- B³ cross-document
- Cluster purity
- Blocking recall (if using LSH/binary)

### Abstract Anaphora
- Nominal vs abstract accuracy
- Per-type breakdown (Event/Fact/Proposition/Situation)
- LEA scores

---

## Current Test Coverage

```
Total tests: 1591
  - Unit tests (lib): 695
  - Integration tests: 896
  - Property tests: 41
  - Ignored (need models/network): 31
```

### Feature Coverage

| Feature | Unit Tests | Integration | Examples |
|---------|------------|-------------|----------|
| eval-advanced | 695 | 44 | eval_basic |
| discourse | 23 | 5 | discourse_pipeline |
| onnx | builds | 7 (ignored) | gliner_candle |
| candle | builds | 5 (ignored) | candle, bert |

## Next Steps

1. **Immediate**: Run GLiNER2 example to verify multi-task extraction works
2. **Short-term**: Add ECB+ loader for CDCR evaluation
3. **Medium-term**: Expand event lexicon based on documented gaps
4. **Long-term**: Build ARRAU loader for abstract anaphora gold standard

