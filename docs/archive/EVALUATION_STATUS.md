# Dataset Evaluation Status

## Progress Summary

### ✅ 1. CLI Hardcoding Issue (FIXED)
**Problem**: `anno dataset eval` ignored the `--dataset` argument and always used hardcoded synthetic data.

**Fix**: 
- CLI now parses dataset name and uses `DatasetLoader` to load real datasets
- Supports all NER datasets (WikiGold, WNUT-17, CoNLL-2003, etc.)
- Gracefully warns when evaluating NER on non-NER datasets (GAP, DocRED)

**Verification**:
```bash
cargo run --features "cli,eval,eval-advanced" -- dataset eval --dataset wikigold --model stacked
```

### ✅ 2. Missing Model Benchmarks (FIXED)
**Problem**: Only `PatternNER` was benchmarked against all datasets. `HeuristicNER` and `StackedNER` (the default models) were never evaluated.

**Fix**:
- Added `benchmark_heuristic_ner_on_datasets()` - evaluates HeuristicNER on all NER datasets
- Added `benchmark_stacked_ner_on_datasets()` - evaluates StackedNER on all NER datasets
- Renamed existing benchmark to `benchmark_pattern_ner_on_datasets()` for clarity

**Verification**:
```bash
cargo test --test real_datasets --features eval-advanced -- --ignored benchmark_heuristic_ner_on_datasets
cargo test --test real_datasets --features eval-advanced -- --ignored benchmark_stacked_ner_on_datasets
```

### ✅ 3. Coreference Evaluation (FULLY IMPLEMENTED)
**Status**: Coreference evaluation fully integrated in CLI and tests.

**What Was Added**:
- `evaluate_coref_on_dataset()` - evaluation function in `tests/real_datasets.rs`
- `benchmark_coreference_on_gap()` - benchmark test using StackedNER + SimpleCorefResolver
- CLI `--task coref` flag for coreference evaluation
- Full CLI integration: `anno dataset eval --dataset gap --task coref --model stacked`

**What Exists**:
- `DatasetLoader::load_or_download_coref()` - loads GAP, PreCo, LitBank
- `CorefEvaluation::compute()` - computes MUC, B³, CEAF, LEA, BLANC, CoNLL F1
- `SimpleCorefResolver` - rule-based resolver for evaluation
- All metric functions (conll_f1, muc_score, b_cubed_score, etc.)
- CLI support with full metrics output

**Verification**:
```bash
cargo run --bin anno --features "cli,eval,eval-advanced" -- dataset eval --dataset gap --task coref --model stacked
```

### ✅ 4. Relation Extraction Evaluation (FULLY IMPLEMENTED)
**Status**: Relation extraction evaluation fully integrated in CLI and tests.

**What Was Added**:
- `load_relation()` and `load_or_download_relation()` methods in `DatasetLoader`
- `parse_docred_relations()` - parser for CrossRE/DocRED format
- `RelationDocument` struct for relation datasets
- `evaluate_relation_on_dataset()` - evaluation function in `tests/real_datasets.rs`
- `benchmark_relation_extraction_on_docred()` - benchmark test
- CLI `--task relation` flag for relation extraction evaluation
- Full CLI integration: `anno dataset eval --dataset docred --task relation --model stacked`

**What Exists**:
- `DatasetLoader` can load DocRED, ReTACRED (via CrossRE proxy)
- `evaluate_relations()` function exists in `src/eval/relation.rs`
- `RelationMetrics`, `RelationEvalConfig` - evaluation types
- Entity-pair heuristic for relation extraction (works with any NER model)
- CLI support with boundary and strict metrics

**Note**: Current implementation uses entity-pair heuristics. For proper relation extraction, a dedicated relation model (like GLiNER multitask) would be needed, but the infrastructure is complete.

**Verification**:
```bash
cargo run --bin anno --features "cli,eval,eval-advanced" -- dataset eval --dataset docred --task relation --model stacked
cargo test --test real_datasets --features eval-advanced -- --ignored benchmark_relation_extraction_on_docred
```

### ❌ 5. AutoNER Model Not Benchmarked
**Status**: `AutoNER` (language-detected routing) is not included in benchmarks.

**What's Missing**:
- No `benchmark_auto_ner_on_datasets()` function
- AutoNER is a router, not a direct model, so evaluation is more complex

### ✅ 6. GLiNER Expanded to All Datasets (FIXED)
**Status**: GLiNER benchmark now tests all NER datasets with comprehensive metrics.

**What Was Added**:
- `benchmark_gliner_on_datasets()` now tests all NER datasets (not just 6)
- Added confusion matrix analysis across all datasets
- Added per-entity-type breakdowns for each dataset
- Shows most confused entity type pairs

**Verification**:
```bash
cargo test --test real_datasets --features "eval-advanced,onnx" -- --ignored benchmark_gliner_on_datasets
```

### ✅ 7. Comprehensive Coreference Evaluation (FIXED)
**Status**: Coreference evaluation now covers all coreference datasets.

**What Was Added**:
- `benchmark_coreference_on_all_datasets()` - evaluates on GAP, PreCo, LitBank
- Comprehensive metrics table showing all 6 coreference metrics
- Individual dataset benchmarks still available

**Verification**:
```bash
cargo test --test real_datasets --features eval-advanced -- --ignored benchmark_coreference_on_all_datasets
```

### ✅ 8. Comprehensive Relation Extraction Evaluation (FIXED)
**Status**: Relation extraction evaluation now covers all relation datasets.

**What Was Added**:
- `benchmark_relation_extraction_on_all_datasets()` - evaluates on DocRED, ReTACRED
- Comprehensive metrics table showing boundary and strict metrics
- Individual dataset benchmarks still available

**Verification**:
```bash
cargo test --test real_datasets --features eval-advanced -- --ignored benchmark_relation_extraction_on_all_datasets
```

### ✅ 9. Enhanced Evaluation Metrics (FIXED)
**Status**: All benchmarks now include comprehensive metrics.

**What Was Added**:
- Confusion matrices for NER models (shows type confusion patterns)
- Per-entity-type breakdowns (P/R/F1 per type)
- Most confused entity type pairs analysis
- Processing time metrics (ms/sentence)

## Current Coverage Matrix

| Model | NER Datasets | Coref Datasets | RE Datasets | Status |
|-------|--------------|----------------|-------------|--------|
| PatternNER | ✅ All | ❌ N/A | ❌ N/A | Complete |
| HeuristicNER | ✅ All | ❌ N/A | ❌ N/A | Complete |
| StackedNER | ✅ All | ✅ All (3) | ✅ All (2) | Complete |
| GLiNER | ✅ All | ❌ N/A | ❌ N/A | Complete |
| AutoNER | ❌ None* | ❌ None* | ❌ None* | N/A* |
| CorefResolver | ❌ N/A | ✅ All (3) | ❌ N/A | Complete |
| RelationExtractor | ❌ N/A | ❌ N/A | ✅ All (2) | Complete* |

*AutoNER is a router to StackedNER, so it doesn't need separate benchmarks
*Uses entity-pair heuristics, not a true relation model

## Verification Commands

### Test CLI Dataset Loading
```bash
# Should load real WikiGold dataset (not synthetic)
cargo run --features "cli,eval,eval-advanced" -- dataset eval --dataset wikigold --model stacked

# Should fail gracefully with clear error
cargo run --features "eval" -- dataset eval --dataset wikigold --model stacked
```

### Test Model Benchmarks
```bash
# Run all NER model benchmarks
cargo test --test real_datasets --features eval-advanced -- --ignored \
  benchmark_pattern_ner_on_datasets \
  benchmark_heuristic_ner_on_datasets \
  benchmark_stacked_ner_on_datasets \
  benchmark_gliner_on_datasets
```

### Test Coreference Evaluation
```bash
# CLI
cargo run --bin anno --features "cli,eval,eval-advanced" -- dataset eval --dataset gap --task coref --model stacked

# Test suite
cargo test --test real_datasets --features eval-advanced -- --ignored benchmark_coreference_on_gap
```

### Test Relation Extraction Evaluation
```bash
# CLI
cargo run --bin anno --features "cli,eval,eval-advanced" -- dataset eval --dataset docred --task relation --model stacked

# Test suite
cargo test --test real_datasets --features eval-advanced -- --ignored benchmark_relation_extraction_on_docred
```

## Next Steps

1. **Improve Relation Extraction**:
   - Integrate GLiNER multitask model for true relation extraction
   - Or document that entity-pair heuristic is the current approach

2. **Add More Evaluation Metrics**:
   - Error analysis (boundary errors, type errors, spurious, missed)
   - Statistical significance testing between models
   - Cross-dataset performance analysis

3. **Performance Optimization**:
   - Batch processing for faster evaluation
   - Parallel dataset loading
   - Caching of model predictions

4. **Documentation**:
   - Add examples of evaluation outputs
   - Document evaluation methodology
   - Create evaluation report templates
