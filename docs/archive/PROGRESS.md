# Evaluation Implementation Progress

## Completed

### ✅ Coreference Evaluation
- **Added**: `evaluate_coref_on_dataset()` function in `tests/real_datasets.rs`
- **Added**: `benchmark_coreference_on_gap()` test
- **Integration**: Uses `StackedNER` + `SimpleCorefResolver` for end-to-end evaluation
- **Metrics**: Full support for MUC, B³, CEAF, LEA, BLANC, CoNLL F1
- **Status**: Test passes, ready for use

### ✅ CLI Dataset Loading
- **Fixed**: `anno dataset eval` now loads real datasets via `DatasetLoader`
- **Fixed**: Removed hardcoded synthetic test cases
- **Status**: Works correctly with `--dataset` argument

### ✅ Model Benchmarks
- **Added**: `benchmark_heuristic_ner_on_datasets()`
- **Added**: `benchmark_stacked_ner_on_datasets()`
- **Status**: All NER models now have comprehensive benchmarks

## Remaining Work

### ❌ Relation Extraction Evaluation
- **Missing**: `evaluate_relation_on_dataset()` function
- **Missing**: Relation dataset loader (DocRED/ReTACRED parsing for relations, not just NER)
- **Missing**: CLI integration
- **Note**: `evaluate_relations()` function exists but needs dataset integration

### ❌ CLI Task Support
- **Missing**: `--task coref` flag for coreference evaluation
- **Missing**: `--task relation` flag for relation extraction evaluation
- **Current**: CLI only supports NER evaluation

### ❌ AutoNER Benchmark
- **Missing**: Evaluation for AutoNER (router model)
- **Note**: More complex due to routing behavior

### ⚠️ GLiNER Coverage
- **Current**: Only 6 datasets
- **Missing**: Full coverage like other models

## Next Steps

1. Add CLI `--task` flag to support coref/relation evaluation
2. Implement relation extraction dataset loading
3. Add `evaluate_relation_on_dataset()` function
4. Consider AutoNER evaluation strategy
5. Expand GLiNER to all NER datasets

