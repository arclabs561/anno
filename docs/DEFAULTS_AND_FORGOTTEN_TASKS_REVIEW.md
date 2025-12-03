# Defaults and Forgotten Tasks Review

**Date**: 2025-01-25  
**Scope**: Review of default configuration values and forgotten tasks from archived documentation

## Default Configuration Review

### TaskEvalConfig Defaults

**Location**: `src/eval/task_evaluator.rs:129-145`

| Setting | Default | Rationale | Assessment |
|---------|---------|-----------|------------|
| `tasks` | `Task::all().to_vec()` | Evaluate all tasks | ✅ **Good** - Comprehensive by default |
| `datasets` | `vec![]` | Empty = all suitable datasets | ✅ **Good** - Flexible, uses all available |
| `backends` | `vec![]` | Empty = all compatible backends | ✅ **Good** - Tests all backends |
| `max_examples` | `None` | No limit | ⚠️ **Questionable** - Could be slow for large datasets |
| `seed` | `Some(42)` | Fixed seed for reproducibility | ✅ **Good** - Reproducible by default |
| `require_cached` | `false` | Allow downloads | ✅ **Good** - User-friendly |
| `relation_threshold` | `0.5` | 50% confidence | ✅ **Good** - Standard threshold |
| `robustness` | `false` | No robustness testing | ✅ **Good** - Expensive, opt-in |
| `compute_familiarity` | `true` | Compute familiarity scores | ✅ **Good** - Useful for zero-shot |
| `temporal_stratification` | `false` | No temporal breakdown | ✅ **Good** - Requires metadata |
| `confidence_intervals` | `true` | Compute CIs | ✅ **Good** - Better reporting |

**Recommendations**:
1. ⚠️ **Consider adding `max_examples: Some(1000)` by default** - Prevents accidentally running on huge datasets
2. ✅ All other defaults are reasonable

### EvalConfig Defaults (Harness)

**Location**: `src/eval/harness.rs:80-93`

| Setting | Default | Rationale | Assessment |
|---------|---------|-----------|------------|
| `max_examples_per_dataset` | `0` (unlimited) | No limit | ⚠️ **Same concern as above** |
| `breakdown_by_difficulty` | `true` | Include difficulty breakdown | ✅ **Good** - Useful analysis |
| `breakdown_by_domain` | `true` | Include domain breakdown | ✅ **Good** - Useful analysis |
| `breakdown_by_type` | `true` | Include type breakdown | ✅ **Good** - Useful analysis |
| `warmup` | `true` | Run warmup iteration | ✅ **Good** - Accurate timing |
| `warmup_iterations` | `1` | Single warmup | ✅ **Good** - Reasonable |
| `min_confidence` | `None` | No filtering | ✅ **Good** - See all predictions |
| `cache_dir` | `None` | Use default cache | ✅ **Good** - Standard behavior |
| `normalize_types` | `false` | Preserve original types | ✅ **Good** - Preserve dataset semantics |

**Recommendations**:
1. ⚠️ **Consider adding `max_examples_per_dataset: 1000` by default** - Same concern

### Statistical Constants

**Location**: `src/eval/task_evaluator.rs:30-40`

| Constant | Value | Rationale | Assessment |
|----------|-------|-----------|------------|
| `DEFAULT_Z_SCORE_95` | `1.96` | 95% CI z-score | ✅ **Correct** - Standard value |
| `DEFAULT_PLACEHOLDER_STD_DEV` | `0.05` | Placeholder when variance unknown | ⚠️ **Arbitrary** - Should be documented |
| `MAX_CI_SAMPLE_SIZE` | `100` | Max samples for CI computation | ✅ **Good** - Performance vs accuracy tradeoff |
| `MIN_CI_SAMPLE_SIZE` | `1` | Min samples for CI | ⚠️ **Questionable** - CI with n=1 is meaningless |
| `ROBUSTNESS_TEST_LIMIT` | `50` | Max examples for robustness | ✅ **Good** - Performance limit |

**Recommendations**:
1. ⚠️ **Change `MIN_CI_SAMPLE_SIZE` to `2`** - CI requires at least 2 samples
2. 📝 **Document `DEFAULT_PLACEHOLDER_STD_DEV`** - Explain why 0.05 was chosen

## Forgotten Tasks from Archived Docs

### High Priority (From `REMAINING_WORK_SUMMARY.md`)

#### 1. Complete Per-Example Score Integration ⚠️ **PARTIALLY DONE**

**Status**: Infrastructure exists, but integration incomplete

**What's Done**:
- ✅ `per_example_scores` tracked in `evaluate_ner_task`
- ✅ `compute_stratified_metrics_from_scores()` function exists
- ✅ `compute_confidence_intervals_from_scores()` function exists
- ✅ Per-example scores cached in `per_example_scores_cache`

**What's Needed**:
- ⚠️ Currently uses cached scores when available, but could be more efficient
- ⚠️ Need to verify that stratified metrics use per-example scores when available

**Current Status**: **Mostly complete** - The code does use per-example scores when available (see `task_evaluator.rs:595-609`), but could be optimized.

**Action**: Verify integration is working correctly, add tests if needed.

#### 2. Temporal Metadata Structure ⚠️ **STRUCTURE READY, DATA MISSING**

**Status**: Framework ready, needs data source

**What's Done**:
- ✅ `StratifiedMetrics.by_temporal_stratum` field exists
- ✅ `compute_temporal_stratification()` function exists
- ✅ Structure ready for temporal stratification

**What's Needed**:
- ❌ Add temporal metadata to `LoadedDataset` or dataset loaders
- ❌ Entity creation date tracking
- ❌ KB version metadata
- ❌ Temporal stratum assignment logic

**Action**: Add optional `temporal_metadata: Option<TemporalMetadata>` to `LoadedDataset`.

### Medium Priority

#### 3. Embedding-Based Familiarity Integration ❌ **NOT DONE**

**Status**: Function exists, not integrated

**What's Done**:
- ✅ `LabelShift::from_type_sets_with_embeddings()` function exists
- ✅ Embedding computation infrastructure available

**What's Needed**:
- ❌ Integration with encoder backends for label embeddings
- ❌ Automatic embedding computation for familiarity
- ❌ Fallback to string-based if embeddings unavailable

**Action**: Integrate embedding-based familiarity when encoder backends are available.

#### 4. KB Version Tracking ❌ **NOT DONE**

**Status**: Not started

**What's Needed**:
- ❌ KB version metadata in datasets
- ❌ URI validation against current KB
- ❌ Emerging entity (NIL) separation
- ❌ URI set expansion (owl:sameAs links)

**Action**: Add `kb_version: Option<String>` to dataset metadata.

### Low Priority

#### 5. Inter-Doc Coref Specific Evaluation ❌ **NOT DONE**

**Status**: Not implemented

**What's Needed**:
- ❌ Distinction between intra-doc and cross-doc coref
- ❌ Cross-doc specific metrics
- ❌ Generalization validation (train/test domain split)

**Action**: Add `coref_type: IntraDoc | CrossDoc` to coref datasets.

#### 6. Improve Confidence Interval Efficiency ⚠️ **MOSTLY DONE**

**Status**: Works but could be optimized

**What's Done**:
- ✅ CI computation from per-example scores (when available)
- ✅ Fallback to sampling and recomputation

**What's Needed**:
- ⚠️ Currently uses cached per-example scores, which is good
- ⚠️ Could avoid recomputation entirely if scores are always available

**Action**: Verify that CI computation always uses cached scores when available.

### Box Embeddings Evaluation Gaps (From `BOX_EVALUATION_GAPS.md`)

#### 1. Standard Coreference Metrics ❌ **NOT DONE**

**Status**: Missing standard metrics

**What's Missing**:
- ❌ MUC (link-based)
- ❌ B³ (mention-based)
- ❌ CEAF-e/m (entity/mention alignment)
- ❌ LEA (link-based entity-aware)
- ❌ BLANC (rand-index based)
- ❌ CoNLL F1 (standard benchmark)
- ❌ Chain-length stratification

**Action**: Add `BoxCorefResolver` to `TaskEvaluator` and use standard metrics.

#### 2. Integration with Evaluation Framework ❌ **NOT DONE**

**Status**: `BoxCorefResolver` not integrated

**What's Needed**:
- ❌ `TaskEvaluator` support for `BoxCorefResolver`
- ❌ Comparison with other resolvers
- ❌ Standard benchmark evaluation

**Action**: Add `BoxCorefResolver` to evaluation framework.

#### 3. Standard Benchmark Evaluation ❌ **NOT DONE**

**Status**: Datasets available but not evaluated

**What's Needed**:
- ❌ GAP test set evaluation
- ❌ PreCo dataset evaluation
- ❌ CoNLL-2012 evaluation (if available)
- ❌ LitBank evaluation

**Action**: Evaluate box embeddings on standard benchmarks.

## Summary of Recommendations

### Immediate Actions (High Priority)

1. **Change `MIN_CI_SAMPLE_SIZE` to `2`** - CI with n=1 is statistically meaningless
2. **Consider default `max_examples: Some(1000)`** - Prevent accidentally slow runs
3. **Verify per-example score integration** - Ensure it's working optimally
4. **Add temporal metadata structure** - Framework ready, just needs data

### Medium Priority

1. **Integrate embedding-based familiarity** - When encoder backends available
2. **Add KB version tracking** - For NED evaluation
3. **Add BoxCorefResolver to evaluation** - Standard metrics integration

### Low Priority

1. **Inter-doc coref evaluation** - Specialized use case
2. **Document placeholder std_dev** - Explain rationale
3. **Standard benchmark evaluation for boxes** - Research priority

## Default Value Changes Proposed

```rust
// In TaskEvalConfig::default()
max_examples: Some(1000),  // Instead of None - prevent slow runs

// In constants
const MIN_CI_SAMPLE_SIZE: usize = 2;  // Instead of 1 - statistical validity
```

## Status Summary

- ✅ **Defaults are mostly reasonable** - Only minor improvements needed
- ⚠️ **Per-example score integration** - Mostly done, verify completeness
- ❌ **Temporal metadata** - Structure ready, needs data
- ❌ **Box embeddings evaluation** - Significant gaps remain
- ❌ **Embedding-based familiarity** - Not integrated
- ❌ **KB version tracking** - Not started

Most forgotten tasks are **enhancements** rather than **critical bugs**. The core evaluation system is complete and working.

