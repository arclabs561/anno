# Evaluation Results Critique

> **Note**: This document critiques specific evaluation results from a particular run. For general evaluation limitations and research findings, see [EVALUATION_CRITIQUE.md](EVALUATION_CRITIQUE.md). For implementation review, see [EVALUATION_CRITICAL_REVIEW.md](EVALUATION_CRITICAL_REVIEW.md).

## Executive Summary

**Total: 258 combinations | ✓: 75 (29%) | ⊘: 177 (69%) | ✗: 6 (2%)**

The evaluation system has several critical issues that make results misleading and the report hard to use.

## Critical Problems

### 1. Backend Factory Mismatch (6 failures)

**Problem**: `coref_resolver` backend doesn't exist in `BackendFactory::create()`.

- `BackendFactory::create()` only creates `Model` backends
- Coreference resolvers use `create_coref_resolver()` which returns `CoreferenceResolver` trait
- Code tries to create `coref_resolver` as a `Model`, causing "Unknown backend" errors

**Fix**: `evaluate_coref_task()` should use `create_coref_resolver()` instead of `BackendFactory::create()`.

**Location**: `src/eval/task_evaluator.rs:989` - The code tries to create `backend_name` (which is "coref_resolver") as a Model backend, but it should:
1. Use a NER backend to extract entities first (e.g., "heuristic" or "stacked")
2. Then use `create_coref_resolver()` to resolve coreference on those entities

The current logic is backwards - it tries to use "coref_resolver" as a Model backend to extract entities, which fails.

### 2. Entity Type Mismatches (58 datasets with 0.0 F1)

**Problem**: RegexNER and HeuristicNER only support limited entity types.

- **RegexNER**: Only extracts structured entities (dates, money, emails, URLs) - NOT named entities
- **HeuristicNER**: Only extracts Person, Organization, Location - NOT domain-specific types
- **Result**: 0.0 F1 on:
  - MIT Movie (actor, director, genre, title, etc.)
  - MIT Restaurant (amenity, cuisine, dish, etc.)
  - Biomedical datasets (BC5CDR, NCBIDisease, GENIA, AnatEM, BC2GM, BC4CHEMD) - Disease, Chemical, Gene types
  - Domain-specific datasets (FabNER, etc.)

**Expected Behavior**: These backends should be skipped for datasets with incompatible entity types, OR the report should clearly indicate "incompatible entity types" rather than showing 0.0 F1.

**Fix**: 
1. Add entity type compatibility checking before evaluation
2. Mark as "⊘ incompatible-types" instead of "✓ 0.0 F1"
3. Or filter out incompatible combinations upfront

### 3. Excessive Noise (69% skipped)

**Problem**: 177 out of 258 combinations are skipped due to missing features.

- Makes report hard to scan
- Most useful information (actual results) is buried
- Should filter or collapse skipped entries

**Fix**: 
1. Add `--hide-skipped` flag to report generation
2. Collapse skipped entries into summary: "177 skipped (onnx/candle features not enabled)"
3. Only show skipped if explicitly requested

### 4. Incomplete Task Coverage

**Problem**: Only 4 of 10 tasks are actually evaluated.

**Evaluated**:
- NER (25 datasets)
- Relation Extraction (2 datasets, both skipped)
- Intra-document Coreference (3 datasets, all failed)
- Abstract Anaphora Resolution (3 datasets, all failed)

**Not Evaluated** (no datasets):
- NED (Named Entity Disambiguation)
- Inter-document Coreference
- DiscontinuousNER (has CADEC dataset but not evaluated)
- Event Extraction
- Text Classification
- Hierarchical Extraction

**Fix**: 
1. Add CADEC evaluation for DiscontinuousNER
2. Document which tasks have no datasets (expected)
3. Don't list tasks with 0 datasets in the report

### 5. Report Readability Issues

**Problems**:
- Too much noise (177 skipped entries)
- 0.0 F1 results not actionable (why did it fail?)
- Missing context (entity type mismatches not explained)
- Backend summary shows "Avg F1" but pattern has 0.0 (misleading)

**Fix**:
1. Filter 0.0 F1 results or mark as "incompatible"
2. Add entity type compatibility column
3. Show only successful results by default
4. Add "Why 0.0?" column explaining failures

### 6. Missing Dataset Coverage

**Problem**: Some datasets aren't being evaluated.

- **CADEC (DiscontinuousNER)**: Dataset exists but task not evaluated (no DiscontinuousNER section in report)
- **Relation Extraction**: Only 2/6 datasets shown (DocRED, ReTACRED). Missing: NYTFB, WEBNLG, GoogleRE, BioRED
  - Likely cause: Datasets not cached or download failing silently
  - Should verify dataset loading for these

**Fix**: 
1. Check why CADEC isn't evaluated for DiscontinuousNER task
2. Verify relation extraction dataset downloads/loading
3. Add dataset availability check before evaluation

## Code Issues

### 1. Backend Factory Doesn't Support Coreference

```rust
// src/eval/task_evaluator.rs:989
let backend = BackendFactory::create(backend_name)?;  // ❌ Wrong - coref_resolver not a Model
```

Should be:
```rust
let resolver = create_coref_resolver(backend_name)?;  // ✅ Correct
```

### 2. No Entity Type Compatibility Check

HeuristicNER and RegexNER are evaluated on datasets with incompatible entity types, resulting in 0.0 F1. Should check compatibility first:

```rust
fn is_compatible(backend: &str, dataset_entity_types: &[&str]) -> bool {
    match backend {
        "heuristic" => {
            // Only supports Person, Organization, Location
            dataset_entity_types.iter().all(|t| 
                matches!(t, "person" | "organization" | "location" | "per" | "org" | "loc")
            )
        }
        "pattern" => {
            // Only supports structured entities
            false  // RegexNER doesn't do named entities
        }
        _ => true  // ML backends are zero-shot or trained
    }
}
```

### 3. Task-Dataset Mapping Issues

- **DiscontinuousNER**: Has CADEC dataset but not being evaluated (no section in report)
- **Relation Extraction**: Shows only 2 datasets (DocRED, ReTACRED) but 6 are mapped (NYTFB, WEBNLG, GoogleRE, BioRED missing)
- **Coref task logic**: Uses `BackendFactory::create()` for entity extraction, then resolves - but should use `create_coref_resolver()` for the resolver itself

## Recommendations

### Immediate Fixes

1. **Fix coref_resolver backend creation** - Use `create_coref_resolver()` instead of `BackendFactory::create()`
2. **Add entity type compatibility checking** - Skip incompatible combinations or mark clearly
3. **Filter 0.0 F1 results** - Either skip or mark as "incompatible entity types"
4. **Collapse skipped entries** - Show summary instead of 177 individual lines

### Report Improvements

1. **Add compatibility column** - Show which entity types backend supports vs dataset requires
2. **Explain 0.0 F1** - Add reason column: "incompatible types", "no entities found", etc.
3. **Filter by default** - Only show successful results, with option to show all
4. **Task coverage summary** - Show which tasks have datasets vs which don't

### Code Improvements

1. **Entity type compatibility trait** - Add `EntityTypeCompatible` trait to backends
2. **Pre-filter incompatible combinations** - Don't even attempt evaluation
3. **Better error messages** - "Backend 'heuristic' only supports [Person, Org, Location], but dataset requires [Disease, Chemical]"

## Metrics

- **Coverage**: 4/10 tasks evaluated (40%)
- **Success Rate**: 75/258 = 29% (but many are incompatible, not failures)
- **Noise Ratio**: 177/258 = 69% skipped (too high)
- **Actionable Results**: ~25 successful NER evaluations (useful)
- **Misleading Results**: 58 datasets with 0.0 F1 (should be marked incompatible)

## Conclusion

The evaluation system works but produces misleading results:
- 0.0 F1 doesn't mean "failed" - it means "incompatible entity types"
- 69% skipped entries make report hard to read
- Coref backend creation is broken
- Missing task coverage (DiscontinuousNER, etc.)

Priority fixes:
1. Fix coref_resolver backend creation
2. Add entity type compatibility checking
3. Improve report filtering and readability

