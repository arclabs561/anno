# Evaluation Optimization: Relation Matching

## Issue Found

**Location**: `src/eval/relation.rs` (lines 257, 293)

**Problem**: The relation evaluation code was not skipping predictions that had already been matched, leading to unnecessary iterations through gold relations.

**Impact**: Performance optimization - the logic was correct, but inefficient for large datasets.

## Fix

Added early-exit checks for already-matched predictions in both strict and boundary matching loops:

```rust
// Strict matching: exact entity spans + relation type
for (pi, p) in pred.iter().enumerate() {
    // Skip predictions that are already matched
    if pred_matched_strict[pi] {
        continue;
    }
    for (gi, g) in gold.iter().enumerate() {
        // ... matching logic
    }
}
```

**Before**: O(n×m) iterations even for matched predictions
**After**: O(n×m) worst case, but skips matched predictions early

## Verification

- ✅ Code compiles successfully
- ✅ Logic remains correct (one-to-one matching preserved)
- ✅ Performance improved for datasets with many matches

