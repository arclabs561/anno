# Bugs Fixed

## Summary

Found and fixed several critical bugs in the evaluation system:

## 1. Deadlock Bug (CRITICAL)

**Location**: `src/eval/task_evaluator.rs:553-558`

**Issue**: Potential deadlock when handling mutex poisoning. The code tried to lock the mutex again in the else branch, which clippy flagged as a potential deadlock.

**Fix**: Simplified to use `unwrap_or_else` directly without nested if/else:

```rust
// Before (buggy):
if let Ok(mut cache) = self.per_example_scores_cache.lock() {
    *cache = None;
} else {
    drop(self.per_example_scores_cache.lock().unwrap_or_else(|e| e.into_inner()));
}

// After (fixed):
let mut cache = self.per_example_scores_cache.lock().unwrap_or_else(|e| e.into_inner());
*cache = None;
```

## 2. Variance Calculation Bug (STATISTICAL ERROR)

**Location**: Multiple locations in `src/eval/task_evaluator.rs`

**Issue**: Using population variance (dividing by n) instead of sample variance (dividing by n-1, Bessel's correction). This causes biased standard deviation estimates, leading to incorrect confidence intervals.

**Fixed Locations**:
1. `compute_confidence_intervals` (line ~2312)
2. `compute_confidence_intervals_from_scores` (line ~2591)
3. `compute_temporal_stratification` - pre_cutoff (line ~2539)
4. `compute_temporal_stratification` - post_cutoff (line ~2555)
5. `compute_stratified_metrics_from_scores` - per entity type (line ~2456)

**Fix**: Changed from:
```rust
let variance = scores.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / scores.len() as f64;
```

To:
```rust
let n = scores.len() as f64;
let variance = if n > 1.0 {
    scores.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
} else {
    0.0
};
```

**Impact**: Confidence intervals are now correctly computed using unbiased sample variance, which is critical for statistical validity.

## Verification

All fixes verified:
- ✅ Deadlock warning eliminated (clippy clean)
- ✅ All variance calculations use sample variance (n-1)
- ✅ Edge case handling (n=0, n=1) added
- ✅ Tests pass
- ✅ Examples run successfully

## Remaining Warnings

Some `unwrap()` calls remain but are intentional (e.g., in test code, or where we've already checked for None/Err). These are acceptable for now but could be improved with better error handling in the future.

