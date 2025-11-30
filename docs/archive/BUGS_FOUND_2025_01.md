# Bugs Found and Fixed - January 2025

**Status**: All bugs fixed, tests added

## Bug 1: Index Out of Bounds in GLiNERCandle::decode_entities ✅ FIXED

**Location**: `src/backends/gliner_candle.rs:711` (original), `src/backends/gliner_candle.rs:716-718` (fixed)

**Issue**: 
The `decode_entities` function was using `end = start + width + 1` and accessing `word_positions.get(end - 1)`. When `start` is near the end of the words array and `width` is at maximum, `end` could exceed `words.len()`, making `end - 1` out of bounds for `word_positions`.

**Root Cause**:
- `generate_spans` uses: `end = start + width` (inclusive end word index)
- `decode_entities` was incorrectly using: `end = start + width + 1` then `end - 1`
- The correct approach: use `end_inclusive = start + width` directly, matching `generate_spans`

**Fix**:
```rust
// Match generate_spans: end_inclusive = start + width (inclusive)
let end_inclusive = start + width;
// Validate bounds before accessing
if start < word_positions.len() && end_inclusive < word_positions.len() {
    if let (Some(&(start_pos, _)), Some(&(_, end_pos))) =
        (word_positions.get(start), word_positions.get(end_inclusive))
    {
        // ... create entity
    }
}
```

**Tests**: `tests/gliner_candle_bug_tests.rs` - 6 comprehensive tests

---

## Bug 2-7: Integer Overflow in Span Count Calculations ✅ FIXED

**Locations**:
- `src/backends/nuner.rs:336, 390, 399`
- `src/backends/gliner2.rs:514, 638, 677`
- `src/backends/gliner_onnx.rs:258, 399, 437`
- `src/backends/session_pool.rs:356, 466, 505`

**Issue**: 
Multiple locations calculate `num_spans = num_words * MAX_SPAN_WIDTH` and `dim = start * MAX_SPAN_WIDTH + width` without overflow protection. While unlikely in practice (would require ~1.5 billion words), this is still a bug.

**Fix Pattern**:
```rust
// Use checked_mul to prevent overflow (consistent with gliner2.rs:2388)
let num_spans = match num_words.checked_mul(MAX_SPAN_WIDTH) {
    Some(v) => v,
    None => {
        return Err(Error::InvalidInput(format!(
            "Span count overflow: {} words * {} MAX_SPAN_WIDTH",
            num_words, MAX_SPAN_WIDTH
        )));
    }
};

// Also check num_spans * 2 for span_idx allocation
let span_idx_len = match num_spans.checked_mul(2) {
    Some(v) => v,
    None => {
        log::warn!("Span idx length overflow, using max");
        usize::MAX
    }
};

// Check dim calculation
let dim = match start.checked_mul(MAX_SPAN_WIDTH) {
    Some(v) => match v.checked_add(width) {
        Some(d) => d,
        None => {
            log::warn!("Dim overflow, skipping span");
            continue;
        }
    },
    None => {
        log::warn!("Dim overflow, skipping span");
        continue;
    }
};
```

**Impact**: Prevents potential panics or incorrect allocations with extremely large inputs.

---

## Summary

- **Total bugs found**: 7 (1 bounds bug + 6 overflow bugs)
- **All fixed**: ✅
- **Tests added**: ✅
- **Consistency**: All fixes follow the same pattern as `gliner2.rs:2388` which already used `checked_mul`

