# Latest Bugs Found and Fixed

**Date**: 2025-01-XX
**Status**: Bug found, fix implemented, tests added

## Bug 1: Index Out of Bounds in GLiNERCandle::decode_entities

**Location**: `src/backends/gliner_candle.rs:711` (original), `src/backends/gliner_candle.rs:717` (fixed)

**Issue**: 
The `decode_entities` function uses `word_positions.get(end - 1)` where `end = start + width + 1`. When `start` is near the end of the words array and `width` is at maximum, `end` can exceed `words.len()`, making `end - 1` out of bounds for `word_positions`.

**Root Cause**:
- `generate_spans` uses: `end = start + width` (inclusive end)
- `decode_entities` was using: `end = start + width + 1` (exclusive end)
- When `start = words.len() - 1` and `width = 1` (max allowed), `end = words.len() + 1`
- Then `end - 1 = words.len()`, which is out of bounds for `word_positions` (valid indices: 0..words.len())

**Example**:
```rust
// words.len() = 5, word_positions.len() = 5
// start = 4, width = 1
// end = 4 + 1 + 1 = 6
// end - 1 = 5, which is >= word_positions.len() (5) ❌
```

**Fix**:
```rust
// Match generate_spans: end_inclusive = start + width
let end_inclusive = start + width;
// Convert to exclusive and clamp to bounds
let end_exclusive = (end_inclusive + 1).min(words.len());
// Use saturating_sub to prevent underflow, clamp to valid range
let end_index = end_exclusive.saturating_sub(1).min(word_positions.len().saturating_sub(1));
```

**Tests Added**: `tests/gliner_candle_bug_tests.rs`
- `test_decode_entities_end_index_bounds` - Detects the bug
- `test_word_positions_get_end_minus_one_safety` - Tests edge cases
- `test_decode_entities_loop_bounds` - Comprehensive bounds checking
- `test_span_definition_consistency` - Verifies generate_spans and decode_entities consistency
- `test_empty_word_positions_edge_case` - Defensive empty check
- `test_end_equals_words_len_boundary` - Boundary condition test

**Impact**: 
- **Severity**: Medium - Can cause panic or incorrect entity extraction when processing text with spans near the end
- **Frequency**: Occurs when extracting entities from text where spans include the last word(s)

---

## Bug 2: Potential Issue in encoder_candle.rs::geglu with Empty Dimensions

**Location**: `src/backends/encoder_candle.rs:464`

**Issue**: 
```rust
let dim = x.dims().last().copied().unwrap_or(0);
let half = dim / 2;
```

If `x.dims()` is empty (shouldn't happen for valid tensors), `dim = 0` and `half = 0`, which might cause issues in the subsequent indexing.

**Analysis**: 
- Tensors should always have at least one dimension, so this is likely safe
- However, defensive code would validate `x.dims().is_empty()` before accessing
- The `.unwrap_or(0)` fallback might mask a real problem

**Status**: Low priority - likely safe but worth documenting

**Recommendation**: Add explicit validation:
```rust
let dims = x.dims();
if dims.is_empty() {
    return Err(Error::Parse("Tensor has no dimensions".into()));
}
let dim = dims.last().copied().unwrap();
let half = dim / 2;
```

---

## Remaining Issues to Investigate

### Issue 1: Word Position Calculation with Repeated Words
**Location**: Multiple files (nuner.rs, w2ner.rs, gliner2.rs, gliner_candle.rs, candle.rs, session_pool.rs, coref_t5.rs)

**Issue**: Using `text[pos..].find(word)` finds the first occurrence after `pos`, not necessarily the correct occurrence for the tokenized sequence.

**Status**: Documented in `SUBTLE_BUGS_FOUND.md` as Bug 1, not yet fixed

### Issue 2: Word Position Vector Length Mismatch
**Location**: Same files as Issue 1

**Issue**: If a word is not found, the code continues but doesn't add a position to `word_positions`. Later code uses `.get()` which returns `None`, causing entities to be silently skipped.

**Status**: Documented in `SUBTLE_BUGS_FOUND.md` as Bug 2, not yet fixed. Note: `gliner_candle.rs` now returns an error if word is not found (line 595), which is better behavior.

---

## Test Coverage

All bugs found have corresponding tests in:
- `tests/gliner_candle_bug_tests.rs` - GLiNERCandle-specific bugs
- `tests/subtle_bugs_tests.rs` - General ML backend bugs
- `tests/offset_bug_tests.rs` - Offset conversion bugs

