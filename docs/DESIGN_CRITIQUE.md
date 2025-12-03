# Design Critique: CLI Implementation Analysis

## Current Issues Identified

### 1. **Cross-Doc Command: Unnecessary Conversion Layer**

**Problem**: `cmd_crossdoc` converts `GroundedDocument` → `CDCR Document` → clusters → converts back. This is inefficient and loses information.

**Root Cause**: The command was built before `Corpus` existed. Now `Corpus::resolve_inter_doc_coref()` does exactly what we need:
- Works directly with `GroundedDocument`
- Uses tracks (Level 2) properly
- Creates `Identity` instances with proper `TrackRef`s
- No conversion overhead

**Impact**: 
- ~200 lines of unnecessary conversion code
- Loss of track/identity information
- Harder to maintain

**Solution**: Refactor to use `Corpus` directly.

### 2. **Extract Command: Manual Validation Instead of Library Methods**

**Problem**: `cmd_extract` manually validates signals and skips invalid ones, but `GroundedDocument::add_signal_validated()` exists and returns errors properly.

**Current Code**:
```rust
if let Some(err) = signal.validate_against(&text) {
    validation_errors.push(err);
} else {
    doc.add_signal(signal);
}
```

**Better Approach**:
```rust
match doc.add_signal_validated(signal) {
    Ok(id) => { /* success */ }
    Err(err) => validation_errors.push(err),
}
```

**Impact**: More consistent error handling, clearer intent.

### 3. **Extract Command: Not Using process_text Utility**

**Problem**: `process_text()` exists and does exactly what `cmd_extract` does (extract + validate), but we're reimplementing it.

**Trade-off**: `process_text()` doesn't support negation/quantifiers, so we can't use it directly. However, we could:
- Use it for the base extraction
- Add negation/quantifiers after
- Or extend `process_text()` to support these features

**Impact**: Code duplication, harder to maintain.

### 4. **Manual Stats Computation**

**Problem**: Some places compute stats manually (e.g., `doc.signals().len()`) instead of using `doc.stats()`.

**Impact**: Inconsistent stats, potential bugs if computation differs.

### 5. **Spatial Index Not Used**

**Problem**: `TextSpatialIndex` exists for efficient range queries, but `query` command uses linear scan.

**Impact**: Performance degradation for large documents.

## Design Principles Violated

1. **DRY (Don't Repeat Yourself)**: Manual validation, stats computation, entity extraction
2. **Use Library Abstractions**: Not using `Corpus`, `process_text`, `add_signal_validated`
3. **Performance**: Not using spatial index for queries
4. **Consistency**: Different validation approaches in different commands

## Recommendations

### High Priority
1. Refactor `cross-doc` to use `Corpus` - eliminates ~200 lines, better architecture
2. Use `add_signal_validated()` in extract - cleaner error handling
3. Use `doc.stats()` everywhere - consistency

### Medium Priority
4. Use `process_text()` where possible - reduce duplication
5. Add spatial index to query command - better performance

### Low Priority
6. Extend `process_text()` to support negation/quantifiers - then use it fully

