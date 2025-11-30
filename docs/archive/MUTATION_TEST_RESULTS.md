# Mutation Testing Results

## Summary

Mutation testing was run on `src/entity.rs` using `cargo-mutants`. Results show:

- **Total mutants tested**: 120
- **Caught (good)**: 80 (66.7%)
- **Missed (needs attention)**: 7 (5.8%)
- **Timeout**: 9 (7.5%)
- **Unviable**: 24 (20.0%)

## Test Coverage Quality

**66.7% kill rate** - This indicates reasonably good test coverage, but there are gaps that need attention.

## Missed Mutants (Critical - Need Tests)

These mutations were not caught by tests, indicating missing or insufficient test coverage:

### 1. `Entity::set_visual_span` (line 2094)
**Mutation**: Replace function body with `()`
**Issue**: No test verifies that `set_visual_span` actually sets the visual span
**Recommendation**: Add test that:
- Creates an entity
- Calls `set_visual_span` with a span
- Verifies `entity.visual_span` is set correctly

### 2. `Span::is_empty` (line 1167)
**Mutation**: Replace return value with `false`
**Issue**: No test checks the `is_empty` method behavior
**Recommendation**: Add tests for:
- Empty span (len == 0) should return true
- Non-empty span should return false

### 3. `DiscontinuousSpan::to_span` (line 1288)
**Mutation**: Replace return value with `None`
**Issue**: No test verifies `to_span` returns `Some` for contiguous spans
**Recommendation**: Add test that:
- Creates a contiguous `DiscontinuousSpan`
- Calls `to_span()`
- Verifies it returns `Some(Span)` with correct values

### 4. `TypeMapper::manufacturing` (line 613)
**Mutation**: Replace function body with `Default::default()`
**Issue**: No test verifies the manufacturing type mapper is correctly initialized
**Recommendation**: Add test that verifies the manufacturing mapper contains expected entity types

### 5. `Entity::validate` boundary checks (line 2351)
**Mutations**: 
- Replace `>` with `==`
- Replace `>` with `>=`
**Issue**: Boundary condition tests may be missing
**Recommendation**: Add tests for:
- Edge cases where `start == end`
- Cases where `start > end` (should fail validation)
- Boundary values at limits

### 6. `TypeMapper::social_media` (line 589)
**Mutation**: Replace function body with `Default::default()`
**Issue**: No test verifies the social_media type mapper
**Recommendation**: Add test similar to manufacturing mapper

## Timeout Mutants (9)

These mutants caused tests to timeout. May indicate:
- Tests that are too slow
- Infinite loops in mutated code
- Tests that need optimization

**Recommendation**: Review timeout mutants to see if they reveal performance issues or if timeouts need to be increased.

## Unviable Mutants (24)

These couldn't be tested due to:
- Compilation errors in mutated code
- Type system constraints
- Mutations that don't make semantic sense

This is normal - not all mutations are viable.

## Recommendations

### Immediate Actions

1. **Add tests for missed mutants** (Priority: HIGH)
   - Focus on the 7 missed mutants listed above
   - These represent real gaps in test coverage

2. **Review timeout mutants** (Priority: MEDIUM)
   - Check if timeouts are legitimate or if tests need optimization
   - Consider increasing timeout for specific test cases if needed

3. **Improve test coverage** (Priority: MEDIUM)
   - Current 66.7% kill rate is decent but could be improved
   - Target: 80%+ kill rate

### Test Additions Needed

```rust
// Example tests to add:

#[test]
fn test_set_visual_span() {
    let mut entity = Entity::new(...);
    let span = Span::new(...);
    entity.set_visual_span(span.clone());
    assert_eq!(entity.visual_span, Some(span));
}

#[test]
fn test_span_is_empty() {
    let empty_span = Span::new(0, 0);
    assert!(empty_span.is_empty());
    
    let non_empty = Span::new(0, 5);
    assert!(!non_empty.is_empty());
}

#[test]
fn test_discontinuous_span_to_span_contiguous() {
    let segments = vec![0..5];
    let dspan = DiscontinuousSpan::new(segments);
    let span = dspan.to_span();
    assert!(span.is_some());
    assert_eq!(span.unwrap().start, 0);
    assert_eq!(span.unwrap().end, 5);
}

#[test]
fn test_entity_validate_boundary_conditions() {
    // Test start == end
    // Test start > end (should fail)
    // Test valid ranges
}
```

## Files Analyzed

- `src/entity.rs` - 120 mutants tested

## Next Steps

1. Review this report
2. Add tests for missed mutants
3. Re-run mutation testing to verify improvements
4. Consider expanding mutation testing to other critical modules

## Fast Iteration Setup

To run mutation tests faster without long hanging tests:

### Quick Commands (via justfile)

```bash
# Fast mutation test on entity.rs only (30s timeout)
just mutants-fast

# Test specific file
just mutants-file src/backends/heuristic.rs

# List mutants without running (quick check)
just mutants-list

# Full mutation test (slower, comprehensive)
just mutants-all
```

### Configuration

Mutation testing is configured in `.cargo/mutants.toml`:
- **Minimum timeout**: 30 seconds (faster iteration)
- **Timeout multiplier**: 2.0x baseline
- **Excludes**: test files, examples, binaries
- **Copy target**: Enabled for faster builds

### Strategy for Fast Iteration

1. **Start small**: Use `just mutants-fast` to test one file at a time
2. **Target critical paths**: Focus on `src/entity.rs` and core backends
3. **Incremental improvement**: Add tests for missed mutants, re-run
4. **Full validation**: Run `just mutants-all` before releases

### Tests Added

The following tests have been added to address missed mutants:

1. ✅ `test_entity_set_visual_span` - Verifies `set_visual_span` actually sets the span
2. ✅ `test_span_is_empty` - Tests empty span detection for both text and visual spans
3. ✅ `test_discontinuous_span_to_span_contiguous` - Tests conversion to regular span
4. ✅ `test_type_mapper_manufacturing` - Verifies manufacturing mapper initialization
5. ✅ `test_type_mapper_social_media` - Verifies social media mapper initialization
6. ✅ `test_entity_validate_boundary_conditions` - Tests boundary edge cases (end == char_count, end > char_count)

These tests should improve the kill rate from 66.7% to ~75%+.

