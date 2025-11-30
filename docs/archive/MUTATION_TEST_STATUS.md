# Mutation Testing Status

## Summary

✅ **Tests Added**: Successfully added tests for missed mutants:
- `test_entity_set_visual_span` - Verifies `set_visual_span` sets the visual span
- `test_span_is_empty` - Tests empty span detection  
- `test_discontinuous_span_to_span_contiguous` - Tests conversion to regular span
- `test_entity_validate_boundary_conditions` - Tests boundary edge cases
- TypeMapper tests (indirect verification)

✅ **Configuration**: Created `.cargo/mutants.toml` with optimized settings:
- Minimum timeout: 60 seconds
- Timeout multiplier: 2.0x
- Copy target: Enabled for faster builds
- Cap lints: Enabled

✅ **Justfile Commands**: Added convenient commands:
- `just mutants-fast` - Fast mutation test on entity.rs
- `just mutants-file FILE` - Test specific file
- `just mutants-list` - List mutants without running
- `just mutants-all` - Full mutation test

## Current Issue

⚠️ **Compilation Requirement**: The codebase requires `eval-advanced` feature to compile:
- `src/eval/task_evaluator.rs` uses `load_or_download()` which is `#[cfg(feature = "eval-advanced")]`
- Mutation tests must run with `--features "eval-advanced"`

## Working Command

To run mutation tests successfully, use:

```bash
# With eval-advanced feature (required for compilation)
cargo mutants --file "src/entity.rs" \
  --timeout 120 \
  --minimum-test-timeout 60 \
  --features "eval-advanced"
```

**Note**: Baseline tests may take 60-120 seconds with `eval-advanced` due to additional dependencies.

## Alternative: Skip Baseline

If baseline tests timeout, you can skip them (not recommended for first run):

```bash
cargo mutants --file "src/entity.rs" \
  --timeout 120 \
  --minimum-test-timeout 60 \
  --features "eval-advanced" \
  --baseline skip
```

## Expected Results

With the new tests added, we expect:
- **Kill rate improvement**: From ~66.7% to ~75%+
- **Missed mutants reduced**: From 7 to ~3-4
- **Better coverage**: Boundary conditions and edge cases now tested

## Next Steps

1. Run mutation tests with `eval-advanced` feature enabled
2. Review results in `mutants.out/` directory
3. Add more tests for any remaining missed mutants
4. Iterate until kill rate reaches 80%+

## Files Modified

- `tests/die_hard.rs` - Added entity validation tests
- `tests/discontinuous_span_tests.rs` - Added span conversion tests
- `tests/type_mapper_tests.rs` - Added TypeMapper tests
- `tests/bounds_validation.rs` - Added boundary condition tests
- `.cargo/mutants.toml` - Mutation test configuration
- `justfile` - Added mutation test commands

