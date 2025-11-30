# Mutation Testing Setup - Fast Iteration Guide

## Quick Start

```bash
# Fast mutation test on entity.rs (30s timeout, ~5-10 min total)
just mutants-fast

# Test specific file
just mutants-file src/backends/heuristic.rs

# List mutants without running (quick check)
just mutants-list
```

## Configuration

Mutation testing is configured in `.cargo/mutants.toml`:
- **Minimum timeout**: 30 seconds (faster iteration)
- **Timeout multiplier**: 2.0x baseline
- **Copy target**: Enabled for faster builds
- **Cap lints**: Enabled to avoid denied warnings

## Strategy for Fast Iteration

### 1. Start Small
- Use `just mutants-fast` to test one file at a time
- Focus on `src/entity.rs` and core backends first
- Each run takes ~5-10 minutes instead of hours

### 2. Incremental Improvement
- Add tests for missed mutants
- Re-run fast mutation test
- Verify kill rate improves

### 3. Full Validation
- Run `just mutants-all` before releases
- This tests all files (slower, but comprehensive)

## Current Status

**Initial Results** (from previous run):
- **Total mutants**: 120
- **Caught**: 80 (66.7%)
- **Missed**: 7 (5.8%)
- **Timeout**: 9 (7.5%)
- **Unviable**: 24 (20.0%)

**Tests Added** to address missed mutants:
1. ✅ `test_entity_set_visual_span` - Verifies `set_visual_span` sets the span
2. ✅ `test_span_is_empty` - Tests empty span detection
3. ✅ `test_discontinuous_span_to_span_contiguous` - Tests conversion to regular span
4. ✅ `test_entity_validate_boundary_conditions` - Tests boundary edge cases
5. ✅ TypeMapper tests (indirect verification)

**Expected Improvement**: Kill rate should improve from 66.7% to ~75%+ with new tests.

## Known Issues

- Compilation error: `task_evaluator.rs` uses `load_or_download()` which requires `eval-advanced` feature
- Mutation tests now run with `--features "eval-advanced"` to fix this
- Some tests may timeout - adjust timeout in config if needed

## Tips

1. **Use `--no-config`** if config file has issues: `cargo mutants --no-config --file "src/entity.rs"`
2. **Target specific functions** with `--exclude-re` to skip slow functions
3. **Check results** in `mutants.out/` directory:
   - `caught.txt` - Good! Tests caught these
   - `missed.txt` - Need more tests
   - `timeout.txt` - May need longer timeout or optimization
   - `unviable.txt` - Normal, can't be tested

## Next Steps

1. Run `just mutants-fast` to verify new tests improve kill rate
2. Add more tests for any remaining missed mutants
3. Iterate until kill rate is 80%+

