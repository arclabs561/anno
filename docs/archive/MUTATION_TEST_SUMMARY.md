# Mutation Testing - Quick Summary

## ✅ What We Accomplished

1. **Fixed Compilation Errors**:
   - Fixed `task_evaluator.rs` to use `sentence.entities()` instead of non-existent method
   - Fixed `Provenance::GoldStandard` → `Provenance::ml("gold", 1.0)`
   - Fixed `SimpleCorefResolver::new()` to use `CorefConfig::default()`
   - All code now compiles with `eval-advanced` feature

2. **Added Tests for Missed Mutants**:
   - `test_entity_set_visual_span` - Tests `set_visual_span` method
   - `test_span_is_empty` - Tests empty span detection
   - `test_discontinuous_span_to_span_contiguous` - Tests span conversion
   - `test_entity_validate_boundary_conditions` - Tests boundary edge cases
   - TypeMapper tests (indirect verification)

3. **Configuration**:
   - `.cargo/mutants.toml` - 60s timeout, optimized settings
   - `just mutants-fast` - Quick command to run tests

## ⚠️ Current Limitation

Mutation testing with `eval-advanced` feature is **very slow** because:
- Many additional dependencies (ICU, URL parsing, etc.)
- Baseline test takes 60-120 seconds
- Each mutant test takes similar time
- For 120 mutants × 60s = ~2 hours minimum

## 🚀 Faster Alternatives

### Option 1: Test Without eval-advanced (if possible)
```bash
# This won't work because code requires eval-advanced to compile
```

### Option 2: Run Just a Few Mutants
```bash
# List mutants first
cargo mutants --file "src/entity.rs" --features eval-advanced --list

# Then test just first 5 mutants
cargo mutants --file "src/entity.rs" --features eval-advanced --timeout 120 --minimum-test-timeout 60 --max-mutants 5
```

### Option 3: Skip Baseline (if you're confident tests pass)
```bash
cargo mutants --file "src/entity.rs" --features eval-advanced --baseline skip --timeout 120
```

### Option 4: Use CI/CD
Run full mutation tests in CI where time doesn't matter as much.

## 📊 Expected Results

With the new tests added, we expect:
- **Kill rate**: ~75%+ (up from 66.7%)
- **Missed mutants**: Reduced from 7 to ~3-4
- **Better coverage**: Boundary conditions now tested

## ✅ Verification

All new tests compile and pass:
```bash
cargo test --test die_hard --test discontinuous_span_tests --test bounds_validation --lib
```

The mutation testing infrastructure is ready - it just needs time to run!

