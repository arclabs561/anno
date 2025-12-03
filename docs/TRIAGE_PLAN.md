# Code Quality Triage Plan

Based on comprehensive code review and expert guidance on Rust error handling.

## Phase 1: Quick Wins (COMPLETED ✓)

- [x] Improved CI caching (cache-all-crates for all jobs)
- [x] Removed empty directories
- [x] Verified hack/ directory documentation

## Phase 2: Error Handling (In Progress)

### Expert Guidance on unwrap()

**When unwrap() is acceptable:**
- Internal invariant violations (bugs in your code)
- Static regex compilation with expect("valid regex") - compile-time constants
- Test code
- Examples/documentation
- CLI binaries (more acceptable than libraries)

**When unwrap() should be replaced:**
- User-facing errors (file I/O, parsing user input)
- External data handling
- Library code that returns Result

### Action Items

1. **Review unwrap() in library code** - Most are in tests (acceptable)
2. **Static regex compilation** - Already using expect() with context (acceptable)
3. **CLI code** - Lower priority, but should improve error messages

## Phase 3: Critical Refactoring (Next)

### 1. Large Functions

**Target:** Functions >100 lines need splitting

**Strategy:**
- Extract helper functions
- Group related logic into modules
- Use builder patterns where appropriate

### 2. src/bin/anno.rs (6689 lines)

**Current structure:** Monolithic CLI file

**Target structure:**
```
src/bin/anno.rs (<100 lines - main entry point)
src/cli/
  ├── parser.rs (clap definitions)
  ├── commands/
  │   ├── extract.rs
  │   ├── debug.rs
  │   ├── eval.rs
  │   └── ...
  └── output.rs (formatting)
```

### 3. src/eval/mod.rs (119 public items)

**Target structure:**
```
src/eval/
  ├── mod.rs (re-exports only)
  ├── metrics/ (NER metrics, coref metrics)
  ├── datasets/ (loaders, adapters)
  ├── backends/ (backend evaluation)
  └── analysis/ (error analysis, calibration)
```

## Phase 4: Performance (Later)

- Audit clone() calls
- Optimize regex compilation (already using lazy_static)
- Pre-allocate vectors where possible

## Metrics

- **Current:** 6689 lines in main binary, 1750 line function, 119 public items in eval
- **Target:** <100 lines main, <50 lines per function, <20 public items per module

## Priority Order

1. CI improvements ✓
2. Error handling review
3. Split large functions
4. Refactor binary structure
5. Split eval module
6. Performance optimizations
