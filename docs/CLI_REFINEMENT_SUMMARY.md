# CLI Refinement Summary

## Current State

✅ **Completed**: All new CLI commands implemented and tested
- `enhance` - Incremental building
- `pipeline` - Unified workflow
- `query` - Entity/cluster querying
- `compare` - Document/model comparison
- `cache` - Cache management
- `config` - Configuration management
- `batch` - Batch processing

## Key Refinements Identified

### 1. **BackendFactory Integration** (High Priority)

**Current**: CLI uses `ModelBackend::create_model()` which duplicates logic

**Refined**: Use `BackendFactory::create()` for consistent backend creation

```rust
// Current (in src/bin/anno.rs)
let model = args.model.create_model()?;

// Refined
use anno::eval::backend_factory::BackendFactory;
let model = BackendFactory::create(&args.model.to_string())?;
```

**Benefits**:
- Single source of truth for backend creation
- Better error messages (factory knows feature requirements)
- Support for backend aliases
- Consistent with evaluation framework

### 2. **Module Organization** (Medium Priority)

**Current**: All CLI code in `src/bin/anno.rs` (5000+ lines)

**Refined**: Extract to library modules

```
src/
├── cli/
│   ├── mod.rs              # Re-exports
│   ├── commands/           # Command implementations
│   │   ├── mod.rs
│   │   ├── extract.rs
│   │   ├── enhance.rs
│   │   ├── pipeline.rs
│   │   └── ...
│   ├── cache/              # Cache management
│   │   └── manager.rs
│   └── config/             # Config management
│       └── loader.rs
└── bin/
    └── anno.rs             # Thin entry point
```

**Benefits**:
- Testable (library code can be unit tested)
- Reusable (other binaries can use CLI functionality)
- Maintainable (smaller, focused modules)

### 3. **Error Handling Consistency** (Medium Priority)

**Current**: CLI uses `Result<(), String>`, library uses `Result<T, Error>`

**Refined**: CLI commands return library `Error` type

```rust
// Current
fn cmd_extract(args: ExtractArgs) -> Result<(), String> {
    // ...
}

// Refined
use anno::Result;
fn cmd_extract(args: ExtractArgs) -> Result<()> {
    // ...
}
```

**Benefits**:
- Consistent error types
- Better error messages (structured errors)
- Easier debugging (error context)

### 4. **GroundedDocument Method Utilization** (Low Priority)

**Current**: CLI manually builds GroundedDocuments

**Refined**: Use GroundedDocument's helper methods

```rust
// Current: Manual signal collection
let signal_ids: Vec<u64> = doc.signals().iter().map(|s| s.id).collect();

// Refined: Use existing methods
let signals = doc.confident_signals(0.5);
let signals = doc.signals_with_label("PER");
```

**Benefits**:
- Less code duplication
- Consistent behavior
- Easier to maintain

### 5. **Evaluation Framework Integration** (Low Priority)

**Current**: Pipeline/batch commands don't leverage evaluation framework

**Refined**: Integrate with `TaskEvaluator` for metrics

```rust
// In pipeline command
if args.evaluate {
    let evaluator = TaskEvaluator::new()?;
    let metrics = evaluator.evaluate_single_document(&doc, Task::NER, None)?;
    report_metrics(&metrics);
}
```

**Benefits**:
- Consistent metrics
- Access to advanced features (bias, calibration)
- Better reporting

## Implementation Priority

### Phase 1: Quick Wins (1-2 hours)
1. ✅ Replace `ModelBackend::create_model()` with `BackendFactory::create()`
2. ✅ Update error handling to use library `Error` type
3. ✅ Use GroundedDocument helper methods where applicable

### Phase 2: Refactoring (4-6 hours)
4. Extract commands to `src/cli/commands/`
5. Extract cache/config to `src/cli/cache/` and `src/cli/config/`
6. Update `src/bin/anno.rs` to use library modules

### Phase 3: Integration (2-4 hours)
7. Integrate with evaluation framework for metrics
8. Use evaluation framework's progress reporting
9. Integrate config with evaluation framework

## Immediate Action Items

1. **BackendFactory Integration** (30 min)
   - Replace all `ModelBackend::create_model()` calls
   - Update error messages to use factory errors
   - Test with all backends

2. **Error Handling** (30 min)
   - Change command signatures to return `Result<()>`
   - Update error formatting in `main()`
   - Test error paths

3. **Module Extraction** (2-3 hours)
   - Create `src/cli/` structure
   - Move one command at a time
   - Update imports and tests

## Testing Strategy

- **Unit Tests**: Test library modules independently
- **Integration Tests**: Test CLI commands end-to-end
- **Regression Tests**: Ensure existing functionality unchanged

## Backward Compatibility

All refinements maintain CLI interface:
- Command names unchanged
- Arguments unchanged
- Output formats unchanged
- Only internal implementation changes

## Documentation Updates

- Update `docs/CLI_DESIGN_REFINEMENT.md` with architecture changes
- Add module-level docs to `src/cli/`
- Update examples to show new patterns

