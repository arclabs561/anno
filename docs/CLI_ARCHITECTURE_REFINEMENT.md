# CLI Architecture Refinement

## Repository Analysis

After reviewing the codebase structure with `eza -T`, key observations:

### Current Structure
- **`src/backends/`**: 30+ backend implementations (Regex, Heuristic, GLiNER, BERT, etc.)
- **`src/eval/`**: Comprehensive evaluation framework (60+ modules)
- **`src/bin/anno.rs`**: CLI implementation (5000+ lines, growing)
- **`src/grounded.rs`**: Core data structure (GroundedDocument)
- **`tests/`**: 116 test files with extensive coverage

### Key Capabilities Available
1. **Evaluation Framework**: `TaskEvaluator`, `EvalSystem`, `UnifiedEvaluator`
2. **Backend Factory**: `BackendFactory` for consistent backend creation
3. **Task Mapping**: Automatic task-dataset-backend mapping
4. **Progress Reporting**: Built-in progress tracking in evaluators
5. **Config Builders**: Structured configuration system

## Refinement Opportunities

### 1. Library/CLI Separation

**Current**: All CLI logic in `src/bin/anno.rs` (5000+ lines)

**Refined**: Extract reusable functionality to library modules

```
src/
├── cli/                    # NEW: CLI-specific modules
│   ├── mod.rs
│   ├── commands/          # Command implementations
│   │   ├── extract.rs
│   │   ├── enhance.rs
│   │   ├── pipeline.rs
│   │   ├── query.rs
│   │   ├── compare.rs
│   │   ├── cache.rs
│   │   ├── config.rs
│   │   └── batch.rs
│   ├── cache/             # Cache management
│   │   ├── mod.rs
│   │   └── manager.rs
│   ├── config/            # Config management
│   │   ├── mod.rs
│   │   └── loader.rs
│   └── progress/           # Progress reporting
│       ├── mod.rs
│       └── reporter.rs
└── bin/
    └── anno.rs            # Thin CLI entry point (arg parsing + dispatch)
```

**Benefits**:
- Testable: Library code can be unit tested
- Reusable: Other binaries can use CLI functionality
- Maintainable: Smaller, focused modules
- Documentable: Library docs can be generated

### 2. Integration with Evaluation Framework

**Current**: CLI commands don't leverage evaluation framework

**Refined**: Pipeline/batch commands integrate with `TaskEvaluator`

```rust
// In src/cli/commands/pipeline.rs
use anno::eval::task_evaluator::TaskEvaluator;
use anno::eval::task_mapping::Task;

pub fn run_pipeline(args: PipelineArgs) -> Result<(), String> {
    // ... extract entities ...
    
    // If evaluation requested, use evaluation framework
    if args.evaluate {
        let evaluator = TaskEvaluator::new()?;
        let metrics = evaluator.evaluate_single_document(
            &doc,
            Task::NER,
            args.gold_standard.as_ref()
        )?;
        
        // Report metrics using evaluation framework's reporting
        report_metrics(&metrics);
    }
    
    // ... rest of pipeline ...
}
```

**Benefits**:
- Consistent metrics across CLI and library
- Access to advanced evaluation features (bias, calibration, etc.)
- Better reporting (HTML, JSON, tables)

### 3. Backend Factory Integration

**Current**: CLI manually creates backends via `ModelBackend::create_model()`

**Refined**: Use `BackendFactory` for consistent backend creation

```rust
// In src/cli/commands/extract.rs
use anno::eval::backend_factory::BackendFactory;

pub fn run_extract(args: ExtractArgs) -> Result<(), String> {
    let factory = BackendFactory::new();
    let model = factory.create_backend(&args.model.to_string())?;
    
    // ... rest of extraction ...
}
```

**Benefits**:
- Consistent backend creation across CLI and library
- Better error messages (factory knows why backends fail)
- Support for backend aliases and auto-selection

### 4. Config System Integration

**Current**: Simple TOML config files

**Refined**: Integrate with evaluation framework's config builders

```rust
// In src/cli/config/loader.rs
use anno::eval::config_builder::EvalConfigBuilder;

pub fn load_config(name: &str) -> Result<PipelineConfig, String> {
    // Load TOML config
    let toml_config = load_toml(name)?;
    
    // Convert to evaluation framework config
    let eval_config = EvalConfigBuilder::new()
        .with_model(&toml_config.model)
        .with_tasks(toml_config.tasks)
        .build()?;
    
    // Merge with CLI defaults
    Ok(PipelineConfig {
        eval: eval_config,
        pipeline: toml_config.pipeline,
        cache: toml_config.cache,
    })
}
```

**Benefits**:
- Reuse evaluation framework's validation
- Consistent config format across CLI and library
- Access to advanced config features

### 5. Progress Reporting Integration

**Current**: Basic progress bars (when indicatif available)

**Refined**: Use evaluation framework's progress reporting

```rust
// In src/cli/progress/reporter.rs
use anno::eval::task_evaluator::ProgressReporter;

pub struct CLIProgressReporter {
    // Wraps evaluation framework's progress reporting
    inner: ProgressReporter,
    // Adds CLI-specific formatting
}

impl CLIProgressReporter {
    pub fn for_pipeline(&self, steps: usize) {
        // Use evaluation framework's progress system
        self.inner.start_task("pipeline", steps);
    }
}
```

**Benefits**:
- Consistent progress reporting
- Better integration with evaluation framework
- Access to detailed progress information

### 6. Error Handling Consistency

**Current**: CLI uses `Result<(), String>`, library uses `Result<T, Error>`

**Refined**: CLI commands return library `Error` type

```rust
// In src/cli/commands/mod.rs
use anno::Result;

pub fn run_extract(args: ExtractArgs) -> Result<()> {
    // ... returns library Error type ...
}

// In src/bin/anno.rs
fn main() -> ExitCode {
    match cli.command {
        Some(Commands::Extract(args)) => {
            match cli::commands::extract::run(args) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("error: {}", e);
                    ExitCode::FAILURE
                }
            }
        }
        // ...
    }
}
```

**Benefits**:
- Consistent error types
- Better error messages (structured errors)
- Easier debugging (error context)

### 7. GroundedDocument Method Utilization

**Current**: CLI manually builds GroundedDocuments

**Refined**: Use GroundedDocument's helper methods

```rust
// In src/cli/commands/enhance.rs
use anno::grounded::GroundedDocument;

pub fn run_enhance(args: EnhanceArgs) -> Result<(), String> {
    let mut doc = load_grounded_document(&args.input)?;
    
    // Use GroundedDocument's built-in methods
    if args.coref {
        // GroundedDocument could have a method for this
        doc.resolve_coreference()?;
    }
    
    if args.link_kb {
        doc.link_to_kb()?;
    }
    
    // Use GroundedDocument's export methods
    doc.export(&args.export, args.export_format)?;
    
    Ok(())
}
```

**Benefits**:
- Less code duplication
- Consistent behavior
- Easier to maintain

### 8. Query System Enhancement

**Current**: Simple filtering in query command

**Refined**: Leverage GroundedDocument's query methods

```rust
// In src/cli/commands/query.rs
use anno::grounded::GroundedDocument;

pub fn run_query(args: QueryArgs) -> Result<(), String> {
    let doc = load_grounded_document(&args.input)?;
    
    // Use GroundedDocument's built-in query methods
    let results = if let Some(typ) = &args.r#type {
        doc.signals_with_label(typ)
    } else if let Some(entity) = &args.entity {
        doc.find_entity(entity)
    } else if let Some(min_conf) = args.min_confidence {
        doc.confident_signals(min_conf as f32)
    } else {
        doc.signals()
    };
    
    // Format and output results
    format_results(&results, args.format)?;
    
    Ok(())
}
```

**Benefits**:
- Reuse existing query methods
- Consistent query behavior
- Less code to maintain

## Implementation Plan

### Phase 1: Extract Library Modules (High Priority)
1. Create `src/cli/` module structure
2. Move command implementations to `src/cli/commands/`
3. Extract cache/config to `src/cli/cache/` and `src/cli/config/`
4. Update `src/bin/anno.rs` to use library modules

### Phase 2: Integration (Medium Priority)
5. Integrate with `BackendFactory`
6. Integrate with `TaskEvaluator` for evaluation
7. Use evaluation framework's progress reporting
8. Integrate config with evaluation framework

### Phase 3: Enhancement (Low Priority)
9. Add GroundedDocument helper methods
10. Enhance query system
11. Improve error handling consistency
12. Add comprehensive documentation

## Migration Strategy

1. **Backward Compatible**: All changes maintain CLI interface
2. **Incremental**: Move one command at a time
3. **Tested**: Each move includes tests
4. **Documented**: Update docs as we go

## Benefits Summary

- **Maintainability**: Smaller, focused modules
- **Testability**: Library code can be unit tested
- **Consistency**: Reuse evaluation framework patterns
- **Extensibility**: Easier to add new commands
- **Documentation**: Library docs can be generated
- **Reusability**: Other binaries can use CLI functionality

