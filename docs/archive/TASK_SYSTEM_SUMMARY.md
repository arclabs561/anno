# Task-Dataset-Backend Mapping System - Summary

## Overview

Created a comprehensive, trait-based system for mapping tasks, datasets, and backends with many-to-many relationships.

## ✅ Completed Components

### 1. Core Mapping System (`src/eval/task_mapping.rs`)
- **Task enum**: 11 tasks (NER, NED, RelationExtraction, IntraDocCoref, InterDocCoref, AbstractAnaphora, DiscontinuousNER, EventExtraction, TextClassification, HierarchicalExtraction)
- **Trait-based capability detection**: Backend capabilities determined by trait implementations
- **Many-to-many mappings**: 
  - `dataset_tasks()`: Dataset → Tasks
  - `task_datasets()`: Task → Datasets
  - `backend_tasks()`: Backend → Tasks
  - `get_task_backends()`: Task → Backends
- **TaskMapping struct**: Comprehensive mapping with all relationships

### 2. Evaluation Framework (`src/eval/task_evaluator.rs`)
- **TaskEvaluator**: Runs evaluations across all valid task-dataset-backend combinations
- **TaskEvalConfig**: Configurable evaluation settings
- **ComprehensiveEvalResults**: Results with summary statistics
- **Markdown report generation**: `to_markdown()` method

### 3. Tests (`tests/task_mapping.rs`)
- Tests for task-dataset mappings
- Tests for backend capabilities
- GLiNER2 multi-task capability verification
- Coreference dataset mappings

### 4. Documentation
- `docs/TASK_DATASET_MAPPING.md`: Comprehensive documentation
- Usage examples and design philosophy

### 5. Dataset Additions
- **CADEC**: Added to `DatasetId` enum for discontinuous NER
- Download URL: `KevinSpaghetti/cadec` on HuggingFace
- Supports: DiscontinuousNER, NER tasks

## GLiNER2 Multi-Task Capabilities

GLiNER2 supports multiple tasks via trait implementations:

| Trait | Task | Status |
|-------|------|--------|
| `Model` | NER | ✅ Implemented |
| `ZeroShotNER` | Zero-shot NER | ✅ Implemented |
| `RelationExtractor` | Relation Extraction | ✅ Implemented (heuristic-based) |
| (Future) | Text Classification | ⏳ Schema support exists |
| (Future) | Hierarchical Extraction | ⏳ Schema support exists |

**Current Implementation**:
- NER: Fully working
- Relation Extraction: Heuristic-based (extracts entities first, then infers relations)
- Text Classification: Schema support in GLiNER2, needs dataset integration
- Hierarchical Extraction: Schema support in GLiNER2, needs dataset integration

## Task-Dataset Mappings

### NER (25 datasets)
- WikiGold, WNUT-17, MIT Movie, MIT Restaurant
- CoNLL-2003, OntoNotes, MultiNERD
- BC5CDR, NCBI Disease, GENIA, AnatEM, BC2GM, BC4CHEMD
- TweetNER7, BroadTwitterCorpus, FabNER
- FewNERD, CrossNER, UniversalNERBench
- WikiANN, MultiCoNER, MultiCoNERv2, WikiNeural, PolyglotNER, UniversalNER

### Discontinuous NER (1 dataset)
- CADEC (Clinical Adverse Drug Events)

### Relation Extraction (2 datasets)
- DocRED, Re-TACRED

### Intra-document Coreference (3 datasets)
- GAP, PreCo, LitBank

### Abstract Anaphora (3 datasets)
- GAP, PreCo, LitBank (same as coreference, but different evaluation)

## Backend Capabilities

| Backend | Tasks Supported | Traits |
|---------|----------------|--------|
| PatternNER | NER (structured only) | `Model` |
| HeuristicNER | NER | `Model` |
| StackedNER | NER | `Model` |
| BertNEROnnx | NER | `Model` |
| CandleNER | NER | `Model` |
| NuNER | NER | `Model`, `ZeroShotNER` |
| GLiNEROnnx | NER | `Model`, `ZeroShotNER` |
| GLiNERCandle | NER | `Model`, `ZeroShotNER` |
| **GLiNER2** | **NER, RE, Classification, Hierarchical** | `Model`, `ZeroShotNER`, `RelationExtractor` |
| W2NER | NER, DiscontinuousNER | `Model`, `DiscontinuousNER` |
| CorefResolver | IntraDocCoref, AbstractAnaphora | `CoreferenceResolver` |

## Usage Examples

### Query Mappings

```rust
use anno::eval::task_mapping::{get_task_datasets, get_task_backends, Task};

// Get all datasets for NER
let ner_datasets = get_task_datasets(Task::NER);

// Get all backends that support relation extraction
let re_backends = get_task_backends(Task::RelationExtraction);
// Returns: ["gliner2"]
```

### Comprehensive Evaluation

```rust
use anno::eval::task_evaluator::{TaskEvaluator, TaskEvalConfig};
use anno::eval::task_mapping::Task;

let evaluator = TaskEvaluator::new()?;
let config = TaskEvalConfig {
    tasks: vec![Task::NER, Task::RelationExtraction],
    datasets: vec![], // All suitable
    backends: vec![], // All compatible
    max_examples: Some(100),
    require_cached: false,
};

let results = evaluator.evaluate_all(config)?;
println!("{}", results.to_markdown());
```

## Integration Status

✅ **Compiles**: All code compiles with `eval-advanced` feature
✅ **Tests**: Test suite created and ready
✅ **Documentation**: Comprehensive docs in place
⏳ **Evaluation Implementation**: Placeholder for actual metric computation (needs backend factory)

## Next Steps

1. **Backend Factory**: Create backend instances from names for evaluation
2. **Metric Computation**: Implement task-specific metrics in `TaskEvaluator`
3. **CADEC Parser**: Add JSONL parsing for CADEC dataset
4. **ShARe13/14**: Add support when public versions found
5. **ACE 2005**: Document LDC requirement or find processed version
6. **NED Datasets**: Add AIDA, TAC-KBP when available
7. **Event Extraction**: Add ACE 2005 event extraction support

## Files Created/Modified

### New Files
- `src/eval/task_mapping.rs` - Core mapping system
- `src/eval/task_evaluator.rs` - Evaluation framework
- `tests/task_mapping.rs` - Test suite
- `examples/task_evaluation.rs` - Usage example
- `docs/TASK_DATASET_MAPPING.md` - Documentation

### Modified Files
- `src/eval/mod.rs` - Added task_mapping and task_evaluator modules
- `src/eval/loader.rs` - Added CADEC dataset support
- `Cargo.toml` - Added task_evaluation example

