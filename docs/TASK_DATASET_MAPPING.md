# Task-Dataset-Backend Mapping System

## Overview

This document describes the comprehensive mapping system that connects:
- **Tasks** (NER, NED, Coreference, etc.)
- **Datasets** (CoNLL-2003, WikiGold, GAP, etc.)
- **Backends** (GLiNER, BERT, W2NER, etc.)

## Design Philosophy

### Trait-Based Capabilities

Backend capabilities are determined by **trait implementations**, not string matching:

- `Model` → All backends support NER
- `ZeroShotNER` → Zero-shot NER capability (GLiNER, NuNER, GLiNER2)
- `RelationExtractor` → Relation extraction (GLiNER2)
- `DiscontinuousNER` → Discontinuous NER (W2NER)
- `CoreferenceResolver` → Coreference resolution

### Many-to-Many Relationships

- **One task → Many datasets**: NER can be evaluated on CoNLL-2003, WikiGold, MultiNERD, etc.
- **One dataset → Many tasks**: GAP supports both IntraDocCoref and AbstractAnaphora
- **One backend → Many tasks**: GLiNER2 supports NER, RelationExtraction, TextClassification, HierarchicalExtraction
- **One task → Many backends**: NER can be evaluated with PatternNER, BERT, GLiNER, etc.

## Tasks

| Task | Description | Example Datasets | Compatible Backends |
|------|-------------|------------------|---------------------|
| **NER** | Named Entity Recognition | CoNLL-2003, WikiGold, MultiNERD | All backends (via `Model` trait) |
| **NED** | Named Entity Disambiguation | (TODO: AIDA, TAC-KBP) | (TODO: Entity linking backends) |
| **RelationExtraction** | Entity-relation-entity triples | DocRED, Re-TACRED | GLiNER2 (via `RelationExtractor` trait) |
| **IntraDocCoref** | Within-document coreference | GAP, PreCo, LitBank | CorefResolver implementations |
| **InterDocCoref** | Cross-document coreference | (TODO: ECB+, WikiCoref) | (TODO: Inter-doc resolvers) |
| **AbstractAnaphora** | Event/proposition anaphora | GAP, PreCo, LitBank | DiscourseAwareResolver |
| **DiscontinuousNER** | Non-contiguous entity spans | (TODO: CADEC, ShARe13/14) | W2NER (via `DiscontinuousNER` trait) |
| **EventExtraction** | Event triggers and arguments | (TODO: ACE 2005) | EventExtractor implementations |
| **TextClassification** | Text/document classification | (TODO: Classification datasets) | GLiNER2 |
| **HierarchicalExtraction** | Nested structure extraction | (TODO: Hierarchical datasets) | GLiNER2 |

## GLiNER2 Multi-Task Capabilities

GLiNER2 (arXiv:2507.18546) is a multi-task model that supports:

1. **NER**: Standard named entity recognition
2. **Text Classification**: Single/multi-label classification
3. **Hierarchical Extraction**: Nested structure extraction
4. **Relation Extraction**: Via GLiREL integration

This is represented in the mapping system as:
```rust
backend_tasks("gliner2") → [
    Task::NER,
    Task::TextClassification,
    Task::HierarchicalExtraction,
    Task::RelationExtraction,
]
```

## Dataset Task Support

Each dataset declares what tasks it can evaluate:

| Dataset | Tasks Supported |
|---------|----------------|
| WikiGold | NER |
| CoNLL-2003 | NER |
| GAP | IntraDocCoref, AbstractAnaphora |
| PreCo | IntraDocCoref, AbstractAnaphora |
| DocRED | RelationExtraction |
| Re-TACRED | RelationExtraction |
| (TODO: CADEC) | DiscontinuousNER, NER |

## Usage Examples

### Query Tasks for a Dataset

```rust
use anno::eval::task_mapping::{get_dataset_tasks, DatasetId};

let tasks = get_dataset_tasks(DatasetId::GAP);
// Returns: [Task::IntraDocCoref, Task::AbstractAnaphora]
```

### Query Datasets for a Task

```rust
use anno::eval::task_mapping::{get_task_datasets, Task};

let datasets = get_task_datasets(Task::NER);
// Returns: [DatasetId::WikiGold, DatasetId::CoNLL2003Sample, ...]
```

### Query Backends for a Task

```rust
use anno::eval::task_mapping::{get_task_backends, Task};

let backends = get_task_backends(Task::RelationExtraction);
// Returns: ["gliner2"]
```

### Comprehensive Evaluation

```rust
use anno::eval::task_evaluator::{TaskEvaluator, TaskEvalConfig};
use anno::eval::task_mapping::Task;

let evaluator = TaskEvaluator::new()?;
let config = TaskEvalConfig {
    tasks: vec![Task::NER, Task::RelationExtraction],
    datasets: vec![], // All suitable datasets
    backends: vec![], // All compatible backends
    max_examples: Some(100),
    require_cached: false,
};

let results = evaluator.evaluate_all(config)?;
println!("{}", results.to_markdown());
```

## Implementation Details

### Trait Detection

The system uses trait objects to detect backend capabilities at runtime:

```rust
pub fn detect_backend_capabilities<M: Model>(backend: &M) -> Vec<Task> {
    let mut tasks = vec![Task::NER]; // All backends implement Model
    
    // Check for RelationExtractor
    if let Some(_) = (backend as &dyn Any).downcast_ref::<dyn RelationExtractor>() {
        tasks.push(Task::RelationExtraction);
    }
    
    // ... check other traits
    tasks
}
```

### Static Mapping

For compile-time queries, we use static mappings based on known backend implementations:

```rust
pub fn backend_tasks(backend_name: &str) -> &'static [Task] {
    match backend_name {
        "gliner2" => &[
            Task::NER,
            Task::RelationExtraction,
            Task::TextClassification,
            Task::HierarchicalExtraction,
        ],
        "w2ner" => &[Task::NER, Task::DiscontinuousNER],
        // ...
    }
}
```

## Future Enhancements

1. **NED Support**: Add entity linking backends and datasets (AIDA, TAC-KBP)
2. **Event Extraction**: Add ACE 2005 dataset support
3. **Discontinuous NER**: Add CADEC, ShARe13/14 datasets
4. **Inter-doc Coref**: Add ECB+, WikiCoref datasets
5. **Dynamic Backend Registration**: Allow runtime backend registration
6. **Metric Computation**: Implement task-specific metric computation in `TaskEvaluator`

## Testing

Run tests with:
```bash
cargo test --features eval-advanced --test task_mapping
```

Test coverage:
- Task-dataset mappings
- Dataset-task mappings
- Backend-task mappings
- GLiNER2 multi-task capabilities
- W2NER discontinuous NER support

