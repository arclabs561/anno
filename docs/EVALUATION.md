# Evaluation Guide

Comprehensive evaluation framework for NER and coreference resolution.

## Quick Start

```rust
use anno::eval::{ReportBuilder, TestCase, SimpleGoldEntity};
use anno::PatternNER;

let model = PatternNER::new();

// Option 1: Built-in synthetic data
let report = ReportBuilder::new("PatternNER").build(&model);

// Option 2: Custom test data
let tests = vec![
    TestCase {
        text: "Meeting on March 15".into(),
        gold_entities: vec![
            SimpleGoldEntity {
                text: "March 15".into(),
                entity_type: "DATE".into(),
                start: 11,
                end: 19,
            },
        ],
    },
];

let report = ReportBuilder::new("PatternNER")
    .with_test_data(tests)
    .with_error_analysis(true)
    .build(&model);

println!("{}", report.summary());
```

## Metrics

### F1 Score Variants

| Metric | Meaning | When to Use |
|--------|---------|-------------|
| **Micro F1** | Aggregate TP/FP/FN across all types | Overall performance |
| **Macro F1** | Average F1 per type (equal weight) | Fairness to rare types |
| **Weighted F1** | F1 weighted by type frequency | Realistic expectation |

### Evaluation Modes (SemEval-2013)

| Mode | Boundary | Type | Use Case |
|------|----------|------|----------|
| **Strict** | Exact | Exact | CoNLL standard (default) |
| **Exact** | Exact | Any | Boundary detection quality |
| **Partial** | Overlap | Exact | Lenient evaluation |
| **Type** | Any | Exact | Type classification only |

```rust
use anno::eval::modes::{EvalMode, MultiModeResults};

let results = MultiModeResults::compute(&predicted, &gold);
println!("Strict F1: {:.1}%", results.strict.f1 * 100.0);
println!("Partial F1: {:.1}%", results.partial.f1 * 100.0);
```

## Coreference Metrics

For coreference resolution tasks (requires `eval` feature):

| Metric | Focus | Notes |
|--------|-------|-------|
| **MUC** | Links | Ignores singletons |
| **B³** | Mentions | Per-mention scores |
| **CEAF** | Entities | Optimal alignment |
| **LEA** | Links+Entities | Entity-aware |
| **BLANC** | Rand index | Best discriminative power |
| **CoNLL F1** | Composite | Average of MUC, B³, CEAF-e |

```rust
use anno::eval::{CorefChain, Mention, conll_f1};

let gold = vec![
    CorefChain::new(0, vec![
        Mention::new("John", 0, 4),
        Mention::new("he", 20, 22),
    ]),
];
let pred = gold.clone();

let (p, r, f1) = conll_f1(&gold, &pred);
```

## BIO Tag Adapter

Convert between BIO-tagged sequences and entity spans:

```rust
use anno::eval::bio_adapter::{bio_to_entities, validate_bio_sequence, BioScheme};

let tokens = ["John", "Smith", "works", "at", "Apple"];
let tags = ["B-PER", "I-PER", "O", "O", "B-ORG"];

let entities = bio_to_entities(&tokens, &tags, BioScheme::IOB2)?;
assert_eq!(entities[0].text, "John Smith");

// Validate sequence
let errors = validate_bio_sequence(&tags, BioScheme::IOB2);
```

## Dataset Support

With `--features network`:

```rust
use anno::eval::loader::{DatasetLoader, DatasetId};

let loader = DatasetLoader::new()?;
let dataset = loader.load_or_download(DatasetId::WikiGold)?;
println!("Loaded {} examples", dataset.sentences.len());
```

| Dataset | Domain | Size | Entity Types |
|---------|--------|------|--------------|
| WikiGold | Wikipedia | ~3.5k | PER, LOC, ORG, MISC |
| WNUT-17 | Social Media | ~2k | Emerging entities |
| CoNLL-2003 | News | ~20k | PER, LOC, ORG, MISC |
| MIT Movie | Movies | ~10k | Actor, Director, Genre |
| MIT Restaurant | Food | ~8k | Cuisine, Location, Price |
| BC5CDR | Biomedical | ~28k | Disease, Chemical |
| FewNERD | Cross-domain | ~188k | 8 coarse + 66 fine types |

### TypeMapper for Domain Datasets

Domain datasets use different entity schemas. TypeMapper normalizes them:

```rust
use anno::{TypeMapper, EntityType};

// MIT Movie: "ACTOR" → Person, "DIRECTOR" → Person
let mapper = TypeMapper::mit_movie();

// Or custom mappings
let mut mapper = TypeMapper::new();
mapper.add("ACTOR", EntityType::Person);
mapper.add("TITLE", EntityType::custom("WORK_OF_ART", EntityCategory::Creative));
```

## Bias Analysis

With `--features eval-bias`:

```rust
use anno::eval::{GenderBiasEvaluator, DemographicBiasEvaluator};

let evaluator = GenderBiasEvaluator::new();
let results = evaluator.evaluate(&model)?;
println!("Gender gap: {:.1}%", results.gap * 100.0);
```

| Module | Purpose |
|--------|---------|
| `GenderBiasEvaluator` | WinoBias-style occupation tests |
| `DemographicBiasEvaluator` | Ethnicity/region fairness |
| `TemporalBiasEvaluator` | Name popularity over time |
| `LengthBiasEvaluator` | Entity length sensitivity |

## Advanced Evaluation

With `--features eval-advanced`:

```rust
use anno::eval::{CalibrationEvaluator, RobustnessEvaluator, ThresholdAnalyzer};

// Confidence calibration (ECE, Brier score)
let cal = CalibrationEvaluator::new();
let results = cal.evaluate(&model, &test_data)?;

// Perturbation robustness
let rob = RobustnessEvaluator::new();
let results = rob.evaluate(&model, &test_data)?;

// Precision-recall curves
let analyzer = ThresholdAnalyzer::new();
let curve = analyzer.analyze(&predictions)?;
```

| Module | Purpose |
|--------|---------|
| `CalibrationEvaluator` | Confidence reliability |
| `RobustnessEvaluator` | Perturbation tolerance |
| `ErrorAnalyzer` | Boundary vs type errors |
| `ThresholdAnalyzer` | P/R tradeoff curves |
| `ActiveLearner` | Uncertainty sampling |
| `LongTailAnalyzer` | Rare entity performance |

## CLI Tool

```bash
cargo install --path . --features eval

# Quick evaluation
anno-eval quick

# Validate BIO sequence
anno-eval bio validate "B-PER I-PER O B-ORG"

# Repair invalid sequence
anno-eval bio repair "O I-PER I-PER O"

# Calculate span overlap
anno-eval overlap 0 10 5 15
```

## Statistical Significance

```rust
use anno::eval::analysis::compare_ner_systems;

let test = compare_ner_systems(
    "ModelA", &scores_a,
    "ModelB", &scores_b,
);

if test.significant_05 {
    println!("Difference is significant (p<0.05)");
}
```

## Error Analysis

```rust
use anno::eval::analysis::{ErrorAnalysis, ConfusionMatrix};

let analysis = ErrorAnalysis::analyze(&text, &predicted, &gold);
println!("{}", analysis.summary());

// Confusion matrix
let mut matrix = ConfusionMatrix::new();
// ... populate ...
let confused = matrix.most_confused(3);
```

## Feature Flag Summary

| Feature | Modules Added |
|---------|---------------|
| `eval` | Core metrics, BIO adapter, coreference, datasets |
| `eval-bias` | Gender, demographic, temporal, length bias |
| `eval-advanced` | Calibration, robustness, active learning, error analysis |
| `eval-full` | All of the above |

## Research

Based on evaluation methodologies from:
- [SemEval-2013 Task 9](https://aclanthology.org/S13-2056/) — Evaluation modes
- [CoNLL-2012](https://aclanthology.org/W12-4501/) — Coreference metrics
- [WinoBias](https://arxiv.org/abs/1804.06876) — Gender bias evaluation

