# Eval Module Architecture

This document describes the architecture of the `anno::eval` module, which provides
comprehensive evaluation tools for NER and coreference resolution systems.

## Module Overview

The eval module is organized into these major subsystems:

```
eval/
├── Core Metrics & Evaluators
│   ├── mod.rs              - Module exports, evaluate_ner_model()
│   ├── evaluator.rs        - NEREvaluator trait, StandardNEREvaluator
│   ├── metrics.rs          - Precision/Recall/F1 calculations
│   ├── modes.rs            - EvalMode (Strict/Exact/Partial/Type)
│   └── types.rs            - MetricValue, MetricWithVariance, GoalCheck
│
├── Data Loading & Datasets
│   ├── datasets.rs         - GoldEntity, ground truth types
│   ├── loader.rs           - DatasetLoader (WikiGold, WNUT-17, CoNLL-2003, etc.)
│   ├── synthetic.rs        - Synthetic test data generation
│   └── validation.rs       - Ground truth validation
│
├── Error Analysis
│   ├── analysis.rs         - ConfusionMatrix, ErrorAnalysis
│   ├── error_analysis.rs   - ErrorAnalyzer, ErrorCategory, ErrorReport
│   └── long_tail.rs        - Rare entity analysis
│
├── Bias & Fairness
│   ├── gender_bias.rs      - WinoBias-style gender bias evaluation
│   ├── demographic_bias.rs - Ethnicity, region, script bias
│   ├── temporal_bias.rs    - Name frequency by decade
│   └── length_bias.rs      - Entity length bias analysis
│
├── Calibration & Confidence
│   ├── calibration.rs      - ECE, MCE, Brier score, reliability diagrams
│   ├── threshold_analysis.rs - Precision-recall curves, optimal thresholds
│   └── ood_detection.rs    - Out-of-distribution detection
│
├── Production Monitoring
│   ├── drift.rs            - Temporal drift detection
│   ├── robustness.rs       - Perturbation testing
│   └── ensemble.rs         - Multi-model disagreement analysis
│
├── Dataset Analysis
│   ├── dataset_comparison.rs - Cross-domain analysis, JS divergence
│   ├── dataset_quality.rs    - Leakage detection, quality metrics
│   ├── learning_curve.rs     - Sample efficiency analysis
│   └── sampling.rs           - Stratified sampling utilities
│
├── Coreference Resolution
│   ├── coref.rs            - CorefChain, Mention types
│   ├── coref_metrics.rs    - MUC, B³, CEAF, LEA, BLANC, CoNLL F1
│   ├── coref_loader.rs     - GAP, PreCo dataset loading
│   └── coref_resolver.rs   - SimpleCorefResolver implementation
│
├── Few-Shot & Active Learning
│   ├── few_shot.rs         - Few-shot evaluation framework
│   └── active_learning.rs  - Annotation selection strategies
│
└── Orchestration
    ├── harness.rs          - EvalHarness for running evaluations
    └── benchmark.rs        - Benchmarking utilities
```

## Key Abstractions

### NEREvaluator Trait (evaluator.rs)

The central abstraction for NER evaluation:

```rust
pub trait NEREvaluator: Send + Sync {
    fn evaluate_test_case(
        &self,
        model: &dyn Model,
        text: &str,
        ground_truth: &[GoldEntity],
        test_case_id: Option<&str>,
    ) -> Result<NERQueryMetrics>;

    fn aggregate(&self, query_metrics: &[NERQueryMetrics]) -> Result<NERAggregateMetrics>;
    
    fn check_goals(
        &self,
        metrics: &NERAggregateMetrics,
        goals: &NERMetricGoals,
    ) -> Result<GoalCheckResult>;
}
```

### EvalHarness (harness.rs)

Orchestrates evaluation across multiple backends:

```rust
let harness = EvalHarness::with_defaults()?;
let results = harness.run_synthetic()?;
```

Features:
- Backend registration
- Parallel evaluation
- HTML report generation
- Breakdown by difficulty/domain

### MetricValue (types.rs)

Type-safe metric representation bounded to [0.0, 1.0]:

```rust
let precision = MetricValue::new(0.85);      // Clamps to bounds
let strict = MetricValue::try_new(0.85)?;    // Returns error if out of bounds
```

### DriftDetector (drift.rs)

Production monitoring for model performance drift:

```rust
let mut detector = DriftDetector::new(DriftConfig::default());
detector.log_prediction(timestamp, confidence, entity_type, entity_text);
let report = detector.analyze();
if report.drift_detected {
    // Take action
}
```

### ThresholdAnalyzer (threshold_analysis.rs)

Analyze precision-recall tradeoffs:

```rust
let analyzer = ThresholdAnalyzer::new(10);
let curve = analyzer.analyze(&predictions);
println!("Optimal threshold: {:.2} (F1: {:.1}%)", 
    curve.optimal_threshold, curve.optimal_f1 * 100.0);
```

## Data Flow

```
              ┌─────────────────┐
              │   DatasetLoader │
              │  (loader.rs)    │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  GoldEntity[]   │
              │ (datasets.rs)   │
              └────────┬────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
         ▼             ▼             ▼
    ┌────────┐   ┌────────┐   ┌────────┐
    │ Model  │   │ Model  │   │ Model  │
    │   A    │   │   B    │   │   C    │
    └────┬───┘   └────┬───┘   └────┬───┘
         │             │             │
         └─────────────┼─────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  NEREvaluator   │
              │ (evaluator.rs)  │
              └────────┬────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
         ▼             ▼             ▼
    ┌────────┐   ┌────────┐   ┌────────┐
    │Metrics │   │ Error  │   │  Bias  │
    │Compute │   │Analysis│   │Analysis│
    └────────┘   └────────┘   └────────┘
```

## Evaluation Modes

From `modes.rs`:

| Mode | Description | Match Criteria |
|------|-------------|----------------|
| Strict | Exact span AND type | Both boundary and type must match |
| Exact | Exact span only | Boundary match, ignore type |
| Partial | Overlapping spans | Any overlap counts as partial match |
| Type | Type match only | Ignore boundaries, just check type |

## Coreference Metrics

From `coref_metrics.rs`:

| Metric | Type | Description |
|--------|------|-------------|
| MUC | Link-based | Counts links needed to reconstruct |
| B³ | Mention-based | Per-mention precision/recall |
| CEAF-e | Entity-based | Optimal alignment of entities |
| CEAF-m | Mention-based | Optimal alignment of mentions |
| LEA | Link+Entity | Link-based entity-aware |
| BLANC | Rand-index | Coreference/non-coreference decisions |
| CoNLL | Aggregate | Mean of MUC, B³, CEAF-e |

## Common Patterns

### Running Standard Evaluation

```rust
use anno::eval::{evaluate_ner_model, GoldEntity};
use anno::PatternNER;

let model = PatternNER::new();
let test_cases = vec![
    ("Meeting on Jan 15".into(), vec![
        GoldEntity::new("Jan 15", EntityType::Date, 11),
    ]),
];

let results = evaluate_ner_model(&model, &test_cases)?;
println!("F1: {:.1}%", results.f1 * 100.0);
```

### Cross-Domain Analysis

```rust
use anno::eval::{compare_datasets, compute_stats, estimate_difficulty};

let comparison = compare_datasets(&news_data, &medical_data);
println!("Type divergence: {:.3}", comparison.type_divergence);
println!("Domain gap: {:.2}", comparison.estimated_domain_gap);
```

### Calibration Analysis

```rust
use anno::eval::calibration::CalibrationEvaluator;

let predictions = vec![(0.95, true), (0.80, true), (0.60, false)];
let results = CalibrationEvaluator::compute(&predictions);
println!("ECE: {:.3}", results.ece);
```

### Production Monitoring

```rust
use anno::eval::{DriftDetector, DriftConfig};

let mut detector = DriftDetector::new(DriftConfig {
    min_samples: 1000,
    window_size: 500,
    ..Default::default()
});

// In production loop:
for prediction in predictions {
    detector.log_prediction(timestamp, confidence, entity_type, text);
}

let report = detector.analyze();
if report.drift_detected {
    alert_team(&report.summary);
}
```

## Adding New Evaluation Modules

1. Create a new file in `src/eval/`
2. Add `pub mod yourmodule;` to `mod.rs`
3. Add re-exports in `mod.rs` for public types
4. Add tests in `tests/eval_integration.rs`
5. Wire up in `quality_bench.rs` example

## Performance Considerations

- Most evaluation is single-threaded by design (predictable, reproducible)
- The harness supports parallel backend evaluation via `--release`
- Drift detection uses a ring buffer to bound memory
- Large datasets should be sampled for quick iteration

## Testing

Unit tests: `cargo test --lib`
Integration tests: `cargo test --test eval_integration`
Example benchmark: `cargo run --example quality_bench`

## References

- SemEval NER evaluation: [CoNLL-2003 shared task](https://aclanthology.org/W03-0419/)
- Coreference metrics: [Pradhan et al. 2014](https://aclanthology.org/P14-2006/)
- Calibration: [Guo et al. 2017](https://arxiv.org/abs/1706.04599)
- Dataset drift: [Chang et al. 2023](https://arxiv.org/abs/2305.17127)

