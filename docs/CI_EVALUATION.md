# CI Evaluation Strategy

## Overview

CI now runs evaluations automatically with a two-tier strategy:
1. **Sanity checks** (on push): Small random samples for quick verification
2. **Full evaluations** (on eval-* branches): Complete task-dataset-backend matrix

## Architecture

### Minimal CI, Maximum Scripts

CI is kept minimal - it just calls scripts/justfile commands:
- All logic lives in `scripts/` and `justfile`
- Easy to test locally: `just eval-sanity` or `./scripts/eval-sanity.sh`
- CI is just orchestration, not implementation

### Scripts

#### `scripts/eval-sanity.sh`
- **Purpose**: Quick sanity checks on push
- **Duration**: ~5-10 minutes
- **Samples**: 20 examples per dataset (random seed: 42)
- **Output**: `eval-sanity-report.md`
- **When**: Runs on every push to main/master

#### `scripts/eval-full.sh`
- **Purpose**: Complete evaluations across all combinations
- **Duration**: ~30-120 minutes (depends on datasets/models)
- **Samples**: All examples (or limit via `MAX_EXAMPLES` env var)
- **Output**: `eval-full-report.md`
- **When**: 
  - Branches starting with `eval-` (e.g., `eval-bugfix`, `eval-feature`)
  - Manual workflow dispatch

## Usage

### Local Testing

```bash
# Run sanity checks locally
just eval-sanity
# or
./scripts/eval-sanity.sh

# Run full evaluations locally
just eval-full
# or
./scripts/eval-full.sh

# Run full with example limit
just eval-full-limit 100
# or
MAX_EXAMPLES=100 ./scripts/eval-full.sh
```

### CI Workflow

#### On Push (main/master)
1. Standard checks (fmt, clippy, tests)
2. **Sanity check evals** (20 examples per dataset)
3. Uploads `eval-sanity-report.md` as artifact

#### On Eval-* Branches
1. Standard checks
2. **Full evaluations** (all combinations)
3. Uploads `eval-full-report.md` as artifact

#### Manual Trigger
- Can trigger full evals via GitHub Actions UI
- Optional `max_examples` input parameter

## Benefits

1. **Actually runs evals**: No longer just tests, but real evaluations
2. **Fast feedback**: Sanity checks catch regressions quickly
3. **On-demand full evals**: Heavy operations only when needed
4. **Maintainable**: All logic in scripts/justfile, not buried in YAML
5. **Testable**: Run same commands locally as CI

## Reports

Both scripts generate markdown reports:
- `eval-sanity-report.md`: Quick overview with small samples
- `eval-full-report.md`: Comprehensive results across all combinations

Reports are uploaded as GitHub Actions artifacts and can be downloaded for analysis.

## Configuration

### Environment Variables

- `MAX_EXAMPLES`: Limit examples per dataset (default: none for full, 20 for sanity)
- `RANDOM_SEED`: Random seed for sampling (default: 42)
- `OUTPUT`: Output file path (default: `eval-*-report.md`)

### Justfile Commands

- `just eval-sanity`: Run sanity checks
- `just eval-full`: Run full evaluations
- `just eval-full-limit N`: Run full with N examples per dataset
- `just ci-eval`: Run full CI + sanity evals

## Branch Strategy

### Regular Development
- Push to `main/master`: Runs sanity checks
- Fast feedback, catches regressions

### Evaluation Work
- Create branch: `eval-<description>` (e.g., `eval-bugfix`, `eval-feature`)
- Push triggers full evaluations
- Review results before merging

### Manual Full Eval
- Use GitHub Actions "Run workflow" button
- Optional: Set `max_examples` to limit scope

## Performance

### Sanity Checks
- **Time**: ~5-10 minutes
- **Examples**: 20 per dataset
- **Coverage**: All task-dataset-backend combinations (small samples)

### Full Evaluations
- **Time**: ~30-120 minutes
- **Examples**: All (or limited via `MAX_EXAMPLES`)
- **Coverage**: Complete task-dataset-backend matrix

## Troubleshooting

### Scripts fail locally
- Check you have `eval-advanced` feature: `cargo build --features eval-advanced`
- Check datasets are cached: `just download-datasets`
- Check models are cached: `cargo test --features onnx` (downloads models)

### CI fails
- Check artifacts for reports
- Run locally: `just eval-sanity` or `just eval-full`
- Check logs for specific errors

### Full eval too slow
- Use `MAX_EXAMPLES` to limit scope
- Run on eval-* branch instead of main
- Use manual trigger with example limit

