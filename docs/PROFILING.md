# Profiling Guide

This guide explains how to profile the evaluation framework to identify performance bottlenecks.

## Quick Start

Build with profiling enabled:

```bash
cargo build --features onnx,eval-profiling
```

Run evaluation with profiling:

```bash
./target/debug/anno benchmark --tasks ner --backends bert_onnx --datasets wikigold --max-examples 50
```

The profiling summary will be printed at the end, showing:
- Total time spent in each operation
- Average time per call
- Minimum and maximum times
- Number of calls

## Example Output

```
=== Profiling Summary ===
Operation                      Count    Total (ms)   Avg (ms)   Min (ms)   Max (ms)
------------------------------------------------------------------------------------------
backend_inference                 50      2500.00      50.00       45.00       55.00
extract_gold_entities             50         5.00       0.10        0.05        0.15
compute_metrics                    1        10.00      10.00       10.00       10.00
evaluate_ner_task                  1      2520.00    2520.00     2520.00     2520.00
```

## Profiled Operations

- `evaluate_ner_task`: Total time for NER evaluation
- `backend_inference`: Time spent in model inference (ONNX, Candle, etc.)
- `extract_gold_entities`: Time spent extracting gold entities from dataset
- `compute_metrics`: Time spent computing precision/recall/F1 metrics

## Advanced Profiling

### Using External Profilers

For more detailed profiling, use external tools:

**macOS (Instruments):**
```bash
cargo build --release --features onnx
instruments -t "Time Profiler" ./target/release/anno benchmark --tasks ner --backends bert_onnx --datasets wikigold
```

**Linux (perf):**
```bash
cargo build --release --features onnx
perf record ./target/release/anno benchmark --tasks ner --backends bert_onnx --datasets wikigold
perf report
```

**Flamegraph:**
```bash
cargo install flamegraph
cargo flamegraph --features onnx -- ./target/release/anno benchmark --tasks ner --backends bert_onnx --datasets wikigold
```

## Interpreting Results

1. **backend_inference** should be the largest time consumer for ML backends
2. **extract_gold_entities** should be very fast (<1ms per sentence)
3. **compute_metrics** should be fast (<100ms total)
4. If **backend_inference** is slow, consider:
   - Using parallel processing (`eval-parallel` feature)
   - Using session pooling (`session-pool` feature)
   - Optimizing model loading/caching

## Performance Targets

For 50 sentences with `bert_onnx`:
- Total time: < 60 seconds
- backend_inference: < 50ms per sentence
- extract_gold_entities: < 0.5ms per sentence
- compute_metrics: < 100ms total

