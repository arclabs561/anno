# /perf -- Performance benchmarking and regression detection

Measure anno's per-backend latency, throughput, and scaling behavior. Compare against saved baselines to detect regressions. This is not a correctness test -- it answers: "is it fast enough, and did it get slower?"

## Execution strategy

- **Run criterion benchmarks first**: they produce stable, statistically valid measurements with confidence intervals. Wall-clock timing is secondary.
- **Capture machine context**: hardware, OS, load average, Rust version, commit SHA. Performance numbers are meaningless without context.
- **Compare against prior reports**: check for prior reports in order: `.claude/reports/`, `qa/reports/`, `.qa/reports/`, `.claude/` root. Read the most recent found. If reports exist in old locations, move them to `.claude/reports/` with dated names before proceeding. Regressions >10% from the prior report are flagged.
- **Separate cold-start from warm**: ONNX backends have ~2s session creation overhead that dominates single-invocation timing. Criterion benchmarks amortize this; CLI wall-clock includes it.
- **Don't optimize prematurely**: report numbers, don't fix them. Performance fixes go in a separate pass after the report is reviewed.

## Report convention

Write to `.claude/reports/perf-YYYY-MM-DD.md` (globally gitignored via `~/.gitignore_global`). For same-day reruns, append `-v2`, `-v3`.

## Procedure

### 0. Read prior reports

```bash
eza --sort=modified -r .claude/reports/perf-*.md qa/reports/perf-*.md 2>/dev/null | head -3
```

Read the most recent report. Note baseline numbers for comparison.

### 1. Capture environment

```bash
echo "=== Environment ==="
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Rust: $(rustc --version)"
echo "OS: $(uname -srm)"
echo "CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || lscpu 2>/dev/null | grep 'Model name' | sed 's/.*: //')"
echo "Cores: $(nproc 2>/dev/null || sysctl -n hw.ncpu)"
echo "Load: $(uptime | sed 's/.*load average/load average/')"
```

### 2. Build release binary

```bash
cargo build --release -p anno-cli --features "eval onnx"
cargo build --release -p anno-lib --benches
```

Note build time. If >5min (clean) or >60s (incremental), flag it.

### 3. Run criterion benchmarks

This is the primary measurement. Criterion handles warmup, iteration count, and statistical analysis.

```bash
# Full benchmark suite
cargo bench -p anno-lib 2>&1 | tee /tmp/anno-perf-criterion.txt

# If time-constrained, run specific groups:
cargo bench -p anno-lib -- backends    # NER backend benchmarks (includes tplinker)
cargo bench -p anno-lib -- coref       # Coreference benchmarks
cargo bench -p anno-lib -- similarity  # Similarity metric benchmarks
cargo bench -p anno-lib -- tplinker    # TPLinker entity-relation benchmarks
```

**What to capture from criterion output**:
- For each benchmark: mean time, standard deviation, throughput (if reported)
- Change vs previous run: "improved", "regressed", "no change" with percentage
- Any benchmark that says "regressed" with >5% change

Criterion stores baselines in `target/criterion/`. The HTML report is at `target/criterion/report/index.html`.

### 4. CLI wall-clock timing (cold-start inclusive)

Measure end-to-end latency including process startup and ONNX model loading. Use `hyperfine` if available, otherwise `time`.

```bash
ANNO=./target/release/anno
TEXT="Apple CEO Tim Cook met Google CEO Sundar Pichai in Seattle on March 15, 2026."

# Check if hyperfine is available
if command -v hyperfine &>/dev/null; then
    TIMER="hyperfine --warmup 1 --runs 5"
else
    TIMER="time"
fi

# Non-ML backends (should be <50ms each)
for model in pattern heuristic crf hmm; do
    echo "=== $model ==="
    $TIMER $ANNO extract -t "$TEXT" --model $model --format json > /dev/null
done

# ML backends (include cold-start ONNX overhead)
for model in bert-onnx gliner ensemble nuner gliner2 stacked tplinker; do
    echo "=== $model ==="
    $TIMER $ANNO extract -t "$TEXT" --model $model --format json \
        $([ "$model" = "gliner" ] && echo "--extract-types person,organization,location") \
        $([ "$model" = "gliner2" ] && echo "--extract-types PER,ORG,LOC") \
        > /dev/null
done
```

**Expected ranges** (single sentence, M-series Mac, cold start):

| Backend | Expected | Red flag |
|---------|----------|----------|
| pattern | <20ms | >50ms |
| heuristic | <10ms | >50ms |
| crf | <15ms | >50ms |
| hmm | <10ms | >50ms |
| bert-onnx | 400-500ms | >800ms |
| gliner | 350-450ms | >1s |
| ensemble | 350-450ms | >1s |
| nuner | 2.0-2.5s | >5s |
| gliner2 | 2.0-2.5s | >4s |
| stacked | 2.5-3.0s | >4s |
| tplinker | 2.5-3.0s | >4s |

### 5. Scaling test

Measure how latency scales with input length. Tests chunking behavior and memory allocation.

```bash
ANNO=./target/release/anno
BASE="Angela Merkel met Emmanuel Macron in Berlin. "

# Generate inputs of increasing size
python3 -c "print('$BASE' * 1)" > /tmp/anno-perf-1x.txt
python3 -c "print('$BASE' * 10)" > /tmp/anno-perf-10x.txt
python3 -c "print('$BASE' * 50)" > /tmp/anno-perf-50x.txt
python3 -c "print('$BASE' * 100)" > /tmp/anno-perf-100x.txt
python3 -c "print('$BASE' * 500)" > /tmp/anno-perf-500x.txt

# Measure scaling for stacked (the default/primary backend)
for size in 1x 10x 50x 100x 500x; do
    echo "=== stacked $size ==="
    time $ANNO extract --file /tmp/anno-perf-${size}.txt --model stacked --format json > /dev/null 2>&1
done

# Measure scaling for bert-onnx (512-token chunking boundary)
for size in 1x 10x 50x 100x 500x; do
    echo "=== bert-onnx $size ==="
    time $ANNO extract --file /tmp/anno-perf-${size}.txt --model bert-onnx --format json > /dev/null 2>&1
done
```

**What to check**:
- Is scaling roughly linear with input size? (Expected for sentence-level models)
- Does 512-token chunking cause a step function at ~50x? (bert-onnx splits at chunk boundaries)
- Does 500x complete at all, or does it OOM/timeout?
- Note any superlinear scaling (>2x time for 2x input) -- indicates O(n^2) or worse

### 6. Memory profiling (optional, when investigating regressions)

```bash
ANNO=./target/release/anno
TEXT="Apple CEO Tim Cook met Google CEO Sundar Pichai in Seattle."

# Peak RSS via /usr/bin/time (macOS: use gtime from gnu-time)
if command -v gtime &>/dev/null; then
    MEMTIME="gtime -v"
elif [[ "$(uname)" == "Linux" ]]; then
    MEMTIME="/usr/bin/time -v"
else
    MEMTIME=""
fi

if [ -n "$MEMTIME" ]; then
    for model in pattern bert-onnx stacked nuner; do
        echo "=== $model ==="
        $MEMTIME $ANNO extract -t "$TEXT" --model $model --format json > /dev/null 2>&1
    done
fi

# For deeper analysis: use heaptrack or dhat (requires debug symbols)
# cargo build --release -p anno-cli --features "eval onnx"  # with debug=1 in profile
# heaptrack ./target/release/anno extract -t "$TEXT" --model stacked --format json
```

**Expected memory ranges** (peak RSS, single sentence):

| Backend | Expected | Red flag |
|---------|----------|----------|
| pattern | <20MB | >50MB |
| bert-onnx | 200-400MB | >600MB |
| stacked | 400-800MB | >1.2GB |
| nuner | 300-500MB | >800MB |

### 7. Batch throughput

Measure documents-per-second for batch processing.

```bash
ANNO=./target/release/anno

# Generate 100 documents
python3 -c "
import json
for i in range(100):
    print(json.dumps({'id': str(i), 'text': f'Document {i}: Apple CEO Tim Cook met Google CEO Sundar Pichai in Seattle on March {i+1}.'}))
" > /tmp/anno-perf-batch.jsonl

# Measure batch throughput per backend
for model in pattern bert-onnx stacked tplinker; do
    echo "=== $model batch (100 docs) ==="
    time cat /tmp/anno-perf-batch.jsonl | $ANNO batch --stdin --model $model --format json > /dev/null 2>&1
done
```

**What to capture**: total time, docs/sec, whether batch mode amortizes model loading (stacked should be much faster per-doc in batch vs single invocation).

### 8. Comparison with prior report

For each measurement:
- **Improved (>5% faster)**: note the improvement
- **Stable (within 5%)**: mark as stable
- **Regressed (>5% slower)**: flag with details
- **Red flag (>10% slower)**: investigate immediately

Create a comparison table:

```markdown
| Benchmark | Previous | Current | Change | Status |
|-----------|----------|---------|--------|--------|
| ... | ... | ... | ... | ... |
```

### 9. Write the report

Save to `.claude/reports/perf-YYYY-MM-DD.md`. Include:

1. **Environment**: date, commit, hardware, Rust version, load average
2. **Criterion summary**: table of mean times + confidence intervals for each benchmark group
3. **CLI cold-start timing**: table of wall-clock times per backend
4. **Scaling analysis**: table or chart of time vs input size
5. **Memory profile**: peak RSS per backend (if measured)
6. **Batch throughput**: docs/sec per backend
7. **Regression table**: comparison against prior report
8. **Anomalies**: anything unexpected (spikes, OOM, timeouts)
9. **Recommendations**: specific optimizations worth investigating (only if data supports it)

## Regression tracking

### Known performance characteristics

- **ONNX cold-start**: ~2s overhead for session creation (bert-onnx, nuner, gliner, gliner2). This is ort crate behavior, not anno code. In daemon/server mode this is amortized.
- **stacked = bert-onnx + nuner + regex + heuristic**: wall-clock is dominated by the slowest ML backend (nuner ~2s) plus bert-onnx (~0.5s), run sequentially. Total ~2.5-3s.
- **tplinker**: similar to stacked (ONNX session + inference).
- **512-token chunking**: bert-onnx splits long text at 512-token boundaries. Each chunk is a separate ONNX inference call. Expect step-function scaling.
- **Criterion baseline drift**: criterion compares against its own saved baselines in `target/criterion/`. If `cargo clean` was run, baselines are lost. Note when this happens.

### Open performance items

Record findings in the report file (`.claude/reports/perf-YYYY-MM-DD.md`), not here.
