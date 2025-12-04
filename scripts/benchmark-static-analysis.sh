#!/usr/bin/env bash
# Benchmark static analysis tools to compare performance
# Creative use: helps identify which tools are worth running in fast CI vs slow CI

set -euo pipefail

echo "=== Static Analysis Tool Benchmarks ==="
echo ""
echo "Benchmarking tools on anno codebase..."
echo ""

RESULTS_FILE="static-analysis-benchmark.txt"
rm -f "$RESULTS_FILE"

benchmark_tool() {
    local tool_name=$1
    local command=$2
    
    echo -n "Benchmarking $tool_name... "
    
    if ! command -v $tool_name &> /dev/null; then
        echo "ERROR: not installed" | tee -a "$RESULTS_FILE"
        return
    fi
    
    local start_time=$(date +%s.%N)
    eval "$command" > /dev/null 2>&1 || true
    local end_time=$(date +%s.%N)
    
    local duration=$(echo "$end_time - $start_time" | bc)
    printf "OK: %.2fs\n" "$duration" | tee -a "$RESULTS_FILE"
}

echo "Tool | Status | Duration" >> "$RESULTS_FILE"
echo "-----|--------|---------" >> "$RESULTS_FILE"

benchmark_tool "cargo-deny" "cargo deny check"
benchmark_tool "cargo-machete" "cargo machete"
benchmark_tool "cargo-geiger" "cargo geiger --quiet"
benchmark_tool "opengrep" "opengrep scan --config auto --quiet anno/ anno-core/ anno-coalesce/ anno-strata/ anno-cli/"

# Clippy benchmark (baseline)
echo -n "Benchmarking clippy (baseline)... "
if command -v cargo &> /dev/null; then
    start_time=$(date +%s.%N)
    cargo clippy --all-targets --quiet 2>&1 | head -5 > /dev/null || true
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc)
    printf "OK: %.2fs\n" "$duration" | tee -a "$RESULTS_FILE"
else
    echo "ERROR: cargo not found" | tee -a "$RESULTS_FILE"
fi

echo ""
echo "=== Results ==="
cat "$RESULTS_FILE"
echo ""
echo "Full results saved to: $RESULTS_FILE"
echo ""
echo "Recommendations:"
echo "- Tools < 5s: Safe for fast CI (every PR)"
echo "- Tools 5-30s: Consider for slower CI or scheduled runs"
echo "- Tools > 30s: Only run on-demand or nightly"

