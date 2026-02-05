#!/usr/bin/env bash
# Profile tests using Rust's native profiling tools
# Uses cargo test with profiling flags and nextest timing

set -euo pipefail

PROFILE="${1:-quick}"
OUTPUT_DIR="target/test-profiles"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUTPUT_DIR"

echo "=== Rust Native Test Profiling ==="
echo "Profile: $PROFILE"
echo ""

# Check Rust version for profiling support
RUST_VERSION=$(rustc --version | cut -d' ' -f2)
echo "Rust version: $RUST_VERSION"

# Method 1: Build with debug symbols and profile flags
echo ""
echo "=== Building with Profiling Symbols ==="
PROFILE_BUILD_LOG="$OUTPUT_DIR/build_profile_${TIMESTAMP}.log"

RUSTFLAGS="-g -C debuginfo=1" cargo build \
    --tests \
    --features "eval discourse" \
    2>&1 | tee "$PROFILE_BUILD_LOG"

# Method 2: Run with nextest JSON output (includes timing)
echo ""
echo "=== Running Tests with Nextest JSON Output ==="
JSON_OUTPUT="$OUTPUT_DIR/nextest_${PROFILE}_${TIMESTAMP}.json"

# Enable experimental libtest JSON support
export NEXTEST_EXPERIMENTAL_LIBTEST_JSON=1

cargo nextest run \
    --profile "$PROFILE" \
    --workspace \
    --features "eval discourse" \
    --message-format libtest-json-plus \
    --status-level all \
    > "$JSON_OUTPUT" 2>&1

if [ -f "$JSON_OUTPUT" ]; then
    echo "✓ JSON output: $JSON_OUTPUT"
fi

# Method 3: Use perf (Linux) or dtrace/sample (macOS) if available
if [[ "$OSTYPE" == "linux-gnu"* ]] && command -v perf >/dev/null 2>&1; then
    echo ""
    echo "=== Generating perf Profile (Linux) ==="
    PERF_DATA="$OUTPUT_DIR/perf_${TIMESTAMP}.data"
    cargo nextest run \
        --profile "$PROFILE" \
        --workspace \
        --features "eval discourse" \
        --no-capture \
        2>&1 | perf record -o "$PERF_DATA" -g -- cargo nextest run \
        --profile "$PROFILE" \
        --workspace \
        --features "eval discourse" || true
    
    if [ -f "$PERF_DATA" ]; then
        perf report -i "$PERF_DATA" > "$OUTPUT_DIR/perf_report_${TIMESTAMP}.txt" 2>&1 || true
        echo "✓ Perf data: $PERF_DATA"
        echo "  View with: perf report -i $PERF_DATA"
    fi
elif [[ "$OSTYPE" == "darwin"* ]] && command -v sample >/dev/null 2>&1; then
    echo ""
    echo "=== Generating sample Profile (macOS) ==="
    SAMPLE_OUTPUT="$OUTPUT_DIR/sample_${TIMESTAMP}.txt"
    # Note: sample requires a running process, so we'd need to run tests differently
    echo "Note: sample profiling requires attaching to running process"
    echo "      Use Instruments.app for GUI profiling instead"
fi

# Summary
echo ""
echo "=== Profiling Complete ==="
echo "Files generated:"
ls -lh "$OUTPUT_DIR"/*${TIMESTAMP}* 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' || echo "  (none)"

