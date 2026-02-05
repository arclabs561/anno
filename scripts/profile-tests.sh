#!/usr/bin/env bash
# Profile test execution using nextest and Rust tooling
# Usage: ./scripts/profile-tests.sh [profile] [filter]

set -euo pipefail

PROFILE="${1:-quick}"
FILTER="${2:-}"
OUTPUT_DIR="target/test-profiles"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUTPUT_DIR"

echo "=== Test Profiling (Nextest + Rust Tooling) ==="
echo "Profile: $PROFILE"
echo "Filter: ${FILTER:-none}"
echo "Output: $OUTPUT_DIR"
echo ""

# Method 1: Nextest JSON output with timing (always available)
echo "=== Generating Nextest JSON Output with Timing ==="
JSON_OUTPUT="$OUTPUT_DIR/nextest_${PROFILE}_${TIMESTAMP}.json"

# Enable experimental libtest JSON support
export NEXTEST_EXPERIMENTAL_LIBTEST_JSON=1

cargo nextest run \
    --profile "$PROFILE" \
    --workspace \
    --features "eval discourse" \
    ${FILTER:+-E "$FILTER"} \
    --message-format libtest-json-plus \
    --status-level all \
    > "$JSON_OUTPUT" 2>&1

if [ -f "$JSON_OUTPUT" ]; then
    echo "✓ JSON output saved: $JSON_OUTPUT"
    
    # Extract timing summary if jq is available
    if command -v jq >/dev/null 2>&1; then
        TOTAL_TESTS=$(jq -r 'select(.type == "test") | .event == "ok" or .event == "failed"' "$JSON_OUTPUT" | grep -c true || echo "0")
        echo "  Total tests: $TOTAL_TESTS"
        echo ""
        echo "Run 'just profile-analyze' for detailed breakdown"
    fi
else
    echo "Warning: JSON output file not generated"
fi

if [ -f "$TIMING_FILE" ]; then
    echo ""
    echo "✓ Timing report saved: $TIMING_FILE"
    
    # Quick summary using jq if available
    if command -v jq >/dev/null 2>&1; then
        TOTAL_TESTS=$(jq -r '.test_executions | length' "$TIMING_FILE")
        TOTAL_TIME=$(jq -r '[.test_executions[].duration_secs] | add' "$TIMING_FILE")
        AVG_TIME=$(echo "$TOTAL_TIME / $TOTAL_TESTS" | bc -l 2>/dev/null || echo "0")
        
        echo ""
        echo "Summary:"
        echo "  Total tests: $TOTAL_TESTS"
        echo "  Total time:   ${TOTAL_TIME}s"
        echo "  Avg time:    ${AVG_TIME}s"
        echo ""
        echo "Run 'just profile-analyze' for detailed breakdown"
    fi
else
    echo "Warning: Timing file not generated"
fi

# Method 2: Rust profiling with --profile test (if supported)
echo ""
echo "=== Rust Profiling (cargo test --profile test) ==="
if cargo test --help 2>&1 | grep -q "profile"; then
    PROFILE_OUTPUT="$OUTPUT_DIR/rust_profile_${TIMESTAMP}"
    echo "Running with Rust profiling..."
    cargo test \
        --profile test \
        --workspace \
        --features "eval discourse" \
        ${FILTER:+--test "$FILTER"} \
        --no-run \
        2>&1 | tee "$OUTPUT_DIR/rust_profile_build_${TIMESTAMP}.log" || {
        echo "Note: --profile test may not be available in this Rust version"
    }
else
    echo "Note: --profile test not available (requires Rust 1.78+)"
fi

# Method 3: Generate HTML report from timing data
if [ -f "$TIMING_FILE" ] && command -v uv >/dev/null 2>&1; then
    echo ""
    echo "=== Generating Analysis Report ==="
    uv run -- python scripts/analyze_test_profile.py "$TIMING_FILE" > "$OUTPUT_DIR/analysis_${TIMESTAMP}.txt" 2>&1 || true
    if [ -f "$OUTPUT_DIR/analysis_${TIMESTAMP}.txt" ]; then
        echo "✓ Analysis report: $OUTPUT_DIR/analysis_${TIMESTAMP}.txt"
    fi
fi

echo ""
echo "=== Profiling Complete ==="
echo "Results in: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  just profile-analyze          # Detailed analysis"
echo "  just profile-slowest          # Show slowest tests"
echo "  cat $OUTPUT_DIR/analysis_${TIMESTAMP}.txt  # View analysis"
