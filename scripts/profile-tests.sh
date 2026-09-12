#!/usr/bin/env bash
# Profile one nextest run and analyze its structured test output.
# Usage: ./scripts/profile-tests.sh [profile] [filterset]

set -euo pipefail

PROFILE="${1:-quick}"
FILTER="${2:-}"
OUTPUT_DIR="target/test-profiles"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
JSON_OUTPUT="$OUTPUT_DIR/nextest_${PROFILE}_${TIMESTAMP}.json"
LOG_OUTPUT="$OUTPUT_DIR/nextest_${PROFILE}_${TIMESTAMP}.log"
ANALYSIS_OUTPUT="$OUTPUT_DIR/analysis_${PROFILE}_${TIMESTAMP}.txt"

mkdir -p "$OUTPUT_DIR"

nextest_args=(
    --profile "$PROFILE"
    --workspace
    --features "eval discourse"
    --message-format libtest-json-plus
    --message-format-version 0.1
    --status-level none
    --final-status-level none
)
if [[ -n "$FILTER" ]]; then
    nextest_args+=(-E "$FILTER")
fi

echo "=== Test Profiling (nextest) ==="
echo "Profile: $PROFILE"
echo "Filter: ${FILTER:-none}"
echo "Structured output: $JSON_OUTPUT"
echo "Diagnostic log: $LOG_OUTPUT"

# Keep stdout as newline-delimited libtest JSON. Cargo and nextest diagnostics
# stay on stderr so they cannot corrupt the analyzer's input.
set +e
NEXTEST_EXPERIMENTAL_LIBTEST_JSON=1 \
    cargo nextest run "${nextest_args[@]}" >"$JSON_OUTPUT" 2>"$LOG_OUTPUT"
nextest_status=$?
set -e

if (( nextest_status != 0 )); then
    echo "nextest failed with status $nextest_status; see $LOG_OUTPUT" >&2
    exit "$nextest_status"
fi

python3 scripts/analyze_test_profile.py "$JSON_OUTPUT" >"$ANALYSIS_OUTPUT"
echo "Analysis: $ANALYSIS_OUTPUT"
