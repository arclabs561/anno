#!/usr/bin/env bash
# Run full evaluations across all task-dataset-backend combinations
# Heavy operation - only run on eval-* branches or manual trigger

set -euo pipefail

MAX_EXAMPLES=${MAX_EXAMPLES:-}
OUTPUT=${OUTPUT:-eval-full-report.md}

echo "Running full evaluations across all task-dataset-backend combinations"

# Build with all features (CLI + core backends)
cargo build --release -p anno-cli --features "eval onnx candle" || {
    echo "Build failed"
    exit 1
}

# The CLI writes diagnostic reports even when individual evaluations fail.
# Check the rows before reporting a successful full evaluation.
result_json=$(mktemp)
trap 'rm -f -- "$result_json"' EXIT
args=(benchmark --output "$OUTPUT" --output-json "$result_json")
if [[ -n "$MAX_EXAMPLES" ]]; then
    echo "Limiting to $MAX_EXAMPLES examples per dataset"
    args+=(--max-examples "$MAX_EXAMPLES")
else
    echo "Running full evaluation (no example limit)"
fi
cargo run --release -p anno-cli --bin anno --features "eval onnx candle" -- "${args[@]}"
if ! jq -e 'any(.results[]; .success == true and (.num_examples | type == "number" and . > 0))' "$result_json" >/dev/null; then
    echo "Full evaluation produced no successful result with evaluated examples." >&2
    exit 1
fi

echo "Full evaluation complete: ${OUTPUT}"
wc -l "${OUTPUT}"

