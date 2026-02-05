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

# Run full benchmark
if [ -n "${MAX_EXAMPLES:-}" ]; then
    echo "Limiting to ${MAX_EXAMPLES} examples per dataset"
    cargo run --release -p anno-cli --bin anno --features "eval onnx candle" -- benchmark \
        --max-examples "${MAX_EXAMPLES}" \
        --output "${OUTPUT}"
else
    echo "Running full evaluation (no example limit)"
    cargo run --release -p anno-cli --bin anno --features "eval onnx candle" -- benchmark \
        --output "${OUTPUT}"
fi

echo "Full evaluation complete: ${OUTPUT}"
wc -l "${OUTPUT}"

