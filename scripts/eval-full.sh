#!/usr/bin/env bash
# Run full evaluations across all task-dataset-backend combinations
# Heavy operation - only run on eval-* branches or manual trigger

set -euo pipefail

MAX_EXAMPLES=${MAX_EXAMPLES:-}
OUTPUT=${OUTPUT:-eval-full-report.md}

echo "Running full evaluations across all task-dataset-backend combinations"

# Build with all features
cargo build --release --features "eval-advanced,onnx,candle" || {
    echo "Build failed"
    exit 1
}

# Run full benchmark
if [ -n "${MAX_EXAMPLES:-}" ]; then
    echo "Limiting to ${MAX_EXAMPLES} examples per dataset"
    cargo run --release --bin anno --features "cli,eval-advanced" -- benchmark \
        --max-examples "${MAX_EXAMPLES}" \
        --output "${OUTPUT}"
else
    echo "Running full evaluation (no example limit)"
    cargo run --release --bin anno --features "cli,eval-advanced" -- benchmark \
        --output "${OUTPUT}"
fi

echo "Full evaluation complete: ${OUTPUT}"
wc -l "${OUTPUT}"

