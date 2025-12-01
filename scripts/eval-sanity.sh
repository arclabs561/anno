#!/usr/bin/env bash
# Run small random sample evaluations for sanity checks
# Used in CI on push to verify everything works

set -euo pipefail

# Small random samples for sanity (fast, ~5-10 min)
MAX_EXAMPLES=${MAX_EXAMPLES:-20}
RANDOM_SEED=${RANDOM_SEED:-42}

echo "Running sanity check evaluations (max ${MAX_EXAMPLES} examples per dataset, seed ${RANDOM_SEED})"

# Run benchmark with small samples
cargo run --release --bin anno --features "cli,eval-advanced" -- benchmark \
    --max-examples "${MAX_EXAMPLES}" \
    --output eval-sanity-report.md \
    --cached-only || {
    echo "Sanity check failed"
    exit 1
}

echo "Sanity check passed"
cat eval-sanity-report.md

