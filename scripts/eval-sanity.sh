#!/usr/bin/env bash
# Run small random sample evaluations for sanity checks
# Used in CI on push to verify everything works

set -euo pipefail

# Small random samples for sanity
MAX_EXAMPLES=${MAX_EXAMPLES:-20}
RANDOM_SEED=${RANDOM_SEED:-42}

echo "Running sanity check evaluations (max ${MAX_EXAMPLES} examples per dataset, seed ${RANDOM_SEED})"

# Keep repo root clean: write reports under ./reports/
mkdir -p reports

# Run benchmark with small samples.
#
# We explicitly pick a small-but-diverse NER slice so this stays bounded and exercises:
# - news + wikipedia + social
# - multilingual/low-resource datasets
# - both classical + ML-capable backends (feature-gated; some may skip)
cargo run --release -p anno-cli --bin anno --features "eval-advanced onnx" -- benchmark \
    --tasks ner \
    --datasets CoNLL2003Sample,WikiGold,Wnut17,WikiANN,MasakhaNER \
    --backends heuristic,stacked,bert_onnx,gliner_onnx \
    --max-examples "${MAX_EXAMPLES}" \
    --output reports/eval-sanity-report.md \
    --cached-only || {
    echo "Sanity check failed"
    exit 1
}

echo "Sanity check passed"
cat reports/eval-sanity-report.md

