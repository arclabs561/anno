#!/usr/bin/env bash
# Run small random sample evaluations for sanity checks
# Used in CI on push to verify everything works

set -euo pipefail

# Bounded, deterministic loader smoke settings.
MAX_EXAMPLES=${MAX_EXAMPLES:-20}
RANDOM_SEED=${RANDOM_SEED:-42}
ANNO_MAX_DOWNLOAD_BYTES=${ANNO_MAX_DOWNLOAD_BYTES:-8000000}
export ANNO_MAX_DOWNLOAD_BYTES
REPORT_MD=reports/eval-sanity-report.md
REPORT_JSON=reports/eval-sanity-report.json

printf 'Running WikiGold sanity evaluation (max %s examples, seed %s, max download %s bytes)\n' \
    "${MAX_EXAMPLES}" "${RANDOM_SEED}" "${ANNO_MAX_DOWNLOAD_BYTES}"

# `jq` is available on GitHub's Ubuntu runner. Local callers need it to validate
# the machine-readable result contract rather than treating a rendered report as success.
if ! command -v jq >/dev/null 2>&1; then
    printf '%s\n' 'eval-sanity requires jq to validate reports/eval-sanity-report.json' >&2
    exit 2
fi

# Keep repo root clean: write reports under ./reports/
mkdir -p reports

# Run benchmark with small samples.
#
# This deliberately permits a normal loader download. It exercises a stable, automatable
# NER source and two supported backends, while the matrix job covers broader datasets
# and ML backends. The loader enforces the per-object download cap above.
cargo run --release -p anno-cli --bin anno --features "eval onnx" -- benchmark \
    --tasks ner \
    --datasets WikiGold \
    --backends heuristic,stacked \
    --max-examples "${MAX_EXAMPLES}" \
    --seed "${RANDOM_SEED}" \
    --output "${REPORT_MD}" \
    --output-json "${REPORT_JSON}"

if ! jq -e '
    def is_expected_result:
        .task == "NER" and
        .dataset == "WikiGold" and
        (.backend == "heuristic" or .backend == "stacked") and
        .success == true and
        .num_examples > 0;
    (.summary | type == "object") and
    (.results | type == "array") and
    (.summary.total_combinations == 2) and
    (.summary.successful == 2) and
    (.summary.failed == 0) and
    (.summary.skipped == 0) and
    (.results | length == 2) and
    (.results | all(.[]; is_expected_result)) and
    ([.results[].backend] | sort == ["heuristic", "stacked"])
' "${REPORT_JSON}" >/dev/null; then
    printf '%s\n' 'Sanity evaluation did not produce two successful WikiGold backend results.' >&2
    jq '{summary, result_count: (.results | length)}' "${REPORT_JSON}" >&2 || true
    exit 1
fi

jq -r '
    .summary as $summary |
    "Sanity check passed: \($summary.successful) successful, \($summary.skipped) skips, \($summary.failed) failures."
' "${REPORT_JSON}"

cat "${REPORT_MD}"
