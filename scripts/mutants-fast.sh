#!/usr/bin/env bash
# Run the bounded discontinuous-span mutation check with cargo-mutants 27.1.0.
set -euo pipefail
cd "$(dirname "$0")/.."
output_dir=${1:-.}
property=discontinuous_span_total_length
args=(-p anno --no-default-features
      --file crates/anno/src/core/entity.rs --re 'DiscontinuousSpan::(total_len|contains)'
      --cargo-arg=--lib --cargo-arg=--locked "--cargo-test-arg=$property")

# Catch a moved target or an unexpected expansion before spending CI time.
listing=$(cargo mutants "${args[@]}" --list)
selected=$(awk 'NF {count++} END {print count+0}' <<< "$listing")
if (( selected < 1 || selected > 6 )); then
    echo "Expected 1–6 span mutants, discovered $selected. Inspect the selection." >&2
    exit 1
fi
printf '%s\n' "$listing"
export PROPTEST_CASES=${PROPTEST_CASES:-100}
export PROPTEST_RNG_SEED=${PROPTEST_RNG_SEED:-48123}
cargo mutants "${args[@]}" --output "$output_dir"

report="$output_dir/mutants.out"
if ! grep -Fq "::${property} ... ok" "$report/log/baseline.log"; then
    echo "The unmutated baseline did not run the expected property test." >&2
    exit 1
fi
caught=$(wc -l < "$report/caught.txt")
if (( caught != selected )); then
    echo "The tests caught $caught of $selected selected mutants." >&2
    exit 1
fi
