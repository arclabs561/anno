#!/usr/bin/env bash
# Run the metric contracts against the evaluator compiled from this workspace.
set -euo pipefail

cases=(
    eval::metrics::tests::classification_f1_handles_empty_and_disjoint_labels
    eval::types::tests::test_metric_with_variance_from_samples
    eval::types::tests::test_metric_with_variance_empty
    eval::types::tests::test_metric_with_variance_single
    eval::task_evaluator::tests::confidence_intervals_bootstrap_pooled_strict_counts
    eval::task_evaluator::tests::stratified_metrics_group_normalized_custom_type_labels_together
)
log=$(mktemp)
trap 'rm -f "$log"' EXIT
cargo test --locked -p anno-eval --lib --features eval -- --format pretty --exact "${cases[@]}" | tee "$log"

# libtest accepts stale filters and exits successfully when they match no test.
for name in "${cases[@]}"; do
    if ! grep -Fxq "test $name ... ok" "$log"; then
        echo "Missing successful evaluation contract: $name" >&2
        exit 1
    fi
done
