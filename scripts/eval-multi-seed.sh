#!/usr/bin/env bash
# Run evaluations with multiple random seeds for comprehensive testing
# Tests all flows with varied samples to catch seed-dependent issues

set -euo pipefail

MAX_EXAMPLES=${MAX_EXAMPLES:-20}
SEEDS=(42 123 456 789 2024)
OUTPUT_DIR=${OUTPUT_DIR:-eval-reports}

echo "=== Multi-Seed Evaluation ==="
echo "Max examples per dataset: ${MAX_EXAMPLES}"
echo "Seeds: ${SEEDS[*]}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

mkdir -p "${OUTPUT_DIR}"

FAILED=0
SUCCESS=0

for seed in "${SEEDS[@]}"; do
    echo "--- Running evaluation with seed ${seed} ---"
    
    OUTPUT_FILE="${OUTPUT_DIR}/eval-seed-${seed}.md"
    
    if cargo run --release --bin anno --features "cli,eval-advanced" -- benchmark \
        --max-examples "${MAX_EXAMPLES}" \
        --seed "${seed}" \
        --output "${OUTPUT_FILE}" \
        --cached-only 2>&1 | tee "${OUTPUT_DIR}/seed-${seed}.log"; then
        SUCCESS=$((SUCCESS + 1))
        echo "✓ Seed ${seed} completed"
    else
        FAILED=$((FAILED + 1))
        echo "✗ Seed ${seed} failed"
    fi
    echo ""
done

echo "=== Summary ==="
echo "Successful: ${SUCCESS}/${#SEEDS[@]}"
echo "Failed: ${FAILED}/${#SEEDS[@]}"

# Aggregate results
echo ""
echo "=== Aggregating Results ==="
if [ ${SUCCESS} -gt 0 ]; then
    # Find common failures across seeds
    echo "## Common Failures Across Seeds" > "${OUTPUT_DIR}/aggregated.md"
    echo "" >> "${OUTPUT_DIR}/aggregated.md"
    
    for seed in "${SEEDS[@]}"; do
        if [ -f "${OUTPUT_DIR}/eval-seed-${seed}.md" ]; then
            echo "### Seed ${seed}" >> "${OUTPUT_DIR}/aggregated.md"
            grep -A 100 "^## Failures" "${OUTPUT_DIR}/eval-seed-${seed}.md" | head -20 >> "${OUTPUT_DIR}/aggregated.md" || true
            echo "" >> "${OUTPUT_DIR}/aggregated.md"
        fi
    done
    
    echo "Aggregated report: ${OUTPUT_DIR}/aggregated.md"
fi

if [ ${FAILED} -gt 0 ]; then
    exit 1
fi

