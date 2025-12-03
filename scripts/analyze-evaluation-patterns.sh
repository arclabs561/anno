#!/usr/bin/env bash
# Analyze evaluation framework patterns for optimization opportunities
# Creative use: identifies performance and correctness issues in evaluation code

set -euo pipefail

echo "=== Evaluation Framework Pattern Analysis ==="
echo ""

ANALYSIS_FILE="evaluation-pattern-analysis.md"
cat > "$ANALYSIS_FILE" <<EOF
# Evaluation Framework Pattern Analysis

Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

## Performance Patterns

EOF

# 1. Check for backend reuse
echo "### Backend Reuse Analysis" >> "$ANALYSIS_FILE"
echo "" >> "$ANALYSIS_FILE"
if rg -q "for.*in.*\{.*BackendFactory::create" --type rust src/eval/task_evaluator.rs 2>/dev/null; then
    echo "WARNING: **Backend Recreation in Loops**" >> "$ANALYSIS_FILE"
    echo "Found backend creation inside loops. This is inefficient." >> "$ANALYSIS_FILE"
    echo "" >> "$ANALYSIS_FILE"
    rg -A 3 "for.*in.*\{.*BackendFactory::create" --type rust src/eval/task_evaluator.rs 2>/dev/null | head -10 >> "$ANALYSIS_FILE" || true
else
    echo "OK: Backends created outside loops" >> "$ANALYSIS_FILE"
fi
echo "" >> "$ANALYSIS_FILE"

# 2. Check for per-example score caching
echo "### Per-Example Score Caching" >> "$ANALYSIS_FILE"
echo "" >> "$ANALYSIS_FILE"
if rg -q "per_example_scores_cache" --type rust src/eval/task_evaluator.rs 2>/dev/null; then
    CACHE_USES=$(rg -c "per_example_scores_cache" --type rust src/eval/task_evaluator.rs 2>/dev/null || echo "0")
    echo "OK: Per-example score caching implemented ($CACHE_USES uses)" >> "$ANALYSIS_FILE"
else
    echo "WARNING: No per-example score caching found" >> "$ANALYSIS_FILE"
fi
echo "" >> "$ANALYSIS_FILE"

# 3. Check for parallel evaluation
echo "### Parallel Evaluation" >> "$ANALYSIS_FILE"
echo "" >> "$ANALYSIS_FILE"
if rg -q "eval-parallel|par_iter|rayon" --type rust src/eval/task_evaluator.rs 2>/dev/null; then
    echo "OK: Parallel evaluation support found" >> "$ANALYSIS_FILE"
    if rg -q "thread_local|ThreadLocal" --type rust src/eval/task_evaluator.rs 2>/dev/null; then
        echo "OK: Thread-local backend caching found" >> "$ANALYSIS_FILE"
    else
        echo "WARNING: Parallel evaluation without thread-local caching" >> "$ANALYSIS_FILE"
    fi
else
    echo "INFO: Sequential evaluation only" >> "$ANALYSIS_FILE"
fi
echo "" >> "$ANALYSIS_FILE"

# 4. Check for confidence interval recomputation
echo "### Confidence Interval Computation" >> "$ANALYSIS_FILE"
echo "" >> "$ANALYSIS_FILE"
if rg -q "compute_confidence_intervals" --type rust src/eval/task_evaluator.rs 2>/dev/null; then
    if rg -q "recreate.*backend|BackendFactory::create.*compute_confidence" --type rust src/eval/task_evaluator.rs 2>/dev/null; then
        echo "WARNING: Confidence intervals may be recomputing predictions" >> "$ANALYSIS_FILE"
        echo "Consider using cached per-example scores instead" >> "$ANALYSIS_FILE"
    else
        echo "OK: Confidence intervals use cached data" >> "$ANALYSIS_FILE"
    fi
else
    echo "INFO: No confidence interval computation found" >> "$ANALYSIS_FILE"
fi
echo "" >> "$ANALYSIS_FILE"

# 5. Check for task-dataset validation
echo "### Task-Dataset Validation" >> "$ANALYSIS_FILE"
echo "" >> "$ANALYSIS_FILE"
if rg -q "is_valid_combination|task.*dataset.*compatible" --type rust src/eval/ 2>/dev/null; then
    echo "OK: Task-dataset validation found" >> "$ANALYSIS_FILE"
else
    echo "INFO: No explicit task-dataset validation found" >> "$ANALYSIS_FILE"
fi
echo "" >> "$ANALYSIS_FILE"

# 6. Check for metric computation patterns
echo "### Metric Computation Patterns" >> "$ANALYSIS_FILE"
echo "" >> "$ANALYSIS_FILE"
F1_CALCS=$(rg -c "f1.*=.*2.*\*|2.*\*.*precision.*recall" --type rust src/eval/metrics.rs 2>/dev/null || echo "0")
if [ "$F1_CALCS" -gt 0 ]; then
    ZERO_CHECKS=$(rg -c "if.*==.*0|if.*\+.*==.*0" --type rust src/eval/metrics.rs 2>/dev/null || echo "0")
    if [ "$ZERO_CHECKS" -lt "$F1_CALCS" ]; then
        echo "WARNING: Some F1 calculations may lack zero-checks" >> "$ANALYSIS_FILE"
    else
        echo "OK: F1 calculations have zero-checks" >> "$ANALYSIS_FILE"
    fi
else
    echo "INFO: No F1 calculations found" >> "$ANALYSIS_FILE"
fi
echo "" >> "$ANALYSIS_FILE"

cat >> "$ANALYSIS_FILE" <<EOF
## Recommendations

1. **Backend Reuse**: Create backends outside loops when possible
2. **Caching**: Use per-example score cache for confidence intervals
3. **Parallelization**: Ensure thread-local backend caching for parallel evaluation
4. **Validation**: Add task-dataset compatibility checks before evaluation
5. **Edge Cases**: Add zero-checks for all metric calculations

## Next Steps

- Review patterns flagged above
- Run OpenGrep with custom rules: \`just opengrep-custom\`
- Check evaluation performance: \`cargo bench --bench evaluation_parallel\`

EOF

echo "OK: Analysis complete: $ANALYSIS_FILE"

