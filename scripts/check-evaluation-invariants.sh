#!/usr/bin/env bash
# Check evaluation framework invariants
# Creative use: validates statistical correctness of evaluation code

set -euo pipefail

echo "=== Evaluation Framework Invariant Checks ==="
echo ""

ISSUES=0

# 1. Check for Bessel's correction in variance
echo "## Variance Calculation (Bessel's Correction)"
echo ""
VARIANCE_PATTERNS=$(rg -c "variance.*sum.*/.*len\(\)" --type rust anno/src/eval/task_evaluator.rs 2>/dev/null || echo "0")
BESSEL_PATTERNS=$(rg -c "variance.*sum.*/.*\(.*len\(\)\s*-\s*1" --type rust anno/src/eval/task_evaluator.rs 2>/dev/null || echo "0")
if [ "$VARIANCE_PATTERNS" -gt 0 ]; then
    if [ "$BESSEL_PATTERNS" -lt "$VARIANCE_PATTERNS" ]; then
        echo "WARNING:  Some variance calculations may use population variance (n) instead of sample variance (n-1)"
        echo "   Found $VARIANCE_PATTERNS variance calculations, $BESSEL_PATTERNS with Bessel's correction"
        ((ISSUES++))
    else
        echo "OK: All variance calculations use Bessel's correction (n-1)"
    fi
else
    echo "INFO:  No variance calculations found"
fi

# 2. Check for confidence interval edge cases
echo ""
echo "## Confidence Interval Edge Cases"
echo ""
CI_FUNCTIONS=$(rg -c "compute_confidence|confidence.*interval" --type rust anno/src/eval/task_evaluator.rs 2>/dev/null || echo "0")
EDGE_CHECKS=$(rg -c "n\s*==\s*0|n\s*==\s*1|len\(\)\s*<=\s*1" --type rust anno/src/eval/task_evaluator.rs 2>/dev/null || echo "0")
if [ "$CI_FUNCTIONS" -gt 0 ]; then
    if [ "$EDGE_CHECKS" -lt 2 ]; then
        echo "WARNING:  Confidence interval functions may lack edge case handling (n=0, n=1)"
        ((ISSUES++))
    else
        echo "OK: Confidence interval edge cases handled"
    fi
else
    echo "INFO:  No confidence interval functions found"
fi

# 3. Check for F1/precision/recall zero-checks
echo ""
echo "## Metric Calculation Zero-Checks"
echo ""
F1_CALCS=$(rg -c "f1.*=.*2.*\*|2.*\*.*precision.*recall" --type rust anno/src/eval/metrics.rs 2>/dev/null || echo "0")
ZERO_CHECKS=$(rg -c "if.*precision.*\+.*recall.*==.*0|if.*tp.*\+.*fp.*==.*0" --type rust anno/src/eval/metrics.rs 2>/dev/null || echo "0")
if [ "$F1_CALCS" -gt 0 ]; then
    if [ "$ZERO_CHECKS" -lt "$F1_CALCS" ]; then
        echo "WARNING:  Some F1 calculations may lack zero-checks"
        echo "   Found $F1_CALCS F1 calculations, $ZERO_CHECKS zero-checks"
        ((ISSUES++))
    else
        echo "OK: F1 calculations have zero-checks"
    fi
else
    echo "INFO:  No F1 calculations found"
fi

# 4. Check for per-example score reuse
echo ""
echo "## Per-Example Score Reuse"
echo ""
CACHE_ACCESS=$(rg -c "per_example_scores_cache" --type rust anno/src/eval/task_evaluator.rs 2>/dev/null || echo "0")
RECOMPUTE_PATTERNS=$(rg -c "recreate.*backend|BackendFactory::create.*confidence" --type rust anno/src/eval/task_evaluator.rs 2>/dev/null || echo "0")
if [ "$CACHE_ACCESS" -gt 0 ]; then
    if [ "$RECOMPUTE_PATTERNS" -gt 0 ]; then
        echo "WARNING:  Some functions may recompute instead of using cached per-example scores"
        ((ISSUES++))
    else
        echo "OK: Per-example scores are cached and reused"
    fi
else
    echo "INFO:  No per-example score caching found"
fi

# 5. Check for stratified metrics computation
echo ""
echo "## Stratified Metrics Computation"
echo ""
STRATIFIED=$(rg -c "StratifiedMetrics|by_entity_type" --type rust anno/src/eval/task_evaluator.rs 2>/dev/null || echo "0")
PER_TYPE=$(rg -c "per.*type.*score|by_type.*compute" --type rust anno/src/eval/task_evaluator.rs 2>/dev/null || echo "0")
if [ "$STRATIFIED" -gt 0 ]; then
    if [ "$PER_TYPE" -lt 2 ]; then
        echo "WARNING:  Stratified metrics may use aggregate values instead of per-type computation"
        ((ISSUES++))
    else
        echo "OK: Stratified metrics computed per-type"
    fi
else
    echo "INFO:  No stratified metrics found"
fi

echo ""
echo "=== Summary ==="
echo "Issues found: $ISSUES"
echo ""

if [ $ISSUES -gt 0 ]; then
    echo "NOTE: Recommendations:"
    echo "   - Use sample variance (n-1) for all variance calculations"
    echo "   - Add edge case handling for small sample sizes"
    echo "   - Add zero-checks for all metric calculations"
    echo "   - Reuse cached per-example scores instead of recomputing"
    echo "   - Compute stratified metrics from actual per-type scores"
    exit 1
else
    echo "OK: Evaluation framework invariants look correct"
    exit 0
fi

