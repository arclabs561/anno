#!/usr/bin/env bash
# Check for patterns that match historical bugs
# Creative use: validates that fixed bugs don't regress

set -euo pipefail

echo "=== Historical Bug Pattern Check ==="
echo ""
echo "Checking for patterns that match bugs that were previously fixed..."
echo ""

ISSUES=0
WORKSPACE_DIRS=(crates/anno/ crates/anno-core/ crates/anno-eval/ crates/anno-cli/ tests/ examples/)
EVAL_DIR="crates/anno-eval/src/eval/"

# 1. Check for mutex double-lock pattern (deadlock bug)
echo "## Mutex Double-Lock Pattern (Deadlock Bug)"
echo ""
if rg -q "if let Ok.*lock.*else.*lock" --type rust "${WORKSPACE_DIRS[@]}" 2>/dev/null; then
    echo "WARNING:  Found potential mutex double-lock pattern"
    echo "   This matches the deadlock bug pattern from BUGS_FIXED.md"
    echo "   Fix: Use single lock with unwrap_or_else"
    rg -A 5 "if let Ok.*lock.*else.*lock" --type rust "${WORKSPACE_DIRS[@]}" 2>/dev/null | sed -n '1,20p'
    ((ISSUES++))
else
    echo "OK: No mutex double-lock patterns found"
fi

# 2. Check for population variance (variance bug)
echo ""
echo "## Variance Calculation (Bessel's Correction)"
echo ""
POPULATION_VAR=$(
    (rg -n "variance.*sum.*/.*len\\(\\)\\s*as\\s*f64" --type rust "$EVAL_DIR" 2>/dev/null || true) | wc -l | tr -d ' '
)
SAMPLE_VAR=$(
    (rg -n "variance.*sum.*/.*\\(.*len\\(\\)\\s*-\\s*1" --type rust "$EVAL_DIR" 2>/dev/null || true) | wc -l | tr -d ' '
)
if [ "$POPULATION_VAR" -gt 0 ]; then
    echo "WARNING:  Found potential population variance (n) instead of sample variance (n-1)"
    echo "   This matches the variance bug pattern from BUGS_FIXED.md"
    echo "   Found $POPULATION_VAR instances, $SAMPLE_VAR with Bessel's correction"
    ((ISSUES++))
else
    echo "OK: No population variance patterns found ($SAMPLE_VAR with Bessel's correction)"
fi

# 3. Check for mutex lock without poison handling
echo ""
echo "## Mutex Poison Handling"
echo ""
LOCKS_WITHOUT_HANDLING=$(
    (rg -n "\\.lock\\(\\)\\.unwrap\\(\\)" --type rust "${WORKSPACE_DIRS[@]}" 2>/dev/null || true) | wc -l | tr -d ' '
)
LOCKS_WITH_HANDLING=$(
    (rg -n "\\.lock\\(\\)\\.unwrap_or_else" --type rust "${WORKSPACE_DIRS[@]}" 2>/dev/null || true) | wc -l | tr -d ' '
)
if [ "$LOCKS_WITHOUT_HANDLING" -gt 0 ]; then
    echo "WARNING:  Found mutex locks without poison handling"
    echo "   $LOCKS_WITHOUT_HANDLING locks without handling, $LOCKS_WITH_HANDLING with handling"
    echo "   Consider using unwrap_or_else(|e| e.into_inner()) for poison handling"
    ((ISSUES++))
else
    echo "OK: All mutex locks have poison handling ($LOCKS_WITH_HANDLING with handling)"
fi

# 4. Check for backend recreation in loops (performance bug)
echo ""
echo "## Backend Recreation in Loops"
echo ""
BACKEND_RECREATION=$(
    (rg -n "for.*in.*\\{.*BackendFactory::create" --type rust "$EVAL_DIR" 2>/dev/null || true) | wc -l | tr -d ' '
)
if [ "$BACKEND_RECREATION" -gt 0 ]; then
    echo "WARNING:  Found backend recreation in loops"
    echo "   This is a performance issue - backends should be created outside loops"
    ((ISSUES++))
else
    echo "OK: No backend recreation in loops found"
fi

echo ""
echo "=== Summary ==="
echo "Issues found: $ISSUES"
echo ""

if [ $ISSUES -gt 0 ]; then
    echo "NOTE: These patterns match bugs that were previously fixed:"
    echo "   - See docs/BUGS_FIXED.md for details"
    echo "   - Review patterns above and apply fixes"
    exit 1
else
    echo "OK: No historical bug patterns found"
    exit 0
fi

