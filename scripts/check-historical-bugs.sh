#!/usr/bin/env bash
# Check for patterns that match historical bugs
# Creative use: validates that fixed bugs don't regress

set -euo pipefail

echo "=== Historical Bug Pattern Check ==="
echo ""
echo "Checking for patterns that match bugs that were previously fixed..."
echo ""

ISSUES=0

# 1. Check for mutex double-lock pattern (deadlock bug)
echo "## Mutex Double-Lock Pattern (Deadlock Bug)"
echo ""
if rg -q "if let Ok.*lock.*else.*lock" --type rust src/ 2>/dev/null; then
    echo "WARNING:  Found potential mutex double-lock pattern"
    echo "   This matches the deadlock bug pattern from BUGS_FIXED.md"
    echo "   Fix: Use single lock with unwrap_or_else"
    rg -A 5 "if let Ok.*lock.*else.*lock" --type rust src/ 2>/dev/null | head -20
    ((ISSUES++))
else
    echo "OK: No mutex double-lock patterns found"
fi

# 2. Check for population variance (variance bug)
echo ""
echo "## Variance Calculation (Bessel's Correction)"
echo ""
POPULATION_VAR=$(rg -c "variance.*sum.*/.*len\(\)\s*as\s*f64" --type rust src/eval/ 2>/dev/null || echo "0")
SAMPLE_VAR=$(rg -c "variance.*sum.*/.*\(.*len\(\)\s*-\s*1" --type rust src/eval/ 2>/dev/null || echo "0")
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
LOCKS_WITHOUT_HANDLING=$(rg -c "\.lock\(\)\.unwrap\(\)" --type rust src/ 2>/dev/null || echo "0")
LOCKS_WITH_HANDLING=$(rg -c "\.lock\(\)\.unwrap_or_else" --type rust src/ 2>/dev/null || echo "0")
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
BACKEND_RECREATION=$(rg -c "for.*in.*\{.*BackendFactory::create" --type rust src/eval/ 2>/dev/null || echo "0")
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

