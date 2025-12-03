#!/usr/bin/env bash
# Check for NLP/ML-specific patterns and anti-patterns
# Creative use: combines OpenGrep with custom analysis for NLP codebase

set -euo pipefail

echo "=== NLP/ML Pattern Analysis ==="
echo ""

ISSUES=0

# 1. Check for text offset validation
echo "## Text Offset Validation"
echo ""
if rg -q "start.*end|Entity.*\{.*start" --type rust src/entity.rs src/grounded.rs 2>/dev/null; then
    if ! rg -q "if.*start.*>.*end|assert.*start.*<=.*end" --type rust src/entity.rs src/grounded.rs 2>/dev/null; then
        echo "WARNING:  Potential missing offset validation in entity creation"
        ((ISSUES++))
    else
        echo "OK: Offset validation found"
    fi
else
    echo "INFO:  No entity creation patterns found"
fi

# 2. Check for confidence score validation
echo ""
echo "## Confidence Score Validation"
echo ""
if rg -q "Confidence::new" --type rust 2>/dev/null; then
    if ! rg -q "0\.0.*<=.*score.*<=.*1\.0|contains.*score" --type rust src/types/confidence.rs 2>/dev/null; then
        echo "WARNING:  Potential missing confidence score range validation"
        ((ISSUES++))
    else
        echo "OK: Confidence score validation found"
    fi
else
    echo "INFO:  No Confidence::new calls found"
fi

# 3. Check for variance calculation (Bessel's correction)
echo ""
echo "## Variance Calculation (Bessel's Correction)"
echo ""
if rg -q "variance.*sum.*/.*len\(\)" --type rust src/eval/task_evaluator.rs 2>/dev/null; then
    if rg -q "variance.*sum.*/.*\(.*len\(\)\s*-\s*1" --type rust src/eval/task_evaluator.rs 2>/dev/null; then
        echo "OK: Bessel's correction (n-1) found"
    else
        echo "WARNING:  Potential population variance (n) instead of sample variance (n-1)"
        ((ISSUES++))
    fi
else
    echo "INFO:  No variance calculations found"
fi

# 4. Check for model download error handling
echo ""
echo "## Model Download Error Handling"
echo ""
if rg -q "\.get\(.*\)\.map_err" --type rust src/backends/ 2>/dev/null; then
    AUTH_HANDLING=$(rg -c "401|Unauthorized|authentication" --type rust src/backends/ 2>/dev/null || echo "0")
    if [ "$AUTH_HANDLING" -lt 2 ]; then
        echo "WARNING:  Limited authentication error handling in model downloads"
        echo "   Consider adding 401/Unauthorized checks with helpful hints"
        ((ISSUES++))
    else
        echo "OK: Authentication error handling found"
    fi
else
    echo "INFO:  No model download patterns found"
fi

# 5. Check for backend recreation in loops
echo ""
echo "## Backend Recreation in Loops"
echo ""
if rg -q "for.*in.*\{.*BackendFactory::create" --type rust src/eval/ 2>/dev/null; then
    echo "WARNING:  Potential backend recreation in loops (performance issue)"
    ((ISSUES++))
else
    echo "OK: No backend recreation in loops found"
fi

# 6. Check for division by zero in metrics
echo ""
echo "## Division by Zero in Metrics"
echo ""
if rg -q "precision.*recall.*/|f1.*=.*2.*\*" --type rust src/eval/metrics.rs 2>/dev/null; then
    if rg -q "if.*==.*0|if.*\+.*==.*0" --type rust src/eval/metrics.rs 2>/dev/null; then
        echo "OK: Division by zero checks found"
    else
        echo "WARNING:  Potential division by zero in metric calculations"
        ((ISSUES++))
    fi
else
    echo "INFO:  No metric calculations found"
fi

# 7. Check for Unicode handling
echo ""
echo "## Unicode/Character Offset Handling"
echo ""
BYTE_INDEXING=$(rg -c "\[.*\.\.\..*\]" --type rust src/entity.rs src/eval/ 2>/dev/null | head -1 || echo "0")
CHAR_OFFSETS=$(rg -c "\.chars\(\)|char_indices" --type rust src/entity.rs src/eval/ 2>/dev/null | head -1 || echo "0")
if [ "$BYTE_INDEXING" -gt 10 ] && [ "$CHAR_OFFSETS" -lt 5 ]; then
    echo "WARNING:  Potential byte indexing instead of character offsets for Unicode"
    ((ISSUES++))
else
    echo "OK: Character offset handling appears correct"
fi

echo ""
echo "=== Summary ==="
echo "Issues found: $ISSUES"
echo ""

if [ $ISSUES -gt 0 ]; then
    echo "NOTE: Run 'just opengrep-custom' for detailed OpenGrep analysis"
    exit 1
else
    echo "OK: No obvious NLP/ML pattern issues found"
    exit 0
fi

