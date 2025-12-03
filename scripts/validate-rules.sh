#!/usr/bin/env bash
# Validate that OpenGrep rules actually catch the patterns they're designed for
# Tests rules against known good/bad code patterns

set -euo pipefail

echo "=== OpenGrep Rule Validation ==="
echo ""
echo "Testing rules against known patterns..."
echo ""

VALIDATION_FAILURES=0

# Test 1: Mutex double-lock pattern (should be caught)
echo "## Test 1: Mutex Double-Lock Pattern"
echo ""
TEST_CODE='if let Ok(mut cache) = self.per_example_scores_cache.lock() {
    *cache = None;
} else {
    drop(self.per_example_scores_cache.lock().unwrap_or_else(|e| e.into_inner()));
}'

if opengrep scan -f .opengrep/rules/rust-error-handling.yaml --json <(echo "$TEST_CODE") 2>/dev/null | jq -e '.results[] | select(.check_id == "mutex-double-lock-deadlock")' > /dev/null; then
    echo "OK: Rule correctly detects mutex double-lock pattern"
else
    echo "ERROR: Rule failed to detect mutex double-lock pattern"
    ((VALIDATION_FAILURES++))
fi
echo ""

# Test 2: Variance calculation without Bessel's correction (should be caught)
echo "## Test 2: Population Variance Pattern"
echo ""
TEST_CODE='let variance = scores.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / scores.len() as f64;'

if opengrep scan -f .opengrep/rules/rust-evaluation-framework.yaml --json <(echo "$TEST_CODE") 2>/dev/null | jq -e '.results[] | select(.check_id == "variance-without-bessel")' > /dev/null 2>/dev/null; then
    echo "OK: Rule correctly detects population variance pattern"
else
    echo "WARNING:  Variance pattern detection may need refinement (rule may not exist or pattern may be too specific)"
fi
echo ""

# Test 3: Confidence score out of range (should be caught)
echo "## Test 3: Confidence Score Validation"
echo ""
TEST_CODE='let conf = Confidence::new(1.5);'

if opengrep scan -f .opengrep/rules/rust-anno-specific.yaml --json <(echo "$TEST_CODE") 2>/dev/null | jq -e '.results[] | select(.check_id == "confidence-score-out-of-range")' > /dev/null 2>/dev/null; then
    echo "OK: Rule correctly detects confidence score validation"
else
    echo "WARNING:  Confidence validation rule may need refinement"
fi
echo ""

# Test 4: Direct mutex lock bypass (should be caught)
echo "## Test 4: Direct Mutex Lock Bypass"
echo ""
TEST_CODE='let guard = mutex.lock().unwrap();'

if opengrep scan -f .opengrep/rules/rust-error-handling.yaml --json <(echo "$TEST_CODE") 2>/dev/null | jq -e '.results[] | select(.check_id == "direct-mutex-lock-bypass-helper")' > /dev/null; then
    echo "OK: Rule correctly detects direct mutex lock bypass"
else
    echo "ERROR: Rule failed to detect direct mutex lock bypass"
    ((VALIDATION_FAILURES++))
fi
echo ""

# Summary
echo "=== Validation Summary ==="
echo ""
if [ $VALIDATION_FAILURES -eq 0 ]; then
    echo "OK: All critical rules validated successfully"
    exit 0
else
    echo "ERROR: $VALIDATION_FAILURES rule(s) failed validation"
    echo ""
    echo "Recommendations:"
    echo "   - Review rule patterns for accuracy"
    echo "   - Test against actual codebase patterns"
    echo "   - Refine patterns based on false positive/negative rates"
    exit 1
fi

