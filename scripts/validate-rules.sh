#!/usr/bin/env bash
# Validate that OpenGrep rules actually catch the patterns they're designed for
# Tests rules against known good/bad code patterns

set -euo pipefail

echo "=== OpenGrep Rule Validation ==="
echo ""
echo "Testing rules against known patterns..."
echo ""

VALIDATION_FAILURES=0

# Several rules restrict `paths.include` (for example `**/eval/**`), so a probe
# in /tmp scans zero files and the test fails for the wrong reason. Write each
# probe under a directory that matches those includes.
PROBE_DIR="$PWD/.rule-probes/eval"
mkdir -p "$PROBE_DIR"
trap 'rm -rf "$PWD/.rule-probes"' EXIT

# opengrep cannot read process substitution (`<(echo ...)`), which made every
# probe scan zero files. Write each probe to a real .rs file first.
scan_code() {
	local code="$1" rules="$2" file
	file="$PROBE_DIR/probe-$RANDOM.rs"
	printf '%s\n' "$code" >"$file"
	opengrep scan -f "$rules" --json "$file"
}

HAS_OPENGREP=0
HAS_JQ=0
HAS_AST_GREP=0

if command -v opengrep &>/dev/null; then
	HAS_OPENGREP=1
fi
if command -v jq &>/dev/null; then
	HAS_JQ=1
fi
if command -v ast-grep &>/dev/null; then
	HAS_AST_GREP=1
fi

if [ $HAS_OPENGREP -eq 0 ] || [ $HAS_JQ -eq 0 ]; then
	echo "WARNING:  OpenGrep validation requires opengrep + jq; skipping those checks."
fi

if [ $HAS_OPENGREP -eq 1 ] && [ $HAS_JQ -eq 1 ]; then
	# Test 1: Mutex double-lock pattern (should be caught)
	echo "## Test 1: Mutex Double-Lock Pattern"
	echo ""
	TEST_CODE='if let Ok(mut cache) = self.per_example_scores_cache.lock() {
        *cache = None;
    } else {
        drop(self.per_example_scores_cache.lock().unwrap_or_else(|e| e.into_inner()));
    }'

	if scan_code "$TEST_CODE" .opengrep/rules/rust-error-handling.yaml | jq -e '.results[] | select(.check_id == "mutex-double-lock-deadlock")' >/dev/null; then
		echo "OK: Rule correctly detects mutex double-lock pattern"
	else
		echo "ERROR: Rule failed to detect mutex double-lock pattern"
		VALIDATION_FAILURES=$((VALIDATION_FAILURES + 1))
	fi
	echo ""

	# Test 2: Variance calculation without Bessel's correction (should be caught)
	echo "## Test 2: Population Variance Pattern"
	echo ""
	TEST_CODE='let variance = scores.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / scores.len() as f64;'

	if scan_code "$TEST_CODE" .opengrep/rules/rust-evaluation-framework.yaml | jq -e '.results[] | select(.check_id == "variance-without-bessel")' >/dev/null; then
		echo "OK: Rule correctly detects population variance pattern"
	else
		echo "WARNING:  Variance pattern detection may need refinement (rule may not exist or pattern may be too specific)"
	fi
	echo ""

	# Test 3: Graph node empty id (should be caught)
	echo "## Test 3: Graph Node Empty ID"
	echo ""
	TEST_CODE='let node = GraphNode::new("", "Person", "John Doe");'

	if scan_code "$TEST_CODE" .opengrep/rules/rust-anno-specific.yaml | jq -e '.results[] | select(.check_id == "graph-node-empty-id")' >/dev/null; then
		echo "OK: Rule correctly detects empty graph node id"
	else
		echo "ERROR: Rule failed to detect empty graph node id"
		VALIDATION_FAILURES=$((VALIDATION_FAILURES + 1))
	fi
	echo ""

	# Test 4: Direct mutex lock bypass (should be caught)
	echo "## Test 4: Direct Mutex Lock Bypass"
	echo ""
	TEST_CODE='let guard = mutex.lock().unwrap();'

	if scan_code "$TEST_CODE" .opengrep/rules/rust-error-handling.yaml | jq -e '.results[] | select(.check_id == "direct-mutex-lock-bypass-helper")' >/dev/null; then
		echo "OK: Rule correctly detects direct mutex lock bypass"
	else
		echo "ERROR: Rule failed to detect direct mutex lock bypass"
		VALIDATION_FAILURES=$((VALIDATION_FAILURES + 1))
	fi
	echo ""

	# Test 5: Byte indexing on text (should be caught) + negative control
	echo "## Test 5: Unicode-Unsafe Text Slice (and negative control)"
	echo ""
	TEST_CODE='fn demo(text: &str, scores: &[f32]) {
        let _a = &text[..1];
        let _b = &scores[..1];
        let test_texts = vec!["a", "b"];
        let _c = &test_texts[..1];
    }'

	RESULTS=$(scan_code "$TEST_CODE" .opengrep/rules/rust-nlp-ml-patterns.yaml)
	if echo "$RESULTS" | jq -e '.results[] | select(.check_id == "byte-indexing-on-text")' >/dev/null; then
		echo "OK: Rule detects byte slicing on text"
	else
		echo "ERROR: Rule failed to detect byte slicing on text"
		VALIDATION_FAILURES=$((VALIDATION_FAILURES + 1))
	fi

	# Negative control: ensure we don't flag slice on non-text variable name
	MATCH_COUNT=$(echo "$RESULTS" | jq -r '[.results[] | select(.check_id == "byte-indexing-on-text")] | length' 2>/dev/null || echo "0")
	if [ "$MATCH_COUNT" -gt 1 ]; then
		echo "ERROR: Rule unexpectedly matched non-text slice(s) too ($MATCH_COUNT matches)"
		VALIDATION_FAILURES=$((VALIDATION_FAILURES + 1))
	else
		echo "OK: Negative control passes (non-text slice not flagged)"
	fi
	echo ""
fi

# Optional: ast-grep validation for unicode/metal rule files
if [ $HAS_AST_GREP -eq 1 ]; then
	echo "## Optional: ast-grep Unicode Rule"
	echo ""
	TEST_CODE='fn demo(text: &str) { let _x = &text[0..1]; }'
	if printf '%s\n' "$TEST_CODE" | ast-grep scan --rule .opengrep/rules/rust-unicode-offsets.yaml --stdin --json=compact | jq -e '.[] | select(.ruleId == "byte-slice-on-string")' >/dev/null; then
		echo "OK: ast-grep unicode rule detects byte slicing"
	else
		echo "WARNING:  ast-grep unicode rule did not match expected pattern"
	fi
	echo ""
fi

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
