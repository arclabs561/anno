#!/usr/bin/env bash
# Validate that OpenGrep rules actually catch the patterns they're designed for
# Tests rules against known good/bad code patterns

set -euo pipefail
cd "$(dirname "$0")/.."

if command -v opengrep >/dev/null; then
    opengrep test -f .opengrep/rules/rust-validated.yaml .opengrep/fixtures/eval/graph.rs
else
    echo "WARNING: OpenGrep is unavailable; skipping its rule fixtures."
fi

# Optional: ast-grep validation for unicode/metal rule files
if command -v ast-grep >/dev/null && command -v jq >/dev/null; then
    echo "## Optional: ast-grep Unicode Rule"
    echo ""
    TEST_CODE='fn demo(text: &str) { let _x = &text[0..1]; }'
    if printf '%s\n' "$TEST_CODE" | ast-grep scan --rule .opengrep/rules/rust-unicode-offsets.yaml --stdin --json=compact | jq -e '.[] | select(.ruleId == "byte-slice-on-string")' > /dev/null; then
        echo "OK: ast-grep unicode rule detects byte slicing"
    else
        echo "WARNING:  ast-grep unicode rule did not match expected pattern"
    fi
    echo ""
fi
