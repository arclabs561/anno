#!/usr/bin/env bash
# Summarize static analysis failures across all jobs
# Used in CI to provide a single summary of all failures

set -euo pipefail

SUMMARY_FILE="static-analysis-failures-summary.md"
rm -f "$SUMMARY_FILE"

echo "# Static Analysis Failures Summary" > "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

FAILURES=0

# Check for cargo-deny failures
if [ -f "cargo-deny-output.txt" ]; then
    if grep -q "error\|denied" cargo-deny-output.txt 2>/dev/null; then
        echo "## ERROR: cargo-deny: Issues Found" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
        grep -i "error\|denied" cargo-deny-output.txt | head -10 >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
        ((FAILURES++))
    fi
fi

# Check for OpenGrep findings
if [ -f "opengrep-results.json" ]; then
    FINDINGS=$(jq '.results | length' opengrep-results.json 2>/dev/null || echo "0")
    if [ "$FINDINGS" -gt 0 ]; then
        echo "## WARNING: OpenGrep: $FINDINGS Findings" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
        jq -r '.results[] | "\(.check_id): \(.path):\(.start.line) - \(.message)"' opengrep-results.json | head -10 >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
        ((FAILURES++))
    fi
fi

# Check for unused dependencies
if [ -f "machete-output.txt" ]; then
    if grep -q "unused" machete-output.txt 2>/dev/null; then
        echo "## WARNING: cargo-machete: Unused Dependencies" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
        grep "unused" machete-output.txt | head -10 >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
        ((FAILURES++))
    fi
fi

# Check for repo-specific issues
if [ -f "repo-specific-analysis.md" ]; then
    if grep -q "WARNING:\|ERROR:" repo-specific-analysis.md 2>/dev/null; then
        echo "## WARNING: Repo-Specific: Issues Found" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
        grep "WARNING:\|ERROR:" repo-specific-analysis.md | head -10 >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
        ((FAILURES++))
    fi
fi

# Summary
echo "## Summary" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
if [ $FAILURES -eq 0 ]; then
    echo "SUCCESS: **No critical failures found**" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    echo "All static analysis checks passed or found only minor issues." >> "$SUMMARY_FILE"
else
    echo "ERROR: **$FAILURES job(s) found issues**" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    echo "Review the sections above for details. Most issues are non-blocking (continue-on-error: true)." >> "$SUMMARY_FILE"
fi

echo "Failure summary generated: $SUMMARY_FILE"
echo "Total failures: $FAILURES"

