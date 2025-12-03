#!/usr/bin/env bash
# Generate unified static analysis report from all tools
# Aggregates results from multiple static analysis tools into a single report

set -euo pipefail

REPORT_FILE="unified-static-analysis-report.md"
rm -f "$REPORT_FILE"

echo "# Unified Static Analysis Report" > "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# 1. Cargo-deny results
echo "## Dependency Security (cargo-deny)" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
if command -v cargo-deny &> /dev/null; then
    cargo deny check 2>&1 | head -50 >> "$REPORT_FILE" || echo "WARNING: cargo-deny check failed or found issues" >> "$REPORT_FILE"
else
    echo "ERROR: cargo-deny not installed" >> "$REPORT_FILE"
fi
echo "" >> "$REPORT_FILE"

# 2. Unused dependencies
echo "## Unused Dependencies (cargo-machete)" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
if command -v cargo-machete &> /dev/null; then
    cargo machete 2>&1 | head -30 >> "$REPORT_FILE" || echo "WARNING: Unused dependencies found" >> "$REPORT_FILE"
else
    echo "ERROR: cargo-machete not installed" >> "$REPORT_FILE"
fi
echo "" >> "$REPORT_FILE"

# 3. Unsafe code statistics
echo "## Unsafe Code Statistics (cargo-geiger)" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
if command -v cargo-geiger &> /dev/null; then
    cargo geiger --quiet 2>&1 | head -30 >> "$REPORT_FILE" || echo "INFO: No unsafe code statistics available" >> "$REPORT_FILE"
else
    echo "ERROR: cargo-geiger not installed" >> "$REPORT_FILE"
fi
echo "" >> "$REPORT_FILE"

# 4. OpenGrep results summary
echo "## Security Pattern Detection (OpenGrep)" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
if command -v opengrep &> /dev/null; then
    echo "### Default Security Rules" >> "$REPORT_FILE"
    opengrep scan --config auto --json src/ 2>/dev/null | jq -r '.results[] | "\(.check_id): \(.path):\(.start.line)"' | head -20 >> "$REPORT_FILE" || echo "OK: No issues found" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    echo "### Custom Rules Summary" >> "$REPORT_FILE"
    for rule_file in .opengrep/rules/*.yaml; do
        if [ -f "$rule_file" ]; then
            rule_name=$(basename "$rule_file" .yaml)
            count=$(opengrep scan -f "$rule_file" --json src/ 2>/dev/null | jq '.results | length' || echo "0")
            echo "- $rule_name: $count findings" >> "$REPORT_FILE"
        fi
    done
else
    echo "ERROR: opengrep not installed" >> "$REPORT_FILE"
fi
echo "" >> "$REPORT_FILE"

# 5. Repo-specific checks
echo "## Repo-Specific Pattern Checks" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
if [ -f "repo-specific-analysis.md" ]; then
    cat repo-specific-analysis.md >> "$REPORT_FILE"
else
    echo "INFO: Repo-specific analysis not available" >> "$REPORT_FILE"
fi
echo "" >> "$REPORT_FILE"

# 6. Summary
echo "## Summary" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "This report aggregates results from:" >> "$REPORT_FILE"
echo "- cargo-deny (dependency security)" >> "$REPORT_FILE"
echo "- cargo-machete (unused dependencies)" >> "$REPORT_FILE"
echo "- cargo-geiger (unsafe code statistics)" >> "$REPORT_FILE"
echo "- opengrep (security pattern detection)" >> "$REPORT_FILE"
echo "- Repo-specific pattern checks" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "For detailed results, see individual tool outputs in CI artifacts." >> "$REPORT_FILE"

echo "Unified report generated: $REPORT_FILE"

