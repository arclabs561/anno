#!/usr/bin/env bash
# Generate repo-specific static analysis report
# Creative use: combines all repo-specific checks into unified report

set -euo pipefail

REPORT_FILE="repo-specific-analysis.md"
TIMESTAMP=$(date -u +"%Y-%m-%d %H:%M:%S UTC")

cat > "$REPORT_FILE" <<EOF
# Repo-Specific Static Analysis Report

Generated: $TIMESTAMP

This report focuses on patterns specific to the anno NLP/ML evaluation framework.

EOF

# Run all repo-specific checks
echo "## NLP/ML Pattern Analysis" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
./scripts/check-nlp-patterns.sh >> "$REPORT_FILE" 2>&1 || true
echo "" >> "$REPORT_FILE"

echo "## Evaluation Framework Analysis" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
./scripts/analyze-evaluation-patterns.sh >> "$REPORT_FILE" 2>&1 || true
echo "" >> "$REPORT_FILE"

echo "## ML Backend Analysis" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
./scripts/check-ml-backend-patterns.sh >> "$REPORT_FILE" 2>&1 || true
echo "" >> "$REPORT_FILE"

echo "## Evaluation Invariant Checks" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
./scripts/check-evaluation-invariants.sh >> "$REPORT_FILE" 2>&1 || true
echo "" >> "$REPORT_FILE"

# Add OpenGrep findings if available
if command -v opengrep &> /dev/null; then
    echo "## OpenGrep Custom Rules Findings" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    for rule_file in .opengrep/rules/*.yaml; do
        if [ -f "$rule_file" ]; then
            rule_name=$(basename "$rule_file" .yaml)
            echo "### $(basename "$rule_file")" >> "$REPORT_FILE"
            echo "" >> "$REPORT_FILE"
            opengrep scan -f "$rule_file" --json src/ 2>/dev/null | \
                jq -r '.results | length' | \
                xargs -I {} echo "Found {} issues" >> "$REPORT_FILE" 2>/dev/null || \
                echo "No issues found or jq not installed" >> "$REPORT_FILE"
            echo "" >> "$REPORT_FILE"
        fi
    done
fi

cat >> "$REPORT_FILE" <<EOF
## Summary

This report combines:
- NLP/ML-specific pattern checks
- Evaluation framework analysis
- ML backend pattern validation
- Evaluation invariant verification
- OpenGrep custom rule findings

## Next Steps

1. Review findings above
2. Address high-severity issues first
3. Run individual checks for detailed analysis:
   - \`just check-nlp-patterns\`
   - \`just analyze-eval-patterns\`
   - \`just check-ml-backends\`
   - \`just check-evaluation-invariants\`

EOF

echo "OK: Repo-specific analysis report generated: $REPORT_FILE"

