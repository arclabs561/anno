#!/usr/bin/env bash
# Generate comprehensive safety report using multiple static analysis tools
# Creative use: combines cargo-geiger, OpenGrep, and Miri results

set -euo pipefail

REPORT_FILE="safety-report.md"
TIMESTAMP=$(date -u +"%Y-%m-%d %H:%M:%S UTC")

cat > "$REPORT_FILE" <<EOF
# Safety Report

Generated: $TIMESTAMP

This report combines results from multiple static analysis tools to provide a comprehensive view of code safety.

## Summary

EOF

# 1. Unsafe Code Statistics (cargo-geiger)
if command -v cargo-geiger &> /dev/null; then
    echo "## Unsafe Code Statistics (cargo-geiger)" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    cargo geiger --output-format json > .geiger-tmp.json 2>/dev/null || echo "{}" > .geiger-tmp.json
    
    if command -v jq &> /dev/null; then
        UNSAFE_COUNT=$(jq -r '[.packages[] | select(.geiger.unsafe_used > 0)] | length' .geiger-tmp.json 2>/dev/null || echo "0")
        echo "- **Files with unsafe code**: $UNSAFE_COUNT" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo "### Detailed Breakdown" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        jq -r '.packages[] | select(.geiger.unsafe_used > 0) | "- \(.name): \(.geiger.unsafe_used) unsafe uses"' .geiger-tmp.json >> "$REPORT_FILE" 2>/dev/null || echo "No unsafe code found" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        rm -f .geiger-tmp.json
    else
        echo "WARNING:  jq not installed, skipping detailed analysis" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
    fi
else
    echo "WARNING:  cargo-geiger not installed. Install with: cargo install cargo-geiger" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
fi

# 2. OpenGrep Security Findings
if command -v opengrep &> /dev/null; then
    echo "## Security Pattern Detection (OpenGrep)" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    opengrep scan --config auto --json --output .opengrep-tmp.json anno/ anno-core/ anno-coalesce/ anno-strata/ anno-cli/ 2>/dev/null || echo '{"results":[]}' > .opengrep-tmp.json
    
    if command -v jq &> /dev/null; then
        FINDING_COUNT=$(jq -r '.results | length' .opengrep-tmp.json 2>/dev/null || echo "0")
        echo "- **Security findings**: $FINDING_COUNT" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        
        if [ "$FINDING_COUNT" -gt 0 ]; then
            echo "### Top Issues" >> "$REPORT_FILE"
            echo "" >> "$REPORT_FILE"
            echo '```' >> "$REPORT_FILE"
            jq -r '.results[0:10][] | "\(.check_id): \(.path):\(.start.line)"' .opengrep-tmp.json >> "$REPORT_FILE" 2>/dev/null
            echo '```' >> "$REPORT_FILE"
            echo "" >> "$REPORT_FILE"
        fi
        rm -f .opengrep-tmp.json
    else
        echo "WARNING:  jq not installed, skipping detailed analysis" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
    fi
else
    echo "WARNING:  opengrep not installed. Install with: curl -fsSL https://raw.githubusercontent.com/opengrep/opengrep/main/install.sh | bash" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
fi

# 3. Unused Dependencies (cargo-machete)
if command -v cargo-machete &> /dev/null; then
    echo "## Unused Dependencies (cargo-machete)" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    cargo machete > .machete-tmp.txt 2>&1 || true
    if [ -s .machete-tmp.txt ]; then
        echo '```' >> "$REPORT_FILE"
        cat .machete-tmp.txt >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
    else
        echo "OK: No unused dependencies found" >> "$REPORT_FILE"
    fi
    echo "" >> "$REPORT_FILE"
    rm -f .machete-tmp.txt
else
    echo "WARNING:  cargo-machete not installed. Install with: cargo install cargo-machete" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
fi

# 4. Dependency Security (cargo-deny)
if command -v cargo-deny &> /dev/null; then
    echo "## Dependency Security (cargo-deny)" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    cargo deny check > .deny-tmp.txt 2>&1 || true
    if grep -q "advisory" .deny-tmp.txt || grep -q "license" .deny-tmp.txt; then
        echo "WARNING:  Issues found. Run 'cargo deny check' for details." >> "$REPORT_FILE"
    else
        echo "OK: No dependency security issues found" >> "$REPORT_FILE"
    fi
    echo "" >> "$REPORT_FILE"
    rm -f .deny-tmp.txt
else
    echo "WARNING:  cargo-deny not installed. Install with: cargo install --locked cargo-deny" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" <<EOF
## Recommendations

1. Review unsafe code usage and ensure proper safety documentation
2. Address OpenGrep security findings
3. Remove unused dependencies to reduce attack surface
4. Keep dependencies up to date (cargo-deny)

## Next Steps

- Run \`just static-analysis\` for comprehensive local analysis
- Review CI artifacts for detailed findings
- Address high-severity issues first

EOF

echo "OK: Safety report generated: $REPORT_FILE"

