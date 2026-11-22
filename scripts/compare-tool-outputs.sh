#!/usr/bin/env bash
# Compare outputs from different static analysis tools
# Creative use: identifies overlapping findings and tool-specific insights

set -euo pipefail

echo "=== Static Analysis Tool Comparison ==="
echo ""
echo "Comparing findings across tools..."
echo ""

COMPARISON_FILE="tool-comparison.md"
cat > "$COMPARISON_FILE" <<EOF
# Static Analysis Tool Comparison

Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

This report compares findings from different static analysis tools to identify:
- Overlapping issues (found by multiple tools)
- Tool-specific insights
- Unique findings per tool

EOF

# 1. Collect unwrap/expect findings from different tools
echo "## Unwrap/Expect Detection Comparison" >> "$COMPARISON_FILE"
echo "" >> "$COMPARISON_FILE"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

if cargo clippy --all-targets --message-format=json -- -W clippy::unwrap_used 2>/dev/null |
   jq -r 'select(.message.code.code == "clippy::unwrap_used") | .message.spans[] | select(.is_primary) | "\(.file_name):\(.line_start)"' > "$work/clippy"; then
    echo "- Clippy unwrap findings: $(wc -l < "$work/clippy")" >> "$COMPARISON_FILE"
else
    rm -f "$work/clippy"
    echo "- Clippy unwrap findings: Not measured" >> "$COMPARISON_FILE"
fi
if opengrep scan --config auto --quiet --json crates/anno/ 2>/dev/null |
   jq -r -L "$script_dir" 'include "scan_metrics"; analyzed_findings[] | select(.check_id | contains("unwrap")) | "\(.path):\(.start.line)"' > "$work/opengrep"; then
    echo "- OpenGrep unwrap findings: $(wc -l < "$work/opengrep")" >> "$COMPARISON_FILE"
else
    rm -f "$work/opengrep"
    echo "- OpenGrep unwrap findings: Not measured" >> "$COMPARISON_FILE"
fi
if [[ -f "$work/clippy" && -f "$work/opengrep" ]]; then
    overlap=$(comm -12 <(sort -u "$work/clippy") <(sort -u "$work/opengrep") | wc -l)
    echo "- Overlapping locations: $overlap" >> "$COMPARISON_FILE"
fi

# 2. Dependency analysis comparison
echo "## Dependency Analysis Comparison" >> "$COMPARISON_FILE"
echo "" >> "$COMPARISON_FILE"

# cargo-audit vs cargo-deny
if command -v cargo-audit &> /dev/null && command -v cargo-deny &> /dev/null; then
    echo "### Security Advisories" >> "$COMPARISON_FILE"
    echo "" >> "$COMPARISON_FILE"
    echo "- **cargo-audit**: Checks RustSec database" >> "$COMPARISON_FILE"
    echo "- **cargo-deny**: Checks RustSec + additional sources" >> "$COMPARISON_FILE"
    echo "" >> "$COMPARISON_FILE"
fi

# cargo-machete vs cargo-udeps
echo "### Unused Dependencies" >> "$COMPARISON_FILE"
echo "" >> "$COMPARISON_FILE"
echo "- **cargo-machete**: Fast heuristic-based (may have false positives)" >> "$COMPARISON_FILE"
echo "- **cargo-udeps**: Accurate compilation-based (slower, requires nightly)" >> "$COMPARISON_FILE"
echo "" >> "$COMPARISON_FILE"

# 3. Unsafe code analysis
echo "## Unsafe Code Analysis" >> "$COMPARISON_FILE"
echo "" >> "$COMPARISON_FILE"

if command -v cargo-geiger &> /dev/null; then
    echo "### cargo-geiger Statistics" >> "$COMPARISON_FILE"
    echo "" >> "$COMPARISON_FILE"
    cargo geiger --output-format json 2>/dev/null | \
        jq -er -L "$script_dir" 'include "scan_metrics"; geiger_rows | map(select(.count > 0)) | "Total packages with unsafe: \(length)"' \
        >> "$COMPARISON_FILE" 2>/dev/null || echo "Unable to generate statistics" >> "$COMPARISON_FILE"
    echo "" >> "$COMPARISON_FILE"
fi

# 4. Tool-specific insights
echo "## Tool-Specific Insights" >> "$COMPARISON_FILE"
echo "" >> "$COMPARISON_FILE"

echo "### OpenGrep Unique Capabilities" >> "$COMPARISON_FILE"
echo "- Custom pattern matching (YAML rules)" >> "$COMPARISON_FILE"
echo "- Cross-file analysis (limited in OSS version)" >> "$COMPARISON_FILE"
echo "- Security-focused patterns" >> "$COMPARISON_FILE"
echo "" >> "$COMPARISON_FILE"

echo "### Miri Unique Capabilities" >> "$COMPARISON_FILE"
echo "- Undefined behavior detection" >> "$COMPARISON_FILE"
echo "- Memory safety validation" >> "$COMPARISON_FILE"
echo "- Data race detection" >> "$COMPARISON_FILE"
echo "" >> "$COMPARISON_FILE"

echo "### Clippy Unique Capabilities" >> "$COMPARISON_FILE"
echo "- Rust-specific idioms" >> "$COMPARISON_FILE"
echo "- Performance suggestions" >> "$COMPARISON_FILE"
echo "- Style enforcement" >> "$COMPARISON_FILE"
echo "" >> "$COMPARISON_FILE"

# Cleanup


echo "OK: Comparison report generated: $COMPARISON_FILE"

