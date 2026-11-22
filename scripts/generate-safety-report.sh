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

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
{
    echo "## Unsafe code in packages (cargo-geiger)"
    if command -v cargo-geiger >/dev/null && cargo geiger --output-format json > "$work/geiger.json" &&
       jq -e -L "$script_dir" 'include "scan_metrics"; geiger_rows' "$work/geiger.json" > "$work/packages.json"; then
        jq -r '"Packages with unsafe code: " + (map(select(.count > 0)) | length | tostring), (.[] | select(.count > 0) | "- \(.name): \(.count) unsafe uses")' "$work/packages.json"
    else
        echo "Not measured. The scanner or its report was unavailable or incomplete."
    fi
    echo
    echo "## OpenGrep findings"
    if command -v opengrep >/dev/null && opengrep scan --config auto --json crates/anno/ > "$work/scan.json" &&
       jq -e -L "$script_dir" 'include "scan_metrics"; analyzed_findings' "$work/scan.json" > "$work/findings.json"; then
        jq -r '"Findings: " + (length | tostring), (.[:10][] | "- \(.check_id): \(.path):\(.start.line)")' "$work/findings.json"
    else
        echo "Not measured. The scanner or its report was unavailable or incomplete."
    fi
    echo
    echo "## Unused dependencies"
    if ! command -v cargo-machete >/dev/null; then
        echo "Not measured. cargo-machete is unavailable."
    elif "$script_dir/static-analysis-common.sh" machete > "$work/machete" 2>&1; then
        echo "cargo-machete completed successfully."
    else
        echo "cargo-machete exited $?; see its output."
        head -30 "$work/machete"
    fi
    echo
    echo "## Dependency policies"
    if ! command -v cargo-deny >/dev/null; then
        echo "Not measured. cargo-deny is unavailable."
    elif cargo deny check > "$work/deny" 2>&1; then
        echo "cargo-deny completed successfully."
    else
        echo "cargo-deny exited $?; see its output."
        head -30 "$work/deny"
    fi
    echo
} >> "$REPORT_FILE"

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

