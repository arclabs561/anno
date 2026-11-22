#!/usr/bin/env bash
# Track unsafe code trends over time using cargo-geiger
# Creative use: generates time-series data for unsafe code usage

set -euo pipefail

TRENDS_DIR=".unsafe-code-trends"
mkdir -p "$TRENDS_DIR"

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
REPORT_FILE="$TRENDS_DIR/trend-$TIMESTAMP.json"

if ! command -v cargo-geiger &> /dev/null; then
    echo "WARNING:  cargo-geiger not installed. Install with: cargo install cargo-geiger"
    exit 1
fi

echo "Generating unsafe code trend snapshot..."

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pending=$(mktemp "$TRENDS_DIR/.pending.XXXXXXXX")
trap 'rm -f "$pending"' EXIT
cargo geiger --output-format json > "$pending"
count_unsafe() {
    jq -er -L "$script_dir" 'include "scan_metrics"; geiger_rows | map(select(.count > 0)) | length' "$1"
}
TOTAL_UNSAFE=$(count_unsafe "$pending")
TOTAL_PACKAGES=$(jq -er '.packages | length' "$pending")
mv "$pending" "$REPORT_FILE"
echo "Snapshot saved: $REPORT_FILE"
echo "Packages with unsafe code: $TOTAL_UNSAFE / $TOTAL_PACKAGES"
for file in "$TRENDS_DIR"/trend-*.json; do
    count=$(count_unsafe "$file") || count="Not measured"
    echo "$(basename "$file" .json) | $count"
done

# Keep only last 30 snapshots
ls -t "$TRENDS_DIR"/trend-*.json 2>/dev/null | tail -n +31 | xargs rm -f 2>/dev/null || true

echo ""
echo "NOTE: Tip: Run this script regularly (e.g., in CI) to track unsafe code trends over time"

