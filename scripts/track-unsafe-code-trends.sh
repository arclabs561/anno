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

# Generate JSON report
cargo geiger --output-format json > "$REPORT_FILE" 2>/dev/null || {
    echo "{}" > "$REPORT_FILE"
}

# Extract summary statistics
if command -v jq &> /dev/null; then
    TOTAL_UNSAFE=$(jq -r '[.packages[] | select(.geiger.unsafe_used > 0)] | length' "$REPORT_FILE" 2>/dev/null || echo "0")
    TOTAL_PACKAGES=$(jq -r '.packages | length' "$REPORT_FILE" 2>/dev/null || echo "0")
    
    echo "Snapshot saved: $REPORT_FILE"
    echo "Packages with unsafe code: $TOTAL_UNSAFE / $TOTAL_PACKAGES"
    
    # Generate trend summary if we have multiple snapshots
    if [ $(ls -1 "$TRENDS_DIR"/trend-*.json 2>/dev/null | wc -l) -gt 1 ]; then
        echo ""
        echo "=== Trend Summary ==="
        echo "Date | Packages with Unsafe" 
        echo "-----|-------------------"
        for file in "$TRENDS_DIR"/trend-*.json; do
            date=$(basename "$file" | sed 's/trend-\(.*\)\.json/\1/')
            count=$(jq -r '[.packages[] | select(.geiger.unsafe_used > 0)] | length' "$file" 2>/dev/null || echo "0")
            echo "$date | $count"
        done | sort
    fi
else
    echo "WARNING:  jq not installed. Install for trend analysis."
    echo "Snapshot saved: $REPORT_FILE"
fi

# Keep only last 30 snapshots
ls -t "$TRENDS_DIR"/trend-*.json 2>/dev/null | tail -n +31 | xargs rm -f 2>/dev/null || true

echo ""
echo "NOTE: Tip: Run this script regularly (e.g., in CI) to track unsafe code trends over time"

