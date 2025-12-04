#!/usr/bin/env bash
# Generate HTML dashboard from static analysis results
# Creative use: visual dashboard for analysis results

set -euo pipefail

DASHBOARD_FILE="static-analysis-dashboard.html"
TIMESTAMP=$(date -u +"%Y-%m-%d %H:%M:%S UTC")

cat > "$DASHBOARD_FILE" <<'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Static Analysis Dashboard</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card h2 {
            margin-top: 0;
            color: #333;
        }
        .metric {
            display: inline-block;
            margin: 10px 20px 10px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .metric-label {
            color: #666;
            font-size: 0.9em;
        }
        .status-ok { color: #28a745; }
        .status-warn { color: #ffc107; }
        .status-error { color: #dc3545; }
        .tool-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .tool-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }
        .tool-card h3 {
            margin-top: 0;
            font-size: 1.1em;
        }
        pre {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Static Analysis Dashboard</h1>
        <p>Generated: TIMESTAMP_PLACEHOLDER</p>
    </div>
EOF

# Collect metrics
UNSAFE_COUNT=0
FINDINGS_COUNT=0
UNUSED_DEPS=0
TOOLS_INSTALLED=0

if command -v cargo-geiger &> /dev/null; then
    cargo geiger --output-format json > .geiger-tmp.json 2>/dev/null || echo "{}" > .geiger-tmp.json
    if command -v jq &> /dev/null; then
        UNSAFE_COUNT=$(jq -r '[.packages[] | select(.geiger.unsafe_used > 0)] | length' .geiger-tmp.json 2>/dev/null || echo "0")
    fi
    ((TOOLS_INSTALLED++))
    rm -f .geiger-tmp.json
fi

if command -v opengrep &> /dev/null; then
    opengrep scan --config auto --json anno/ anno-core/ anno-coalesce/ anno-strata/ anno-cli/ 2>/dev/null | jq -r '.results | length' > .opengrep-tmp.txt 2>/dev/null || echo "0" > .opengrep-tmp.txt
    FINDINGS_COUNT=$(cat .opengrep-tmp.txt)
    ((TOOLS_INSTALLED++))
    rm -f .opengrep-tmp.txt
fi

if command -v cargo-machete &> /dev/null; then
    cargo machete 2>&1 | grep -c "unused" > .machete-tmp.txt 2>/dev/null || echo "0" > .machete-tmp.txt
    UNUSED_DEPS=$(cat .machete-tmp.txt)
    ((TOOLS_INSTALLED++))
    rm -f .machete-tmp.txt
fi

# Generate dashboard content
cat >> "$DASHBOARD_FILE" <<EOF
    <div class="card">
        <h2>📊 Overview</h2>
        <div class="metric">
            <div class="metric-value">$TOOLS_INSTALLED</div>
            <div class="metric-label">Tools Installed</div>
        </div>
        <div class="metric">
            <div class="metric-value">$UNSAFE_COUNT</div>
            <div class="metric-label">Packages with Unsafe</div>
        </div>
        <div class="metric">
            <div class="metric-value">$FINDINGS_COUNT</div>
            <div class="metric-label">Security Findings</div>
        </div>
        <div class="metric">
            <div class="metric-value">$UNUSED_DEPS</div>
            <div class="metric-label">Unused Dependencies</div>
        </div>
    </div>

    <div class="card">
        <h2>Available Tools</h2>
        <div class="tool-grid">
EOF

# Check each tool
for tool in "cargo-deny:cargo deny check" "cargo-machete:cargo machete" "cargo-geiger:cargo geiger" "opengrep:opengrep scan" "cargo-nextest:cargo nextest" "cargo-llvm-cov:cargo llvm-cov"; do
    tool_name=$(echo $tool | cut -d: -f1)
    tool_cmd=$(echo $tool | cut -d: -f2)
    
    if command -v $tool_name &> /dev/null; then
        status="OK: Installed"
        status_class="status-ok"
    else
        status="ERROR: Not Installed"
        status_class="status-error"
    fi
    
    cat >> "$DASHBOARD_FILE" <<EOF
            <div class="tool-card">
                <h3>$tool_name</h3>
                <p class="$status_class">$status</p>
            </div>
EOF
done

cat >> "$DASHBOARD_FILE" <<EOF
        </div>
    </div>

    <div class="card">
        <h2>📝 Quick Actions</h2>
        <pre>
# Run all static analysis
just static-analysis

# Generate safety report
just safety-report-full

# Validate setup
just validate-setup
        </pre>
    </div>
</body>
</html>
EOF

# Replace timestamp
sed -i.bak "s/TIMESTAMP_PLACEHOLDER/$TIMESTAMP/g" "$DASHBOARD_FILE"
rm -f "${DASHBOARD_FILE}.bak"

echo "OK: Dashboard generated: $DASHBOARD_FILE"
echo "   Open in browser: open $DASHBOARD_FILE"

