#!/usr/bin/env bash
# Run report entry points with measured, missing, and malformed scanner output.
set -euo pipefail
scripts=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
mkdir bin
# Limit tool discovery so this regression never needs a toolchain or remote rules.
for tool in bash dirname date cat jq sed rm head mktemp basename mv ls tail wc xargs mkdir tr sort comm ln chmod grep cut; do
    ln -s "$(command -v "$tool")" "bin/$tool"
done
export PATH="$work/bin"
bash "$scripts/generate-analysis-dashboard.sh" >/dev/null
grep -q 'metric-value">Not measured' static-analysis-dashboard.html

cat > bin/cargo <<'MOCK'
#!/bin/sh
case "$1" in
    geiger) cat geiger.json ;;
    *) exit 2 ;;
esac
MOCK
cat > bin/opengrep <<'MOCK'
#!/bin/sh
cat scan.json
MOCK
chmod +x bin/cargo bin/opengrep
ln -s cargo bin/cargo-geiger
jq -n '{safe: 0, unsafe_: 0} as $zero | {packages:[{package:{id:{name:"demo"}},unsafety:{used:{functions:$zero,exprs:{safe:0,unsafe_:2},item_impls:$zero,item_traits:$zero,methods:$zero}}}],packages_without_metrics:[],used_but_not_scanned_files:[]}' > geiger.json
echo '{"results":[],"errors":[]}' > scan.json
bash "$scripts/generate-safety-report.sh" >/dev/null
grep -q 'Packages with unsafe code: 1' safety-report.md
grep -q 'Findings: 0' safety-report.md
bash "$scripts/track-unsafe-code-trends.sh" >/dev/null
[[ -s "$(echo .unsafe-code-trends/trend-*.json)" ]]
rm -f .unsafe-code-trends/trend-*.json
# Partial JSON from a scanner lacks the arrays that attest coverage.
jq 'del(.used_but_not_scanned_files)' geiger.json > partial.json
mv partial.json geiger.json
echo '{"results":[]}' > scan.json
bash "$scripts/generate-safety-report.sh" >/dev/null
[[ $(grep -c 'Not measured. The scanner' safety-report.md) == 2 ]]
echo '{}' > geiger.json
echo '{"results":[],"errors":["scan failed"]}' > scan.json
bash "$scripts/generate-safety-report.sh" >/dev/null
[[ $(grep -c 'Not measured. The scanner' safety-report.md) == 2 ]]
if bash "$scripts/track-unsafe-code-trends.sh" >/dev/null 2>&1; then exit 1; fi
[[ ! -e "$(echo .unsafe-code-trends/trend-*.json)" ]]
rm -f geiger.json scan.json
bash "$scripts/compare-tool-outputs.sh" >/dev/null
grep -q 'OpenGrep unwrap findings: Not measured' tool-comparison.md
