#!/usr/bin/env bash
# Validate publish readiness for all crates in the workspace
# Checks cargo publish --dry-run and version requirements

set -euo pipefail

echo "=== Publish Validation ==="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
ERRORS=0
WARNINGS=0

# Function to check publish readiness
check_publish() {
    local crate=$1
    local crate_path=$2
    
    echo "Checking $crate..."
    
    if [ -d "$crate_path" ]; then
        if cargo publish --dry-run -p "$crate" 2>&1 | grep -q "aborting upload due to dry run"; then
            echo -e "${GREEN}✅ $crate: Ready to publish${NC}"
            return 0
        else
            echo -e "${RED}❌ $crate: Publish validation failed${NC}"
            cargo publish --dry-run -p "$crate" 2>&1 | grep -E "(error|warning)" | head -5 || true
            ((ERRORS++))
            return 1
        fi
    else
        echo -e "${YELLOW}⚠️  $crate: Directory not found${NC}"
        ((WARNINGS++))
        return 1
    fi
}

# Function to check version requirements
check_version_requirement() {
    local file=$1
    local dep_name=$2
    local expected_version=$3
    
    if [ ! -f "$file" ]; then
        echo -e "${YELLOW}⚠️  $file: File not found${NC}"
        ((WARNINGS++))
        return 1
    fi
    
    if grep -q "$dep_name.*version.*=.*\"$expected_version\"" "$file" 2>/dev/null || \
       grep -q "$dep_name.*version.*workspace" "$file" 2>/dev/null; then
        echo -e "${GREEN}✅ $file: Version requirement for $dep_name present${NC}"
        return 0
    else
        echo -e "${RED}❌ $file: Missing version requirement for $dep_name${NC}"
        ((ERRORS++))
        return 1
    fi
}

# Check workspace structure
echo "## Workspace Structure"
cargo metadata --format-version 1 | jq -r '.workspace_members[]' | sort
echo ""

# Check publish readiness
echo "## Publish Readiness"
echo ""

check_publish "anno-core" "anno-core"
check_publish "anno-coalesce" "coalesce"
check_publish "anno-strata" "strata"
check_publish "anno" "anno"

echo ""

# Check version requirements
echo "## Version Requirements"
echo ""

check_version_requirement "coalesce/Cargo.toml" "anno-core" "0.2.0"
check_version_requirement "strata/Cargo.toml" "anno-core" "0.2.0"
check_version_requirement "anno/Cargo.toml" "anno-coalesce" "0.2.0"
check_version_requirement "anno/Cargo.toml" "anno-strata" "0.2.0"

echo ""

# Check crates.io status
echo "## Crates.io Status"
echo ""

for crate in anno anno-core anno-coalesce anno-strata; do
    if curl -s "https://crates.io/api/v1/crates/$crate" 2>/dev/null | jq -r '.crate | "\(.name): v\(.max_version // "not published")"' 2>/dev/null | grep -q "not published"; then
        echo -e "${YELLOW}⚠️  $crate: Not published${NC}"
    else
        version=$(curl -s "https://crates.io/api/v1/crates/$crate" 2>/dev/null | jq -r '.crate.max_version // "unknown"')
        echo -e "${GREEN}✅ $crate: Published (v$version)${NC}"
    fi
done

echo ""

# Summary
echo "## Summary"
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed!${NC}"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠️  $WARNINGS warning(s), but no errors${NC}"
    exit 0
else
    echo -e "${RED}❌ $ERRORS error(s), $WARNINGS warning(s)${NC}"
    echo ""
    echo "See docs/PUBLISH_STATUS.md for details on fixing publish issues."
    exit 1
fi

