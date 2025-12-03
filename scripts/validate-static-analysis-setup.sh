#!/usr/bin/env bash
# Validate that all static analysis tools are properly configured
# Creative use: ensures CI will work and provides setup instructions

set -euo pipefail

echo "=== Static Analysis Setup Validation ==="
echo ""

ERRORS=0
WARNINGS=0

check_tool() {
    local tool=$1
    local install_cmd=$2
    local required=${3:-false}
    
    if command -v "$tool" &> /dev/null; then
        echo "OK: $tool installed"
        return 0
    else
        if [ "$required" = "true" ]; then
            echo "ERROR: $tool NOT installed (REQUIRED)"
            echo "   Install: $install_cmd"
            ((ERRORS++))
        else
            echo "WARNING:  $tool NOT installed (optional)"
            echo "   Install: $install_cmd"
            ((WARNINGS++))
        fi
        return 1
    fi
}

check_file() {
    local file=$1
    local required=${2:-false}
    
    if [ -f "$file" ]; then
        echo "OK: $file exists"
        return 0
    else
        if [ "$required" = "true" ]; then
            echo "ERROR: $file NOT found (REQUIRED)"
            ((ERRORS++))
        else
            echo "WARNING:  $file NOT found (optional)"
            ((WARNINGS++))
        fi
        return 1
    fi
}

echo "## Required Tools"
check_tool "cargo" "Already installed with Rust" true
check_tool "rustup" "Already installed with Rust" true

echo ""
echo "## Static Analysis Tools"
check_tool "cargo-deny" "cargo install --locked cargo-deny" false
check_tool "cargo-machete" "cargo install cargo-machete" false
check_tool "cargo-geiger" "cargo install cargo-geiger" false
check_tool "cargo-nextest" "cargo install cargo-nextest" false
check_tool "cargo-llvm-cov" "cargo install cargo-llvm-cov" false
check_tool "opengrep" "curl -fsSL https://raw.githubusercontent.com/opengrep/opengrep/main/install.sh | bash" false

echo ""
echo "## Advanced Tools"
check_tool "cargo-miri" "rustup component add miri && cargo install cargo-miri" false
check_tool "jq" "brew install jq (macOS) or apt-get install jq (Linux)" false

echo ""
echo "## Configuration Files"
check_file "deny.toml" false
check_file ".opengrep/rules/rust-security.yaml" false
check_file ".opengrep/rules/rust-anno-specific.yaml" false
check_file ".pre-commit-config.yaml" false

echo ""
echo "## CI Integration"
if [ -f ".github/workflows/ci.yml" ]; then
    if grep -q "cargo-deny" ".github/workflows/ci.yml"; then
        echo "OK: cargo-deny integrated in CI"
    else
        echo "WARNING:  cargo-deny not found in CI workflow"
        ((WARNINGS++))
    fi
    
    if grep -q "opengrep" ".github/workflows/ci.yml"; then
        echo "OK: opengrep integrated in CI"
    else
        echo "WARNING:  opengrep not found in CI workflow"
        ((WARNINGS++))
    fi
else
    echo "WARNING:  CI workflow not found"
    ((WARNINGS++))
fi

echo ""
echo "## Justfile Commands"
if [ -f "justfile" ]; then
    if grep -q "static-analysis" "justfile"; then
        echo "OK: static-analysis command exists"
    else
        echo "WARNING:  static-analysis command not found"
        ((WARNINGS++))
    fi
else
    echo "WARNING:  justfile not found"
    ((WARNINGS++))
fi

echo ""
echo "=== Summary ==="
echo "Errors: $ERRORS"
echo "Warnings: $WARNINGS"
echo ""

if [ $ERRORS -gt 0 ]; then
    echo "ERROR: Setup incomplete. Please install required tools."
    exit 1
elif [ $WARNINGS -gt 0 ]; then
    echo "WARNING:  Setup mostly complete. Some optional tools/configurations missing."
    exit 0
else
    echo "OK: Setup complete! All tools and configurations in place."
    exit 0
fi

