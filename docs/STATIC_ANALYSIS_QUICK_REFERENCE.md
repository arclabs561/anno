# Static Analysis Quick Reference

Quick commands for common static analysis tasks.

## One-Liners

```bash
# Run all static analysis
just static-analysis

# Generate comprehensive safety report
just safety-report-full

# Quick pre-commit check
just pre-commit-check

# Full analysis (everything)
just analysis-full

# Validate setup
just validate-setup
```

## Individual Tools

```bash
# Dependency security
just deny

# Unused dependencies (fast)
just machete

# Unsafe code statistics
just geiger

# Security patterns
just opengrep
just opengrep-custom

# Undefined behavior (unsafe code)
just miri-unsafe

# Better test output
just test-nextest

# Code coverage
just coverage
```

## Creative Tools

```bash
# Benchmark tool performance
just benchmark-tools

# Compare tool outputs
just compare-tools

# Track unsafe code trends
just track-unsafe-trends
```

## CI Integration

All tools are integrated into GitHub Actions:
- **Every PR/Push**: cargo-deny, unused-deps, safety-report, opengrep
- **Weekly**: Comprehensive analysis (see `.github/workflows/static-analysis-weekly.yml`)
- **On-demand**: Miri (when unsafe code changes), coverage (manual trigger)

## Installation

```bash
# Required
cargo install --locked cargo-deny
cargo install cargo-machete
cargo install cargo-geiger
cargo install cargo-nextest
cargo install cargo-llvm-cov
rustup component add miri
cargo install cargo-miri

# OpenGrep
curl -fsSL https://raw.githubusercontent.com/opengrep/opengrep/main/install.sh | bash

# Optional helpers
brew install jq  # macOS
# or
apt-get install jq  # Linux
```

## Pre-commit Setup

```bash
pip install pre-commit
pre-commit install
```

## Troubleshooting

**Tool not found?**
- Run `just validate-setup` to see what's missing
- Check installation commands above

**CI failing?**
- All static analysis jobs use `continue-on-error: true`
- Check artifacts for detailed results

**Slow CI?**
- Fast tools run on every PR
- Slow tools (Miri, coverage) run on schedule or manual trigger

## File Locations

- Configuration: `deny.toml`, `.opengrep/rules/*.yaml`
- Scripts: `scripts/*.sh`
- Reports: `safety-report.md`, `tool-comparison.md`, `.unsafe-code-trends/`
- CI: `.github/workflows/ci.yml`, `.github/workflows/static-analysis-weekly.yml`

