# Static Analysis Tools Setup

This document describes the static analysis tools integrated into the `anno` project and how to use them.

## Quick Start

Run all static analysis tools:
```bash
just static-analysis
```

Generate comprehensive safety report:
```bash
just safety-report-full
```

## Installed Tools

### 1. cargo-deny
**Purpose:** Comprehensive dependency linting (licenses, security, duplicates)

**Configuration:** `deny.toml`

**Usage:**
```bash
just deny
# or
cargo deny check
```

**What it checks:**
- Security vulnerabilities (advisories)
- License compatibility
- Duplicate dependencies
- Wildcard version requirements
- Unknown sources

### 2. cargo-machete
**Purpose:** Fast unused dependency detection

**Usage:**
```bash
just machete
# or
cargo machete
```

**Note:** Fast but may have false positives. For accurate results, use `cargo-udeps` (slower, requires nightly).

### 3. cargo-geiger
**Purpose:** Count unsafe code in dependency tree

**Usage:**
```bash
just geiger
# or
cargo geiger
```

**Creative Use:** Generate safety report with statistics:
```bash
just safety-report
```

### 4. OpenGrep
**Purpose:** Security pattern detection with custom rules

**Installation:**
```bash
curl -fsSL https://raw.githubusercontent.com/opengrep/opengrep/main/install.sh | bash
```

**Usage:**
```bash
# Default security rules
just opengrep

# Custom rules (project-specific)
just opengrep-custom
```

**Custom Rules:**
- `.opengrep/rules/rust-security.yaml` - General Rust security patterns
- `.opengrep/rules/rust-anno-specific.yaml` - Anno-specific patterns

### 5. Miri
**Purpose:** Undefined behavior detection in unsafe code

**Usage:**
```bash
just miri-unsafe
# or
cargo miri test --lib --features onnx
```

**Note:** Runs selectively on unsafe code paths. Slow but catches memory safety issues.

### 6. cargo-nextest
**Purpose:** Faster test runner with better output

**Usage:**
```bash
just test-nextest
# or
cargo nextest run --all-features
```

### 7. cargo-llvm-cov
**Purpose:** Code coverage via LLVM instrumentation

**Usage:**
```bash
just coverage
# or
cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info
```

## CI/CD Integration

All tools are integrated into GitHub Actions (`.github/workflows/ci.yml`):

1. **cargo-deny** - Runs on every PR/push
2. **unused-deps** - Fast check for unused dependencies
3. **safety-report** - Generates unsafe code statistics
4. **miri-unsafe** - Validates unsafe code (selective, on-demand)
5. **opengrep** - Security pattern detection with SARIF upload
6. **coverage** - Code coverage (runs on schedule/manual trigger)

## Creative Integrations

### Comprehensive Safety Report

The `generate-safety-report.sh` script combines multiple tools:

```bash
just safety-report-full
```

This generates `safety-report.md` with:
- Unsafe code statistics (cargo-geiger)
- Security pattern findings (OpenGrep)
- Unused dependencies (cargo-machete)
- Dependency security status (cargo-deny)

### Selective Miri Testing

Miri runs selectively on unsafe code files to avoid long CI times:
- Only runs when unsafe code changes
- Can be triggered with label `test-unsafe` on PRs
- Focuses on files with `unsafe` blocks

### OpenGrep Custom Rules

Two rule sets:
1. **rust-security.yaml** - General Rust security patterns
   - Dangerous unwrap patterns
   - Unsafe code without documentation
   - Mutex poison handling
   - Unchecked array access

2. **rust-anno-specific.yaml** - Project-specific patterns
   - Entity offset validation
   - Session pool leaks
   - Confidence score range validation
   - Graph node validation

## Workflow Recommendations

### Daily Development
```bash
# Quick check before commit
just check          # fmt + clippy + tests
just machete        # Fast unused deps check
```

### Pre-Push
```bash
# Comprehensive check
just static-analysis
```

### Weekly Review
```bash
# Full safety audit
just safety-report-full
just coverage
```

### Before Release
```bash
# Everything
just ci-eval        # Full CI simulation
just static-analysis
just safety-report-full
just coverage
```

## Tool Comparison

| Tool | Speed | Accuracy | CI-Friendly | Purpose |
|------|-------|----------|-------------|---------|
| cargo-deny | Medium | High | OK: | Dependency security |
| cargo-machete | Very Fast | Good | OK: | Unused deps (fast) |
| cargo-udeps | Slow | Excellent | WARNING: | Unused deps (accurate) |
| cargo-geiger | Medium | High | OK: | Unsafe code stats |
| OpenGrep | Medium | High | OK: | Security patterns |
| Miri | Slow | Excellent | WARNING: | Undefined behavior |
| cargo-nextest | Fast | High | OK: | Test runner |
| cargo-llvm-cov | Medium | High | OK: | Coverage |

## Troubleshooting

### Tool Not Found
Most tools need to be installed:
```bash
cargo install --locked cargo-deny
cargo install cargo-machete
cargo install cargo-geiger
cargo install cargo-nextest
cargo install cargo-llvm-cov
rustup component add miri
```

### OpenGrep Installation
```bash
curl -fsSL https://raw.githubusercontent.com/opengrep/opengrep/main/install.sh | bash
```

### CI Failures
All static analysis jobs use `continue-on-error: true` so they won't block CI. Check artifacts for details.

## References

- [cargo-deny docs](https://github.com/EmbarkStudios/cargo-deny)
- [cargo-machete](https://github.com/bnjbvr/cargo-machete)
- [cargo-geiger](https://github.com/rust-secure-code/cargo-geiger)
- [OpenGrep](https://github.com/opengrep/opengrep)
- [Miri](https://github.com/rust-lang/miri)
- [cargo-nextest](https://nexte.st/)
- [cargo-llvm-cov](https://github.com/taiki-e/cargo-llvm-cov)

