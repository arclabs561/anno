# Creative Uses of Static Analysis Tools

This document describes creative and advanced ways we use static analysis tools in the anno project.

## 1. Comprehensive Safety Report

**Script:** `scripts/generate-safety-report.sh`

**What it does:**
Combines results from multiple tools into a single markdown report:
- Unsafe code statistics (cargo-geiger)
- Security pattern findings (OpenGrep)
- Unused dependencies (cargo-machete)
- Dependency security (cargo-deny)

**Usage:**
```bash
just safety-report-full
```

**Creative aspect:** Single command gives you a complete picture of code safety from multiple angles.

## 2. Tool Performance Benchmarking

**Script:** `scripts/benchmark-static-analysis.sh`

**What it does:**
Measures execution time of each static analysis tool to help decide:
- Which tools are fast enough for every-PR CI
- Which tools should run on schedule
- Which tools are too slow for regular use

**Usage:**
```bash
just benchmark-tools
```

**Creative aspect:** Data-driven decision making about CI pipeline optimization.

## 3. Tool Output Comparison

**Script:** `scripts/compare-tool-outputs.sh`

**What it does:**
Compares findings across different tools to identify:
- Overlapping issues (found by multiple tools)
- Tool-specific insights
- Unique findings per tool

**Usage:**
```bash
just compare-tools
```

**Creative aspect:** Helps understand which tools complement each other and which provide unique value.

## 4. Unsafe Code Trend Tracking

**Script:** `scripts/track-unsafe-code-trends.sh`

**What it does:**
Generates time-series snapshots of unsafe code usage:
- Tracks number of packages with unsafe code over time
- Stores historical data for trend analysis
- Keeps last 30 snapshots

**Usage:**
```bash
just track-unsafe-trends
```

**Creative aspect:** Enables tracking whether unsafe code is increasing or decreasing over time, helping make informed decisions about code safety.

**CI Integration:** Can be run in weekly workflow to track trends automatically.

## 5. Selective Miri Testing

**Implementation:** `.github/workflows/ci.yml` - `miri-unsafe` job

**What it does:**
- Only runs when unsafe code is present
- Can be triggered with PR label `test-unsafe`
- Focuses on files with `unsafe` blocks

**Creative aspect:** Avoids slow CI times by being selective about when to run expensive analysis.

## 6. OpenGrep Custom Rules for Project Patterns

**Files:**
- `.opengrep/rules/rust-security.yaml` - General Rust security
- `.opengrep/rules/rust-anno-specific.yaml` - Project-specific patterns

**What it does:**
Catches project-specific issues like:
- Entity offset validation
- Session pool leaks
- Confidence score range validation
- Graph node validation

**Creative aspect:** Tailored rules for this specific codebase's common patterns and potential issues.

## 7. Pre-commit Integration

**File:** `.pre-commit-config.yaml`

**What it does:**
Runs static analysis before commits:
- Fast tools (cargo-deny, cargo-machete) run always
- OpenGrep runs only on Rust files
- All tools are optional (won't block if not installed)

**Creative aspect:** Catches issues early without being too strict.

## 8. Weekly Comprehensive Analysis

**Workflow:** `.github/workflows/static-analysis-weekly.yml`

**What it does:**
Runs comprehensive analysis weekly:
- All tools
- Trend tracking
- Tool comparison
- Coverage generation
- Benchmarking

**Creative aspect:** Separates fast (PR) checks from comprehensive (weekly) analysis, balancing speed and thoroughness.

## 9. SARIF Integration for GitHub Security Tab

**Implementation:** OpenGrep CI job uploads SARIF

**What it does:**
- Uploads findings to GitHub Security tab
- Makes security issues visible in GitHub UI
- Integrates with GitHub's security features

**Creative aspect:** Leverages GitHub's native security features for better visibility.

## 10. Setup Validation

**Script:** `scripts/validate-static-analysis-setup.sh`

**What it does:**
Validates that all tools and configurations are in place:
- Checks tool installation
- Verifies configuration files
- Confirms CI integration
- Provides setup instructions

**Usage:**
```bash
just validate-setup
```

**Creative aspect:** Helps ensure CI will work and provides clear setup instructions.

## Best Practices

### Fast Feedback Loop
- Use fast tools (cargo-machete, cargo-deny) in pre-commit
- Run comprehensive analysis weekly, not on every PR

### Tool Selection
- Benchmark tools to understand performance
- Compare outputs to avoid redundancy
- Use tool-specific strengths (e.g., Miri for unsafe, OpenGrep for patterns)

### Trend Analysis
- Track metrics over time (unsafe code, findings, etc.)
- Use trends to make informed decisions
- Store historical data for comparison

### CI Optimization
- Separate fast checks from slow analysis
- Use conditional execution (e.g., Miri only when unsafe code changes)
- Upload artifacts for later review

## Future Enhancements

1. **Automated Rule Generation**: Generate OpenGrep rules from common patterns in codebase
2. **ML-Based Issue Prioritization**: Use ML to prioritize findings by severity/impact
3. **Cross-Tool Correlation**: Correlate findings across tools to identify root causes
4. **Historical Comparison**: Compare current findings to historical baselines
5. **Custom Dashboards**: Generate visual dashboards from analysis results

