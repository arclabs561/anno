# OpenGrep Integration Research

## Executive Summary

OpenGrep is a static code analysis tool (fork of Semgrep) that can find security vulnerabilities and code patterns across 30+ languages, including Rust. This document researches integration options for the `anno` repository.

**Key Findings:**
- OpenGrep supports Rust (experimental but functional)
- LGPL-2.1 license is compatible with MIT/Apache-2.0 (can use as external tool)
- Can be integrated via CI/CD, pre-commit hooks, or local development
- Provides security-focused static analysis complementary to existing Rust tooling (clippy, rustfmt, cargo-audit)

## What is OpenGrep?

OpenGrep is a fork of Semgrep created by a collective of AppSec organizations (Aikido.dev, Arnica, Amplify, Endor, Jit, Kodem, Mobb, Orca Security) to provide open-source static analysis under LGPL-2.1 license.

**Key Features:**
- Pattern-based code search (semantic grep)
- Security vulnerability detection
- Custom rule creation (YAML-based)
- SARIF output support
- 30+ language support including Rust

**Installation:**
```bash
# Quick install script
curl -fsSL https://raw.githubusercontent.com/opengrep/opengrep/main/install.sh | bash

# Or download binary from releases
# https://github.com/opengrep/opengrep/releases
```

## License Compatibility

**OpenGrep License:** LGPL-2.1  
**Anno License:** MIT OR Apache-2.0

**Analysis:**
- LGPL-2.1 allows linking with proprietary code
- Using OpenGrep as an external tool (CLI) doesn't create license conflicts
- If we were to embed OpenGrep as a library, we'd need to consider LGPL requirements
- **Recommendation:** Use as external CLI tool (no license issues)

## Current Static Analysis in Anno

**Existing Tools:**
1. **rustfmt** - Code formatting
2. **clippy** - Linting and style checks
3. **cargo-audit** - Security vulnerability scanning (via `rustsec/audit-check`)
4. **cargo check** - Compilation checks

**Gaps OpenGrep Could Fill:**
- Custom security patterns (beyond cargo-audit's dependency scanning)
- Code pattern detection (e.g., unsafe unwrap patterns, error handling anti-patterns)
- Cross-file analysis (though limited in open-source version)
- Custom rule enforcement for project-specific standards

## Integration Options

### Option 1: CI/CD Integration (Recommended)

Add OpenGrep as a GitHub Actions job in `.github/workflows/ci.yml`:

```yaml
  opengrep:
    name: OpenGrep Static Analysis
    runs-on: ubuntu-latest
    continue-on-error: true  # Don't fail CI on findings
    steps:
      - uses: actions/checkout@v4
      
      - name: Install OpenGrep
        run: |
          curl -fsSL https://raw.githubusercontent.com/opengrep/opengrep/main/install.sh | bash
          echo "$HOME/.local/bin" >> $GITHUB_PATH
      
      - name: Run OpenGrep scan
        run: |
          opengrep scan \
            --config auto \
            --json --output opengrep-results.json \
            --sarif-output opengrep-results.sarif \
            src/ tests/ examples/
      
      - name: Upload SARIF results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: opengrep-results.sarif
      
      - name: Upload JSON results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: opengrep-results
          path: opengrep-results.json
```

**Pros:**
- Automated scanning on every PR/push
- Results visible in GitHub Security tab (via SARIF upload)
- Non-blocking (can use `continue-on-error: true`)

**Cons:**
- Adds ~30-60 seconds to CI pipeline
- Requires network access to download OpenGrep

### Option 2: Pre-commit Hook

Add to `.pre-commit-config.yaml` (if using pre-commit framework):

```yaml
repos:
  - repo: https://github.com/opengrep/opengrep
    rev: v1.12.1
    hooks:
      - id: opengrep
        args: ['--config', 'auto', '--json']
```

**Pros:**
- Catches issues before commit
- Fast feedback loop

**Cons:**
- Requires developers to install pre-commit framework
- May slow down commit process

### Option 3: Local Development Tool

Add to `justfile` for local scanning:

```makefile
# Run OpenGrep static analysis
opengrep:
    @which opengrep > /dev/null || (echo "Install opengrep: curl -fsSL https://raw.githubusercontent.com/opengrep/opengrep/main/install.sh | bash" && exit 1)
    opengrep scan --config auto --json --output opengrep-results.json src/ tests/ examples/
    @echo "Results saved to opengrep-results.json"
    @cat opengrep-results.json | jq '.results | length' | xargs -I {} echo "Found {} issues"

# Run with custom rules
opengrep-custom:
    opengrep scan -f .opengrep/rules --json --output opengrep-results.json src/
```

**Pros:**
- Developer-friendly
- Optional (doesn't block workflow)
- Can be run on-demand

**Cons:**
- Requires manual execution
- No automated enforcement

## Custom Rules for Rust

Create `.opengrep/rules/rust-security.yaml`:

```yaml
rules:
  - id: dangerous-unwrap
    pattern: $VAR.unwrap()
    message: "Unwrap detected - potential panic risk. Consider using expect() with context or proper error handling."
    languages: [rust]
    severity: WARNING
    metadata:
      category: reliability
      cwe: "CWE-248: Uncaught Exception"

  - id: unsafe-block
    pattern: unsafe { $BODY }
    message: "Unsafe block detected. Ensure proper safety invariants are documented."
    languages: [rust]
    severity: INFO
    metadata:
      category: safety

  - id: unwrap-in-production
    patterns:
      - pattern-either:
          - pattern: $VAR.unwrap()
          - pattern: $VAR.expect(...)
      - not:
          - pattern: |
              #[cfg(test)]
              ...
    message: "Unwrap/expect in non-test code. Consider proper error handling."
    languages: [rust]
    severity: WARNING

  - id: panic-in-library
    patterns:
      - pattern: panic!(...)
      - not:
          - pattern: |
              #[cfg(test)]
              ...
    message: "Panic in library code. Libraries should return Result types instead."
    languages: [rust]
    severity: ERROR
    metadata:
      category: reliability

  - id: missing-error-handling
    pattern: |
      fn $F(...) -> $TYPE {
        $VAR.unwrap()
      }
    message: "Function returns non-Result but uses unwrap. Consider returning Result<...>."
    languages: [rust]
    severity: WARNING
```

## Rust-Specific Considerations

**Important Limitations:**
1. **Pattern Context:** Rust patterns must be valid Rust syntax within their context. Standalone statements won't work - they need to be within functions, blocks, etc.

2. **Experimental Support:** Rust support in Semgrep/OpenGrep is marked as experimental. Some complex patterns may not work perfectly.

3. **Macro Handling:** Rust macros are expanded before analysis, so pattern matching on macro invocations may be limited.

**Best Practices:**
- Start with simple patterns (unwrap detection, unsafe blocks)
- Test rules on your codebase before committing
- Use metavariables (`$VAR`, `$F`) for flexibility
- Combine with existing Rust tooling (clippy, rustfmt) rather than replacing

## Additional Rust Static Analysis Tools

Beyond the standard `rustfmt` and `clippy`, there are many specialized tools for different aspects of Rust code quality. Here's a comprehensive overview:

### Dependency Management Tools

#### cargo-deny
**Purpose:** Comprehensive dependency linting (licenses, bans, advisories, duplicates)

**Installation:**
```bash
cargo install --locked cargo-deny
```

**Setup:**
```bash
cargo deny init  # Creates deny.toml
cargo deny check
```

**Capabilities:**
- Security vulnerability auditing (complements cargo-audit)
- License compatibility checking (SPDX standards)
- Dependency banning (block specific crates)
- Duplicate dependency detection
- Source verification

**CI Integration:**
```yaml
- name: Install cargo-deny
  run: cargo install --locked cargo-deny
- name: Check dependencies
  run: cargo deny check
```

**Recommendation:** ✅ **High Priority** - More comprehensive than cargo-audit alone

#### cargo-udeps
**Purpose:** Accurate unused dependency detection

**Installation:**
```bash
cargo install cargo-udeps --locked
```

**Usage:**
```bash
cargo +nightly udeps
```

**Characteristics:**
- Very accurate (compiles entire crate)
- Requires nightly compiler
- Slow (compiles from scratch)
- Best for periodic checks, not every CI run

**Recommendation:** ⚠️ **Low Priority** - Too slow for CI, use cargo-machete instead

#### cargo-machete
**Purpose:** Fast unused dependency detection

**Installation:**
```bash
cargo install cargo-machete
```

**Usage:**
```bash
cargo machete
```

**Characteristics:**
- Very fast (searches `use` statements, no compilation)
- Some false positives (but acceptable for CI)
- Works with stable Rust
- Perfect for CI/CD pipelines

**Recommendation:** ✅ **Medium Priority** - Good for CI, faster than cargo-udeps

#### cargo-geiger
**Purpose:** Count unsafe code in dependency tree

**Installation:**
```bash
cargo install cargo-geiger
```

**Usage:**
```bash
cargo geiger
```

**Characteristics:**
- Tracks unsafe code exposure
- Helps understand safety surface area
- Useful for security audits

**Recommendation:** ⚠️ **Low Priority** - Informational only, not blocking

### Advanced Analysis Tools

#### Miri
**Purpose:** Interpreter that detects undefined behavior in unsafe code

**Installation:**
```bash
rustup component add miri
cargo install cargo-miri
```

**Usage:**
```bash
cargo miri test
```

**Capabilities:**
- Detects memory leaks, data races, undefined behavior
- Runs tests in interpreted mode
- Catches issues that normal tests miss

**CI Integration:**
```yaml
- name: Install Miri
  run: |
    rustup component add miri
    cargo install cargo-miri
- name: Run Miri tests
  run: cargo miri test
```

**Recommendation:** ✅ **High Priority** - Essential for unsafe code validation

#### MIRAI
**Purpose:** Abstract interpretation-based analyzer for unsafe code and soundness

**Installation:**
```bash
cargo install --locked mirai-cargo
```

**Usage:**
```bash
cargo mirai
```

**Characteristics:**
- Lightweight formal verification
- Contract-based (preconditions, postconditions)
- Good for simple requirements
- No soundness guarantees over symbolic inputs

**Recommendation:** ⚠️ **Low Priority** - Niche use case, consider Kani/Prusti instead

#### Prusti
**Purpose:** Formal verification using contracts/specifications

**Installation:**
```bash
# Requires Rust nightly
cargo install prusti-rustc
```

**Characteristics:**
- Rigorous formal verification
- Proves correctness under all conditions
- Reduces annotation burden via type system
- Does not verify unsafe blocks
- Limited CI/CD support

**Recommendation:** ⚠️ **Very Low Priority** - Research tool, not practical for most projects

#### Kani
**Purpose:** Model checker for Rust (bounded model checking)

**Installation:**
```bash
cargo install --locked kani-verifier
```

**Usage:**
```bash
cargo kani
```

**Characteristics:**
- Systematic exploration of program states
- Used by AWS Firecracker team
- Proves properties about code
- Does not support concurrent code verification

**Recommendation:** ⚠️ **Low Priority** - Advanced use case, consider for critical code paths

#### Flowistry
**Purpose:** Dataflow visualization for ownership/borrowing patterns

**Installation:**
```bash
cargo install flowistry
```

**Characteristics:**
- Visualizes data movement through code
- Helps debug borrow checker errors
- Learning tool for ownership system
- Not a linting tool
- Performance degrades on large codebases

**Recommendation:** ⚠️ **Very Low Priority** - Development tool, not CI/CD

### Development Workflow Tools

#### Dylint
**Purpose:** Write custom Rust lints as dynamic libraries

**Installation:**
```bash
cargo install cargo-dylint dylint-link
```

**Characteristics:**
- Create custom lints without forking Clippy
- Dynamic loading of lint libraries
- Useful for project-specific rules

**Recommendation:** ⚠️ **Low Priority** - Only if custom lints needed beyond Clippy

#### cargo-nextest
**Purpose:** Faster test runner with better output

**Installation:**
```bash
cargo install cargo-nextest
```

**Usage:**
```bash
cargo nextest run
```

**Characteristics:**
- Parallel test execution
- Better test output formatting
- Faster than standard `cargo test`

**Recommendation:** ✅ **Medium Priority** - Improves developer experience

#### cargo-llvm-cov
**Purpose:** Code coverage via LLVM instrumentation

**Installation:**
```bash
cargo install cargo-llvm-cov
```

**Usage:**
```bash
cargo llvm-cov --all-features --workspace
```

**Recommendation:** ✅ **Medium Priority** - Useful for coverage tracking

#### bacon
**Purpose:** Background code checker, reruns on save

**Installation:**
```bash
cargo install bacon
```

**Characteristics:**
- Faster feedback than cargo-watch
- Reruns checks automatically
- Development tool only

**Recommendation:** ⚠️ **Very Low Priority** - Personal preference tool

#### taplo
**Purpose:** TOML formatter/linter for Cargo.toml

**Installation:**
```bash
cargo install taplo-cli
```

**Usage:**
```bash
taplo format Cargo.toml
taplo lint Cargo.toml
```

**Recommendation:** ⚠️ **Low Priority** - Nice to have, not essential

## Comprehensive Tool Comparison

| Tool | Purpose | CI/CD | Priority | Speed | Notes |
|------|---------|-------|----------|-------|-------|
| **rustfmt** | Formatting | ✅ | Required | Fast | Already in use |
| **clippy** | Linting | ✅ | Required | Fast | Already in use |
| **cargo-audit** | Security advisories | ✅ | Required | Fast | Already in use |
| **cargo-deny** | Dependency linting | ✅ | **High** | Medium | More comprehensive than audit |
| **cargo-machete** | Unused deps (fast) | ✅ | Medium | Very Fast | Good for CI |
| **cargo-udeps** | Unused deps (accurate) | ⚠️ | Low | Slow | Too slow for CI |
| **cargo-geiger** | Unsafe code stats | ✅ | Low | Medium | Informational |
| **Miri** | Undefined behavior | ✅ | **High** | Slow | Essential for unsafe code |
| **MIRAI** | Abstract interpretation | ⚠️ | Low | Medium | Niche use case |
| **Prusti** | Formal verification | ⚠️ | Very Low | Very Slow | Research tool |
| **Kani** | Model checking | ⚠️ | Low | Very Slow | Advanced use case |
| **Flowistry** | Dataflow viz | ❌ | Very Low | Medium | Dev tool only |
| **Dylint** | Custom lints | ⚠️ | Low | Medium | Only if needed |
| **cargo-nextest** | Test runner | ✅ | Medium | Fast | Better DX |
| **cargo-llvm-cov** | Coverage | ✅ | Medium | Medium | Coverage tracking |
| **bacon** | Background checker | ❌ | Very Low | N/A | Dev tool |
| **taplo** | TOML formatter | ⚠️ | Low | Fast | Nice to have |
| **OpenGrep** | Security patterns | ✅ | Medium | Medium | Custom security rules |

## Recommended Integration Priority

### Phase 1: Essential (Already Have)
- ✅ rustfmt
- ✅ clippy
- ✅ cargo-audit

### Phase 2: High Value Additions
1. **cargo-deny** - Comprehensive dependency checking
2. **Miri** - Undefined behavior detection (if using unsafe code)

### Phase 3: Quality of Life
3. **cargo-machete** - Fast unused dependency detection
4. **cargo-nextest** - Better test runner
5. **cargo-llvm-cov** - Code coverage tracking

### Phase 4: Optional/Advanced
6. **OpenGrep** - Custom security patterns
7. **cargo-geiger** - Unsafe code statistics
8. **taplo** - TOML formatting

### Not Recommended for CI/CD
- **cargo-udeps** - Too slow (use cargo-machete instead)
- **MIRAI/Prusti/Kani** - Research/advanced verification tools
- **Flowistry** - Development visualization tool
- **bacon** - Personal development tool
- **Dylint** - Only if custom lints needed

## Implementation Recommendation

**Phase 1: CI/CD Integration (Low Risk)**
1. Add OpenGrep job to CI workflow
2. Use `--config auto` (default security rules)
3. Set `continue-on-error: true` initially
4. Upload SARIF results to GitHub Security tab

**Phase 2: Custom Rules (Medium Risk)**
1. Create `.opengrep/rules/` directory
2. Add Rust-specific security rules
3. Test rules locally before committing
4. Gradually tighten severity levels

**Phase 3: Pre-commit (Optional)**
1. Add pre-commit hook if team adopts pre-commit framework
2. Make it optional (not blocking)

## Example Workflow

```bash
# 1. Install OpenGrep locally (one-time)
curl -fsSL https://raw.githubusercontent.com/opengrep/opengrep/main/install.sh | bash

# 2. Test default rules
opengrep scan --config auto src/

# 3. Create custom rules directory
mkdir -p .opengrep/rules

# 4. Add custom rules (see rust-security.yaml above)

# 5. Test custom rules
opengrep scan -f .opengrep/rules src/

# 6. Integrate into CI (see CI example above)
```

## Cost/Benefit Analysis

**Benefits:**
- ✅ Additional security layer beyond cargo-audit
- ✅ Custom rule enforcement for project standards
- ✅ SARIF integration with GitHub Security tab
- ✅ No license conflicts (external tool)
- ✅ Free and open-source

**Costs:**
- ⚠️ ~30-60s added to CI pipeline
- ⚠️ Learning curve for custom rule creation
- ⚠️ Rust support is experimental (may have edge cases)
- ⚠️ Maintenance overhead (updating rules, OpenGrep version)

**Recommendation:** 
Start with Phase 1 (CI integration with default rules). This provides immediate value with minimal risk. Add custom rules (Phase 2) only if default rules prove useful and team wants more control.

## References

- [OpenGrep GitHub](https://github.com/opengrep/opengrep)
- [OpenGrep Manifesto](https://opengrep.dev/)
- [Semgrep Rust Support](https://kudelskisecurity.com/research/advancing-rust-support-in-semgrep)
- [LGPL-2.1 License](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html)
- [SARIF Format](https://docs.github.com/en/code-security/code-scanning/integrating-with-code-scanning/sarif-support-for-code-scanning)

## Recommended CI/CD Integration Plan

### Immediate Additions (High ROI)

**1. cargo-deny** - Replace/enhance cargo-audit
```yaml
cargo-deny:
  name: Cargo Deny
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@stable
    - run: cargo install --locked cargo-deny
    - run: cargo deny check
```

**2. Miri** - If using unsafe code
```yaml
miri:
  name: Miri (undefined behavior)
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@stable
    - run: rustup component add miri
    - run: cargo miri test --lib
```

**3. cargo-machete** - Fast unused dependency check
```yaml
unused-deps:
  name: Unused Dependencies
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@stable
    - run: cargo install cargo-machete
    - run: cargo machete
```

### Optional Additions (Medium ROI)

**4. cargo-nextest** - Better test output
```yaml
test-nextest:
  name: Tests (nextest)
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@stable
    - run: cargo install cargo-nextest
    - run: cargo nextest run --all-features
```

**5. cargo-llvm-cov** - Code coverage
```yaml
coverage:
  name: Code Coverage
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@stable
    - run: cargo install cargo-llvm-cov
    - run: cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info
    - uses: codecov/codecov-action@v3
      with:
        files: ./lcov.info
```

**6. OpenGrep** - Custom security patterns (see earlier section)

## Configuration Files

### deny.toml (cargo-deny)
```toml
[advisories]
vulnerability = "deny"
unmaintained = "warn"
notice = "warn"
ignore = []

[licenses]
default = "deny"
copyleft = "deny"
allow = ["MIT", "Apache-2.0"]
deny = ["GPL-3.0"]

[bans]
multiple-versions = "deny"
wildcards = "deny"
```

### .cargo/config.toml (optional)
```toml
[build]
# Enable unused dependency warnings
rustflags = ["-W", "unused-crate-dependencies"]
```

### clippy.toml (enhanced)
```toml
cognitive-complexity-threshold = 25
too-many-arguments-threshold = 8
```

## Next Steps

1. **Immediate:** Add cargo-deny to CI (replaces/enhances cargo-audit)
2. **If using unsafe code:** Add Miri to CI
3. **Quality of life:** Add cargo-machete for unused dependency checks
4. **Optional:** Evaluate cargo-nextest and cargo-llvm-cov
5. **Future:** Consider OpenGrep if custom security patterns needed

