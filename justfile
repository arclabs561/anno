# anno development tasks
# Run `just` to see available commands

default:
    @just --list

# === Quick Commands ===

# Run fast checks (fmt + clippy + quick tests)
check:
    cargo fmt --all -- --check
    cargo clippy --all-targets
    cargo test --lib

# Format all code
fmt:
    cargo fmt --all

# Check formatting without modifying
fmt-check:
    cargo fmt --all -- --check

# Run all unit tests
test:
    cargo test --lib --features "eval-advanced discourse"

# Run all tests including integration
test-all:
    cargo test --features "eval-advanced discourse"

# === CI Simulation ===

# Simulate full CI pipeline locally (fast checks only)
ci: fmt
    cargo check --all-targets
    cargo clippy --all-targets
    cargo test --lib
    cargo test --test no_features
    ANNO_MAX_EXAMPLES=10 cargo test --lib --features "eval-advanced discourse"
    cargo test --test eval_integration --features "eval-advanced"
    cargo test --test coref_integration --features "eval-advanced"
    cargo test --test discourse_comprehensive --features "discourse"
    cargo test --test new_features_integration --features "eval-advanced"
    cargo test --test regression_f1 --features eval
    @echo "CI simulation passed"

# Simulate CI with sanity evals (includes small random sample evals)
ci-eval: ci
    just eval-sanity

# === Evaluation ===

# Run evaluation on synthetic data (fast, no downloads)
eval-quick:
    ANNO_MAX_EXAMPLES=20 cargo run --example eval_basic --features eval

# Run sanity check evaluations (small random samples, ~5-10 min)
# Used in CI on push
eval-sanity:
    ./scripts/eval-sanity.sh

# Run full evaluations (all task-dataset-backend combinations)
# Heavy operation - only run on eval-* branches or manual trigger
eval-full:
    ./scripts/eval-full.sh

# Run full evaluations with example limit
eval-full-limit MAX_EXAMPLES:
    MAX_EXAMPLES={{MAX_EXAMPLES}} ./scripts/eval-full.sh

# Run evaluation with specific seed
eval-seed SEED MAX_EXAMPLES="20":
    cargo run --release --bin anno --features "cli,eval-advanced" -- benchmark \
        --max-examples {{MAX_EXAMPLES}} \
        --seed {{SEED}} \
        --cached-only \
        --output eval-seed-{{SEED}}.md

# Run abstract anaphora evaluation
eval-anaphora:
    cargo run --example abstract_anaphora_eval --features discourse

# === Backend Tests ===

# Test ONNX backend (build only, no models)
test-onnx:
    cargo build --features onnx
    cargo test --lib --features onnx

# Test Candle backend (build only, no models)  
test-candle:
    cargo build --features candle
    cargo test --lib --features candle

# Test with model downloads (slow, requires network)
test-models:
    cargo test --features onnx -- --ignored --nocapture

# === Documentation ===

# Build docs
docs:
    cargo doc --no-deps --features "eval-full discourse"

# Open docs in browser
docs-open:
    cargo doc --no-deps --features "eval-full discourse" --open

# Preview README in browser (requires grip or similar)
readme-preview:
    @which grip > /dev/null && grip README.md || \
    (python3 -m http.server 8000 > /dev/null 2>&1 & echo "Preview at http://localhost:8000/README.md" && sleep 2 && open http://localhost:8000/README.md || echo "Open README.md in your editor for preview")

# === Benchmarks ===

# Run NER benchmark (no execution, just compile)
bench-check:
    cargo bench --no-run --features eval

# Run benchmarks
bench:
    cargo bench --features eval

# === Utilities ===

# Download evaluation datasets
download-datasets:
    cargo test --test real_datasets --features eval-advanced -- --ignored download

# Clean build artifacts
clean:
    cargo clean

# Check MSRV (1.75)
msrv:
    cargo +1.75.0 check

# Run property tests with more cases
proptest:
    PROPTEST_CASES=1000 cargo test --lib --features "eval-advanced" -- proptest

# === Release ===

# Build release binary
build-release:
    cargo build --release --features "eval-full discourse onnx"

# Run clippy with stricter lints
clippy-strict:
    cargo clippy --all-targets -- -W clippy::pedantic -W clippy::nursery

# === Code Quality ===

# Count lines of code
loc:
    @tokei src/ tests/ examples/ benches/ --compact

# Check for TODO/FIXME comments
todos:
    @rg -i "(TODO|FIXME|HACK|XXX)" --type rust -c | sort -t: -k2 -rn | head -15

# Show test coverage summary
test-count:
    @echo "Tests:" && rg "^#\[test\]" --type rust -c | awk -F: '{sum += $2} END {print sum}'

# === Quick Examples ===

# Run quickstart example (no deps)
example-quickstart:
    cargo run --example quickstart

# Run eval example (needs eval feature)
example-eval:
    cargo run --example eval_basic --features eval

# Run GLiNER2 example (needs onnx feature + model download)
example-gliner2:
    cargo run --example gliner2_multitask --features onnx

# === Mutation Testing ===

# Run mutation tests on entity.rs (fast, targeted)
mutants-fast:
    cargo mutants --file "src/entity.rs" --timeout 120 --minimum-test-timeout 60 --features "eval-advanced"

# Run mutation tests on specific file
mutants-file FILE:
    cargo mutants --file "{{FILE}}" --timeout 30 --minimum-test-timeout 20

# Run mutation tests on all source files (slow, comprehensive)
mutants-all:
    cargo mutants --timeout 60 --minimum-test-timeout 30

# List mutants without running tests (quick check)
mutants-list:
    cargo mutants --list

# === Static Analysis Tools ===

# Run cargo-deny (dependency linting)
# Rule validation and reporting
validate-rules:
    @echo "Validating OpenGrep rules against known patterns..."
    @./scripts/validate-rules.sh

unified-report:
    @echo "Generating unified static analysis report..."
    @./scripts/generate-unified-report.sh
    @echo "Report generated: unified-static-analysis-report.md"

failure-summary:
    @echo "Summarizing static analysis failures..."
    @./scripts/summarize-failures.sh
    @echo "Summary generated: static-analysis-failures-summary.md"

# Static analysis tools
deny:
    @which cargo-deny > /dev/null || (echo "Install: cargo install --locked cargo-deny" && exit 1)
    cargo deny check

# Run cargo-machete (fast unused dependencies)
machete:
    @which cargo-machete > /dev/null || (echo "Install: cargo install cargo-machete" && exit 1)
    cargo machete

# Run cargo-geiger (unsafe code statistics)
geiger:
    @which cargo-geiger > /dev/null || (echo "Install: cargo install cargo-geiger" && exit 1)
    cargo geiger

# Generate unsafe code safety report (creative use of cargo-geiger)
safety-report:
    @which cargo-geiger > /dev/null || (echo "Install: cargo install cargo-geiger" && exit 1)
    @echo "Generating safety report..."
    @cargo geiger --output-format json > .safety-report.json 2>/dev/null || true
    @echo "Unsafe code statistics:"
    @cat .safety-report.json | jq -r '.packages[] | select(.geiger.unsafe_used > 0) | "\(.name): \(.geiger.unsafe_used) unsafe uses"' 2>/dev/null || echo "No unsafe code found or jq not installed"
    @echo ""
    @echo "Full report saved to .safety-report.json"

# Run OpenGrep static analysis
opengrep:
    @which opengrep > /dev/null || (echo "Install: curl -fsSL https://raw.githubusercontent.com/opengrep/opengrep/main/install.sh | bash" && exit 1)
    opengrep scan --config auto --json --output .opengrep-results.json src/ tests/ examples/
    @echo "Results saved to .opengrep-results.json"
    @cat .opengrep-results.json | jq -r '.results | length' | xargs -I {} echo "Found {} issues" 2>/dev/null || echo "Run: opengrep scan --config auto"

# Run OpenGrep with custom rules
opengrep-custom:
    @which opengrep > /dev/null || (echo "Install: curl -fsSL https://raw.githubusercontent.com/opengrep/opengrep/main/install.sh | bash" && exit 1)
    opengrep scan -f .opengrep/rules --json --output .opengrep-custom-results.json src/
    @echo "Custom rules results saved to .opengrep-custom-results.json"

# Run Miri on unsafe code files (selective)
miri-unsafe:
    @rustup component list | grep -q "miri.*installed" || (echo "Install: rustup component add miri" && exit 1)
    @echo "Running Miri on unsafe code files..."
    @cargo miri test --lib --features onnx -- --test-threads=1 2>&1 | head -50 || true
    @echo "Miri check complete (see output above)"

# Run all static analysis tools (comprehensive check)
static-analysis:
    @echo "=== Running Static Analysis Tools ==="
    @echo ""
    @echo "1. cargo-deny (dependency linting)..."
    @just deny || echo "⚠️  cargo-deny failed or not installed"
    @echo ""
    @echo "2. cargo-machete (unused dependencies)..."
    @just machete || echo "⚠️  cargo-machete failed or not installed"
    @echo ""
    @echo "3. cargo-geiger (unsafe code stats)..."
    @just geiger || echo "⚠️  cargo-geiger failed or not installed"
    @echo ""
    @echo "4. OpenGrep (security patterns)..."
    @just opengrep || echo "⚠️  OpenGrep failed or not installed"
    @echo ""
    @echo "=== Static Analysis Complete ==="

# Run tests with cargo-nextest (better output)
test-nextest:
    @which cargo-nextest > /dev/null || (echo "Install: cargo install cargo-nextest" && exit 1)
    cargo nextest run --all-features

# Generate code coverage report
coverage:
    @which cargo-llvm-cov > /dev/null || (echo "Install: cargo install cargo-llvm-cov" && exit 1)
    cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info
    @echo "Coverage report generated: lcov.info"
    @echo "View with: genhtml lcov.info -o coverage-html && open coverage-html/index.html"

# Generate comprehensive safety report (creative: combines multiple tools)
safety-report-full:
    @./scripts/generate-safety-report.sh
    @echo "Full safety report: safety-report.md"

# Benchmark static analysis tools (creative: performance comparison)
benchmark-tools:
    @./scripts/benchmark-static-analysis.sh

# Compare tool outputs (creative: identify overlapping findings)
compare-tools:
    @./scripts/compare-tool-outputs.sh

# Track unsafe code trends over time (creative: time-series analysis)
track-unsafe-trends:
    @./scripts/track-unsafe-code-trends.sh

# Validate static analysis setup
validate-setup:
    @./scripts/validate-static-analysis-setup.sh

# === All-in-One Commands ===

# Run everything: static analysis + safety report + trends
analysis-full:
    @echo "Running comprehensive static analysis..."
    @just static-analysis
    @echo ""
    @echo "Generating safety report..."
    @just safety-report-full
    @echo ""
    @echo "Tracking unsafe code trends..."
    @just track-unsafe-trends
    @echo ""
    @echo "✅ Comprehensive analysis complete!"
    @echo "   - Reports: safety-report.md, tool-comparison.md"
    @echo "   - Trends: .unsafe-code-trends/"

# Quick validation before commit
pre-commit-check:
    @echo "Running pre-commit checks..."
    @cargo fmt --all -- --check
    @cargo clippy --all-targets -- -D warnings
    @just machete || echo "⚠️  cargo-machete not installed, skipping"
    @echo "✅ Pre-commit checks passed"

# Generate HTML dashboard (creative: visual analysis results)
dashboard:
    @./scripts/generate-analysis-dashboard.sh
    @echo "Dashboard: static-analysis-dashboard.html"

# === NLP/ML-Specific Analysis ===

# Check NLP/ML-specific patterns
check-nlp-patterns:
    @./scripts/check-nlp-patterns.sh

# Analyze evaluation framework patterns
analyze-eval-patterns:
    @./scripts/analyze-evaluation-patterns.sh
    @echo "Analysis: evaluation-pattern-analysis.md"

# Check ML backend patterns
check-ml-backends:
    @./scripts/check-ml-backend-patterns.sh

# Check evaluation framework invariants
check-eval-invariants:
    @./scripts/check-evaluation-invariants.sh

# Comprehensive NLP/ML analysis (combines all checks)
analysis-nlp-ml:
    @echo "=== NLP/ML Pattern Analysis ==="
    @just check-nlp-patterns || echo "⚠️  Some NLP pattern issues found"
    @echo ""
    @echo "=== Evaluation Framework Analysis ==="
    @just analyze-eval-patterns
    @echo ""
    @echo "=== ML Backend Analysis ==="
    @just check-ml-backends || echo "⚠️  Some ML backend issues found"
    @echo ""
    @echo "=== Evaluation Invariants ==="
    @just check-eval-invariants || echo "⚠️  Some invariant issues found"
    @echo ""
    @echo "=== OpenGrep Custom Rules ==="
    @just opengrep-custom || echo "⚠️  OpenGrep not installed"
    @echo ""
    @echo "✅ NLP/ML analysis complete"

# Generate repo-specific analysis report
repo-analysis:
    @./scripts/generate-repo-specific-report.sh
    @echo "Report: repo-specific-analysis.md"

# Integrate static analysis with evaluation framework
integrate-analysis-eval:
    @./scripts/integrate-with-evaluation.sh
    @echo "Integration guide: static-analysis-eval-integration.md"

# Check for historical bug patterns (regression prevention)
check-historical-bugs:
    @./scripts/check-historical-bugs.sh

# === Publish Validation ===

# Validate publish readiness for all crates
validate-publish:
    @./scripts/validate-publish.sh
