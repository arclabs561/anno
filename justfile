# anno development tasks
# Run `just` to see available commands

default:
    @just --list

# === Quick Commands ===

# Run fast checks (fmt + clippy + quick tests) - matches pre-push hook
check:
    #!/usr/bin/env bash
    set -e
    just docs-audit
    cargo fmt --manifest-path Cargo.toml -p anno -- --check
    cargo clippy --manifest-path Cargo.toml --workspace --all-targets --features "eval-advanced discourse" -- -D warnings
    if command -v cargo-nextest >/dev/null 2>&1; then
        cargo nextest run --manifest-path Cargo.toml --profile quick --workspace --features "eval-advanced discourse"
    else
        cargo test --manifest-path Cargo.toml --workspace --lib --features "eval-advanced discourse"
    fi

# Run fast checks without features (minimal, for quick iteration)
check-minimal:
    cargo fmt --manifest-path Cargo.toml -p anno -- --check
    cargo clippy --manifest-path Cargo.toml -p anno --all-targets --no-default-features
    cargo test --manifest-path Cargo.toml -p anno --no-default-features --lib

# Format all code
fmt:
    cargo fmt --manifest-path Cargo.toml -p anno

# Check formatting without modifying
fmt-check:
    cargo fmt --manifest-path Cargo.toml -p anno -- --check

# Run all unit tests (prefers nextest)
test:
    #!/usr/bin/env bash
    if command -v cargo-nextest >/dev/null 2>&1; then
        cargo nextest run --profile quick --lib --features "eval-advanced discourse"
    else
    cargo test --lib --features "eval-advanced discourse"
    fi

# Run all tests including integration (prefers nextest)
test-all:
    #!/usr/bin/env bash
    if command -v cargo-nextest >/dev/null 2>&1; then
        cargo nextest run --profile quick --workspace --features "eval-advanced discourse"
    else
    cargo test --features "eval-advanced discourse"
    fi

# Quick single-test run with filter (e.g., just t test_name)
# Uses nextest with minimal features to avoid full workspace scan
t FILTER:
    #!/usr/bin/env bash
    if command -v cargo-nextest >/dev/null 2>&1; then
        cargo nextest run --profile quick -p anno -E 'test(/{{FILTER}}/)' --no-default-features --features "cli eval"
    else
        cargo test -p anno --no-default-features --features "cli eval" -- '{{FILTER}}'
    fi

# Quick single-test run with full features
tf FILTER:
    #!/usr/bin/env bash
    if command -v cargo-nextest >/dev/null 2>&1; then
        cargo nextest run --profile quick -p anno -E 'test(/{{FILTER}}/)' --features "eval-advanced discourse"
    else
        cargo test -p anno --features "eval-advanced discourse" -- '{{FILTER}}'
    fi

# === Test Profiling (Nextest + Rust Tooling) ===

# Profile tests with nextest timing (recommended)
profile-tests PROFILE="quick" FILTER="":
    @./scripts/profile-tests.sh {{PROFILE}} {{FILTER}}

# Profile with Rust native tools (debug symbols, perf/sample)
profile-tests-rust PROFILE="quick":
    @./scripts/profile-tests-rust.sh {{PROFILE}}

# Profile quick tests only
profile-quick:
    @just profile-tests quick

# Profile CI tests
profile-ci:
    @just profile-tests ci

# Profile ML tests (slow, model loading)
profile-ml:
    @just profile-tests ml

# Profile specific test filter
profile-filter FILTER:
    @just profile-tests quick "{{FILTER}}"

# Quick timing report (no full run, just analyze existing)
profile-timing:
    @NEXTEST_EXPERIMENTAL_LIBTEST_JSON=1 cargo nextest run --profile quick --workspace --features "eval-advanced discourse" --message-format libtest-json-plus --status-level all

# Show slowest tests from last profile run
profile-slowest:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -z "$(ls -A target/test-profiles/timing_*.json 2>/dev/null || true)" ]; then
        echo "No timing files found. Run 'just profile-tests' first."
        exit 1
    fi
    LATEST="$(ls -t target/test-profiles/timing_*.json | sed -n '1p')"
    echo "Analyzing: $LATEST"
    if command -v jq >/dev/null 2>&1; then
        echo ""
        echo "=== Slowest Tests ==="
        jq -r '.test_executions[] | select(.duration_secs > 0.1) | "\(.duration_secs | tostring | .[0:6])s  \(.test_name)"' \
            "$LATEST" | sort -rn
    else
        echo "Install jq for analysis: brew install jq"
    fi

# Analyze test profile with detailed breakdown
profile-analyze FILE="":
    @if [ -n "{{FILE}}" ]; then \
        uv run -- python scripts/analyze_test_profile.py "{{FILE}}"; \
    else \
        uv run -- python scripts/analyze_test_profile.py; \
    fi

# Install profiling tools
profile-install:
    @echo "Installing profiling tools..."
    @echo ""
    @echo "1. cargo-flamegraph (cross-platform flamegraphs):"
    @echo "   cargo install flamegraph"
    @echo ""
    @echo "2. cargo-instruments (macOS only, requires Xcode):"
    @echo "   cargo install cargo-instruments"
    @echo ""
    @echo "3. hyperfine (benchmarking):"
    @echo "   brew install hyperfine"
    @echo ""
    @echo "4. jq (JSON analysis):"
    @echo "   brew install jq"

# === CI Simulation ===

# Simulate full CI pipeline locally (fast checks only)
ci: fmt
    just docs-audit
    cargo check --all-targets
    cargo clippy --all-targets
    cargo test --lib
    cargo test --test no_features
    ANNO_MAX_EXAMPLES=10 cargo test --lib --features "eval-advanced discourse"
    cargo test --test eval_integration --features "eval-advanced"
    cargo test --test coreference_tests --features "eval-advanced"
    cargo test --test discourse_proptest --features "discourse"
    cargo test --test features_comprehensive --features "eval-advanced"
    cargo test --test regression_tests --features eval
    @echo "CI simulation passed"

# Simulate CI with sanity evals (includes small random sample evals)
ci-eval: ci
    just eval-sanity

# === Evaluation ===

# Run randomized matrix test (backends x datasets x tasks)
# Strategies: random, ml-only, worst-first, ml-all
# Example: just matrix worst-first 42
matrix strategy="random" seed="" perspective="ner":
    #!/usr/bin/env bash
    echo "Running randomized matrix test (strategy: {{strategy}})..."
    export ANNO_SAMPLE_STRATEGY={{strategy}}
    export ANNO_MATRIX_PERSPECTIVE={{perspective}}
    if [ -n "{{seed}}" ]; then export ANNO_CI_SEED={{seed}}; fi
    cargo test -p anno --lib --features "eval-advanced" test_randomized_matrix_sample -- --nocapture

# Run matrix test with ML backends (requires onnx/candle features)
matrix-ml:
    @echo "Running ML-focused matrix test..."
    @ANNO_SAMPLE_STRATEGY=ml-only ANNO_ML_IN_MATRIX=1 cargo test -p anno --lib --features "eval-advanced onnx" test_randomized_matrix_sample -- --nocapture

# Show backend availability matrix
matrix-backends:
    @echo "Deprecated: legacy backend availability matrix was removed with archive cleanup."

# Run evaluation on synthetic data (fast, no downloads)
eval-quick:
    # Fast, bounded benchmark run that writes an artifact under ./reports/.
    # Note: this may still download datasets/models unless caches are already warm.
    mkdir -p reports
    cargo run --release -p anno-cli --bin anno --features "eval-advanced onnx" -- benchmark \
        --tasks ner \
        --datasets CoNLL2003Sample,WikiGold,Wnut17,WikiANN,MasakhaNER \
        --backends heuristic,stacked,bert_onnx,gliner_onnx \
        --max-examples 20 \
        --output reports/eval-quick-report.md

# Wider local eval (still bounded, but broader than eval-quick)
eval-wide MAX_EXAMPLES="50":
    mkdir -p reports
    cargo run --release -p anno-cli --bin anno --features "eval-advanced onnx" -- benchmark \
        --tasks ner \
        --datasets CoNLL2003Sample,WikiGold,Wnut17,WikiANN,MasakhaNER,MultiCoNERv2,MultiNERD \
        --backends heuristic,stacked,bert_onnx,gliner_onnx,nuner \
        --max-examples {{MAX_EXAMPLES}} \
        --output reports/eval-wide-report.md

# Run sanity check evaluations (small random samples)
# Used in CI on push
eval-sanity:
    ./scripts/eval-sanity.sh

# Regenerate dataset registry exports and derived tooling files.
regenerate-datasets:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo test -p anno-eval generate_datasets_json -- --ignored
    cargo test -p anno-eval generate_datasets_jsonl -- --ignored
    cargo test -p anno-eval generate_datasets_markdown -- --ignored
    python3 scripts/generate_download_configs.py \
        --input generated/datasets_generated.json \
        --output generated/download_configs_generated.json \
        --stats
    python3 scripts/generate_catalog_html.py

# Curated benchmark profiles (official; avoids parsing markdown and avoids “sweep everything”).
# Examples:
# - just eval-profile ner-standard 20
# - just eval-profile ner-zeroshot-multilingual 20
eval-profile PROFILE MAX_EXAMPLES="20":
    mkdir -p reports
    cargo run --release -p anno-cli --bin anno --features "eval-advanced onnx" -- benchmark \
        --profile {{PROFILE}} \
        --max-examples {{MAX_EXAMPLES}} \
        --output reports/eval-{{PROFILE}}.md \
        --output-json reports/eval-{{PROFILE}}.json

# Run full evaluations (all task-dataset-backend combinations)
# Heavy operation - only run on eval-* branches or manual trigger
eval-full:
    ./scripts/eval-full.sh

# Run full evaluations with example limit
eval-full-limit MAX_EXAMPLES:
    MAX_EXAMPLES={{MAX_EXAMPLES}} ./scripts/eval-full.sh

# Run evaluation with specific seed
eval-seed SEED MAX_EXAMPLES="20":
    cargo run --release -p anno-cli --bin anno --features "eval-advanced onnx" -- benchmark \
        --max-examples {{MAX_EXAMPLES}} \
        --seed {{SEED}} \
        --cached-only \
        --output eval-seed-{{SEED}}.md

# Run abstract anaphora evaluation
eval-anaphora:
    cargo run --example abstract_anaphora_eval --features discourse

# Run comprehensive local evaluation (resumable)
# Example: just eval-comprehensive 50
eval-comprehensive MAX_EXAMPLES="50":
    uv run scripts/eval_comprehensive.py --max-examples {{MAX_EXAMPLES}}

# Resume comprehensive evaluation from where it left off
eval-resume MAX_EXAMPLES="50":
    uv run scripts/eval_comprehensive.py --resume --max-examples {{MAX_EXAMPLES}}

# View current evaluation results
eval-results:
    @cat reports/RESULTS.md 2>/dev/null || echo "No results yet. Run 'just eval-comprehensive' first."

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
    cargo doc -p anno -p anno-core -p anno-eval --no-deps --features "eval-advanced discourse"

# Open docs in browser
docs-open:
    cargo doc -p anno -p anno-core -p anno-eval --no-deps --features "eval-advanced discourse" --open

# Check internal docs markdown links (fast, no network).
docs-links:
    @if command -v uv > /dev/null; then \
        uv run -- python scripts/check_docs_links.py; \
    else \
        python3 scripts/check_docs_links.py; \
    fi

# Docs hygiene (links + stale path checks). Fast, offline.
docs-audit:
    @if command -v uv > /dev/null; then \
        uv run -- python scripts/docs_audit.py; \
    else \
        python3 scripts/docs_audit.py; \
    fi

# Preview README in browser with GitHub-style rendering (auto-reloads)
# Auto-finds free port starting from 8000
readme-preview:
    @uv run scripts/serve_readme.py > /tmp/serve_readme.log 2>&1 & \
    sleep 3 && \
    PORT=$$(cat /tmp/serve_readme_port.txt 2>/dev/null || echo "8000") && \
    open http://localhost:$$PORT/README_github_style.html && \
    echo "ok: Preview at http://localhost:$$PORT/README_github_style.html (auto-reloads)"

# Run e2e test with Playwright + Gemini VLM
readme-test:
    @uv run scripts/e2e_readme_test.py

# Type-check Python scripts with ty (optional).
# Notes:
# - Uses `uvx` so you don't have to install ty into your repo venv.
# - Runs on `scripts/` only (this repo is not a Python package).
typecheck-python:
    @which uvx > /dev/null || (echo "Install uv (provides uvx): https://docs.astral.sh/uv/" && exit 1)
    uvx ty check scripts

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

# Warm local dataset cache (and optionally S3 mirror).
# Example:
#   ANNO_WARM_PER_TASK=2 ANNO_WARM_SEED=42 cargo run --example cache_warm --features "eval-advanced"
cache-warm:
    cargo run -p anno --example cache_warm --features "eval-advanced"

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
    @rg -i "(TODO|FIXME|HACK|XXX)" --type rust -c | sort -t: -k2 -rn

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
    ./scripts/static-analysis-common.sh machete

# Run cargo-geiger (unsafe code statistics)
geiger:
    @which cargo-geiger > /dev/null || (echo "Install: cargo install cargo-geiger" && exit 1)
    cargo geiger

# Run ast-grep checks (optional, local)
ast-grep-unicode:
    @which ast-grep > /dev/null || (echo "Install: https://ast-grep.github.io/ (or: brew install ast-grep)" && exit 1)
    ast-grep scan --rule .opengrep/rules/rust-unicode-offsets.yaml --report-style short crates/anno/

ast-grep-unicode-all:
    @which ast-grep > /dev/null || (echo "Install: https://ast-grep.github.io/ (or: brew install ast-grep)" && exit 1)
    ast-grep scan --rule .opengrep/rules/rust-unicode-offsets.yaml --report-style short crates/anno/ tests/ examples/

ast-grep-metal:
    @which ast-grep > /dev/null || (echo "Install: https://ast-grep.github.io/ (or: brew install ast-grep)" && exit 1)
    ast-grep scan --rule .opengrep/rules/rust-candle-metal.yaml --report-style short crates/anno/

ast-grep-metal-all:
    @which ast-grep > /dev/null || (echo "Install: https://ast-grep.github.io/ (or: brew install ast-grep)" && exit 1)
    ast-grep scan --rule .opengrep/rules/rust-candle-metal.yaml --report-style short crates/anno/ tests/ examples/

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
    opengrep scan --config auto --json --output opengrep-results.json crates/anno/ tests/ examples/
    @echo "Results saved to opengrep-results.json"
    @if command -v jq > /dev/null; then \
        echo "Found $$(jq -r '.results | length' opengrep-results.json) issues"; \
    else \
        echo "Install jq to summarize: opengrep-results.json"; \
    fi

# Run OpenGrep with custom rules
opengrep-custom:
    @which opengrep > /dev/null || (echo "Install: curl -fsSL https://raw.githubusercontent.com/opengrep/opengrep/main/install.sh | bash" && exit 1)
    opengrep scan -f .opengrep/rules/rust-security.yaml --json --output opengrep-security-results.json crates/anno/
    opengrep scan -f .opengrep/rules/rust-nlp-ml-patterns.yaml --json --output opengrep-nlp-results.json crates/anno/
    opengrep scan -f .opengrep/rules/rust-evaluation-framework.yaml --json --output opengrep-eval-results.json crates/anno-eval/
    opengrep scan -f .opengrep/rules/rust-anno-specific.yaml --json --output opengrep-anno-results.json crates/anno/
    opengrep scan -f .opengrep/rules/rust-error-handling.yaml --json --output opengrep-error-results.json crates/anno/
    opengrep scan -f .opengrep/rules/rust-memory-patterns.yaml --json --output opengrep-memory-results.json crates/anno/
    @echo "Custom rules results saved to opengrep-*-results.json"
    @if command -v jq > /dev/null; then \
        echo "Counts:"; \
        echo "  security: $$(jq -r '.results | length' opengrep-security-results.json)"; \
        echo "  nlp:      $$(jq -r '.results | length' opengrep-nlp-results.json)"; \
        echo "  eval:     $$(jq -r '.results | length' opengrep-eval-results.json)"; \
        echo "  anno:     $$(jq -r '.results | length' opengrep-anno-results.json)"; \
        echo "  error:    $$(jq -r '.results | length' opengrep-error-results.json)"; \
        echo "  memory:   $$(jq -r '.results | length' opengrep-memory-results.json)"; \
    fi

# Run Miri on unsafe code files (selective)
miri-unsafe:
    @rustup component list | rg -q "miri.*installed" || (echo "Install: rustup component add miri" && exit 1)
    @echo "Running Miri on unsafe code files..."
    @cargo miri test --lib --features onnx -- --test-threads=1 --nocapture || true
    @echo "Miri check complete (see output above)"

# Run all static analysis tools (comprehensive check)
static-analysis:
    @echo "=== Running Static Analysis Tools ==="
    @echo ""
    @echo "1. cargo-deny (dependency linting)..."
    @just deny || echo "warning:  cargo-deny failed or not installed"
    @echo ""
    @echo "2. cargo-machete (unused dependencies)..."
    @just machete || echo "warning:  cargo-machete failed or not installed"
    @echo ""
    @echo "3. cargo-geiger (unsafe code stats)..."
    @just geiger || echo "warning:  cargo-geiger failed or not installed"
    @echo ""
    @echo "4. OpenGrep (security patterns)..."
    @just opengrep || echo "warning:  OpenGrep failed or not installed"
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
    @echo "ok: Comprehensive analysis complete!"
    @echo "   - Reports: safety-report.md, tool-comparison.md"
    @echo "   - Trends: .unsafe-code-trends/"

# === Git Hook Checks ===

# Check for invalid filenames (spaces, duplicates)
check-filenames:
    @echo "Checking for files with spaces in names..."
    @first=$$(find . -type f \( -name "*.rs" -o -name "*.toml" \) -name "* *" ! -path "./.git/*" ! -path "./target/*" ! -path "./.cargo/*" -print -quit 2>/dev/null) && \
    if [ -n "$$first" ]; then \
        echo "error: Error: Found files with spaces in names (invalid for Rust):"; \
        find . -type f \( -name "*.rs" -o -name "*.toml" \) -name "* *" ! -path "./.git/*" ! -path "./target/*" ! -path "./.cargo/*"; \
        exit 1; \
    fi
    @echo "Checking for duplicate test files..."
    @dup=$$(find . -type f -name "* 2.rs" ! -path "./.git/*" ! -path "./target/*" -print -quit 2>/dev/null) && \
    if [ -n "$$dup" ]; then \
        echo "error: Error: Found duplicate test files (likely backups):"; \
        find . -type f -name "* 2.rs" ! -path "./.git/*" ! -path "./target/*"; \
        echo "Please remove these files before committing."; \
        exit 1; \
    fi

# Check compilation (fast, catches syntax/type errors)
check-compile:
    @echo "Checking compilation..."
    @cargo check --workspace --all-targets --message-format=short --quiet

# Check test compilation
check-tests-compile:
    @echo "Checking test compilation..."
    @cargo check --workspace --tests --message-format=short --quiet

# Check for potential secrets (warn only, non-blocking)
check-secrets:
    @echo "Checking for potential secrets..."
    @if command -v rg &> /dev/null; then \
        files=$$(rg -i --files-with-matches "api[_-]?key\\s*=|password\\s*=|secret\\s*=|token\\s*=|credential\\s*=" \
           --glob '!*.md' --glob '!*.txt' --glob '!target/**' \
           --glob '!.git/**' --glob '!*.lock' --glob '!justfile' \
           --glob '!*.sh' --glob '!docs/**' 2>/dev/null || true); \
        if [ -n "$$files" ]; then \
            echo "warning:  Warning: Potential secrets found. Review before committing."; \
            echo "warning:  (Listing files only; not printing match contents.)"; \
            echo "$$files"; \
        fi; \
    else \
        echo "warning:  ripgrep not found, skipping secrets check"; \
    fi

# Check for large files (warn only, non-blocking)
check-large-files:
    @echo "Checking for large files..."
    @first=$$(find . -type f -size +1M ! -path "./target/*" ! -path "./.git/*" ! -path "./.cargo/*" ! -path "./*.lock" ! -path "./assets/*" ! -path "./.mypy_cache/*" ! -path "./.pytest_cache/*" ! -path "./__pycache__/*" -print -quit 2>/dev/null) && \
    if [ -n "$$first" ]; then \
        echo "warning:  Warning: Large files detected (>1MB):"; \
        find . -type f -size +1M ! -path "./target/*" ! -path "./.git/*" ! -path "./.cargo/*" ! -path "./*.lock" ! -path "./assets/*" ! -path "./.mypy_cache/*" ! -path "./.pytest_cache/*" ! -path "./__pycache__/*" 2>/dev/null; \
        echo "   (This is a warning, not blocking)"; \
    fi

# Run clippy with warnings only (non-blocking)
check-clippy-warn:
    @echo "Running clippy (warnings only)..."
    @cargo clippy --workspace --all-targets --quiet 2>&1 || echo "warning:  Clippy found warnings (not blocking)"

# Full pre-commit checks (all blocking checks)
pre-commit-full:
    @just check-filenames
    @just check-compile
    @just check-tests-compile
    @just fmt-check
    @echo "ok: Pre-commit checks passed!"

# Pre-commit with warnings (blocking + non-blocking)
pre-commit-all: pre-commit-full
    @just check-secrets
    @just check-large-files
    @just check-clippy-warn

# Validate commit message format
validate-commit-msg COMMIT_MSG:
    @len=$$(printf "%s" "{{COMMIT_MSG}}" | wc -c | tr -d ' ') && \
    if [ "$$len" -lt 10 ]; then \
        echo "error: Error: Commit message too short (minimum 10 characters)"; \
        echo "   Current message: '{{COMMIT_MSG}}'"; \
        exit 1; \
    fi
    @if printf "%s" "{{COMMIT_MSG}}" | rg -q "^[a-z]+(\\(.+\\))?: .{10,}"; then \
        exit 0; \
    fi
    @if printf "%s" "{{COMMIT_MSG}}" | rg -q "^(Merge|Revert|Release|chore\\(release\\))"; then \
        exit 0; \
    fi
    @echo "warning:  Warning: Commit message doesn't follow conventional format"
    @echo "   Recommended: type(scope): description"
    @echo "   Examples:"
    @echo "     - feat(api): add new endpoint"
    @echo "     - fix: resolve compilation error"
    @echo "     - docs: update README"
    @echo "   Current message: '{{COMMIT_MSG}}'"
    @echo ""
    @echo "   (This is a warning, commit will proceed)"

# Quick validation before commit (legacy, kept for compatibility)
pre-commit-check:
    @echo "Running pre-commit checks..."
    @cargo fmt --all -- --check
    @cargo clippy --workspace --all-targets --features "eval-advanced discourse" -- -D warnings
    @just machete || echo "warning:  cargo-machete not installed, skipping"
    @echo "ok: Pre-commit checks passed"

# === Git Hook Setup ===

# Install git hooks (run once after clone)
setup-hooks:
    @echo "Installing git hooks from scripts/hooks/..."
    @cp scripts/hooks/pre-commit .git/hooks/pre-commit
    @cp scripts/hooks/pre-push .git/hooks/pre-push
    @cp scripts/hooks/commit-msg .git/hooks/commit-msg
    @chmod +x .git/hooks/pre-commit
    @chmod +x .git/hooks/pre-push
    @chmod +x .git/hooks/commit-msg
    @echo ""
    @echo "Hooks installed:"
    @echo "  pre-commit   fast checks (format, compile)"
    @echo "  pre-push     full checks (clippy, tests)"
    @echo "  commit-msg   message format hints"
    @echo ""
    @echo "To bypass: git commit --no-verify"

# Show hook status
hook-status:
    @echo "Git hook status:"
    @ls -la .git/hooks/pre-commit .git/hooks/pre-push .git/hooks/commit-msg 2>/dev/null || echo "No hooks installed"

# Run what pre-commit hook runs (for debugging)
run-pre-commit-hook:
    @echo "Simulating pre-commit hook..."
    @cargo fmt --all
    @cargo check --workspace --all-targets --quiet

# Run what pre-push hook runs (for debugging)
run-pre-push-hook:
    @echo "Simulating pre-push hook..."
    @cargo fmt --all -- --check
    @cargo clippy --workspace --all-targets --features "eval-advanced discourse" -- -D warnings
    @cargo test --workspace --lib --features "eval-advanced discourse" --quiet
    @cargo test --workspace --doc --features "eval-advanced discourse" --quiet || echo "warning:  Doc test warnings"

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
    @just check-nlp-patterns || echo "warning:  Some NLP pattern issues found"
    @echo ""
    @echo "=== Evaluation Framework Analysis ==="
    @just analyze-eval-patterns
    @echo ""
    @echo "=== ML Backend Analysis ==="
    @just check-ml-backends || echo "warning:  Some ML backend issues found"
    @echo ""
    @echo "=== Evaluation Invariants ==="
    @just check-eval-invariants || echo "warning:  Some invariant issues found"
    @echo ""
    @echo "=== OpenGrep Custom Rules ==="
    @just opengrep-custom || echo "warning:  OpenGrep not installed"
    @echo ""
    @echo "ok: NLP/ML analysis complete"

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

# === AWS Spot Instance Evaluation ===

# One-time spot infrastructure setup (IAM, SQS, EBS, launch template)
spot-setup:
    @chmod +x scripts/spot/setup.sh
    @./scripts/spot/setup.sh

# Pre-download datasets to S3 (avoids HuggingFace API rate limits on spot instances)
# Run this before spot-eval to ensure datasets are available in S3
spot-prepare-datasets:
    @uv run scripts/prepare_datasets_s3.py

# Run comprehensive evaluation on spot instances (full pipeline)
# Generates tasks, launches fleet, waits for completion, aggregates results
spot-eval:
    @uv run scripts/spot/orchestrate.py full

# Run spot eval with custom parameters
# Example: just spot-eval-custom "gliner,nuner" "WikiGold,Wnut17" 2
spot-eval-custom BACKENDS DATASETS FLEET_SIZE="4":
    @uv run scripts/spot/orchestrate.py full \
        --backends "{{BACKENDS}}" \
        --datasets "{{DATASETS}}" \
        --fleet-size "{{FLEET_SIZE}}"

# Generate evaluation tasks and enqueue (without launching fleet)
spot-generate:
    @uv run scripts/spot/orchestrate.py generate

# Generate tasks for specific backends/datasets
spot-generate-custom BACKENDS="" DATASETS="" SEEDS="42,123,456":
    @uv run scripts/spot/orchestrate.py generate \
        --backends "{{BACKENDS}}" \
        --datasets "{{DATASETS}}" \
        --seeds "{{SEEDS}}"

# Preview tasks without enqueueing
spot-generate-dry:
    @uv run scripts/spot/orchestrate.py generate --dry-run

# Launch spot fleet (requires tasks in queue)
spot-launch FLEET_SIZE="4":
    @uv run scripts/spot/orchestrate.py launch --fleet-size "{{FLEET_SIZE}}"

# Check evaluation progress (queue depth, fleet status, results count)
spot-status:
    @uv run scripts/spot/orchestrate.py status

# Monitor workers via SSM (no SSH required)
spot-monitor:
    @uv run scripts/spot/monitor.py

# Monitor workers with live updates
spot-monitor-watch:
    @uv run scripts/spot/monitor.py --watch

# Tail logs from a specific worker
spot-logs INSTANCE:
    @uv run scripts/spot/monitor.py --logs "{{INSTANCE}}" --follow

# Execute command on a worker via SSM
spot-exec INSTANCE CMD:
    @uv run scripts/spot/monitor.py --exec "{{INSTANCE}}" "{{CMD}}"

# Aggregate and display results from S3
spot-results OUTPUT="reports/spot-eval-results.json":
    @uv run scripts/spot/orchestrate.py results --output "{{OUTPUT}}"

# Download results from S3 and aggregate
spot-aggregate:
    @uv run scripts/spot/aggregate.py --download

# Regenerate summary and open in browser
spot-summary:
    @uv run scripts/spot/aggregate.py --open

# Show LLM-generated summary of results
spot-summarize:
    @uv run scripts/spot/aggregate.py --llm

# Merge prediction cache shards from all workers
spot-merge-cache:
    @uv run scripts/spot/merge_cache.py

# Cancel fleet and clean up
spot-teardown:
    @uv run scripts/spot/orchestrate.py teardown

# Cancel fleet and purge task queue
spot-teardown-full:
    @uv run scripts/spot/orchestrate.py teardown --purge-queue

## Note: runctl integration removed.
##
## This repo previously supported runctl-managed spot evals. That path is intentionally
## deleted now; keep orchestration under `scripts/spot/orchestrate.py` (SQS/SSM-based).

# Quick spot eval (3 fast backends, 2 datasets, 1 seed) - good for testing
spot-eval-quick:
    @uv run scripts/spot/orchestrate.py full \
        --backends "pattern,heuristic,stacked" \
        --datasets "WikiGold,Wnut17" \
        --seeds "42" \
        --fleet-size 1

# Local evaluation (no AWS, runs on this machine)
eval-local BACKENDS="heuristic,stacked" DATASETS="WikiGold" MAX="50":
    @cargo build --release -p anno-cli --features "eval-advanced onnx" > /dev/null
    @uv run scripts/spot/orchestrate.py local \
        --backends "{{BACKENDS}}" \
        --datasets "{{DATASETS}}" \
        --max-examples "{{MAX}}"

# Local quick eval (zero-dep backends only, fast)
eval-local-quick:
    @cargo build --release -p anno-cli --features "eval-advanced onnx" > /dev/null
    @uv run scripts/spot/orchestrate.py local \
        --profile quick \
        --datasets "WikiGold,CoNLL2003Sample" \
        --max-examples 30

# ML-focused spot eval (ONNX/Candle backends only)
spot-eval-ml:
    @uv run scripts/spot/orchestrate.py full \
        --backends "gliner,nuner,w2ner,gliner2,bert_onnx,gliner_candle" \
        --fleet-size 4

# Sync local dataset/model cache to S3 (for spot instances)
spot-cache-upload:
    @./scripts/sync_datasets_s3.sh upload

# Download cached datasets/models from S3
spot-cache-download:
    @./scripts/sync_datasets_s3.sh download

# Show S3 cache status
spot-cache-status:
    @./scripts/sync_datasets_s3.sh status

# Upload current source code to S3 (required before launching spot instances)
spot-upload-src:
    @git archive --format=tar.gz HEAD -o /tmp/anno-src.tar.gz
    @aws s3 cp /tmp/anno-src.tar.gz s3://arc-anno-data/src/anno-src.tar.gz
    @echo "Source uploaded to s3://arc-anno-data/src/anno-src.tar.gz"

# Run CI-style muxer-backed sampler locally.
# This runs a small cached-only slice and persists muxer state to `ANNO_HISTORY_FILE` (JSON)
# or to the default `muxer_history.json` location when unset.
ci-matrix-local SEED="42" PERSPECTIVE="ner":
    ANNO_CI_SEED="{{SEED}}" \
    ANNO_MATRIX_PERSPECTIVE="{{PERSPECTIVE}}" \
    ANNO_SAMPLE_STRATEGY=worst-first \
    cargo test -p anno --lib --features "eval-advanced" matrix_muxer_ci::test_randomized_matrix_sample -- --nocapture

# Legacy: spot “badness history” export is not wired into the muxer JSON format by default.
spot-export-badness:
    @echo "Not implemented: spot -> muxer history export. Run ci-matrix-local to generate muxer_history.json from local runs."

# Compare spot results against a baseline
spot-compare BASELINE:
    @uv run scripts/spot/orchestrate.py results --compare "{{BASELINE}}"

# =============================================================================
# Spot + trainctl Integration (Enhanced Features)
# =============================================================================

# Launch trainctl interactive dashboard for spot monitoring
spot-dash:
    @scripts/spot/trainctl-bridge.sh dashboard

# Sync datasets from S3 using trainctl (faster parallel downloads)
spot-sync-datasets:
    @scripts/spot/trainctl-bridge.sh sync-datasets

# Download spot results using trainctl
spot-sync-results DIR="reports/spot-results":
    @scripts/spot/trainctl-bridge.sh sync-results "{{DIR}}"

# Upload source using trainctl (faster upload)
spot-upload-src-fast:
    @scripts/spot/trainctl-bridge.sh upload-src

# Show processes on spot worker (trainctl)
spot-ps INSTANCE="":
    @scripts/spot/trainctl-bridge.sh processes "{{INSTANCE}}"

# Interactive top for spot worker (trainctl)
spot-top INSTANCE="":
    @scripts/spot/trainctl-bridge.sh top "{{INSTANCE}}"

# Show fleet costs (trainctl)
spot-cost:
    @scripts/spot/trainctl-bridge.sh cost

# === trainctl Integration (Alternative) ===

# Launch workers via trainctl (better SSM, monitoring, dashboard)
# Requires: cd ../trainctl && cargo build --release
spot-trainctl WORKERS="1" PROFILE="full":
    @./scripts/spot/launch-trainctl.sh "{{WORKERS}}" "{{PROFILE}}"

# Quick trainctl launch (1 worker, quick profile)
spot-trainctl-quick:
    @./scripts/spot/launch-trainctl.sh 1 quick

# ML trainctl launch (4 workers, ML backends)
spot-trainctl-ml:
    @./scripts/spot/launch-trainctl.sh 4 ml

# Check if trainctl is available
spot-trainctl-check:
    @if command -v trainctl &>/dev/null; then \
        echo "trainctl: $(which trainctl)"; \
        trainctl --version; \
    elif [ -f ../trainctl/target/release/trainctl ]; then \
        echo "trainctl: ../trainctl/target/release/trainctl (local build)"; \
        ../trainctl/target/release/trainctl --version; \
    else \
        echo "trainctl not found. Build it:"; \
        echo "  cd ../trainctl && cargo build --release"; \
    fi
