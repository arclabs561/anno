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

# Run evaluations with multiple random seeds (comprehensive testing)
eval-multi-seed MAX_EXAMPLES="20":
    MAX_EXAMPLES={{MAX_EXAMPLES}} ./scripts/eval-multi-seed.sh

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
