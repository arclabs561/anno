# Contributing to anno

Thanks for your interest. anno is a focused information-extraction library: NER, coreference, relation extraction, PII, and zero-shot entity types over a unified `Entity` model. Backends compose; the API stays small.

## Before you start

For non-trivial work (new backends, API changes, new features, large refactors), open an issue first to align on scope. Drive-by bug fixes and doc patches don't need an issue.

## Setup

- Rust toolchain: stable, MSRV `1.88`. Use `rustup` to manage.
- Optional: `cargo-nextest` for faster test runs (`cargo install cargo-nextest`).
- Optional: `just` for the canonical recipes (`brew install just` or `cargo install just`).

Clone, then:

```
just check          # fmt + clippy + quick tests with default features
just check-minimal  # without ML feature flags
```

## Workspace layout

- `crates/anno`: the main library. Type foundation, backends, metrics, graph export, orchestration. Published as `anno`.
- `crates/anno-eval`: evaluation harness, dataset registry, task wiring.
- `crates/anno-cli`: `anno` binary. CLI parsing and command implementations.

The earlier split into `anno-core`, `anno-metrics`, and `anno-graph` was folded back into `anno` in Phase B of the 2026-04-26 consolidation. Their public surfaces now live at `anno::core::*`, `anno::metrics::*`, and `anno::graph::*` respectively. The legacy `anno-core` / `anno-metrics` / `anno-graph` crates remain frozen on crates.io at 0.8.0.

## Where backends live

`crates/anno/src/backends/<name>/`: one module per backend. Each implements the `Model` trait (and `ZeroShotNER` / `RelationExtractor` where applicable). Existing backends are the reference: `gliner_onnx`, `gliner_candle`, `gliner_multitask`, `nuner`, `bert_onnx`, `tplinker`, `crf`, `heuristic`.

To add a backend:

1. Create the module under `crates/anno/src/backends/<name>/`.
2. Implement `Model` (and any extension traits).
3. Wire it into `crates/anno/src/backends/mod.rs`.
4. Register in `crates/anno-cli/src/cli/parser.rs` (`ModelBackend` enum) if user-selectable.
5. Register in `crates/anno-eval/src/eval/backend_factory.rs` and `backend_name.rs` for the eval harness.
6. Add a row to `docs/BACKENDS.md`.
7. Add tests under the module.

### `Model` is a sealed trait

`anno::Model` is sealed (`Model: sealed::Sealed + Send + Sync`) where the
`Sealed` trait is `pub(crate)`. External crates **cannot** add their own
`impl Model for ...` directly. This is intentional: it keeps the set of
in-tree backends auditable (each backend lives under
`crates/anno/src/backends/`), keeps the trait's invariants under one
roof, and lets us evolve `Model`'s method set without breaking
downstream impls.

To plug an external backend in, wrap it in `anno::AnyModel` (the public,
non-sealed adapter) -- this preserves the audit story while still
letting anno-cli or downstream consumers hand a custom implementation
to the extraction pipeline.

For tests inside the `anno` crate, two patterns:

- Stubs that need `Model`: register them via
  `#[cfg(test)] impl Sealed for super::MockModel {}` inside `mod sealed`
  (see `lib.rs:151-190`).
- Stubs that only need an extension trait (`ZeroShotNER`,
  `RelationExtractor`, `DiscontinuousNER`): those traits are not
  sealed, so the stub can `impl ZeroShotNER for MyStub` directly. See
  `crates/anno/src/backends/inference/traits.rs::tests::CapturingNer`
  for the canonical pattern.

## Feature flags

`crates/anno/Cargo.toml` defines the feature set. Common combos:

- `--no-default-features`: pattern + heuristic only, no ML.
- `--features onnx` (default): ONNX runtime backends.
- `--features candle`: pure-Rust Candle backends.
- `--features "onnx candle"`: both.
- `--features "eval discourse"`: full eval harness + discourse module (used by CI).
- `--features analysis`: metrics-only diagnostics via `anno::metrics`.
- `--features graph`: graph/KG export via `anno::graph` (pulls in `lattix`).

The `just check-feature-matrix` recipe sweeps the gating combinations that have regressed before.

## Style

- Direct, lowercase prose in docs and commits. No marketing words ("powerful", "robust", "elegant"). No em-dashes in prose.
- Commit messages: `scope: short lowercase description`. Examples: `anno: rename GLINER2 → GLINER_MULTITASK`, `docs: fix QUICKSTART relation example`.
- One commit per logical change. Don't mix renames with behavioral changes.
- `cargo fmt` and `cargo clippy --all-targets -- -D warnings` must pass before `git add`.
- Test names should describe the property under test, not the function under test.
- Avoid `Box<dyn Model>` in new internal APIs unless the dyn-dispatch is load-bearing; static dispatch is preferred.

## Testing

- `just check`: fmt + clippy + quick tests.
- `just test`: unit tests with `eval discourse` features.
- `just test-all`: full workspace including integration.
- Integration tests live in `crates/anno/tests/integration/main.rs` (single-binary consolidation; new tests go in `tests/integration/`, not new test crates).

## Models

External model weights are downloaded from HuggingFace via `hf-hub` on first use and cached under `~/.cache/anno/models/`. For offline or CI environments, set `ANNO_NO_DOWNLOADS=1`; the loader will use cached weights and fall back to non-ML backends if none exist.

To prefetch:

```
cargo run -p anno-cli --features onnx -- models download <backend> [<backend> ...]
```

## Pull requests

- Keep PRs scoped to one concern.
- Show before/after for behavior changes.
- Link the related issue.
- CI must be green before requesting review.

## License

Dual-licensed under MIT or Apache-2.0 at your option. By contributing you agree your contributions are licensed under both.
