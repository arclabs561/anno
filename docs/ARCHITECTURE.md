# Architecture

This repo is **pre-1.0** and prioritizes long-term maintainability over API stability.

## Crate layout (dependency boundaries)

- `crates/anno-core` (**core invariants + data model**)
  - Owns: entity/coref/grounded types, dataset/spec metadata, coalescing primitives.
  - Must not depend on: CLI, evaluation harnesses, heavy ML backends, or OS-specific glue.

- `crates/anno-metrics` (**shared analysis primitives**)
  - Owns: dependency-light metrics + cross-context clustering primitives shared by `anno` and
    `anno-eval` (e.g. coref metrics, cluster encoders).
  - Depends on: `anno-core` (and serde), but **must not** depend on `anno` or `anno-eval`
    (to avoid cycles).

- `crates/anno` (**library + backends**)
  - Owns: runtime backends (regex/heuristic/onnx/candle/llm), ingest/linking/joint pipeline, env/offset helpers.
  - Depends on: `anno-core`.
  - Notes: exposes a small `anno::eval` module behind `analysis` (alias: legacy `eval`) for
    analysis-oriented helpers; it re-exports shared primitives from `anno-metrics`.

- `crates/anno-eval` (**evaluation + datasets + muxer integration**)
  - Owns: dataset loaders/registries, metrics, evaluation orchestration, muxer-backed sampling.
  - Depends on: `anno`, `anno-core`, and `anno-metrics`.

- `crates/anno-cli` (**the `anno` binary**)
  - Owns: CLI UX, command wiring, output formatting, file I/O.
  - Depends on: `anno`, `anno-core`, and optionally `anno-eval` behind features.

### Intended direction of dependencies

```
anno-cli  ─┬─> anno-eval ─┬─> anno
           │             ├─> anno-metrics ──> anno-core
           │             └─> anno-core
           └────────────────> anno-core

anno ───────────────┬──────> anno-core
                    └──────> anno-metrics ──> anno-core
```

If you feel pressure to add a dependency “upwards” (e.g. `anno-core -> anno`), that’s a design smell.

## Design rules (what goes where)

- **Types and invariants live in `anno-core`**
  - If a concept is reused across modules/crates (IDs, slugs, spans, confidence/coverage concepts), prefer encoding it as a type in `anno-core`.
  - Offsets/spans are **character offsets** (Unicode scalar values / Rust `char` count), not bytes.

- **Backends and execution live in `anno`**
  - Feature-gate heavyweight dependencies (onnx/candle/burn/llm) in `anno`.
  - Keep “business logic” out of the CLI; the CLI should orchestrate calls into `anno`/`anno-eval`.

- **Evaluation lives in `anno-eval`**
  - Dataset downloading/parsing, metrics, aggregation, and muxer selection live here.
  - The eval code should call into `anno` backends via the `anno` API surface.

- **UX and I/O live in `anno-cli`**
  - Reading/writing files, rendering tables/HTML, progress bars, and argument parsing live here.
  - Do not leak CLI-specific dependencies into `anno`/`anno-core`.

## Notes on doctests

- `anno-core` doctests are kept enabled.
- `anno-eval` doctests are currently **disabled** (`[lib] doctest = false`) while the split is still settling.
  - The full evaluation harness lives in `anno-eval::eval` (crate name `anno_eval`), while
    `anno::eval` is a small analysis-primitives module behind `analysis`.

