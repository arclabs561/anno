# Architecture

This repo is **pre-1.0** and prioritizes long-term maintainability over API stability.

## Crate layout (dependency boundaries)

After the 2026-04-26 Phase B consolidation, the workspace ships **three** crates. The earlier `anno-core`, `anno-metrics`, and `anno-graph` packages were folded back into `anno` (their content lives at `anno::core::*`, `anno::metrics::*`, and `anno::graph::*`, the latter behind `feature = "graph"`).

- `crates/anno` (**library + type foundation + backends + metrics + graph export**)
  - Owns: extraction types (entity/coref/grounded), runtime backends (regex/heuristic/onnx/candle/llm), ingest pipeline, env/offset helpers, coreference scoring metrics, KG export adapters.
  - Internal layout:
    - `anno::core::*` — types, traits, errors, dataset/spec metadata, coalescing, span candidate generation. Dependency-light, no ML.
    - `anno::metrics::*` — coref scoring (MUC, B^3, CEAF, LEA, BLANC) and cluster encoders, behind `feature = "analysis"`.
    - `anno::graph::*` — `lattix`-backed adapters from extraction output to `KnowledgeGraph` / `GraphDocument` / N-Triples, behind `feature = "graph"`.
  - The `anno::core` module surface must not depend on heavy ML deps or OS-specific glue.

- `crates/anno-eval` (**evaluation + datasets + muxer integration**)
  - Owns: dataset loaders/registries, evaluation orchestration, muxer-backed sampling.
  - Depends on: `anno`.

- `crates/anno-cli` (**the `anno` binary**)
  - Owns: CLI UX, command wiring, output formatting, file I/O.
  - Depends on: `anno`, `anno-eval`. `anno/graph` is forwarded behind the `graph` feature.

### Intended direction of dependencies

```
anno-cli  ──> anno-eval ──> anno
       └─────────────────> anno
```

If you feel pressure to add a dependency "upwards" (e.g. anno -> anno-eval), that's a design smell.

## Design rules (what goes where)

- **Types and invariants live in `anno::core`**
  - If a concept is reused across modules (IDs, slugs, spans, confidence/coverage concepts), prefer encoding it as a type in `anno::core`.
  - Offsets/spans are **character offsets** (Unicode scalar values / Rust `char` count), not bytes.
  - The `core` module must stay dependency-light: no heavy ML deps, no OS-specific glue.

- **Backends and execution live in `anno::backends`**
  - Feature-gate heavyweight dependencies (onnx/candle/llm) at the `anno` feature surface.
  - Keep "business logic" out of the CLI; the CLI should orchestrate calls into `anno`/`anno-eval`.
  - Model loading should try the target format first (ONNX/safetensors), then auto-convert
    from PyTorch via `uv run scripts/export_*.py` with local caching. See GLiNER ONNX
    (`export_pytorch_to_onnx`) and Candle (`convert_pytorch_to_safetensors`) for the pattern.

- **Evaluation lives in `anno-eval`**
  - Dataset downloading/parsing, metrics aggregation, and muxer selection live here.
  - The eval code should call into `anno` backends via the `anno` API surface.

- **UX and I/O live in `anno-cli`**
  - Reading/writing files, rendering tables/HTML, progress bars, and argument parsing live here.
  - Do not leak CLI-specific dependencies into `anno`.

## Notes on doctests

- `anno` doctests are kept enabled.
- `anno-eval` doctests are currently **disabled** (`[lib] doctest = false`).
  - The full evaluation harness lives in `anno_eval::eval`, while `anno::metrics` (behind the `analysis` feature) provides the dependency-light coref scoring primitives.

