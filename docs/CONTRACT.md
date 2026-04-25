# Contract

This document is the **interface contract** for `anno`: what it does, what it guarantees, and what it intentionally does *not* do.

## What `anno` is

`anno` turns **UTF-8 text** into **structured extractions**:

- **NER**: span detection + entity typing (fixed or zero-shot custom types)
- **Relation extraction**: typed `(head, relation, tail)` triples; available on `RelationCapable` backends (`tplinker`, `gliner_multitask`)
- **Within-document coreference**: cluster mentions into tracks
- **Cross-document coalescing**: cluster tracks across documents into identities

## Primary interoperability contract

- **Offsets are character offsets** (Unicode scalar values), not byte offsets.
- **Core types are the interface**: downstream code should integrate against the stable shapes re-exported by `anno` (prefer `anno::core::*`) (`Entity`, `Signal`, `Track`, `Identity`, `GroundedDocument`, `Corpus`) and treat them as the stable "shape".

## Input text contract (what backends expect)

`anno` backends operate on **plain UTF-8 text**. They will run on "messy" text, but you should be
explicit about what the *authoritative* input string is, because offsets are always relative to the
exact string you pass in.

- **Accepted**: raw text, OCR text, and extracted HTML/PDF text *after* you've turned it into a
  single plain string.
- **Recommended upstream normalization** (in your ingestion layer, e.g. `textprep`):
  - normalize newlines (`\r\n`/`\r` → `\n`) if your sources vary
  - remove bidi controls / suspicious invisibles if your corpora come from untrusted sources
  - avoid "pretty reflow" that changes character positions after extraction (it invalidates spans)
- **CLI convenience**: the CLI exposes `--clean` and `--normalize` flags as opt-in helpers;
  offsets in the output are relative to the post-normalized string.
- **Not a product goal**: `anno` does not promise to be an HTML/PDF parser or a crawler. Feed it
  text; keep parsing/cleaning upstream.

## Scope (what's in / out)

**In scope**
- Inference-time extraction (regex/heuristics/ML backends).
- Zero-shot extraction with custom entity types (`ZeroShotNER` backends: GLiNER, GLiNER multi-task, NuNER -- use `extract_with_types`; CLI `--extract-types`).
- Relation extraction (`RelationCapable` backends: `tplinker`, `gliner_multitask`).
- Graph/KG export: the `graph` feature exposes `GraphDocument` and N-Triples export.
- Evaluation + dataset loading behind feature flags (for benchmarking, not required for usage).

**Out of scope by design**
- Training.
- Document parsing as a product (HTML/PDF pipelines, crawling, etc.). Feed `anno` text; keep ingestion upstream.
- Heavy graph algorithms (community detection, node ranking, etc.). Export via N-Triples and run algorithms downstream (e.g. in `lattix`).

## Feature gating (how to depend on it)

`anno` is a **facade crate** for the workspace: it re-exports the internal implementation crate
(`anno-lib`) and forwards feature flags down to it.

Important default: the facade keeps defaults minimal.

- The `anno` **package** has `default = ["onnx"]` — ONNX ML backends are on by default.
- It depends on `anno-lib` with `default-features = false`.
- Use `default-features = false` in your `Cargo.toml` to opt out of ONNX and pull only what you need.

Major feature flags:

- `onnx`: ONNX Runtime backends (GLiNER, BERT-NER, etc.)
- `candle`: pure-Rust transformer backend (GPU via platform support)
- `analysis`: lightweight analysis primitives (metrics, encoders) — available at inference time, safe to include in production
- `eval`: evaluation harnesses (dataset loading, benchmarking) — only needed for benchmarking runs; pulls in heavier dataset/IO deps
- `discourse`: discourse-level utilities
- `graph`: graph/KG export surface

Treat feature flags as **capability toggles**: depend on the narrowest set you need. In particular, do not include `eval` in a production dependency; use `analysis` instead if you need metrics primitives.

## CLI packages (two `anno` binaries)

The workspace contains **two binaries named `anno`**:

- **Minimal facade CLI**: package `anno` (this crate). Supports `anno extract` with a small dependency set.
- **Full CLI**: package `anno-cli` (`crates/anno-cli/`). Includes benchmarking/eval/datasets tooling and richer commands.

## Integration posture

- **Upstream**: `textprep` handles text cleaning and normalization; `anno` consumes the resulting text. (Internally, `anno-core` uses `sketchir` for similarity sketching during coalescing — this is not a user-facing dependency.)
- **Downstream**: other code can safely:
  - index/store entities/tracks/identities (using the stable `anno` shapes),
  - join with other signals (audio/vision/etc.) using the shared offset discipline,
  - export graphs via `anno-graph::entities_to_knowledge_graph` (which owns the triple construction) or `GraphDocument`, then run algorithms on the exported data.

## Minimal usage obligations

- If you persist results, persist both the **source text identity** (doc id / provenance) and the **character-offset spans**.
- Don't reinterpret spans as byte offsets.

## CLI default (best available)

The default CLI model (`--model stacked`) prefers the **best available** ML backend.

- By default, model downloads are allowed (so first run may be slower).
- To force cached-only / offline behavior: set `ANNO_NO_DOWNLOADS=1` (or `HF_HUB_OFFLINE=1`).
- To prefetch explicitly: use the **full CLI** (`anno-cli`): `anno models download gliner gliner_multitask bert-onnx` (then `stacked` will pick it up).

## Evaluation (two layers)

`anno` has two eval layers with very different runtimes:

- **Single-text eval** (`anno eval`): inline gold spans against one text; typically milliseconds–seconds.
- **Benchmark matrix** (`anno benchmark`): many backends × datasets × seeds; can take minutes to hours. See `docs/QUICKSTART.md` for bounded `just eval-*` profiles and artifact paths.
