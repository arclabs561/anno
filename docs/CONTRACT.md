# Contract

This document is the **interface contract** for `anno`: what it does, what it guarantees, and what it intentionally does *not* do.

## What `anno` is

`anno` turns **UTF-8 text** into **structured extractions**:

- **NER**: span detection + entity typing
- **Within-document coreference**: cluster mentions into tracks
- **Cross-document coalescing**: cluster tracks across documents into identities

## Primary interoperability contract

- **Offsets are character offsets** (Unicode scalar values), not byte offsets.
- **Core types are the interface**: downstream code should integrate against the stable shapes re-exported by `anno` (or via `anno::core::*`) (`Entity`, `Signal`, `Track`, `Identity`, `GroundedDocument`, `Corpus`, `GraphDocument`) and treat them as the stable “shape”.

## Scope (what’s in / out)

**In scope**
- Inference-time extraction (regex/heuristics/ML backends).
- Evaluation + dataset loading behind feature flags (for benchmarking, not required for usage).

**Out of scope by design**
- Training.
- Document parsing as a product (HTML/PDF pipelines, crawling, etc.). Feed `anno` text; keep ingestion upstream.
- Heavy graph/community-detection toolchains. `GraphDocument` exists for interop/export; run graph algorithms elsewhere.

## Feature gating (how to depend on it)

`anno` is a single publishable crate. Major feature flags:
- `default = ["onnx"]`
- `candle`: pure-Rust transformer backend (GPU via platform support)
- `eval-advanced`: enables evaluation-adjacent helpers (used by `anno-cli` benchmarking)
- `discourse`: discourse-level analysis

Treat feature flags as **capability toggles**: depend on the narrowest set you need.

Note: the `anno` binary lives in the separate `anno-cli` crate (package `anno-cli`, bin `anno`).

## Integration posture

- **Upstream**: `textprep` / `sketchir` handle text cleaning + lightweight structure; `anno` consumes the resulting text.
- **Downstream**: other code can safely:
  - index/store entities/tracks/identities (using the stable `anno` shapes),
  - join with other signals (audio/vision/etc.) using the shared offset discipline,
  - export graphs via `GraphDocument` without importing graph-algorithm choices into `anno`.

## Minimal usage obligations

- If you persist results, persist both the **source text identity** (doc id / provenance) and the **character-offset spans**.
- Don’t reinterpret spans as byte offsets.

## CLI default (best available)

The default CLI model (`--model stacked`) prefers the **best available** ML backend.

- By default, model downloads are allowed (so first run may be slower).
- To force cached-only / offline behavior: set `ANNO_NO_DOWNLOADS=1` (or `HF_HUB_OFFLINE=1`).
- To prefetch explicitly: use `anno models download ...` (then `stacked` will pick it up).

## Evaluation (expected runtime + artifacts)

`anno` has two “eval” layers; they serve different goals and have very different runtimes.

### 1) Local, single-text eval (fast)

Use the CLI `eval` command when you have **one** text and **gold spans** (inline or from a file).
This is typically milliseconds–seconds plus model inference time.

```bash
anno eval --help
```

### 2) Benchmark/eval matrix (can be slow by design)

The “real evaluation” pipeline runs `anno benchmark` across many backends × datasets × seeds and writes a report file.
This can take minutes to hours depending on:
- how many combinations you run,
- model downloads/warmup,
- dataset downloads/IO,
- and whether caches are already populated.

Recommended bounded profiles (generated artifacts; avoid prose claims):

```bash
# Fast, bounded local benchmark (writes reports/eval-quick-report.md)
just eval-quick

# CI-ish sanity (small sample; writes reports/eval-sanity-report.md; cached-only)
just eval-sanity

# Full matrix but bounded (writes your chosen OUTPUT file)
just eval-full-limit MAX_EXAMPLES=50
```

**Source of truth**: read the generated report files under `reports/` (or whatever output path you set), not markdown claims embedded in docs.

If you use spot evaluation, run the aggregator to regenerate:
- `reports/eval-results.jsonl` (source of truth)
- `reports/eval-summary.json`
- `reports/RESULTS.md`

See `scripts/spot/README.md` and run `uv run scripts/spot/aggregate.py --download` (requires AWS credentials and access to the configured bucket).