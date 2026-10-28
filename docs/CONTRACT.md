# Contract

This contract describes the input text, offsets, and feature flags that callers
must account for when using `anno`. Start with the [quickstart](QUICKSTART.md)
for runnable examples.

## What `anno` is

`anno` turns **UTF-8 text** into **structured extractions**:

- **NER**: span detection + entity typing (fixed or zero-shot custom types)
- **Relation extraction**: typed `(head, relation, tail)` triples; available on `RelationCapable` backends (`tplinker`, `gliner_multitask`)
- **Within-document coreference**: cluster mentions into tracks
- **Cross-document coalescing**: cluster tracks across documents into identities

## Primary interoperability contract

- **Offsets are character offsets** (Unicode scalar values), not byte offsets.
- **Core types are the interface**: use the types re-exported through
  `anno::core::*`: `Entity`, `Signal`, `Track`, `Identity`, `GroundedDocument`,
  and `Corpus`.

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
- Constituency tree interchange with external parsers; see [CONSTITUENCY.md](CONSTITUENCY.md).
- Evaluation + dataset loading behind feature flags (for benchmarking, not required for usage).

**Out of scope by design**
- Training.
- Document parsing as a product (HTML/PDF pipelines, crawling, etc.). Feed `anno` text; keep ingestion upstream.
- Heavy graph algorithms (community detection, node ranking, etc.). Export via N-Triples and run algorithms downstream (e.g. in `lattix`).

## Feature gating (how to depend on it)

The published `anno` crate is the main library. Feature flags select optional capabilities.

- `anno` has `default = ["onnx", "heuristic-fr"]`. ONNX ML backends and French heuristics are on by default.
- Use `default-features = false` in your `Cargo.toml` to opt out of ONNX and pull only what you need.

Major feature flags:

- `onnx`: ONNX Runtime backends (GLiNER, BERT-NER, etc.)
- `candle`: pure-Rust transformer backend (GPU via platform support)
- `analysis`: lightweight analysis primitives (metrics, encoders), available at inference time for production use
- `eval` on `anno-cli` / `anno-eval`: evaluation harnesses (dataset loading, benchmarking); this is not a feature of the main `anno` library.
- `discourse`: discourse-level utilities
- `graph`: graph/KG export surface

Treat feature flags as **capability toggles**: depend on the narrowest set you need. In particular, do not include `eval` in a production dependency; use `analysis` instead if you need metrics primitives.

## CLI

The CLI lives in `anno-cli` (`crates/anno-cli/`). Install with
`cargo install anno-cli` for extraction and batch processing. Add
`--features eval` for evaluation, datasets, and cross-document clustering.

## Integration posture

- **Upstream**: `textprep` handles text cleaning and normalization; `anno` consumes the resulting text. (Internally, `anno::core` uses `sketchir` for similarity sketching during coalescing. This is not a user-facing dependency.)
- **Downstream**: other code can safely:
  - index/store entities/tracks/identities (using the stable `anno` shapes),
  - join with other signals (audio/vision/etc.) using the shared offset discipline,
  - export graphs via `anno::graph::entities_to_knowledge_graph` (which owns the triple construction) or `GraphDocument`, then run algorithms on the exported data. Requires `anno = { features = ["graph"] }`.

## Minimal usage obligations

- If you persist results, keep the document ID and provenance of the source text
  alongside the spans and their character offsets.
- Don't reinterpret spans as byte offsets.

## PII pattern limits

Pattern-based redaction recognizes these formats:

| Format | Validation |
| --- | --- |
| Email addresses | Pattern match |
| US phone numbers and addresses | Pattern match |
| US Social Security numbers | Structure check |
| Card numbers with 16 digits | Luhn checksum |
| IBANs | Mod-97 checksum |

Other national ID formats are not covered.

Printed IBANs use uppercase groups of four separated by single ASCII spaces,
with a shorter final group when needed. Recognition stops at the registered
country length. Text after a space at that endpoint is separate, including
numeric text. Compact IBANs use the existing compact-format recognizer.
These checks validate format and checksum, not account existence or
country-specific bank and account rules.

## CLI default

With `--model stacked`, the CLI tries available model backends before falling
back to patterns and heuristics.

- By default, model downloads are allowed (so first run may be slower).
- To force cached-only / offline behavior: set `ANNO_NO_DOWNLOADS=1` (or `HF_HUB_OFFLINE=1`).
- To prefetch explicitly: use the **full CLI** (`anno-cli`): `anno models download gliner gliner_multitask bert-onnx` (then `stacked` will pick it up).

## CLI JSON provenance

`anno extract --format json` and `--format jsonl` retain the requested CLI backend in
top-level `provenance.model`. Each emitted entity that carries library provenance also has a
nested `provenance` object with its actual `source`, extraction `method`, and available
pattern, raw-confidence, model-version, and timestamp metadata.

Top-level `provenance.entity_sources`, when present, is a sorted unique list of source names
from the entities actually emitted after filtering and validation. It is omitted when no emitted
entity has source metadata. It does not infer a single producing model for stacked output, a
model artifact version, or provider/device placement.

## Evaluation (two layers)

`anno` has two eval layers with very different runtimes:

- **Single-text eval** (`anno eval`): inline gold spans against one text; typically milliseconds–seconds.
- **Benchmark matrix** (`anno benchmark`): many backends × datasets × seeds;
  can take minutes to hours. See [evaluation sampling](EVALUATION_SAMPLING.md)
  for bounded runs and the records needed to interpret them.
