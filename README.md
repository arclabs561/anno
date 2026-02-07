# anno

Information extraction: named entity recognition and coreference.

Dual-licensed under MIT or Apache-2.0.

API docs (may lag behind `main` while crates.io publishing is paused): [docs.rs/anno](https://docs.rs/anno)

## Install

This repo is **not** using crates.io as its primary distribution path right now; see `docs/PUBLISH_STATUS.md`.

### Full CLI (`crates/anno-cli`)

This is the recommended `anno` binary (many commands: `extract`, `debug`, `benchmark`, `models`, etc.):

```sh
# From this repo:
cargo install --path crates/anno-cli --bin anno --features "onnx eval"
```

Or without cloning manually:

```sh
cargo install --git https://github.com/arclabs561/anno --package anno-cli --bin anno --features "onnx eval"
```

### Minimal facade CLI (package `anno`)

This binary is intentionally small and currently supports `anno extract` only:

```sh
# From this repo:
cargo install --path . --bin anno
```

Optional (enable ONNX-backed ML selection for `stacked` at build time):

```sh
cargo install --path . --bin anno --features onnx
```

## What it does

- Named entity recognition (NER): people, orgs, locations, etc.
- Coreference: resolve mentions like "Sophie Wilson" → "She".
- Structured pattern extraction (dates, money, emails).

## Backends

| Backend | Label surface | Structure | Weights | Notes |
|---------|---------------|-----------|---------|-------|
| `stacked` (default) | Mixed (best available) | Flat spans | HuggingFace (when ML enabled) | Selector/fallback: chooses an ML backend when available, otherwise regex+heuristic |
| `gliner` | Custom (zero-shot) | Flat spans | [onnx-community/gliner_small-v2.1](https://huggingface.co/onnx-community/gliner_small-v2.1) | Span classifier; `--extract-types` |
| `gliner2` | Custom (zero-shot) | Flat spans | [onnx-community/gliner-multitask-large-v0.5](https://huggingface.co/onnx-community/gliner-multitask-large-v0.5) | Multi-task (NER + classification) |
| `nuner` | Custom (zero-shot) | Flat spans | [deepanwa/NuNerZero_onnx](https://huggingface.co/deepanwa/NuNerZero_onnx) | Token classifier (BIO), arbitrary-length entities |
| `w2ner` | Fixed (trained labels) | Nested/discont. | [ljynlp/w2ner-bert-base](https://huggingface.co/ljynlp/w2ner-bert-base) | Word-word grids; supports nested spans |
| `bert-onnx` | Fixed (PER/ORG/LOC/MISC) | Flat spans | [protectai/bert-base-NER-onnx](https://huggingface.co/protectai/bert-base-NER-onnx) | Classic CoNLL-style NER |
| `pattern` | Fixed (patterns) | Flat spans | None | Regex for dates, emails, money |
| `heuristic` | Fixed (heuristics) | Flat spans | None | Capitalization + context baseline |
| `crf` | Fixed (trained labels) | Flat spans | Bundled (`bundled-crf-weights`) | Classical baseline; can load custom weights |
| `hmm` | Fixed (trained labels) | Flat spans | Bundled (`bundled-hmm-params`) | Classical baseline/education |
| `ensemble` | Mixed | Flat spans | Varies | Parallel combiner: weighted voting across backends |

Notes:

- All NER backends return **variable-length spans** (start/end offsets). Some are token-labeling models internally.
- Offsets are **character offsets** (Unicode scalar values), not byte offsets; see `docs/CONTRACT.md`.
- ML backends are feature-gated behind `onnx` or `candle`. The published `anno` crate keeps defaults minimal; enable `onnx` explicitly when you want ML backends.
- ML model weights download from HuggingFace on first use (see “Offline / downloads” below).
- The table is the **NER backend surface**; for a fuller capability/provenance discussion see `docs/BACKENDS.md`.
- Evaluation tooling adds additional notions (dataset/backend compatibility gates, label-shift “true zero-shot” accounting); those live in `anno-eval`, not in the runtime `Model` trait surface.

## Offline / downloads

- The facade CLI (package `anno`) does not enable ML backends by default.
- If you enable ML backends (`--features onnx` / `candle`), weights may download on first use.
- Force cached-only / offline behavior:
  - `ANNO_NO_DOWNLOADS=1` (preferred), or
  - `HF_HUB_OFFLINE=1`

## Examples

Note: Most examples below assume the **full CLI** (package `anno-cli`). If you installed the
minimal facade CLI, only `anno extract` is available.

Named entities (human output is compact and may vary by backend/build):

```sh
anno extract --text "Lynn Conway worked at IBM and Xerox PARC in California."
```

```text
PER:1 "Lynn Conway"
ORG:2 "IBM" "Xerox PARC"
LOC:1 "California"
```

Machine-readable output (schema-stable; values vary). This example uses `pattern` to be offline and reproducible; other models use the same JSON shape:

```sh
anno extract --model pattern --format json --text "Contact jobs@acme.com by March 15 for the \$50K role."
```

```json
[
  {
    "text": "jobs@acme.com",
    "entity_type": "EMAIL",
    "start": 8,
    "end": 21,
    "confidence": 0.98
  },
  {
    "text": "March 15",
    "entity_type": "DATE",
    "start": 25,
    "end": 33,
    "confidence": 0.95
  },
  {
    "text": "$50K",
    "entity_type": "MONEY",
    "start": 42,
    "end": 46,
    "confidence": 0.95
  }
]
```

Structured entities (dates, money, emails):

```sh
anno extract --model pattern --text "Contact jobs@acme.com by March 15 for the \$50K role."
```

```text
EMAIL:1 "jobs@acme.com" DATE:1 "March 15" MONEY:1 "$50K"
```

Zero-shot extraction (full CLI; define your own entity types):

```sh
anno extract --model gliner --extract-types "DRUG,SYMPTOM" \
  --text "Aspirin can treat headaches and reduce fever."
```

```text
drug:1 "Aspirin" symptom:2 "headaches" "fever"
```

Coreference resolution (full CLI):

```sh
anno debug --coref -t "Sophie Wilson designed the ARM processor. She revolutionized mobile computing."
```

```text
Coreference: "Sophie Wilson" → "She"
```

## Library (Rust)

Add the library:

```toml
[dependencies]
# Git (recommended; matches this repo while publishing is paused):
anno = { git = "https://github.com/arclabs561/anno", rev = "<commit>" }

# crates.io (may lag behind `main`):
# anno = "0.2"
```

```rust
use anno::{Model, StackedNER};

let m = StackedNER::default();
let ents = m.extract_entities("Sophie Wilson designed the ARM processor.", None)?;
assert!(!ents.is_empty());
# Ok::<(), anno::Error>(())
```

More examples: `docs/QUICKSTART.md`.

## Advanced: sampler (muxer)

If you build with `eval`, `anno sampler` (alias `anno muxer`) exposes the same randomized
matrix sampler used in CI, with **two modes**:

- **triage**: quick regression-hunting defaults (worst-first)
- **measure**: stable measurement defaults (ml-only)

This command is hidden from the top-level help; use `anno help sampler`.

## Docs

- `docs/QUICKSTART.md`
- `docs/CONTRACT.md` — interface contract
- `docs/BACKENDS.md` — backend selection
