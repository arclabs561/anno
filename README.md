# anno

Information extraction: named entity recognition and coreference.

Dual-licensed under MIT or Apache-2.0.

API docs: [docs.rs/anno](https://docs.rs/anno)

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
- ML backends are feature-gated behind `onnx` or `candle`. The published `anno` crate enables `onnx` by default; disable it with `default-features = false`.
- ML model weights download from HuggingFace on first use (see “Offline / downloads” below).
- The table is the **NER backend surface**; for a fuller capability/provenance discussion see `docs/BACKENDS.md`.
- Evaluation tooling adds additional notions (dataset/backend compatibility gates, label-shift “true zero-shot” accounting); those live in `anno-eval`, not in the runtime `Model` trait surface.

## Offline / downloads

- Prefetch models explicitly: `anno models download gliner gliner2 bert-onnx`
- Force cached-only / offline behavior:
  - `ANNO_NO_DOWNLOADS=1` (preferred), or
  - `HF_HUB_OFFLINE=1`

## Examples

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
{
  "entities": [
    {
      "confidence": 0.9800000190734863,
      "end": 21,
      "id": "xxh3:6518c623a81baa6b",
      "negated": false,
      "quantifier": null,
      "start": 8,
      "text": "jobs@acme.com",
      "type": "EMAIL"
    },
    {
      "confidence": 0.949999988079071,
      "end": 33,
      "id": "xxh3:04671e09ebe227c9",
      "negated": false,
      "quantifier": null,
      "start": 25,
      "text": "March 15",
      "type": "DATE"
    },
    {
      "confidence": 0.949999988079071,
      "end": 46,
      "id": "xxh3:59f7cd731485be0b",
      "negated": false,
      "quantifier": null,
      "start": 42,
      "text": "$50K",
      "type": "MONEY"
    }
  ],
  "provenance": {
    "confidence_stats": {
      "count": 3,
      "max": 0.9800000190734863,
      "mean": 0.9599999984105428,
      "median": 0.949999988079071,
      "min": 0.949999988079071,
      "std_dev": 0.014142150234638435
    },
    "elapsed_ms": 79.89025,
    "entity_count": 3,
    "model": "pattern",
    "result_hash": "xxh3:dbe0e2bc98ef6916",
    "text_chars": 52
  }
}
```

Structured entities (dates, money, emails):

```sh
anno extract --model pattern --text "Contact jobs@acme.com by March 15 for the \$50K role."
```

```text
EMAIL:1 "jobs@acme.com" DATE:1 "March 15" MONEY:1 "$50K"
```

Zero-shot extraction (define your own entity types):

```sh
anno extract --model gliner --extract-types "DRUG,SYMPTOM" \
  --text "Aspirin can treat headaches and reduce fever."
```

```text
drug:1 "Aspirin" symptom:2 "headaches" "fever"
```

Coreference resolution:

```sh
anno debug --coref -t "Sophie Wilson designed the ARM processor. She revolutionized mobile computing."
```

```text
Coreference: "Sophie Wilson" → "She"
```

## Library (Rust)

Add the library (from crates.io):

```toml
[dependencies]
anno = "0.2"
```

```rust
use anno::{Model, StackedNER};

let m = StackedNER::default();
let ents = m.extract_entities("Sophie Wilson designed the ARM processor.", None)?;
assert!(!ents.is_empty());
# Ok::<(), anno::Error>(())
```

## Install

```sh
# From this repo:
cargo install --path crates/anno-cli --bin anno --features "onnx eval-advanced"
```

More examples: `docs/QUICKSTART.md`.

## Advanced: sampler (muxer)

If you build with `eval-advanced`, `anno sampler` (alias `anno muxer`) exposes the same randomized
matrix sampler used in CI, with **two modes**:

- **triage**: quick regression-hunting defaults (worst-first)
- **measure**: stable measurement defaults (ml-only)

This command is hidden from the top-level help; use `anno help sampler`.

## Docs

- `docs/QUICKSTART.md`
- `docs/CONTRACT.md` — interface contract
- `docs/BACKENDS.md` — backend selection
