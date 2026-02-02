# Backends

This page avoids benchmark numbers and "working set" claims that drift. Use `anno benchmark` for measurements.

## Model Families

### Neural (2018+, feature-gated)

| Model | Architecture | Zero-shot | Weights |
|-------|--------------|-----------|---------|
| GLiNER | Bi-encoder span classifier | Yes | HuggingFace |
| GLiNER2 | Multi-task span classifier | Yes | HuggingFace |
| NuNER | Token classifier (BIO) | Yes | HuggingFace |
| BERT-NER | Sequence labeling | No | HuggingFace |
| W2NER | Word-word grids (nested) | No | HuggingFace |

### Classical (pre-2015, algorithms with heuristic parameters)

| Model | Method | Notes |
|-------|--------|-------|
| CRF | Conditional Random Fields | Dominant 2001-2015; can load trained weights |
| HMM | Hidden Markov Model | First statistical NER (1997) |

These are methods, not specific trained models. Default parameters are hand-tuned heuristics.
CRF can optionally load trained weights via `CrfNER::with_weights("crf_weights.json")`.

### Rule-based (no weights)

| Model | Method | Entity Types |
|-------|--------|--------------|
| Pattern | Regex | DATE, MONEY, EMAIL, URL, PHONE |
| Heuristic | Capitalization + context | PER, ORG, LOC |

## Choose by constraints

- **No ML deps**: `--model pattern`, `heuristic`, or `stacked` with `default-features = false`
- **Zero-shot custom types**: `--model gliner --extract-types "TYPE1,TYPE2"` (requires `onnx`)
- **Nested entities**: `--model w2ner` (requires `onnx`)
- **Pure Rust inference**: Candle backends (requires `candle`)
- **Offline**: set `ANNO_NO_DOWNLOADS=1` after prefetching with `anno models download`

## Where weights come from

All ML models download from HuggingFace on first use. Default models:

- GLiNER: `onnx-community/gliner_small-v2.1`
- GLiNER2: `onnx-community/gliner-multitask-large-v0.5`
- NuNER: `deepanwa/NuNerZero_onnx`
- BERT-NER: `protectai/bert-base-NER-onnx`

Override with model-specific flags or environment variables.

## Source of truth (generated at runtime)

Use the CLI to see what's available in *your build*:

```bash
anno backends
anno models list
anno models recommend
```

## Measuring performance

Run your own benchmark/eval and keep the results as artifacts:

```bash
anno eval --help
anno benchmark --help  # requires --features eval-advanced
```

Output goes to `reports/`. Treat generated files as the source of truth.

## See also

- [Contract](CONTRACT.md) — scope + guarantees
