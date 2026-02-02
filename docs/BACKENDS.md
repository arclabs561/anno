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

### Classical (pre-2015)

| Model | Method | Notes |
|-------|--------|-------|
| CRF | Conditional Random Fields | Common pre-neural baseline for sequence labeling (incl. NER); ships heuristic params, can load trained |
| HMM | Hidden Markov Model | Historical baseline for sequence labeling; useful for comparison/education |

**Status**:

- HMM ships with hand-tuned heuristic parameters (baseline/education).
- HMM can optionally use **bundled trained params** (priors + transitions, compact) when the `bundled-hmm-params` feature is enabled.
- CRF can optionally use **bundled trained weights** (compact) when the `bundled-crf-weights` feature is enabled, and can also load custom weights.

- CRF can load trained weights: `CrfNER::with_weights("crf_weights.json")`
- Training script: `uv run scripts/train_crf_weights.py`
  - Default training data: WikiANN (PAN-X) via `unimelb-nlp/wikiann` (config `en`)
  - License note: the packaged dataset’s license is discussed in `https://huggingface.co/datasets/unimelb-nlp/wikiann/discussions/6`
  - CoNLL-2003 note: CoNLL-2003’s English text is derived from Reuters/RCV1 and is commonly treated as redistribution-restricted; the CoNLL site notes it “only make[s] available the annotations” and requires separate Reuters corpus access: `http://www.clips.uantwerpen.be/conll2003/ner/`

- Training script (HMM params): `uv run scripts/train_hmm_params.py`
  - Output: `crates/anno/src/backends/hmm_params.json` (priors + transitions only; emissions remain heuristic)

Pointers (for “what good looks like” in classical NER):

- Stanford NER describes itself as a **CRF sequence model** and ships trained English models. See: `https://techfinder.stanford.edu/technology/stanford-named-entity-recognizer`
- The McCallum CRF tutorial discusses the relationship between **HMMs** and **CRFs** in NLP. See: `https://people.cs.umass.edu/~mccallum/papers/crf-tutorial.pdf`
- The CoNLL-2003 shared task paper summarizes baseline behavior and the variety of systems used at the time. See: `https://ar5iv.labs.arxiv.org/html/cs/0306050`

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
