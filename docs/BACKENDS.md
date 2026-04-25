# Backends

This page avoids benchmark numbers and "working set" claims that drift. Use `anno benchmark` for measurements.

## Model Families

### Neural -- ONNX (feature `onnx`)

| Backend | Architecture | Zero-shot | Status | Default model |
|---------|--------------|-----------|--------|---------------|
| `gliner` | Bi-encoder span classifier | Yes | stable | `onnx-community/gliner_small-v2.1` |
| `gliner_multitask` | GLiNER v1 with task-conditioned label prompts (NER + classification + structure; Stepanov & Shtopko 2024) | Yes | beta | `onnx-community/gliner-multitask-large-v0.5` |
| `nuner` | Token classifier (BIO) | Yes | stable | `numind/NuNER_Zero` (also: `NuNER_Zero-4k` 4096 ctx, `NuNER_Zero-span`) |
| `bert_onnx` | BERT sequence labeling | No | beta | `protectai/bert-base-NER-onnx` |
| `w2ner` | Word-word grids (nested) | No | beta | `ljynlp/w2ner-bert-base` |
| `tplinker` | Handshaking tagging (joint entity+relation) | No | beta | -- |
| `glirel` | DeBERTa encoder + scoring head (relations) | Yes | beta | `jackboyla/glirel-large-v0` |
| `gliner_poly` | Poly-encoder with label attention fusion | Yes | WIP | `knowledgator/gliner-bi-large-v1.0` (also: `gliner-bi-small-v1.0`, `modern-gliner-bi-large-v1.0`, `modern-gliner-bi-base-v1.0`; the `gliner-poly-*-v1.0` repos are model cards only with no weights) |
| `gliner_onnx` | GLiNER manual ONNX impl | Yes | beta | `onnx-community/gliner_small-v2.1` |
| `gliner_pii` | GLiNER PII Edge (60+ PII categories) | Yes | beta | `knowledgator/gliner-pii-edge-v1.0` |
| `gliner_relex` | GLiNER-RelEx joint NER+RE | Yes | beta | `knowledgator/gliner-relex-large-v1.0` |
| `deberta_v3` | DeBERTa-v3 NER (local export) | No | WIP | -- |
| `albert` | ALBERT NER (local export) | No | WIP | -- |

### Neural -- Candle (feature `candle`)

| Backend | Architecture | Zero-shot | Status | Default model |
|---------|--------------|-----------|--------|---------------|
| `gliner_candle` | GLiNER via Candle (pure Rust) | Yes | beta | `urchade/gliner_small-v2.1` (also: `knowledgator/gliner-bi-base-v2.0`, `gliner-bi-large-v2.0`) |
| `candle_ner` | BERT NER via Candle | No | beta | `dslim/bert-base-NER` |

### Neural -- LLM (feature `llm`)

| Backend | Architecture | Zero-shot | Status | Default model |
|---------|--------------|-----------|--------|---------------|
| `universal_ner` | LLM-backed zero-shot (OpenRouter/Anthropic/Groq/Ollama) | Yes | beta | `google/gemini-2.5-flash-lite` |

### Classical (no feature gate)

| Backend | Method | Status | Notes |
|---------|--------|--------|-------|
| `crf` | Conditional Random Fields | stable | Ships heuristic params, can load trained |
| `hmm` | Hidden Markov Model | stable | Historical baseline; optional bundled trained params |
| `heuristic_crf` | CRF + heuristic emissions | stable | CRF sequence labeling with gazetteer/word-shape features |
| `ensemble` | Weighted voting across backends | beta | Combines multiple backend outputs |

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
  - Output: `crates/anno/src/backends/hmm_params.json` (priors + transitions + compact emission backoff; no word-identity emissions)
  - Default behavior (when `bundled-hmm-params` is enabled): `HmmNER::new()` uses the bundled params for a real end-to-end baseline.
    - You can disable bundled dynamics via `HmmConfig { use_bundled_dynamics: false, ..Default::default() }` (or `ANNO_HMM_NO_BUNDLED_DYNAMICS=1`).

Pointers (for “what good looks like” in classical NER):

- Stanford NER describes itself as a **CRF sequence model** and ships trained English models. See: `https://techfinder.stanford.edu/technology/stanford-named-entity-recognizer`
- The McCallum CRF tutorial discusses the relationship between **HMMs** and **CRFs** in NLP. See: `https://people.cs.umass.edu/~mccallum/papers/crf-tutorial.pdf`
- The CoNLL-2003 shared task paper summarizes baseline behavior and the variety of systems used at the time. See: `https://ar5iv.labs.arxiv.org/html/cs/0306050`

### Rule-based (no feature gate)

| Backend | Method | Entity Types |
|---------|--------|--------------|
| `stacked` (default) | Pattern + heuristic combined | PER, ORG, LOC, DATE, MONEY, etc. |
| `pattern` | Regex | DATE, MONEY, EMAIL, URL, PHONE |
| `heuristic` | Capitalization + context | PER, ORG, LOC |

## GLiNER entity type limit

Cross-encoder GLiNER models (e.g. `gliner_small-v2.1`) encode entity type labels
jointly with the input text. Performance degrades beyond ~30 entity types per
inference call. If you need more types, batch them into groups of 20-30 and merge
results across calls.

The `knowledgator/gliner-bi-*-v2.0` bi-encoder models pre-compute label
embeddings independently from the input text. This gives ~130x speedup at high
label counts since label embeddings can be cached and reused across inputs. These
models are available for the `gliner_candle` backend (safetensors). Pre-converted
ONNX exports are not yet available for the `gliner`/`gliner_onnx` backends.

Source: practitioner findings from the GLiNER community and "Illustrated GLiNER"
(Shahrukh Khan). Bi-encoder speedup figure from Knowledgator's model card.

## Backend setup (export scripts and weights)

Most backends auto-download their default model from HuggingFace on first
load. A few require a one-time local ONNX export via a Python script
(those models either lack a ready-to-use ONNX repo on HF, or use an
architecture that needs a specific export tweak). After the export, the
runtime loader picks the artifact up from the documented path or env var.

| Backend | Auto-download | Script | Default output | Path override |
|---------|---------------|--------|----------------|---------------|
| `gliner` / `gliner_onnx` | yes | (auto-export on first load if no ONNX cached) | hf-hub cache | `--` |
| `gliner_multitask` | yes | `--` | hf-hub cache | `--` |
| `bert_onnx` | yes | `--` | hf-hub cache | `--` |
| `gliner_candle` / `candle_ner` | yes | `--` | hf-hub cache | `--` |
| `gliner_pii` | yes | `--` | hf-hub cache | `--` |
| `gliner_relex` | yes (script needed only for re-export) | `export_gliner_relex_onnx.py` | hf-hub cache | `--` |
| `nuner` | yes (script needed only for unsupported variants) | `export_nuner_to_onnx.py` (delegates to `export_gliner_poly_onnx.py` for GLiNER-based variants) | hf-hub cache | `--` |
| `gliner_poly` | no | `export_gliner_poly_onnx.py` | `~/.cache/anno/models/gliner-poly/` | (cache probe) |
| `glirel` | no | `export_glirel_onnx.py` | `~/.cache/anno/models/glirel/` | (cache probe) |
| `deberta_v3` | no | `export_deberta_ner_to_onnx.py` | `~/.cache/huggingface/hub/models--deberta-v3-ner/onnx/` | `DEBERTA_MODEL_PATH` |
| `biomedical` | no | `export_biomedical_ner_to_onnx.py` | `~/.cache/anno/models/biomedical-ner/` | `BIOMEDICAL_MODEL_PATH` |
| `w2ner` | no | `export_w2ner_to_onnx.py` | path argument | `W2NER_MODEL_PATH` |
| `tplinker` | no (heuristic-mode fallback always works) | `export_tplinker_onnx.py` | `~/.cache/anno/models/tplinker/` | (cache probe) |
| `fcoref` (coref) | no | `export_fcoref.py` | `./fcoref_onnx/` (relative) or pass `--output-dir` | `FCOREF_MODEL_PATH` |

Run any script via `uv run`:

```sh
uv run scripts/export_<backend>_onnx.py [--model <hf-id>] [--output <dir>]
```

The scripts are PEP 723-annotated so `uv` resolves their Python deps in an
isolated environment. The runtime loader prints the exact script command
in its error message when the artifact is missing.

Notes:

- `gliner_poly` uses bi-encoder model weights (`gliner-bi-large-v1.0` family)
  even though the backend name says "poly" — the `gliner-poly-*-v1.0` HF repos
  are model cards only with no weights, per the export script's docstring.
- `tplinker` exports with random weights if no `--checkpoint` is provided;
  the runtime falls back to a heuristic mode when no ONNX is present.
- `gliner_onnx` runs `export_gliner_poly_onnx.py` automatically on first
  load if no ONNX file is in the cache (no manual step required).

## Choose by constraints

- **No ML deps**: `--model pattern`, `heuristic`, or `stacked` with `default-features = false`
- **Zero-shot custom types**: `--model gliner --extract-types "TYPE1,TYPE2"` (requires `onnx`)
- **Relations (best-effort)**: `--model gliner_multitask --extract-relations` (requires `onnx`) or `--model tplinker --extract-relations` (heuristic baseline). Use `--relation-types "FOUNDED,WORKS_FOR"` to constrain labels.
- **Nested entities**: `--model w2ner` (requires `onnx`)
- **Pure Rust inference**: Candle backends (requires `candle`)
- **Offline**: set `ANNO_NO_DOWNLOADS=1` after prefetching with `anno models download`

## Helpers (not NER backends)

Some optional modules are *helpers* that operate over the same span/offset contract, but they are
not “backends” in the NER table sense:

- **Chunking helpers**: `anno::backends::semantic_chunking` always provides a lightweight
  rule-based chunker (paragraph boundaries + size limits + overlap). The `semantic-chunking`
  feature adds a sentence-similarity strategy (still dependency-light; no embedding model).
  Chunking does not change extraction shapes; it only decides which slices of text to run
  extraction over.
- **`discourse` feature**: discourse-level utilities (centering, shell nouns, abstract referents).
  These operate on **character-offset spans** (events/propositions still need localization), and
  are primarily used by evaluation tooling.

## Where weights come from

All ML models download from HuggingFace on first use. See the tables above for default model IDs per backend.

Override with model-specific flags or environment variables.

## ONNX export scripts

Some models only distribute PyTorch weights. Export scripts in `scripts/` convert them to ONNX for use with anno's inference backends. All scripts use PEP 723 inline metadata and run with `uv run`.

| Script | Target model | Notes |
|--------|-------------|-------|
| `export_gliner_poly_onnx.py` | GLiNER bi-encoder (v1/v2) | Dual strategy: library export, then manual fallback. Produces `model.onnx` + `label_encoder.onnx` |
| `export_nuner_to_onnx.py` | NuNER Zero / Zero-4k | Auto-detects architecture (token classifier vs GLiNER). Delegates GLiNER variants to `export_gliner_poly_onnx.py` |
| `export_deberta_ner_to_onnx.py` | DeBERTa-v3 NER | Standard token classifier export |
| `export_biomedical_ner_to_onnx.py` | d4data/biomedical-ner-all | Uses Optimum; optional INT8 quantization |
| `export_w2ner_to_onnx.py` | W2NER | Simplified architecture (fixed-length inputs for ONNX compat) |
| `export_tplinker_onnx.py` | TPLinker | Joint entity-relation extraction |
| `export_glirel_onnx.py` | GLiREL | Relation extraction; falls back to PyTorch weights if ONNX export fails |
| `export_gliner_relex_onnx.py` | GLiNER-RelEx (joint NER+RE) | Dual output: entity_scores + relation_scores. Falls back to PyTorch |
| `export_fcoref.py` | f-coref | Splits encoder (ONNX) from scorer heads (safetensors) |

GLiNER ONNX backends (`gliner_onnx`) auto-export on first load if no ONNX file is cached. The auto-export calls `export_gliner_poly_onnx.py` via `uv run` or `python3`.

## Source of truth (generated at runtime)

Use the CLI to see what's available in *your build*:

```bash
anno models list
anno models info gliner
anno info
```

## Measuring performance

Run your own benchmark/eval and keep the results as artifacts:

```bash
anno eval --help
anno benchmark --help  # requires --features eval
```

Output goes to `reports/`. Treat generated files as the source of truth.

## See also

- [Quickstart](QUICKSTART.md) — getting started + common flags
- [Contract](CONTRACT.md) — scope + guarantees
- [Architecture](ARCHITECTURE.md) — how the pieces fit together
- [Publish status](PUBLISH_STATUS.md) — what’s stable vs experimental
