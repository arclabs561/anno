# Backends

This page avoids benchmark numbers and "working set" claims that drift. Use `anno benchmark` for measurements.

## Choosing a backend

Start with the labels and domain you need, then compare explicit backends on
your own annotated text. `bert_onnx` is a candidate for conventional named
entities; GLiNER supports configurable labels. `pattern` is useful for structured
values such as emails and URLs. Capability matching determines which evaluations
are meaningful; it does not establish accuracy.

`stacked` combines outputs and is convenient for mixed extraction, but is not a
fixed model or a guarantee of higher F1. With the `onnx` feature, its default
construction attempts BERT and NuNER, then GLiNER if neither loads, alongside
patterns and heuristics. Without an available ML layer it falls back to patterns
and heuristics. `ANNO_NO_DOWNLOADS=1` still permits already-cached models, so
record the entity provenance and available model artifacts when comparing runs.

For a repeatable starting comparison, use the tracked
[QA procedure](../.agents/skills/anno-qa/SKILL.md) and
[fixed panel](../scripts/qa/core-panel.json). The `ner-baseline` suite compares
four backends on news, Wikipedia, and social-media data; `classical-ner` adds
the classical implementations. Both use fixed seeds and example budgets, and
produce validated JSON plus a Markdown table. Prepare datasets and models first;
the panel itself disables downloads. Do not interpret a successful run, a
declared incompatibility, or a small sampled score as broad benchmark coverage.

Evaluation backend IDs and extraction CLI names differ: use `bert_onnx` and
`gliner_onnx` in the evaluation panel, versus `--model bert-onnx` and
`--model gliner` with `anno extract`.

## Model Families

### Neural: ONNX (feature `onnx`)

| Backend | Architecture | Zero-shot | Status | Default model |
|---------|--------------|-----------|--------|---------------|
| `gliner` (canonical) / `gliner_onnx` (low-level) | Bi-encoder span classifier | Yes | stable | `onnx-community/gliner_small-v2.1` |
| `gliner_multitask` | GLiNER v1 with task-conditioned label prompts (NER + classification + structure; Stepanov & Shtopko 2024) | Yes | beta | `onnx-community/gliner-multitask-large-v0.5` |
| `gliner2_fastino` | Fastino GLiNER2, eight-graph ONNX pipeline | Yes | experimental | `jugaadsrl/gliner2-multi-v1-onnx` at `4241d7c66b648e618c89c150bf4cf418d2f83159` |
| `nuner` | Token classifier (BIO) | Yes | stable | `numind/NuNER_Zero` (also: `NuNER_Zero-4k` 4096 ctx, `NuNER_Zero-span`) |
| `bert_onnx` | BERT sequence labeling | No | beta | `protectai/bert-base-NER-onnx` |
| `w2ner` | Word-word grids (nested) | No | beta | `ljynlp/w2ner-bert-base` |
| `tplinker` | Handshaking tagging (joint entity+relation) | No | beta | -- (no public pre-trained weights; runs a heuristic fallback unless you supply a checkpoint to `export_tplinker_onnx.py`) |
| `glirel` | DeBERTa encoder + scoring head (relations) | Yes | beta | `jackboyla/glirel-large-v0` |
| `gliner_poly` | Poly-encoder with label attention fusion | Yes | WIP | `knowledgator/gliner-bi-large-v1.0` (also: `gliner-bi-small-v1.0`, `modern-gliner-bi-large-v1.0`, `modern-gliner-bi-base-v1.0`; the `gliner-poly-*-v1.0` repos are model cards only with no weights) |
| `gliner_pii` | GLiNER PII Edge (60+ PII categories) | Yes | beta | `knowledgator/gliner-pii-edge-v1.0` |
| `gliner_relex` | GLiNER-RelEx joint NER+RE | Yes | beta | `knowledgator/gliner-relex-large-v1.0` |
| `deberta_v3` | DeBERTa-v3 NER (local export) | No | WIP | -- |
| `albert` | ALBERT NER (local export) | No | WIP | -- |

Note: `gliner` is a smart-default alias. Under `--features onnx` it
resolves to `anno::GLiNEROnnx`; under `--features candle` only (no
`onnx`) it falls back to `anno::GLiNERCandle`. Pick `gliner` when you
want GLiNER and don't mind which backend serves it. Pick `gliner_onnx`
or `gliner_candle` explicitly to force one backend and fail at
construction time if its feature isn't enabled.

### Neural: Candle (feature `candle`)

| Backend | Architecture | Zero-shot | Status | Default model |
|---------|--------------|-----------|--------|---------------|
| `gliner_candle` | GLiNER via Candle (pure Rust) | Yes | experimental | `urchade/gliner_small-v2.1` is configured, but checkpoint compatibility is not validated |
| `candle_ner` | BERT NER via Candle | No | beta | `dslim/bert-base-NER` |

`GLiNERCandle::from_assets` accepts configuration, tokenizer JSON, and
safetensors bytes without downloading files or spawning Python. Native
`from_pretrained` retains its memory-mapped weight loading. The GLiNER loader
requires its existing BERT tensor layout; byte-backed construction does not
add DeBERTa or ModernBERT support.

The experimental Candle target build excludes native network and
filesystem model constructors:

```sh
rustup target add wasm32-unknown-unknown
cargo build -p anno --target wasm32-unknown-unknown --no-default-features --features candle
```

Use Cargo and rustc from the toolchain where the target is installed. This is
library build support, not a JavaScript binding or a verified browser demo.
A compatible trained checkpoint and browser inference test are still required.
The public browser feature remains deferred until those gates pass.

### ONNX execution providers

`create_onnx_session_with_provider` accepts an explicit `OnnxExecutionProvider`
and overrides the older CUDA/CoreML preference flags in `OnnxSessionConfig`.
CUDA, CoreML, DirectML, and ROCm require `onnx-cuda`, `onnx-coreml`,
`onnx-directml`, and `onnx-rocm`, respectively. An unavailable requested
provider returns an error. Successful registration still permits CPU
execution of unsupported graph nodes; it does not prove full GPU placement.

The `onnx_cuda_smoke` and `onnx_coreml_smoke` examples accept an optional
local ONNX path to verify registration without a download. CUDA and DirectML
runtime validation requires the corresponding hardware and drivers.

DirectML sessions disable memory patterns and use sequential execution, as
required by the [DirectML provider](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html).
Real-model placement on Windows remains unverified.

The `onnx-rocm` feature is a legacy binding, not a supported AMD runtime path
with the current dependency: `ort` targets ONNX Runtime 1.24, while the
[ROCm provider was removed in 1.23](https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html).
Upstream recommends MIGraphX. Selecting and validating a compatible AMD
provider is tracked in [issue #19](https://github.com/arclabs561/anno/issues/19).

For a cached BERT graph, `onnx_bert_cuda_provider_probe` retains CPU/CUDA
profiles, artifact hashes, label agreement, and the strict logit comparison.
Run a matched TF32 pair on Linux/CUDA with separate receipt directories:

```sh
cargo run -p anno --release --example onnx_bert_cuda_provider_probe --features onnx,onnx-cuda -- /model-dir /receipts/tf32-on --tf32=true
cargo run -p anno --release --example onnx_bert_cuda_provider_probe --features onnx,onnx-cuda -- /model-dir /receipts/tf32-off --tf32=false
```

The model directory must contain `model.onnx`, `tokenizer.json`, and
`config.json`. The probe's `cuda_tf32` receipt field records the explicit
choice; omitting the option leaves the runtime default unchanged and records
`null`. The tolerance stays `1e-4` in both runs. Tolerance, label-agreement,
or placement failures return a nonzero exit after saving the receipt and
profiles. Invalid logits in the initial comparison (empty, mismatched, or
non-finite) terminate with an error before producing a receipt. Warm and
timed iterations measure latency; they do not repeat the comparison.
Results cover the fixed probe input,
not general model quality or throughput. These are diagnostic options, not
changes to the library's provider defaults.

### Neural: Fastino GLiNER2 (feature `gliner2-fastino`)

`GLiNER2Fastino` is available behind the `gliner2-fastino` feature. Its
optional Candle implementation is behind `gliner2-fastino-candle`; see
[the feature-gating contract](CONTRACT.md) for the supported feature surface.
The ONNX evaluator uses the pinned `jugaadsrl/gliner2-multi-v1-onnx`
snapshot `4241d7c66b648e618c89c150bf4cf418d2f83159`; it requires about
1.25 GB for the selected `fp32_v2` files on first use.

### Neural: LLM (feature `llm`)

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

### Patterns, heuristics, and composed defaults

| Backend | Method | Entity Types |
|---------|--------|--------------|
| `stacked` (default) | Pattern + heuristic, plus available ML layers with `onnx` | PER, ORG, LOC, DATE, MONEY, etc. |
| `pattern` | Regex | DATE, MONEY, EMAIL, URL, PHONE |
| `heuristic` | Capitalization + context | PER, ORG, LOC |

## GLiNER entity type limit

Cross-encoder GLiNER models (e.g. `gliner_small-v2.1`) encode entity type labels
jointly with the input text. For large label sets, measure the chosen model and
batch labels when that fits the workload.

The `knowledgator/gliner-bi-*-v2.0` bi-encoder models pre-compute label
embeddings independently from the input text. These models are available for the
`gliner_candle` backend (safetensors). Pre-converted ONNX exports are not yet
available for the `gliner`/`gliner_onnx` backends.

## Backend setup (export scripts and weights)

Most backends auto-download their default model from HuggingFace on first
load. A few require a one-time local ONNX export via a Python script
(those models either lack a ready-to-use ONNX repo on HF, or use an
architecture that needs a specific export tweak). After the export, the
runtime loader picks the artifact up from the documented path or env var.

### Why we need the scripts

anno runs models through `ort` (ONNX Runtime), so every backend needs
an `.onnx` file on disk. Some HuggingFace repos already ship a
pre-converted ONNX (`gliner`, `gliner_multitask`, `bert_onnx`, etc.) and
the runtime just downloads them. The rest ship only PyTorch
(`pytorch_model.bin` / `model.safetensors`), which anno can't consume
directly. Those are the ones that need a one-time Python script to
convert.

Why not link PyTorch in anno itself? It would push the binary from ~50
MB to multi-GB, pull in CUDA-toolkit deps even on CPU-only hosts, and
slow startup. Pre-converting once means the production runtime ships
only `ort` plus the model artifact.

#### Architecture-specific gotchas

A few backends need more than a stock `optimum-cli export onnx`:

- **DeBERTa-v3**: pin `transformers==4.47.0` to avoid a fast-tokenizer
  bug; copy SentencePiece artifacts from the HF cache because the
  fast-tokenizer conversion is broken in newer versions.
- **GLiNER-RelEx**: scatter-based subword pooling, so `torch.onnx.
  export` needs a real tokenized input (random dummy inputs cause
  index-out-of-bounds during JIT tracing).
- **GLiNER bi-encoder** (`gliner_poly`): produces **two** ONNX graphs --
  a text encoder and a separate label encoder for the BGE
  sentence-transformer label embeddings.
- **F-coref**: bypass `fastcoref`'s spaCy-dependent high-level API and
  load via `transformers` directly; exports the DistilRoBERTa encoder
  as ONNX and the mention-ranking scorer as safetensors.
- **W2NER**: the simplified export removes the LSTM head and substitutes
  attention pooling. The runtime expects this simplified form.
- **TPLinker**: no public HF distribution of trained weights; the
  script requires a `--checkpoint` argument and otherwise exports with
  random weights (heuristic-mode fallback covers users who don't have
  a checkpoint).

#### Model-card-only repos

Some HF repos exist as documentation but ship no weights. Example:
`knowledgator/gliner-poly-{base,small}-v1.0`. The actual loadable
weights live under different repo names (`gliner-bi-*-v1.0`).

#### GLiNER auto-export shortcut

`gliner_onnx` runs `export_gliner_poly_onnx.py` automatically on first
load if no ONNX is cached, so end users don't need to invoke the script
manually for that one backend.

### Nuances

- Each script is annotated with PEP 723 inline metadata, so `uv run
  scripts/export_*.py` resolves Python deps in a hermetic env. No
  `pip install` polluting the global Python.
- Opset target varies: gliner_poly uses 17, the others mostly target 14.
  ort handles both; the difference is invisible at runtime.
- Output layout varies: some scripts write `model.onnx` flat, some
  write `onnx/model.onnx`. The runtime loader's
  `hf_loader::download_onnx_model` tries both candidates.
- Tokenizer fallback: `tokenizer.json` is preferred. A few older BERT
  models (`dslim/bert-base-NER`) only ship `vocab.txt`; CandleNER's
  loader handles both.
- The default load path for some scripts (`gliner_poly`, `glirel`,
  `tplinker`) is anno's own cache (`~/.cache/anno/models/<name>/`),
  which is separate from the hf-hub cache (`~/.cache/huggingface/hub/`).
  Both are honored; just different conventions.
- Quantization: most scripts accept `--quantize` to emit an int8 or
  fp16 ONNX variant. The runtime loader prefers
  `model_quantized.onnx` / `model_int8.onnx` when present.

| Backend | Auto-download | Script | Default output | Path override |
|---------|---------------|--------|----------------|---------------|
| `gliner` / `gliner_onnx` | yes | (auto-export on first load if no ONNX cached) | hf-hub cache | `--` |
| `gliner_multitask` | yes | `--` | hf-hub cache | `--` |
| `gliner2_fastino` | yes | `--` | hf-hub cache | `--` |
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
  even though the backend name says "poly". The `gliner-poly-*-v1.0` HF repos
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

- **Chunking helpers**: `anno::backends::chunking` splits text into overlapping
  chunks with source character offsets. The `chunking` feature adds
  `chunk_text_semantic`, which uses `text-splitter` for Unicode sentence, word,
  and grapheme boundaries.
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

- [Quickstart](QUICKSTART.md): getting started + common flags
- [Contract](CONTRACT.md): scope + guarantees
- [Architecture](ARCHITECTURE.md): how the pieces fit together
- [Publish status](PUBLISH_STATUS.md): what’s stable vs experimental
