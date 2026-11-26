# Backends

## Choosing a backend

On “Ada founded Acme. She leads it.”, NER can return the typed spans “Ada” and “Acme”. A coreference resolver can group “Ada” with “She”. A relation extractor can return a directed `founded` edge from the Ada mention to the Acme mention. A document-level model can also predict an edge between two clusters. Each output answers a different question, and each can be evaluated separately. [Grounded representation](decisions/grounded-extraction.md) · [Joint extraction design](decisions/joint-extraction.md)

The [backend error analysis](eval/backend-residuals.md) counts missing and extra outputs by stage. To choose a checkpoint or change a threshold, start with the comparison that matches your input and output.

| Decision | Reproducible comparison |
|---|---|
| Pick a NER backend for short and long text in a particular language | [Source-verified multilingual NER panels](eval/ner-language-length.md) |
| Retain more mention candidates or use dependency heads for coreference | [Raw-text mention budget](eval/coref-panels.md#doubling-the-fcoref-mention-budget) · [Predicted heads on supplied and returned mentions](../crates/anno-eval/research/reference/adjacent/multilingual/README.md#predicted-dependency-heads) |
| Rank candidate directed relations with a fixed budget | [JointIE argument and ranking trial](eval/jointie-role-trial.md) |
| Separate a clustering error from a relation threshold or annotation gap | [JEREX controls on two annotation editions](eval/jerex-controls.md) |

### Which inputs do you have?

| You have | You need | Route to try | What controls the result |
|---|---|---|---|
| Text containing emails, dates, URLs or amounts | Typed occurrences, without model weights | `anno extract --model pattern` | Fixed rules determine coverage. `heuristic` adds capitalization/context-based names. `crf` and `hmm` provide classical sequence baselines. [Classical models](#classical-no-feature-gate) |
| Text and a fixed tag inventory matching a trained model | Conventional named entities | `anno extract --model bert-onnx` with a pinned checkpoint | The checkpoint’s BIO mapping defines the available labels. [Checkpoint selection](#selecting-ner-checkpoints) |
| Text and entity labels supplied at runtime | Typed mention spans | `anno extract --model gliner --extract-types person,organization` or `--model nuner` | GLiNER v1 scores prompted spans. NuNER Zero and Zero-4k score words and merge adjacent same-label words. NuNER Zero-span scores trained multiword candidates directly. Verify the actual checkpoint and language on your data. [Span export details](#span-width-exports-max_width) · [Native checkpoint replays](../scripts/qa/checkpoint_replay/README.md) |
| Text and custom labels in the GLiNER2.5 schema budget | A flat set of labeled spans | `anno extract --model gliner25` with `GLINER25_MODEL_DIR` | The native boundary export uses a shared 512-token prompt/text window and rejects oversized input. The separate `anno joint` entry point accepts a relation schema. [GLiNER2.5](#fastino-gliner25) |
| Text, known entity occurrences and predicate labels | Directed edges among those occurrences | Rust `GLiREL::extract_relations(text, entities, relation_types, threshold)` | Give each occurrence its source span. Missed or misaligned upstream mentions affect which pairs can be scored. [GLiREL API](../crates/anno/src/backends/glirel/onnx.rs) |
| Text and entity/predicate vocabularies | Mentions and directed relations from a trained joint export | `anno extract --model gliner-relex --extract-types person,organization --extract-relations --relation-types founded` | The entity-only export serves NER. A relation request requires the `--joint` export and a compatible relation head. The prompt’s label inventory materially affects scores. [Export and replay](#architecture-specific-gotchas) · [CrossRE experiments](eval/crossre-full-prompt-comparison.md) |
| Text and a typed relation schema with graph constraints | Mention IDs and a constrained document graph | `anno joint --checkpoint CHECKPOINT --schema schema.json --text 'Ada founded Acme.'` | The upstream Python JointIE provider scores overlapping windows, then decodes one graph. Only co-window candidates have direct model scores. Output exposes scored windows, feasibility and derived edges. [Schema, example and runtime](#fastino-gliner25) |
| DocRED sentences in token order | Mention detections, coreference clusters and directed **concept** edges | JEREX Python prediction, then the Rust `import_jerex` example | The predictor takes tokens with a closed label set. The importer aligns them against your exact source and retains endpoint memberships. Upstream supplies no relation probability or mention-pair witness. [Real checkpoint and import commands](eval/joint-stage-baseline.md) |

The GLiNER name covers several interfaces. When labels repeat across many documents, `gliner-poly` can use a split text encoder, label encoder and fusion head. When the desired output is a document category or structured fields, the Fastino GLiNER2 Rust backend exposes `classify` and `extract_structure`. Those return different output types from the GLiNER2.5 boundary NER and JointIE routes above. [Split encoder setup](#selecting-ner-checkpoints) · [Fastino API](../crates/anno/src/backends/gliner2_fastino/mod.rs)

For NuNER Zero-4k, FP32 and QUInt8 both scored 50.7% exact span-and-label F1 on 24 hash-selected English MultiNERD test sentences at threshold 0.5. Across all 38 inputs, including separate Unicode and constructed long-text cases, the two graphs returned different spans on ten. The checkpoint card lists English, and one long-input case uses Russian. The smaller graph is an experimental option for callers who can check their own labels and threshold. FP32 remains the reference. [Labeled comparison and artifact sizes](eval/nuner-4k-labeled.md)

For overlapping or nested mentions, inspect W2NER’s word-grid output and the simplifications in its local export. For a hosted model, UniversalNER uses a configured chat-completions endpoint. API availability, model ID and offset recovery then become part of the run. [W2NER export](#architecture-specific-gotchas) · [LLM setup](#neural-llm-feature-llm)

The extraction default `stacked` combines rules and available models. Its ONNX configuration attempts BERT and NuNER, then tries GLiNER if neither loads. Its rule layers can still extract entities without a neural model. For a deterministic rule-only run, select `pattern` explicitly. `ANNO_NO_DOWNLOADS=1` permits cached weights. Evaluation IDs can differ from CLI names, for example `bert_onnx` and `gliner_onnx` in evaluations versus `--model bert-onnx` and `--model gliner` in extraction. [Evaluation panels](../scripts/qa/core-panel.json) · [CLI model selectors](../crates/anno-cli/src/cli/parser.rs)

### Which mentions should coreference use?

| Resolver | Input → output | Calling and interpreting it |
|---|---|---|
| Native mention-ranking | Raw text → predicted mention clusters. The entity-only entry point also accepts supplied detections. | Runs without model weights. The entity-only arm receives surfaces/types in source order and lacks surrounding prose, which changes the available antecedent features. [Resolver](../crates/anno/src/backends/coref/mention_ranking) · [Comparison](eval/coref-panels.md) |
| FCoref | Raw text → detected mentions grouped by antecedent links | Anno implements the f-coref DistilRoBERTa encoder in ONNX with Rust mention and antecedent scoring. Call `FCoref::from_path(dir)?.resolve(text)` to require that model. The public resolver proposes its own mentions. An input beyond the configured document token budget returns an error. [Upstream model](https://huggingface.co/biu-nlp/f-coref) · [Rust implementation and input limits](../crates/anno/src/backends/coref/fcoref/mod.rs) |
| Maverick | Raw text → predicted clusters, or source text plus supplied spans → clusters over caller-proposed mentions | Select the persistent Python worker with `anno pipeline --model heuristic --coref --maverick-model CHECKPOINT`. The Rust `resolve_mentions` entry point checks supplied character offsets against spaCy tokens. Token-splitting spans return an error. Singleton handling depends on the checkpoint and entry point. [Runtime, license and API](#maverick-coreference) |

In `anno pipeline`, `--model` selects mention extraction. Plain `--coref` tries FCoref on ONNX builds and can fall back to native mention-ranking. A grounded document’s `coref_run` distinguishes a completed run with zero tracks, a failed run, and prior failed attempts before a fallback completed. Explicit Maverick worker errors propagate to the caller. [Pipeline routing](../crates/anno-cli/src/cli/commands/pipeline.rs) · [Default coreference attempts](../crates/anno-cli/src/cli/utils.rs)

With the `discourse` feature, speaker-turn mode links English first-person singular mentions to the supplied speaker and omits general name and third-person chains. For identities spanning files, `pipeline --cross-doc` takes a separate step over document tracks. [Pipeline options](../crates/anno-cli/src/cli/commands/pipeline.rs) · [Cross-document identity](decisions/cross-document-coreference.md)

For English pronoun resolution, a fixed GAP panel of 128 Wikipedia excerpts yielded raw-text pair F1 of 0.094 for native mention-ranking, 0.745 for FCoref, and 0.784 for Maverick. Maverick reached 0.930 when supplied the annotated mention boundaries. That measures a different input contract. The native raw clusters contained both exact candidate spans for only 8 of 117 true GAP links. In 65 cases the exact name was missing at proposal. In another 44 both endpoints survived proposal and deduplication but at least one was absent from the emitted clusters. Maverick’s supplied-span LitBank arm accepted nine of ten works because a supplied boundary split a token in the tenth. Cluster-level CoNLL scores and document refusals are reported separately. Both panels identify the weights, mention source, exact matching rules and singleton policy. [Measurements and replay](eval/coref-panels.md) · [Native loss trace](eval/native-gap-loss.md)

On another 128 GAP excerpts from disjoint articles, FCoref raw text linked 68 of 118 true pairs with three false links. Native raw text linked seven with two false links. Supplying the three spans to native mention-ranking made all 118 true pairs visible, but its entity-only entry point linked just one. For callers who already have mentions and contextual text, evaluate the supplied-span interface directly. [Independent GAP replay](eval/gap-decision-audit.md)

### How do I verify what produced an answer?

For `extract --extract-relations`, a backend without a relation interface can produce `relation-heuristic` edges over its extracted mentions. GLiNER multitask can score entities with its own weights and use GLiREL or heuristics for relations. TPLinker can return `tplinker-heuristic` when its graph is absent. Check the producer on the **edge**, the actual checkpoint and the prompt. In the full-domain CrossRE AI comparison, the trained GLiNER-Relex export scored 24.1% strict F1 with the training inventory and 56.3% with labels drawn from each test document’s gold relations. The latter assumes knowledge of the answer vocabulary. The TPLinker heuristic emitted no edges with either inventory on that domain. [CLI routing](../crates/anno-cli/src/cli/commands/extract.rs) · [Multitask scoring](../crates/anno/src/backends/gliner_multitask/mod.rs) · [CrossRE results](eval/crossre-full-prompt-comparison.md)

A source map locates the original bytes that support an analyzed span. This offline example extracts an email at analyzed character range `6..21` and resolves it to original HTML bytes `12..27`.

```sh
printf '<p>Email <b>ada@example.org</b>.</p>\n' > page.html
ANNO_NO_DOWNLOADS=1 anno extract --model pattern --file page.html \
  --html-source-spans --clean --normalize --format grounded > doc.json
anno query source-span doc.json 6 21 --original page.html
```

A normalized character can have several disjoint original byte ranges. Transformed, approximate, synthetic and unmapped fragments carry their own kinds. An original digest identifies a file even when a PDF or HTML extraction path cannot locate an individual span in it. Keep the model’s analyzed span, source edition and projected fragments together when showing evidence to a reader. [Source-span contract](CONTRACT.md) · [HTML workflow](EXTRACTION_WORKFLOWS.md#trace-a-signal-to-its-original-html)

For JointIE, inspect candidate and decoded edges separately. With a 384-word ceiling on the first 64 CrossRE AI test sentences, all 161 annotated edges fit in a scored window. The broad training vocabulary produced 22 exact annotated edge candidates and the decoder selected zero. Shorter windows produced one exact match in 27 predictions. For JEREX, inspect clusters before interpreting relation recall. On the first 32 DocRED joint test documents, 140 of 406 directed labeled relations matched exact cluster endpoints. Another 110 lost an endpoint at cluster construction even though every required mention survived. These are two distinct tasks, corpora and denominators. [Expanded stage counts and schema comparison](eval/joint-edge-loss.md)

For a reproducible comparison, use the tracked QA procedure with fixed example budgets and seeds. A replay against real weights establishes that a checkpoint runs on its fixtures. A labeled held-out comparison measures a task on its sampled domain. A synthetic decoder test can establish offset or edge conventions even when trained weights are unavailable. For example, TPLinkerPlus has a verified sparse-tag decoder. A compatible, licensed trained joint checkpoint and reference replay are still needed for a text-to-edge claim. The Grenander 2025 generative coreference and Du 2026 document-level joint models also lack verified end-to-end artifacts in Anno. [Artifact inventory](eval/outstanding-model-artifacts.md) · [QA procedure](../.agents/skills/anno-qa/SKILL.md) · [Joint extraction stage replay](eval/joint-stage-baseline.md)

## Model Families

### Neural: ONNX (feature `onnx`)

| Backend | Architecture | Zero-shot | Status | Default model |
|---------|--------------|-----------|--------|---------------|
| `gliner` (canonical) / `gliner_onnx` (low-level) | Shared text/label encoder with a span classifier | Yes | stable | `onnx-community/gliner_small-v2.1` |
| `gliner_multitask` | GLiNER v1 with task-conditioned label prompts (NER + classification + structure; Stepanov & Shtopko 2024) | Yes | beta | `onnx-community/gliner-multitask-large-v0.5` |
| `gliner2_fastino` | Fastino GLiNER2, eight-graph ONNX pipeline | Yes | experimental | `jugaadsrl/gliner2-multi-v1-onnx` at `4241d7c66b648e618c89c150bf4cf418d2f83159` |
| `nuner` | Label-conditioned word or span scoring | Yes | stable | `deepanwa/NuNerZero_onnx` (default export of `numind/NuNER_Zero`), or a prepared NuNER Zero-4k FP32 export for `nuner_4k` |
| `bert_onnx` | BERT sequence labeling | No | beta | `protectai/bert-base-NER-onnx` |
| `w2ner` | Word-word grids (nested) | No | beta | `ljynlp/w2ner-bert-base` |
| `tplinker` | Handshaking tagging (joint entity+relation) | No | beta | Local three-head checkpoint; runtime uses a heuristic when no ONNX graph is present |
| `glirel` | DeBERTa encoder + scoring head (relations) | Yes | beta | `jackboyla/glirel-large-v0` |
| `gliner_poly` | Poly-encoder with label attention fusion | Yes | WIP | `knowledgator/gliner-bi-large-v1.0` (also: `gliner-bi-small-v1.0`, `modern-gliner-bi-large-v1.0`, `modern-gliner-bi-base-v1.0`; the `gliner-poly-*-v1.0` repos are model cards only with no weights) |
| `gliner_pii` | GLiNER PII Edge (60+ PII categories) | Yes | beta | `knowledgator/gliner-pii-edge-v1.0` |
| `gliner_relex` | Span NER and directed pair scoring from a local joint export | Yes | beta | `knowledgator/gliner-relex-large-v1.0` |
| `deberta_v3` | DeBERTa-v3 NER (local export) | No | WIP | -- |
| `albert` | ALBERT NER (local export) | No | WIP | -- |

In the Rust backend factory, `gliner` is a smart-default alias. Under `--features onnx` it
resolves to `anno::GLiNEROnnx`; under `--features candle` only (no
`onnx`) it falls back to `anno::GLiNERCandle`. Pick `gliner` when you
want GLiNER and don't mind which backend serves it. Pick `gliner_onnx`
or `gliner_candle` explicitly to select a runtime. The extraction CLI exposes
`--model gliner` with `onnx`. [CLI selectors](../crates/anno-cli/src/cli/parser.rs)

#### Selecting NER checkpoints

`anno extract --model bert-onnx` accepts `BERT_ONNX_MODEL_PATH` as a local
ONNX export directory or a Hub model ID. For fixed CoNLL labels, available
exports include `protectai/bert-large-NER-onnx` and
`onnx-community/modernbert-ner-conll2003-ONNX`. For a reproducible run,
pass a downloaded revision's directory. JSON and JSONL extraction provenance
records the explicit path or Hub ID in `checkpoint`; file contents can
change independently of that selector. The loader reads `tokenizer.json`
and the BIO label mapping in `config.json`.

`anno extract --model nuner` accepts `NUNER_MODEL_PATH` with either input
form, including when `--extract-types` supplies the labels. NuNER Zero
merges adjacent width-one predictions of the same type. The export
`onnx-community/NuNER_Zero-span` uses a trained 12-word span head and
scores multiword candidates directly. Include `gliner_config.json` with
local exports so the loader constructs the trained grid width.

```sh
NUNER_MODEL_PATH=onnx-community/NuNER_Zero-span anno extract --model nuner \
    --extract-types person,location 'Marie Curie worked in Paris.'
BERT_ONNX_MODEL_PATH=onnx-community/modernbert-ner-conll2003-ONNX \
    anno extract --model bert-onnx 'Marie Curie worked in Paris.'
```

NuNER Zero-4k uses Longformer. In the standard GLiNER 0.2.30 export the
first query projection matched PyTorch, but its first sliding attention window
already differed. The TorchScript ONNX conversion flattened the projected heads
before applying their original storage strides. Anno's exporter gathers query,
key and value windows by logical token position and verifies every FP32 logit
against unmodified PyTorch before installing the graph. Inputs of 14, 20,
512, 895 and 4096 tokens matched every threshold decision, with a maximum
absolute logit difference of 0.000299 on CPU. The source checkpoint revision is
`7a7cd8d65af2572c297054dbaf8f25c0d46da55d`.

```sh
uv run scripts/export_nuner_to_onnx.py --model numind/NuNER_Zero-4k --output /tmp/nuner-4k
NUNER_MODEL_PATH=/tmp/nuner-4k anno extract --model nuner \
    --extract-types person,location 'Marie Curie worked in Paris.'
uv run scripts/export_nuner_to_onnx.py --model numind/NuNER_Zero-4k --output /tmp/nuner-4k --verify-only
```

Dynamic unsigned INT8 reduced the graph from 1.52 GB to 407 MB with ONNX
Runtime 1.32.0 on CPU. At threshold 0.5 it retained all seven positive FP32
word-label decisions across the five fixtures and added one. On the 4096-token
input, the score for “Paris” as a location went from 0.425 to 0.519. The largest
absolute probability change was 0.108. The comparison measures agreement with
FP32. These fixtures have no gold accuracy labels. The exporter currently
releases FP32. [Per-case counts and artifact hashes](eval/nuner-4k-quantization.json)

Run the comparison against a verified FP32 export.

```sh
uv run scripts/qa/check_nuner_4k_quantization.py /tmp/nuner-4k/model.onnx /tmp/nuner-4k-u8.onnx --quantize
```

In the evaluation backend factory, select `nuner_4k` with
`NUNER_4K_MODEL_PATH=/tmp/nuner-4k`. The pinned PyTorch
adapter emits half-open character offsets and merges adjacent same-label
words according to the checkpoint convention. Merged predictions omit
confidence because the checkpoint only scores individual words.

```sh
uv run scripts/nuner_4k_import.py letter.txt person location > mentions.jsonl
anno dataset import --format jsonl --input mentions.jsonl --source-text letter.txt --output imported.jsonl
```

For a compact two-encoder alternative, run the following with the `onnx` feature:

```sh
uv run scripts/export_gliner_poly_onnx.py --model knowledgator/gliner-bi-edge-v2.0 \
    --strategy manual --output /tmp/gliner-bi-edge
GLINER_POLY_MODEL_PATH=/tmp/gliner-bi-edge anno extract --model gliner-poly \
    --extract-types person,location 'Marie Curie worked in Paris.'
```

#### Span-width exports (`max_width`)

`gliner`, `gliner_poly`, and `gliner_multitask` score a fixed grid of word
spans per position. The grid width is an **export property**, not a runtime
knob: the decoder reads it from the output tensor (`[1, words, max_width,
classes]`, or derives it from `[1, words * max_width, classes]`). An export
built with `max_width = 1` scores only single-word spans. Returning a multiword
mention from such scores requires a merging rule, as in NuNER Zero.

For parity with a checkpoint trained at width 12, keep all 12 candidates per word position in the export. The default
`onnx-community/gliner_small-v2.1` export satisfies this (its
`gliner_config.json` records `max_width: 12`). Check any other model
directory before benchmarking:

```sh
python3 - <<'PY'
import json, pathlib, sys
cfg = json.loads(pathlib.Path("<model-dir>/gliner_config.json").read_text())
width = cfg.get("max_width")
sys.exit(0 if isinstance(width, int) and width >= 12 else f"max_width={width} < 12")
PY
```

To regenerate a local poly/bi-encoder export at the required width:

```sh
uv run scripts/export_gliner_poly_onnx.py \
  --model knowledgator/gliner-bi-large-v2.0 \
  --output ~/.cache/anno/models/gliner-poly \
  --strategy manual --max-width 12 --quantize
```

`--strategy manual` is required for a specific width: the GLiNER library's
`export_to_onnx` path takes no width override. The script now rejects an
export whose `gliner_config.json` records a width below the requested
`--max-width`, and the decoder logs a warning when it sees a one-word span
grid. A narrow graph cannot be widened at load time; rebuild the export.
`gliner2_fastino` is separate: its eight-graph pipeline bakes
`MAX_WIDTH = 8` into the export (see `gliner2_fastino/pipeline.rs`), and its
decoder is consistent with that value.

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
The DirectML registration fallback workflow loads a pinned Windows CPU runtime,
checks 81 known logits from a synthetic ONNX fixture, then requests DirectML
through the same Anno API. The job fails if that request silently returns a CPU
session. Each run retains its runtime build, file hashes and registration result.
For a Windows host with a DX12 GPU, point `ORT_DYLIB_PATH` at the DLL from
[ONNX Runtime DirectML 1.24.4](https://pypi.org/project/onnxruntime-directml/1.24.4/#files)
and use a local BERT NER export containing `model.onnx`,
`tokenizer.json`, and `config.json`.

```powershell
cargo run --locked --release -p anno --no-default-features --features onnx-directml,ort/load-dynamic --example onnx_bert_provider_probe -- C:\models\bert C:\receipts\directml
```

The probe uses Anno’s session factory for CPU and DirectML. It saves both raw
profiles and a receipt with the artifact hashes, node assignments, argmax
agreement and maximum absolute logit difference. A passing receipt requires
at least one DirectML node and agreement within `1e-4` for the fixed example.
Record the DX12 adapter, driver, Anno commit and command alongside the receipt
to identify the hardware that executed the graph. The [hosted CPU control](../.github/workflows/directml-fallback.yml)
also records runner adapter inventory and verifies that profiling captured inference.

On Windows with Cargo 1.100, set `__CARGO_TEMPORARY_BUILD_DIR_NEW_LAYOUT_OPT_OUT=1`
for the DirectML build. The pinned `ort-sys` 2.0.0-rc.12 copies `DirectML.dll`
using Cargo’s previous build directory layout. See [upstream issue #619](https://github.com/pykeio/ort/issues/619).

The `onnx-rocm` feature is a legacy binding, not a supported AMD runtime path
with the current dependency: `ort` targets ONNX Runtime 1.24, while the
[ROCm provider was removed in 1.23](https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html).
The selected Linux x86-64 stack for AMD hardware testing is ONNX Runtime
1.29.0, AMD's `onnxruntime-ep-migraphx==1.0.0+rocm10.2.0` plugin and
`migraphx-libs==2.19.0+rocm10.2.0` with ROCm 10.2. Anno's pinned Rust
binding, `ort` 2.0.0-rc.12, loaded that runtime, discovered the CPU device,
selected it with `SessionBuilder::with_devices` and returned all 81 expected
fixture logits. AMD requires separate EP registration. Its plugin exports
`CreateEpFactories` and depends on `libmigraphx_c.so.3` and `libamdhip64.so.7`.
On a supported host, register `libmigraphx-ep.so` with
`Environment::register_ep_library("MIGraphXExecutionProvider", path)`, then
select a discovered GPU with that EP in `SessionBuilder::with_devices`.
This is the candidate for device testing. Anno’s public provider enum still
uses the legacy ROCm EP and GPU placement remains open.
[AMD installation and supported devices](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/onnxruntime.html) · [Pinned Rust registration API](https://github.com/pykeio/ort/blob/v2.0.0-rc.12/src/environment.rs#L179) · [Package digests and hardware acceptance](https://github.com/arclabs561/anno/issues/19)

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

### Fastino GLiNER2.5

With the `gliner2-fastino` feature, `anno extract --model gliner25` loads the
local FP32 multilingual boundary export from `GLINER25_MODEL_DIR`. The
directory may contain `fp32_25/` or be that directory. Keep ONNX external
data files beside their graphs. The [reference export](https://huggingface.co/jugaadsrl/gliner2.5-multi-v1-onnx/tree/3de2c11f55853b7919f985a59bfc8df27b5ec991/fp32_25)
uses first-token pooling, per-label abstention, and flat weighted interval
selection. The schema prompt and text share the 512-token encoder window.
Oversized inputs return an error.

```bash
GLINER25_MODEL_DIR=/models/gliner25 anno extract --model gliner25 \
  --text 'Éva works for Acme.' --extract-types person,organization
```

For a typed relation graph, `anno joint` uses the upstream Python
`gliner2[local]==2.2.0` runtime with a local PyTorch boundary checkpoint.
The schema uses the upstream JointSchema JSON format, including relation
endpoint types, optional cardinality limits, symmetry, inverses and acyclicity.

```json
{"entities":["person","organization"],"relations":{"works_for":{"head":"person","tail":"organization"}}}
```

```bash
anno joint --python .venv/bin/python --checkpoint /models/gliner2.5-small-v1 \
  --schema schema.json --text 'Éva works for Acme.'
```

Omit `--text` to stream `{"doc_id":"…","text":"…"}` records on stdin. Each
output preserves the result-local entity IDs, character offsets, directed
edges, feasibility and any upstream derived-edge or rescued-mention flags.
For a long document, the worker scores overlapping word windows sized to fit
the encoder with the schema prompt. `--window-words` sets a ceiling (default
384) and `--overlap-words` sets the desired overlap (default up to 64).
`scored_windows` records the actual half-open character ranges. Each candidate
keeps its source position and label. For identical mention or typed edge
candidates in overlapping windows, the higher logit wins. One upstream beam
decode applies endpoint cardinality, symmetry, inverses and acyclicity across
the complete candidate graph. Its default candidate budgets grow with the
number of windows. Explicit schema constraints apply across the document.

An edge can be selected only from a scored window or derived under schema
rules from such an edge. Two endpoints that never share a window have no
direct model score. Endpoints identify mentions by occurrence. A coreference
resolver can subsequently link those mentions into tracks. With the Python
provider installed, run `python crates/anno-cli/tests/test_gliner25_window_candidates.py`
to exercise cross-window graph constraints without downloading a checkpoint.
[Upstream JointIE schema and decoder](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/15-joint_ie.md)

### Neural: Fastino GLiNER2 (feature `gliner2-fastino`)

`GLiNER2Fastino` is available behind the `gliner2-fastino` feature. Its
optional Candle implementation is behind `gliner2-fastino-candle`; see
[the feature-gating contract](CONTRACT.md) for the supported feature surface.
The ONNX evaluator uses the pinned `jugaadsrl/gliner2-multi-v1-onnx`
snapshot `4241d7c66b648e618c89c150bf4cf418d2f83159`; it requires about
1.25 GB for the selected `fp32_v2` files on first use.

Use this ONNX backend for CPU inference with the multilingual GLiNER2
checkpoint. Candle 0.11.1's CPU kernel takes a mutable slice of
uninitialized values when DeBERTa broadcasts the scalar attention-mask
fill. The Candle loader rejects CPU devices before loading weights. Enable
`gliner2-fastino-candle-cuda` or `gliner2-fastino-candle-metal` to load
the Candle backend on an accelerator. The ONNX export runs the attention operation in ONNX Runtime. For
a LoRA adapter, this route requires an ONNX export with the adapter merged.
[Dependency review](https://github.com/arclabs561/anno/pull/168)

The optional GLiNER2.5 Multi Python importer uses Fastino's boundary decoder. It
extracts caller-supplied labels from one UTF-8 document and writes character
spans for `anno dataset import --format jsonl`. The example defaults to a
confidence threshold of 0.50 and accepts `--threshold`.
[Import example](../scripts/gliner25_import.py) · [Pinned checkpoint](https://huggingface.co/fastino/gliner2.5-multi-v1/tree/7b005274e82da37dcf1f5fd9d6a2ef413c6a004c)

Calibrate each checkpoint on separate documents using the intended labels and
the number of false suggestions the application can absorb. On 119 held-out
Dutch CoNLL-2002 articles, the two upstream Python checkpoints scored about
0.42 exact-span F1 at 0.50. At thresholds chosen for development F1, the older
Multi model (0.95) returned 203 correct and 62 extra mentions. GLiNER2.5
(0.90) returned 204 correct and 201 extra. On a separate English WikiGold
panel, F1 moved from 0.723 to 0.741, with a paired 95% interval for the
difference of −0.040 to +0.073. A target-domain comparison at an agreed
recall floor or suggestion budget gives a firmer basis for choosing a model.
[Thresholds, pinned datasets and per-label results](https://github.com/arclabs561/anno/pull/219)

### Features not validated in CI

CI compile-checks the hardware-gated features a GitHub-hosted runner can build:
`onnx-coreml`, `gliner2-fastino-coreml`, `gliner2-fastino-candle-metal`,
`onnx-cuda`, `onnx-rocm`, and `gliner2-fastino-cuda` (the `test-metal` job in
`.github/workflows/ci.yml`). They are checked on macOS because all six were
verified to compile there when the checks were added. A green check means the
gated code and its dependency features build; it does not mean a provider
registers, places nodes on a GPU, or produces CPU-matching logits. `onnx-rocm`
deserves a second caveat: see the legacy-binding note above.

Unvalidated features: not compiled or exercised by CI at all.

| Feature | Why CI does not validate it |
|---------|-----------------------------|
| `gliner2-fastino-candle-cuda` | `cudarc`'s build script runs `nvcc --version`, so the build requires a CUDA toolkit. Hosted runners ship none; installing one would add a multi-GB toolchain to the PR path. A local `cargo check --features gliner2-fastino-candle-cuda` fails the same way. |

Do not read a compile check, a declared feature, or a listed backend as runtime
coverage. Provider behaviour needs evidence from the host you intend to run on;
use `onnx_cuda_smoke`, `onnx_coreml_smoke`, and the provider probes for that.

### Neural: LLM (feature `llm`)

| Backend | Architecture | Zero-shot | Status | Default model |
|---------|--------------|-----------|--------|---------------|
| `universal_ner` | LLM-backed zero-shot (OpenRouter/Anthropic/Groq/Gemini/Ollama) | Yes | beta | `google/gemini-2.5-flash-lite` |

Select a chat-completions endpoint for the CLI or evaluation with
`ANNO_LLM_ENDPOINT` (the full request URL) and `ANNO_LLM_MODEL` (the model ID
accepted by that endpoint). Without a model ID Anno uses the native provider
default, or the literal model ID `default` for a custom URL. The optional `ANNO_LLM_API_KEY` is scoped to this
explicit route. With a custom endpoint, leaving the key unset sends an
unauthenticated request. An explicitly configured Rust `LlmConfig` takes
precedence over these three variables. Without an endpoint override Anno uses
its existing provider discovery.

#### Test UniversalNER with Codex CLI

Run `just test-llm-codex` with Python 3, Cargo and Codex CLI on your PATH.
The check runs `anno extract` on “Ada left.” It explicitly selects a temporary
loopback endpoint while a second listener stands in for ambient Ollama. The
selected server checks its model and bearer credential and forwards the prompt to
`codex exec`. A second loopback listener supplies a deterministic Responses
fixture. The check requires the person span at character offsets 0..3 and one
request at each hop, with zero requests to the ambient listener. It exercises the CLI protocol and the Anno parser.

Use `just test-llm-codex-live` to send that synthetic prompt to the model
configured for Codex. This uses the CLI's own authentication and can consume
provider quota. Both modes start in a temporary directory and close the listeners
when the test ends. With a prebuilt `anno` binary that includes the `llm`
feature, run `python3 scripts/verification/check_codex_llm.py --anno /path/to/anno`.
The deterministic path was exercised with Codex CLI 0.172.0-alpha.2.

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

#### Maverick coreference

`uv run scripts/download_maverick_weights.py --model ontonotes` prepares the pinned MES checkpoint in the local Anno cache and prints its safetensors path. The converter verifies the upstream SHA-256 and reads the model tensors with Candle. Upstream weights use CC BY-NC-SA 4.0. Pass the prepared file to `anno pipeline --model heuristic --coref --maverick-model CHECKPOINT --format json "Ada arrived. She spoke."`. Anno loads the fixed MES architecture with strict parameter validation. Once prepared in the default Anno cache, the official OntoNotes, LitBank and PreCo Hub IDs also work offline. Pass the printed file path when choosing a custom output directory. The worker checks the complete prepared-file digest whenever you select an official model ID. Original Lightning training archives require preparation because they include serialized training configuration.

Set `MAVERICK_PYTHON` to the Python executable for the persistent CPU worker. The evaluator selects backend `maverick` with `MAVERICK_MODEL_PATH` set to the prepared file. The preparation command also caches the pinned DeBERTa encoder and tokenizer for offline use. See the [CPU environment](../scripts/qa/maverick/requirements-cpu-py312.txt), [model and encoder hashes](../crates/anno/src/backends/coref/maverick_models.json) and [upstream package license](https://github.com/SapienzaNLP/maverick-coref#license).

The Rust `MaverickCoref::resolve_mentions` API accepts half-open Unicode character spans for clustering-only runs. The worker uses English spaCy tokenization and a sentencizer, sends word-tokenized sentences to the model, and maps its inclusive token predictions back to source character ranges. A supplied span that splits a token raises an error. The pipeline publishes multi-mention tracks. Callers can request singleton predictions through the Rust constructor when using singleton-trained LitBank or PreCo weights.

Maverick allows 15 minutes for model startup and five minutes for a request, including time queued behind another caller. Rust callers can select different durations with `MaverickCoref::with_timeouts`. If worker I/O times out, the pipe breaks, or the reply is malformed, the adapter returns an error and reaps that worker. The next document starts a fresh worker. A caller that times out in the queue leaves the active request running. A model-reported input error also leaves the worker available.

The [Maverick workflow](../.github/workflows/maverick-coref.yml) exercises worker hangs and recovery on pull requests. Manually dispatch it to replay the real-weight test with a verified encoder and checkpoint. To run locally on Linux x86-64 with Python 3.12, use the workflow’s CPU installation commands and run `python scripts/qa/maverick/verify.py /path/to/workdir`. The script creates an isolated model cache, checks each pinned artifact and runs inference offline. It checks the worker’s token IDs and source offsets for ligatures, nonbreaking hyphens and fullwidth digits. Transformers 4.57.6 preserves the recorded encodings. Versions 5.10.1 and 5.24.1 changed them.

For T5, `T5Coref::resolve_neural` preserves the ONNX model’s empty predictions and inference errors for evaluation; `resolve` retains its legacy heuristic fallback. [Grenander et al.’s incremental seq2seq method](https://arxiv.org/abs/2510.14504) uses compressed entity histories. An Anno-compatible trained checkpoint and decoding contract for that method have not been verified.

#### Architecture-specific gotchas

A few backends need more than a stock `optimum-cli export onnx`:

- **DeBERTa-v3**: pin `transformers==4.47.0` to avoid a fast-tokenizer
  bug; copy SentencePiece artifacts from the HF cache because the
  fast-tokenizer conversion is broken in newer versions.
- **GLiNER-RelEx**: scatter-based subword pooling, so `torch.onnx.
  export` needs a real tokenized input (random dummy inputs cause
  index-out-of-bounds during JIT tracing). Run `uv run scripts/export_gliner_relex_onnx.py`
  to write `model.onnx`, `gliner_config.json` and `tokenizer.json` to
  `$HF_HOME/hub/models--knowledgator--gliner-relex-large-v1.0/onnx/`
  (`HF_HOME` defaults to `~/.cache/huggingface`). The sampler offers this
  model for NER when these files are present. Pass `--joint` to export the
  trained relation head alongside the span encoder in the `onnx/joint/`
  directory. The exporter compares directed edges and scores with upstream
  Python on multiple texts and schemas before publishing the graph.
  The CPU parity run used checkpoint revision
  `4aedc9226a5ac9e2f6b5ea3e91c1ee577c88a290`; pass it with `--revision` to reproduce the export.
  Use `anno extract --model gliner-relex --extract-relations --extract-types "person,organization" --relation-types "works for,founded" --text "Marie Curie worked at the University of Paris."`.
  After export, run `python scripts/qa/check_gliner_relex_cli.py --anno target/debug/anno --receipt result.json`.
  The replay checks five upstream predictions, including repeated surface names and empty
  results. It reloads each grounded export and records the checkpoint, artifact generation,
  native binary, ONNX Runtime and score differences in the receipt.
- **GLiNER bi-encoder** (`gliner_poly`): produces **two** ONNX graphs --
  a text encoder and a separate label encoder for the BGE
  sentence-transformer label embeddings.
- **F-coref**: bypass `fastcoref`'s spaCy-dependent high-level API and
  load via `transformers` directly; exports the DistilRoBERTa encoder
  as ONNX and the mention-ranking scorer as safetensors.
- **W2NER**: the simplified export removes the LSTM head and substitutes
  attention pooling. The runtime expects this simplified form.
- The TPLinker authors originally linked evaluation checkpoints. In the repository’s issue tracker they subsequently confirmed that the saved files were lost. The original download folder returns HTTP 404. Anno’s
  exporter checks the tensor names and shapes of its handshaking kernel and
  three trained heads. The selected dataset must use the checkpoint’s relation
  ordering. TPLinkerPlus uses a single classifier and its own tag-to-index
  mapping. Feed upstream sparse predictions to the Rust decoder with
  `cargo run -p anno --no-default-features --example decode_tplinker_plus < prediction.json`.
  Each input supplies `text`, `tok2char_span` (Unicode code-point spans),
  `entity_type2id`, `rel2id`, and `spots` from upstream's `get_spots_fr_shaking_tag`.
  The output preserves window-local entity types, spans and directed predicates. Fixture parity
  covers self-relations, nested spans, ambiguous same-span types, reverse links, and empty
  output. This verifies tag decoding. A trained TPLinkerPlus model replay remains open.
  [Checkpoint availability](https://github.com/131250208/TPlinker-joint-extraction/issues/89#issuecomment-3695039976)
  · [TPLinkerPlus implementation](https://github.com/131250208/TPlinker-joint-extraction/blob/master/tplinker_plus/tplinker_plus.py)

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
| `bert_onnx` | yes | `--` | hf-hub cache | `BERT_ONNX_MODEL_PATH` (CLI) |
| `gliner_candle` / `candle_ner` | yes | `--` | hf-hub cache | `--` |
| `gliner_pii` | yes | `--` | hf-hub cache | `--` |
| `gliner_relex` | local ONNX export; relations require `--joint` | `export_gliner_relex_onnx.py` | hf-hub cache | `--` |
| `nuner` | yes (script needed only for unsupported variants) | `export_nuner_to_onnx.py` (uses GLiNER’s shared-encoder exporter for NuNER Zero variants) | hf-hub cache | `NUNER_MODEL_PATH` (CLI) |
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
  are model cards only with no weights. The default checkpoint requires an ONNX
  export. Use `export_gliner_poly_onnx.py --strategy manual` for the split graph
  format. The exporter writes `anno_poly_export.json`, which declares whether the
  main graph uses embedded labels or token IDs and lists external weight files.
  Automatic evaluation checks this inventory, both tokenizers and the declared files
  in `GLINER_POLY_MODEL_PATH`, the default cache, or the model's local Hugging Face
  snapshot. For an older self-contained export, run the same script with
  `--inventory-only --output DIRECTORY`. Explicit loading supports older exports
  without an inventory. The native ONNX fixture checks label-sensitive
  decoding and source offsets. Pretrained accuracy requires separate measurement.
- `tplinker` exports with random weights if no `--checkpoint` is provided;
  the runtime falls back to a heuristic mode when no ONNX is present.
- `gliner_onnx` runs `export_gliner_poly_onnx.py` automatically on first
  load if no ONNX file is in the cache (no manual step required).

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

Several ML backends download their default weights from Hugging Face on first use. NuNER 4k and GLiNER Poly require prepared ONNX exports. See the tables above for model IDs.

Override with model-specific flags or environment variables.

## ONNX export scripts

Some models only distribute PyTorch weights. Export scripts in `scripts/` convert them to ONNX for use with anno's inference backends. All scripts use PEP 723 inline metadata and run with `uv run`.

| Script | Target model | Notes |
|--------|-------------|-------|
| `export_gliner_poly_onnx.py` | GLiNER bi-encoder (v1/v2) | Dual strategy: library export, then manual fallback. Produces `model.onnx` + `label_encoder.onnx` |
| `export_nuner_to_onnx.py` | NuNER Zero, Zero-span and Zero-4k | Uses Optimum for fixed labels and GLiNER for label-conditioned exports. The Zero-4k path verifies long-window FP32 scores. See the NuNER section for evaluator discovery and the measured INT8 comparison. |
| `export_deberta_ner_to_onnx.py` | DeBERTa-v3 NER | Standard token classifier export |
| `export_biomedical_ner_to_onnx.py` | d4data/biomedical-ner-all | Uses Optimum; optional INT8 quantization |
| `export_w2ner_to_onnx.py` | W2NER | Simplified architecture (fixed-length inputs for ONNX compat) |
| `export_tplinker_onnx.py` | TPLinker | Joint entity-relation extraction |
| `export_glirel_onnx.py` | GLiREL | Relation extraction; falls back to PyTorch weights if ONNX export fails |
| `export_gliner_relex_onnx.py` | GLiNER-RelEx | Exports entity logits or a trained joint graph with `--joint` |
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
