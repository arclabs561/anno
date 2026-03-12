# anno

[![crates.io](https://img.shields.io/crates/v/anno.svg)](https://crates.io/crates/anno)
[![Documentation](https://docs.rs/anno/badge.svg)](https://docs.rs/anno)
[![CI](https://github.com/arclabs561/anno/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/anno/actions/workflows/ci.yml)

Information extraction for unstructured text: named entity recognition (NER), coreference resolution, entity linking, relation extraction, and zero-shot entity types.

Dual-licensed under MIT or Apache-2.0. MSRV: 1.85.

## Library

The crate is published as **`anno`** on crates.io.

```toml
[dependencies]
anno = "0.3.9"
```

The default feature (`onnx`) pulls in `ort` (ONNX Runtime C++ bindings). For builds without a C++ runtime, use `default-features = false`.

```rust
use anno::{Model, StackedNER};

let m = StackedNER::default();
let ents = m.extract_entities("Sophie Wilson designed the ARM processor.", None)?;
assert!(!ents.is_empty());
# Ok::<(), anno::Error>(())
```

`StackedNER::default()` selects the best available backend at runtime: GLiNER (if `onnx` enabled and model cached), falling back to heuristic + pattern extraction. Set `ANNO_NO_DOWNLOADS=1` or `HF_HUB_OFFLINE=1` to force cached-only behavior.

Zero-shot custom types via `DynamicLabels` (GLiNER, GLiNER2, NuNER):

```rust
use anno::{DynamicLabels, GLiNEROnnx};

let m = GLiNEROnnx::new("onnx-community/gliner_small-v2.1")?;
let ents = m.extract_with_labels(
    "Aspirin treats headaches.",
    &["drug", "symptom"],
    None,
)?;
# Ok::<(), anno::Error>(())
```

### Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `onnx` | Yes | ONNX Runtime backends (GLiNER, NuNER, BERT, W2NER, FCoref) via `ort` |
| `candle` | No | Pure-Rust Candle backends (no C++ runtime needed) |
| `analysis` | No | Evaluation helpers, coref metrics, RAG rewriting |
| `eval` | No | Superset of `analysis`: adds network support for dataset fetching |
| `graph` | No | Keywords, summarization, salience (TextRank/LexRank/YAKE) |
| `schema` | No | JSON Schema generation for output types via `schemars` |
| `discourse` | No | Centering theory, abstract anaphora, shell nouns |

## Tasks

**Named entity recognition.** Given input text, identify spans `(start, end, type, confidence)` where each span denotes a named entity [1, 2]. Entity types follow standard taxonomies (PER, ORG, LOC, MISC for CoNLL-style [2]) or caller-defined labels for zero-shot extraction. Offsets are **character offsets** (Unicode scalar values), not byte offsets; see [CONTRACT.md](docs/CONTRACT.md).

**Coreference resolution.** Identify mention spans and group them into clusters, where each cluster tracks a consistent discourse referent within the document [3, 4]. Coreferring mentions span proper names, definite descriptions, and pronouns: "Sophie Wilson", "the designer", and "She" form one cluster. Multiple backends: rule-based sieves (`SimpleCorefResolver`), neural (`FCoref`, 78.5 F1 on CoNLL-2012), and mention-ranking (`MentionRankingCoref`).

**Relation extraction.** Extract typed `(head, relation, tail)` triples from text. Available on `RelationCapable` backends (`gliner2`, `tplinker`); others fall back to co-occurrence edges for graph export.

**Structured pattern extraction.** Dates, monetary amounts, email addresses, URLs, phone numbers via deterministic regex grammars.

## Backends

All backends produce the same output type: variable-length spans with character offsets.

| Backend | Architecture | Labels | Zero-shot | Relations | Weights | Reference |
|---|---|---|---|---|---|---|
| `stacked` (default) | Selector/fallback | GLiNER > BERT > heuristic+pattern | -- | -- | HuggingFace (when ML enabled) | -- |
| `gliner` | Bi-encoder span classifier | Custom | Yes | -- | [gliner_small-v2.1](https://huggingface.co/onnx-community/gliner_small-v2.1) | Zaratiana et al. [5] |
| `gliner2` | Multi-task span classifier | Custom | Yes | Heuristic | [gliner-multitask-large-v0.5](https://huggingface.co/onnx-community/gliner-multitask-large-v0.5) | [11] |
| `nuner` | Token classifier (BIO) | Custom | Yes | -- | [NuNerZero_onnx](https://huggingface.co/deepanwa/NuNerZero_onnx) | Bogdanov et al. [6] |
| `bert-onnx` | Sequence labeling (BERT) | PER/ORG/LOC/MISC | No | -- | [bert-base-NER-onnx](https://huggingface.co/protectai/bert-base-NER-onnx) | Devlin et al. [8] |
| `pattern` | Regex grammars | DATE/MONEY/EMAIL/URL/PHONE/PERCENT | N/A | -- | None | -- |
| `tplinker` | Joint entity-relation (heuristic) | Custom | -- | Heuristic | None | [10] |

ML backends are feature-gated (`onnx` or `candle`). Weights download from HuggingFace on first use. See [BACKENDS.md](docs/BACKENDS.md) for the full list (including experimental backends) and feature-flag details.

## CLI

```sh
cargo install --git https://github.com/arclabs561/anno --package anno-cli --bin anno --features "onnx eval"
```

### Examples

```sh
anno extract --text "Lynn Conway worked at IBM and Xerox PARC in California."
```

```text
PER:1 "Lynn Conway"
ORG:2 "IBM" "Xerox PARC"
LOC:1 "California"
```

JSON output (`--format json`):

```sh
anno extract --model pattern --format json --text "Contact jobs@acme.com by March 15 for the \$50K role."
```

Zero-shot custom entity types (via GLiNER [5]):

```sh
anno extract --model gliner --extract-types "DRUG,SYMPTOM" \
  --text "Aspirin can treat headaches and reduce fever."
```

```text
drug:1 "Aspirin" symptom:2 "headaches" "fever"
```

Coreference:

```sh
anno debug --coref -t "Sophie Wilson designed the ARM processor. She revolutionized mobile computing."
```

```text
Coreference: "Sophie Wilson" → "She"
```

## Coreference

Three coref backends with different tradeoffs:

| Backend | Type | Quality | Speed | Weights |
|---------|------|---------|-------|---------|
| `SimpleCorefResolver` | Rule-based (6 sieves) | Low | Fast | None |
| `FCoref` | Neural (DistilRoBERTa) | 78.5 F1 | Medium | ONNX export |
| `MentionRankingCoref` | Mention-ranking | Medium | Medium | None |

**FCoref** (Otmazgin et al., AACL 2022) requires a one-time model export. The export script is in the repo at `scripts/export_fcoref.py` (not shipped in the crates.io package -- clone the repo to run it):

```sh
# From a repo clone:
uv run scripts/export_fcoref.py
```

```rust
let coref = FCoref::from_path("fcoref_onnx")?;
let clusters = coref.resolve("Marie Curie was born in Warsaw. She discovered radium.")?;
// clusters[0] = { mentions: ["Marie Curie", "She"], canonical: "Marie Curie" }
```

**RAG preprocessing** (`rag::resolve_for_rag()`, requires `analysis` feature): rewrites pronouns with their antecedents so document chunks remain self-contained after splitting. Supports English, French, Spanish, German. Includes pleonastic "it" filter and overlap guard.

## Downstream

Filter and pipe JSON output:

```sh
# All person entities, text only
anno extract --format json --file article.txt \
  | jq '[.[] | select(.entity_type == "PER") | .text]'

# Unique organizations, sorted
anno extract --format json --file article.txt \
  | jq '[.[] | select(.entity_type == "ORG") | .text] | unique | sort[]'
```

Batch a directory (parallel, cached):

```sh
anno batch --dir docs/ --parallel 4 --cache --output results/

# Stream stdin JSONL: {"id":"…","text":"…"} per line
cat corpus.jsonl | anno batch --stdin --parallel 4 --cache --output results/
```

### Knowledge Graph (RDF)

> The following export commands require `--features graph` at install time (add `graph` to the features list in the `cargo install` command above).

`anno export` emits standard **N-Triples** or **JSON-LD** -- loadable into any RDF store (Oxigraph, Jena, Blazegraph, etc.). `--base-uri` sets the IRI namespace:

```sh
# Own namespace (recommended for private corpora)
anno export --input docs/ --output /tmp/kg/ --format ntriples \
  --base-uri https://myproject.example.com/kg/

# Align to DBpedia (widely used LOD namespace)
anno export --input docs/ --output /tmp/kg/ --format ntriples \
  --base-uri https://dbpedia.org/resource/
```

Default (`urn:anno:`) produces stable URNs suitable for local use without a registered domain. The output is standard W3C N-Triples, so any SPARQL-capable store can query it:

```sh
# Example: load into a SPARQL store and query
cat /tmp/kg/*.nt | your-rdf-store load --format ntriples

your-rdf-store query 'SELECT ?label WHERE {
  ?e a <https://myproject.example.com/kg/vocab#PERType> ;
     <http://www.w3.org/2000/01/rdf-schema#label> ?label .
}'
```

**Semantic relation triples** — `RelationCapable` backends (`tplinker`, `gliner2`) produce typed `(head, relation, tail)` triples instead of co-occurrence edges. Use `--format graph-ntriples` (`--features graph`) for routing through the internal graph substrate:

```sh
anno export --input docs/ --output /tmp/kg/ \
  --format graph-ntriples --model gliner2 \
  --base-uri https://myproject.example.com/kg/
# → <entity/person/0_Lynn_Conway_0> <rel/works_for> <entity/org/1_IBM_13> .
```

### Property Graph CSV

Export node and edge tables as CSV for import into any property-graph database:

```sh
# Semantic edges (gliner2 or tplinker)
anno export --input docs/ --output /tmp/kg/ --format kuzu --model gliner2

# Co-occurrence edges (any other backend)
anno export --input docs/ --output /tmp/kg/ --format kuzu
```

Each file produces `{stem}-nodes.csv` + `{stem}-edges.csv` with columns:
- **Nodes**: `id`, `entity_type`, `text`, `start`, `end`, `source`
- **Edges**: `from`, `to`, `rel_type`, `confidence`

## Architecture

| Crate | Published | Purpose |
|---|---|---|
| `anno` (root) | Yes | Backends, `Model` trait, extraction pipeline, coref resolvers. |
| `anno-core` | No (yanked) | Data model (`Entity`, `Relation`, `Mention`, `CorefChain`, `Signal`, `Track`). Re-exported by `anno`. |
| `anno-graph` | No | Graph/KG export adapters (N-Triples, JSON-LD, CSV) |
| `anno-eval` | No | Evaluation harnesses, dataset loaders, matrix sampling |
| `anno-cli` | No | CLI binary |
| `anno-metrics` | No | Shared evaluation primitives (CoNLL F1, encoders) |

`anno-core` exists so that crates needing only data types (`Entity`, `Relation`, etc.) don't pull in ML dependencies. `anno-lib` depends on `anno-core` and re-exports all its types at the crate root.

Pipeline: Text -> Extract -> Coalesce -> structured output. See [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Evaluation

`anno-eval` provides dataset loading, backend-vs-dataset compatibility gating, and CoNLL-style [2] span-level evaluation (precision, recall, F1) with label mapping between backend and dataset taxonomies.

```sh
anno benchmark --model gliner --dataset conll2003
```

`anno sampler` (`--features eval`) provides a randomized matrix sampler with two modes: **triage** (worst-first) and **measure** (ML-only stable measurement).

### Sampler examples

```sh
# Deterministic offline decision-loop smoke example (also run in CI).
cargo run -p anno-eval --example muxer_decision_loop --features eval

# Run the CI sampler harness locally (uses ~/.anno_cache for history/cache).
just ci-matrix-local 42 ner
```

## Scope

Inference-time extraction. Training pipelines are out of scope -- use upstream frameworks (Hugging Face Transformers, Flair, etc.) and export ONNX weights for consumption. (The repo contains some experimental training code for box embeddings; this is research-only and not part of the public API.)

## Troubleshooting

**ONNX Runtime linking errors.** The `onnx` feature pulls in `ort`, which needs the ONNX Runtime C++ library. If linking fails, use `default-features = false` for a minimal build without native deps. If you need ONNX backends, ensure `ort` can download the prebuilt runtime -- check proxy/firewall settings and the `ORT_DYLIB_PATH` env var.

**Model download failures.** First use of ML backends downloads weights from HuggingFace to `~/.cache/huggingface/hub/`. Behind firewalls, pre-fetch models on a connected machine, then set `HF_HUB_OFFLINE=1` (or `ANNO_NO_DOWNLOADS=1`) to force cached-only mode.

**FCoref export script not found.** The export script (`scripts/export_fcoref.py`) is only present in the git repository, not in the crates.io package. Clone the repo to access it: `uv run scripts/export_fcoref.py`.

**"Feature not available" errors.** Most ML backends are feature-gated behind `onnx` or `candle`. If a backend constructor returns an error about a missing feature, check which features are enabled. The feature flags table above lists what each feature unlocks.

**Character offset mismatches.** All spans use character offsets (Unicode scalar values), not byte offsets. If offsets look wrong, ensure you are indexing by `chars()` count, not by byte position. See [CONTRACT.md](docs/CONTRACT.md).

## Documentation

- [QUICKSTART](docs/QUICKSTART.md)
- [CONTRACT](docs/CONTRACT.md) — offset semantics, scope, feature gating
- [BACKENDS](docs/BACKENDS.md) — backend selection, architecture, feature flags
- [ARCHITECTURE](docs/ARCHITECTURE.md) — crate layout, dependency flow
- [REFERENCES](docs/REFERENCES.md) — full bibliography (NER, coref, relation extraction, software)
- [API docs](https://docs.rs/anno)
- [Changelog](CHANGELOG.md)

## References

[1] Grishman & Sundheim, *COLING* 1996 (MUC-6 NER).
[2] Tjong Kim Sang & De Meulder, *CoNLL* 2003.
[3] Lee et al., *EMNLP* 2017 (end-to-end coref).
[4] Jurafsky & Martin, *SLP3* 2024.
[5] Zaratiana et al., *NAACL* 2024 (GLiNER).
[6] Bogdanov et al., 2024 (NuNER).
[7] Li et al., *AAAI* 2022 (W2NER).
[8] Devlin et al., *NAACL* 2019 (BERT).
[9] Lafferty et al., *ICML* 2001 (CRF).
[10] Wang et al., *COLING* 2020 (TPLinker).
[11] Zaratiana et al., 2025 (GLiNER2).
[12] Rabiner, *Proc. IEEE* 1989 (HMM).

Full list with links: [docs/REFERENCES.md](docs/REFERENCES.md). Citeable via [CITATION.cff](CITATION.cff).

## License

Dual-licensed under MIT or Apache-2.0.
