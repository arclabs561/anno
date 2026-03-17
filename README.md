# anno

[![crates.io](https://img.shields.io/crates/v/anno.svg)](https://crates.io/crates/anno)
[![Documentation](https://docs.rs/anno/badge.svg)](https://docs.rs/anno)
[![CI](https://github.com/arclabs561/anno/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/anno/actions/workflows/ci.yml)

Extract named entities, relations, coreference chains, and PII from unstructured text. Fixed entity types (PER/ORG/LOC/MISC) or zero-shot custom labels.

Dual-licensed under MIT or Apache-2.0. MSRV: 1.85.

## Quickstart

```toml
[dependencies]
anno = "0.3.9"
```

```rust
let entities = anno::extract("Sophie Wilson designed the ARM processor.")?;
for e in &entities {
    println!("{} [{}] ({},{}) {:.2}", e.text, e.entity_type, e.start(), e.end(), e.confidence);
}
// Sophie Wilson [PER] (0,13) 0.95
// ARM [ORG] (27,30) 0.90
# Ok::<(), anno::Error>(())
```

Filter results with `prelude`:

```rust
use anno::prelude::*;

let people: Vec<_> = entities.of_type(&EntityType::Person).collect();
let confident: Vec<_> = entities.above_confidence(0.8).collect();
```

For backend control, construct a model directly:

```rust
use anno::{Model, StackedNER};

let m = StackedNER::default();
let ents = m.extract_entities("Sophie Wilson designed the ARM processor.", None)?;
# Ok::<(), anno::Error>(())
```

`StackedNER::default()` selects the best available backend at runtime: BERT or NuNER (if `onnx` enabled and models cached), then GLiNER, falling back to heuristic + pattern extraction. Set `ANNO_NO_DOWNLOADS=1` or `HF_HUB_OFFLINE=1` to force cached-only behavior.

Zero-shot custom types via GLiNER:

```rust
use anno::GLiNEROnnx;

let m = GLiNEROnnx::new("onnx-community/gliner_small-v2.1")?;
let ents = m.extract("Aspirin treats headaches.", &["drug", "symptom"], 0.5)?;
for e in &ents {
    println!("{}: {}", e.entity_type, e.text);
}
// drug: Aspirin
// symptom: headaches
# Ok::<(), anno::Error>(())
```

### Custom backends

`AnyModel` wraps a closure into a `Model`, bypassing the sealed trait when you need to plug in an external NER system:

```rust
use anno::{AnyModel, Entity, EntityType, Language, Model, Result};

let model = AnyModel::new(
    "my-ner",
    "REST API wrapper",
    vec![EntityType::Person, EntityType::Organization],
    |text: &str, _lang: Option<Language>| -> Result<Vec<Entity>> {
        Ok(vec![]) // call your backend here
    },
);
let ents = model.extract_entities("test", None)?;
# Ok::<(), anno::Error>(())
```

## What it does

**Named entity recognition.** Spans `(start, end, type, confidence)` with character offsets (Unicode scalar values, not bytes). Fixed taxonomies (PER/ORG/LOC/MISC) or caller-defined labels for zero-shot extraction [1, 2].

**Coreference resolution.** Group mentions into clusters tracking the same referent. Rule-based sieves (`SimpleCorefResolver`), neural (`FCoref`, 78.5 F1 on CoNLL-2012 [3]), and mention-ranking (`MentionRankingCoref`).

**Structured patterns.** Dates, monetary amounts, emails, URLs, phone numbers via deterministic regex grammars.

**Relation extraction.** `(head, relation, tail)` triples via `RelationCapable` backends (`gliner2`, `tplinker`). Other backends produce co-occurrence edges for graph export.

**PII detection.** Classify NER entities as PII and scan for structured patterns (SSN, credit card, IBAN, email, phone). Redact or pseudonymize in one call:

```rust
use anno::{pii, Model, StackedNER};

let text = "John Smith's SSN is 123-45-6789.";
let m = StackedNER::default();
let ents = m.extract_entities(text, None)?;

let mut pii_ents: Vec<_> = ents.iter().filter_map(pii::classify_entity).collect();
pii_ents.extend(pii::scan_patterns(text));
let redacted = pii::redact(text, &pii_ents);
// "[REDACTED]'s SSN is [REDACTED]."
# Ok::<(), anno::Error>(())
```

**Export.** Brat standoff, CoNLL BIO tags, JSONL, N-Triples, JSON-LD, and graph CSV via pure functions in `anno::export`.

## Backends

| Backend | Zero-shot | Weights | Reference |
|---|---|---|---|
| `stacked` (default) | -- | HF (when ML enabled) | -- |
| `gliner` | Yes | [gliner_small-v2.1](https://huggingface.co/onnx-community/gliner_small-v2.1) | Zaratiana et al. [5] |
| `gliner2` | Yes | [gliner-multitask-large-v0.5](https://huggingface.co/onnx-community/gliner-multitask-large-v0.5) | [11] |
| `nuner` | Yes | [NuNerZero_onnx](https://huggingface.co/deepanwa/NuNerZero_onnx) | Bogdanov et al. [6] |
| `bert-onnx` | No | [bert-base-NER-onnx](https://huggingface.co/protectai/bert-base-NER-onnx) | Devlin et al. [8] |
| `pattern` | N/A | None | -- |
| `universal-ner` | Yes | None (LLM API) | -- |

Statistical baselines (`crf` [9], `hmm` [12], `bilstm-crf`) and structural backends (`w2ner` [7], `tplinker` [10]) are also available. See [BACKENDS.md](docs/BACKENDS.md) for the full list.

ML backends are feature-gated (`onnx` or `candle`). Weights download from HuggingFace on first use.

### Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `onnx` | Yes | ONNX Runtime backends via `ort` |
| `candle` | No | Pure-Rust backends (no C++ runtime) |
| `metal` | No | Metal GPU acceleration (enables `candle`) |
| `cuda` | No | CUDA GPU acceleration (enables `candle`) |
| `analysis` | No | Coref metrics, RAG rewriting |
| `schema` | No | JSON Schema for output types |
| `llm` | No | LLM-based extraction (OpenRouter, Anthropic, Groq, Gemini, Ollama) |
| `production` | No | `parking_lot` locks + `tracing` instrumentation |

## CLI

```sh
cargo install --git https://github.com/arclabs561/anno --package anno-cli --bin anno --features "onnx"
```

```sh
anno extract --text "Lynn Conway worked at IBM and Xerox PARC in California."
# PER:1 "Lynn Conway"
# ORG:2 "IBM" "Xerox PARC"
# LOC:1 "California"

anno extract --model gliner --extract-types "DRUG,SYMPTOM" \
  --text "Aspirin can treat headaches and reduce fever."
# drug:1 "Aspirin" symptom:2 "headaches" "fever"

anno debug --coref -t "Sophie Wilson designed the ARM. She revolutionized mobile computing."
# Coreference: "Sophie Wilson" -> "She"
```

JSON output with `--format json`. Batch processing with `anno batch`. Graph export (N-Triples, JSON-LD, CSV) with `anno export --features graph`.

## Coreference

| Backend | Type | Quality | Speed |
|---------|------|---------|-------|
| `SimpleCorefResolver` | Rule-based (9 sieves) | Low | Fast |
| `FCoref` | Neural (DistilRoBERTa) | 78.5 F1 [3] | Medium |
| `MentionRankingCoref` | Mention-ranking | Medium | Medium |

`FCoref` requires a one-time model export: `uv run scripts/export_fcoref.py` (from a repo clone).

RAG preprocessing (`rag::resolve_for_rag()`, `analysis` feature): rewrites pronouns for self-contained chunks after splitting.

## Scope

Inference-time extraction. Training pipelines are out of scope -- use upstream frameworks and export ONNX weights.

## Troubleshooting

- **ONNX linking errors**: use `default-features = false` for builds without C++, or check `ORT_DYLIB_PATH`.
- **Model downloads**: set `HF_HUB_OFFLINE=1` for cached-only mode behind firewalls.
- **Feature errors**: most backends are gated behind `onnx` or `candle`.
- **Offset mismatches**: all spans use character offsets, not byte offsets. See [CONTRACT.md](docs/CONTRACT.md).

## Documentation

- [QUICKSTART](docs/QUICKSTART.md) -- getting started
- [CONTRACT](docs/CONTRACT.md) -- offset semantics, scope
- [BACKENDS](docs/BACKENDS.md) -- backend details, feature flags
- [ARCHITECTURE](docs/ARCHITECTURE.md) -- crate layout
- [REFERENCES](docs/REFERENCES.md) -- full bibliography
- [API docs](https://docs.rs/anno)

## References

[1] Grishman & Sundheim, *COLING* 1996.
[2] Tjong Kim Sang & De Meulder, *CoNLL* 2003.
[3] Otmazgin et al., *AACL* 2022 (F-COREF).
[4] Jurafsky & Martin, *SLP3* 2024.
[5] Zaratiana et al., *NAACL* 2024 (GLiNER).
[6] Bogdanov et al., 2024 (NuNER).
[7] Li et al., *AAAI* 2022 (W2NER).
[8] Devlin et al., *NAACL* 2019 (BERT).
[9] Lafferty et al., *ICML* 2001 (CRF).
[10] Wang et al., *COLING* 2020 (TPLinker).
[11] Zaratiana et al., 2025 (GLiNER2).
[12] Rabiner, *Proc. IEEE* 1989 (HMM).

Full list: [docs/REFERENCES.md](docs/REFERENCES.md). Citeable via [CITATION.cff](CITATION.cff).

## License

Dual-licensed under MIT or Apache-2.0.
