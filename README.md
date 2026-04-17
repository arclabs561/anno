# anno

[![crates.io](https://img.shields.io/crates/v/anno.svg)](https://crates.io/crates/anno)
[![Documentation](https://docs.rs/anno/badge.svg)](https://docs.rs/anno)
[![CI](https://github.com/arclabs561/anno/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/anno/actions/workflows/ci.yml)

Text annotation and entity extraction. Covers NER, coreference resolution,
PII detection, relation extraction, and export to standard formats.

Multiple backends (ML, statistical, rule-based) are tried at runtime;
works without model downloads via built-in fallbacks.

Dual-licensed under MIT or Apache-2.0. MSRV: 1.88.

## Quickstart

```toml
[dependencies]
anno = "0.6.0"
```

```rust
let entities = anno::extract("Sophie Wilson designed the ARM processor.")?;
for e in &entities {
    println!("{} [{}] ({},{}) {:.2}", e.text, e.entity_type, e.start(), e.end(), e.confidence);
}
// Sophie Wilson [PER] (0,13) 1.00
// ARM [misc] (27,30) 0.99
// Output varies by backend. With `onnx` feature and models cached,
// the ML backends produce more specific types (e.g. ARM -> ORG).
# Ok::<(), anno::Error>(())
```

Filter results with `prelude` (re-exports common types including `Result`):

```rust
use anno::prelude::*;

# let entities = anno::extract("Sophie Wilson designed the ARM processor.")?;
let people: Vec<_> = entities.of_type(&EntityType::Person).collect();
let confident: Vec<_> = entities.above_confidence(0.8).collect();
# Ok::<(), Error>(())
```

For backend control, construct a model directly:

```rust
use anno::{Model, StackedNER};

let m = StackedNER::default();
let ents = m.extract_entities("Sophie Wilson designed the ARM processor.", None)?;
# Ok::<(), anno::Error>(())
```

`StackedNER::default()` selects the best available backend at runtime: BERT ONNX and NuNER (both tried independently when `onnx` enabled and models cached), then GLiNER if neither loaded, falling back to pattern + heuristic extraction. Set `ANNO_NO_DOWNLOADS=1` to force cached-only behavior.

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

## PII detection

Classify NER entities as PII and scan for structured patterns (SSN, credit card, IBAN, email, phone). Redact or pseudonymize in one call:

```rust
use anno::{pii, Model, StackedNER};

let text = "John Smith's SSN is 123-45-6789.";
let m = StackedNER::default();
let redacted = pii::scan_and_redact(text, &m)?;
// "[PERSON_1]'s SSN is [ID_NUMBER_1]."
# Ok::<(), anno::Error>(())
```

## Backends

17 backends spanning ML (GLiNER, NuNER, BERT, W2NER), statistical (CRF, HMM), rule-based (pattern, heuristic), and LLM-based extraction. ML backends are feature-gated (`onnx` or `candle`); weights download from HuggingFace on first use. See [BACKENDS.md](docs/BACKENDS.md) for the full list, default models, and status.

### Feature flags

`onnx` (default) -- ONNX Runtime backends. `candle` -- pure-Rust backends, no C++ runtime. `metal`/`cuda` -- GPU acceleration (enables `candle`). `llm` -- LLM-based extraction via OpenRouter, Anthropic, Groq, Gemini, or Ollama. `discourse` -- centering theory, abstract anaphora, dialogue acts. `analysis` -- coref metrics and cluster encoders. `schema` -- JSON Schema for output types. `production` -- `tracing` instrumentation.

## CLI

```sh
cargo install anno-cli --features onnx
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

Three resolvers: `SimpleCorefResolver` (rule-based, 9 sieves; requires `analysis` feature), `FCoref` (neural, 78.5 F1 on CoNLL-2012 [3]; requires `onnx`), and `MentionRankingCoref`. `FCoref` requires a one-time model export: `uv run scripts/export_fcoref.py` (from a repo clone).

RAG preprocessing (`rag::resolve_for_rag()`, `rag::preprocess()`): rewrites pronouns for self-contained chunks after splitting. Always available (no feature flag required).

## Scope

Inference-time extraction. Training pipelines are out of scope -- use upstream frameworks and export ONNX weights.

## Troubleshooting

- **ONNX linking errors**: use `default-features = false` for builds without C++, or check `ORT_DYLIB_PATH`.
- **Model downloads**: set `ANNO_NO_DOWNLOADS=1` for cached-only mode behind firewalls.
- **Feature errors**: most backends are gated behind `onnx` or `candle`.
- **Offset mismatches**: all spans use character offsets, not byte offsets. See [CONTRACT.md](docs/CONTRACT.md).

## Examples

All examples live in `crates/anno/examples/`. Run with `cargo run --example <name>`.

| Example | Feature | What it shows |
|---------|---------|---------------|
| `quickstart` | -- | One-line extraction, filtering with `EntitySliceExt` |
| `pii_redact` | -- | Detect names, SSNs, emails; redact or pseudonymize |
| `zero_shot` | `onnx` | Custom entity types ("drug", "symptom") via GLiNER |
| `relations` | -- | Entity-pair relation extraction with TPLinker |
| `coref` | `analysis` | Coreference chains linking "Marie Curie" and "Curie" |
| `export_formats` | -- | brat standoff, CoNLL BIO, JSONL, graph CSV |
| `rag_preprocess` | -- | Chunking + pronoun rewriting for self-contained RAG chunks |
| `batch` | -- | Parallel extraction over multiple documents |

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
