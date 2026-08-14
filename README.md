# anno

[![crates.io](https://img.shields.io/crates/v/anno.svg)](https://crates.io/crates/anno)
[![Documentation](https://docs.rs/anno/badge.svg)](https://docs.rs/anno)

Text annotation and entity extraction.

`anno` extracts entity spans from text, classifies common PII, and exports
annotations for downstream tools. It works offline with rule-based fallbacks; ML
backends are optional and feature-gated.

## Install

```toml
[dependencies]
anno = "0.11"
```

MSRV: 1.88. Dual-licensed under MIT or Apache-2.0.

## Extract Entities

```rust
let entities = anno::extract("Sophie Wilson designed the ARM processor.")?;
for e in &entities {
    println!("{} [{}] ({},{}) {:.2}", e.text, e.entity_type, e.start(), e.end(), e.confidence);
}
// Results include character offsets and backend-specific confidence scores.
// `ANNO_NO_DOWNLOADS=1` blocks new Hugging Face fetches but still loads
// cached or locally exported models.
# Ok::<(), anno::Error>(())
```

Filter results with the `prelude` extension traits:

```rust
use anno::prelude::*;

# let entities = anno::extract("Sophie Wilson designed the ARM processor.")?;
let people: Vec<_> = entities.of_type(&EntityType::Person).collect();
let confident: Vec<_> = entities.above_confidence(0.8).collect();
# Ok::<(), Error>(())
```

## Redact PII

```rust
use anno::{pii, Model, StackedNER};

let text = "John Smith's SSN is 123-45-6789.";
let m = StackedNER::default();
let redacted = pii::scan_and_redact(text, &m)?;
// "[PERSON_1]'s SSN is [ID_NUMBER_1]."
# Ok::<(), anno::Error>(())
```

PII scanning combines extracted entities with structured patterns such as SSNs,
credit cards, IBANs, email addresses, and phone numbers.

## Backends

`StackedNER::default()` selects an available backend at runtime. With
default features it tries cached ONNX models when available, then falls back to
pattern and heuristic extraction. Confidence scores are backend-specific and
are not calibrated across backends. Set `ANNO_NO_DOWNLOADS=1` to prevent new
model downloads while still allowing cached or local models.

Feature flags:

- `onnx` (default): ONNX Runtime backends.
- `candle`: pure-Rust backends.
- `metal` / `cuda`: GPU acceleration through `candle`.
- `llm`: LLM-based extraction providers.
- `discourse`: centering theory, abstract anaphora, and dialogue acts.
- `analysis`: coreference metrics and cluster encoders.
- `schema`: JSON Schema for output types.

See [docs/BACKENDS.md](docs/BACKENDS.md) for model IDs, status, and backend
selection details.

## Custom Backends

`AnyModel` wraps a closure into a `Model` when you need to plug in an external
NER system:

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

## CLI

```sh
cargo install anno-cli
```

```sh
anno extract --text "Lynn Conway worked at IBM and Xerox PARC in California."
# PER:1 "Lynn Conway"
# ORG:2 "IBM" "Xerox PARC"
# LOC:1 "California"

anno extract --model gliner --extract-types "DRUG,SYMPTOM" \
  --text "Aspirin can treat headaches and reduce fever."
# Output depends on the installed model weights.

anno debug --coref -t "Sophie Wilson designed the ARM. She revolutionized mobile computing."
# Coreference: "Sophie Wilson" -> "She"
```

JSON output with `anno extract --format json`. Batch processing with `anno
batch`. Graph-oriented exports use `anno export --format ntriples`, `jsonld`,
or `graph-csv`; `graph-ntriples` additionally requires installing `anno-cli`
with its `graph` feature.

## Coreference

Coreference is available through rule-based and neural resolvers. RAG
preprocessing (`rag::resolve_for_rag()`, `rag::preprocess()`) rewrites pronouns
for self-contained chunks after splitting.

## Scope

Inference-time extraction. Training pipelines are out of scope. Use upstream
frameworks and export ONNX weights.

## Troubleshooting

- **ONNX linking errors**: use `default-features = false` for builds without C++, or check `ORT_DYLIB_PATH`.
- **Model downloads**: set `ANNO_NO_DOWNLOADS=1` for cached-only mode behind firewalls.
- **Feature errors**: most backends are gated behind `onnx` or `candle`.
- **Offset mismatches**: all spans use character offsets, not byte offsets. See [CONTRACT.md](docs/CONTRACT.md).

## More

- [docs/QUICKSTART.md](docs/QUICKSTART.md): a longer walkthrough.
- [docs/CONTRACT.md](docs/CONTRACT.md): span and offset contracts.
- [docs/REFERENCES.md](docs/REFERENCES.md): cited papers and model references.
- `crates/anno/examples/`: runnable examples.

## License

Dual-licensed under MIT or Apache-2.0.
