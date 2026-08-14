# anno

[![crates.io](https://img.shields.io/crates/v/anno.svg)](https://crates.io/crates/anno)
[![Documentation](https://docs.rs/anno/badge.svg)](https://docs.rs/anno)

Text annotation and entity extraction.

`anno` extracts entity spans, resolves coreference, and finds common forms of
personally identifiable information. Model-backed extractors are optional; the
rule-based extractors work offline.

## Library

```toml
[dependencies]
anno = "0.11"
```

```rust
let entities = anno::extract("Sophie Wilson designed the ARM processor.")?;
for entity in entities {
    println!(
        "{} [{}] {}..{}",
        entity.text,
        entity.entity_type,
        entity.start(),
        entity.end()
    );
}
# Ok::<(), anno::Error>(())
```

Offsets are character offsets, not UTF-8 byte offsets. Confidence scores depend
on the selected backend and are not calibrated across backends.

Pattern-based PII redaction covers values such as email addresses, phone
numbers, and identification numbers:

```rust
use anno::pii;

let text = "John Smith's SSN is 123-45-6789.";
let redacted = pii::redact_patterns(text);
assert_eq!(redacted, "John Smith's SSN is [ID_NUMBER_1].");
```

## CLI

```sh
cargo install anno-cli
```

```console
$ anno extract --text "Lynn Conway worked at IBM and Xerox PARC in California."
PER:1 "Lynn Conway"
ORG:2 "IBM" "Xerox PARC"
LOC:1 "California"
```

`anno extract --format json` emits JSON. Run `anno help <command>` for command
options.

## Backends

The default `onnx` feature may download model weights on first use. Set
`ANNO_NO_DOWNLOADS=1` to allow only cached or local models; extraction then
falls back to pattern and heuristic backends when those models are unavailable.
Other optional features include `candle`, GPU support, discourse analysis,
metrics, graph export, and JSON Schema. Their requirements and model identifiers
are listed in [the backend guide](docs/BACKENDS.md).

This crate provides inference and annotation utilities, not model training.
The minimum supported Rust version is 1.88.

Further documentation:

- [Quickstart](docs/QUICKSTART.md)
- [Span and offset contract](docs/CONTRACT.md)
- [References](docs/REFERENCES.md)
- [Examples](crates/anno/examples/)

## License

MIT or Apache-2.0.
