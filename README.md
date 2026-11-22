# anno

[![crates.io](https://img.shields.io/crates/v/anno.svg)](https://crates.io/crates/anno)
[![Documentation](https://docs.rs/anno/badge.svg)](https://docs.rs/anno)

Anno extracts entity spans, links coreferent mentions, and redacts common forms
of personally identifiable information. Pattern and heuristic backends run offline.

## Library

```toml
[dependencies]
anno = "0.14"
```

```rust,no_run
fn main() -> anno::Result<()> {
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
    Ok(())
}
```

Offsets count Unicode scalar values in the exact input text. For repeated calls,
[reuse a model](docs/QUICKSTART.md#library-usage). Each backend has its own
confidence scale.

Redact supported PII formats without downloading weights.
[Pattern coverage and validation](docs/CONTRACT.md#pii-pattern-limits)

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
$ ANNO_NO_DOWNLOADS=1 anno extract --model heuristic --text "Lynn Conway worked at IBM and Xerox PARC in California."
PER:1 "Lynn Conway"
ORG:2 "IBM" "Xerox PARC"
LOC:1 "California"
```

Use `--format grounded` to retain the input text alongside its annotations.
The default stack may download weights on first use. `ANNO_NO_DOWNLOADS=1`
limits it to available local models, patterns and heuristics.

Anno supports Rust 1.91 and later.

- [Quickstart and examples](docs/QUICKSTART.md)
- [Source text, offsets and provenance](docs/CONTRACT.md)
- [Backends and feature flags](docs/BACKENDS.md)
- [Evaluation panels and sampling](docs/EVALUATION_SAMPLING.md)
- [References](docs/REFERENCES.md) and [runnable examples](crates/anno/examples/)

## License

MIT or Apache-2.0.
