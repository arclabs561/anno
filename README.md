# anno

[![crates.io](https://img.shields.io/crates/v/anno.svg)](https://crates.io/crates/anno)
[![Documentation](https://docs.rs/anno/badge.svg)](https://docs.rs/anno)

Text annotation and entity extraction.

Extract entity spans, coreference links, and common forms of personally
identifiable information from text. Model-backed extractors are optional; the
rule-based extractors work offline.

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

For repeated extraction, create and reuse a model rather than calling
`anno::extract` each time. Offsets are character offsets, not UTF-8 byte
offsets. Confidence scores are backend-local and are not calibrated across
backends.

Redact email addresses, US Social Security numbers, and IBANs without
downloading a model.
The patterns check formats and checksums, not whether an account exists.
See the [PII pattern limits](docs/CONTRACT.md#pii-pattern-limits) for supported
formats and validation rules.

```rust
use anno::pii;

let text = "John Smith's SSN is 123-45-6789.";
let redacted = pii::redact_patterns(text);
assert_eq!(redacted, "John Smith's SSN is [ID_NUMBER_1].");
```

### Coreference with your own tokenizer

Aurelius keeps UTF-8 byte ranges so it can quote the original document. When
you already have word boundaries, pass those byte ranges to `analyze_words`.
Its mentions and clusters use half-open Unicode scalar offsets. Convert them
back with `SpanConverter` before slicing a Rust string.

The [word spans example](crates/anno/examples/fcoref_word_spans.rs) runs on a
local FP32 [FCoref export](scripts/export_fcoref.py):

```sh
cargo run -p anno --example fcoref_word_spans -- /path/to/fcoref_onnx
```

For `Émilie\nDurand reviewed the proposal. Émilie Durand said she approved it.`
(the first name spans a newline), the English checkpoint produced these groups:

| Mention | Characters | UTF-8 bytes | Cluster |
| --- | --- | --- | --- |
| `Émilie\nDurand` | 0..13 | 0..14 | 1 |
| `the proposal` | 23..35 | 24..36 | 2 |
| `Émilie Durand` | 37..50 | 38..52 | 1 |
| `she` | 56..59 | 58..61 | 1 |
| `it` | 69..71 | 71..73 | 2 |

The repeated name has its own source range. Joining the words with spaces
would lose the first mention’s newline. The model proposes links and prints
raw mention scores. These scores are uncalibrated and cluster numbers are local
to the document. Supply every non-whitespace token, including punctuation.
This version accepts one encoder segment (512 tokens by default, including
special tokens).

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

This uses the offline heuristic backend. `anno extract --format json` emits
JSON. Run `anno help <command>` for command options.

## Backends

The default `onnx` feature may download model weights on first use. Set
`ANNO_NO_DOWNLOADS=1` to use only cached or local models; the default stack then
falls back to pattern and heuristic extractors when a model is unavailable.
The [backend guide](docs/BACKENDS.md) lists model identifiers, feature flags,
and hardware requirements.

This crate provides inference and annotation utilities, not model training.
The minimum supported Rust version is 1.91.

Further documentation:

- [Quickstart](docs/QUICKSTART.md)
- [Span and offset contract](docs/CONTRACT.md)
- [References](docs/REFERENCES.md)
- [Examples](crates/anno/examples/)

## License

MIT or Apache-2.0.
