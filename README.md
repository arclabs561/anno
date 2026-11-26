# anno

[![crates.io](https://img.shields.io/crates/v/anno.svg)](https://crates.io/crates/anno)
[![Documentation](https://docs.rs/anno/badge.svg)](https://docs.rs/anno)

Anno extracts entity spans, links coreferent mentions, and redacts common forms
of personally identifiable information. Pattern and heuristic backends run offline.

## Library

Run the example with offline backends from the repository root.

```sh
cargo run -p anno --no-default-features --example quickstart
```

Run `cargo add anno` in your application to use the latest published Rust library.
This checkout’s release line has the following manifest entry once published.

```toml
[dependencies]
anno = "0.16"
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
The default document budget is 512 tokens, including special tokens. Use
`FCoref::from_path_with_document_budget` to accept longer documents. The encoder reads
the document in segments and the scorer links mentions across those segments.

## CLI

```sh
cargo install anno-cli --locked
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
- For agent clients that load local plugins, select `.agents/` from this checkout. [CLI skill](.agents/skills/anno-cli/SKILL.md) · [QA skill](.agents/skills/anno-qa/SKILL.md)
- [Compare two editions of a harbor bulletin with the published Python wheel](crates/anno-py/README.md#compare-dates-in-two-bulletins)
- [Source text, offsets and provenance](docs/CONTRACT.md)
- [Backends and feature flags](docs/BACKENDS.md)
- [Fixtures, datasets and QA](docs/QA.md) and [evaluation sampling](docs/EVALUATION_SAMPLING.md)
- [References](docs/REFERENCES.md) and [runnable examples](crates/anno/examples/)

## License

MIT or Apache-2.0.
