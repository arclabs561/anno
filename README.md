# anno

[![crates.io](https://img.shields.io/crates/v/anno.svg)](https://crates.io/crates/anno)
[![Documentation](https://docs.rs/anno/badge.svg)](https://docs.rs/anno)

Anno is a Rust library for finding entities in text and linking mentions of the
same person, place, or thing. Each result points back to the words it came from.

## Find mentions

> **Sophie Wilson** designed the **ARM** processor.

Pass the sentence to `anno::extract`. The offline extractor finds a person
and an organization, each with a range in the original sentence.

```text
Sophie Wilson   PER    0..13
ARM             ORG   27..30
```

Try the example from a checkout with Rust 1.91 or later:

```sh
cargo run -p anno \
  --no-default-features \
  --example quickstart
```

In your own Rust project, install the library with `cargo add anno`.
The range `0..13` covers the first 13 Unicode scalar values. To highlight
a mention in a Rust string, convert its range to byte offsets.
[Use the library](docs/QUICKSTART.md#library-usage) · [Work with offsets](docs/CONTRACT.md#source-text-and-offsets)

## Follow a pronoun across captions

> **Émilie Durand** inspected *the east gate*.\
> **She** said *it* would reopen on Monday.

FCoref groups the two mentions of Durand and the two mentions of the gate.
The caption file supplies the playback times. Together they let an editor
select *She* in the second cue and jump to *Émilie Durand* in the first.

Add a microphone emoji and a space before *Émilie*. In the full transcript,
*She* moves from character 39 to 41. In the recording, its cue still begins at
14.9 seconds. The replay checks the links and cue ranges in both versions
using a local FP32 model. [Run the caption example](crates/anno/examples/caption_links/README.md)

## Where do pronoun links go missing?

In Anno’s replay of 128 English passages from the GAP test split, 117
pronoun–name pairs were labeled as referring to the same person. FCoref
received the raw passages and predicted its own mentions.

| FCoref output | Pairs |
| --- | ---: |
| Both mentions linked | 73 |
| Mentions in separate clusters | 3 |
| At least one span missing | 41 |

Of the 44 true links FCoref missed, 41 had at least one mention missing
at its exact source range. In the other three, both mentions appeared and
FCoref placed them in different clusters. The study also reports false
links and compares three resolvers on these passages.
[Results, scoring method, and replay](docs/eval/coref-panels.md#gap-pronoun-decisions)

## Command line

```sh
cargo install anno-cli --locked
```

```sh
ANNO_NO_DOWNLOADS=1 anno extract --model heuristic \
  --text "Lynn Conway worked at IBM and Xerox PARC in California."
```

```text
PER:1 "Lynn Conway"
ORG:2 "IBM" "Xerox PARC"
LOC:1 "California"
```

Use `--format grounded` to retain the input text alongside its annotations.
The default stack may download weights on first use. `ANNO_NO_DOWNLOADS=1`
limits it to available local models, patterns and heuristics.

## More examples

- [Compare two harbor bulletins in Python](crates/anno-py/README.md#compare-dates-in-two-bulletins). The deadline moved from December 4 to December 7. Find the line that changed in each edition.
- [Choose your own entity types](docs/QUICKSTART.md#cli-extract-entities). Give GLiNER labels such as `DRUG` and `SYMPTOM` to look for in a passage.
- [Correct one occurrence of a name](docs/QUICKSTART.md#review-a-catalog-link-from-a-source-mention). Two entries read “Ora Venn.” Assign the signature to the clerk while keeping the other entry under the musician.
- [Use your own tokenizer](crates/anno/examples/fcoref_word_spans.rs) or [process a directory of documents](docs/QUICKSTART.md#cli-batch-processing).

<details>
<summary>Cargo dependency for the 0.16 release line</summary>

Once published, this checkout’s release line can be selected in `Cargo.toml`:

```toml
[dependencies]
anno = "0.16"
```

</details>

## Documentation

[Quickstart](docs/QUICKSTART.md) · [Agent guide](docs/README.md#use-anno-with-an-agent) · [Source text and offsets](docs/CONTRACT.md) · [Backends and features](docs/BACKENDS.md)

[Evaluation and datasets](docs/QA.md) · [Evaluation sampling](docs/EVALUATION_SAMPLING.md) · [Research references](docs/REFERENCES.md) · [All examples](crates/anno/examples/)

## License

MIT or Apache-2.0.
