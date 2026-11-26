# anno

[![crates.io](https://img.shields.io/crates/v/anno.svg)](https://crates.io/crates/anno)
[![Documentation](https://docs.rs/anno/badge.svg)](https://docs.rs/anno)

Anno is a Rust library for finding entities in text and linking mentions of the
same person, place, or thing. Each result points back to the words it came from.

## Find a name

With Rust 1.91 or later, run `cargo add anno`. Extract the mentions in a
sentence, then select the people:

```rust,no_run
use anno::prelude::*;

fn main() -> anno::Result<()> {
    let mentions = anno::extract("Sophie Wilson designed the ARM processor.")?;
    for person in mentions.of_type(&EntityType::Person) {
        println!("{} ({}..{})", person.text, person.start(), person.end());
    }
    Ok(())
}
```

The offline example finds two mentions, *Sophie Wilson* and *ARM*. Selecting
people leaves one:

```text
Sophie Wilson (0..13)
```

The range counts Unicode scalar values from the start of the sentence. To
highlight a mention in a Rust string, convert its range to byte offsets first.
[Working with offsets](docs/CONTRACT.md#source-text-and-offsets)

From a checkout, try the same sentence without downloading a model:

```sh
cargo run -p anno --no-default-features --example quickstart
```

## Follow a pronoun across captions

> Émilie Durand inspected the east gate.
>
> She said it would reopen on Monday.

Here FCoref links *She* to *Émilie Durand* and *it* to *the east gate*. A
caption editor can use either link to highlight the earlier words or play
the cue that contains them.

Add a microphone emoji and a space before *Émilie*. In the full transcript,
*She* moves from character 39 to 41. In the recording, its cue still begins at
14.9 seconds. The replay checks the links and cue ranges in both versions
using a local FP32 model. [Run the caption example](crates/anno/examples/caption_links/README.md)

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

## More examples

- [Compare two harbor bulletins in Python](crates/anno-py/README.md#compare-dates-in-two-bulletins). The deadline moved from December 4 to December 7. Find the line that changed in each edition.
- [Choose your own entity types](docs/QUICKSTART.md#cli-extract-entities). Give GLiNER labels such as `DRUG` and `SYMPTOM` to look for in a passage.
- [Correct one occurrence of a name](docs/QUICKSTART.md#review-a-catalog-link-from-a-source-mention). Two entries read “Ora Venn.” Assign the signature to the clerk while keeping the other entry under the musician.
- [Use your own tokenizer](crates/anno/examples/fcoref_word_spans.rs) or [process a directory of documents](docs/QUICKSTART.md#cli-batch-processing).

## Documentation

[Quickstart](docs/QUICKSTART.md) · [Agent guide](docs/README.md#use-anno-with-an-agent) · [Source text and offsets](docs/CONTRACT.md) · [Backends and features](docs/BACKENDS.md)

[Evaluation and datasets](docs/QA.md) · [Evaluation sampling](docs/EVALUATION_SAMPLING.md) · [Research references](docs/REFERENCES.md) · [All examples](crates/anno/examples/)

## License

MIT or Apache-2.0.
