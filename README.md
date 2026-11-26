# anno

[![crates.io](https://img.shields.io/crates/v/anno.svg)](https://crates.io/crates/anno)
[![Documentation](https://docs.rs/anno/badge.svg)](https://docs.rs/anno)

Anno finds names, places, and organizations in text. Each mention has a label
and a range you can use to highlight the original words. Anno can also link
mentions that refer to the same person or thing.

## Find mentions in text

Add the library with `cargo add anno` (Rust 1.91 or later).

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

With the offline backends, this finds `Sophie Wilson` at `0..13` and `ARM` at
`27..30`. The ranges count Unicode scalar values. To slice a Rust string,
convert them to UTF-8 byte offsets first. [Working with offsets](docs/CONTRACT.md#source-text-and-offsets)

Run the offline example from a checkout:

```sh
cargo run -p anno --no-default-features --example quickstart
```

## Link mentions across captions

> Émilie Durand inspected the east gate.
>
> She said it would reopen on Monday.

On this passage, the FCoref model links *She* to *Émilie Durand* and *it* to
*the east gate*. The caption example uses the position of *She* to find its
cue, which starts at 14.9 seconds.

Add a microphone emoji and a space to the start of the first caption.
*She* moves from character 39 to character 41 in the transcript. It still
begins the same cue. The replay checks both versions using a local FP32 model. [Run the caption example](crates/anno/examples/caption_links/README.md)

If your tokenizer already supplies word boundaries, pass its UTF-8 byte
ranges to `analyze_words`. [Use your own word spans](crates/anno/examples/fcoref_word_spans.rs)

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

- [Choose your own entity types](docs/QUICKSTART.md#cli-extract-entities)
- [Review a catalog link for one occurrence of a name](docs/QUICKSTART.md#review-a-catalog-link-from-a-source-mention)
- [Process a directory of documents](docs/QUICKSTART.md#cli-batch-processing)
- [Quickstart and examples](docs/QUICKSTART.md)
- [Use Anno with an agent](docs/README.md#use-anno-with-an-agent)
- [Compare two editions of a harbor bulletin with the published Python wheel](crates/anno-py/README.md#compare-dates-in-two-bulletins)
- [Source text, offsets and provenance](docs/CONTRACT.md)
- [Backends and feature flags](docs/BACKENDS.md)
- [Fixtures, datasets and QA](docs/QA.md) and [evaluation sampling](docs/EVALUATION_SAMPLING.md)
- [References](docs/REFERENCES.md) and [runnable examples](crates/anno/examples/)

## License

MIT or Apache-2.0.
