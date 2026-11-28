# anno

[![crates.io](https://img.shields.io/crates/v/anno.svg)](https://crates.io/crates/anno)
[![Documentation](https://docs.rs/anno/badge.svg)](https://docs.rs/anno)

Anno finds names in text and records where each one appears. It can also
link mentions of the same person, place, or thing across sentences.

[Rust](#find-mentions-in-rust) · [Command line](docs/QUICKSTART.md#install) · [Python](#compare-two-notices-in-python)

## Find mentions in Rust

In a Rust project, run `cargo add anno@0.15 --no-default-features` for the
offline extractor. Put this in `src/main.rs` and run `cargo run`:

```rust
use anno::extract;

fn main() -> anno::Result<()> {
    let text =
        "Lynn Conway worked at IBM.";
    for mention in extract(text)? {
        println!(
            "{} [{}] {}..{}",
            mention.text,
            mention.entity_type,
            mention.start(),
            mention.end()
        );
    }
    Ok(())
}
```

```text
Lynn Conway [PER] 0..11
IBM [ORG] 22..25
```

Ranges count Unicode scalar values from zero. The end is exclusive.
[Using these ranges in Rust](docs/CONTRACT.md#source-text-and-offsets)

## Follow a pronoun across captions

> **Émilie Durand** inspected *the east gate*.\
> **She** said *it* would reopen on Monday.

Select *She* in the cue at **14.9 seconds**. FCoref links it to *Émilie
Durand* in the cue at **12.3 seconds**. The caption adapter returns both cue
ranges so an editor can jump to the earlier line.
[Reproduce with FCoref](crates/anno/examples/caption_links/README.md)

## Compare two notices in Python

A harbor office extended its application deadline. Install the bindings with
`python -m pip install anno-py`. From the repository root, run:

```sh
cd crates/anno-py/examples
python bulletin_diff.py \
  bulletins/bulletin-edition-1.txt \
  bulletins/bulletin-edition-2.txt
```

The script compares the dates in the two notices and returns the source lines
for the dates whose counts changed:

> Issued 2026-11-18. Applications for a winter berth close on **2026-12-04**.\
> Issued 2026-11-18. Applications for a winter berth now close on **2026-12-07**.

Applications stay open three days longer. The drawing is still on December 9.
[Compare your own notices](crates/anno-py/README.md#compare-dates-in-two-bulletins)

## Would supplying the missing names recover a pronoun link?

In an earlier test, FCoref missed 44 true pronoun links. In 41 of those cases,
at least one of the two mentions was absent at its exact range in the output.

On a separate set of 256 GAP development passages, the researchers gave FCoref
the locations of the pronouns and candidate names. They reserved space for
those spans without increasing the total number of candidates the model could
consider. **186 additional annotated spans made the cut. None of the 512
labeled link decisions changed.**
[Original observation](docs/eval/coref-panels.md#gap-pronoun-decisions) ·
[Experiment and replay](docs/eval/fcoref-marked-rescue.md)

## More examples

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
