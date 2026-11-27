# anno

[![crates.io](https://img.shields.io/crates/v/anno.svg)](https://crates.io/crates/anno)
[![Documentation](https://docs.rs/anno/badge.svg)](https://docs.rs/anno)

Anno finds mentions in text and links them to the words they refer to.
Each mention carries its range in the original text.

[Rust](#find-mentions-in-rust) · [Command line](docs/QUICKSTART.md#install) · [Python](#compare-two-notices-in-python)

## Find mentions in Rust

In a Rust project, run `cargo add anno@0.15 --no-default-features` for the
offline extractor. Put this in `src/main.rs` and run `cargo run`:

```rust
use anno::extract;

fn main() -> anno::Result<()> {
    let text =
        "Lynn Conway worked at IBM.";
    for e in extract(text)? {
        println!("{} [{}]",
            e.text, e.entity_type);
    }
    Ok(())
}
```

```text
Lynn Conway [PER]
IBM [ORG]
```

## Compare two notices in Python

A harbor office issued a revised bulletin. Which date changed?
Install the Python bindings with `python -m pip install anno-py`.
The [bulletin example](crates/anno-py/README.md#compare-dates-in-two-bulletins)
uses `anno_py.Extractor()` on both editions and returns the source line for
each changed date.

> Applications for a winter berth close on **2026-12-04**.\
> Applications for a winter berth now close on **2026-12-07**.

The lines tell you what moved: applications stay open three days longer.
Both notices still schedule the public drawing for December 9.

## Follow a pronoun across captions

> **Émilie Durand** inspected *the east gate*.\
> **She** said *it* would reopen on Monday.

Select *She* in the cue at **14.9 seconds**. FCoref links it to *Émilie
Durand* in the cue at **12.3 seconds**. The caption adapter returns both cue
ranges so an editor can jump to the earlier line.
[Reproduce with FCoref](crates/anno/examples/caption_links/README.md)

## Can a missing mention be rescued?

In one FCoref replay, many missed pronoun links lacked an exact name or
pronoun span in the output. Would giving those spans a place among the
candidates recover the links?

On a separate panel of 256 GAP development passages, Anno reserved slots for
the annotated names and pronouns while keeping the candidate budget fixed.
**186 marked spans entered the candidate set. None of the 512 labeled link
decisions changed.** The project kept the existing selection policy.
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
