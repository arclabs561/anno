# anno

Information extraction for Rust. Find named entities, link mentions, and
extract relations. Keep the words behind each result.

[Rust](#find-mentions-in-rust) · [CLI](docs/QUICKSTART.md#install) · [Python](#compare-two-notices-in-python) · [API](https://docs.rs/anno)

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

## Same text, different source

Two addresses on a page both read `alice@example.org`. Select the `@` in
each one. Where should an editor put the cursor in the original HTML?

| On screen | In the HTML file | UTF-8 bytes |
| --- | --- | --- |
| First `@` | `&#64;` | `20..25` |
| Second `@` | `@` | `48..49` |

The checkout CLI can save the source map with the annotations and look up
either occurrence after cleaning the page. [Replay both lookups](docs/EXTRACTION_WORKFLOWS.md#trace-a-signal-to-its-original-html)

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

## What survives an export?

Give two exporters the same annotations for *cyclophilin gene*. One mention
covers the whole phrase. Another covers just *cyclophilin*. Anno’s HTML
preview highlights the longer span once. Its JSONL export keeps both records.

On the GENIA corpus, HTML displayed **50,709 of 55,740** supplied mentions.
JSONL preserved **all 55,740**. Both exporters received the reference
annotations directly, so the difference measures what a reviewer can see.
[Inspect the overlaps and replay the exports](docs/EXTRACTION_WORKFLOWS.md#review-overlapping-highlights)

## More examples

- [Save annotations once and export them again](docs/EXTRACTION_WORKFLOWS.md#saving-and-converting-annotations). Keep the source text in grounded JSON, then write BRAT annotations or graph CSV without rerunning extraction.
- [Compare relation extraction on CrossRE](docs/eval/crossre-full-prompt-comparison.md). See how much the score changes when the model receives the labels from each test sentence.
- [Test whether supplying missing mentions recovers pronoun links](docs/eval/fcoref-marked-rescue.md). On 256 GAP passages, 186 annotated spans entered the candidate set. None of the 512 labeled decisions changed.
- [Visualize evaluation results](docs/QUICKSTART.md#visualize-evaluation-results). Compare overlapping labels, extraction errors, and fallback predictions.
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
