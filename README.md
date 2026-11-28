# anno

Anno finds named entities in text, links mentions, and extracts relations.
Use the Rust library in an application or the `anno` command on a directory
of documents. Each mention points back to the words in its source.

[Rust](#find-mentions-in-rust) · [CLI](#extract-a-directory) · [Python](#compare-two-notices-in-python) · [Models](#choose-what-to-extract) · [API](https://docs.rs/anno)

## Find mentions in Rust

In a Rust project, run `cargo add anno@0.15 --no-default-features` for the
offline extractor. Put this in `src/main.rs` and run `cargo run`:

```rust
use anno::extract;

fn main() -> anno::Result<()> {
    let text = "Lynn Conway worked at IBM.";
    for m in extract(text)? {
        let span = m.start()..m.end();
        println!("{} [{}] {span:?}", m.text, m.entity_type);
    }
    Ok(())
}
```

```text
Lynn Conway [PER] 0..11
IBM [ORG] 22..25
```

Count from zero: the letters in `IBM` occupy positions 22, 23, and 24.
The range ends just before the period. Anno counts Unicode scalar values.
[Using these ranges in Rust](docs/CONTRACT.md#source-text-and-offsets)

## Extract a directory

Who appears in all three reports? From the root of this checkout, extract the
people in the sample documents. Each row carries its source and character range.

```sh
cargo install --locked --path crates/anno-cli \
  --no-default-features --features extractor-readability
anno batch --dir testdata/fixtures --model heuristic --format jsonl |
  jq -r '.id as $doc | .entities[] | select(.type == "PER") |
         [$doc, .text, .start, .end] | @tsv'
```

| Document | Person | Character range |
| --- | --- | --- |
| `cross_doc/doc1` | Jensen Huang | `0..12` |
| `cross_doc/doc2` | Jensen Huang | `0..12` |
| `cross_doc/doc3` | Sundar Pichai | `0..13` |
| `cross_doc/doc3` | Jensen Huang | `18..30` |

For your own archive, pass `--dir notes --output results` to save one
annotation file per document. A source at `notes/letters/bio.txt` becomes
`results/letters/bio.json`. The published 0.15 CLI scans only the top level.
[Batch processing](docs/QUICKSTART.md#cli-batch-processing)

## Choose what to extract

Give GLiNER2.5 the labels
`person`, `organization`, and `location`. In Anno’s pinned checkpoint replay,
it picks out *Mario Rossi*, *Apple*, and *Cupertino* from a sentence. To try the same input,
build the checkout CLI with `gliner2-fastino` and point `GLINER25_MODEL_DIR`
at the FP32 ONNX export:

```sh
anno extract --model gliner25 \
  --extract-types person,organization,location \
  --text 'Mario Rossi works at Apple in Cupertino'
```

For a directed edge such as `person → works_for → organization`, the
separate `anno joint` command accepts a relation schema and a local PyTorch
checkpoint. [Model setup and examples](docs/BACKENDS.md#fastino-gliner25) ·
[Checkpoint replay](scripts/qa/checkpoint_replay/README.md)

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

Already have word boundaries? Aurelius passes its document text and word
spans to Anno’s `analyze_words`. [Read the caller](https://github.com/arlyon/aurelius/blob/1319bcdcee28ee5b668b41a2a4f85dd0722b460f/lanes/src/language.rs#L317)

## Compare two notices in Python

Two notices give different deadlines to apply for a winter boat space.
Which sentence changed? Install the bindings with
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

Two annotations can start at the same letter:

```text
cyclophilin gene
^^^^^^^^^^^^^^^^  DNA
^^^^^^^^^^^       protein
```

Anno’s HTML preview highlights the longer phrase once. Its JSONL exporter
writes a record for each annotation.

On the GENIA corpus, HTML displayed **50,709 of 55,740** supplied mentions.
JSONL preserved **all 55,740**. Both exporters received the reference
annotations directly, so the difference measures what a reviewer can see.
[Inspect the overlaps and replay the exports](docs/EXTRACTION_WORKFLOWS.md#review-overlapping-highlights)

## What else can you do with a mention?

- **Correct a namesake.** Two entries in the same log say “Ora Venn.” Assign the signature to the clerk while keeping the other entry under the musician. [Review one occurrence](docs/QUICKSTART.md#review-a-catalog-link-from-a-source-mention)
- **Hand annotations to another tool.** Save a grounded document once. Write BRAT annotations for an editor or graph CSV for analysis without rerunning the extractor. [Convert saved annotations](docs/EXTRACTION_WORKFLOWS.md#saving-and-converting-annotations)
- **Find out what a stronger model buys.** Compare recovered labels with the extra predictions someone must review, or inspect where two labels overlap. [Visualize evaluation results](docs/QUICKSTART.md#visualize-evaluation-results)
- **Check what the benchmark gave the model.** On CrossRE, giving the same checkpoint each test sentence’s annotated labels raised strict relation F1 from 29.5% to 58.2%. [Inspect the paired comparison](docs/eval/crossre-full-prompt-comparison.md)

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
