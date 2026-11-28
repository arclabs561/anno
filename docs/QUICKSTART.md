# Quickstart

Extract entities with the CLI or a reusable Rust model. To try the offline
Python bindings on a pair of documents, [compare dates in two bulletins](../crates/anno-py/README.md#compare-dates-in-two-bulletins).

Find any invocation and its flags in the [command and option index](CLI_OPTIONS.md). For direct BRAT output or conversion of reviewed predictions, see [saving and converting annotations](EXTRACTION_WORKFLOWS.md#saving-and-converting-annotations).

## Install

These commands select the releases published on crates.io. Run the first
from the Rust project that will use Anno.

```bash
cargo add anno
cargo install anno-cli --locked
```

To use the library from a local clone, run `cargo add anno --path /path/to/anno/crates/anno` in your project.

For the English heuristic and pattern examples, this smaller build includes
the Readability HTML extractor and skips the ONNX runtime.

```bash
cargo install anno-cli --locked --no-default-features --features extractor-readability
```

Single-text scoring with `anno eval` works in the standard CLI. Add
`--features eval` for dataset benchmarks or cross-document clustering.
For commands marked as using this checkout, install the CLI from the repository
root with those features.

```bash
cargo install --locked --path crates/anno-cli --bin anno --features "onnx eval"
```

## Reuse a CLI workflow profile

In this checkout, save the model and output format once and select the profile
before or after the command. Each explicit option overrides the corresponding
saved setting, even when the explicit value equals the command's built-in default.
Omitted fields keep their command-specific defaults.

```sh
anno config save standard --model stacked --format json
anno config ls
anno config show standard
anno --config standard -- "Zoë met Jane in Lyon."
anno batch --config standard --dir ./docs --quiet
```

| Command | Saved settings used |
| --- | --- |
| Positional text, `extract` | `model`, `format`, `threshold` |
| `batch`, `pipeline` | `model`, `format`, `coref`, `link_kb` |
| `debug` | `model`, `coref`, `link_kb` |
| `watch` | `model`, `format` (JSON or JSONL) |
| `enhance` | `format`, `coref`, `link_kb` |
| `cross-doc` (eval build) | `model` for raw inputs, `format` |
| `query FILE` | `format` |
| `eval` with live extraction, `dataset eval`, `privacy`, `export` and `dataset export` with raw input, `dataset context`, `explain`, `domain`, `singleton` | `model` |

Use `--coref=false` or `--link-kb=false` to override a saved switch. The profile
threshold is extraction confidence. Cross-document similarity, relation distance,
checkpoint paths, and benchmark sampling have their own command options.
Export encodings and command-specific report formats also use their own options.
Source offsets, runtime provenance and observed-language accounting apply automatically.
Operations outside the table, including scoring saved predictions, reject an
explicit profile. Use `anno help COMMAND` for the options available in your build.

When requesting relations with `anno extract`, `--relation-threshold` overrides the extraction threshold. With neither supplied, the relation cutoff is 0.55. The heuristic relation gap defaults to 120 source characters. [Relation behavior and distance trial](CONTRACT.md#cli-json-provenance)

## CLI: extract entities

```console
$ ANNO_NO_DOWNLOADS=1 anno extract --model heuristic --text "Lynn Conway worked at IBM and Xerox PARC in California."
PER:1 "Lynn Conway"
ORG:2 "IBM" "Xerox PARC"
LOC:1 "California"
```

Pattern-only extraction (offline, no weights):

```bash
anno extract --model pattern --text "Email jobs@acme.com by 2024-01-15 for \$100."
```

Extract entity types that you supply (requires the default `onnx` feature and
model weights):

```bash
anno extract --model gliner --extract-types "DRUG,SYMPTOM" \
  --text "Aspirin can treat headaches and reduce fever."
```

## Save annotations with the source text

```bash
anno extract --model pattern --format grounded --text '🦊 Email ada@example.org' > annotation.json
```

The email signal has surface `ada@example.org`, source `pattern`, and character
range `8..23`. The document includes the exact text used for extraction. Keep
your source document ID and revision alongside this file. For a compact entity
list use `--format json`. [Offset and provenance contract](CONTRACT.md)

## Review a catalog link from a source mention

This Rust example opens an original synthetic quay log. Two occurrences of “Ora Venn” start with the same catalog key. Inspect the proposal for the occurrence that signed the cargo log, then choose whether to apply it. Run these commands from the repository root.

```sh
review_mention() { cargo run --quiet --locked -p anno --no-default-features --example review_mention -- "$@"; }
review_mention preview
review_mention review > correction.json
review_mention show < correction.json
```

During `review`, read the source, selected occurrence, and affected tracks in the terminal. Enter `apply` or `defer`, then enter your reason on the next line. The same process holds the live proposal until you decide. For example, applying with the reason “The selected occurrence signed the cargo log” changes its binding to `quay-catalog/clerk`. Deferring with “The signature alone is inconclusive” keeps `quay-catalog/musician`. The other occurrence keeps its original binding in both cases.

The caller saves the grounded document, the selected source occurrence, the recorded origin of its prior identity, the reader’s reason, and the decision together. The saved quote, Unicode scalar range, and source-text digest let a later reader check which occurrence was reviewed. Only an applied decision contains a write receipt. The example file represents one decision on one document. [Example source](../crates/anno/examples/review_mention.rs) · [Catalog write contract](CONTRACT.md#grouping-flat-entities)

## Read changes to a document's catalog results

With the CLI built from this checkout, a subscriber can ask which source occurrences entered or left a person-specific search. The inputs are two complete grounded document snapshots. The publisher assigns an increasing sequence within the feed and can identify the cited source edition independently. To try a reassignment on the unchanged quay log above, run:

```sh
printf 'defer\nKeep the existing catalog assignment.\n' | review_mention review > deferred.json
printf 'apply\nUse the signed cargo log.\n' | review_mention review > applied.json
jq '{stream:"quay", sequence:1, source_revision:"log-edition-1", document:.document}' deferred.json > before.json
jq '{stream:"quay", sequence:2, source_revision:"log-edition-1", document:.document}' applied.json > after.json
anno query publication before.json --catalog-record quay-catalog musician
anno query publication after.json --since before.json --catalog-record quay-catalog musician
```

The first response contains both occurrences under the musician's record. The delta removes the one that signed the cargo log. Search the clerk's record instead to see that occurrence added. Omitting the catalog filter returns the complete occurrence view, including unresolved names. Each selection has its own checkpoint. The source digest and locations refer to the text in the grounded document.

Install a delta's removals, additions and target checkpoint in one transaction after checking the stored checkpoint against its base. Redelivery to the same target can be skipped. If the stored state diverged, obtain a full snapshot. Rust consumers can use `anno::publication::IndexSnapshot` and `IndexCheckpoint::needs_delta` to make these decisions. [Rust publication contract](../crates/anno/src/publication.rs) · [Chronological catalog pilot](decisions/catalog-publication-pilot.md)

## CLI: redact a document

`anno extract --threshold` filters which entities appear in the extraction
result. To produce a copy of the text with structured PII replaced, use the
offline privacy command.

```console
$ anno privacy --model pattern --action redact --text "Send a message to sample@example.org. Reference 123-45-6789."
Send a message to [CONTACT_1]. Reference [ID_NUMBER_1].
```

For a UTF-8 file, use `--input ./note.txt` in place of `--text`. The command
prints the redacted text by default. Add `--output ./redacted.txt` to write it
to a file.

For a support handoff that keeps dates and case references visible, select
`CONTACT` and write a separate copy. Edit the source ticket and rerun the
command when the request changes. Review the resulting file before sharing it.

```sh
anno privacy --model pattern --action redact --types CONTACT \
  --input ticket.txt --output handoff.txt
```

## CLI: relation extraction

Extract typed `(head, relation, tail)` triples with a `RelationCapable` backend:

```bash
# tplinker: heuristic baseline, no extra deps
anno extract --extract-relations --model tplinker \
  --text "Steve Jobs founded Apple in 1976."

# gliner_multitask: zero-shot entity types + heuristic relations (requires onnx)
anno extract --extract-relations --model gliner-multitask \
  --relation-types "FOUNDED,WORKS_FOR" \
  --text "Steve Jobs founded Apple in 1976."
```

## CLI: coreference

Group repeated mentions in a single document. With the smaller CLI build,
the output includes a coreference track for the two mentions of “Curie”.

```bash
ANNO_NO_DOWNLOADS=1 anno debug --model heuristic --coref -t "Marie Curie moved to Paris. Marie Curie later returned."
```

## Visualize evaluation results

To see what an extraction got wrong, pipe its evaluation into the errors view.
This example runs offline and produces one wrong entity type and one missed name.

```sh
anno eval --model pattern --text 'Mail x@y.com to Ada.' \
  --gold 'x@y.com:ORG:5:12' --gold 'Ada:PER:16:19' --json | anno viz errors -o errors.svg
```

Use `--metric char-masking` on `anno eval` to also see Unicode
character coverage. An errors input may also contain a whitespace-separated
stream of reports, including JSONL. It pools their counts before computing
percentages and counts a repeated input once per report. Save a compact,
replayable summary with `anno viz errors reports.jsonl --title 'Validation errors' -o summary.json`,
then use `anno viz render summary.json -o errors.svg` to redraw it from any directory.
Current native reports break errors down by gold label and record whether a
partial boundary match also had the wrong type. A pooled report needs that
detail in every input before it can show the per-type result.

To decide whether a change helps, compare a baseline and candidate on the same
documents. The following example scores two offline backends on identical gold
mentions and writes a self-contained comparison you can open in a browser.

```sh
for model in pattern heuristic; do
  anno eval --model "$model" --text 'Mail x@y.com to Ada.' --pair-id mail-1 \
    --gold 'x@y.com:EMAIL:5:12' --gold 'Ada:PER:16:19' --json > "$model.jsonl"
done
anno viz paired pattern.jsonl --against heuristic.jsonl \
  --baseline-name Pattern --candidate-name Heuristic --cohort 'Mail fixture' \
  --role 'Worked example' -o comparison.html
```

Each JSONL stream may contain multiple documents in any order. Current
`anno eval --json` records a source-text digest and a fingerprint for each
typed gold span. A stable occurrence ID distinguishes repeated copies of the
same text. Use `--pair-id` on both runs when an input lacks such an ID.
The comparison checks the document and reference sets before counting gained,
lost and retained exact matches. To compare older evaluations, replay their
saved predictions and gold through the current evaluator.

The paired view shows all suggestions in the scoring inventory, including
correct ones. It also shows component calls and median latency when measured.
A cohort can supply a signed uncertainty interval with its method. Inferring
a sampling interval requires paired outcomes at the resampling unit. One
document can simultaneously gain and lose a gold match.
[Fixed-budget CrossRE example](eval/crossre-predicate-ranking.md) · [WikiGold routing costs](eval/wikigold-nuner-routing.md)

For a whole NER dataset, use an eval-enabled build and write one report for
every loaded sentence, including sentences with no gold mentions. With WikiGold
cached, the following runs cover the full input twice. The HTML comparison
checks that both runs evaluated the same occurrences against the same gold.

```sh
anno dataset eval --dataset WikiGold --task ner --model pattern --reports pattern.jsonl
anno dataset eval --dataset WikiGold --task ner --model heuristic --reports heuristic.jsonl
anno viz paired pattern.jsonl --against heuristic.jsonl --cohort WikiGold --unit sentences \
  --baseline-name Pattern --candidate-name Heuristic -o wikigold.html
anno viz errors heuristic.jsonl -o wikigold-errors.svg
```

On the pinned WikiGold release this example pairs 1,696 sentences, including
355 with no gold mentions. The heuristic scores 1,394 exact matches and the
pattern backend scores zero against WikiGold's typed gold spans. Pattern is a
useful control here. Its structured identifiers fall outside much of this label
inventory. Per-case timings cover extraction on the machine running the command.

For ordinary extraction output, `anno viz overlap documents.jsonl -o overlap.svg`
measures how many mentions share a character and which labels are affected.
It reads grounded documents directly and works without a gold reference.
[Grounded extraction example and full GENIA replay](EXTRACTION_WORKFLOWS.md#review-overlapping-highlights)

For a sweep over several operating points, plot correct matches against total
suggestions with `--x-axis suggestions`. The default horizontal axis counts
unmatched predictions after evaluation. A filled point yields the largest
observed correct count at or below its horizontal budget. Axes use rates per
100 input units. The named settings give exact counts and recall.

```sh
anno viz tradeoff docs/eval/crossre-full-test-tradeoff.json --x-axis suggestions -o queue.html
```

To supply your own study, write `anno.tradeoff/v1` JSON with one series per
cohort. Each series holds settings scored on the same observations and gold
items. `observations` includes inputs with zero gold. `correct` and
`unmatched` count matched and unmatched predictions under your scoring rule.
Recall is `correct / gold` and appears as a dash when gold is zero. For example,
these illustrative counts compare two settings on ten sentences.

```json
{
  "schema": "anno.tradeoff/v1",
  "title": "What does the fallback recover?",
  "unit": "sentences",
  "series": [{
    "label": "Review sample", "observations": 10, "gold": 8,
    "points": [
      {"label": "Stock", "correct": 4, "unmatched": 1},
      {"label": "Fallback", "correct": 6, "unmatched": 3}
    ]
  }]
}
```

All four views accept a JSON file or stdin and write standalone SVG to stdout by
default. A `.html` output file or `--format html` produces a self-contained
page with an accessible data table. A `.json` output file or `--format json`
writes a reusable count summary (`anno.paired/v1`, `anno.tradeoff/v1`,
`anno.errors/v1` or `anno.overlap/v1`). Rust callers can use `paired::Paired`,
`Tradeoff`, `errors::EvalErrors` or `overlap::Overlap` from `anno::viz`. Each exposes `from_json`, `render_svg`
and `render_html`. Rendering runs offline in the standard CLI.

### Browse the published studies

Run these commands in an Anno checkout or unpacked `dataset-figures` CI artifact.
Each includes a dated catalog of the studies and registry evidence in that build.

```sh
anno dataset evidence WikiGold --status scored
anno viz list --dataset WikiGold
anno viz render wikigold-routing-paired -o routing.html
```

Use `anno dataset evidence --status unindexed` to find gaps, or add `--backend`,
`--task` and `--split` to require a match on the same recorded run. Add `--json`
for sample sizes and source receipts. A loader check establishes that an input
opened. A scored record identifies the backend and measured outcome.

To select a particular research snapshot, pass `--catalog PATH` or set
`ANNO_FIGURE_CATALOG=/path/to/figure-catalog.json`. The catalog and the
installed Anno version can be updated independently. [Figure generators and CI downloads](figures/README.md)

## CLI: batch processing

Process a directory of `.txt` or `.md` files:

```bash
# Sequential, write per-document JSON to results/
anno batch --model heuristic --dir docs/ --output results/

# 4 parallel workers; reuse eligible cached results
anno batch --model heuristic --dir docs/ --parallel 4 --cache --output results/

# Stream JSONL from stdin: {"id":"…","text":"…"} per line
cat corpus.jsonl | anno batch --model heuristic --stdin --parallel 4 --cache --output results/
```

Only deterministic backends use the result cache: `pattern`, `heuristic`,
`minimal`, and `heuristic-fr`. Runs using model weights, remote services, or
coreference bypass it. Entries live under `{cache_dir}/results/v2/`; the cache
key includes the document, CLI build, and extraction settings.
`anno cache clear --confirm` deletes the entire CLI cache directory, including
any models and batch results stored there.

## CLI: cross-document clustering

Cluster entities across multiple documents (requires `--features eval`):

```bash
anno cross-doc ./docs --threshold 0.6 --format tree
```

## Library usage

```rust
use anno::{Model, StackedNER};

let ner = StackedNER::default();
let entities = ner.extract_entities("Lynn Conway worked at IBM and Xerox PARC.", None)?;

for e in entities {
    println!("{} [{}..{}] {:?}", e.text, e.start(), e.end(), e.entity_type);
}
```

## Offsets

Offsets count characters (Unicode scalar values), not UTF-8 bytes. See the
[span and offset contract](CONTRACT.md).

## Next

- [Contract](CONTRACT.md): scope, offset guarantees, feature flags
- [Backends](BACKENDS.md): model selection and hardware requirements
