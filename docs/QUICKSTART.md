# Quickstart

Extract entities with the CLI or a reusable Rust model. The examples below
start with offline heuristics; model downloads and evaluation are optional.

## Install

Run `cargo add anno` for the current published release. To target the 0.16 release line, use this dependency.

```toml
[dependencies]
anno = "0.16"
```

Install the CLI:

```bash
cargo install anno-cli --locked
```

For the English heuristic and pattern examples, this smaller build includes
the Readability HTML extractor and skips the ONNX runtime.

```bash
cargo install anno-cli --locked --no-default-features --features extractor-readability
```

Single-text scoring with `anno eval` works in the standard CLI. Add
`--features eval` for dataset benchmarks or cross-document clustering.
To build from a local clone with those features:

```bash
cargo install --path crates/anno-cli --bin anno --features "onnx eval"
```

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
