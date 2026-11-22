# Extraction workflows

This page separates a deterministic local check from an evaluation that can
support a deployment decision.

## Offline deterministic profile

Run the built-in profile without downloading a model or dataset:

```sh
cargo run -p anno-eval --example offline_extraction_profile
```

It compares `RegexNER`, `HeuristicNER`, and an explicit pattern-plus-heuristic stack on
`anno_eval::synthetic::news_dataset()`. The profile reports strict,
micro-averaged precision, recall, and F1: a prediction earns credit only when
both its character span and entity label exactly match a gold annotation.

It reports two latency values per backend:

- **first pass** includes the first traversal of the fixture and any lazy
  initialization it triggers;
- **warm pass average** is the mean of ten subsequent full-fixture passes on
  the same model instance.

The comparison is useful as a reproducible regression signal. It is not a
model benchmark or a production-quality claim: the fixture has eight short,
synthetic news examples and 23 authored annotations. Its labels and phrasing
are part of the repository, so they cannot establish performance on an unseen
or domain-specific corpus. Record the command, commit, host, Rust version, and
complete stdout with any result used for a decision.

The stack in this profile is constructed with explicit regex and heuristic
layers. It does not download or select an ML model. Test an ML-backed
stack separately and record the exact model artifact, configuration, runtime,
and whether every requested backend actually ran.

The `Offline extraction benchmark` GitHub Actions workflow runs the same
profile on a pinned WikiGold corpus. To repeat it locally, pass the downloaded
`wikigold.conll` file as the example's sole argument. For each gold mention the
profile checks whether either layer proposed its span and type and whether the
stack returned it. The four counts distinguish retained, lost, gained, and
missed mentions. The last line counts output surfaces that disagree with the
original text at their reported offsets.

## Application workflow

For an application, instantiate one model, retain it for the process lifetime,
and preserve the returned character offsets with the source text. A compact
starting point is [the batch example](../crates/anno/examples/batch.rs): it
keeps one extractor instance and returns one result per input document.

For identified documents, `annotate_grounded_with` and
`annotate_grounded_batch_with` return `GroundedDocument` values, preserving
entity metadata and canonical grouping. Ungrouped entities get separate
singleton tracks. [The grounded example](../crates/anno/examples/grounded.rs)
demonstrates reuse of a local model. The CLI uses the signal-only conversion
`GroundedDocument::from_entity_signals`, leaving track creation to its explicit
coreference step.

`anno batch --batch-size 32` bounds each call to the backend's batch API.
`--parallel` bounds concurrent extraction chunks and subsequent enrichment workers.
Result caching is limited
to deterministic local backends; model-backed and coreference runs bypass it
until their effective artifact/configuration identity can be established.
Eligible cache keys also include a digest of the executable, so rebuilding the
same package version does not reuse results from different extraction code.

For stacked extraction, select a failure policy explicitly:

```rust
use anno::{StackedExtractionPolicy, StackedNER};

let model = StackedNER::builder()
    .layer(anno::RegexNER::new())
    .layer(anno::HeuristicNER::new())
    .build();
match model.extract_entities_with_report(
    "Grace Hopper developed COBOL.",
    None,
    StackedExtractionPolicy::Strict,
) {
    Ok(report) => println!("{} entities", report.entities().len()),
    Err(error) => {
        // Rejected runs retain the successful, failed, and skipped layer outcomes.
        eprintln!("{error}: {:?}", error.report().layer_outcomes());
    }
}
```

`BestEffort` permits partial results and exposes failures in the report. `Strict`
rejects a run if an attempted layer fails. Adaptive skips are recorded separately.
These outcomes describe inference on an already constructed stack; they are not
an artifact fingerprint or a record of model-construction fallback attempts.

Before using results for routing, redaction, or a knowledge graph, add a held
out set annotated to the application's label policy. Score exact span-and-label
P/R/F1 first, then inspect false positives and misses by entity type and source
format. Measure first and warm latency separately on representative document
lengths. If the workflow permits partial results, record which backends
succeeded and failed; otherwise make a failed backend a hard error.

The CLI fixture documents under `crates/anno-cli/tests/fixtures/` are useful
for regression assertions such as required mentions and minimum counts, but
they do not contain exhaustive gold annotations. Do not compute precision from
them.

## Carrying source revisions through brat and dataset JSONL

Keep one directory per source revision. From a common corpus root, export the
grounded document and import the resulting annotation file.

```sh
anno export --input v1/report.json --output v1/brat --format brat --include-confidence
anno dataset import --input v1/brat/report.ann --output v1/rows.jsonl --format brat --include-text
sha256sum v1/brat/report.txt
```

The companion `report.txt` contains the grounded document’s source text.
Keep the SHA-256 digest of those exact UTF-8 bytes with your document ID,
revision ID and annotation path in the corpus manifest. The digest distinguishes
text changes, including changes in Unicode normalization. Two documents can
contain the same text, and an annotation edit can leave the text digest unchanged.
Version the annotation file separately when tracking annotation edits.

Each imported row’s `source` records the annotation path supplied to the
importer. Running from the same root retains the revision directory in that
path. brat’s `T1`, `T2` identifiers are local to the annotation file.
The grounded document ID, signal IDs and local brat IDs are absent from these
JSONL rows. Use the manifest to recover document identity after moving files.

For a mention that skips intervening words, `segments` lists its half-open
Unicode character ranges. `start` and `end` bound the whole mention, and
`text` joins the selected pieces with spaces. Use the individual ranges when
repeated mentions have the same surface. Offsets count Unicode code points,
so a decomposed accent and its base character occupy two positions.

## CoNLL offsets and the original source revision

By default, `anno dataset import --format conll` counts Unicode characters in
the token sequence reconstructed with one ASCII space between tokens. Blank
lines and `-DOCSTART-` reset entity boundaries without adding characters.
This convention retains offsets for existing token-only corpora.

When original document offsets matter, supply the matching UTF-8 text for a
single CoNLL file.

```sh
anno dataset import --format conll --input v1/report.conll \
  --source-text v1/report.txt --output v1/rows.jsonl
```

Every token, including tokens tagged `O`, must match the next source text
after any intervening Unicode whitespace. Adjacent tokens may touch in the
original. The importer checks the whole document and preserves tabs, multiple
spaces and line endings inside entities. The output ranges count Unicode
scalar values from the start of the original file, with an exclusive end.
CRLF counts as two characters. Keep the text revision and its digest alongside
the CoNLL file in the corpus manifest. The row’s `source` identifies the CoNLL file.

## Identifying RDF source documents

Provide the IRI of the exact source revision when exporting one file.

```sh
anno export --input report.txt --output rdf --format ntriples \
  --document-iri https://example.org/reports/42/revisions/3
```

N-Triples, JSON-LD and graph N-Triples share the same mention IDs. By default
the CLI hashes the canonical input path and extracted text to identify the
source revision. Moving a file or changing its text changes that identity.
An explicit versioned IRI lets different machines identify the same revision.

To migrate stored RDF from the earlier index-based IDs, re-export each original
document with its annotations in their original order using the earlier release
and this release. Within each document, use the index in each old mention ID
to find the original annotation and its new ID. Several annotations can have
the same type and outer span. A previously merged graph can contain several
occurrences under one old ID. Rebuild from the originals to recover them.
An external reference that recorded only the old ID can remain ambiguous.
