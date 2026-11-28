# Extraction workflows

This page separates a deterministic local check from an evaluation that can
support a deployment decision.

## Saving and converting annotations

For a BRAT annotation project, extract directly into a directory. To keep a
reviewable intermediate document, save grounded JSON. Subsequent exports read
the saved text, spans and recorded relations without loading a backend.

```sh
anno extract --model pattern --file notice.txt --format brat --export annotations
anno extract --model pattern --file notice.txt --format grounded > reviewed.json
anno export --input reviewed.json --output graph --format graph-csv
anno pipeline --model pattern --files notice.txt --format jsonl --output documents.jsonl
anno export --input documents.jsonl --output annotations-from-pipeline --format brat
```

To export an entire tree, save one grounded document per input file:

```sh
anno batch --dir docs/examples/computing-history --model heuristic --format grounded --output saved
anno export --input saved --format brat --output annotations
```

Open `annotations/processors/arm.txt` alongside `arm.ann` in the same
directory. The note credits Sophie Wilson and Steve Furber. The annotation
file locates their names at character ranges `0..13` and `18..30`.

Compact extraction JSON omits the complete source text. Grounded JSON retains
it, so conversion can reproduce whitespace and character offsets. Pipeline
JSONL contains one grounded document per line in document mode. With multiple
saved documents in one file, export numbers artifacts in input order starting
at one. Graph CSV preserves an empty recorded edge set. Exporting raw text can
generate co-occurrence edges when extraction yields no semantic relations.

### Review overlapping highlights

**Use JSONL alongside flat HTML when reviewing nested annotations.** A flat highlight can hide an entire gold mention. In GENIA test sentence 600, two mentions start at character 67. The native HTML exporter highlights **cyclophilin gene** once as DNA, covering `[67, 83)`. JSONL contains that DNA interval and the nested protein **cyclophilin** at `[67, 78)`. Both exports preserve a separate DNA mention later in the sentence. [Exported example and source fingerprint](eval/genia-native-exports.json)

![In test sentence 600, HTML marks cyclophilin gene as DNA. JSONL contains both that DNA span and the nested protein span cyclophilin. Across all splits, HTML displays 50,709 of 55,740 gold mentions and JSONL preserves every mention. The split rows give hidden mentions and affected sentences.](figures/anno-genia-nesting.png)

On **18,546 sentences** from all three GENIA splits, the HTML exporter hid **5,031 of 55,740 mentions (9.0%)**. At least one mention disappeared from **3,816 sentences (20.6%)**. The replay checked every visible HTML interval and every JSONL row against four published gold annotation layers, including the 1,983 sentences without gold mentions. [Native replay and split counts](eval/genia-native-exports.json)

To measure how often spans compete for a highlight in your own data, pipe grounded extraction output into the overlap view. Save the aggregate counts to compare labels or reopen them in a browser.

```sh
anno extract --model pattern --format grounded --text 'Mail Ada at ada@example.org.' \
  | anno viz overlap -o overlap.svg
anno viz overlap documents.jsonl --title 'Support ticket extractions' -o overlap-counts.json
anno viz overlap overlap-counts.json -o overlap.html
```

In GENIA, **9,652 of 55,740 mentions (17.3%)** shared a character with another mention. Those overlaps covered **47,678 of 3,103,238 character positions (1.54%)**. Sampling by mention will expose more overlapping cases than sampling random character offsets. Within the 47,678 shared positions, **855 (1.79%)** lay under at least three mentions. Protein supplies the most overlapping mentions (5,738 of 34,097). RNA has the highest affected share (418 of 933, or 44.8%). [Overlap counts](eval/genia-span-overlap.json) · [Grounded replay](../scripts/genia_grounded_overlap.py)

![GENIA overlap by mention, sentence and character. A depth table shows 855 positions with at least three mentions. A separate bar and percentages use all 47,678 shared positions as the denominator. RNA leads the label table by affected share and protein has the largest affected count.](figures/genia-span-overlap.svg)

Character counts use Unicode scalar positions and include whitespace. Touching spans share zero characters. Gaps between segments of a discontinuous signal keep their own coverage. The SVG lists up to eight labels by affected share. HTML gives complete rankings by share and count. Saved summaries contain aggregate counts, label names and the chart title. [CLI examples](QUICKSTART.md#visualize-evaluation-results)

<details>
<summary>Token-layer depths and common nested label pairs</summary>

Of **148,148 annotated tokens**, 136,462 belonged to one active layer, 11,447 to two and 239 to three. These denominators count publisher tokens. The overlap view above counts character positions in reconstructed text.

| Outer label | Inner label | Strict containment pairs |
|---|---|---:|
| protein | protein | 1,863 |
| DNA | protein | 1,533 |
| DNA | DNA | 383 |
| RNA | protein | 360 |
| cell_type | cell_type | 316 |
| cell_line | protein | 195 |
| protein | DNA | 89 |

The table pools all three splits. On test alone, DNA containing protein led with 190 pairs. A mention can participate in multiple pairs. [Complete split and label counts](eval/genia-nesting-counts.json)

</details>

To replay the exports, build the example and run the figure script from the repository root.

```sh
cargo build --locked -p anno-eval --example genia_export
python3 scripts/plot_genia_nesting.py target/genia-review \
  --native-export "${CARGO_TARGET_DIR:-target}/debug/examples/genia_export"
python3 scripts/genia_grounded_overlap.py target/genia-review/genia-span-overlap.json
anno viz overlap target/genia-review/genia-span-overlap.json -o target/genia-review/genia-span-overlap.svg
```

The script checks all three publisher files against their SHA-256 digests. The Rust example reads each of the four IOB2 layers with `AnnotatedSentence::entities()` and passes the original labels to both exporters. Character offsets refer to a sentence reconstructed with one space between tokens. An independent decoder checks the entire text recovered from the HTML, the span and label of every highlight, and each JSONL row. [Replay example](../crates/anno-eval/examples/genia_export.rs) · [Figure and checks](../scripts/plot_genia_nesting.py)

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

## Caption playback links

A caption editor can use FCoref to suggest which earlier mention a pronoun refers to. The adapter returns the original cue IDs, character ranges, and playback times in milliseconds. [Run the caption example](../crates/anno/examples/caption_links/README.md)

## Migrating RDF relation labels

N-Triples, JSON-LD and graph N-Triples preserve letters, numbers, underscores
and hyphens in relation IRIs, and percent-encode other characters as UTF-8.
For example, `WORKS_AT` keeps `/rel/WORKS_AT`, `part:of` becomes
`/rel/part%3Aof`, and literal `part%3Aof` becomes `/rel/part%253Aof`.

Older releases replaced punctuation with underscores, so `part_of` and
`part:of` both used `/rel/part_of`. To migrate, re-export the original labeled
relations and update stored RDF and queries using those labels. If both labels
occurred, the old predicate alone cannot identify which original label a
particular edge used.

For a discontinuous mention all three RDF writers add a `vocab#sourceSegments`
literal. Its ordered, half-open Unicode character offsets use BRAT notation,
such as `2 5;10 13`. The start and end offsets still describe the envelope.
The dataset importer checks the segment bounds and preserves them in JSONL.
It also reads segments from Anno’s older v2 occurrence IRIs when the literal
is absent. Contiguous mentions need only the start and end offsets.

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

## Comparing contract passage suggestions

Use the CUAD retrieval example's `matched_passages_per_question` field to
compare BM25 and literal category lookup at equal suggestion volume. For each
question both methods return as many windows as the shorter ranking. A hit
requires a window to contain the complete annotated answer. Each annotation
counts once even when windows overlap.
[Runnable example](../crates/anno-eval/examples/cuad_retrieval.rs) · [Pinned CUAD release](https://github.com/TheAtticusProject/cuad/tree/67faa0e6023b04fcaae6cc09497ab00e5d63a2a2)

Across Governing Law, Termination For Convenience and Non-Compete on the 102
test contracts, the unrestricted top-three lists recovered 100 answers with
867 BM25 windows and 43 with 136 literal windows. At equal volume, BM25
recovered 45 of 190 answers and literal lookup recovered 43, each in 136
windows. Summing across the three categories per contract, BM25 gained one
answer on six contracts and two on one, lost one on six and tied on 89.
An exact two-sided sign test on the 13 non-ties gave *p* = 1.00. This sample
provides little evidence that switching rankers improves recall at equal
volume. The counts measure machine suggestions. [Replay and category results](https://github.com/arclabs561/anno/pull/217)

## Carrying source revisions through brat and dataset JSONL

Keep one directory per source revision. From a common corpus root, export the
grounded document and import the resulting annotation file.

```sh
anno export --input v1/report.json --output v1/brat --format brat --include-confidence
anno dataset import --input v1/brat/report.ann --output v1/rows.jsonl --format brat --include-text --require-brat-manifest
```

Anno writes `report.anno-pair` with the SHA-256 digests of the annotation
and the exact UTF-8 source bytes. Keep it with `report.ann` and `report.txt`.
All three can move or take a new common stem. Import checks a present manifest
against the files it actually reads, including an explicit `--source-text`.
A changed file invalidates that pairing. Preserve the original trio when editing
annotations and track the edited pair as a new revision.

Legacy BRAT input without a manifest remains readable. Normal progress output
marks the revision unverified. Use `--require-brat-manifest` when every input
must have verified digests. Keep the publisher's document and revision IDs in
your corpus manifest because distinct documents can contain identical bytes.

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

For one brat file, use an explicit original when checking an annotation before import.

```sh
anno dataset import --format brat --input reviewed.ann \
  --source-text editions/original.txt --output checked.jsonl
```

The importer checks supplied reference text and Unicode character ranges against
that document, including discontinuous annotations. An empty reference field
takes its surface from the document. For review of an exact edition, retain the
digests of both the document and the annotation file. A span can still match
after the surrounding passage or the annotation label changes.

## Verify mention JSONL against a source document

For one document and one entity per JSONL row, supply the exact UTF-8 text used
during extraction. The command checks every surface against its half-open
Unicode character offsets and preserves the supplied type label.

```sh
anno dataset import --format jsonl --input mentions.jsonl \
  --source-text post.txt --output checked.jsonl
```

JSONL export preserves discontinuous character ranges. Segmented rows may use space-joined pieces. When the gaps contain only line
breaks, the original multiline surface also passes. The row's `source` field
is an optional producer string. The optional `confidence` field accepts a number
or null. The importer validates mention rows against the file named by
`--source-text`. Additional row fields, including document IDs and hashes, are
ignored. A changed passage outside all mentions can still pass, so verify the
document ID and digest of the original bytes against your corpus manifest.

A successfully processed document with no predictions has an empty JSONL file.
Record its completion and zero mention count against the same ID and source
digest when scoring. A missing or failed run has no measured count.

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

## Trace a signal to its original HTML

A page displays `alice@example.org` twice. Its first address contains an
inline bold tag and spells the at sign `&#64;`. Its second uses a literal
`@`. Searching the rendered text gives you the same address both times.

With the checkout CLI, run this in a scratch directory to select each at
sign in the cleaned text and locate it in the saved HTML:

```sh
printf '<p>🦊 al<b>ice</b>&#64;example.org</p><p>alice@example.org</p>' > page.html
anno extract --file page.html --model pattern --html-source-spans --clean --format grounded > doc.json
anno query source-span doc.json 7 8 --original page.html
anno query source-span doc.json 25 26 --original page.html
```

| Cleaned text (characters) | Saved HTML (bytes) | Source spelling |
| --- | --- | --- |
| `7..8` | `20..25` | `&#64;` |
| `25..26` | `48..49` | `@` |

Check the returned ranges directly against the file:

```python
from pathlib import Path

source = Path("page.html").read_bytes()
assert source[20:25] == b"&#64;"
assert source[48:49] == b"@"
```

### Use the map on your own page

`anno extract --file page.html --model pattern --html-source-spans` renders a
local UTF-8 HTML file with deformat's mapped strip renderer and writes grounded
JSON. Use `--export result.json` to save it. The renderer choice can change which
page content the model sees. For a fetched page, save the UTF-8 HTML you want to
identify and pass that file to `--file`. Its digest and offsets refer to the
saved bytes.

Use `anno query source-span result.json START END --original page.html` to project
a half-open character range in the analyzed text to the original file. This
also works when `--clean` or `--normalize` changes the rendered text. The
grounded `source_map` composes those steps. Preserve individual fragments
when inline markup interrupts a mention.

When the analyzed text still matches the mapped renderer output, the legacy
`html_source` also records SHA-256 digests of the source file and
rendered `text`. Match a signal's half-open character range to a `spans` entry's
`rendered_chars`. Discontinuous mentions have a separate entry for each component.
Each fragment identifies UTF-8 byte ranges in both the rendered text and the
original file. An inline tag can split one mention into several fragments.
`direct` means byte-for-byte correspondence. `entity_decoded` points to the
complete character reference, such as `&#64;`. `whitespace_collapsed`,
`synthetic`, and `unknown` identify an approximate origin, which can include
an entire text run or tag. For a gap in the renderer map, `unmapped` has a rendered byte
range and `null` source bytes. Check the source digest before using these byte ranges. Direct
fragments can be compared byte for byte with the rendered text.
