# Contract

This contract describes the input text, offsets, and feature flags that callers
must account for when using `anno`. Start with the [quickstart](QUICKSTART.md)
for runnable examples.

## Source text and offsets

Anno consumes plain UTF-8 text. Entity offsets are half-open character ranges
`[start, end)` counted in Unicode scalar values. In `🦊 Email ada@example.org`,
the email spans characters `8..23` and UTF-8 bytes `11..26`. Rust string slicing
uses byte indices. Convert at that boundary using `anno::offset::SpanConverter`.
Use `CharOffset` and `ByteOffset` when passing positions between Rust components.

Keep the document ID, exact text revision and coordinate convention with saved
annotations. When OCR cleanup, newline normalization, `--clean` or `--normalize`
changes the input, the positions refer to the processed text. To highlight the
original, keep the mapping back to that revision. Grounded JSON stores the text
with its signals and tracks. [Runnable example](QUICKSTART.md#save-annotations-with-the-source-text)

## Library and CLI features

| Need | Enable or use |
|---|---|
| Offline pattern and heuristic extraction | `anno` with `default-features = false` |
| ONNX backends and French heuristics | Default `anno` features |
| Candle inference | `anno/candle` |
| Within-document and cross-context scoring primitives | `anno/analysis` exposes `anno::metrics` |
| Graph construction and export | `anno/graph` |
| Discourse utilities | `anno/discourse` |
| Compare predictions with gold spans for one text | `anno eval` in the standard CLI |
| Dataset benchmarks and cross-document clustering | Install `anno-cli` with `--features eval` |

Use `Entity` for extraction results and `GroundedDocument` for text, signals,
tracks and identities. For custom entity types use `extract_with_types` or CLI
`--extract-types`. Relation-capable backends return typed triples. The graph
feature exposes `GraphDocument` and N-Triples export. Keep training, document parsing and downstream graph algorithms in their
respective tools. [Backend selection](BACKENDS.md) · [Constituency interchange](CONSTITUENCY.md)

## Grouping flat entities

For reviewed catalog writes, `Corpus::preview_kb_assignment` identifies the track bindings in scope. `Corpus::apply_kb_assignment` checks the preview's owner and revision before applying it. [Executable verification contract](../verification/reviewed_kb/README.md)

`GroundedDocument::from_entities` groups mentions with equal canonical IDs and
gives each unassigned mention a singleton track. Track IDs are allocated in
first-seen order. The first populated KB ID in each group becomes its link.
`to_entities()` gives every member the track ID and the shared KB link.

Use `GroundedDocument::try_from_entities` to detect conflicting populated KB IDs,
zero mentions without a nonempty source span, visual spans, and flat offsets that would change when
reconstructed from discontinuous segments. Each signal preserves its producer’s
linguistic mention type across grouping and JSON. A name and a pronoun can share
a track while retaining different mention types. Older documents omit
`mention_type` and read as an unspecified form. An explicit `"unknown"` survives
the round trip.
Errors identify the original input indices. A populated link and a missing link
in the same group are compatible. Both constructors use the same grouping and
preserve the detection confidence and explicit custom entity categories. Entity
types and grounded labels normally serialize as strings. A custom spelling or
category that would be lost uses the tagged form, for example
`{"Custom":{"name":"DISEASE","category":"Agent"}}`. Existing string labels remain readable.

## Grounded relation evidence

A mention-level relation keeps its original directed endpoints, predicate, trigger
and extraction score. Call `record_relation` on the source document, or use
`record_relation_by_signal_ids` when the producer selected known signals.
Coincident signals with the same geometry, label and surface require the
producer's IDs. Both calls accept provenance for the relation decision.

Grounded JSON saves these observations in `relation_evidence.mentions`. The
collection records the document ID and SHA-256 of its exact UTF-8 text.
`project_recorded_relations` looks up the current track for each original
signal, including after regrouping and reloading. An untracked signal still
supplies a source highlight. Raw edits that invalidate an occurrence make
projection and serialization fail. A file without `relation_evidence` reads
with empty evidence collections.

Use `record_concept_relation` for source annotations whose endpoints are
concept IDs. A concept can have a supplied set of source signals, an explicitly
empty set, or unspecified membership. Grounded JSON preserves these distinctions
in `relation_evidence.concepts`. An endpoint projects onto a track only when the
complete, nonempty member set exactly matches that track. A split or merge can
change this projection. The concept edge persists without adding a mention pair
or a score to a gold annotation.

Given a DWIE annotation with its reconstructed article in `content`, these
commands import and reopen the source concepts and their directed edges.

```sh
cargo run -p anno --no-default-features --example import_dwie_relations -- article.json > grounded.json
anno query relations grounded.json
```

The example uses the supplied mention’s presence as a binary signal score of
1.0 and labels it `mention`. DWIE's multi-label concept taxonomy and candidate
link scores are outside this relation import. Concept-edge confidences are
absent. Query output includes the source text digest, original endpoint
observations, and current projected track IDs. On DWIE article `DW_1095032`,
the import produces 37 source signals, 14 tracks and five source concept edges.
The edge `(7, gpe0, 14)` survives reloading with an explicitly empty member set
and no track for concept 14. [Original DWIE annotation](https://github.com/klimzaporojets/DWIE/blob/master/data/annos/DW_1095032.json)

## PII pattern limits

Pattern-based redaction recognizes these formats:

| Format | Validation |
| --- | --- |
| Email addresses | Pattern match |
| US phone numbers and addresses | Pattern match |
| US Social Security numbers | Structure check |
| Card numbers with 16 digits | Luhn checksum |
| IBANs | Mod-97 checksum |

Other national ID formats are not covered.

Printed IBANs use uppercase groups of four separated by single ASCII spaces,
with a shorter final group when needed. Recognition stops at the registered
country length. Text after a space at that endpoint is separate, including
numeric text. Compact IBANs use the existing compact-format recognizer.
These checks validate format and checksum, not account existence or
country-specific bank and account rules.

## CLI default

With `--model stacked`, the CLI tries available model backends before falling
back to patterns and heuristics.

- By default, model downloads are allowed (so first run may be slower).
- To force cached-only / offline behavior: set `ANNO_NO_DOWNLOADS=1` (or `HF_HUB_OFFLINE=1`).
- To prefetch explicitly: use the **full CLI** (`anno-cli`): `anno models download gliner gliner_multitask bert-onnx` (then `stacked` will pick it up).

## CLI JSON provenance

`anno extract --format json` and `--format jsonl` retain the requested CLI backend in
top-level `provenance.model`. Each emitted entity that carries library provenance also has a
nested `provenance` object with its actual `source`, extraction `method`, and available
pattern, raw-confidence, model-version, and timestamp metadata.

With `--detect-lang`, a fetched HTML page's recognized language declaration
supplies `provenance.language` (its primary language code) and
`provenance.language_source = "html_declaration"`. Other inputs use
`"text_heuristic"`, which estimates language from Unicode scripts and can
identify Latin-script languages as English. These fields describe the input.

Top-level `provenance.entity_sources`, when present, is a sorted unique list of source names
from the entities actually emitted after filtering and validation. It is omitted when no emitted
entity has source metadata. It does not infer a single producing model for stacked output, a
model artifact version, or provider/device placement.

`anno pipeline --coref --format json` and `--format grounded` record the successful
resolver in each document’s optional `coref_backend` field. `fcoref` identifies the
neural model and `mention-ranking` the heuristic fallback. Successful runs record
the backend even when they produce zero tracks. Documents without a recorded
successful coreference run omit this field.

The optional `coref_run` object describes the latest reported invocation. A
`completed` result includes the producing `backend` and `tracks_added`, even
when that count is zero. On `failed` results, `last_failure` identifies the
terminal backend and stage. Either outcome can include `prior_failures` in
attempt order. Each failure identifies a backend and the stage (`load`,
`inference`, or `invalid_output`). A failed retry preserves the earlier
successful `coref_backend`. Older documents can have a producer without an
attempt report.

### Speaker-attributed first person

Build the CLI with `--features discourse` to use `anno pipeline --coref --speaker-turns turns.json --files transcript.txt --format json`.
Use one transcript per invocation. For example, the text `Éva: I took my bag.` takes this sidecar:

```json
[{"speaker": "channel-A", "text": "I took my bag.", "start": 5, "end": 19}]
```

Offsets count Unicode scalar values in the original text, with an exclusive end. Turns must be in order and disjoint. Leave turns with unknown speakers out of the array. For a recorder that supplies audio times, compute text offsets as you assemble the transcript and keep the audio times separately.

Providing this sidecar selects a speaker-only resolver. General name and third-person chains from the ordinary resolver are omitted. The resolver records `speaker-turns` in `coref_backend` and uses document-local speaker IDs as track names. It groups English `I`, `me`, `my`, `mine`, and `myself` when at least two mentions belong to one speaker. A missing turn language defaults to English. Turns bearing quotation marks and turns marked with another language are skipped. The resolver leaves second-person references unresolved. Its output depends on the accuracy of the supplied speaker annotations.

## CoNLL BIO export

`anno export --format conll` and `anno::export::to_conll` split the source at
contiguous mention boundaries. The CLI rejects discontinuous mentions and
supports brat output for preserving their separate segments. For example, a pair of parentheses around an email address
gets separate `O` rows, and the address gets a `B-EMAIL` row. BIO carries one
label per token. When mentions overlap, the exporter keeps the earliest
starting mention, preferring the longer span for a tied start.

These rows depend on the supplied mentions. For evaluation on a pretokenized
corpus, align predictions to the corpus's token rows and sentence boundaries
before scoring. The shared-task definition assigns `B` to the first word of
each entity and `I` to its remaining words.
[CoNLL-2002 shared task, section 2](https://aclanthology.org/W02-2024.pdf)

## Evaluation (two layers)

`anno` has two eval layers with very different runtimes:

- **Single-text eval** (`anno eval`): inline gold spans against one text; typically milliseconds–seconds.
- **Benchmark matrix** (`anno benchmark`): many backends × datasets × seeds;
  can take minutes to hours. See [evaluation sampling](EVALUATION_SAMPLING.md)
  for bounded runs and the records needed to interpret them.
