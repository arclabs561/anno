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
coreference run omit this field.

### Speaker-attributed first person

Build the CLI with `--features discourse` to use `anno pipeline --coref --speaker-turns turns.json --files transcript.txt --format json`.
Use one transcript per invocation. For example, the text `Éva: I took my bag.` takes this sidecar:

```json
[{"speaker": "channel-A", "text": "I took my bag.", "start": 5, "end": 19}]
```

Offsets count Unicode scalar values in the original text, with an exclusive end. Turns must be in order and disjoint. Leave turns with unknown speakers out of the array. For a recorder that supplies audio times, compute text offsets as you assemble the transcript and keep the audio times separately.

Providing this sidecar selects a speaker-only resolver. General name and third-person chains from the ordinary resolver are omitted. The resolver records `speaker-turns` in `coref_backend` and uses document-local speaker IDs as track names. It groups English `I`, `me`, `my`, `mine`, and `myself` when at least two mentions belong to one speaker. A missing turn language defaults to English. Turns bearing quotation marks and turns marked with another language are skipped. The resolver leaves second-person references unresolved. Its output depends on the accuracy of the supplied speaker annotations.

## Evaluation (two layers)

`anno` has two eval layers with very different runtimes:

- **Single-text eval** (`anno eval`): inline gold spans against one text; typically milliseconds–seconds.
- **Benchmark matrix** (`anno benchmark`): many backends × datasets × seeds;
  can take minutes to hours. See [evaluation sampling](EVALUATION_SAMPLING.md)
  for bounded runs and the records needed to interpret them.
