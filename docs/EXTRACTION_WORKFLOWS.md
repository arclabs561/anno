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
