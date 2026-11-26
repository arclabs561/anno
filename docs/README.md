# Anno documentation

Run the [quickstart](QUICKSTART.md) for a first CLI or Rust extraction. Before saving annotations or highlighting a result, check which source text and offset units the caller must retain. [Source text and offsets](CONTRACT.md#source-text-and-offsets)

## Verify a user-visible result

| Goal | Replay and expected result |
|---|---|
| Highlight the correct occurrence in the original HTML | Follow the [two-address replay](EXTRACTION_WORKFLOWS.md#trace-a-signal-to-its-original-html). Both occurrences display the same address. Their `@` characters point to different source bytes, one encoded and one literal. |
| Render a saved constituency parse | From the checkout root, run `anno parse --input testdata/constituency/benepar_en3.json`. It prints two trees, one for “Mira builds small robots.” and one for “She smiles.” [Parser interchange](CONSTITUENCY.md) · [Fixture](../testdata/constituency/README.md) |

## Integrate extraction

| Task | Guide |
|---|---|
| Select a model, feature flags and runtime | [Backends](BACKENDS.md) |
| Reuse a model across documents and interpret a measured extraction profile | [Extraction workflows](EXTRACTION_WORKFLOWS.md) |
| Work with entities, signals, tracks and identities | [Architecture](ARCHITECTURE.md) · [Public contract](CONTRACT.md) |
| Represent text and other media locations | [Location model](LOCATION.md) |
| Exchange constituency trees with an external parser | [Constituency parsing](CONSTITUENCY.md) |

## Evaluate a result

| Task | Guide |
|---|---|
| Choose a regression fixture, a labeled comparison or a live check | [Fixtures, datasets and QA](QA.md) |
| Compare extraction across labeled domains | [Domain QA](DOMAIN_QA.md) |
| Allocate evaluation runs under a fixed budget | [Evaluation sampling](EVALUATION_SAMPLING.md) |
| Investigate a recurring failure or challenge a quality claim | [Adversarial QA runbook](ADVERSARIAL_QA.md) |
| Interpret a worked example of redaction scoring | [PII character-masking evaluation](eval/pii-char-masking.md) |

Run `just regenerate-datasets` to produce the dataset registry in the generated documentation directory.

## Build, operate and release

- [Build instructions and local checks](BUILDING.md)
- [Dataset cache in CI](S3_CACHE_CI.md)
- [Publishing and validation](PUBLISH_STATUS.md) · [Git tags and releases](TAGS.md)

## Understand a design decision

Versioned decisions in `docs/decisions/` connect a choice to its public verification, and `docs/dev-notes/` holds implementation plans. Use a durable project archive for private research and raw receipts. A cloud workspace is only a working copy.

- [Grounded extraction and evaluation decisions](decisions/grounded-extraction.md)
- [Fastino backend implementation plan](dev-notes/fastino-backend-plan.md)
- [Research references](REFERENCES.md)

Run `just docs-audit` after editing or moving documentation.
