# Anno documentation

Run the [quickstart](QUICKSTART.md) for a first CLI or Rust extraction. Before saving annotations or highlighting a result, check which source text and offset units the caller must retain. [Source text and offsets](CONTRACT.md#source-text-and-offsets)

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

Versioned decisions in `docs/decisions/` connect a choice to its public verification, `docs/dev-notes/` holds implementation plans, and private research and raw experiment records stay in the project workspace.

- [Grounded extraction and evaluation decisions](decisions/grounded-extraction.md)
- [Fastino backend implementation plan](dev-notes/fastino-backend-plan.md)
- [Research references](REFERENCES.md)

Run `just docs-audit` after editing or moving documentation.
