# Anno documentation

Run the [quickstart](QUICKSTART.md) for a first CLI or Rust extraction. Before saving annotations or highlighting a result, check which source text and offset units the caller must retain. [Source text and offsets](CONTRACT.md#source-text-and-offsets)

## Use Anno with an agent

For clients that load a local Agent Plugins package, select the checkout’s `.agents/` directory. Choose a skill by the result you need.

| Goal | Skill |
|---|---|
| Run an operation with the installed binary and inspect its output | [Anno CLI](../.agents/skills/anno-cli/SKILL.md) |
| Use Anno from Rust or Python, highlight a source occurrence or exchange annotations | [Integrate Anno](../.agents/skills/anno-integrate/SKILL.md) |
| Reproduce a failure or compare two backend or scorer revisions | [Anno QA](../.agents/skills/anno-qa/SKILL.md) |

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
| Check why an HTML highlight is missing for a nested mention | [Review overlapping highlights](EXTRACTION_WORKFLOWS.md#review-overlapping-highlights) |
| Work with entities, signals, tracks and identities | [Architecture](ARCHITECTURE.md) · [Public contract](CONTRACT.md) |
| Represent text and other media locations | [Location model](LOCATION.md) |
| Exchange constituency trees with an external parser | [Constituency parsing](CONSTITUENCY.md) |

## Evaluate a result

| Task | Guide |
|---|---|
| Find an experiment’s notes, results and replay command, or record a new trial | [Experiment records](eval/README.md) |
| Choose a regression fixture, a labeled comparison or a live check | [Fixtures, datasets and QA](QA.md) |
| Choose a benchmark for mention spans, clusters or directed relations | [Worked example](QA.md#find-a-dataset-or-add-a-case) |
| Find the files and splits behind a figure and see which datasets were left out | [Dataset coverage and exclusions](eval/dataset-coverage.md) |
| Compare extraction across labeled domains or test recognition of unfamiliar names | [Domain QA](DOMAIN_QA.md) · [WNUT-17 example](DOMAIN_QA.md#test-recognition-of-unfamiliar-names) |
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
