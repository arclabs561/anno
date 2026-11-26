# Experiment records

Start with the study closest to the decision you are making. Each study names what it scores. A coreference link, an exact mention, and a directed relation have different denominators. For a comparison across backends, start with [where predictions failed](backend-residuals.md).

| If you are investigating | Read the notes | Inspect the recorded result |
|---|---|---|
| Entity recognition across languages and text lengths | [NER panels and replay](ner-language-length.md) | [Counts by model and panel](ner-language-length.json) |
| Where a coreference link disappears | [Native mention and link losses](native-gap-loss.md) | [Stage counts](native-gap-loss.json) · [FCoref proposal budget](fcoref-proposal-budget.json) |
| Whether a relation was missing from the reference | [JEREX original and revised annotations](jerex-controls.md) | [Reannotation counts](../../scripts/qa/joint_baseline/receipts/jerex_controls/revised_reference.json) |
| Which cells Muxer sampled | [Replay and export the sampler summary](../EVALUATION_SAMPLING.md#checked-in-public-evidence) | [Pinned offline replay](muxer-public-summary.json) |

The sampler summary covers two WikiGold backends and three seeds. For a particular CI run, inspect that run’s uploaded decision journal and dataset-pool evidence while its artifacts are retained. The [matrix workflow](../../.github/workflows/ci.yml) sets a 60-day lifetime for the separate trusted sampler checkpoint. Git preserves selected summaries and the reasoning behind them. Use `git log -- docs/eval/` for the history of those notes. The [research reference index](../../crates/anno-eval/research/reference/README.md) also covers experiments maintained with the evaluator.

## Record a trial

Put the question, compared arms, and replay command alongside the relevant study. Record the revision and input pins, cohort selection, observed backend and language where available, and the unit being scored. Give the paired counts for what improved and what regressed, the decision they support, and any arm that could not run. A result that preserves the current default is worth recording.

Link the note to a compact result file where sharing is permitted. Source text, per-document predictions and debug logs can expose the original material. Keep those with the authorized inputs and record how a reviewer can regenerate them. Project-authored regression fixtures and third-party materials with redistribution rights follow the [fixture and source policy](../QA.md#decide-what-to-keep).
