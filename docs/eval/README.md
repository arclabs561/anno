# Experiment records

Start with the study closest to the decision you are making. Each study names what it scores. A coreference link, an exact mention, and a directed relation have different denominators. For a comparison across backends, start with [where predictions failed](backend-residuals.md).

| If you are investigating | Read the notes | Inspect the recorded result |
|---|---|---|
| Entity recognition across languages and text lengths | [NER panels and replay](ner-language-length.md) | [Counts by model and panel](ner-language-length.json) |
| Whether an Anno release recovers unfamiliar names | [Native WNUT-17 release comparison](../DOMAIN_QA.md#compare-native-anno-on-unfamiliar-names) | [Heuristic counts](wnut-native-comparison.json) · [Offline default-stack counts](wnut-stacked-comparison.json) |
| When the default stack should pay for a skipped NuNER call | [Exact names, extra predictions and latency with BERT present or absent](nuner-fallback-routing.md) | [Paired native routing counts and timings](nuner-fallback-routing.json) |
| Whether NuNER routing transfers to encyclopedia sentences | [All WikiGold sentences with BERT present or absent](wikigold-nuner-routing.md) | [Routing across WikiGold and WNUT](../figures/wikigold-routing-tradeoff.svg) · [Where the WikiGold stock stack lost names](../figures/wikigold-entity-errors.svg) |
| Whether relation scores depend on seeing test labels | [CrossRE prompt comparison](crossre-full-prompt-comparison.md) | [Paired counts by domain](crossre-full-prompt-comparison.json) |
| Which extra directed relation to present for review | [Predicate and direction scores across six CrossRE domains](crossre-predicate-ranking.md) | [Development and balanced test panels](../figures/crossre-predicate-tradeoff.svg) · [Additional correct and unmatched suggestions on the full release](../figures/crossre-full-test-tradeoff.svg) |
| Which sources contributed examples or tagged-token counts | [Dataset coverage and exclusions](dataset-coverage.md) | [All registry IDs and source outcomes](dataset-coverage.csv) |
| Where a coreference link disappears | [Follow a native link from proposal through clustering](native-gap-endpoint-stages.md) · [Original 128-excerpt study](native-gap-loss.md) | [All three GAP splits](native-gap-endpoint-stages.json) · [FCoref proposal budget](fcoref-proposal-budget.json) |
| When the native resolver should abstain between two supplied names | [Choose a margin on GAP development](native-gap-margin.md) · [Test that choice on held-out books](litbank-pronoun-transfer.md) | [Frozen policy](native-gap-margin-policy.json) · [Literary transfer counts](litbank-pronoun-transfer-score.json) |
| How many correct links survive an abstention rule | [Separate the two score margins on GAP and transfer them to Winogender](native-gap-link-retention.md) | [Correct and wrong pronoun links at each cutoff](../figures/native-gap-margin-tradeoff.svg) |
| Whether FCoref generated the right name before pruning | [Trace the 69 missing-endpoint cases](fcoref-endpoint-stages.md) · [Reserve exact spans at the same budget](fcoref-marked-rescue.md) | [Counts by first loss and endpoint](fcoref-endpoint-stages.json) |
| Whether a wider coreference mention identifies the right person | [Blinded referent and exact-highlight review](fcoref-identity-review.md) | [Optional highlight trial](fcoref-identity-review.md#does-a-single-contained-person-span-make-a-usable-highlight) · [Agreement and character overhang](fcoref-identity-review.json) · [Native scores against null](fcoref-rival-pair-scores.md) |
| Whether changing a returned mention boundary recovers a link | [Repeated names in FCoref clusters](fcoref-emitted-boundaries.md) | [Before and after on 256 articles](fcoref-emitted-boundaries.json) |
| Whether a relation was missing from the reference | [JEREX original and revised annotations](jerex-controls.md) | [Reannotation counts](../../scripts/qa/joint_baseline/receipts/jerex_controls/revised_reference.json) |
| Which cells Muxer sampled | [Replay and export the sampler summary](../EVALUATION_SAMPLING.md#checked-in-public-evidence) | [Pinned offline replay](muxer-public-summary.json) |

The sampler summary covers two WikiGold backends and three seeds. For a particular CI run, inspect that run’s uploaded decision journal and dataset-pool evidence while its artifacts are retained. The [matrix workflow](../../.github/workflows/ci.yml) sets a 60-day lifetime for the separate trusted sampler checkpoint. Git preserves selected summaries and the reasoning behind them. Use `git log -- docs/eval/` for the history of those notes. The [research reference index](../../crates/anno-eval/research/reference/README.md) also covers experiments maintained with the evaluator.

For a new extraction, use the built-in defaults or a [named extraction profile](../QUICKSTART.md#reuse-a-cli-workflow-profile). For a sampling campaign, choose the task and budget, then reserve [task and language coverage](../EVALUATION_SAMPLING.md#reserve-part-of-a-trial-for-coverage) when needed. A study's recorded checkpoints, thresholds and cohort pins reconstruct that experiment. Changing them defines a new comparison.

## Record a trial

Put the question, compared arms, and replay command alongside the relevant study. Record the revision and input pins, cohort selection, observed backend and language where available, and the unit being scored. Give the paired counts for what improved and what regressed, the decision they support, and any arm that could not run. A result that preserves the current default is worth recording.

Link the note to a compact result file where sharing is permitted. Source text, per-document predictions and debug logs can expose the original material. Keep those with the authorized inputs and record how a reviewer can regenerate them. Project-authored regression fixtures and third-party materials with redistribution rights follow the [fixture and source policy](../QA.md#decide-what-to-keep).

Use `anno viz errors` to see which labels and spans an evaluation got wrong. To compare policies on the same inputs, put their correct matches, unmatched predictions and denominators in a tradeoff report. The WikiGold, GAP and CrossRE studies use the same report format. For an extraction without gold annotations, `anno viz overlap` measures how often signals share characters. [Command examples and report format](../QUICKSTART.md#visualize-evaluation-results) · [Figure inputs and regeneration](../figures/README.md)
