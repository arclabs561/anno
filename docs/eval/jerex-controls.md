# JEREX cluster and relation controls

## Independent reference for 81 unmatched predictions

The historical DocRED evaluation found 81 JEREX predictions whose directed endpoint pair carried no relation in the original labels, although both endpoint clusters matched exactly. We compared every prediction with the published Re-DocRED revision. The audit requires identical sentence tokens and uniquely identical sets of mention spans at both endpoints. It preserves direction and predicate identity across potentially reordered entity indices. All 32 source documents matched. Re-DocRED places 25 in its train split, three in dev and four in test. The checkpoint and predictions came from the original JEREX study. These splits describe the revision used for scoring, and do not establish performance on the Re-DocRED test split alone. [Original evaluation](joint-edge-loss.md) · [Tan et al., EMNLP 2022](https://arxiv.org/abs/2205.12696) · [Revised annotations](https://github.com/tonytan48/Re-DocRED/tree/ccfb54f5ddf5836027c87badda10f6dfc56efaac/data)

| Revised-reference result | Predictions |
|---|---:|
| Same directed predicate explicitly annotated | 65 |
| Forward pair has a different predicate | 2 |
| Forward pair still has no annotation | 14 |

Thus 65 of the original reference misses are positives under independent reannotation. The remaining 16 require evidence beyond this comparison to judge their factual accuracy. This fraction concerns the 81 aligned, originally unannotated pairs. The source data and per-document labels stay in the caller’s authorized external directory. [Aggregate audit receipt](../../scripts/qa/joint_baseline/receipts/jerex_controls/revised_reference.json)

## Two controlled interventions

Each CPU forward pass uses the frozen checkpoint and upstream JEREX revision. First run stock inference. On the paired pass, preserve every detected mention and change only the partition handed to entity and relation scoring. Group surviving detections by the original reference entity and retain every unannotated detection as its own singleton. The oracle receives no relation labels, adds no spans, and recomputes downstream entity types and relation logits. This measures how much an improved partition *could* change the trained scorer.

Within each partition, classify the same valid pair/predicate probabilities at the published 0.60 cutoff and prespecified alternatives of 0.25, 0.50 and 0.75. Score the identical predictions against each annotation edition. Only the original DocRED membership sets supply the oracle partition, including when Re-DocRED supplies the scoring labels. Exact matches require the two directed endpoint membership sets and predicate. The replay checks identical detector output between arms, exact agreement with the stock relation decoder at its published cutoff, and aggregate agreement with the independent writer scorer on the original split. Gains and losses use the stock partition and checkpoint cutoff under the same reference as the baseline. Paired percentile intervals resample documents 2,000 times with seed 53. [Experiment](../../scripts/qa/joint_baseline/jerex_controls.py) · [Original stage definitions](joint-edge-loss.md)

### First 32 documents

The stock run at 0.60 reproduced the historical 140 matches and 204 extras. Changing only the partition recovered 63 original-reference edges and lost zero. Micro-F1 rose from 37.3 to 49.9. The document-bootstrap interval on the difference was +6.5 to +19.8 percentage points. Against Re-DocRED the same partition change gained 84 matches and lost one. Some endpoint gains therefore survive independent reannotation. [Original and revised counts, all thresholds](../../scripts/qa/joint_baseline/receipts/jerex_controls/docred_0_31.json)

| Reference | Gold | Stock matches / extras | Reference partition, same detections and cutoff | Stock partition, cutoff 0.25 |
|---|---:|---:|---:|---:|
| Original DocRED | 406 | 140 / 204 | 203 / 205 | 160 / 267 |
| Re-DocRED | 1,044 | 207 / 137 | 290 / 118 | 242 / 185 |

The threshold-only change at 0.25 added 20 original matches and 63 extras. Its original-reference F1 difference interval was −1.2 to +3.4 points. The revised-reference interval was +1.6 to +4.6. Scoring both editions matters for an intervention that deliberately increases recall.

### Next 32 documents

The same interventions ran on the next 32 documents in published order, with identical cutoffs. Both cohorts are English. The clusters use reference annotations and cannot be supplied by the deployed extractor. The result identifies an intervention worth testing with a learned or deterministic partitioner. [Replication counts and paired intervals](../../scripts/qa/joint_baseline/receipts/jerex_controls/docred_32_63.json)

| Reference | Gold | Stock matches / extras | Reference partition, same detections and cutoff | Stock partition, cutoff 0.25 |
|---|---:|---:|---:|---:|
| Original DocRED | 536 | 154 / 249 | 234 / 198 | 172 / 350 |
| Re-DocRED | 1,208 | 211 / 192 | 310 / 122 | 245 / 277 |

On the second cohort, the partition intervention gained 80 original-reference matches and lost zero (micro-F1 32.8 to 48.3, paired interval +7.5 to +23.6 points). With revised labels it gained 101 and lost two. The stock partition at 0.25 added 18 original-reference matches and 101 unmatched predictions. Its original-reference F1 difference interval again included zero (−2.2 to +1.6 points). Both replications support improving clustering before changing the default relation cutoff.

## Reproduce

For an application with its own validation data and tolerance for extra predictions, the public CPU adapter accepts an explicit cutoff and records the override in each output row. Omitting it uses the checkpoint value.

```sh
python crates/anno-eval/examples/docred/predict_jerex.py "$MODEL" "$INPUT" "$OUTPUT" --relation-threshold 0.25
```

Use the pinned CPU environment and authors’ checkpoint from the [JEREX setup](../QA.md#run-the-jerex-joint-checkpoint). Place the three Re-DocRED JSON files from the linked revision in an authorized external directory. Both scripts verify the pinned file digests. The optional progress file is created exclusively and contains per-document aggregate counts. Use a fresh external path for each run.

```sh
qa=scripts/qa/joint_baseline
python "$qa/jerex_revised_reference.py" "$GOLD" "$PREDICTION_RECEIPT" "$REVISED_DIR"
python "$qa/jerex_controls.py" "$MODEL" "$GOLD" --start 0 --documents 32 --revised-directory "$REVISED_DIR"
python "$qa/jerex_controls.py" "$MODEL" "$GOLD" --start 32 --documents 32 --revised-directory "$REVISED_DIR"
```

`GOLD` is the authors’ original `test_joint.json`. Build the external full prediction receipt with the [stage replay](joint-edge-loss.md#source-terms-and-replay), or supply the exact historical receipt matching the public digest. Only aggregate scores and digests belong in this result set.
