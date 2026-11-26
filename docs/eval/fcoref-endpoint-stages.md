# Where FCoref lost a marked endpoint

On the [256 English GAP articles used in the emitted-boundary trial](fcoref-emitted-boundaries.md), FCoref missed 82 of 231 annotated pronoun–name links. We traced each positive pair from token-span enumeration to the final cluster. The ordered cohort and every returned stock cluster matched the original run. GAP also marks negative pairs. Across all 512 decisions the stock resolver produced 149 true links, 12 false links, 82 missed links and 269 true negatives. [Aggregate receipt](fcoref-endpoint-stages.json)

| First stage where a positive pair was lost | Gold links | Which endpoint was missing at this stage |
|---|---:|---|
| No legal token proposal for an exact source span | 0 | — |
| An exact proposal ranked below the stock cutoff | 57 | 55 name only, 1 pronoun only, 1 both |
| Both exact proposals survived but a final cluster lacked an endpoint | 12 | 10 name only, 2 both |
| Both endpoints were returned in separate clusters | 13 | Both present |
| Linked in the same returned cluster | 149 | Both present |

The 57 and 12 explain the previous trial's 69 missing-endpoint cases. For all 56 pairs whose exact name ranked below the cutoff, at least one longer retained span strictly enclosed that occurrence. None had a shorter retained span inside the exact name. In 43 of those pairs, the pronoun's own returned cluster contained a longer rival. Names fell a median of 559.5 proposal slots below the cutoff (range 12–1,850). Doubling the stock slot count would admit both exact endpoints for only one of the 57 pairs. That count concerns admission. Whether an admitted pair links depends on antecedent scoring and cluster construction.

A boundary-aware name scorer is worth testing at the existing budget. Track wrong links and exact links already recovered by the stock model when changing which proposals get a slot. A broad or nested referring expression can be linguistically valid while missing the particular name occurrence that GAP scores. This English panel measures exact agreement with those marked occurrences.

## Replay the stage counts

Use the three GAP TSVs and pinned FP32 FCoref export from the [original trial](fcoref-emitted-boundaries.md#replay). In a disposable checkout, apply the existing rank readout to the normal native example. Install the Python `tokenizers` package in the environment that will run the scorer.

```sh
git apply scripts/qa/fixtures/fcoref-boundary-ranks.patch
cargo build --locked -p anno --features onnx --example coref_gap_panel
python3 scripts/qa/run_fcoref_endpoint_stages.py \
  "$GAP/gap-development.tsv" "$GAP/gap-test.tsv" "$GAP/gap-validation.tsv" \
  "$FCOREF_MODEL_PATH" target/debug/examples/coref_gap_panel
```

The [runner](../../scripts/qa/run_fcoref_endpoint_stages.py) sends one unlabeled document at a time and collects exactly one rank table for its response. The [scorer](../../scripts/qa/fcoref_endpoint_stages.py) checks that the trace contains every legal token pair once, in native score order. Several token pairs can trim to the same character span. Its best rank determines whether that occurrence survives. Every emitted span must have a retained proposal. The public receipt records aggregates and fingerprints. The passages, ranks and predictions stay with the authorized evaluator. A rebuild can produce a different diagnostic binary digest.
