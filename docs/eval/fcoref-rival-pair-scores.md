# How FCoref scored the exact name and the enclosing mention

The [endpoint trace](fcoref-endpoint-stages.md) found 43 positive GAP decisions where the annotated name ranked below the proposal cutoff and the pronoun's returned cluster contained a longer mention enclosing that name. We replayed precisely those cases from the 256-article English panel. When several cluster members enclosed the name, we chose the shortest, breaking ties by source start and then end. The pronoun and both candidate names received slots under the [fixed-budget reservation](fcoref-marked-rescue.md). The native resolver saw source text and the three marked boundaries. The evaluation script used the reference links to choose which decisions to inspect.

For each candidate pair, FCoref scores the later token mention against the earlier one. It adds the two mention logits to a learned pair term. The highest-scoring previous mention wins if its total is at least zero, otherwise the later mention chooses null. An earlier candidate wins an exact tie with null. On 35 cases the later mention was the pronoun. On eight it was the name or the enclosing expression.

| Median across 43 decisions | Exact marked name | Enclosing mention |
|---|---:|---:|
| Name or enclosing mention logit | −13.50 | −1.69 |
| Learned pair term with the pronoun | 4.78 | 8.10 |
| Total for that direct edge (null = 0) | −9.76 | 6.32 |
| Gap to the native winning score for the later mention | 14.12 | 0.00 |

All 43 exact edges scored below null. The best of the 43 reached −1.62. With the exact name present, the later mention chose another antecedent on 35 cases and null on eight. Every enclosing mention had a higher mention logit, and its learned pair term was higher on 41 cases. The median *paired* differences (enclosing minus exact) were 11.94 for the mention logit, 3.60 for the pair term and 15.34 for the total. The paired medians are computed separately. The mention-score disadvantage exceeded the pair-term disadvantage in 42 of 43 cases. The exact pair term alone was positive in 36 cases. That term is an uncalibrated component of the model score. Choosing an antecedent uses the complete edge total.

The enclosing edge was the native direct choice in 38 cases. In the other five, the two spans met through the cluster, including two whose direct score was below null. Reserving the marked names preserved all 43 enclosing-edge totals from stock. A low exact-name score explains why guaranteed admission alone failed on this slice. The [separate blinded review](fcoref-identity-review.md) assesses whether the enclosing expression denotes the person and measures exact highlighting independently. [Aggregate values and extrema](fcoref-rival-pair-scores.json)

## Replay

Use the pinned GAP files, tokenizer and FP32 model from the [original trial](fcoref-emitted-boundaries.md#replay). Build the rank instrument and pair instrument in **separate disposable checkouts** of the study revision. The examples have the same executable name. Keep their paths distinct. For the first executable, follow the [rank-readout build](fcoref-endpoint-stages.md#replay-the-stage-counts) and set `RANK_BIN` to its absolute path. In the second checkout run the following commands.

```sh
git apply scripts/qa/fixtures/fcoref-marked-rescue.patch
git apply scripts/qa/fixtures/fcoref-pair-scores.patch
cargo build --locked -p anno --no-default-features --features onnx --example coref_gap_panel
python3 scripts/qa/fcoref_rival_pair_scores.py \
  "$GAP/gap-development.tsv" "$GAP/gap-test.tsv" "$GAP/gap-validation.tsv" \
  "$FCOREF_MODEL_PATH" target/debug/examples/coref_gap_panel "$RANK_BIN"
```

The [scorer](../../scripts/qa/fcoref_rival_pair_scores.py) checks the selected tokens and logits against their independent rank trace, the unmodified stock clusters, both components of every native direct edge, and the antecedent chosen by the decoder. Multiple token spans that project to the same trimmed occurrence use the highest mention score, with the first source-ordered representation winning ties. Add `--private-rows /absolute/private/new-file.json` to join case scores with independent judgments in authorized storage. The public result contains aggregates and digests. Rebuilding can change the executable digest.
