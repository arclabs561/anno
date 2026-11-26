# Can the FCoref linker use an exact mention that missed its candidate budget?

The earlier [emitted-boundary experiment](fcoref-emitted-boundaries.md) found 69 missed gold links for which the resolver did not emit both exact endpoints. We tested whether admitting the three benchmark-marked spans to the native antecedent scorer could change final links while keeping the stock candidate count. Marking a span supplies its precise location, including the locations of distractors. The scorer alone receives the gold link labels.

## Method

Both arms use the same 256 GAP development articles, with the same text, native CPU encoder, tokenizer, FP32 mention and antecedent weights, mention logits and per-document candidate count. The control supplies an empty list of reserved spans. Its complete ordered cluster partitions and confusion counts must reproduce the recorded stock panel. Each treatment input contains the pronoun and the A and B name spans as UTF-8 byte ranges in that source. Among legal token spans that map to each exact marked range after Anno's normal trimming, the treatment reserves its highest-ranked token span. It fills all other slots by stock logit rank, restores ordinary token order and runs ordinary antecedent scoring and clustering. A mark without an exact legal token span cannot reserve a slot. Candidate scores, tie order, null decisions and clustering are unchanged by the selection code.

This diagnostic measures how the existing linker responds to a supplied mention inventory. GAP marks two candidate names and a pronoun per excerpt. A changed link that does not correspond to one of the two marked decisions has no correctness judgment in this experiment. The checkpoint's training exposure is unknown. [Panel and source rights](fcoref-emitted-boundaries.md#replay)

## Result

All 768 marked occurrences had a legal exact token realization. The native reservation promoted 186 into the top K across 146 of 256 documents. Both arms kept 6,460 total slots. Independent projection of the stock rank traces reproduced all three proposal counts. The no-hints arm reproduced the frozen stock partition commitment.

| Observable outcome | Stock | Reserved marked boundaries |
|---|---:|---:|
| Correct / false / missed marked links | 149 / 12 / 82 | 149 / 12 / 82 |
| Gold pairs with both exact endpoints in the returned clusters | 162 of 231 | 162 of 231 |
| Returned mention spans | 2,311 | 2,309 |

All 512 labeled decisions stayed identical. Reservation recovered zero of the 69 missed links with a missing emitted endpoint and zero of the 13 with already-emitted endpoints in separate clusters. It removed one unscored mention pair in one document. GAP cannot judge that change. [Aggregate result](fcoref-marked-rescue.json)

The [native antecedent scorer](../../crates/anno/src/backends/coref/fcoref/scoring.rs) adds both mention logits to the learned pair logit. The best earlier candidate links at a total score of zero or greater. Lower totals choose the null antecedent, and the cluster builder omits singletons. A promoted name keeps its original mention logit. Giving it a slot therefore tests admission alone. These results support keeping the current selection policy and examining mention scores, competing antecedent scores and their margins against null before changing inference defaults.

## Replay

From a disposable Anno checkout, apply the research patch and build its example. The marked-reservation test exercises exact selection at a fixed budget, including an unrepresentable mark and two token variants of the same mark. Set `GAP` and `FCOREF_MODEL_PATH` to the authorized external sources documented in the panel. The replay verifies the original file pins, independent stock result and cohort digest. The harness emits aggregate counts. Source text, labels and native partitions stay outside the tracked results.

```sh
git apply scripts/qa/fixtures/fcoref-marked-rescue.patch
cargo test --locked -p anno --no-default-features --features onnx --lib marked_reservation_keeps_best_variant_at_fixed_budget
cargo build --locked -p anno --no-default-features --features onnx --example coref_gap_panel
python3 scripts/qa/fcoref_marked_rescue.py \
  "$GAP/gap-development.tsv" "$GAP/gap-test.tsv" "$GAP/gap-validation.tsv" \
  "$FCOREF_MODEL_PATH" target/debug/examples/coref_gap_panel
```
