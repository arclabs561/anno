# What does a track score measure?

For evidence on when to publish or revise a track assignment, see the [index reconciliation experiments](index-reconciliation.md), [downstream search, archive and review workflows](index-consumers.md) and the [cross-document coreference audit](cross-document-coreference.md).

A `Track` records the within-document membership a caller can act on. Three different questions arise when a model produces it. Did the detector find a mention? Do two mentions refer to the same entity? Does the entire predicted track equal a gold track, including all its members? Each question needs its own labels and denominator.

## Evidence available in anno

`Track::new` initializes `cluster_confidence` to 1.0. The native mention-ranking projection can publish this default without measuring the correctness of the chain. Consumers cannot recover a calibrated probability from that value alone. [Track constructor](../../crates/anno/src/core/grounded/mod.rs) · [Mention-ranking projection](../../crates/anno/src/backends/coref/mention_ranking/types.rs)

At the next level, the inter-document resolver measures the weakest comparison among all pairs of tracks assigned to an identity. Its regression joins “Ada”, “Ada Byron” and “Byron” at a Jaccard threshold of 0.5. The two adjacent similarities are 0.5 and the endpoints score zero. All three belong to the committed component, whose `minimum_pair_similarity` is `Some(0)`. A singleton has no pair to compare and yields `None`. This diagnostic exposes how far a transitive decision can extend beyond its supporting edges. [Resolver and regression](../../crates/anno/src/coalesce/resolver.rs)

Even calibrated pair probabilities leave a whole-track forecast undetermined. Consider three mentions A, B and C. In one distribution, exactly one uniformly chosen pair belongs together. In another, all three belong together with probability one-third and all three are separate otherwise. Every pair has match probability one-third under both distributions. The probability that all three belong together is zero in the first and one-third in the second. A model that supplies such a group forecast needs assumptions about complete assignments. [Bayesian entity resolution](https://proceedings.mlr.press/v33/steorts14.html) · [Decision theory for partitions](https://doi.org/10.1214/17-BA1073)

## Model choices and their type implications

| Approach | What the model can supply | Implication for anno |
|---|---|---|
| Fellegi–Sunter record linkage | A likelihood ratio for agreements among matched versus unmatched pairs. A common shared name can carry less weight than a rare shared alias. Converting the ratio to posterior odds also needs prior match odds for the candidate population. | Keep the comparison and its candidate population with the pair. Record whether the pair passed blocking so the evaluator can count links the scorer never saw. [Fellegi & Sunter](https://doi.org/10.1080/01621459.1969.10501049) |
| Learned antecedent ranking | A distribution over earlier candidates and a null choice. “Ada” in “Ada arrived. She spoke.” can select null and still end up in a two-mention track. A singleton-aware model can learn a separate mention or singleton decision. | Store candidate choices against document-scoped signal IDs. Keep null, a genuinely annotated singleton, and a decision to abstain distinguishable. Preserve the score's scale, including logits. [Lee et al.](https://aclanthology.org/D17-1018/) · [Zou et al.](https://aclanthology.org/2024.naacl-short.19/) |
| Multiple-hypothesis tracking | Mutually exclusive association histories let later evidence revise an earlier decision. In text, a later sentence may disambiguate which of two namesakes “she” refers to. | Keep alternative assignments in the resolver run. Publish one committed membership per signal into the document. Sensor motion, survival and per-scan detection assumptions need independent justification for discourse entities, including fictional and negated referents. [Reid](https://doi.org/10.1109/TAC.1979.1102177) · [Reuter et al.](https://doi.org/10.1109/TSP.2014.2323064) |

A future evidence API should name the event, identify its members, record the producer and score scale, and identify the evaluation population for calibrated forecasts. An antecedent candidate can carry a raw logit. A pair forecast needs a defined pair event. An exact-track forecast needs the complete member set and annotation policy. Sorting members preserves membership evidence. Adding, removing or replacing a member invalidates evidence about that exact set. These are proposed semantics for a future producer and consumer.

One statistical alternative to the current threshold-and-union resolver is to choose the partition with the smallest expected pairwise loss. For a calibrated co-membership probability `p`, joining the pair costs `false_merge_cost × (1 − p)` in expectation. Separating it costs `false_split_cost × p`. If a false merge costs nine times as much as a false split, an isolated pair needs a probability above 0.9 to favor joining. With three or more mentions, minimize the sum over valid partitions because choices must be transitive. Pair marginals suffice for this particular expected loss even though they cannot answer the whole-track probability question above. Compare this decision rule with the existing resolver on the same candidate set. [Wade & Ghahramani](https://doi.org/10.1214/17-BA1073)

## What the namesake fixture shows

Two offline runs used the same ONNX-enabled anno CLI at `d232e80`, identical UTF-8 input and the heuristic NER model. The grounded run receipts identify which coreference backend completed. One used native mention-ranking. The other used the CPU FP32 export of `biu-nlp/f-coref` at revision `e91dfff12879495d882ba9460d0c5d5dd44ade59` with ONNX Runtime 1.24.2. The original checkpoint and all four exported files matched the workflow's pinned hashes. Each backend published two tracks. Exact spans below count all pipeline signals, including untracked NER detections. [Namesake fixture](../../testdata/grounded/namesakes.json) · [Model export and checksums](../../.github/workflows/fcoref-cli.yml)

| Method | Exact assigned spans | Correct links | False merges | Missed links | Correct nonlinks |
|---|---:|---:|---:|---:|---:|
| Native mention-ranking pipeline | 3/6 | 1 | 0 | 5 | 9 |
| Pinned F-coref pipeline | 4/6 | 0 | 0 | 6 | 9 |
| Casefolded surface equality, supplied gold spans | 6/6 supplied | 1 | 1 | 5 | 8 |

The denominator is the 15 unordered pairs among six assigned mentions, with six positive and nine negative pairs. A pair receives a predicted link when both exact spans occur in one emitted track. The seventh mention has unknown membership and is excluded. The native resolver linked “Finch” to “She”. Its other track joined two surname fragments outside the exact gold spans. Surface equality joined the two pilot descriptions correctly and the two namesakes incorrectly.

F-coref linked “Dr. Léa Noor (alias Finch)” to “She” and “A different Léa Noor, the pilot” to “The pilot”. Both first spans are wider than the gold highlights. The two edges therefore remain unjudged under this boundary rule, and precision on the restricted pair set is undefined. This example motivates annotating compact highlights and accepted referring expressions separately before judging association on either. A single document cannot rank the two models. The separate 12-document person panel supplies both boundary policies and marked distractors. Its pinned F-coref run covered 22 of 37 compact highlights and 33 of 37 accepted referring expressions. [Panel and scoring protocol](../QA.md#score-person-highlights-and-referring-expressions) · [Peer evaluation](https://github.com/arclabs561/anno/pull/273)

To reproduce, build the CLI with ONNX support, extract the source without a trailing newline and run it once with each model-path setting. Start in a working directory without a fallback model and check the resulting `coref_run.backend`. The `--model` option selects NER in both runs.

```sh
jq -j .text testdata/grounded/namesakes.json > /tmp/anno-namesakes.txt
ANNO_NO_DOWNLOADS=1 HF_HUB_OFFLINE=1 FCOREF_MODEL_PATH=/nonexistent \
  anno pipeline --files /tmp/anno-namesakes.txt --model heuristic --coref --format grounded
ANNO_NO_DOWNLOADS=1 HF_HUB_OFFLINE=1 FCOREF_MODEL_PATH="$FCOREF_MODEL_DIR" \
  anno pipeline --files /tmp/anno-namesakes.txt --model heuristic --coref --format grounded
```

## A comparison that could change the design

1. Run the native mention ranker and a pinned English F-coref model on the same document groups. Record the backend that actually ran. Report exact-span detection precision and recall first. On fixed mentions, score an antecedent as correct if it selects any earlier member of the gold chain. Track first mentions separately. The existing CLI oracle-mentions path uses the simple resolver, so a fixed-mention F-coref comparison requires an adapter. [F-coref](https://aclanthology.org/2022.aacl-demo.6/) · [Evaluation path](../../crates/anno-cli/src/cli/commands/dataset.rs)
2. Hold out complete document groups and keep repeated identities out of fitting and testing for the unseen-identity comparison. Include aliases, pronouns, same-name different people, ordinary negatives and rejected candidates. Report unjudged examples separately. For calibrated pair forecasts, report Brier loss and observed match rates by probability bin, with counts. Restore deployment prevalence with sampling weights when negatives were deliberately oversampled. [Proper scoring rules](https://doi.org/10.1198/016214506000001437)
3. Score the published partition with LEA and separate false-merge and false-split counts. Report chain length, candidate coverage and annotated singleton policy. For a system that can defer a link, plot error among accepted links against coverage. Use grouped uncertainty intervals over documents or entity groups. Choose an operating point against a caller's stated cost of joining namesakes versus splitting aliases. [LEA](https://aclanthology.org/P16-1060/) · [Anno metrics](../../crates/anno/src/metrics/coref_metrics.rs)
