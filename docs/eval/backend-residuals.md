# Which Anno predictions failed, and where

The unit of error matters. GAP scores pronoun–candidate decisions; LitBank scores mention clusters; CrossRE and JointIE score directed, labeled relations between text occurrences; JEREX scores relations between whole clusters. NuNER scores spans with labels. All counts below use the individual experiment's reference annotations, including its omissions and scope. A false positive in these tables means a prediction unmatched by that reference.

The replay and classification utilities preserve duplicate multiplicity and verify original scoring totals. [Reproduction commands](../../scripts/qa/backend_residuals/README.md)

## Coreference

For raw GAP input, all three resolvers saw the same 128 excerpts. The 117 positive decisions reconcile as follows.

| Resolver | Missing at least one returned endpoint | Both returned, separate | Linked correctly | False links |
|---|---:|---:|---:|---:|
| Native | 109 | 2 | 6 | 4 |
| FCoref | 41 | 3 | 73 | 6 |
| Maverick | 40 | 1 | 76 | 1 |

Native proposed all 117 pronouns. For 65 pairs it missed the exact name at proposal. For 44, both exact mentions survived proposal filtering but a required endpoint had no selected incident link and disappeared from the output clusters. All four false native links came from direct antecedent selections. FCoref omitted only the name on 37 positive pairs and both endpoints on four. Maverick omitted only the name on 40. These returned-output observations cannot distinguish proposal from internal filtering in the two neural models.

A frozen possessive-name boundary change increased exact native name/pronoun proposals from 211 to 238 on 392 true validation pairs. It produced the same 27 correct links and one extra incorrect link. A separately sealed 128-excerpt test gained one correct link. The proposal improvement has been reproduced; the validation result gives no net correct-link gain for adopting the change by default. [Native trace and validation](https://github.com/arclabs561/anno/blob/543ac50b/docs/eval/native-gap-loss.md)

### Literary mentions and links

Every cell below comes from replayed cluster outputs. A missing gold pair counts once under either absent endpoint or split cluster. An extra link counts once under unannotated endpoint or merger of two annotated clusters.

| Resolver and input | Works scored | Matched pairs | Misses with absent endpoint | Misses split despite both endpoints | Extras with unannotated endpoint | Extras merging annotated clusters |
|---|---:|---:|---:|---:|---:|---:|
| native-predicted | 10 | 50 | 262 | 79 | 43 | 4 |
| native-supplied | 10 | 48 | 0 | 343 | 0 | 19 |
| fcoref-predicted | 10 | 226 | 165 | 0 | 23 | 8 |
| maverick-predicted | 10 | 278 | 105 | 8 | 60 | 17 |
| maverick-supplied | 9 | 289 | 0 | 2 | 0 | 282 |

FCoref’s 165 missed pairs all have a missing endpoint. Reducing linkage thresholds alone would help only if that also changes which mentions are emitted. Maverick supplied with gold spans illustrates the other failure mode. It missed just two pairs on nine admissible works, yet emitted 282 links between different annotated clusters. Three works accounted for 196 of those extras. Pair counts amplify the effect of merging large clusters. The refusal on *Dear Enemy* is recorded separately and accounts for the different denominator. [Original CoNLL scores and alignment policy](https://github.com/arclabs561/anno/blob/9973b2a5/docs/eval/coref-panels.md)

## Joint models

**GLiNER2.5 JointIE.** With 384-word windows, every one of the 161 gold edges in the 64-sentence cohort fit inside a scored window. Two were outside the declared predicate vocabulary. Another 137 never appeared in the sparse output despite both typed mentions sharing a window. The cap removed 17 and the optimizer selected none of the five remaining. The model emitted two other edges. One occurred in a sentence without annotated relations; the other had just one endpoint appearing anywhere in that sentence's gold relations. The restricted endpoint schema cut the number of typed candidates by 85%, also excluded 51 gold edges at declaration, and produced zero matches from four predictions.

Short windows yielded one match among 27 predictions. Of the 26 extras, 13 had just one exact span occurring in any gold relation. One was a same-predicate reversal with the exact typed endpoints. The other 12 are individually classified in the JointIE extras ledger. The direct target for diagnosis on the broad schema is the sparse pair output: the trace localizes an observable loss but does not expose which internal proposal operation omitted each edge. [All stage counts](https://github.com/arclabs561/anno/blob/e1e9e7bd/docs/eval/joint-edge-loss.md)

**JEREX.** On 32 held-out documents, 140 predicted labeled relations matched, 266 gold relations were missed, and 204 predictions were unmatched. The disjoint miss counts are 57 missing a required mention, 110 losing exact cluster membership and 99 below threshold after both clusters aligned. Of the 204 extras, 121 lacked an exact gold endpoint cluster, two used a different predicate on an annotated directed pair, and 81 had no annotated forward relation between the aligned endpoints. None of those 83 aligned-endpoint errors was explained solely by reversing the predicted predicate. The references establish an unmatched prediction; judging whether those 81 relations are actually false would require document-level adjudication. [Gold- and prediction-side diagnostics](https://github.com/arclabs561/anno/blob/e1e9e7bd/docs/eval/joint-edge-loss.md)

## Named entities and quantization

NuNER FP32 and QUInt8 processed the same 38 labeled inputs under three requested entity classes. At threshold 0.5 they missed 50 identical labeled mentions. Two other misses were unique to FP32 and one to QUInt8. They emitted 32 identical extras, plus 17 specific to FP32 and 13 specific to QUInt8. Both missed 28 of 29 gold mentions in the single long Russian stress input, which accounts for 28 of 30 no-overlap misses in each arm. That one case confounds language, length, content and text packing. Its result cannot isolate the cause or estimate performance across Russian documents.

Quantization reduced peak RSS on the measured host from 4,000 to 3,084 MiB. It changed exact predicted sets on 10 of 38 inputs. Default thresholds and precision choices need evaluation on the caller's target distribution. [Labeled results and error taxonomy](https://github.com/arclabs561/anno/blob/31af7a24/docs/eval/nuner-4k-labeled.md)

## Runtime, provenance and unavailable checkpoints

The flat extraction CLI could emit a relation without naming the scorer that produced it. The CLI now includes the observed relation producer when available. Existing tests distinguish propagated model errors from a successful run returning zero retained edges. [CLI fix](https://github.com/arclabs561/anno/pull/369)

The TPLinker heuristic runs have no measured trained TPLinkerPlus accuracy. The original Plus decoder can be exercised on supplied pair logits. Reproducible end-to-end inference still requires compatible trained weights, tag vocabulary, preprocessing and usable terms. The Grenander and Du candidates have different missing artifacts documented in the inventory. These are availability gaps with no test-set quality denominator. [Artifact inventory](https://github.com/arclabs561/anno/blob/31af7a24/docs/eval/outstanding-model-artifacts.md)

## CrossRE relation errors

We reran each of the eight native backend/prompt/domain configurations with a passive observer at the input to the original scorer. Across the 847 distinct source sentences there are 3,388 arm–sentence observations. An independent audit reconstructed gold Unicode offsets from upstream tokens and verified every missed and extra triplet, including duplicates. This recovers item-level evidence that was absent from the original aggregate reports.

| Domain | Producer | Label prompt | Matched | Missed | Unmatched predictions |
|---|---|---|---:|---:|---:|
| ai | TPLinker heuristic | training inventory | 0 | 1164 | 0 |
| ai | TPLinker heuristic | document gold labels | 0 | 1164 | 0 |
| ai | gliner_relex | training inventory | 256 | 908 | 707 |
| ai | gliner_relex | document gold labels | 680 | 484 | 571 |
| literature | TPLinker heuristic | training inventory | 0 | 1623 | 0 |
| literature | TPLinker heuristic | document gold labels | 0 | 1623 | 1 |
| literature | gliner_relex | training inventory | 614 | 1009 | 1560 |
| literature | gliner_relex | document gold labels | 964 | 659 | 696 |

The largest fixed-prompt miss category in both domains had one exact endpoint occurring elsewhere in the model’s emitted relations: 324/908 misses on AI and 391/1,009 on literature. Another 239 AI misses and 41 literature misses occurred in sentences with no predicted relation. The counts identify what the caller received and where to instrument next. They do not establish whether a missing endpoint was ever proposed by the mention detector.

The prompt comparison is paired by exact gold triplet in the same sentence. Giving a model the document’s gold label inventory recovered many relations and also lost some relations found with the training vocabulary.

| Domain | Matched under both prompts | Only training inventory | Only document-gold labels | Missed under both |
|---|---:|---:|---:|---:|
| ai | 237 | 19 | 443 | 465 |
| literature | 591 | 23 | 373 | 636 |

Document-gold labels disclose which categories occur in the answers. Treat the improvement as a prompt-information experiment. Crossing domains or selecting a model for an unknown document calls for the training-inventory results. The heuristic arms emitted zero relations in three settings and one unmatched prediction in literature under the document-gold prompt. Those zero scores establish failure of this fallback on these corpora; the trained TPLinkerPlus checkpoint was unavailable.

## Priorities supported by the evidence

1. For coreference, separately measure exact name proposal and the fate of proposed mentions with no selected links. Supplied mentions can trade missed links for many spurious links. Retain the existing native default until a candidate improves end-to-end outcomes on independent articles.
2. For JointIE, instrument which head, tail or ordered pair disappeared inside the sparse scorer. Tuning windows or the global optimizer can address only edges that reach those stages. For JEREX, test improvements to cluster membership independently from relation scoring and adjudicate a sample of the 81 forward-unannotated predictions before treating them as false facts.
3. For GLiNER-Relex, log the full mention proposals and predicate candidates on the fixed-schema cases where only one endpoint appears in the emitted relations. The new per-sentence ledger supplies exact reproducible inputs for that investigation.
4. For NuNER, separate language and length using naturally delimited short and long documents with the same annotation policy before changing the default precision or threshold.
