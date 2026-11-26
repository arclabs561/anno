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

A frozen possessive-name boundary change increased exact native name/pronoun proposals from 211 to 238 on 392 true validation pairs. It produced the same 27 correct links and one extra incorrect link. A separately sealed 128-excerpt test gained one correct link. The proposal improvement has been reproduced; the validation result gives no net correct-link gain for adopting the change by default. [Native trace and validation](native-gap-loss.md)

### Literary mentions and links

Every cell below comes from replayed cluster outputs. A missing gold pair counts once under either absent endpoint or split cluster. An extra link counts once under unannotated endpoint or merger of two annotated clusters.

| Resolver and input | Works scored | Matched pairs | Misses with absent endpoint | Misses split despite both endpoints | Extras with unannotated endpoint | Extras merging annotated clusters |
|---|---:|---:|---:|---:|---:|---:|
| native-predicted | 10 | 50 | 262 | 79 | 43 | 4 |
| native-supplied | 10 | 48 | 0 | 343 | 0 | 19 |
| fcoref-predicted | 10 | 226 | 165 | 0 | 23 | 8 |
| maverick-predicted | 10 | 278 | 105 | 8 | 60 | 17 |
| maverick-supplied | 9 | 289 | 0 | 2 | 0 | 282 |

All 165 missed FCoref pairs lack at least one exact endpoint in the final linked clusters. These outputs do not show whether the shortlist lacked the span or whether it was dropped as unlinked. Maverick supplied with gold spans illustrates the other failure mode. It missed just two pairs on nine admissible works, yet emitted 282 links between different annotated clusters. Three works accounted for 196 of those extras. Pair counts amplify the effect of merging large clusters. The refusal on *Dear Enemy* is recorded separately and accounts for the different denominator. [Original CoNLL scores and alignment policy](coref-panels.md)

## Joint models

**GLiNER2.5 JointIE.** With 384-word windows, every one of the 161 gold edges in the 64-sentence cohort fit inside a scored window. Two were outside the declared predicate vocabulary. Another 137 never appeared in the sparse output despite both typed mentions sharing a window. The cap removed 17 and the optimizer selected none of the five remaining. The model emitted two other edges. One occurred in a sentence without annotated relations; the other had just one endpoint appearing anywhere in that sentence's gold relations. The restricted endpoint schema cut the number of typed candidates by 85%, also excluded 51 gold edges at declaration, and produced zero matches from four predictions.

Short windows yielded one match among 27 predictions. Of the 26 extras, 13 had just one exact span occurring in any gold relation. One was a same-predicate reversal with the exact typed endpoints. The external JointIE replay classifies the other 12 using the same rules. A later replay traced all 137 broad-schema sparse losses to the 0.20 argument gate. Lowering it recovered additional scored candidates on a disjoint cohort and yielded zero additional correct decoded edges. [All stage counts](joint-edge-loss.md)

**JEREX.** On 32 held-out documents, 140 predicted labeled relations matched, 266 gold relations were missed, and 204 predictions were unmatched. The disjoint miss counts are 57 missing a required mention, 110 losing exact cluster membership and 99 below threshold after both clusters aligned. Of the 204 extras, 121 lacked an exact gold endpoint cluster, two used a different predicate on an annotated directed pair, and 81 had no annotated forward relation between the aligned endpoints. None of those 83 aligned-endpoint errors was explained solely by reversing the predicted predicate. A later audit matched all 32 source documents against independent Re-DocRED annotations. The revised labels include the exact directed predicate for 65 of the 81 forward-unannotated predictions. Two have another forward predicate and 14 still have no annotated forward pair. [Revised-reference audit](jerex-controls.md) [Gold- and prediction-side diagnostics](joint-edge-loss.md)

## Named entities and quantization

The historical NuNER FP32 and QUInt8 comparison processed the same 38 labeled inputs under three requested entity classes. At threshold 0.5 they missed 50 identical labeled mentions. Two other misses were unique to FP32 and one to QUInt8. They emitted 32 identical extras, plus 17 specific to FP32 and 13 specific to QUInt8. Both missed 28 of 29 gold mentions in the single long Russian stress input, which accounts for 28 of 30 no-overlap misses in each arm. That one case confounds language, length, content and text packing. Its result cannot isolate the cause or estimate performance across Russian documents.

Quantization reduced peak RSS on the measured host from 4,000 to 3,084 MiB. It changed exact predicted sets on 10 of 38 inputs in the historical decoder. The aggregate report identifies the scoring difference at FP32 ties and thresholds. Default thresholds and precision choices need evaluation on the caller's target distribution. [Labeled results and error taxonomy](nuner-4k-labeled.md)

## Runtime, provenance and unavailable checkpoints

The flat extraction CLI could emit a relation without naming the scorer that produced it. The CLI now includes the observed relation producer when available. Existing tests distinguish propagated model errors from a successful run returning zero retained edges. [CLI fix](https://github.com/arclabs561/anno/pull/369)

The TPLinker heuristic runs have no measured trained TPLinkerPlus accuracy. The original Plus decoder can be exercised on supplied pair logits. Reproducible end-to-end inference still requires compatible trained weights, tag vocabulary, preprocessing and usable terms. The Grenander and Du candidates have different missing artifacts documented in the inventory. These are availability gaps with no test-set quality denominator. [Artifact inventory](outstanding-model-artifacts.md)

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

The largest fixed-prompt miss category in both domains had one exact endpoint occurring elsewhere in the model’s emitted relations: 324/908 misses on AI and 391/1,009 on literature. Another 239 AI misses and 41 literature misses occurred in sentences with no predicted relation. The counts identify what the caller received. For a local, single-document inspection of the native ONNX path, build the CLI with `onnx,instrument` and run `RUST_LOG=anno::relex_candidates=trace anno extract --model gliner-relex --extract-types person,organization --extract-relations --relation-types founded --text "Ada founded Acme."`. In prompt order, the trace records zero-based label indices, word character offsets, span-grid slots, pre-threshold mention scores, overlap suppression, and the directed pair/predicate scores even for dropped endpoints. Keep source-derived traces in an external working directory. They do not establish whether a missing endpoint was ever proposed by the mention detector.

The prompt comparison is paired by exact gold triplet in the same sentence. Giving a model the document’s gold label inventory recovered many relations and also lost some relations found with the training vocabulary.

| Domain | Matched under both prompts | Only training inventory | Only document-gold labels | Missed under both |
|---|---:|---:|---:|---:|
| ai | 237 | 19 | 443 | 465 |
| literature | 591 | 23 | 373 | 636 |

Document-gold labels disclose which categories occur in the answers. Treat the improvement as a prompt-information experiment. Crossing domains or selecting a model for an unknown document calls for the training-inventory results. The heuristic arms emitted zero relations in three settings and one unmatched prediction in literature under the document-gold prompt. Those zero scores establish failure of this fallback on these corpora; the trained TPLinkerPlus checkpoint was unavailable.

## Priorities supported by the evidence

1. For FCoref, prioritize the learned score of the exact name span. Doubling the proposal budget changed none of the marked decisions across the independent article panel. For native coreference, predicted dependency heads improved retrieval over supplied spans across 18 languages, but yielded zero additional correct cluster pairs when applied to the spans the native resolver actually returned. [Budget trial](coref-panels.md#doubling-the-fcoref-mention-budget) · [Predicted-head trial](../../crates/anno-eval/research/reference/adjacent/multilingual/README.md#predicted-dependency-heads)
2. For JointIE, test ranking of alternative endpoint types at a fixed cap. Lowering the argument threshold alone increased scored candidates 5.4-fold without improving final recall on the second cohort. For JEREX, test improvements to cluster membership independently from relation scoring and preserve the independent reannotation when interpreting unmatched predictions.
3. On the fixed-schema Relex diagnostic slice, 49 of 73 missed relations reached both surviving endpoints and fell below the predicate threshold. Another 16 lost an endpoint to overlap suppression. On the next 32 sentences, lowering only the predicate gate gained five gold edges and 22 unmatched predictions. [Relex loss and threshold trial](relex-role-trial.md)
4. Compare NER backends within the same language, length band and label inventory. The fixed multilingual panel verifies runtime identity and reports recovered and lost exact spans. Scheduled CI exercises the licensed RED-FM input. Optional corpora and checkpoints have their own acquisition terms. [NER comparison and replay](ner-language-length.md)
