# Independent audit of native GAP link coverage

On the original 128 GAP passages, native mention-ranking returned both exact endpoints for **8 of the 117 annotated links**. It linked 6 of those 8 pairs. The 109 remaining gold pairs lacked at least one endpoint in the returned partition. All 128 calls succeeded. Eight returned empty partitions and count toward recall. The distinction matters because an output partition alone cannot say whether a missing mention was originally proposed, pruned or intentionally suppressed as a singleton. [Aggregate evidence](gap-decision-audit.json) · [Original panel and scorer](coref-panels.md)

A second set covers 128 article URLs outside the original panel, from the same published GAP test split. The selection rule was fixed before this replay and the article IDs were withheld from selection of changes to the native resolver. There are 118 gold links. We selected the next eligible records in the original hash order, excluding every URL in the original set and every URL already selected for the second set. These two native panels are article-disjoint within one test split. Comparing the second panel with the 256 independent FCoref controls in issue #288 gives eight shared rows and ten shared article URLs. Cross-study totals would count some articles twice. Exposure during model pretraining is unknown. [Pinned GAP source](https://github.com/google-research-datasets/gap-coreference/tree/83135f26025cdd705c04f77e940d0c9201116b80)

| Panel | Producer and input | Returned gold endpoint pairs / gold links | TP / FP / FN / TN | Both decisions correct |
|---|---|---:|---:|---:|
| Original | Native raw text | 8 / 117 | 6 / 4 / 111 / 135 | 16 / 128 |
| Original | Native, three supplied spans | 117 / 117 | 0 / 1 / 117 / 138 | 11 / 128 |
| Original | FCoref raw text | 76 / 117 | 73 / 6 / 44 / 133 | 82 / 128 |
| Article-disjoint | Native raw text | 11 / 118 | 7 / 2 / 111 / 136 | 16 / 128 |
| Article-disjoint | Native, three supplied spans | 118 / 118 | 1 / 1 / 117 / 137 | 11 / 128 |
| Article-disjoint | FCoref raw text | 73 / 118 | 68 / 3 / 50 / 135 | 78 / 128 |

The independent scorer reconstructs the two gold decisions from each source TSV record and validates their Unicode character offsets. It checks all five published arms against the saved per-ID observations. It also regenerates the two native partitions from the saved panel executable and derives the five observations per document from cluster membership, checking every bit against the published receipt. On the second panel it derives those bits from fresh native and FCoref partitions. Rows with neither annotated link contribute two negative decisions. The native supplied arm receives entity spans with a generic type and uses its entity-only resolver entry point. [Audit implementation](../../scripts/qa/audit_gap_decisions.py) · [Native adapter](../../crates/anno/examples/coref_gap_panel.rs)

A native candidate was frozen on the original evidence before scoring the article-disjoint set. It trims a final English possessive clitic from heuristic proper-name mentions. Against the frozen baseline on those 128 articles, the candidate increased gold-pair endpoint coverage from 11 to 14 and true links from 7 to 8, with two false links in both runs. One additional document gained a correct link. No document lost a correct decision. Exact pronoun coverage fell from 63 to 62, while exact A and B coverage rose from 14 to 19 and from 19 to 21. Supplied results were unchanged on every item. The receipt includes the number and digest of the seven changed observations, the executable digests and paired decision counts. This check addresses one boundary rule on English examples. [Paired candidate receipt](gap-decision-audit.json)

A successful empty partition supplies two negative predictions. A documented Maverick supplied-span alignment refusal contributes one attempted document and its attempted gold links, with no predicted decisions or scored TP/FP/FN/TN. The GAP scorer retains a per-ID refusal and rejects records marked both refused and successful. The published original GAP arms contained no refusals. The companion LitBank panel does contain one supplied-Maverick refusal, on *Dear Enemy*. Its reported supplied-Maverick CoNLL F1 covers nine accepted works. [Scorer regression](../../scripts/qa/test_coref_panels.py) · [LitBank common subset](coref-panels.md)

To replay, use the pinned GAP test TSV, native example and FCoref checkpoint described in the original panel. The runner verifies the data and model file hashes, records the executable digests and emits the aggregate receipt. The FCoref encoder in this panel has SHA-256 `74b98b0f03b27154065d0918a566ed5968d14665e601ecaea6905014bfe6f065`. Issue #288 traces another FP32 export on a different selection. Its stage counts describe that separate run. Both GAP source splits contain only ASCII article text. [Independent FCoref proposal audit](https://github.com/arclabs561/anno/issues/288#issuecomment-7176979503)

```sh
python3 scripts/qa/audit_gap_decisions.py gap-test.tsv target/debug/examples/coref_gap_panel \
  /tmp/gap-decision-audit.json --extra-holdout fcoref/predicted --candidate /path/to/frozen-candidate
```


To compare rebuilds, compare the `panels` and `article_holdout_candidate.arms` fields with the tracked receipt. Executable digests identify the build used. Add `--local-details /tmp/gap-decision-details.json` to inspect rows on your machine.

## Source and rights

The GAP benchmark is by Webster et al. (2018). Google Research publishes the repository under Apache-2.0 and identifies Wikipedia as the source of the excerpts. Each upstream TSV records its source articles. Obtain the TSVs directly from the pinned repository and consult the article attribution and reuse terms before sharing any prose. The three new GAP receipts publish aggregate measurements and whole-panel digests. IDs, gold decisions, source spans and per-document observations from these replays stay in local working files. The digest for a structured result uses SHA-256 over compact, key-sorted ASCII JSON in original row order. [GAP release](https://github.com/google-research-datasets/gap-coreference/tree/83135f26025cdd705c04f77e940d0c9201116b80) · [Upstream license](https://github.com/google-research-datasets/gap-coreference/blob/83135f26025cdd705c04f77e940d0c9201116b80/LICENSE) · [Webster et al.](https://aclanthology.org/Q18-1042/) · [Wikimedia terms of use](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use)
