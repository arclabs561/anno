# Independent audit of native GAP link coverage

On the original 128 GAP passages, native mention-ranking returned both exact endpoints for **8 of the 117 annotated links**. It linked 6 of those 8 pairs. The 109 remaining gold pairs lacked at least one endpoint in the returned partition. All 128 calls succeeded. Eight returned empty partitions and count toward recall. The distinction matters because an output partition alone cannot say whether a missing mention was originally proposed, pruned or intentionally suppressed as a singleton. [Per-ID evidence](gap-decision-audit.json) · [Original panel and scorer](coref-panels.md)

A second set covers 128 previously unused article URLs from the same published GAP test split. The selection rule was fixed before this replay and the article IDs were withheld from selection of changes to the native resolver. There are 118 gold links. We selected the next eligible records in the original hash order, excluding every URL in the original set and every URL already selected for the second set. This is an article-disjoint check within one test split. Exposure during model pretraining is unknown. [Pinned GAP source](https://github.com/google-research-datasets/gap-coreference/tree/83135f26025cdd705c04f77e940d0c9201116b80)

| Panel | Producer and input | Returned gold endpoint pairs / gold links | TP / FP / FN / TN | Both decisions correct |
|---|---|---:|---:|---:|
| Original | Native raw text | 8 / 117 | 6 / 4 / 111 / 135 | 16 / 128 |
| Original | Native, three supplied spans | 117 / 117 | 0 / 1 / 117 / 138 | 11 / 128 |
| Original | FCoref raw text | 76 / 117 | 73 / 6 / 44 / 133 | 82 / 128 |
| Article-disjoint | Native raw text | 11 / 118 | 7 / 2 / 111 / 136 | 16 / 128 |
| Article-disjoint | Native, three supplied spans | 118 / 118 | 1 / 1 / 117 / 137 | 11 / 128 |
| Article-disjoint | FCoref raw text | 73 / 118 | 68 / 3 / 50 / 135 | 78 / 128 |

The independent scorer reconstructs the two gold decisions from each source TSV record and validates their Unicode character offsets. It checks all five published arms against the saved per-ID observations. It also regenerates the two native partitions from the saved panel executable and derives the five observations per document from cluster membership, checking every bit against the published receipt. On the second panel it derives those bits from fresh native and FCoref partitions. Rows with neither annotated link contribute two negative decisions. The native supplied arm receives entity spans with a generic type and uses its entity-only resolver entry point. [Audit implementation](../../scripts/qa/audit_gap_decisions.py) · [Native adapter](../../crates/anno/examples/coref_gap_panel.rs)

A native candidate was frozen on the original evidence before scoring the article-disjoint set. It trims a final English possessive clitic from heuristic proper-name mentions. Against the frozen baseline on those 128 articles, the candidate increased gold-pair endpoint coverage from 11 to 14 and true links from 7 to 8, with two false links in both runs. *test-1337* gained one correct B link. No document lost a correct decision. Exact pronoun coverage fell from 63 to 62, while exact A and B coverage rose from 14 to 19 and from 19 to 21. Supplied results were unchanged on every item. The receipt includes all seven changed observations, both binary digests and the paired decision counts. This check addresses one boundary rule on English examples. [Paired candidate receipt](gap-decision-audit.json)

A successful empty partition supplies two negative predictions. A documented Maverick supplied-span alignment refusal contributes one attempted document and its attempted gold links, with no predicted decisions or scored TP/FP/FN/TN. The GAP scorer retains a per-ID refusal and rejects records marked both refused and successful. The published original GAP arms contained no refusals. The companion LitBank panel does contain one supplied-Maverick refusal, on *Dear Enemy*. Its reported supplied-Maverick CoNLL F1 covers nine accepted works. [Scorer regression](../../scripts/qa/test_coref_panels.py) · [LitBank common subset](coref-panels.md)

To replay, use the pinned GAP test TSV, native example and FCoref checkpoint described in the original panel. The runner checks data and model hashes and records the executable digest. It stores IDs, labels and membership observations without the original prose.

```sh
python3 scripts/qa/audit_gap_decisions.py gap-test.tsv target/debug/examples/coref_gap_panel \
  gap-decision-audit.json --extra-holdout fcoref/predicted --candidate /path/to/frozen-candidate
```
