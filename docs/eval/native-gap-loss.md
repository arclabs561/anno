# Where native GAP links disappear

The raw native baseline returned the exact pronoun and correct name for **8 of 117 true GAP pairs**. It linked six. This trace reproduces all 128 published per-excerpt coverage and decision masks, including the four false links, before attributing any loss. It uses default English mention-ranking without a NER model. Only the excerpt ID and original source text reach the Rust resolver. The scorer joins gold boundaries and links afterward. [Original panel and model comparison](coref-panels.md) · [Per-pair trace receipt](native-gap-loss.json)

| First observed outcome for a true pair | Pairs | Evidence |
|---|---:|---|
| Exact name absent at proposal | 65 | All 117 exact pronouns were proposed. |
| Exact proposal removed during overlap deduplication | 0 | Both exact endpoints survived whenever both were proposed. |
| Both exact endpoints retained, at least one absent from returned clusters | 44 | Neither an outgoing nor an incoming selected link touched each omitted endpoint. |
| Both endpoints returned in different clusters | 2 | Their character spans exist in the model output. |
| Both endpoints returned in the same cluster | 6 | Six correct positive link decisions. |

Among the 65 missing exact names, 41 were multiword names for which the detector proposed only subspans. In ten others, the detector included the English possessive clitic in the proposed span. Twelve had no overlapping proposals at sentence starts and two had none immediately after an opening parenthesis. These describe the observed source shapes and spans. A corrected boundary does not establish that the ranker would link the name to the right pronoun. GAP leaves most other spans and relations unlabeled.

The raw default uses a left-to-right linker, a 100-character distance cap and type-specific antecedent limits. Its returned clusters contain only mentions participating in a selected link. The trace records the scored candidates, the effective threshold and the selected antecedent using the same production ranking routine. Looking separately at the direct edge between the two gold endpoints after deduplication yields 35 pairs scored at or below threshold, 11 beyond the distance cap, and six selected. Either endpoint can be the later mention. An indirect path could also put the endpoints in one cluster. For an omitted endpoint, the observed absence of a qualifying earlier candidate or a selected later link does not identify which linguistic feature ought to change. No threshold adjustment follows from these counts.

To reproduce the baseline, use the version of the driver and native resolver in the commit accompanying this trace, and the pinned GAP test TSV from the [panel replay](coref-panels.md#replay).

```sh
python3 scripts/qa/native_gap_loss.py /path/to/gap-test.tsv /tmp/native-gap-loss.json
cmp /tmp/native-gap-loss.json docs/eval/native-gap-loss.json
```

The receipt includes one row per true pair with the GAP ID and the designated candidate. Rows are ordered by the original frozen panel, and include the earliest loss, each endpoint’s state, and the direct edge decision when both exact endpoints entered ranking.
