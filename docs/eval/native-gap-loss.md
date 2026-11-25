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

The raw default uses a left-to-right linker, a 100-character distance cap and type-specific antecedent limits. Its returned clusters contain only mentions participating in a selected link. The trace records the scored candidates, the effective threshold and the selected antecedent using the same production ranking routine. Looking separately at the direct edge between the two gold endpoints after deduplication yields 35 pairs scored at or below threshold, 11 beyond the distance cap, and six selected. Within the 44 pairs missing a returned endpoint, 34 direct edges scored at or below threshold and ten lay beyond the distance cap. Of the two pairs returned separately, one fell into each category. Either endpoint can be the later mention. An indirect path could also put the endpoints in one cluster. For an omitted endpoint, the observed absence of a qualifying earlier candidate or a selected later link does not identify which linguistic feature ought to change. No threshold adjustment follows from these counts.

To reproduce the baseline, use the version of the driver and native resolver in the commit accompanying this trace, and the pinned GAP test TSV from the [panel replay](coref-panels.md#replay).

```sh
python3 scripts/qa/native_gap_loss.py /path/to/gap-test.tsv /tmp/native-gap-loss.json
cmp /tmp/native-gap-loss.json docs/eval/native-gap-loss.json
```

The receipt includes one row per true pair with the GAP ID and the designated candidate. Rows are ordered by the original frozen panel, and include the earliest loss, each endpoint’s state, and the direct edge decision when both exact endpoints entered ranking.

## Removing English possessive clitics from proposed names

Ten of the diagnostic panel's missing name proposals included their following possessive marker. We froze one change to the English proper-name heuristic before scoring new excerpts. After its existing trailing-punctuation cleanup, the candidate strips a final ASCII or curly apostrophe followed by lowercase `s`. It preserves internal apostrophes, as in `O'Neil's`, and leaves ranking, multiword detection and sentence-initial detection alone. The fixture includes a character-boundary regression covering both apostrophe forms and Unicode names. [Frozen candidate](../../scripts/qa/fixtures/native-gap-possessive.patch)

All 454 rows in the upstream validation split qualified for evaluation. They represent 328 distinct article URLs, none shared with the full test split. Every row contains the same original text in each arm. The resolver receives no gold boundaries. [Validation split](https://github.com/google-research-datasets/gap-coreference/blob/83135f26025cdd705c04f77e940d0c9201116b80/gap-validation.tsv) · [Results and changed target rows](native-gap-possessives.json)

| Validation measure | Baseline | Possessive candidate |
|---|---:|---:|
| True pairs with both exact boundaries proposed | 211/392 | 238/392 |
| True pairs with both exact boundaries in output clusters | 48/392 | 50/392 |
| Correct pronoun–name links | 27 | 27 |
| Incorrect pronoun–name links | 40 | 41 |
| Distinct returned mentions, summed by excerpt | 2,691 | 2,798 |
| Exact two-label decisions | 80/454 | 80/454 |

On the separately sealed 128-excerpt test cohort with fresh article URLs, returned gold pairs increased from 11 to 14 of 118. Correct links increased from seven to eight and false links stayed at two. The ten LitBank excerpts had identical raw native mention and cluster scores in both arms, including 71 exact mention matches among 118 outputs and CoNLL F1 of 0.314809. The validation panel shows a proposal gain, a smaller gain in observable linked endpoints and no net gain in correct links. GAP cannot classify most additional output mentions as correct or false. Lexicalized names can include a final apostrophe-s. These results support retaining the existing default and keeping clitic handling as a testable candidate.

To replay the candidate, build the native test executable at the baseline trace revision and copy the executable path printed by Cargo to a temporary directory. In a separate worktree at the same revision, apply the [candidate patch](../../scripts/qa/fixtures/native-gap-possessive.patch), run the mention-ranking tests and copy that test executable as well. Run the scorer with the two pinned GAP TSVs and the two binaries.

```sh
cargo test --locked -p anno --lib --no-default-features --no-run
# In the candidate worktree:
git apply scripts/qa/fixtures/native-gap-possessive.patch
cargo test --locked -p anno --lib --no-default-features mention_ranking
python3 scripts/qa/native_gap_possessives.py gap-test.tsv gap-validation.tsv \
  /tmp/baseline-test /tmp/candidate-test /tmp/native-gap-possessives.json
cmp /tmp/native-gap-possessives.json docs/eval/native-gap-possessives.json
```

The three proposal bits in each changed validation row refer to the pronoun, A and B. The five output bits add the two pronoun–candidate link decisions to those three returned-endpoint bits. A changed row can reflect a change in proposal coverage even when its emitted decisions match exactly.
