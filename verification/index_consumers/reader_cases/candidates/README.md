# What a spelling suggestion adds to a reader’s answer

Across five held-out source items, expanded spelling suggestions helped **40** queries retrieve their first supported occurrence and increased complete exact answers from **15 to 29 of 240** eligible queries. They also increased the number of queries showing a known wrong occurrence from **28 to 45**. Even a unique spelling candidate in the available gallery can point to a different person. The experiment keeps suggestions separate from native membership and preserves an occurrence reference and its original spelling for every addition. [Frozen design](DESIGN.md) · [Candidate policy](policy.py)

## Reader results

For a lookup on your own local data, see [Run one query](#run-one-query).

Each panel uses the same 331 known-QID queries. A query is eligible if the selected gallery has at least one other occurrence with its QID. “Exact” requires all supported occurrences and zero extras, including unscored extras. “Wrong” and “unscored” count queries showing at least one such occurrence and can overlap. “Wrong pages” counts query–page deliveries with no known support anywhere on that page and with every returned hit there verifiably wrong. A wrong hit on a page that also has gold support still counts in “wrong.” The evaluator reports such occurrences separately.

| Gallery | Policy | Eligible | Shown | Any correct | Exact | Wrong | Unscored | Wrong pages |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Other pages | Native | 269 | 224 | 215 | 18 | 53 | 7 | 35 |
| Other pages | Fold | 269 | 224 | 215 | 18 | 62 | 7 | 43 |
| Other pages | Expanded | 269 | 244 | 235 | 27 | 64 | 9 | 59 |
| Other languages | Native | 233 | 83 | 81 | 15 | 9 | 0 | 10 |
| Other languages | Fold | 233 | 92 | 90 | 15 | 9 | 0 | 10 |
| Other languages | Expanded | 233 | 138 | 125 | 26 | 20 | 2 | 39 |
| Other source items | Native | 240 | 131 | 124 | 15 | 28 | 4 | 15 |
| Other source items | Fold | 240 | 148 | 141 | 15 | 37 | 4 | 23 |
| Other source items | Expanded | 240 | 177 | 164 | 29 | 45 | 6 | 50 |

The predeclared language holdouts run each query page against pages in the two other languages. Their expanded arm gained 13 exact answers and spoiled two existing exact answers. The original design called the three files volumes. Inspection of their source metadata established five source items, so the five-item split was added after the first results, using unchanged policies. In that split each query page runs against pages of the other four items. The all-page panel replays the previously studied native baseline. All arms receive gold person spans. Candidate matching sees only original spellings and opaque native groups. [Baseline, corpus rights and grouping correction](../AJMC.md)

| Held-out source item | Eligible | Any correct, native → expanded | Exact, native → expanded | Queries with a wrong occurrence, native → expanded |
| --- | ---: | ---: | ---: | ---: |
| German, Wecklein 1894 | 53 | 27 → 44 | 0 → 1 | 18 → 27 |
| German, sophokle1v3soph | 50 | 28 → 38 | 1 → 1 | 3 → 4 |
| English, cu31924087948174 | 45 | 35 → 36 | 3 → 8 | 4 → 4 |
| English, sophoclesplaysa05campgoog | 26 | 21 → 23 | 3 → 5 | 3 → 3 |
| French, lestragdiesdeso00tourgoog | 66 | 13 → 23 | 8 → 14 | 0 → 7 |

## Which occurrences changed?

Across source-item folds, the fold arm added 174 correct and 36 wrong occurrences. The expanded arm added 241 correct, 74 wrong and two unscored. Every addition had a different raw spelling. The expanded policy abstained on 37 ambiguous abbreviation queries and two ambiguous edit queries. An ambiguity abstention suppresses only the spelling suggestion. The native result is still available.

The gate counts eligible **additional** native groups at each tier. For a query *Téra* whose own group already contains *Téra*, a single different group containing *Tera* can still yield a fold suggestion. For *Groven*, if only its own group matches at the fold tier, *Grovon* in a different group can still yield an edit suggestion. These original test cases show why a displayed suggestion can coexist with a more strongly supported native answer. A second additional group at the same tier triggers abstention. [Policy regression tests](test_policy.py)

- A single deletion linked *Achilleus* to *Achilles*, both labeled Q41746. A prefix linked *Homère* to *Hom.*, both labeled Q6691.
- The same long-s fold linked *Odyſſeus* (Q47231) to occurrences of *Odysseus* carrying Q47231 on one page and Q35160 on another. The single native donor group concealed the conflicting gold annotations. The scorer judges these against the pinned source labels.
- French *Sophocle* (Q7235) drew German *Soph.* (Q11950683). *Athena* (Q37122) drew *Athen.* with explicit NIL. The first addition is a verified mismatch under the annotations. The second supplies insufficient evidence for a correctness judgment.

These short spellings come from the AjMC team’s HIPE-2022 test annotations, released under CC BY 4.0. The scorer prints document IDs and zero-based mention indices for the first examples in each fold. Shared pages and repeated referents limit generalization. The comparison uses Python’s Unicode name-family prefix as a script proxy for its one-edit gate. A separate evaluation applies the same candidate policy to a place-name registry with known absent targets. [AjMC component attribution](https://github.com/hipe-eval/HIPE-2022-data/blob/43185e4c0886d867a6db640a478f06583043bc9e/documentation/README-ajmc.md) · [Independent confirmation](../confirmation/)

## Serving suggestions only on an empty native answer

After inspecting the results, we applied one presentation rule to the same lookup traces. It displays expanded suggestions only if the eligible gallery contains zero hits in the query’s native group. Even an unscored native hit makes the answer nonempty. Each pair of values below compares unrestricted expansion with this rule. The implementation reuses the same per-query expanded decision. All previously reported counts were reproduced.

| Gallery | Any correct | Exact | Queries with wrong hits | Queries with unscored hits | Wrong pages | Added correct occurrences | Added wrong occurrences |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Other pages | 235 → 231 | 27 → 24 | 64 → 55 | 9 → 9 | 59 → 51 | 311 → 59 | 64 → 28 |
| Other languages | 125 → 125 | 26 → 25 | 20 → 18 | 2 → 2 | 39 → 35 | 207 → 134 | 39 → 35 |
| Other source items | 164 → 160 | 29 → 25 | 45 → 45 | 6 → 6 | 50 → 42 | 241 → 149 | 74 → 56 |

In the source-item holdouts, **56 of 74** additional wrong occurrences were delivered to queries whose native answer was empty. All 17 newly wrong queries survived the gate. The language holdouts similarly retained 35 of 39 wrong occurrences. In the all-page gallery, most wrong additions went to already populated answers. A reader can use this rule to protect existing answers, but an empty result offers little assurance about the correctness of the first suggestion. This is a post-hoc comparison on reused data.

## Offline replay

Use the three checksum-verified TSVs and Rust example binary described in the baseline. The JSON output includes per-fold counts, decision tiers, unscored extras and case anchors. [Data and binary setup](../AJMC.md#replay) · [Scorer](evaluate.py)

```sh
python3 -m unittest verification.index_consumers.reader_cases.candidates.test_policy
python3 -m verification.index_consumers.reader_cases.candidates.evaluate \
  "$AJMC_DIR" target/debug/examples/cross_doc_track_partition
```

## Run one query

From the repository root, send one query and its permitted donor occurrences to the local JSON reader. Supply the caller's native group IDs as opaque strings. A query with no native assignment uses `null`. Choose `native`, `fold` or `expanded`. The latter two use the frozen supplementary gate. Each gallery locator combines a source, revision and occurrence, all strings. The caller selects the audience and revisions when constructing the gallery. A gallery row may include a source `snippet` for the caller’s own reference. Matching and output use its original `surface`. [Command implementation](read.py)

```sh
python3 -m verification.index_consumers.reader_cases.candidates.read <<'JSON'
{
  "variant": "fold",
  "query": {"surface": "Éra", "group": "g1"},
  "gallery": [
    {"source":"register","revision":"r2","occurrence":"n1","surface":"Éra","group":"g1"},
    {"source":"folio","revision":"9","occurrence":"w1","surface":"ERA","group":"g2"}
  ]
}
JSON
```

The response returns `register/r2/n1` under `native_hits` and `folio/9/w1`, spelled `ERA`, under `proposal.hits` with decision `suggested` and tier `fold`. The locators remain separate JSON fields. With an additional `ERA` occurrence in a third native group, the decision becomes `ambiguous` and the suggested hit list is empty. Both lists sort by source, revision and occurrence lexically. Invalid requests, including duplicate locators, exit with status 2.
