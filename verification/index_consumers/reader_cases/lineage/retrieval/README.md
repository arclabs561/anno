# When a similar organization name should become a link

For 781 changed ROR names, a character-trigram search retrieved the correct earlier ID within five suggestions 522 times. Exact lookup across all earlier names included it 160 times. Automatically accepting high-scoring, well-separated matches recovered 61 correct links the exact policy had deferred. It also made 62 new errors on newly listed accessions. The cost of a wrong binding changes which policy a reader should use.

## Intake and policy

Each query searches only the active records in the immediately preceding edition. The two existing baselines accept a unique normalized display name or a unique name among all published labels, aliases and acronyms. Their rules and the original line-of-succession query stay as documented in the [exact lookup study](../README.md). The Rust retrieval process receives normalized prior names, query strings and any accession to exclude. Labels, later aliases and successor IDs stay outside retrieval. A supplied exclusion removes the cited source itself when looking for its successor.

Before observing trigram scores, we fixed the following policy and costs.

1. Normalize names with Unicode NFKC, case folding and whitespace collapse. Pad each end with two spaces and construct a set of Unicode character trigrams. Each distinct normalized name contributes once to a trigram’s document frequency.
2. Include all exact aliases. For the other candidates, take the three rarest query trigrams present in at most 2,048 indexed names. Use trigram order to resolve frequency ties. Any indexed name sharing one of the three enters the candidate set.
3. Score each surviving name by Sørensen–Dice overlap, twice the shared trigram count divided by the sum of the two set sizes. Rank distinct accessions by their best alias. Use accession order only to make equal-scoring suggestions reproducible. Keep the first five for the suggestion measure.
4. Issue a link only when the first score is at least 0.90 and exceeds the next distinct accession’s score by at least 0.10. The runner-up score is zero for a sole candidate. These thresholds use exact integer comparisons. A wrong issued link costs 10, deferring an available labeled target costs 1, and a correct answer costs zero. An abstention on a newly issued accession costs zero under this accession-level measure. These costs are illustrative.

## Registry results

The six later transitions, ending at v2.11–v2.16, reuse the population previously inspected with exact names. The separate, outcome-uninspected replication covers v2.0→v2.1 and v2.1→v2.2. Its first transition is date-disjoint from the previous study, and its final endpoint previously provided only the initial registry. Neither group establishes prospective accuracy on a future release. ROR v2.16 was the latest available at the November 25, 2026 check. [Additional CC0 release pins](../ngram-earlier.csv) · [ROR release archive](https://ror.readme.io/docs/zenodo)

| Cohort | Queries | Gold in block | Gold in top five | Display correct / wrong | All names correct / wrong | Trigram correct / wrong / deferred |
|---|---:|---:|---:|---:|---:|---:|
| Later changed active name | 781 | 632 | 522 | 15 / 35 | 146 / 31 | 206 / 28 / 547 |
| Later sole available active successor | 179 | 98 | 64 | 12 / 0 | 22 / 0 | 24 / 0 / 155 |
| Later target named by duplicate issue | 43 | 35 | 30 | 9 / 0 | 13 / 0 | 15 / 0 / 28 |
| Earlier changed active name | 66 | 55 | 42 | 0 / 0 | 3 / 2 | 12 / 5 / 49 |
| Earlier sole available active successor | 4 | 3 | 1 | 0 / 0 | 0 / 0 | 0 / 0 / 4 |

All 43 later curator-named duplicate targets were available at intake. Two directives lacked the prescribed successor in the new published source record, and another listed an inactive target. Suggestions express name similarity. The reader can show the released state and curator directive separately. The stale preceding edition would change 426 current-reference answers in the six later cuts, including six through a changed target on an untouched source. The two older transitions changed 14 answers. The date-disjoint v2.0→v2.1 transition alone contained 63 changed-name queries, 40 top-five hits, 12 correct issued links and five wrong ones. Its 418 new accessions attracted five wrong links. The successor and issue cohorts overlap. Each is scored separately. [Lineage controls and issue links](../README.md)

For each later changed name, the block contained a median of 385 distinct accessions and a 95th percentile of 2,261. Among the 19,227 newly listed IDs the corresponding counts were 477 and 2,113. A common alias can belong to several accessions. Of the later changed names, 149 targets failed to enter the block and 110 more fell below rank five. The top-ranked ID was correct for 419 queries before abstention. Compared with unique exact aliases, the gate retained 145 correct links, recovered 61, deferred one formerly correct link and suppressed three wrong ones. On newly issued accessions, however, the gate proposed 130 earlier IDs among 19,227 queries. The exact display and all-name baselines proposed 50 and 69. The gate added 62 wrong links and suppressed one. Across disjoint changed-name and new-accession cohorts, the all-name policy incurred 1,604 cost units and the gate incurred 2,127. On the two older transitions, these costs were 141 and 189 respectively, including six and nine wrong links on 1,471 newly issued accessions. A new ROR accession establishes that this exact ID was unavailable then. It cannot resolve whether the real organization previously existed under another ID.

For example, v2.13 introduced the accession for “Canon Foundation”. The gate returned the earlier “Cannon Foundation” with a score of 32/35 and next-best score of 32/44. Both exact baselines deferred. In the same release, “Dongwha Pharm (South Korea)” retrieved the curator-designated “Dong Wha Pharm (South Korea)” that exact lookup missed. Inspect the row-level predictions and cost before selecting an automatic action. [Canon Foundation](https://ror.org/00za1cb91) · [Cannon Foundation](https://ror.org/0222bz953) · [Dongwha curation issue](https://github.com/ror-community/ror-updates/issues/40399)

## External affiliation strings

We also applied the frozen policy to the 644 test rows of Allen Institute for AI’s S2AFF manual annotation file, published in the pinned June 27, 2022 commit. It receives each complete raw affiliation string without a learned parser or segmentation. The independent catalog is ROR v1.1 from June 16, before that commit. Some strings represent several organizations. A returned ID counts as correct if it is one of that row’s annotated IDs. The file supplies no per-row publication date. [Pinned annotations](https://github.com/allenai/S2AFF/blob/ac8d6b58f42b253821c26fff8b013913317c5c72/data/gold_affiliation_annotations.csv) · [Pinned ROR data](https://doi.org/10.5281/zenodo.6657125)

Of 606 rows with at least one annotated ROR ID, every row had an available target. The block included a target for 186 and the top five included one for 164. Exact display accepted 19 correct, exact aliases 32 and the gate 44. All three issued zero wrong links on this panel. The gate deferred 562 of the 606. Another 38 rows supply organization text without an explicit ROR ID. All policies deferred on all 38, and we report those rows separately. S2AFF’s original loader also separates rows lacking ROR labels. This is a check on a published annotated test split. Its input distribution and target policy differ from a registry rename. [Original evaluation loader](https://github.com/allenai/S2AFF/blob/ac8d6b58f42b253821c26fff8b013913317c5c72/s2aff/data.py)

ROR registry data are CC0. S2AFF’s pinned repository carries Apache 2.0. Its affiliation file is fetched only on request and verified by SHA-256. The external ledger records row numbers and predictions, so source text stays in its original archive. The study contains no new human ratings of these queries. [ROR data license](https://ror.readme.io/docs/data-dump) · [S2AFF repository license](https://github.com/allenai/S2AFF/blob/ac8d6b58f42b253821c26fff8b013913317c5c72/LICENSE)

## Reproduce

The offline controls require Python 3.11 or newer and a Rust compiler; the native index uses the Rust standard library. The two evaluations accept the same cache directory as the exact-name study. From the parent `lineage` directory run

```sh
python3 ngram_check.py
mkdir -p /tmp/ror-retrieval
python3 ngram_retrieval.py /path/to/cache --fetch \
  --ledger /tmp/ror-retrieval/ror-queries.jsonl --summary /tmp/ror-retrieval/ror-results.json
python3 s2aff_names.py /path/to/cache --fetch \
  --ledger /tmp/ror-retrieval/s2aff-queries.jsonl --summary /tmp/ror-retrieval/s2aff-results.json
```

Omit `--fetch` to use verified local inputs offline. The [expected SHA-256 file](../ngram-expected.sha256) pins both summaries and every query outcome. Run `sha256sum -c /absolute/path/to/lineage/ngram-expected.sha256` from the output directory to check all four files. Use `--binary` to supply an existing index executable built with `rustc --edition 2021 -O ngrams.rs`. The ROR ledger includes every test query, candidate count, labeled candidate inclusion, up to five scored suggestions, the three answers and their individual costs. The summary includes each release’s denominators, acceptance, costs, and paired outcome changes. The two evidence sets serve distinct reader tasks.
