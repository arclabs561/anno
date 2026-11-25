# Does the address move the right organization into view?

With one candidate inspection per affiliation, the location-aware order showed an annotated organization in 164 of 537 validation queries, up from 161 with native name similarity. Four queries gained a first-place hit and one lost it. With five inspections the orders tied at 177. In 360 queries neither five-item shortlist included an annotated target. The largest loss of coverage occurred before reranking. [Frozen protocol](protocol.txt) · [Aggregate results](context-results.json)

## Equal inspection budgets

The primary panel includes all 588 rows of the S2AFF validation split. Of these, 537 contain at least one available annotated ROR accession and 51 have no annotated ROR accession. The comparison panel is the previously studied test split of 644 rows, with 606 available-target rows and 38 rows lacking an accession. Both splits come from the same publication. All annotated IDs were active in the frozen catalog. The natural unavailable-target stratum therefore has zero observations. Thirty-six validation and 16 test rows have more than one annotated ID. A hit means at least one appears in view. [Source annotations](https://github.com/allenai/S2AFF/blob/ac8d6b58f42b253821c26fff8b013913317c5c72/data/gold_affiliation_annotations.csv)

| Available-target panel | Budget | Options shown per arm | Hits, native → location | Gained / lost | Missed, native → location |
|---|---:|---:|---:|---:|---:|
| Validation, 537 | 1 | 537 | 161 → 164 | 4 / 1 | 376 → 373 |
| | 3 | 1,602 | 170 → 171 | 1 / 0 | 367 → 366 |
| | 5 | 2,549 | 177 → 177 | 0 / 0 | 360 → 360 |
| Validation IDs absent from test, 476 | 1 | 476 | 153 → 156 | 4 / 1 | 323 → 320 |
| | 3 | 1,420 | 160 → 161 | 1 / 0 | 316 → 315 |
| | 5 | 2,256 | 164 → 164 | 0 / 0 | 312 → 312 |
| Previously studied test, 606 | 1 | 606 | 133 → 140 | 9 / 2 | 473 → 466 |
| | 3 | 1,787 | 157 → 161 | 4 / 0 | 449 → 445 |
| | 5 | 2,818 | 164 → 164 | 0 / 0 | 442 → 442 |

Options shown sum the number actually available up to each budget for every row. The count charges for all displayed options even when the first is correct. The study measures neither human review time nor whether a reader would recognize a correct candidate. The full native block included an annotated ID on 184 of 537 validation rows. Seven more were lost between the block and the first five. The location score changed the order on 86 available-target validation rows, including 77 in the entity-disjoint subset, and on 84 available-target test rows. The sensitivity subset requires every annotated ID on the validation row to be absent from the entire test split. [Aggregate results](context-results.json)

## Automatic links and unavailable organizations

| Panel | Rows | Native correct / wrong / deferred | Location correct / wrong / deferred |
|---|---:|---:|---:|
| Validation, target available | 537 | 45 / 0 / 492 | 52 / 0 / 485 |
| Validation IDs absent from test | 476 | 42 / 0 / 434 | 51 / 0 / 425 |
| Previously studied test, target available | 606 | 44 / 0 / 562 | 53 / 0 / 553 |
| Validation, all annotated owners withheld | 537 | 0 / 2 / 535 | 0 / 2 / 535 |
| Previously studied test, all annotated owners withheld | 606 | 0 / 1 / 605 | 0 / 3 / 603 |

On validation, the reranker issued ten correct links that native scoring deferred and deferred three that native scoring got right. With the gold owners withheld, both arms issued links on the same two validation rows. In the secondary test challenge the bonus triggered two additional wrong links. These are errors against the published annotations under deliberately simulated target absence. They do not estimate the prevalence of naturally absent organizations. On the 51 validation strings lacking a ROR label, each arm issued one unjudged link and deferred 50. Both deferred on all 38 test strings lacking a ROR label. These 89 rows cannot establish whether the catalog contains the organization. [Aggregate results](context-results.json)

## Ranking and checks

The original native ranker sees the complete affiliation string and the 102,742 active entries in ROR v1.1. It constructs its blocked candidates from normalized organization names covering display names, aliases, acronyms and labels. It ranks distinct owners by their best character-trigram Dice similarity and returns up to five. No annotations, split, correctness flag or other text enter its natural ranking. With five or fewer options fixed, each candidate gets 0.08 for any catalog address city in the affiliation and 0.04 for its catalog English country name. Each bonus applies once. Cap the summed score at one and keep native order on ties. Both scores use the original gate of at least 0.90 and at least a 0.10 lead over the runner-up, with zero as the runner-up for a sole option. All gate and rerank comparisons use exact rationals. These scores are heuristic. [Native retrieval](../retrieval/README.md) · [Frozen specification](protocol.txt)

Phrase matching applies the existing Unicode NFKC, casefold and whitespace key, then extracts contiguous Unicode alphanumeric runs. Punctuation separates words, including trailing periods, apostrophes, hyphens and underscores. A place must contain at least three alphabetic characters and match consecutive whole tokens. The same affiliation can supply city and country evidence. No weights or thresholds were fitted to either split. The choices were written down before computing geographic outcomes. This is an organization-linking task using published organization labels. The file provides no per-row date or byline or person-identity judgment. [Implementation freeze](implementation.txt) · [Scorer](rerank.py)

For the masked challenge the system removes **all annotated owners for the current query before generating candidates**. The Rust index reduces trigram document frequencies for each normalized alias whose complete owner set disappeared. Shared aliases with any remaining owner keep their frequency. The reduced frequencies control both the 2,048-surface cutoff and the three rare-gram choices. The scorer omits every removed owner before taking the best five distinct surviving IDs. Other eligible owners can enter the shortlist. The audit compares complete native outputs against the original executable for every natural query and against four physically rebuilt original catalogs for real masked queries. The synthetic tests also exercise crossing the 2,048 boundary and recovering candidates beyond the original fifth position. [Mask implementation](masked.rs) · [Native parity receipt](context-native-audit.json) · [Offline tests](check.py)

## Source rights and reproduction

S2AFF's annotations come from the pinned June 27, 2022 repository commit, which carries Apache 2.0. The ROR v1.1 registry comes from June 16, 2022 and is CC0. The runner checks the SHA-256 of both cached inputs on every invocation. The repository contains aggregates and receipts, with no copied affiliation strings. The optional local row ledger contains source row numbers, published IDs, candidate IDs, counts and rational scores; it omits source text. The frozen catalog predates the pinned annotation commit. Row-level affiliation publication dates and annotation completeness are unknown. [Pinned S2AFF file](https://github.com/allenai/S2AFF/blob/ac8d6b58f42b253821c26fff8b013913317c5c72/data/gold_affiliation_annotations.csv) · [Repository license](https://github.com/allenai/S2AFF/blob/ac8d6b58f42b253821c26fff8b013913317c5c72/LICENSE) · [ROR archive](https://doi.org/10.5281/zenodo.6657125) · [ROR license](https://ror.readme.io/docs/data-dump)

Run from the repository root with Python 3.11+ and a current Rust compiler. Offline regression takes no downloads. Use `--fetch` on the full evaluator to populate a new cache, or omit it to enforce local-only input. Use a fresh output directory. The complete two-arm replay with cached inputs took about ten seconds on the study host.

```sh
lineage=verification/index_consumers/reader_cases/lineage
study="$PWD/$lineage/context"
out=/tmp/anno-affiliation-context-results
cache=/tmp/anno-affiliation-cache
mkdir -p "$out"
python3 "$lineage/ngram_check.py"
python3 "$study/rerank.py" "$cache" "$out" --fetch
git show "7f752a2df7f499e8f41f309e6ee69133f6aaa3d5:$lineage/ngrams.rs" > "$out/reference.rs"
rustc --edition 2021 -O "$out/reference.rs" -o "$out/reference"
python3 "$study/audit_native.py" "$cache" "$out/reference" > "$out/context-native-audit.json"
(cd "$out" && sha256sum -c "$study/expected.sha256")
```

The final check covers the complete local ledger, aggregate JSON, original Rust source and independent audit. Natural parity compares the entire candidate block, exact alias and display hits, all ranked IDs, and the exact native score fractions. The four masked audits select the first eligible single-ID and multi-ID row from each split in source order, without reference to outcomes. [Expected digests](expected.sha256)
