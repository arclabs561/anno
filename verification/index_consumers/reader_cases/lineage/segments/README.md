# What changes when a long affiliation gets five suggestions?

On 1,571 distinct, verified zbMATH affiliations, the whole-string query showed a labeled organization within five options on 787 inputs (50.1%). Searching the same index with the whole string and up to three parts found one on 1,416 (90.1%). It recovered 631 previously missed inputs and displaced two previous hits. All first choices stayed the same. A reader who inspects only the top suggestion sees no benefit on this panel. [Aggregate receipt](results.json) · [Source dataset and scoring labels](https://doi.org/10.5281/zenodo.18403221)

| Inspections allowed | Whole-string hits | Segmented hits | Gained | Lost |
|---:|---:|---:|---:|---:|
| 1 | 644 | 644 | 0 | 0 |
| 3 | 765 | 1,277 | 530 | 18 |
| 5 | 787 | 1,416 | 631 | 2 |

At five, the two methods displayed 7,699 and 7,838 cards respectively. Among the 1,146 inputs whose complete sets of source-labeled organizations also had no ID in common with any S2AFF gold label, hits increased from 574 to 1,023. All 312 inputs without a usable extra part had identical suggestions under the two methods. Both found an organization on 263. These subsets overlap. The missing-target, wrong-commitment and geographic-score questions have separate controls in the preceding study. [Affiliation comparison and target withholding](../context/README.md)

The follow-up scores commitments on these verified targets, the same targets withheld, and affiliations whose source marks the organization absent. Authored controls retain independent source affiliation references for a single contributor. [Commitments and deferrals](../commitments/README.md)

## How the pairs were scored

Use the unchanged compiled Rust trigram index on all 102,742 active ROR v1.1 owners. Search the full input first, then up to three distinct nonempty comma- or semicolon-delimited parts in source order, excluding any part equal to the full input under the existing Unicode normalization. Interleave by depth, taking a whole-query candidate first, skipping repeated owners and stopping after five. Both arms use the same frozen ranking rules. This study measures suggestions displayed before an automatic or human decision. [Native ranking](../ngrams.rs) · [Executable comparison](replicate.py)

The native candidate set omitted the correct organization on 746 of its 784 top-five misses. Another 38 targets were present below position five. The segmented arm's 155 misses comprised 102 targets outside all candidate sets, 23 below every component list's five, and 30 displaced when combining those lists. Results are any-hit against the source's published IDs, including its allowance for a parent organization. These numbers measure agreement with that policy.

## Source selection and rights

Madhurima Deb, Yun Hao and Moritz Schubotz released *Gold Datasets of Verified Affiliation Strings with ROR Matches from zbMATH Open Publications*, version I, 19 March 2026, under CC BY 4.0. This study grouped the published strings, compared their labels to new predictions and reports derived measurements. The input file remains with its authors. Their G2 labels verify an earlier matcher's proposals. They mark partial or erroneous IDs, missed organizations and absent registry targets separately. Only unanimously verified positive records supply positive labels here. This selects cases another matcher already handled, and limits generalization to the broader population of mathematical affiliations. [Source and license](https://zenodo.org/records/18403221) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

The protocol fixed the two search methods, selection rule, inspection budgets and sensitivity subsets after inspecting file structure and label counts, before inspecting either method's predictions. It considers all 2,061 G2 rows from 1,008 publications. Grouping by the existing NFKC/casefold/whitespace key produced 1,905 distinct inputs. We excluded 302 groups with a non-TP or mixed source status, one with inconsistent TP label sets, five with missing or malformed IDs, 22 with no available target in ROR v1.1, and four otherwise eligible normalized strings that also occur in S2AFF. The 1,571 retained groups represent 1,703 original rows. For each group we used its first raw spelling. Two retained groups had a mixture of available and unavailable IDs. A hit requires at least one available ID. Exact normalized overlap checks cover all published S2AFF splits, including training. Publication-level overlap, paraphrases and human recognition of a correct suggestion were not measured.

The catalog and S2AFF exclusion reference are the pinned ROR v1.1 (16 June 2022, CC0) and S2AFF annotation CSV (27 June 2022 repository commit, Apache-2.0). The authors' 2026 reference may include later accessions. The runner checks all three input hashes before evaluating. The local row ledger contains numeric source row, counts, availability and ranking outcomes. [ROR release](https://doi.org/10.5281/zenodo.6657125) · [S2AFF pinned annotations and license](https://github.com/allenai/S2AFF/tree/ac8d6b58f42b253821c26fff8b013913317c5c72) · [Expected output hashes](expected.sha256)

## Search work and reproduction

The segmented arm issued 4,762 lookups for 1,571 inputs. The median number of distinct candidates per input grew from 139 to 2,133, and the 95th percentile from 849 to 4,910. The optional benchmark captures warm single-client turnaround, including Unicode normalization, each local Python/Rust pipe round trip, serialization and parsing of the native diagnostic candidate lists, and assembling the final five. It excludes catalog load and compilation. The benchmark flushes each original Rust query response and verifies that the displayed owners match the unmodified batch executable on every query in every round. It runs one untimed warmup and five timed, alternating-order pairs per input. Percentiles summarize per-input median times, equally weighting inputs. The released counts are deterministic. Timings vary with the host and do not set a CI threshold.

On the shared x86-64 container (Python 3.12.14, rustc 1.100.0, four-CPU quota), the full-query median and p95 were 0.207 and 0.970 ms. The segmented arm took 2.617 and 6.540 ms respectively. On the 312 whole-only controls, medians were 0.408 and 0.403 ms. No network calls or compilation enter these timings. A production service returning only the shortlist may have a different cost profile.

Run from the repository root with Python 3.11+ and a current rustc on the path. Download only when populating a new cache. Use a writable external output directory. Add `--timing` to generate machine-specific timing measurements.

```sh
study=verification/index_consumers/reader_cases/lineage/segments
cache=/tmp/anno-affiliation-cache
out=/tmp/anno-zbmath-segments
python3 "$study/replicate.py" --cache "$cache" --out "$out" --fetch
sha256sum "$out/rows.csv" "$out/results.json"
python3 "$study/replicate.py" --cache "$cache" --out "$out" --timing
```

Compare the two output digests to `expected.sha256`. All three input files can be reused offline.
