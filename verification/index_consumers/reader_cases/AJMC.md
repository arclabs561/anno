# Native person-track retrieval on classical commentaries

With gold mentions supplied, the default resolver retrieved at least one correct other-page occurrence for **215 of 269 eligible queries (79.9%)**. Only **18 of 269 (6.7%)** returned the full supported set without an extra occurrence. Every correctly grouped different-page pair in this corpus had exactly the same reconstructed spelling. A public reader can use these groups to find corroborating occurrences, but a negative lookup would miss many references to the same figure.

## Data and scoring

The Ajax MultiCommentary (AjMC) component v0.3 in HIPE-2022 v2.1 provides gold annotations on randomly chosen pages of three nineteenth-century commentaries on Sophocles’ *Ajax*. All the official unmasked German, English and French test files at upstream commit `43185e4c0886d867a6db640a478f06583043bc9e` were selected before running the native resolver. Each language represents one commentary volume. Cross-volume differences therefore combine language, author and edition effects. The sample includes mythological figures, authors, editors and others. The provider licenses this component CC BY 4.0 and each file identifies its license as CC-BY. This checkout contains the evaluator, checksums and aggregate results. [Component description, annotation rules and license](https://github.com/hipe-eval/HIPE-2022-data/blob/43185e4c0886d867a6db640a478f06583043bc9e/documentation/README-ajmc.md) · [Release](https://github.com/hipe-eval/HIPE-2022-data/releases/tag/v2.1-test-all-unmasked)

The evaluator reconstructed the 363 outer literal person spans from BIO tags, `Partial` character ranges and `NoSpaceAfter`. It supplied one unbound PER track per span on the 43 of 44 pages with a person mention. The existing Rust example received page keys and OCR spellings, with no gold QIDs, corrected transcriptions, catalog, context embeddings or previous assignments. All three files were resolved together at the unchanged default threshold 0.7. All 363 spans received a local group. Span extraction from unannotated prose and assignment to Wikidata were outside this run. [Native input and resolver invocation](../../../crates/anno/examples/cross_doc_track_partition.rs) · [Evaluator](ajmc.py)

Of 363 mentions, 331 (91.2%) carried a single gold QID. Ten had explicit NIL labels, and 22 had no link annotation (`_`), all in secondary references that AjMC excludes from linking. A local group assignment carries no catalog-membership or NIL prediction. Pair scores below use only pairs for which both mentions have a QID. All 363 mentions could affect grouping. Of 86 distinct QIDs, 30 appeared on multiple pages and 20 in multiple volumes.

| Comparison | Scorable pairs | Gold positives | Correctly grouped | Wrongly grouped | Missed positives | Correctly separate | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Different pages, including other volumes | 53,011 | 2,951 | 1,194 | 56 | 1,757 | 50,004 | 95.5% | 40.5% |
| Different volumes only | 36,286 | 1,592 | 158 | 14 | 1,434 | 34,680 | 91.9% | 9.9% |

For each of the 331 known-ID mentions, the reader returns every other-page occurrence in the same predicted group. Of these queries, 269 had another occurrence with the same gold QID and 215 had a correctly matching occurrence with the same spelling. The resolver showed results for 224 of the 331 queries. Fifty-three shown queries had at least one verifiably wrong QID, and seven had an extra occurrence whose ID could not be scored. These counts may overlap. The 18 exact answers above require agreement with the supported set including no extra unscored occurrence. No gold QID is imputed to a NIL or unannotated mention. Pairs and queries within a volume share sources and referents, so these counts do not provide independent trials for population-level confidence intervals.

## Replay

Download the three named upstream test files to a local directory. The evaluator checks their SHA-256 digests and runs offline. The commands below run from the repository root with the default Cargo target directory. Set `AJMC_DIR` to the directory containing the files. [German TSV](https://raw.githubusercontent.com/hipe-eval/HIPE-2022-data/43185e4c0886d867a6db640a478f06583043bc9e/data/v2.1/ajmc/de/HIPE-2022-v2.1-ajmc-test-de.tsv) · [English TSV](https://raw.githubusercontent.com/hipe-eval/HIPE-2022-data/43185e4c0886d867a6db640a478f06583043bc9e/data/v2.1/ajmc/en/HIPE-2022-v2.1-ajmc-test-en.tsv) · [French TSV](https://raw.githubusercontent.com/hipe-eval/HIPE-2022-data/43185e4c0886d867a6db640a478f06583043bc9e/data/v2.1/ajmc/fr/HIPE-2022-v2.1-ajmc-test-fr.tsv)

```sh
cargo build --locked -p anno --no-default-features --example cross_doc_track_partition
python3 -m verification.index_consumers.reader_cases.ajmc \
  "$AJMC_DIR" target/debug/examples/cross_doc_track_partition
```
