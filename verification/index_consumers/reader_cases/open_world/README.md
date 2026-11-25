# When an unassigned mention acquires a catalog ID

NILK provides a case that a deleted catalog row or an artificially withheld person cannot. Its older and newer Wikidata dumps let the authors mark a mention as outside the earlier knowledge base (NIL) and also give it a later ID. The later ID lets an evaluator check whether two such mentions belong together without granting the reader that ID at the earlier cut. [Dataset and terms](https://doi.org/10.18419/darus-3454) · [Iurshina et al., CIKM 2022](https://doi.org/10.1145/3511808.3557659) · [Upstream construction](https://github.com/iurshina/NILK)

We froze [the protocol](sampling.txt) before examining the training and test windows. Each split uses four deterministic 2 MiB byte windows of the official JSONL. We indexed only the printed spelling and labeled ID of non-NIL training mentions. Of 7,412 test records, 384 shared a source page with this training sample and were excluded. The resulting 7,028 queries include one empty mention, which was retained and yielded no candidate. No observation IDs were duplicated within either set. The input windows group adjacent records, so these counts describe this panel.

| Decision | Exact spelling | Casefold, decompose and remove combining marks |
|---|---:|---:|
| Correct unique catalog links emitted | 0 | 0 |
| Wrong unique links on earlier-catalog entities | 116 | 120 |
| Wrong unique links on earlier-catalog NIL | 1 | 1 |
| Cross-page NIL pairs correctly grouped, given the NIL labels (30 available) | 11 | 21 |
| Wrong cross-page NIL pairs grouped in these windows | 0 | 0 |

The index contained 5,275 distinct IDs. **None of the 6,925 earlier-catalog non-NIL queries had its target in this sampled index.** These windows therefore test what a small donor catalog does when the target is absent. They cannot estimate a linker's sensitivity when the correct record is available. A system defining NIL as an empty spelling lookup would conflate missing donor coverage with the provider's 103 genuine temporal NIL labels. Every one of the 103 had a later ID available to the scorer. Clustering pairs conditional on a correct NIL decision is a separate problem. Folding found ten more correct pairs here; it also assigned four additional wrong earlier-catalog IDs in the lookup task.

## Source replay

Run from the repository root. The first command downloads 16 MiB into the chosen external cache, validates the HTTP byte ranges and verifies all eight SHA-256 digests. Omit `--fetch` for an offline replay. The second uses only original synthetic input, including a known positive and two different entities with the same normalized name.

```sh
python3 -m verification.index_consumers.reader_cases.open_world.replay /tmp/anno-nilk --fetch
python3 -m unittest verification.index_consumers.reader_cases.open_world.test_replay
```

The [result and range hashes](receipt.json) contain aggregate measurements. The DaRUS version 1.0 deposit by Iurshina et al. lists CC BY 4.0. The JSONL includes Wikipedia prose subject to its source terms. Downloaded windows stay in the local cache. The repository distributes the original evaluator, synthetic controls and aggregates. [Publisher file metadata](https://darus.uni-stuttgart.de/api/datasets/:persistentId/?persistentId=doi:10.18419/darus-3454) · [Wikimedia content licensing](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content)

## Diagnostic with reachable targets

After the official train/test comparison returned zero reachable targets, we froze a second protocol before scoring it. We reused the four cached official **test** windows, assigned source pages to two folds by a fixed SHA-256 hash, and used each fold in turn to index known mentions for queries on the opposite fold. This scores 7,412 occurrences on 1,871 pages exactly once, including the 384 occurrences excluded from the earlier comparison because they shared an official training page. The primary comparison restricts donors to the same 2 MiB window. The secondary comparison pools donor pages from all four windows. [Frozen page protocol](page_control_design.txt) · [Page-control evaluator](page_control.py)

| Gallery and stratum | Queries | Exact correct | Fold correct | Exact wrong | Fold wrong | Exact ambiguous | Fold ambiguous |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Same window, known target present | 1,512 | 1,194 | 1,203 | 2 | 2 | 90 | 90 |
| Same window, known target absent | 5,792 | 0 | 0 | 14 | 14 | 4 | 4 |
| Same window, source temporal NIL | 108 | 0 | 0 | 0 | 0 | 0 | 0 |
| Pooled windows, known target present | 2,085 | 1,614 | 1,678 | 13 | 13 | 144 | 144 |
| Pooled windows, known target absent | 5,219 | 0 | 0 | 33 | 33 | 24 | 24 |
| Pooled windows, source temporal NIL | 108 | 0 | 0 | 0 | 0 | 0 | 0 |

All remaining decisions had no matching spelling. Among reachable targets in the same window (291 distinct IDs), exact and folded spelling recovered 78.97% and 79.56% of mentions respectively. Their precision across *all emitted links* was 98.68% and 98.69%. Pooling exposed 402 distinct reachable IDs. Sensitivity there was 77.41% and 80.48%, with precision of 97.23% and 97.33%. All 108 source-labeled NIL mentions yielded no match in both conditions. The 16 or 46 wrong answers instead came from earlier-known targets, some of which were present in the gallery.

This is a retrospective control for candidate availability. Neighboring records within each byte window can share topics and spellings, even across mutually exclusive page folds. The reader receives previously labeled mentions from that test file as donors. These measurements answer a different question from transfer across the official train/test boundary and cannot establish production sensitivity or NIL prevalence. [Dataset construction](https://doi.org/10.18419/darus-3454)

After the original source replay has filled the cache, run the control on demand. It uses the same digest-verified window reader and exact/fold scorer. Its complete aggregate output is checked against the SHA-256 of an independent implementation’s result.

```sh
python3 -m verification.index_consumers.reader_cases.open_world.page_control /tmp/anno-nilk
```
