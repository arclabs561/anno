# Cross-spelling confirmation on an unseen catalog

How often does a name-only spelling rule return the correct catalog entry when a reader supplies a registered alternate name? On 250 Swiss place queries, folding matched 46 entries and all 46 matched the source's place ID. Adding abbreviation and one-edit suggestions recovered five more correct entries and produced two wrong answers. All seven extra suggestions used the edit rule. Neither the abbreviation tier nor its precision was exercised here.

| Reader | Indexed queries with the right place / 200 | Wrong suggestions / covered queries | Absent-place queries receiving a suggestion / 50 | Abstentions / 250 |
| --- | ---: | ---: | ---: | ---: |
| Literal exact | 0 | Undefined (0 covered) | 0 | 250 |
| Case-insensitive exact | 0 | Undefined (0 covered) | 0 | 250 |
| Exact plus fold | 46 | 0/46 | 0 | 204 |
| Exact plus fold, abbreviation and one edit | 51 | 2/53 | 1 | 197 |

The errors trace to a single ambiguous string. Two different GeoNames records, one inside the catalog and one outside, both supply the alias “Halde”. The edit rule suggested the same unrelated catalog entry “Walde” for both. There are 250 entity queries and 249 distinct strings. A single plausible spelling in the catalog can still be the wrong answer. [Indexed query record](https://www.geonames.org/6295358/) · [Withheld query record](https://www.geonames.org/6295372/) · [Suggested record](https://www.geonames.org/7670010/)

For a reader who values one additional correct answer at one unit, the expanded policy's incremental sample utility is five minus twice the cost assigned to one wrong answer. The two policies tie at a cost of 2.5. That comparison weights each entity query equally and assigns zero to abstention and review effort. It describes this sample only. Repeated strings make the two wrong answers dependent. The observed zero errors in the fold arm covers 46 returned candidates. Natural unknown rates and the costs of involving a human remain unmeasured.

## Sampling and evidence

The query and catalog selection rule was frozen before inspecting individual rows, labels or output. A separate contributor fixed the candidate rules using a different text collection. We ran their shared [candidate policy](../candidates/policy.py) unchanged on this source. The tested policy digest and sample digests are in the [receipt](receipt.json). [Frozen selection](sampling.txt)

The 2026-11-25 GeoNames CH country export has 82,456 rows. Of 13,200 populated-place records within Switzerland, 8,712 lack a distinct nonempty alternate surface under the frozen rule and 4,488 qualify. Hash ranking sampled the first 250 eligible IDs. The first 200 enter the catalog with their primary name as the only evidence. The remaining 50 IDs are deliberately absent. Each query uses one independently hash-ranked distinct alternate name for its source record.

The gold label means this exact record listed the alias. Distinct registered IDs provide a check on returning an unrelated record. Withheld IDs give a known catalog-absence condition. The source supplies no naturally unlinked mention labels. Catalog coverage here is deliberately 200/250, and only alias-bearing places are eligible. The challenge excludes aliases literally identical to their own candidate name by design, so the zero-correct literal baseline is expected. These selected labels cannot measure performance on ordinary place mentions, raw text extraction, people, human review outcomes or identity resolution across documents.

Each query supplies its surface alone to the policy. The index exposes a primary name and an opaque donor key per included place. The evaluator preserves direct case-insensitive exact candidates and adds the helper's suggestions. An additional literal arm checks case-sensitive equality. Query labels, coordinates, alternate lists, withheld canonical names and GeoNames IDs are absent from the helper inputs. Scoring uses the reserved source ID afterward. Abstentions, wrong-only and mixed candidate sets, indexed and absent query coverage, and candidate counts are recorded for each arm. All returned sets in this sample were singletons.

## Reproduction and attribution

Run the default regression check offline. It recomputes choices from the included attributed projection and checks every arm's metrics. To verify frozen selection from the original country file, supply the dated archive. The public download gets replaced daily, so a newer source can have a different digest.

```sh
python3 verification/index_consumers/reader_cases/confirmation/check.py
python3 verification/index_consumers/reader_cases/confirmation/check.py --source /path/to/CH.zip
```

[places.tsv](places.tsv) is derived from GeoNames and distributed under Creative Commons Attribution 4.0. Changes include feature filtering, sampling, alias selection and retaining only the ID, catalog name for included records and selected alternate name. GeoNames supplies and aggregates the underlying data. The archive and extracted text SHA-256 digests identify the original inputs. The projected data retain the source license. [GeoNames](https://www.geonames.org/) · [Official CH export](https://download.geonames.org/export/dump/CH.zip) · [GeoNames field definitions and license](https://download.geonames.org/export/dump/readme.txt) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
