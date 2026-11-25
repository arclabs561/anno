# Datasets for identity and reader evaluation

The ranks below prioritize questions the contributor-credit and ROR cases leave open. Fetch source data separately for an evaluation and record the source version, selected files and checksums alongside its results. The reader fixtures include a PLOS adaptation under CC BY and original fictional cases. [PLOS correction and terms](https://doi.org/10.1371/journal.pone.0337268)

## Executed follow-ups

| Reader decision | Source and evaluation |
| --- | --- |
| Show a reviewable cross-spelling suggestion with its source occurrence | [AjMC source-item and language holdouts and local JSON reader](candidates/README.md) · [Swiss GeoNames alias confirmation](confirmation/README.md) |
| Return a shortlist or automatically assign an organization | [ROR release transitions and S2AFF labeled affiliations](lineage/retrieval/README.md) |
| Distinguish an incomplete lookup from an entity outside the dated catalog | [NILK byte-window replay and cross-page NIL clustering](open_world/README.md) |
| Open a former area code, preserve a billed artist spelling, or identify a documented role | [ONS succession and MusicBrainz billing, guest-vocal and work-credit pilots](catalog_pilots/README.md) |

| Rank | Source | Reader question | What the label establishes |
| --- | --- | --- | --- |
| 1 | HIPE-2022 historical newspapers | Can a reader group separately observed mentions and abstain on an unlinked one? | A literal or metonymic mention has an annotated Wikidata referent or a NIL label under the task’s KB convention. |
| 2 | UK Office for National Statistics (ONS) geography history | Which administrative area covered a former district after a reorganization? | A dated, source-asserted replacement between named classes of administrative areas. |
| 3 | NILK | Can a reader abstain today and still group mentions of the same later-known entity? | An entity first found in the newer of two Wikidata dumps supplies a retrospective cluster label. |
| 4 | MusicBrainz | What spelling was billed on this release, and what role did this artist have on this work? | A release has ordered artist credits. A work can separately have a typed artist relationship. |

## 1. HIPE-2022

The AjMC component also has a native, three-language test on all its official test pages. It measures retrieval with supplied person mentions. [AjMC results and offline replay](AJMC.md)

The English `hipe2020` development TSV at commit `43185e4c0886d867a6db640a478f06583043bc9e` has 80 distinct document IDs. In `NEL-LIT`, 1,162 token rows carried a single Wikidata QID and 1,613 carried `NIL`. Of 291 distinct single QIDs, 36 occurred in multiple documents. These are token counts. Reconstruct BIO spans before scoring mention-level links. Retain the document ID, date, source offsets or token positions, and separate literal and metonymic annotations. File SHA-256 is `0fedc96d176dcebefb0742c31e0d9e47acffb8544468eb72776e1f23f0920b87`. [Pinned TSV](https://github.com/hipe-eval/HIPE-2022-data/blob/43185e4c0886d867a6db640a478f06583043bc9e/data/v2.1/hipe2020/en/HIPE-2022-v2.1-hipe2020-dev-en.tsv) · [Task definitions](https://hipe-eval.github.io/HIPE-2022/tasks)

Ask which documents support one QID and whether a NIL mention pollutes that answer. Repeated `NIL` values leave their mutual identity undetermined. Report known-ID retrieval and abstention separately. Group duplicate newspaper material across splits and report repeat-entity and unseen-entity tests separately. The collection and this newspaper subset declare CC BY-NC-SA 4.0. The repository lists different terms for other components, including CC BY 4.0 for AjMC. Use the selected artifact under its terms and retain its citation. [Release and component licenses](https://github.com/hipe-eval/HIPE-2022-data/blob/43185e4c0886d867a6db640a478f06583043bc9e/README.md)

## 2. ONS geography history

In 2023 Cumberland (`E06000063`) replaced Allerdale, Carlisle and Copeland. Westmorland and Furness (`E06000064`) replaced Barrow-in-Furness, Eden and South Lakeland. Both ONS area pages explicitly identify the three former district codes. A lookup for former Carlisle should reach Cumberland. Returning both new authorities because both belong to the wider Cumbria reorganization overstates the source. A historical Carlisle record should keep its original code. Score each dated, directed source–target pair and its area type. [Cumberland](https://www.ons.gov.uk/explore-local-statistics/areas/E06000063-cumberland) · [Westmorland and Furness](https://www.ons.gov.uk/explore-local-statistics/areas/E06000064-westmorland-and-furness)

Select the December 2025 Code History Database, extract the Geography History and related metadata, and retain the release date separately from the change’s effective date. Coverage begins on 1 January 2009 and concerns changes that received new codes. Keep an entire reorganization on one side of a holdout. A contemporary hierarchy lookup alone cannot establish succession. The product’s rights field points to ONS geography licenses. ONS specifies OGL v3.0 and the credit “Source: Office for National Statistics licensed under the Open Government Licence v.3.0” for ordinary lookups. Check the specific export before redistributing any material from the mixed database. The cited area pages were checked directly; the Access tables were not sampled. [December 2025 product](https://www.data.gov.uk/dataset/4ec2169f-1a72-4c2d-be60-a74174902593/code-history-database-december-2025-for-the-uk) · [Database scope](https://www.ons.gov.uk/methodology/geography/geographicalproducts/namescodesandlookups/codehistorydatabasechd) · [Product rights pointer](https://www.arcgis.com/sharing/rest/content/items/c59bd9b324644773b92d4f06d1f48dd7?f=pjson) · [ONS conditions by product](https://www.ons.gov.uk/methodology/geography/licences)

## 3. NILK

DaRUS version 1.0 compares the 2017 and 2021 Wikidata dumps. In the first 2 MiB of `valid_1000.jsonl` (file 237900, bytes 0–2097151), 1,752 complete JSON lines included 24 with `nil: true`. All 24 carried a QID. The QID enables retrospective scoring even though the older catalog lacks the entry. The prefix establishes field semantics and supplies no estimate of benchmark-wide prevalence. The test JSONL alone is 11,660,991,447 bytes. Use a local sample for development. [Versioned deposit](https://doi.org/10.18419/darus-3454) · [File metadata](https://darus.uni-stuttgart.de/api/datasets/:persistentId/?persistentId=doi:10.18419/darus-3454)

Hide `wikidata_id`, `nil`, source hyperlink targets and newer aliases from the resolver. Build candidates from the 2017 snapshot. Score accepted links, abstentions and same-entity NIL clustering as separate decisions. Hold out source pages and target entities for the unseen-entity test. A new KB entry dates a change in catalog coverage; it gives no birth or incorporation date. The deposit labels the dataset CC BY 4.0. Its contexts contain Wikipedia prose, which carries source attribution and share-alike terms. Keep the original rights and revision information if redistributing passages. The deposit’s data license is independent of the processing code. [Dataset construction](https://github.com/iurshina/NILK) · [Wikimedia licensing terms](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content)

## 4. MusicBrainz

The official credit guide uses release `0e97456e-9471-4869-bf43-10ea495f666a` as an example where inflection expresses a collaboration. Preserve ordered artist identifiers, credited spellings and join phrases on each release. For a lyricist or translator query, evaluate the relevant artist–work relationship and record its type and target. A billing string alone supplies insufficient evidence for that query. A missing relationship leaves the role unasserted. These examples were checked against the documentation; a live API request returned HTTP 503. [Credit model and examples](https://musicbrainz.org/doc/Artist_Credits) · [Role definitions](https://musicbrainz.org/doc/Artist_Relationship_Guide_for_Artists)

Choose a dated core dump and checksum before building paired release/work examples. Keep connected release groups, reused recordings and near-identical credits together when splitting. Store evidence at release, recording or work scope as applicable. MusicBrainz publishes core artist credits and relationships under CC0. Its supplementary edit history, user annotations and ratings carry CC BY-NC-SA 3.0. A pair of core snapshots can demonstrate a changed credit, while the reason for the change needs independent evidence. [Data partitions, releases and licenses](https://musicbrainz.org/doc/MusicBrainz_Database)

## Interpreting missing links

A third meaning matters when comparing these evaluations with production services. OpenAlex reports that most author entries on works first ingested from 4 November through 19 December 2025 lack an author ID because of a matcher transition. Use the work’s ingestion date to construct this operational control. A missing profile here supports “unassigned by this provider.” HIPE’s annotated NIL and NILK’s later-resolvable NIL support different scoring decisions. Preserve the source, catalog version and reason when available. [OpenAlex author assignment history](https://help.openalex.org/data/authors/disambiguation/)
