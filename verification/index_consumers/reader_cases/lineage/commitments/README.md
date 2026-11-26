# When does an affiliation suggestion support a committed link?

The existing [trigram study gate](../retrieval/README.md) selects the leading organization when its whole-string Dice score reaches 0.90 and its lead reaches 0.10. We compared that rule to always selecting the first available suggestion. On the 1,571 distinct [verified positive inputs](../segments/README.md), the gate committed 92 correct links and deferred the other 1,479. When all their source-labeled owners were withheld before candidate generation, the same gate deferred on every row. Both arms searched the unmodified ROR v1.1 study index. The numbers describe these research policies. The official ROR API has its own selection indicator. [Aggregate receipt](results.json) · [ROR matching guidance](https://ror.readme.io/docs/matching)

| Panel | Inputs | First suggestion, correct / wrong / deferred | Gate, correct / wrong / deferred |
|---|---:|---:|---:|
| Verified available target | 1,571 | 644 / 927 / 0 | 92 / 0 / 1,479 |
| Same inputs, all labeled owners withheld | 1,571 | 0 / 1,571 / 0 | 0 / 0 / 1,571 |
| Source marks affiliation absent from its registry | 34 | 0 / 34 / 0 | 0 / 1 / 33 |

The additional 34 are distinct nonempty G2 strings unanimously marked TN, with no normalized query overlap with S2AFF. G2 defines TN relative to the authors’ registry version and accepts an exact or parent organization as a positive. The frozen index contains 102,742 active organizations from June 2022. On row 798, a generic department name exactly matched the display name of record 02xkdev90. Its score was 72/72 and the runner-up scored 56/68. The gate therefore disagreed with the source's absence judgment. That row alone cannot identify the author’s institution. Inputs without positive or negative judgments stay outside this comparison. [Source definitions](https://doi.org/10.5281/zenodo.18403221) · [Candidate record](https://ror.org/02xkdev90)

## Multiple affiliations on one byline

The [authored fixture](fixture.json) supplies 12 bylines and 18 separately asserted affiliation references. We queried those source fields independently and compared them with the same text joined by semicolons. The per-field gate committed 11 correct references, committed two wrong references and deferred five. The two wrong answers repeat the same mechanism. An absent organization shares its printed name with an active catalog owner. For an ambiguous alias shared by two active owners the gate instead deferred on both appearances. On the two fully resolvable multi-affiliation bylines it recovered all two and all three owners respectively. The flattened gate deferred on both. These constructed cases test behavior under known source boundaries. A real publication can link one contributor to independently identified affiliations. [JATS affiliation examples](https://jats.nlm.nih.gov/publishing/tag-library/1.4/element/aff.html)

## Reproduction and rights

The small namesake, masked-identity and multiple-reference controls run in the offline index verification track. For the public panel, use Python 3.11+ and rustc. Add `--fetch` to populate a new cache with the pinned inputs. The output directory receives the aggregates, numeric source-row ledger, and authored per-byline receipts.

~~~sh
lineage=verification/index_consumers/reader_cases/lineage
study="$PWD/$lineage/commitments"
python3 "$lineage/ngram_check.py"
python3 "$study/replicate.py" --cache /path/to/ror-cache --out /tmp/anno-commitments
(cd /tmp/anno-commitments && sha256sum -c "$study/expected.sha256")
~~~

Madhurima Deb, Yun Hao and Moritz Schubotz published the G2 labels under CC BY 4.0. The reported numbers are new grouped evaluations of their data. ROR supplies the CC0 catalog. Allen Institute for AI supplies the Apache-2.0 S2AFF exclusion reference. The fixture and scripts are originally authored under MIT OR Apache-2.0. The pinned inputs remain with their publishers. [G2 and license](https://doi.org/10.5281/zenodo.18403221) · [ROR v1.1](https://doi.org/10.5281/zenodo.6657125) · [S2AFF source and license](https://github.com/allenai/S2AFF/tree/ac8d6b58f42b253821c26fff8b013913317c5c72)
