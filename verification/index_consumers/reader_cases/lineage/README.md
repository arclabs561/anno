# Affiliation names and catalog corrections across ROR releases

A repository that saves an affiliation can receive a new name, a retired identifier, or an explicit curator correction. This study asks whether a newly printed name retrieves its catalog record from the preceding edition and what a reader of the originally cited identifier can show after the next release. Inputs are 15 consecutive Research Organization Registry (ROR) releases, v2.2 through v2.16. The ZIP archives total 522,596,398 bytes and contain the CC0 registry. The factual issue manifest contains only release numbers, ROR IDs and public issue numbers. [Source and license](https://ror.readme.io/docs/data-dump) · [Release and curation process](https://ror.readme.io/docs/updates) · [Archive manifest with SHA-256 checksums](releases.csv)

## Held-out results

Six consecutive later releases supplied 781 queries with a changed active display name. Adding earlier aliases, labels and acronyms raised the number of correct unique links from 15 to 146. Another 14 correct targets appeared in ambiguous sets. That gives candidate recall of 160/781 and 146 correct answers among 177 unique answers. Most changed names still lacked their correct ID in the candidate set.

| Later query and prior lookup | Queries | Correct unique | Wrong unique | Ambiguous | Empty |
|---|---:|---:|---:|---:|---:|
| Changed display name, earlier display only | 781 | 15 | 35 | 9 | 722 |
| Changed display name, all earlier names | 781 | 146 | 31 | 18 | 586 |
| First appearance of an active accession, earlier display only | 19,227 | 0 | 50 | 13 | 19,164 |
| First appearance of an active accession, all earlier names | 19,227 | 0 | 69 | 25 | 19,133 |
| Available sole active successor, earlier display only | 179 | 12 | 0 | 0 | 167 |
| Available sole active successor, all earlier names | 179 | 22 | 0 | 0 | 157 |
| Target named in completed duplicate issue, earlier display only | 43 | 9 | 0 | 0 | 34 |
| Target named in completed duplicate issue, all earlier names | 43 | 13 | 0 | 0 | 30 |

Across the six releases, 304 previously active records became inactive and 62 became withdrawn. Of these 366 retired records, 128 had zero active direct successors, six had several, and 232 had one. Of the 232 single active targets, 53 were unavailable in the prior active catalog. The qualified 179 form the final two rows above. The reader followed the published release state at every cutoff. Keeping prior answers would have left 426 existing ID/edition answers stale, including six whose source projection stayed identical while a referenced target changed. The updated and rebuilt readers agreed on all 858,366 later record/edition answers and on both complete name indexes. Across the eight earlier boundaries those figures were 210 stale, two from changed targets, and 1,009,678 checked. Counts accumulate across releases. All previously published IDs remained present in the next snapshot.

| Later edition | Active name changes | New active IDs | Active retirements | Stale prior answers |
|---|---:|---:|---:|---:|
| v2.11 | 31 | 3,092 | 25 | 30 |
| v2.12 | 142 | 1,650 | 58 | 63 |
| v2.13 | 120 | 4,088 | 101 | 113 |
| v2.14 | 126 | 3,122 | 26 | 33 |
| v2.15 | 139 | 1,938 | 31 | 39 |
| v2.16 | 223 | 5,337 | 125 | 148 |

All 43 explicitly named duplicate targets were active in the preceding release. All-name retrieval missed 30 of the 43. Of the 41 decisions whose relationship also appeared in the published snapshot, 13 targets were retrieved and 28 were missed. At v2.13 and v2.14 respectively, the records for the Max Planck Institute of Geoanthropology and China Telecom were withdrawn with an empty successor list despite an explicit target in each completed issue. A third issue's target for the Protein Engineering Research Institute was present and marked inactive at v2.15. The direct relationship therefore agrees with the issue while an active-successor lookup returns no result. A reader can report the published status while preserving the separate curation evidence. A current lookup cannot establish what an individual reader saw at an earlier cutoff. [Max Planck issue](https://github.com/ror-community/ror-updates/issues/39541) · [China Telecom issue](https://github.com/ror-community/ror-updates/issues/40727) · [Protein Engineering issue](https://github.com/ror-community/ror-updates/issues/42175)

## Evaluation populations

The earlier period starts with v2.2 on January 29, 2026, and ends with v2.10 on July 20. Two already-known duplicate corrections in v2.3 and v2.10 motivated choosing ROR. The later evaluation covers every intervening release from v2.11 on August 3 to v2.16 on November 19. Cohort and matching rules were fixed before examining those registry records. This is a retrospective split within a second publisher family, following the selection of ROR for its observable corrections. [Earlier correction](https://github.com/ror-community/ror-updates/issues/29292) · [Last earlier release](https://doi.org/10.5281/zenodo.21458494) · [Last later release](https://doi.org/10.5281/zenodo.24378467)

Each name query uses Unicode NFKC, case folding and collapsed whitespace. The first lookup includes only earlier active display names. The second includes every previously published name, such as labels, aliases, acronyms and names in other languages. Each distinct ROR ID counts once. The matcher returns a link only for a unique candidate. IDs, relationship targets, locations and text from the future edition are excluded from its index. Published ROR IDs and release relationships supply the scoring labels.

* **Changed active name.** For every retained ID active in both releases whose verbatim display name changed, issue its newly printed display name to the prior active index. A correct result retrieves the same ID. Identical names on different IDs can yield an ambiguity or a wrong unique match.
* **Newly listed active ID.** Query every new active ID's printed name against the preceding release. Its accession was absent from that release by construction. Count unique links to earlier IDs and ambiguous candidate sets. A new accession can refer to a previously known real organization, so this measures unseen accession handling. These labels do not measure recognition of real-world entities outside the catalog.
* **Retired active ID.** For every active record that becomes withdrawn or inactive, count its published active direct successors. Restrict name-retrieval scoring to a sole active successor that was already active in the earlier release. Query the former display name and exclude the former ID itself. A successor can continue another organization’s work. Treating general succession as equivalent identity would combine different reader decisions. Zero or multiple active targets produce a separate eligibility count.
* **Explicit duplicate.** For a smaller subset, a completed ROR curation issue explicitly identifies one ID as a duplicate of the other and prescribes withdrawing the source with that target as successor. The fixed issue retrieval searches six later release milestones for the phrase “duplicate of”. A conservative parser requires a source-ID field, a target ID in that phrase's description, a withdrawal update, and the same target in a successor directive. Additional wording, unmatched issues and unrecorded decisions leave this subset incomplete. The six searches returned 85 issues. Forty-four qualifying issues identify 43 distinct source/target/release decisions because two issues document the same transition. The issue-target cohort requires a source and target active at intake. The narrower snapshot-confirmed cohort also requires the prescribed relationship on the newly withdrawn source. [Decision manifest](duplicate_decisions.csv) · [Issue collection and selection](collect_decisions.py)

All remaining IDs with earlier and later records also serve as current-reader controls. For an originally submitted ROR ID the reader retains that key and returns the record's published status plus every direct successor with its status at this edition. A historical reference may retain a retired ID. Returning one automatic replacement would lose information for zero-, multi-target and non-equivalent successions. The evaluation checks materialized answers and candidate indexes against independently rebuilt projections at each identical release cutoff.

## Reproduction

Use Python 3.11 or newer. From this directory, the first command runs the small offline ambiguity, stale-reference, swapped-label and bulk-replay controls. The second reproduces the release counts and case evidence.

```sh
python check.py
python evaluate.py /path/to/ror-cache --fetch --verify --out /tmp/ror-results.json --cases /tmp/ror-cases.jsonl
```

The download happens only for missing files. Each pinned archive is SHA-256 checked before parsing. Re-running without `--fetch` uses a populated cache offline. The optional cases file includes every observed transition from active to inactive or withdrawn, each published target and the two candidate sets for eligible records. The file uses CC0 source data and factual issue references. The output JSON separates every release and gives both aggregate cohorts. The verifier checks a pinned digest of all counts and, when requested, the ordered per-case evidence. [Expected output digests](expected.sha256) `python collect_decisions.py --out /tmp/current-decisions.csv` repeats issue retrieval if you need to compare the pinned manifest with current GitHub text. GitHub issue text can change after release.

This Python consumer compares an update of the projected source rows and name postings with a fresh build. A target-status change also refreshes the reader answers of its predecessors. ROR publishes full snapshots. The replay derives field differences by scanning consecutive snapshots, and constructs reference answers for verification. Reader agreement establishes behavior for these projected fields and editions. The evaluation has no end-user intervention data, publication-affiliation mention offsets, or ground-truth automatic approval policy.
