# Replay the catalog publication pilot

The required index suite runs the small historical withdrawal and three checks on committee publication order offline. Run those checks from the anno repository root with Cargo and Python 3.10 or newer:

```sh
pilot=verification/index_consumers/congress
python3 "$pilot/test_committee_pilot.py"
python3 "$pilot/check_catalog_history.py"
```

The larger replay uses pinned upstream data on demand. It needs Git, Python, PyYAML, and a clone of the congressional directory with its history available. From the repository root:

```sh
pilot=verification/index_consumers/congress
scratch=$(mktemp -d)
git clone --bare https://github.com/unitedstates/congress-legislators.git "$scratch/legislators"
python3 "$pilot/extract_roster.py" "$scratch/legislators" "$scratch/roster.jsonl"
python3 "$pilot/committee_observations.py" "$scratch/legislators" "$scratch/committee.jsonl"
python3 "$pilot/committee_pilot.py" "$scratch/committee.jsonl" "$scratch/roster.jsonl" \
  --time-search --pairs "$scratch/pairs.tsv" > "$scratch/policy.json"
cargo run --quiet --locked -p anno --no-default-features --example congress_name_scores < "$scratch/pairs.tsv"
python3 "$pilot/check_rollcalls.py" "$scratch/legislators" --output-dir "$scratch/house"
cargo build --release --locked -p anno-cli --no-default-features --features extractor-readability
python3 "$pilot/check_catalog_history.py" --anno target/release/anno --roll-calls "$scratch/house"
```

For a custom Cargo target directory, pass the resulting binary's path to `--anno`. The two Clerk files are fetched from their publisher, verified by SHA-256 and projected to printed member names. To run offline, pass a local XML cache as the second positional argument to `check_rollcalls.py`. The timing includes starting the CLI on ready grounded input, validating and projecting that input, transferring JSON, committing postings and a checkpoint in SQLite, and checking a lookup through a second connection. Each trial uses a fresh database. Time to obtain the source or construct the catalog assignments is measured separately from this local path in the historical replay.

| Material | Source, reuse and receipt |
|---|---|
| Congressional directory and committee editions | Community-maintained data dedicated under CC0 1.0. The scripts pin both ends of the first-parent interval, select only fields used in the experiment and retain the source revisions. The generated JSONL stays in the caller's working directory. [Source license](https://github.com/unitedstates/congress-legislators/blob/82b1eaccd2cad9e4172059507f5129fc513962af/LICENSE) · [Expected checksums and counts](committee-findings.md) |
| Historical correction fixture | The checked fixture contains attributed CC0 structured fields from Wikidata and the congressional project, their pinned versions, links to terms, and explicitly identified pilot decisions. [Source manifest](attribution-correction.json) · [Wikidata structured data terms](https://www.wikidata.org/wiki/Wikidata:Copyright) |
| Official House roll calls | Government source files remain at the Clerk. Only aggregates, receipts and independently authored projection code are distributed here. See the source-work and jurisdiction notes in the receipt. [Clerk sources and receipts](rollcall-receipt.md) |
| Independent collector | The optional harness invokes upstream code in a separate local checkout against independently verified input. Its results describe that program's fallback when a supplied ID is withheld. [Setup and results](collector-study.md) |

The project-authored harnesses use `MIT OR Apache-2.0`. The local tests need no model weights. [Design decision and limitations](../../../docs/decisions/catalog-publication-pilot.md)

## Score native cross-document coreference

The earlier committee replay ranks candidates from a known catalog. This replay asks which unlinked document-local tracks the native `anno::coalesce::Resolver` puts together. Run it with the same two projected JSONL inputs created above:

```sh
cargo build --release --locked -p anno --no-default-features --example cross_doc_track_partition
python3 "$pilot/cross_doc_partition.py" "$scratch/roster.jsonl" "$scratch/committee.jsonl" \
  target/release/examples/cross_doc_track_partition --timing-runs 5
```

At midnight UTC on July 1 in 2021, 2023 and 2025, we reconstructed the latest published view strictly before that date. Each cut has four projected documents: the House and Senate sections of the full current roster, House Homeland Security (HSHM), and Senate Finance (SSFI). We selected these committees to exercise the known Menendez collision. Every row with a Bioguide ID and a printed name enters the evaluation. One roster occurrence lacked `official_full` and was omitted. The twelve documents contribute 1,798 occurrences for 707 publisher-assigned IDs. Publications and people recur across dates. These counts describe this selected workload.

The native process receives a projected occurrence per printed name, a supplied singleton PER track, and no catalog assignments or embeddings. Only the evaluator has the gold ID. The metric compares every unordered pair of occurrences from different documents, including negatives. There are 1,310,369 such pairs, of which 1,858 share the publisher's ID. Within-document pairs can affect the native grouping and are excluded from the evaluation denominator. Span detection, creation of local tracks, and prose understanding are outside this experiment. [Native caller](../../../crates/anno/examples/cross_doc_track_partition.rs) · [Evaluator](cross_doc_partition.py)

| Publication or reader rule | Correct links | False links | Missed links |
|---|---:|---:|---:|
| Native default threshold (0.7) | 1,810 | 12 | 48 |
| Native threshold 1.0 | 1,810 | 12 | 48 |
| Native threshold 0.5 | 1,851 | 101 | 7 |
| Native default, one independent pool per chamber across all years | 1,788 | 0 | 70 |
| Native default, divide a group by chamber only after observing a simultaneous roster conflict | 1,810 | 0 | 48 |

All twelve default false links involve the two records printed as “Robert Menendez.” Five missed pairs involve the same record printed as “Carlos A. Gimenez” and “Carlos Giménez.” Lowering the threshold recovered 41 correct pairs and added 89 incorrect ones. Twelve of the lower-threshold errors joined Mark E. Amodei to Mark E. Green. A permanent chamber boundary resolved the Menendez collision but lost another 22 pairs across ten actual transfers, including Andy Kim and Elissa Slotkin.

The last row is a post-hoc, retrospective reader experiment. It trusts differently labeled chamber sections of the same dated roster as evidence that two tracks need separate review. Exactly one native group triggered the rule. Two people can serve in the same chamber, and a namesake can coincide with a transfer. In that combined case, splitting every historical member by chamber can introduce a new error. [Scoring and policy controls](test_cross_doc_partition.py)

Resolving the entire accumulated corpus after each dated cut in forward and reverse arrival order yielded the same final memberships as bulk. Final numeric IDs differed from bulk on 1,785 and 1,798 occurrences, respectively. The median default run in five fresh optimized processes took 1.821 s on the shared four-CPU Xeon container. This duration includes pipe I/O, checked track construction, resolving and reading back identities. It excludes reconstruction of the dated views and gold scoring. The experiment appends distinct dated documents; it does not model source replacement, a catalog authority, or a latency guarantee. The input licenses and acquisition receipts are above.

### What an earlier-record lookup returns

The same command emits `native-reader` and `relaxed-reader` results from the default and relaxed native partitions. B³ precision and recall average over all 1,798 tracks, including the query track itself and same-document neighbors. B³ F1 is the harmonic mean of those two averages. At thresholds 0.7 and 0.5, B³ F1 was 0.99178 and 0.98876, respectively.

For each of the 598 occurrences in the latest cut, the simulated reader returns earlier occurrences in its predicted group. One query returning three occurrences counts as one shown page and three hits. A target may also be relevant to another query. The reference set includes 1,090 such query–target pairs across 509 queries with earlier same-ID records. The other 89 queries remain in the coverage denominator. The filter admits a whole result when its candidates span the specified number of distinct earlier dates. Multiple documents from the same date count once.

| Resolver threshold | Earlier dates required | Pages shown / 598 | Shown pages with any wrong hit | Correct hits | Wrong hits | Missed hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7 | 1 | 507 | 1 | 1,073 | 4 | 17 |
| 0.7 | 2 | 415 | 1 | 959 | 4 | 131 |
| 0.5 | 1 | 513 | 22 | 1,088 | 54 | 2 |
| 0.5 | 2 | 424 | 21 | 979 | 53 | 111 |

At 0.7 the two-date rule removed 114 correct hits. The four wrong hits survived on the same namesake page because the mistaken association appeared across both earlier dates. The dates come from one directory project. Repetition alone provides little protection against its recurring namesake error. This is a retrospective read of one partition built from all three cuts on a panel selected to include that collision. It does not estimate performance on unseen people or an online resolver with only earlier observations.

## Look up an earlier House member on held-out people

We used the first House roll call of 2003 as the prior directory and the first of 2005 as the query. We fixed those rolls, the unique-answer rule, and the native default threshold (0.7) before scoring. Every recorded member with a printed label and Clerk `name-id` was eligible, including members marked “Not Voting”. To hold out people, we removed every Bioguide ID ever present in either complete 2021–25 pilot stream from both rolls before resolving. That excluded 89 of 435 earlier rows and 101 of 434 later rows. The remaining 346 prior and 333 query rows represented 376 people, with zero ID overlap with the 724-person exclusion set. The same spelling can occur on both sides of this identity split. [2003 source roll](https://clerk.house.gov/evs/2003/roll001.xml) · [2005 source roll](https://clerk.house.gov/evs/2005/roll001.xml) · [Pilot editions and source rights](committee-findings.md)

The native resolver received a singleton PER track for each printed member label and grouped both complete documents together. For each 2005 track the reader returned a prior track only when its component contained exactly one 2003 occurrence. The scorer compared that occurrence’s Clerk ID with the query’s ID. Case-sensitive literal grouping produced the same partition. [Native caller](../../../crates/anno/examples/cross_doc_track_partition.rs) · [Aggregate receipt and source hashes](transfer-rollcalls.json)

| 2005 queries | Total | Correct unique | Wrong unique | Deferred, no prior occurrence | Deferred, multiple prior occurrences |
|---|---:|---:|---:|---:|---:|
| Target appeared in 2003 | 303 | 296 | 0 | 7 | 0 |
| Target absent from 2003 | 30 | 0 | 1 | 29 | 0 |

The wrong link joined identical printed labels attached to different IDs. All seven missed returning members had changed printed labels. For six, removing a final state abbreviation in parentheses made the two forms equal. This last comparison describes the misses after scoring. Across all 115,218 cross-roll pairs, native grouped 296 true pairs and one false pair, missed seven true pairs, and separated 114,914 other pairs. The lookup measures retrospective component membership for these two House rosters. An online reader with one arriving query at a time, a direct pair explanation, or a corpus of prose mentions has a different information set.

To reproduce the aggregate measurement, start with the two pilot JSONL files and the native example built above. The evaluator verifies the hashes of all four files and checks the membership outcomes. Only aggregate results are stored in the repository. The authored collision, ambiguity and leakage controls run in the ordinary index suite. [Evaluator](transfer_rollcalls.py) · [Offline controls](test_cross_doc_partition.py)

```sh
curl --fail --location https://clerk.house.gov/evs/2003/roll001.xml --output "$scratch/roll001-2003.xml"
curl --fail --location https://clerk.house.gov/evs/2005/roll001.xml --output "$scratch/roll001-2005.xml"
python3 "$pilot/transfer_rollcalls.py" "$scratch" target/release/examples/cross_doc_track_partition --check
```

## Check predecessor suggestions across roster corrections

With the locally reconstructed `roster.jsonl` and CLI from the replay above, run the read-only lookup against each immediately preceding edition.

```sh
python3 verification/index_consumers/congress/predecessor_lookup.py "$scratch/roster.jsonl" --anno target/release/anno
```

Each named update is queried twice, using the defaults and then `--limit 5 --minimum-score 0`. An independent word-Jaccard oracle checks the returned track handles, source revisions, text digests, surfaces, scores, withheld ties and truncation. Track IDs are renumbered at each cut. The CLI receives printed names and source coordinates. Catalog IDs stay in the evaluator for edition reconstruction and outcome scoring.

Across 80 pinned publications, the default showed the correct prior record for 26 of 32 renamed entries and offered a suggestion for 3 of 173 entries absent from the previous roster. The wider rule covered 32 of 32 renames and offered 242 candidate cards across 100 of those 173 absent entries. Together with 36 updates to existing records with unchanged or previously missing names, this covers 241 named queries. The source's catalog continuity supplies the reference labels. [Directory data license](https://github.com/unitedstates/congress-legislators/blob/82b1eaccd2cad9e4172059507f5129fc513962af/LICENSE)
