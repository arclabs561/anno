# Linking printed committee assignments to a changing congressional roster

The independent congress-legislators project publishes public committee printings and a separate roster of current members. We evaluated whether a caller could recover each printed name's upstream Bioguide record when the ID was hidden. The ID establishes the publisher's attribution. The two files share an upstream project and can incorporate the same edits. [Committee source](https://github.com/unitedstates/congress-legislators/blob/82b1eaccd2cad9e4172059507f5129fc513962af/committee-membership-current.yaml) · [Roster source](https://github.com/unitedstates/congress-legislators/blob/82b1eaccd2cad9e4172059507f5129fc513962af/legislators-current.yaml) · [CC0 1.0 license](https://github.com/unitedstates/congress-legislators/blob/82b1eaccd2cad9e4172059507f5129fc513962af/LICENSE)

We initialized both files at first-parent revision `adcd211c` (2021-01-20 15:43:50 UTC) and replayed through `82b1eacc` (2025-12-05 21:02:36 UTC). The committee file was empty at baseline. Sixty-three committee editions yielded 59 updates affecting printed assignments, 13,142 added occurrences, 9,235 removals and 837 distinct name–target pairs. Every addition had a target. None of the original committee rows changed its ID while retaining the same committee, printed name and within-name ordinal. Dates measure commits recorded by the source. A removal ends a published committee occurrence.

Each decision used the committee prefix for chamber when available and ranked the current roster's first/last and official full names by word Jaccard. It required a unique maximum. The Rust scoring function agreed with the replay on all 10,660 distinct scored name pairs. Two people appeared as “Robert Menendez,” in 45 Senate and ten House additions. Removing the chamber hint from the strict-prior 0.5 policy increased unresolved occurrences from 62 to 84.

| Policy | Correct | Wrong | Unresolved | Median / p95 scheduled delay |
|---|---:|---:|---:|---:|
| Prior roster, exact token-set match | 13,045 | 0 | 97 | 0 / 0 h |
| Prior roster, score ≥ 0.5 | 13,080 | 0 | 62 | 0 / 0 h |
| Same Git revision, score ≥ 0.5 | 13,087 | 0 | 55 | 0 / 0 h |
| Next Monday at 00:00 UTC, score ≥ 0.5 | 13,087 | 0 | 55 | 27.37 / 123.67 h |
| Prior roster, accept any unique positive score | 13,098 | 1 | 43 | 0 / 0 h |

Strict-prior queries used roster commits strictly earlier than the committee timestamp. The same-revision policy also admitted a roster update from exactly the same Git commit. It rejected an unrelated edition even with an equal timestamp. Independent period summaries cover 9,162 occurrences in 2021–2024 and 3,980 in 2025. All seven improvements at the 0.5 threshold came before 2025. The weekly and same-revision policies made identical individual decisions at both the 0.5 and exact-match thresholds. The similarity cutoffs were fixed for the replay. Repeated occurrences across committees and related publications limit generalization to other writing.

Before the September 2024 finance-committee release, a unique positive score linked “George S. Helmy” to Angus S. King, Jr. at 1/6 through the shared middle initial. The 0.5 rule deferred. That Git commit also admitted George Helmy to the roster. Re-querying the historic printing against the updated roster corrected the weak assignment. All seven conservative recoveries in the 30-day audit used roster data from the original publication instant; no transport latency was observed. This audit holds the archived occurrence fixed even if a subsequent committee edition withdraws it. [Source release](https://github.com/unitedstates/congress-legislators/commit/e6540a18a50128f015d13e733dd6182cdfd35e24)

No reviewer staffed the pilot. Under the strict-prior 0.5 rule, the unstaffed queue retained 62 cases. At five simulated perfect decisions per UTC day, the FIFO queue resolved 42 within a week. At twenty, it resolved 61 within a week. Both eventually resolved 61 by the 2026-01-01 cutoff. One target was unavailable at service time. Each occurrence used one review slot; a reviewer could select any current catalog record, including one the ranker missed. Unused slots expired at midnight. These are conditional upper bounds on review accuracy. The separate retrospective recovery audit and queue are alternative policies.

Exact source search used a fresh local disk SQLite database, applied real additions and removals in one transaction per publication, and verified a printed-name lookup on another connection. Across five full replays of 59 updates, the medians of the five per-run p50 and p95 publication-to-visible-query durations were 2.28 and 9.94 ms (Python 3.12.14, SQLite 3.53.1). Timing began after preparing each change list and excluded extraction, transport and scoring. The decoded committee JSONL has SHA-256 `1e78450e71aa73c920f2386b4af31ae12ae693199b034c89a3fdb3da0617a3e8`; the roster JSONL has SHA-256 `f6c4246cdaff7b3094ae00cf820b9ed27462970c75bda0319460e0350290a337`.

With a local source clone in `CONGRESS_REPO`, run the collector once and use the generated JSONL inputs. The example is designed to run from the anno workspace; substitute paths to these scripts when invoking from its root.

```sh
python3 extract_roster.py "$CONGRESS_REPO" /tmp/roster.jsonl
python3 committee_observations.py "$CONGRESS_REPO" /tmp/committee.jsonl
python3 committee_pilot.py /tmp/committee.jsonl /tmp/roster.jsonl --time-search --pairs /tmp/name-pairs.tsv
cargo run --locked --quiet -p anno --no-default-features --example congress_name_scores < /tmp/name-pairs.tsv
python3 -m unittest test_committee_pilot
```

The public input data use CC0 1.0. The newly authored collector, analysis, verification and examples use anno's MIT OR Apache-2.0 license.
