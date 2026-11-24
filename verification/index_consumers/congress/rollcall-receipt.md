# House voter names against the catalog available on vote day

Two official House Clerk roll calls offer an independent source of written names with source-supplied Bioguide identifiers. The opening quorum of the 118th Congress tests a term transition. A later roll call in the same Congress checks how the policy behaves away from that transition. These were deliberately selected dates. The Clerk files identify 432 of the same people on both days. Each observation is a member row, including an entry marked as not voting. These separately published sources share legislative identifiers and may share upstream information.

The matcher receives the exact printed name and the Clerk's state abbreviation. Its candidates come from a separate, dated roster. Only the evaluator reads the Clerk's identifiers. The matcher and writer do not receive them, party, district, or vote choice. Directory identifiers on candidate records provide opaque output labels. The fixed rule folds case, accents and punctuation, uses the surname and state, and, when a comma-delimited given name is printed, requires a matching first name, nickname or given portion of the catalog's full name. It abstains unless exactly one person matches. This tests directory attribution on structured mentions. Source extraction and voter identity verification are outside the experiment.

| Source | Receipt | SHA-256 of the original XML |
|---|---|---|
| [January 3 quorum, roll 1](https://clerk.house.gov/evs/2023/roll001.xml) | 12:28 Eastern, 435 occurrences | `d5600044c18f39e9d263b336b8bc88210773dac127b5529903041dfd8191b812` |
| [September 20, roll 400](https://clerk.house.gov/evs/2023/roll400.xml) | 14:16 Eastern, 433 occurrences | `7c4bf38a7cd038b3aeeed3164f4d3205e09d452740bf8e6594358fcd53fb0408` |

Each roster version is the newest first-parent commit before the Clerk's action time as recorded in the source, on history pinned through `82b1eaccd2cad9e4172059507f5129fc513962af`. The January as-of and September as-of commits are `238b522772f264f3eb59b1ee83f9308f96248a03` and `98e8dceb1a11f11efec77b616e6bc344d38e2233`. The next January change to the roster was the election update at 14:54:55 Eastern, approximately 2 h 27 min after the quorum. Treat the Git committer timestamp as observed catalog time. Clerk times have minute resolution. Network delivery and indexing latency are unmeasured by these two clocks. [Election update](https://github.com/unitedstates/congress-legislators/commit/c2bd05283e3dbf18dde3134df1710421c959a890)

| Query and roster policy | Correct | Wrong | Unresolved |
|---|---:|---:|---:|
| January as-of, enforce House term and printed name | 0 | 0 | 435 |
| January as-of, carry latest stored House term forward, surname and state | 356 | 1 | 78 |
| January as-of, carry latest stored House term forward and respect printed given names | 358 | 0 | 77 |
| Same January occurrences after the election update, enforce House term and printed name | 430 | 0 | 5 |
| September as-of, enforce House term and printed name | 430 | 0 | 3 |

The January catalog contained no House term covering the instant of the quorum under half-open term dates. Carrying names forward while dropping an explicit given name assigned “Garcia, Robert” to Mike Garcia. Preserving the full source form let the policy abstain and subsequently assign the member correctly after the catalog update. Of the five unresolved January occurrences after the update, four have different printed and directory surnames and one source-listed member is absent from the current roster file. Each roll includes 80 parenthesized state qualifiers and five non-ASCII source names. No gold identifier supplied a forecast or selected a candidate.

The optional publication envelopes contain a newline-separated projection of printed member labels under one working document per roll. Its offsets and source digest refer to these UTF-8 documents. A consumer citing the official XML must retain the original XML receipt and an occurrence-level mapping. The two January editions have identical document IDs, ordering, source text and signal IDs. Only the later resolver’s catalog memberships differ.

## Reproduce

Clone `https://github.com/unitedstates/congress-legislators.git` with its history available. Run with Python 3 and PyYAML. The adapter fetches and verifies the two Clerk XML files, or a local cache can supply both. Output consists of aggregates unless publication envelopes are requested.

```sh
python3 check_rollcalls.py /path/to/congress-legislators
python3 check_rollcalls.py /path/to/congress-legislators --output-dir /tmp/anno-house
```

The roster publisher dedicates the data under CC0. The Clerk hosts the official government record. U.S. law defines federal government works in §101 and addresses domestic copyright in §105. The pilot distributes its original code and aggregate results. It fetches the Clerk XML from the source and excludes the individual votes from all generated inputs. [Pinned roster license](https://github.com/unitedstates/congress-legislators/blob/238b522772f264f3eb59b1ee83f9308f96248a03/LICENSE) · [17 U.S.C. §§101 and 105](https://www.copyright.gov/title17/92chap1.html)
