# The upstream congressional vote collector

The independent public-domain `unitedstates/congress` data collector consumes this
same legislator directory. Its House parser normally keeps the Clerk's supplied
Bioguide ID. With an empty or sentinel ID it calls a lookup parameterized by
chamber, state, party, vote date and printed name, plus an exclusion set from
earlier inferred voters. The lookup checks both current and historical rosters,
uses inclusive term-end comparison, and considers dated alternate names. A
nonunique or missing candidate returns no ID. The House parser then raises
rather than returning an unresolved voter. This project does not use anno. Its pinned collector version was replayed retrospectively against the dated inputs.
[Pinned parser](https://github.com/unitedstates/congress/blob/8184bcb13160da2389dea4c902de67817db681fc/congress/tasks/vote_info.py#L417-L476)
· [Pinned lookup](https://github.com/unitedstates/congress/blob/8184bcb13160da2389dea4c902de67817db681fc/congress/tasks/utils.py#L736-L829)
· [Upstream CC0 dedication](https://github.com/unitedstates/congress/blob/8184bcb13160da2389dea4c902de67817db681fc/LICENSE)

The original harness calls that parser on the complete House Clerk XML for
roll calls 1 and 400 of 2023. Each baseline must preserve every Clerk ID and
make zero calls to the fallback. The harness then removes exactly one voter's
ID per run, calls the same parser, restores the supplied ID, and compares the
returned ID or explicit failure to the held-out Clerk field. Each run keeps
the source's printed name, party, state and vote date unchanged. Multiple
editions of the same opening roll are repeated observations. Actual Clerk
records in these files already have IDs. The withholding probe measures
conditional fallback behavior, and supplies no estimate of real missing-ID
frequency or operator capacity.

| Published vote and roster edition | Unchanged Clerk IDs | Fallback exact with one ID removed | Parser failure | Wrong ID |
|---|---:|---:|---:|---:|
| January 3 opening roll, last available roster at voting time (`238b5227`) | 435/435 | 360/435 | 75 | 0 |
| Same January roll, roster published about 2 h 27 min after voting time (`c2bd0528`) | 435/435 | 433/435 | 2 | 0 |
| Same January roll, after March 29 directory correction (`a156ccbd`) | 435/435 | 434/435 | 1 | 0 |
| September 20 roll 400, previously published June roster (`98e8dceb`) | 433/433 | 433/433 | 0 | 0 |

The earlier roster could resolve “Garcia, Mike” to G000061. “Garcia, Robert”
failed until the new directory was available, then returned G000598. A
comparison limited to half-open, date-active terms had zero candidates on
the earlier opening-day roster, where the previous terms ended on January 3.
The external caller includes both term endpoints. It also uses historical
records, party, alternate names and given-name rules. The combined policy
differences preclude attributing the change in accuracy to any one of them.
A daily cutoff cannot order these two January publications correctly. The
Clerk recorded 12:28 Eastern and the roster update was committed at
14:54:55 Eastern.
[Opening-date source](https://clerk.house.gov/evs/2023/roll001.xml)
· [Roster update](https://github.com/unitedstates/congress-legislators/commit/c2bd05283e3dbf18dde3134df1710421c959a890)
· [September source](https://clerk.house.gov/evs/2023/roll400.xml)

The two failures after the January refresh had different causes. In that
edition, K000400's printed full name already included “Kamlager-Dove” while
the family-name field used by the lookup still read “Kamlager.” The official
roll also named McEachin under “Not Voting.” His final recorded term had
ended on November 28, 2022. Replaying the same vote after the March 29
name correction recovered K000400 and still declined M001200. A historical mention can still identify a person
without asserting that the person held office on the day of publication.
[Directory name correction](https://github.com/unitedstates/congress-legislators/commit/984ba89c3a99492e63c61a9bfbb9a92225fdb644)
· [Historical register](https://github.com/unitedstates/congress-legislators/blob/c2bd05283e3dbf18dde3134df1710421c959a890/legislators-historical.yaml)

A separate deliberate mutation swapped the two California García voters’ nonempty
IDs while preserving both printed names and the set of IDs in the roll. The
collector retained the crossed attributions with zero fallback calls. An
authentic supplied ID and a corroborated attribution are different assertions.
No error in the original Clerk records was observed.

The harness isolates the collector's YAML and pickle caches by edition. It
provides matching current *and historical* files directly from the pinned
roster Git repository. The caller's update setting disables its network
refresh. The original upstream code is fetched only when reproducing the
study. The fixture includes no upstream Python source.

After obtaining the repositories and two Clerk XML files, use Python 3.11+
in a dedicated virtual environment. The resulting parser run is offline.

```sh
git clone https://github.com/unitedstates/congress.git /tmp/vote-collector
git -C /tmp/vote-collector checkout --detach 8184bcb13160da2389dea4c902de67817db681fc
git clone --bare https://github.com/unitedstates/congress-legislators.git /tmp/congress-legislators.git
mkdir -p /tmp/house-2023
curl -fsSL https://clerk.house.gov/evs/2023/roll001.xml -o /tmp/house-2023/roll001-2023.xml
curl -fsSL https://clerk.house.gov/evs/2023/roll400.xml -o /tmp/house-2023/roll400-2023.xml
python3 -m venv /tmp/vote-collector-venv
/tmp/vote-collector-venv/bin/python -m pip install PyYAML lxml pytz scrapelib==2.4.1
/tmp/vote-collector-venv/bin/python check_collector.py /tmp/vote-collector /tmp/congress-legislators.git /tmp/house-2023
```
