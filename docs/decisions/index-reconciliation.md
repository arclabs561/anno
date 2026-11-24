# Indexing mentions as evidence changes

A source mention can be searchable before a resolver knows whom it describes. Use that distinction to offer immediate text search, provisional person suggestions, and a revisioned person index. Publish a binding only for a stated use and with a way to replace or withdraw it. The experiments below exercise the existing Rust resolvers and small reference policies. They use original synthetic data and do not change production linking defaults.

## Which updates converge?

Against anno `813df66`, a harpist's “Lira Voss”, a pilot's “Lira Voss Pell” and a watch report's “Voss Pell” have adjacent word-Jaccard similarities of 2/3. The endpoints score 1/3. At threshold 0.60, the native resolver puts all three in one connected component. All six arrival orders with three flush sizes reproduced that partition. Against the independently authored roster there was one correct pair and two false pairs. Three different numeric cluster IDs represented the same final membership across runs. Comparing source occurrences and partitions made it possible to measure convergence and accuracy separately. [Native replay and gold](../../crates/anno/tests/index_reconciliation.rs) · [Resolver implementation](../../crates/anno/src/coalesce/resolver.rs)

A corrected manifest changes the pilot's name to “Voss Pell”. Across 24 delivery orders and three flush sizes, native document replacement recovered the newest authored partition in **36/72** runs. A caller that accepted only the newest source revision recovered it in **72/72**. Native replacement implements arrival order. Delivering an exact duplicate can retire a track handle despite identical source text. The small caller's revision filter deduplicates a repeated head and skips older editions; it assumes a monotone per-source revision sequence. Source revisions are separate from annotation revisions: the latter can change on identical bytes.

Decisions introduce another input. Applying the same roster entry to the harpist after one, two or three arrivals affected respectively one, two or three tracks. Every receipt correctly described what changed. An externally assigned identity is protected by the native resolver, so these different decisions legitimately lead to different later partitions. A replay equivalence test must fix the source revisions, model and configuration, and the **decision history including the approved membership**. A saved approval cannot simply be replayed against today's larger group. [Reviewed write contract](../../crates/anno/src/core/grounded/kb_assignment.rs)

For long transcripts the eval resolver has another boundary. A single call over two four-token windows linked two occurrences of Nadia and kept Omar separate. Calling the same resolver object independently for the two excerpts yielded separate chains for Nadia and reused a local number for different people. Rebasing text offsets cannot join those chains. Applications that ingest independent fragments need either a session that retains matching evidence or later document-level reconciliation. [Window experiment](../../crates/anno-eval/src/eval/incremental_coref.rs)

## What repeated resolution costs

A sequential release-mode benchmark supplied one grounded mention per document, cycling through 16 people with distinct names. Every schedule produced the same intended partition and support diagnostics. Wall-clock values are medians of five runs of the resolver calls alone on an x86-64 Xeon Platinum 8573C container, four-CPU quota, Rust 1.95. Source ingestion, querying, index I/O and building embeddings are outside the timing. Pair counts in this table are computed from the native loops and fixture sizes. Changed prior bindings were counted by comparing published snapshots. [Benchmark source](../../crates/anno/examples/index_reconcile_cost.rs)

| 256 documents, flush size | Full resolver calls | Threshold pairs | Additional within-group support pairs | Earlier postings whose numeric identity changed, summed across flushes | Resolver time |
|---|---:|---:|---:|---:|---:|
| 1 | 256 | 2,796,160 | 159,680 | 1,920 | 996 ms |
| 32 | 8 | 103,872 | 5,952 | 896 | 36.7 ms |
| 256 | 1 | 32,640 | 1,920 | 0 previously published | 11.8 ms |

At 128 documents the corresponding medians were 180, 7.26 and 3.30 ms. At 256, zero of the final numeric IDs on the two periodic schedules matched their independent bulk counterparts across 256 postings. Those IDs remain useful inside their own resolver generation. Downstream joins need a generation or an explicit remapping. These controlled inputs establish neither production throughput nor a recommended batch size.

**Targeted repair.** On a second fixture, a token-to-track lookup narrowed candidate scores from 8,128 possible pairs to 432 on the initial 128 names. Inserting one two-token bridge joined two groups of four with eight actual score calls. Removing it and rechecking its eight former neighbors took twelve. Simply dropping the bridge from lookup left the two groups fused. All three repaired partitions matched a fresh native rebuild. This bound relies on nonempty, already normalized ASCII tokens, a single type, a positive Jaccard threshold and no embeddings. With these assumptions any accepted edge must share a token; a semantic embedding can link two names without one. The fused group's weakest-pair score was zero even though every accepted edge scored at least 0.5. A membership optimization must also preserve whatever support statistics callers consume. [Runnable repair and negative control](../../crates/anno/examples/index_component_repair.rs) · [Incremental clustering and repair in FAMER](https://dbs.uni-leipzig.de/files/research/publications/2020-6/pdf/FAMER_Incremental_eswc2020.pdf)

## When should a caller wait?

The policy experiment supplies the track membership and up to three name observations per track. It uses anno's string scorer on two district-blocked candidates, and compares linking, explicitly declaring the person absent and deferring a decision. It trains score/margin thresholds on 1,152 tracks from 48 synthetic families and evaluates on 2,304 tracks from 96 disjoint families. The adaptive policy learns from development data when to wait using only the score and margin already visible. The full-track policy sees three observations. It does not optimize a joint partition. All families use the same generator, so new entities are held out while observation templates are shared. [Model, generator and oracle tests](../../crates/anno/examples/index_link_policy.rs)

| Loss units per held-out track | Decide after one observation | Buffer two | Wait for all three | Adaptive stopping |
|---|---:|---:|---:|---:|
| High delay cost | 17.35 | 19.14 | 22.93 | **14.13** |
| High false-link cost | 28.96 | 21.85 | 22.85 | **21.71** |

Weights for wrong link, false absence, review and each interval of delay were respectively (50, 35, 25, 7) and (240, 60, 30, 1). Under high delay cost, adaptive stopped after one/two/three observations for 1,223/593/488 cases. It made 180 wrong links, the same count as immediate decisions, and cut reviews from 1,081 to 345. Its loss difference from the best development-selected fixed rule was −3.22 units per track (approximate paired 95% interval −3.62 to −2.82 across test families). Under high false-link cost, adaptive matched buffering's 1,106 correct links, 69 wrong links and 823 reviews. The saving was entirely 306 wait intervals, a difference of −0.13 per track (−0.15 to −0.12). That small advantage could disappear under even modest implementation overhead.

Candidate retrieval supplied the correct record for 1,708 of 1,954 cataloged people (87.4%). Another 350 people genuinely had no catalog record. Both adaptive policies declared 113 of the 246 blocked people absent. An empty candidate set describes the search that ran; establishing catalog absence needs independent evidence. Preserve the candidates, catalog revision, retrieval policy and reason for deferral. Only call a score a probability after calibrating it against appropriate held-out outcomes. A useful next comparator is a time-aware, calibrated linker evaluated on held-out sources and time periods with separately measured blocking recall, accepted-link risk, review load and final partition error. [NIL evaluation](https://aclanthology.org/2023.findings-acl.690/) · [Multifile linkage and partition decisions](https://arxiv.org/abs/2110.03839) · [Matching records as their attributes change](https://www.vldb.org/pvldb/vol7/p469-chiang.pdf)

## Time, retractions and the consumer boundary

Separate the source edition, the time period for which a claim applies, the time the system recorded that evidence, and the generation of the published answer. An author can use a pen name and a legal name concurrently. A later correction of an earlier spelling revises a prior assertion; a real-world name change starts another effective period. The source byline remains available under its original edition. Valid/recorded time answers what a given catalog supported at a chosen cutoff. It does not settle which source to believe. [Temporal database taxonomy](https://www2.cs.arizona.edu/~rts/pubs/SIGMOD85.pdf) · [Temporal reference experiment](../../verification/temporal_identity/check_replay.py)

In a one-subject temporal fixture with supplied claims, both views received the same six receipts. A model retaining separate claim histories and evaluating both time cutoffs matched 35/35 hand-labeled date pairs across all 720 storage orders. Keeping only the latest received row for the subject matched 18/35. One old duplicate could not revive a withdrawn alias. An added conflicting body with the same source revision was rejected in all 5,040 storage permutations, without changing the answer before its recorded arrival. The arrival timestamps were fixed during permutation. The 25,200 successful queries are repeated checks of 35 expectations, not independent people.

A stream-processing watermark can trigger publication at a chosen delay. Later records can still change the answer. Keep exact search available when the source's access terms permit it, then publish identity assignments and corrections with distinct versions. Consumers must be able to remove a prior search posting or count and add its replacement exactly once in effect. Give each delivery a replayable ID, include its predecessor or snapshot generation, and advance the consumer checkpoint together with the projection. A lagging consumer that missed retained deletions needs a fresh snapshot. [Dataflow model](https://vldb.org/pvldb/vol8/p1792-Akidau.pdf) · [Differential dataflow](https://www.microsoft.com/en-us/research/publication/differential-dataflow/) · [Changelog retention](https://debezium.io/documentation/reference/stable/transformations/applying-transformations-selectively.html)

## Decision for anno

The subsequent [catalog publication pilot](catalog-publication-pilot.md) replayed historical directory, committee and House Clerk records through a Rust publication API and the shipped query command. It measures when a corrected result becomes readable on a local index.

Keep `Signal`, `Track` and `Identity` as the grounded, current view. Pilot the additional contract in an index consumer before adding public Rust types. That consumer should accept versioned source updates and withdrawals, retain anchored occurrences, expose possible matches separately from committed memberships, and publish a versioned snapshot or changes to earlier bindings. Carry the dependencies that authorize and support an attribution all the way to facets and citations. Explicit external assignments and corrections are inputs to reconciliation. This design lets a caller tune latency and review independently of the physical write or flush schedule. [Worked downstream journeys and release tests](index-consumers.md)

A deployment decision needs the caller's read-freshness budget, update distribution (including bursts, deletes and repeats), false-merge cost, available review capacity and label policy. Replay a licensed workload chronologically with burst and disconnection periods. Compare full reconciliation, bounded repair and an adaptive policy against a gold partition as well as against each other. Measure p95 time to searchable text and to corrected person results, stale postings after recovery, per-audience precision, and throughput including persistence. The synthetic timing and loss units here cannot select those budgets for a real caller.

## Publish and read a coherent collection

The publisher chooses the collection, source, audience, edition number and complete member list. An `anno::publication::IndexEdition` pins one exact native checkpoint per required stream. A zero-row selection still needs its checkpoint. A manifest with no members publishes an empty collection. A reader can use independent document streams when its query permits different refresh times.

For example, generate two full receipts from `anno query publication`, then stage and publish a release using the SQLite subscriber. Each input file here is a grounded publication envelope containing a stream, sequence and document.

```bash
anno query publication first-document.json > first-snapshot.json
anno query publication second-document.json > second-snapshot.json
edition=(anno query edition --collection program --source editor --audience public)
"${edition[@]}" manifest 12 first-snapshot.json second-snapshot.json > program-12.json
"${edition[@]}" stage catalog.sqlite first-snapshot.json
"${edition[@]}" stage catalog.sqlite second-snapshot.json
"${edition[@]}" publish catalog.sqlite program-12.json
"${edition[@]}" read catalog.sqlite 12
```

The producer can supply ordinary `--since` delta receipts to `stage`. The subscriber reconstructs the full projection, including its source marker, and checks the native fingerprint. A missing delta base can be recovered by staging the target snapshot directly. Staging can occur in any order for full snapshots. A delta requires its exact retained base. Stage scope-qualified receipts supplied by a trusted publisher. Enforce reader authorization before opening the selected audience projection.

Publication verifies the required versions and inserts the manifest in one SQLite write transaction. A `waiting` result names missing checkpoints and the currently visible edition. Retry the manifest after staging the missing versions. Reading without a number returns the latest published manifest and all its members in one read transaction. Republishing a historical manifest reports the current edition. Each read includes the edition it actually returned.

Retain versions needed for historical reads and replay. To release rows, pass a native checkpoint JSON file to `evict catalog.sqlite checkpoint.json`. The current edition's members remain pinned. Historical manifests and checkpoint identities remain available after row eviction. Reading an evicted edition reports the missing member until the exact full snapshot is restored. SQLite can reuse the freed pages. Keep an assertion history when a query needs to explain changes between published editions. In the fixture the direct endpoint delta for Blair is empty even though the intervening release credits Blair.

The [CLI replay](../../verification/index_consumers/check_editions.py) compares 540 fresh reader processes against the publisher's bulk receipts across all 90 causal delivery orders. It also kills a writer while another process holds open the previous edition. The optional `--measure` workload stages an update to one of 250 postings in each member, publishes a second edition and recovers a new subscriber from full snapshots. One local debug build produced the following measurements. Times include CLI startup, SQLite commits and native verification. The publication interval starts with delivery of the first delta. The catch-up interval starts with delivery of the first full target snapshot.

| Members | Postings per edition | Two editions of retained snapshot bodies | SQLite file | First delta to publication | Last member staged to publication | Full snapshot catch-up |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1,000 | 579,658 B | 610,304 B | 0.259 s | 0.079 s | 0.199 s |
| 24 | 6,000 | 3,484,988 B | 3,592,192 B | 1.444 s | 0.517 s | 1.423 s |

Each figure comes from one synthetic local run. Deployments choose retention from their required historical coverage, delivery lag and storage budget.

## Reproduce

From the repository root with stable Rust and Python 3.10 or newer:

```sh
bash scripts/verification/check_index_replay.sh
cargo run --locked -p anno --no-default-features --example index_link_policy
cargo run --locked --release -p anno --no-default-features --example index_reconcile_cost
cargo test --locked -p anno-eval --no-default-features --lib document_windows_share_memory_but_separate_calls_start_fresh
```

The first command runs the short checks used in CI. The policy command prints all rule choices, decisions, delays and uncertainties. The release benchmark prints host-dependent timings; CI imposes no speed threshold. The fixtures and executable examples in this study were authored for anno under its project license. The external papers and standards are cited as design sources. No model weights, outside source passages, or benchmark corpora are needed to run the experiments.
