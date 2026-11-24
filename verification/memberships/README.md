# Publishing cross-document track memberships

A source occurrence can keep its text and acquire different neighbors after another document changes. Publish a complete corpus edition through `anno query memberships`. A full result contains a checkpoint and ordered `[key, row]` pairs. The source key consists of document ID, SHA-256 of the document text, optional source revision, and the source's submitted track ID. A later revision of identical text is distinguishable when its publisher supplies a distinct source revision.

```sh
anno query memberships edition-2.json
anno query memberships edition-2.json --since edition-1.json
anno query memberships edition-2.json --neighbors transcription-a 0
```

Each input contains a nonempty stream, numeric sequence and threshold, boolean `require_type_match`, and a `documents` array. Each element contains one `document` in GroundedDocument JSON format and an optional `source_revision`. Document IDs must be unique within an edition. Withdrawing a document means omitting it from the next complete edition. An empty corpus publishes an authoritative empty membership result. The checkpoint scopes the stream to this projection version, threshold and type policy. Include the source collection, audience and extraction policy in the supplied stream name. The Rust equivalent for a retained live corpus is `anno::coalesce::memberships::publish_memberships`.

- `direct` gives the exact edges accepted during native resolution, their similarities and the measure used. Qualifying values are inclusive at the threshold.
- A shared `component` digest identifies the same complete source-addressed member set. The digest changes on a merge or split. Neighbor queries return null direct evidence for an inferred peer, or for preserved catalog-anchored membership whose edge was not reevaluated.
- `catalog_record` gives the qualified external assignment carried by that track. Independently cataloged postings with the same name can have no shared model component. Source signal IDs and locations remain available for citation.

The CLI rebuilds current model memberships from the submitted documents. A live corpus can also retain a reviewed hybrid identity spanning documents. Use its Rust publication operation when that corpus-owned state matters. Catalog-only rows remain available with null model component. Independent callers must define their issuer and source-document IDs upstream.

Apply removals then upserts from the expected predecessor and advance the checkpoint atomically. Redeliveries whose target already matches need no writes. Consumers caching a neighbor's display information must invalidate cached queries for affected **old and new** components, including when member metadata changes. The SQLite replay consumes intact locally generated CLI output, indexes postings by component, and compares every cached response to fresh bulk and native queries. It includes redundant paths, a disjoint insertion, source-edition replacement, ties, reversed input, an empty final corpus, and incompatible predecessors.

```sh
cargo test --locked -p anno --no-default-features --test membership_publication
cargo test --locked -p anno-cli --no-default-features --features 'eval extractor-readability' --test membership_queries
cargo build --locked -p anno-cli --no-default-features --features 'eval extractor-readability' --bin anno
python3 verification/memberships/replay.py target/debug/anno --cost
python3 verification/memberships/compare.py target/debug/anno --roster /path/to/roster.jsonl
```

## Resolver comparison

The production resolver takes the transitive closure of every threshold-passing edge. The comparison method exactly minimizes the number of pair decisions changed from that same edge set to form a partition. It enumerates partitions inside each connected component and breaks equal-cost ties by the first assignment in source order. Both methods received every possible pair of the same supplied person tracks at threshold 0.5. Neither method deferred a pair.

| Panel | Scored pairs | Gold links | Native false joins / misses | Cluster editing false joins / misses |
|---|---:|---:|---:|---:|
| Frozen synthetic registers | 67 | 11 | 17 / 3 | 14 / 4 |
| Subsequent synthetic registers | 35 | 15 | 16 / 0 | 4 / 0 |
| Published 2023 roster against 2021 | 208 | 13 | 3 / 0 | 3 / 0 |
| Published 2025 roster against both earlier editions | 522 | 25 | 8 / 0 | 8 / 0 |

The real temporal holdout uses the existing pinned CC0 Congress roster. Every record whose printed name starts with the token Mark or Robert is included at each July 1 cut. This is a targeted slice of a panel that previously identified namesakes and shared tokens. Names and source positions enter resolution. Bioguide IDs are withheld for scoring. For each later cut, the table counts all pairs from that cut to earlier cuts. There was one synthetic development topic with multiple optimum partitions and three in the subsequent synthetic panel. Neither real cut had a tied optimum. [Roster extraction, custody and reproduction](../index_consumers/congress/README.md)

For an illustrative discovery loss of one unit per false join and two per miss, native and cluster editing cost 23 and 22 on the first panel, 16 and 4 on the subsequent synthetic panel and 8 each on the 2025 holdout. At eight units per false join and one per miss, those losses were 139/116, 128/32 and 64/64. The absence of improvement on the dated holdout and the additional development miss give no basis for a default change. Discovery can present the current component and its supporting edges. A consumer publishing a committed attribution can read the qualified catalog column or require an explicit review.

## Discovery and application cost

The publisher recomputes the native corpus and accepted edges, hashes each component's sorted source keys and captures all rows. A retained older IndexSnapshot avoids re-resolving the previous endpoint in Rust. The stateless CLI `--since` rebuilds both endpoints before comparing fingerprints and rows. Its application writes only the changed postings. Exhaustive scoring is quadratic in the number of unanchored tracks. A dense component can itself contain quadratically many direct neighbor entries.

In the benchmark, four tracks share each base name. Adding one two-token bridge connects the first eight tracks. Both the 32- and 128-document base require eight removals and nine upserts. In the larger current edition the resolver examined 8,256 candidate pairs. Run `--cost` to print separately timed full capture, discovery of the delta from two complete inputs, and SQLite application. The timing includes CLI startup, JSON and local pipe I/O in discovery, and an in-memory SQLite transaction in application. It does not model disk durability or asynchronous delivery.


On the shared x86-64 development container with Rust 1.95, dev profile with symbols disabled, five-run medians were as follows. Each `--since` process receives two independent complete inputs.

| Base documents | Full current publication | Two publications and diff | SQLite apply |
|---:|---:|---:|---:|
| 32 | 20.54 ms | 23.24 ms | 0.39 ms |
| 128 | 164.32 ms | 295.98 ms | 0.53 ms |
