# What a cross-document identity tells a reader

A signal identifies an observation in one source. A track groups signals within that document. A native cross-document identity groups document-qualified tracks under a particular resolution. A catalog record has its own namespace and accession. Two tracks can refer to the same person before anyone links either to a catalog. Two people can also share a printed name and receive one mistaken identity.

## Choosing an entry point

| Entry point | Input and decision | What the reader receives |
|---|---|---|
| `anno pipeline --cross-doc` | Extracts detections from each input and resolves their tracks in one corpus. `--coref` runs within-document coreference first. `--link-kb` attaches offline demo identifiers, including generated ones. | `--format grounded` returns the resolved source documents in input order, including the local coreference receipt if requested. `--format json` returns cluster rows from those same assignments. |
| `anno cross-doc --import ...` or `--stdin` | Runs native track resolution on supplied grounded documents. A second edition of the same document ID replaces the earlier edition. | The JSON report groups shared assignments and equal **qualified** catalog references. Source membership is identified by document and signal position. Unassigned signals remain available in the source document. |
| `anno cross-doc DIRECTORY` | Runs the flat mention matcher on extracted entities. Lexical blocking limits which pairs reach the scorer. | The JSON report includes metadata and cluster rows. `anno query REPORT --format json` reads those rows. Source offsets and document text require a grounded document. |
| `anno::coalesce::Resolver` | Compares eligible tracks using compatible embeddings or token overlap, applies a threshold and takes connected components. Already linked tracks stay assigned. | Each measured group can report its weakest pair. The numeric identity and its representative name belong to this corpus. |
| `anno query publication` | Diffs a source-anchored projection of occurrences and catalog assignments against a previous edition. | A reader gets removals, upserts and checkpoints for that projection. Changing an unlinked cluster alone yields no posting changes. |

A cluster report’s type filter uses the type on the projected identity. A catalog record can have no type even when its supporting signals are labeled. Inspect grounded signals for a mention-level type query.

The native Rust default uses threshold 0.7 with type matching. Both CLI commands default to 0.6 with type matching off and expose `--require-type-match`. With native type matching, an explicit track label takes precedence, otherwise the resolver checks its member labels for agreement. A within-document model can emit a generic coreference mention label. That label carries less information than a person or organization annotation. [Native resolver](../../crates/anno/src/coalesce/resolver.rs) · [Pipeline](../../crates/anno-cli/src/cli/commands/pipeline.rs) · [Raw and imported reports](../../crates/anno-cli/src/cli/commands/crossdoc.rs)

Raw directory mode compares every pair through 100 extracted mentions (4,950 pairs) and selects lexical locality-sensitive hashing (LSH) from 101 mentions. Files yielding no mentions add no pairs. A single dense document can cross the cutoff because the flat resolver also compares mentions within each document. Pin `--candidates exhaustive` or `--candidates lsh` when comparing editions that cross the cutoff. JSON metadata records the actual policy and mention count. The regression checks that adding a file with no detections preserves the “Dr. Nara Pell” / “Dr. Pell” link, and that explicit LSH still misses this pair. [CLI regression](../../crates/anno-cli/tests/crossdoc_candidates.rs)

## Inspecting one flat mention pair

Use the document ID and zero-based entity index from the JSON cluster rows to ask how the raw directory resolver treated two mentions.

```sh
anno cross-doc ./reports --format json --explain-pair first 0 second 0
```

The optional `pair_explanation` includes each extracted surface and type. Its `decision.outcome` records `not_proposed` when candidate generation omitted the pair, `type_mismatch` when the type guard rejected it, or `below_threshold` or `accepted` with the measured `similarity` and its `basis`. If an encoder produces a non-finite similarity, the receipt records `non_finite_similarity`, its `basis` and the original comparison’s `passed_threshold` result. The report metadata supplies the candidate mode and threshold. An accepted edge includes equality at the threshold. `same_component` records the final membership before display filters. An indirect connection through a third mention can produce a shared component even when the pair itself was rejected. The Rust entry point is `CDCRResolver::resolve_with_pair`. [Flat resolver](../../crates/anno-eval/src/eval/cdcr.rs)

For two **grounded tracks**, use `anno query memberships EDITION --explain-pair LEFT_DOC LEFT_TRACK RIGHT_DOC RIGHT_TRACK` on the retained edition. Its direct decision distinguishes catalog exclusion, type mismatch, missing evidence and the score comparison. The returned component flag includes transitive membership. Both CLI forms use source document IDs, but raw entity indices and grounded track IDs address different objects. [Grounded pair query](../../verification/memberships/README.md)

## Looking for a predecessor in a prior edition

A reviewer can ask which tracks in an earlier roster have names resembling a particular current track. Supply two complete membership-edition JSON files from the same publication stream. Use the document ID and submitted local track ID from the current edition. The prior sequence must be strictly earlier. The sequence describes publication order; a source can publish a correction long after the events it describes.

```sh
anno query predecessors current.json --prior previous.json --doc-id roster --track-id 7
anno query predecessors current.json --prior previous.json --doc-id roster --track-id 7 --limit 5 --minimum-score 0
```

The default shows one candidate with a word-set Jaccard score of at least 0.5. Widening as in the second command shows up to five tracks with positive lexical overlap. The lookup compares nonblank canonical surfaces using anno's native word normalization (Unicode lowercase, whitespace-separated words, ASCII possessive suffixes). Catalog assignments, embeddings and type labels do not contribute to this name score. The lookup uses its own display threshold independently of the two editions' membership publication settings.

Equal-score groups stay together. If a top tie exceeds the available slots, the reviewer gets an empty list and the size and score of that tie. If a lower tie exceeds the remaining slots, higher groups are retained. If the list is already full and further candidates qualify, `truncated` is true. The maximum limit is 128, so a top tie larger than 128 stays withheld.

The response records the stream, both publication sequences, lookup settings, scorer, and the surface and source address of each displayed track. A text digest identifies the document text. An annotation can change a track's canonical surface without changing that digest. Retain the response to explain a displayed score, and retain the supplied editions to reproduce the ranking after a correction. Each lookup scans the prior tracks and sorts the qualifying matches; the display limit bounds the response size in candidates.

This query reads the supplied editions and makes no assignments. To confirm a catalog binding, use the independently established catalog name and accession, obtain the current **live** document and track IDs from the receiving corpus, then use its journal preview and approval flow. Import can remap a submitted handle. A publication sequence and a journal head identify different states. [Reviewed corpus operations](index-consumers.md#keeping-a-reviewed-corpus-across-restarts)

## Which changes reach consumers?

* **A source is corrected or withdrawn.** Within the same live corpus, a surviving resolver-owned identity can keep its number while its name, type, representative embedding and pair support change. The resolver preserves externally supplied representative metadata. Refresh a displayed identity when its evidence changes. [Lifecycle regressions](../../crates/anno/src/coalesce/resolver.rs)
* **A new source supplies a bridge.** Thresholded pair matches are transitive in both engines. In a flat diagnostic, “Nara Pell” and “Ari Pell” were separate until a third document supplied “Pell”. The resulting group reported the flat constructor's confidence of 1.0 and no weakest-pair measurement. On the native path, the weakest pair can score below the threshold that connected the group. Neither score estimates the probability of a shared real-world identity. [Flat matcher](../../crates/anno-eval/src/eval/cdcr.rs) · [Score semantics](track-association.md)
* **A new edition changes only another document.** In the pipeline regression, two documents shared an email address. Editing the other document split the group while leaving the first document's bytes and catalog postings identical. A consumer that promises cluster navigation must version or diff the membership it publishes. A catalog posting feed alone cannot invalidate that navigation. [Process regression](../../crates/anno-cli/tests/pipeline_cross_doc.rs) · [Publication contract](catalog-publication-pilot.md)
* **The same sources are rebuilt.** Compare document-qualified memberships under the same input edition and resolver settings. Importing document snapshots begins a fresh corpus. Report ordinals and corpus identity numbers can change with ordering or allocation history. Even a stable group number cannot supply a permanent identifier for a person. [Dated bulk and incremental replay](../../verification/index_consumers/congress/README.md#score-native-cross-document-coreference)

## Time, namesakes and abstention

The native resolver received 1,798 unlinked person tracks from twelve dated House and Senate directory and committee sections. Bioguide accessions were withheld until scoring. Among 1,310,369 eligible cross-document pairs, 1,858 shared an accession. At 0.7 there were **1,810 correct links, 12 false links and 48 missed links**. All twelve false links involved two namesakes printed as “Robert Menendez”. Lowering the threshold to 0.5 recovered 41 true links and introduced 89 false links. These figures assess linkage conditional on supplied tracks. [Inputs, scorer and reproduction](../../verification/index_consumers/congress/README.md#score-native-cross-document-coreference)

Permanent chamber boundaries removed the false links and lost 22 additional true ones across ten people who changed chambers. A retrospective rule that split groups only after observing simultaneous membership in both chambers removed all twelve false links in this panel while preserving the transfers. It depends on synchronized roster editions. A counterexample combining a transfer with an overlapping namesake defeats it. A downstream archive could flag the conflicting interval for review and keep the source citations accessible while the link is unsettled. There is no general impossibility rule for changing offices, names or identities. [Policy experiment and counterexample](../../verification/index_consumers/congress/test_cross_doc_partition.py)

For a search interface, a provisional name cluster can be useful for discovery. Open the underlying occurrences and label the grouping with its source edition. For an index joining facts about people, publish a qualified catalog record or a revisioned decision with its evidence, issuer and scope. An append-only alias list cannot say when a name or role applied, when the catalog learned it, or who can see it. Keep effective dates, publication order and visibility in the consuming domain until the producer can supply them. A future matching API should expose candidate inclusion, supporting and conflicting evidence, and an unresolved outcome before offering a confidence threshold for automatic joins.

## Delivering grounded documents from independent issuers

Use `anno cross-doc --delivery-manifest deliveries.json` for a caller-managed import. Each array member names one grounded document relative to the manifest and carries its own request. For example, a manifest containing one independent document can be written as

```json
[
  {
    "path": "catalog/page-7.json",
    "request": {
      "issuer": "city-library",
      "delivery_id": "shipment-45/page-7",
      "source_revision": 3,
      "ownership": {"mode": "independent"}
    }
  }
]
```

The default JSON report includes an ordered `deliveries` array with the original receipt for each applied or previously applied operation. Within one invocation, repeating the entire manifest or any previously accepted member preserves the existing corpus. Each separate invocation starts a new session. Durable consumers must recover their journal of accepted deliveries and catalog decisions before handling late retries.

In Rust, pass the request and the original JSON bytes to `Corpus::deliver_document`. Get the qualified live document ID from the receipt. Each new delivery ID must advance the issuer-local document revision. Reusing an accepted token checks the original bytes and intent even after a newer revision arrives. Byte differences, including changes in JSON formatting, count as different payloads.

Independent import allocates owners per document and preserves a shared assignment between tracks within that document. An authorized restore uses `{"mode":"restore","origin":"catalog-backup-8"}`. Each issuer, origin and source identity ID identifies an original record. Its members follow the current canonical catalog assignment in the receiving corpus. A fork that copied an entire origin must declare independent import. The caller authenticates the issuer and decides whether it can assert that origin.

A newer delivery that disputes reviewed bindings returns a review with the current and proposed scopes, the source assertion, and its effective catalog assignment on approval. Keeping the current edition discards the proposal. Accepting it calls `Corpus::accept_document_replacement` with the still-current review. The replacement retires superseded track handles. Raw `--import` continues to use `Corpus::add_document` with its established explicit replacement behavior.
## Publishing changed memberships

`anno query memberships` publishes source-addressed track rows across complete corpus editions. Direct scores, transitive component membership and qualified catalog assignment are available independently. A corrected bridge changes the component digest on all affected rows, including tracks in documents with unchanged source text. A downstream subscriber can retract old memberships, install new ones and advance its checkpoint in one transaction. The reference publisher uses full discovery and an incremental downstream write. [API, reader replay, cluster-editing comparison and cost measurement](../../verification/memberships/README.md)
