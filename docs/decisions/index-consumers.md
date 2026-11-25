# What a downstream reader needs from an identity index

A source search, a person search and a count of people answer different questions. The same mention can belong in the first result while its membership in the other two is unresolved. The journeys below are proposed product uses. The linked Rust paths and replay results identify which pieces were actually exercised. A documented external integration is Aurelius consuming source-bound words from a pinned copy of anno. Its source establishes an integration path; its deployed revision and production latency target are unknown. [Caller and deployment evidence](grounded-extraction.md#resource-assumptions)

## Search and question answering with citations

A researcher asks what a particular person recorded. A name-expanded version of a passage can retrieve a pronoun that a literal name search misses. Opening the citation must still show the author's actual words. Two catalogs can also use the same accession for different people.

In the [native RAG example](../../crates/anno/examples/rag_citation.rs), literal queries found the two observation sentences after pronoun expansion and zero before it. Resolving on bare accession `17` returned two observations for `music/17`, only one of which was attributable to that person. Following the subject mention to its current **catalog and accession** returned the one supported sentence. Its subject in the source is “She”. Present expanded prose as a search aid and quote the original occurrence in the answer. These counts cover two hand-authored passages with supplied tracks and a literal matcher.

A separate three-sentence control found a production attribution error. With Iris Moss untracked and a supplied track joining Éva Lorn to “She”, a newly allocated cluster ID collided with the supplied ID. The public RAG path rewrote the pronoun to Iris before the fix and to Éva after it. The regression covers the supplied-antecedent case; a pronoun without a supplied named antecedent can still use heuristic fallback. [Fix and before/after regression](https://github.com/arclabs561/anno/pull/298)

Store source revision, coordinate unit, range and exact quote with a citation. Also retain the revision of the annotation or link used for person-specific retrieval. An edit elsewhere can leave the same quote at the same offsets while changing its context. Conversely, correcting an attribution can leave the original bytes untouched. An old citation can still verify the words while no longer supporting the answer about that person. Measure passage recall, exact-quote validity and person-attribution precision separately. Test the latter two after each kind of revision. [W3C annotation selectors and states](https://www.w3.org/TR/annotation-model/#selectors) · [KILT provenance evaluation](https://aclanthology.org/2021.naacl-main.200/) · [ALCE citation evaluation](https://aclanthology.org/2023.emnlp-main.398/)

## Archives, changing names and statements about a role

An editor preparing an exhibit wants the byline as printed, aliases used in 1997, and what the catalog supported when the exhibit was assembled in 2014. Those can be three different strings. Show the original byline at its source anchor. Show names and roles as separate sourced claims, each with its effective period. Let two claims overlap, and keep a withdrawal or correction distinguishable from a genuine later change. The [two-time replay](../../verification/temporal_identity/check_replay.py) exercises those distinctions.

For transcripts, a channel identifying the speaker, a person mentioned by the speaker, and the person speaking inside a quotation can all differ. Treating each first-person pronoun as a reference to the channel owner could put a resignation or confession in the wrong person's mouth. Anno's existing speaker-turn rules have explicit limits on quotation turns and document-local channel identifiers. A release test for an exhibit would keep the printed byline and quoted words intact, tie any speaker attribution to its turn and source, and allow “unresolved” when a quotation or reused role defeats the available evidence. [Speaker-turn contract](../CONTRACT.md) · [International Council on Archives, Records in Contexts](https://www.ica.org/ica-network/expert-groups/egad/records-in-contexts-conceptual-model/)

## Search facets, dashboards and index migrations

A catalog maintainer needs to count *people mentioned*, then click a count to see its contributions. Three mentions, two local tracks and one qualified catalog key describe different denominators. One catalog key can cover separate internal identity records. Report what was counted and how many occurrences remain unresolved. A count without its contributing source occurrences is hard to correct or explain.

During a rebuild, use an occurrence key scoped to the source document, source revision and span, with a discriminator for multiple assertions on the same span. Keep the annotation version alongside it. Store identity bindings under the resolver generation that produced them. Removing a bridge can change bindings on documents whose text never changed. Publish removals and additions together with a checkpoint or atomically switch to the rebuilt snapshot. A restarted consumer should replay the same revision twice without double-counting and should handle an empty snapshot or explicit deletion without resurrecting old contributions. A memory-limited incremental resolver also needs an explicit session boundary: two independent calls can reuse the same cluster number for different people. [Native replay](../../crates/anno/tests/index_reconciliation.rs) · [Existing graph export](../EXTRACTION_WORKFLOWS.md) · [Debezium delete and tombstone semantics](https://debezium.io/documentation/reference/stable/transformations/applying-transformations-selectively.html)

We fed four native resolver snapshots to a single-host SQLite dashboard. Retiring a bridge split two unchanged documents between different identities. Updating only the retired document left the dashboard reporting one person. We killed a separate writer after removing old postings and again after inserting their replacements. With statement-by-statement commits, the dashboard reopened at the old checkpoint with either zero postings or the new postings. With one transaction for removals, additions and the checkpoint, it reopened with the complete old answer at both interruption points. Retrying delivered the split, and delivering it again wrote zero rows. A missed revision required a snapshot to recover. Publishing the final empty snapshot removed every posting. These results cover this local writer and storage engine. [Native publisher](../../crates/anno/examples/index_publish_snapshots.rs) · [SQLite recovery experiment](../../verification/index_consumers/check_recovery.py)

## A public reader and a staff-only register

A public page contains an ambiguous byline. A private register disambiguates it. A public reader may be allowed to search for the printed name while a person-specific search, a suggested alias or a count identifying the writer discloses the private assertion. Filtering documents only after computing results or facets would expose it. Calculate each displayed inference from an independently sufficient set of evidence the viewer may use. A later public corroboration could change the public answer.

The [projection experiment](../../verification/index_consumers/check_projection.py) checks literal hits, attributed citations, visible facets and totals in 26 expected views. Checking only the source page's permissions leaked attribution in four public views. Ignoring dependency revisions restored stale support in three staff views. The revision- and dependency-aware version matched all 26. This is a small reference projection with fixed audiences; a deployment still needs authorization, cache isolation and invalidation when permissions change. Include shared search caches, autocomplete and counts in release tests, and exercise restoration from an older checkpoint after a deletion. A replay tombstone prevents an old link from reappearing in a rebuilt index. Erasing the original text, exports, retained evidence and backups is a separate storage and retention decision.

A related mistake is to treat a release pseudonym as a person ID. Two separate native CLI runs on different synthetic email addresses can both produce `contact@example.com`. Joining on that output invents a shared identity. If a release promises unlinkability, validate the release itself, including retained metadata and alias tables. If the caller instead needs within-release consistency, state the population and lifetime in which that promise holds. [Privacy command](../../crates/anno-cli/src/cli/commands/privacy.rs)

## An operator with a finite decision budget

A cataloger has time for twenty ambiguous cases before a release. Give them the source occurrence, the proposed target, the plausible alternatives, the reach of the edit, and the option to defer it. “No candidate retrieved”, “the person is absent from this catalog”, and “nobody has decided yet” require different next actions. The first can justify searching another source, the second may justify creating a record, and the third can remain an unresolved case. Record what was presented, the catalog and extraction revisions, and the decision actually made. If two people disagree, preserve the individual judgments and record which decision governs this release.

Compare queues at equal operator time or equal case volume. Measure omitted gold candidates, wrong automatic assignments, unresolved cases and reversals after later evidence. Assuming somebody can review every deferred case for a constant cost needs validation with a real queue. Anno's existing suggestion study illustrates why equal budgets can change the apparent benefit of a policy. [Review-budget study](../EXTRACTION_WORKFLOWS.md) · [Decision costs in the index experiment](index-reconciliation.md)

## A dataset maintainer evaluating a new model

A team corrects the gold labels on a document and wants to compare two models under the revised policy. The source digest stays the same. In an original CLI probe the pattern model extracted `bench@example.org` from “Return sensor logs to bench@example.org.” The first gold snapshot labeled character range `[22, 39)` as `EMAIL`. The second used `entities: []` after the caller elected to exclude shared mailboxes. The model's prediction stayed fixed. Scoring against the first and second snapshots independently yielded F1 scores of 1.0 and 0.0. Passing both rows together yielded 1.0. Repeating only the first row yielded 0.667, with an extra false negative. The single-text gold loader combines entity arrays from every row, including valid fragments of one annotation. It verifies a supplied source digest and document ID but has no operation to supersede an earlier judgment. [Gold loader](../../crates/anno-cli/src/cli/utils.rs) · [Evaluation caller](../../crates/anno-cli/src/cli/commands/eval.rs)

A corpus adapter should identify the dataset policy and adjudicated revision for each document, select that snapshot, and then pass its fragments to the scorer. Keep an empty accepted snapshot distinguishable from a document that nobody annotated. Two adjudicators can label the same span differently. Preserve both judgments until the evaluation declares which policy or adjudication governs the run. An acceptance test can replay either delivery twice, select the revised policy, and still count exactly one false positive. Selecting the historical policy should still yield the original score. Deduplicating arbitrary annotations by span would erase legitimate overlapping labels.

### Keeping a reviewed corpus across restarts

`anno corpus` stores accepted grounded document deliveries and reviewed catalog assignments in one SQLite journal. For example, use this request file to deliver the exact bytes of `program.json`. Give each member of a batch its own delivery ID so a subset can be retried independently.

```json
{"kind":"delivery","request":{"issuer":"programs","delivery_id":"issue-4/doc-7","source_revision":4,"ownership":{"mode":"independent"}}}
```

```sh
anno corpus --store program.sqlite apply request.json --document program.json
```

Save the returned `receipt.document_id` to read the current grounded document with `anno corpus --store program.sqlite read <document-id>`. An exact replay returns the original receipt, even after a review has changed the current state. Use a later source revision and a fresh delivery ID for a genuine update. The caller authenticates the issuer and decides who may restore the same original corpus or access the database.

An assignment file identifies the live track and proposed catalog binding. Copy the track ID from the current document. In this example the document's embedded ID was `doc-7`.

```json
{"kind":"assignment","trigger":{"doc_id":"8:programs/doc-7","track_id":0},"kb_name":"artists","kb_id":"42","canonical_name":"Nera"}
```

```sh
anno corpus --store program.sqlite preview correction.json
anno corpus --store program.sqlite apply correction.json --approval "$REVIEWED_APPROVAL"
```

Inspect the affected tracks before supplying the approval token. Follow the same preview/apply sequence with `--document` for an incoming revision that conflicts with a reviewed binding. The token binds the operation, payload, displayed scope, and exact journal head. The assignment and delivery commands serialize writers. Each accepted journal row commits the operation, the original document bytes, the scope and the receipt together. A reader reconstructs its corpus from one SQLite snapshot. Replaying an already committed approval returns its original receipt.

For a review spanning several documents, take the distinct document IDs from the assignment preview's `proposal.scope.affected_tracks`. Pass its `head` as the read precondition. For example, if the scope includes two programs, inspect both before applying the reviewed approval.

```sh
anno corpus --store program.sqlite read-many --at-head "$PREVIEW_HEAD" 8:programs/doc-7 8:programs/doc-9
```

The response contains `head` and `documents` in request order, including duplicates. The command accepts 1–128 IDs and verifies the full journal once per call. An unknown document or mismatched head fails before any documents are printed. Omitting `--at-head` reads the current head. For a larger selection, use the same preview head for every request. If the journal advances before inspection finishes, obtain a fresh preview and inspect its scope again. The response covers only the requested documents.

This adapter targets local corpora where replaying the retained history per command is affordable. All accepted deliveries stay recognizable for the life of the journal. Back up the whole SQLite database with SQLite's backup facilities. Reconstructing an empty database from current document exports loses the delivery and review history. If a later version computes a different review scope or receipt during recovery, the command reports the incompatible event. Keep the prior executable and database available for a reviewed migration.
