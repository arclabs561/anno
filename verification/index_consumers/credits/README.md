# Who gets credit for a work?

A reader can ask what a source printed, which catalog record a mention links to, or which publisher attributes a work to that record. Each question can have a different answer. This corpus exercises a reader that must update those answers after corrections and changes in visibility. [Research and design findings](../../../docs/decisions/credit-corrections.md)

## Reproduce

From the repository root, run the same replay used by the index verification suite. The default Cargo build may fetch dependencies. The replay runs offline with a built CLI. To supply one, add `--anno /path/to/anno`.

```sh
python3 verification/index_consumers/credits/check_cases.py
python3 verification/index_consumers/credits/check_search.py
cargo test --locked -p anno --no-default-features --test membership_publication
```

The three fictional cases contain 13 source editions and 17 deliveries. Each step specifies the expected public and staff answers. The runner obtains complete memberships and changes from `anno query memberships`, applies the changes to a reader with cached answers, and compares its results with a fresh read of the complete snapshots and the authored expectations. A deliberate stale-cache fault checks whether the fixtures detect a broken reader. Each venue publishes into its own stream. The fixtures give each credited occurrence its own document and each source a contiguous delivery sequence. [Fixtures](cases.json) · [Runner](check_cases.py)

| Case | Question the reader must answer |
| --- | --- |
| A program, an interview and a register | When the interview withdraws its attribution, can we still quote the printed collective and find the independent register credit? Does an uncredited reappearance of the person's name restore the withdrawn support? |
| Two exhibition venues | Can two venues disagree about one work when both use the same name and accession string? Does an unrelated work or a staff-only label enter the public answer? |
| A byline and a ledger | If access to the supporting ledger is granted and revoked while its membership rows stay identical, does an already-open public search lose the credit? Can an old delivery restore it? |

The consumer supplies the work, relationship and audience for each source row. The native CLI supplies grounded signals, track membership, qualified catalog bindings and checkpoints. The Rust regression pairs a fixed membership checkpoint with a separate checkpoint covering the publisher's credit decision. The Python reader uses trusted local fixtures and an in-memory cache. The queries cover exact catalog lookups and printed credits. [Rust regression](../../../crates/anno/tests/membership_publication.rs) · [Existing replay suite](../../../scripts/verification/check_index_replay.sh)

A separate SQLite reader installs the ledger editions and reopens the database before each observation. It checks exact search, the count of credited works, autocomplete and the printed byline. Its negative control retains an obsolete autocomplete entry after revocation. Exact search and counts still pass in that faulty reader. Both readers run against a private local database with two simulated audiences. [SQLite search check](check_search.py)

## Correct a bibliography at a retained edition

Two authored articles credit “Rowan Mire”. Their catalog records initially assign both articles to the same person. A correction moves one credit to a namesake with the same accession in a different namespace. The article and its printed affiliation are unchanged. A third index edition receives another affiliation that already applied on the original publication day, a later name, and an affiliation with unknown dates. [Articles and catalog receipts](bibliography/)

| Reader edition | Person credited on the tide paper | Affiliations supported for its publication day |
|---|---|---|
| 1, initial import | archive / 17 | Quay Archive |
| 2, corrected attribution | university / 17 | Harbor Institute |
| 3, additional authority evidence | university / 17 | Harbor Institute and Reed Observatory. The date of the Marsh Library affiliation is undetermined. |

The [replay](check_attribution_history.py) reads the authored JATS through the bibliography byline reader and verifies each projected byline using `anno query publication`. It also asks `anno query memberships` to publish the qualified assignments. The online reader starts from one complete native snapshot and receives subsequent native row deltas alongside the application's complete credit and dated-claim manifest. At each delivery the test reopens every retained view in memory and compares its answers with a fresh full native publication and hand-labeled expectations. It checks both authors’ work lists, source locators, half-open interval boundaries and duplicate delivery. The third native delta has no changed rows. The additional authority evidence still changes the historical-day answer in that reader edition.

The article’s affiliation comes from its contributor reference. Dated institutional claims have independent evidence. The reference reader keeps an undated affiliation visible as undetermined for the requested day. Its edition counter orders what the application learned. The fixture treats a stated period as nonempty and uses whole days, with an exclusive end. A known start confirms that day. A known end confirms the previous day. It assumes trusted native CLI output; the reader checks the scope, transition, and the person named by each authority decision. Each claim is public in this fixture. Each retained view occupies memory. [JATS affiliations](https://jats.nlm.nih.gov/publishing/tag-library/1.4/chapter/tag-affiliations.html) · [ORCID affiliation dates and overlapping roles](https://info.orcid.org/documentation/integration-guide/admin-guide-to-affiliations/) · [Example of a publisher changing an author list](https://elifesciences.org/articles/90408)

The date check generates 15 possible reports with known or missing bounds. It asks about six days spanning a leap day, through both name and affiliation lookup. Its answer key enumerates the compatible nonempty intervals. A citation is confirmed if every interval covers the day, undetermined if some do, and absent if none do. Several sources use the same display value, so the check also exercises retaining each citation and keeping the other namesake's answers separate. The replay includes a faulty reader that discards uncertain names and checks that it fails on those answers.

A missing end here means that the source did not supply one. The [temporal alias example](../../temporal_identity/check_replay.py) models a different source convention in which a dated alias can continue indefinitely.

Run `python3 verification/index_consumers/credits/check_attribution_history.py --anno target/debug/anno`. All article text, example dates, authority decisions and expected answers for this case were written for anno under MIT OR Apache-2.0.

## Provenance and reuse

The fictional records, expected answers, scripts and original research annotations are available under the repository's `MIT OR Apache-2.0` license. The source register links public decisions, studies, policies and catalog records. Its entries describe the evidence and its limits. The fixtures borrow decision patterns and use invented works, names, catalogs and source events. No source passages, scans or third-party record dumps are included. Each linked work retains its own terms. These shared examples are regression seeds. A model evaluated on them after seeing them would need a separate held-out corpus for a performance claim. [Source register](sources.json) · [Repository license](../../../LICENSE) · [Source rights and regression seeds](../../source_rights/README.md)
