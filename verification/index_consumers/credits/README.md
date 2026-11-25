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

## Provenance and reuse

The fictional records, expected answers, scripts and original research annotations are available under the repository's `MIT OR Apache-2.0` license. The source register links public decisions, studies, policies and catalog records. Its entries describe the evidence and its limits. The fixtures borrow decision patterns and use invented works, names, catalogs and source events. No source passages, scans or third-party record dumps are included. Each linked work retains its own terms. These shared examples are regression seeds. A model evaluated on them after seeing them would need a separate held-out corpus for a performance claim. [Source register](sources.json) · [Repository license](../../../LICENSE) · [Source rights and regression seeds](../../source_rights/README.md)
