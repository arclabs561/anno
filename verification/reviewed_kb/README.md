# Verifying reviewed catalog assignments

A successful `Corpus::apply_kb_assignment` must change only the track bindings in the approved scope. A stale or foreign preview must fail before writing. The required **Test (minimal)** job verifies both properties on pull requests and pushes to `main`.

Run `just verify-reviewed-kb` from the repository root with Rust, Python 3, Git and the Z3 command-line tool on your path. You can also invoke the CI script directly.

```sh
bash scripts/verification/check_reviewed_kb.sh
```

| Check | What a failure tells you |
|---|---|
| [Reader correction](../../scripts/verification/check_review_mention.py) | A saved apply or deferral disagrees with the selected source mention, the other namesake, or the catalog binding read in a fresh process. [Try the example](../../docs/QUICKSTART.md#review-a-catalog-link-from-a-source-mention) |
| [Assignment scope](../../crates/anno/tests/reviewed_kb_assignment.rs) | The Rust API reported the wrong members, changed a binding outside the receipt, or accepted a stale or ambiguous assignment. The fixture includes distinct identities with identical catalog keys and labels. |
| [Action sequences](../../crates/anno/tests/reviewed_kb_sequences.rs) | An actual edit, review or apply disagreed with the independent oracle. The test tries all 4,681 sequences of up to four actions from an eight-action alphabet on a two-document corpus. The failing prefix identifies the operations to reproduce. |
| [Shared fuzz seeds](../../crates/anno/tests/reviewed_kb_seeds.rs) | A retained operation trace changed the wrong binding or produced an unexpected review outcome on a corpus containing both local and shared identities. [Run a local campaign or promote a minimized input](../../fuzz/README.md) |
| [Document delivery model](../../crates/anno/tests/document_delivery_seeds.rs) | The observed receipt, rejection, source mention binding, track handle or review scope disagreed with the issuer and ownership history. Stable Cargo and the coverage-guided job run the same model. |
| [Compiler boundary](../../scripts/verification/check_review_boundary.py) | Rust rejected a valid caller or allowed a forbidden operation, such as a sibling writing through shared access or a caller submitting an edited display copy. The fixture compiles the production state module and checks the expected diagnostics for each case. |
| [Z3 model](revision_guard.smt2) | A counterexample exists within the model bounds, an expected witness disappeared, or a solver query did not run. The named queries cover rejection, the scope of a fresh write, and deliberate defects in ownership, revision or write scope. A base case and inductive step prove that revision equality tracks a clean history for any finite sequence of modeled operations. |
| [Invalidation mutation](../../scripts/verification/check_review_invalidation.py) | The unchanged code failed its positive control, or a test failed to detect removing invalidation from the production mutation gateway in a temporary source copy. |

When adding a corpus writer, obtain mutable storage through the private state module. Add a concrete action to the Rust sequence test when the writer introduces a new way to invalidate or preserve a review. Change the SMT transitions when the observable contract changes. The model assumes the implementation invalidates its revision correctly and finds all members. The concrete Rust checks exercise those assumptions.

The project-authored models, verification code and synthetic seed fixtures use the repository's `MIT OR Apache-2.0` license. Source rights for other corpora and the seed admission properties are documented separately. [Source-rights verification](../source_rights/README.md)
