# Source rights and regression seeds

The source-rights model checks Anno's routing decisions after a maintainer has reviewed the source and destination. It takes authorization, source class, retained notices, and regression/holdout scope as inputs. It cannot determine copyright ownership, grant permission, or inspect a deployed IAM policy.

Run the same offline check as the required CI job with Python 3 and Z3 on your path.

```sh
python3 verification/source_rights/check.py
```

The negative queries cover replay of a restricted local source, a public publisher download, a missing license declaration, an unreviewed suggestion, and a held-out case. A separate equivalence query checks that replay preserves the original admission decision in both directions, including registration and notice requirements. Satisfiable witnesses cover authorized local use and the reviewed synthetic path. A deliberately permissive rule demonstrates how using replay success alone would allow a restricted source into shared state. The runner also compares the solver with the actual reviewed-span registry predicate on all 36 combinations of source, license declaration, registration and holdout scope. The replay adapter separately checks the frozen oracle, provenance and native output. The checkpoint tests exercise seed identities, corpus version and restore behavior.

For dataset decisions, run `cargo test --locked -p anno-eval --lib source_rights_route_conformance`. This checks the production registry's local-only and shared-read exclusions for the reviewed datasets and absence of an approved raw S3 publication package across the registry. The concrete loader tests cover the transfer entry points. Default CI uses project-controlled fixtures and publisher material only where the corresponding terms and source notices have been reviewed.

## License and provenance

Anno's project-authored SMT models, verification scripts, synthetic test fixtures, and seven reviewed-KB seed byte sequences are available under `MIT OR Apache-2.0`. The reviewed-KB seeds encode operation sequences. Their bytes contain no third-party corpus passages. The reviewed NER registry records the rights and scope of its project-authored synthetic fixture alongside the evidence digests. A new seed needs its own provenance review. Retain source notices when distributing any third-party material under its applicable terms. [Repository license](../../LICENSE) · [MIT text](../../LICENSE-MIT) · [Apache-2.0 text](../../LICENSE-APACHE) · [Seed runner](../../fuzz/README.md)
