# Fuzzing anno

## Reviewed catalog assignments

The `reviewed_kb` target mutates short sequences of catalog operations. Each input starts with both document-local identities and identities shared across documents. Several records have identical visible names and catalog keys. The oracle checks who actually acquired the new binding, who kept the old one, and whether a stale or foreign approval was rejected.

The required `Test (minimal)` CI job replays the [shared seeds](seeds/reviewed_kb/) with ordinary Cargo. A separate coverage-guided job runs 1,000 inputs on relevant pull requests and five-minute campaigns weekly or when manually dispatched. CI retains failing inputs as a workflow artifact. [Seed replay](../crates/anno/tests/reviewed_kb_seeds.rs) · [Campaign workflow](../.github/workflows/reviewed-kb-fuzz.yml)

### Comparing seed selection locally

Run paired rounds from the repository root. The first command starts a local checkpoint. Repeat the second with consecutive round numbers. A repeated round leaves the counters unchanged.

```sh
cargo run --locked -p anno-eval --features eval --example reviewed_seed_campaign -- /tmp/reviewed-seeds.json 0
cargo run --locked -p anno-eval --features eval --example reviewed_seed_campaign -- /tmp/reviewed-seeds.json 1
```

Each round replays the seven unmodified regression seeds, then applies the same deterministic mutation recipe to one parent chosen by Muxer UCB and one parent chosen in rotation. Each lane gets one probe and maintains its own exposure and discovery counts. A seed receives one win when the native oracle confirms a previously unseen combination of track and receipt outcome in that lane. Outcomes cover stale reviews and successful single- or two-track writes. Smoke-seed outcomes, repeated discoveries and successful execution alone add no wins. A typed visible-binding mismatch must recur on a second execution to count as a failure. It stops the campaign and identifies the lane and round for reproduction. Other panics stop the run before a checkpoint is written. These are semantic outcomes; compiler edge coverage belongs to the separate instrumented fuzz campaign.

Only the seven named, first-party synthetic seeds compiled into the example can receive credit. To admit a proposed QA or judge case, reduce it to a repository-owned operation trace, review its provenance and expected outcome, add it to the shared native seed replay and then add its ID to the campaign. Holdouts and unreviewed inputs stay outside the registry. State contains the version, corpus fingerprint, round and finite per-seed counts and outcome bits. Keep the checkpoint in a trusted local directory. Schema validation does not establish the authenticity of a checkpoint from another writer. Reinitialize on a changed corpus or oracle, and bump the schema when changing the mutator or reward definition.

To reproduce a campaign from the repository root, use the pinned nightly and the fuzz workspace's lockfile.

```sh
rustup toolchain install nightly-2026-05-01 --profile minimal
cargo install --locked cargo-fuzz --version 0.13.2
cargo +nightly-2026-05-01 fetch --locked --manifest-path fuzz/Cargo.toml
mkdir -p fuzz/corpus/reviewed_kb
cargo +nightly-2026-05-01 fuzz run --dev --sanitizer address reviewed_kb \
  fuzz/corpus/reviewed_kb fuzz/seeds/reviewed_kb -- -runs=1000 -max_len=48
```

The first corpus directory is writable. libFuzzer puts new coverage inputs there, and Git ignores the directory. The second directory supplies small, reviewed synthetic inputs. Keep the named regression cases even if a coverage minimizer considers two of them equivalent.

Seeds can be read as operations. Digits `0` through `5` select a track, `p` previews a reviewed assignment, and the end of the input applies it. For example, `0pBA` previews track 0, changes its key, then restores it. The approval must still be stale. In `0p2A`, track 2 already has the requested binding on an independent identity. That write changes nothing, so the pending approval on track 0 still applies to its two shared tracks. [Operation definitions](../crates/anno/tests/support/reviewed_kb_fuzz.rs)

When a campaign finds a failure, reproduce it with the failing file in place of the two corpus directories. Use `cargo +nightly-2026-05-01 fuzz tmin --dev reviewed_kb <failing-file>` to minimize it. Promote the shortest trace that still demonstrates the behavior to `fuzz/seeds/reviewed_kb/` and record the expected outcome in the Cargo replay test. An adversarial reviewer can contribute a scenario or a new assertion through the same path. [libFuzzer corpus handling](https://llvm.org/docs/LibFuzzer.html#running)

## PII

Feeds arbitrary bytes (decoded with `String::from_utf8_lossy`) through
`anno::pii::scan_patterns` followed by `anno::pii::redact`, asserting that the
process does not panic and that the output is valid UTF-8.

The fuzz crate uses a standalone Cargo workspace. The PII target runs manually.

### Run

```sh
cargo install cargo-fuzz   # once
cd fuzz
cargo +nightly fuzz run pii
```

A short smoke run instead of fuzzing indefinitely:

```sh
cargo +nightly fuzz run pii -- -runs=1000
```

To only check that the target compiles:

```sh
cargo check --manifest-path fuzz/Cargo.toml
```

`cargo-fuzz` needs a nightly toolchain (`cargo +nightly`). If the toolchain or
`cargo-fuzz` is unavailable on your machine, the target is unverified there;
the source remains a reproducible harness.

The seven reviewed-KB seed files are project-authored synthetic operation sequences under the repository's `MIT OR Apache-2.0` license. Additional corpora need their own recorded source and redistribution terms before inclusion. [Model and fixture licensing](../verification/source_rights/README.md)
