# PII fuzz target

Feeds arbitrary bytes (decoded with `String::from_utf8_lossy`) through
`anno::pii::scan_patterns` followed by `anno::pii::redact`, asserting that the
process does not panic and that the output is valid UTF-8.

This crate is standalone: it is **not** a member of the root workspace and it is
**not** run by CI.

## Run

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
