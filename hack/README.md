# Hack Folder - Local Data Testing

This folder is for **local experiments** (scratch scripts, one-off runs, and optional local data).
The repo includes a small, tracked fixture set under `testdata/fixtures/` for reproducible examples.

## Notes

Design/UX/testing notes that used to live in `hack/` have been moved to `docs/notes/hack/`.

## Fixtures (tracked)

Use these for docs/tests without licensing ambiguity:

- `testdata/fixtures/cross_doc/` (3 small `.txt` files)

## Usage

```bash
# Batch extraction over a small directory
./target/debug/anno batch --dir testdata/fixtures/cross_doc --format human

# Cross-document clustering (requires `eval-advanced`)
./target/debug/anno cross-doc testdata/fixtures/cross_doc --format tree --threshold 0.3
```

## Building the binary

```bash
cargo build -p anno-cli --bin anno
```

Note: some popular datasets/news sources have redistribution restrictions. Keep any "real data"
you download/scrape under `hack/real_data/` as **local-only** unless you've verified licensing.

## Quick Test Commands

For fast iteration during development:

```bash
# Run a single test by name (minimal features, fastest)
just t test_name

# Run a single test with full features
just tf test_name

# Run all lib tests
just test

# Full check (format + clippy + tests)
just check
```

These use `cargo nextest` when available (faster, better output) and fall back to `cargo test`.

