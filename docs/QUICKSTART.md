# Quickstart

This is a short, practical starting point for using `anno` via the CLI and as a Rust library.

## Install

As a Rust library (from crates.io):

Add `anno` to your `Cargo.toml`:

```toml
[dependencies]
anno = "0.2"
```

CLI (from source): the `anno` binary lives in the `anno-cli` crate (package `anno-cli`, bin `anno`).

```bash
git clone https://github.com/arclabs561/anno
cd anno

# Minimal build (no ML backends):
cargo build --release -p anno-cli --bin anno

# Recommended for most workflows (enables downloads + ONNX ML backends):
cargo build --release -p anno-cli --bin anno --features "eval-advanced onnx"
```

## CLI: extract entities

Prefer `--text` / `--file` over positional text for now (it avoids known arg-order/input-parsing pitfalls).

```bash
anno extract --text "Marie Curie was born in Paris."
```

Pattern-only extraction (emails/dates/money/urls, etc.):

```bash
anno extract --model pattern --text "Email bob@acme.com on 2024-01-15 for $100."
```

Machine-readable output:

```bash
anno extract --model pattern --format tsv --text "Email bob@acme.com on 2024-01-15 for $100."
```

## CLI: within-document coreference

```bash
anno pipeline --coref --text "Marie Curie was born in Paris. She moved to Paris."
```

## CLI: cross-document clustering

Cross-document clustering requires the `eval-advanced` feature (see `docs/BACKENDS.md`).

```bash
anno cross-doc ./docs --threshold 0.6 --format tree
```

## Offsets (Unicode)

Offsets are **character offsets**, not byte offsets. See `docs/CONTRACT.md`.

## Next docs

- `docs/CONTRACT.md` — scope + guarantees
- `docs/BACKENDS.md` — backend selection and feature flags

