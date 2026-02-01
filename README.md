# anno

Information extraction: named entity recognition and coreference.

Dual-licensed under MIT or Apache-2.0.

API docs: `https://docs.rs/anno` (rustdoc).

```sh
anno extract --text "Marie Curie was born in Warsaw."
```

```text
[Marie Curie] PERSON
[Warsaw] LOCATION
```

More: `docs/QUICKSTART.md`.
Docs index: `docs/README.md`.
Contract: `docs/CONTRACT.md`.

Build rustdocs locally:

```sh
cargo doc -p anno --no-deps --features "eval-advanced discourse"
```

Install the CLI:

```sh
# This repo’s CLI binary lives in `crates/anno-cli` (bin name: `anno`).
# Install from source:
cargo install --path crates/anno-cli --bin anno --features "eval-advanced onnx"
```