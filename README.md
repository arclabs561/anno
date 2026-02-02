# anno

Information extraction: named entity recognition and coreference.

Dual-licensed under MIT or Apache-2.0.

API docs: [docs.rs/anno](https://docs.rs/anno)

```sh
anno extract --text "Marie Curie was born in Warsaw."
```

```text
[Marie Curie] PERSON
[Warsaw] LOCATION
```

See `docs/QUICKSTART.md` for 5-minute setup.
Full docs: `docs/README.md`.
Interface contract: `docs/CONTRACT.md`.

## Install

The CLI binary lives in `crates/anno-cli` (bin name: `anno`).

```sh
cargo install --path crates/anno-cli --bin anno --features "eval-advanced onnx"
```

Build rustdocs locally:

```sh
cargo doc -p anno --no-deps --features "eval-advanced discourse"
```
