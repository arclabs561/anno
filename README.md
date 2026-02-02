# anno

Information extraction: named entity recognition and coreference.

Dual-licensed under MIT or Apache-2.0.

API docs: [docs.rs/anno](https://docs.rs/anno)

## Examples

Named entities:

```sh
anno extract --text "Ada Lovelace worked with Charles Babbage in London."
```

```text
PER:2 "Ada Lovelace" "Charles Babbage"
LOC:1 "London"
```

Structured entities (dates, money, emails):

```sh
anno extract --model pattern --text "Contact jobs@acme.com by March 15 for the \$50K role."
```

```text
EMAIL:1 "jobs@acme.com"
DATE:1 "March 15"
MONEY:1 "$50K"
```

Coreference (linking "She" to its referent):

```sh
anno debug --coref -t "Marie Curie discovered radium. She won two Nobel Prizes."
```

```text
Coreference: "Marie Curie" → "She"
```

## Install

The CLI binary lives in `crates/anno-cli`.

```sh
cargo install --path crates/anno-cli --bin anno --features "onnx eval-advanced"
```

See `docs/QUICKSTART.md` for 5-minute setup.
Full docs: `docs/README.md`.
Interface contract: `docs/CONTRACT.md`.

Build rustdocs locally:

```sh
cargo doc -p anno --no-deps --features "eval-advanced discourse"
```
