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

Zero-shot extraction (define your own entity types):

```sh
anno extract --model gliner --extract-types "DRUG,SYMPTOM" \
  --text "Aspirin can treat headaches and reduce fever."
```

```text
drug:1 "Aspirin"
symptom:2 "headaches" "fever"
```

Coreference resolution:

```sh
anno debug --coref -t "Sophie Wilson designed the ARM processor. She revolutionized mobile computing."
```

```text
Coreference: "Sophie Wilson" → "She"
```

## Install

```sh
cargo install --path crates/anno-cli --bin anno --features "onnx eval-advanced"
```

See `docs/QUICKSTART.md` for library usage and more examples.

## Docs

- `docs/QUICKSTART.md` — 5-minute setup
- `docs/CONTRACT.md` — interface contract
- `docs/BACKENDS.md` — backend selection
