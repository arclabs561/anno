# anno

Information extraction: named entity recognition and coreference.

Dual-licensed under MIT or Apache-2.0.

API docs: [docs.rs/anno](https://docs.rs/anno)

## Backends

| Backend | Zero-shot | Weights | Notes |
|---------|-----------|---------|-------|
| `stacked` (default) | No | HuggingFace | BERT/GLiNER + regex + heuristic |
| `gliner` | Yes | [onnx-community/gliner_small-v2.1](https://huggingface.co/onnx-community/gliner_small-v2.1) | Span classifier, custom entity types |
| `gliner2` | Yes | [onnx-community/gliner-multitask-large-v0.5](https://huggingface.co/onnx-community/gliner-multitask-large-v0.5) | Multi-task (NER + classification) |
| `nuner` | Yes | [deepanwa/NuNerZero_onnx](https://huggingface.co/deepanwa/NuNerZero_onnx) | Token classifier, arbitrary-length entities |
| `w2ner` | No | [ljynlp/w2ner-bert-base](https://huggingface.co/ljynlp/w2ner-bert-base) | Nested/discontinuous entities |
| `bert-onnx` | No | [protectai/bert-base-NER-onnx](https://huggingface.co/protectai/bert-base-NER-onnx) | Traditional fixed-label NER |
| `pattern` | No | None | Regex (dates, emails, money) |
| `heuristic` | No | None | Capitalization + context |
| `crf` | No | Bundled (`bundled-crf-weights`) | CRF with bundled trained weights when enabled; can load custom weights |
| `hmm` | No | Bundled (`bundled-hmm-params`) | HMM with optional bundled params (compact); baseline/education |
| `ensemble` | No | Varies | Weighted voting across backends |

ML backends require `--features onnx`. ONNX model weights download from HuggingFace on first use.

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
# From this repo:
cargo install --path crates/anno-cli --bin anno --features "onnx eval-advanced"
```

See `docs/QUICKSTART.md` for library usage and more examples.

## Docs

- `docs/QUICKSTART.md` — 5-minute setup
- `docs/CONTRACT.md` — interface contract
- `docs/BACKENDS.md` — backend selection
