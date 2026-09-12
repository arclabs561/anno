# anno-py

Python bindings for [anno](https://github.com/arclabs561/anno).

The default build uses pattern + heuristic backends: no model downloads and it
works offline. Offsets are character offsets, so
`text[e.start:e.end] == e.text`.

## Install (development)

```sh
cd crates/anno-py
uv venv && maturin develop --uv
```

## Usage

```python
import anno_py

ents = anno_py.extract("Contact Jane Doe at jane.doe@example.com.")
for e in ents:
    print(e.text, e.label, e.start, e.end, e.confidence)
```

For repeated calls, reuse an extractor: `ex = anno_py.Extractor()`, then `ex.extract(text)`.

## Optional ONNX models

Model-backed extraction is opt-in. Build the extension with the `onnx` Cargo
feature:

```sh
cd crates/anno-py
maturin develop --uv --features extension-module,onnx
```

Then select a backend explicitly. BERT accepts a HuggingFace model ID or a
local model directory containing `model.onnx`, `tokenizer.json`, and optionally
`config.json`. GLiNER accepts a HuggingFace model ID and zero-shot labels;
those labels replace GLiNER's default labels for that extractor.

```python
# BERT's fixed label taxonomy; the default model is used when model is omitted.
bert = anno_py.Extractor(backend="bert", model="protectai/bert-base-NER-onnx")

# Zero-shot labels are supported by GLiNER only.
gliner = anno_py.Extractor(
    backend="gliner",
    model="onnx-community/gliner_small-v2.1",
    labels=["person", "organization", "drug"],
    threshold=0.5,
)
```

`threshold` is a GLiNER-only option. BERT uses the model's fixed BIO label
taxonomy and does not accept zero-shot labels or a threshold override.

Model construction may download an uncached HuggingFace model. Set
`ANNO_NO_DOWNLOADS=1` to forbid new downloads; cached models and local BERT
model directories still work. The `offline` backend always constructs only
the pattern + heuristic stack, including in an ONNX-enabled wheel. An offline
wheel raises `RuntimeError` if asked for `bert` or `gliner`; it never silently
switches backends. Invalid backend options raise `ValueError`.

## Tests

```sh
pytest tests/
```
