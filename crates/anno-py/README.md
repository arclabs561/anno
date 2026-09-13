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
switches backends. Invalid backend options raise `ValueError`. Rust
`InvalidInput` errors map to `ValueError`; other variants, including parsing
and inference errors, map to `RuntimeError`.

## Experimental Fastino source build

Fastino provides zero-shot typed extraction and single-label classification.
It is an explicit `FastinoExtractor` API because classification is not a
capability of every `Extractor` backend. The standard wheel stays offline-only:
this source build includes native ONNX Runtime support but does not bundle
model weights.

```sh
cd crates/anno-py
maturin develop --uv --features extension-module,fastino
```

The default model is the verified pinned
`jugaadsrl/gliner2-multi-v1-onnx` export. Its first use downloads roughly
1.25 GB of assets into the Hugging Face cache. To require a pre-populated
cache, set `ANNO_NO_DOWNLOADS=1`; a missing asset then raises `RuntimeError`
instead of downloading or falling back.

```python
from anno_py import FastinoExtractor

fastino = FastinoExtractor()
entities = fastino.extract(
    "Acme Corp signed a deal in Paris.",
    labels=["organization", "location"],
    threshold=0.5,
)
classes = fastino.classify(
    "This product is wonderful, I love it.",
    labels=["positive", "negative", "neutral"],
)
print(classes[0].label, classes[0].probability)
```

`classify` returns all supplied labels in descending softmax probability. The
current Fastino classifier is single-label; it has no threshold argument. If
Fastino's internal count head produces no classification, `classify` raises
`RuntimeError`; it does not fabricate or normalize an all-zero score vector.
`FastinoExtractor` releases the GIL while loading and running the native
model. The backend is experimental, and this source-build path has only been
validated on the build host; do not treat it as a standard wheel distribution
commitment.

## Tests

```sh
pytest tests/
```

CI builds the Fastino extension and tests its conversion and offline error
contracts without model downloads. The cached-model parity gate is manual:
with the Fastino extension installed and the pinned assets already cached,
run from the repository root:

```sh
ANNO_NO_DOWNLOADS=1 cargo run -p anno-py --example fastino_reference \
  --features fastino > /tmp/anno-fastino-reference.json
ANNO_NO_DOWNLOADS=1 ANNO_PY_TEST_MODELS=1 \
  ANNO_PY_FASTINO_REFERENCE=/tmp/anno-fastino-reference.json \
  python -m pytest crates/anno-py/tests -q
```

The oracle comparison checks entity labels, character spans, confidence scores,
and ordered classification probabilities against the direct Rust API.
