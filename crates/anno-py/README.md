# anno-py

Python bindings for [anno](https://github.com/arclabs561/anno).

The default extractor runs offline. Offsets count Unicode code points and
index Python strings directly.

## Compare dates in two bulletins

A harbor office extended its application deadline. From the repository root,
install the released wheel and run the example on the two original notices:

```sh
cd crates/anno-py
python3 -m venv .venv
.venv/bin/python -m pip install anno-py==0.1.0
.venv/bin/python examples/bulletin_diff.py \
  examples/bulletins/bulletin-edition-1.txt examples/bulletins/bulletin-edition-2.txt
grep -n 'public drawing' examples/bulletins/bulletin-edition-*.txt
```

The JSON shows December 4 removed and December 7 added. Each occurrence
includes its source line and Unicode character ranges. Each file has a SHA-256.
The PyPI wheel supplies contiguous spans and leaves the example's `detector`
field null. The checkout build also supplies the detector when available.

The final command shows that both notices still schedule the drawing on
December 9. The example compares counts of detected date strings. Passing the
same file twice gives an empty `date_count_changes` list, as does moving a date
to another event while preserving its count. Substitute your own UTF-8 files
to review another pair of editions.
[Example source](examples/bulletin_diff.py) · [Notice license](examples/bulletins/LICENSE)

## Install from a checkout

From the repository root with `uv` and `maturin` available:

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

In a checkout build, a contiguous entity has `e.segments == [(e.start, e.end)]`.
For a discontinuous entity, `e.start` and `e.end` bound the whole mention,
including any gaps. Get the exact source pieces with
`[text[start:end] for start, end in e.segments]`.

Entities from a checkout build also expose optional `source`, `method`, and `model_version`
fields when its producing backend recorded provenance. `source` identifies the
layer that produced that entity, which can differ across results from a stacked
extractor. It does not identify the requested extractor configuration or prove
model artifact selection or device placement. `model_version` remains `None`
when the backend did not record one; it is never inferred from a backend name.

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
