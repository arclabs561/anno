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
To hand the comparison to another reader, put both editions in one directory.
Run the example from that directory with the two filenames and redirect stdout
to `changed-dates.json`. Send all three files together. The reader can check
each original file against its SHA-256 in `sources`, slice the decoded UTF-8 text
at each reported range, and open both drawing lines. Preserve the original
bytes when copying the notices, including line endings.

[Example source](examples/bulletin_diff.py) · [Notice license](examples/bulletins/LICENSE)

## Install from a checkout

From the repository root with `uv` and `maturin` available:

```sh
cd crates/anno-py
uv venv && maturin develop --uv
```

To test the installed wheel from a fresh checkout, run the following from the repository root. The wheel and test environment are created outside the source tree.

```sh
work=$(mktemp -d)
(cd crates/anno-py && maturin build --locked --out "$work/wheels")
python3 -m venv "$work/venv"
"$work/venv/bin/python" -m pip install "$work"/wheels/*.whl pytest
"$work/venv/bin/python" -I -m pytest -q crates/anno-py/tests
```

To exercise ONNX and Fastino together, configure ONNX Runtime for the build and add `--features extension-module,onnx,fastino` to the Maturin command. With local checkpoints and worker interpreters configured as described below, the following test opens each supplied model, extracts twice, and checks the resulting Unicode source spans. Set `ANNO_PY_TEST_BERT_DIR` to include a local BERT NER ONNX checkpoint. Set `ANNO_PY_TEST_GLINER_EDGE=knowledgator/gliner-pii-edge-v1.0` to include a cached GLiNER PII Edge checkpoint. Set `ANNO_PY_TEST_CLI` to an Anno executable to compare its worker results with the installed wheel. The worker-checkpoint CI job verifies the two Otter snapshots, GLiNEREL and Kniv’s fixed-label NER assets before running that comparison offline. [CI workflow](../../.github/workflows/worker-checkpoint-replay.yml)

```sh
ANNO_PY_TEST_MODELS=1 "$work/venv/bin/python" -I -m pytest -q \
  crates/anno-py/tests/test_local_checkpoints.py
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

## Select a model in a checkout build

`Extractor()` and `anno_py.extract()` run the pattern and heuristic stack offline.
Reuse an `Extractor` to keep a selected checkpoint loaded between documents.
All selectors return entities with end-exclusive Python string offsets.

| Backend | Extraction schema | Setup |
|---|---|---|
| `bert` | Fixed checkpoint labels | Build with `onnx`. Supply a Hugging Face model ID or a local directory with `model.onnx` and `tokenizer.json`. |
| `gliner` | Prompted, defaults to person, organization and location | Build with `onnx`. Supply a Hugging Face ONNX model ID or use the default. |
| `otter-cross`, `otter-bi` | Prompted multilingual NER | [Prepare the matching snapshot and worker](../../docs/models/otter.md). Set `OTTER_CROSS_MODEL_DIR` or `OTTER_BI_MODEL_DIR`. |
| `glinerel-72m` | Prompted English NER | [Install the worker](../../docs/models/glinerel-72m.md). Uses the pinned Hub snapshot or `ANNO_GLINEREL_MODEL_DIR`. |
| `kniv` | Fixed English BIO labels | [Prepare the checkpoint and worker](../../docs/models/kniv.md). Set `KNIV_MODEL_DIR`. |
| `adaptner` | Fixed Arabic Wojood labels, including nested spans | Build with `onnx` and [prepare the local export](../../docs/models/adaptner.md). Set `ADAPTNER_MODEL_DIR`. |
| `gliner2-pii` | Prompted PII extraction, defaults to email and phone | Build with `fastino`. Uses the [pinned Hub export or a local snapshot](../../docs/models/gliner2-pii.md) at `GLINER2_PII_MODEL_DIR`. |

For the six named specialists, pass `model="/path/to/checkpoint"` to override
the directory setting. Worker-backed selectors start the current Python
interpreter. Set `ANNO_OTTER_PYTHON`, `ANNO_GLINEREL_PYTHON`, or `KNIV_PYTHON`
to use a separate environment with the backend's dependencies. The native
ONNX and Fastino selectors are available together in this checkout build:

```sh
cd crates/anno-py
maturin develop --uv --features extension-module,onnx,fastino
```

```python
import anno_py

arabic = anno_py.Extractor(backend="adaptner")
otter = anno_py.Extractor(backend="otter-bi", labels=["person", "location"])
text = "🙂 Angela Merkel besuchte Berlin."
for entity in otter.extract(text):
    print(entity.label, text[entity.start:entity.end])
```

For prompted models, `labels` replaces the default schema and `threshold`
overrides the checkpoint's detection threshold. Otter-bi defaults to 0.2.
Fixed-schema models raise `ValueError` for either option. Use
`ANNO_NO_DOWNLOADS=1` to require cached or local model assets. An unavailable
native feature or model raises `RuntimeError`. The GLiNEREL and Kniv
integration guides also describe relation extraction through the CLI and Rust.

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
