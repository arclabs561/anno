# GLiNER2-PII

`anno extract --model gliner2-pii` runs Fastino’s multilingual PII checkpoint with the native GLiNER2 ONNX backend. Its default schema is `email,phone_number` at threshold 0.5. Pass `--extract-types` to request a different schema. The Apache-2.0 checkpoint's model card lists 42 PII categories across seven languages. [Original checkpoint and label list](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi/tree/1cb4166094dc58fa8d836429f060d6c95f62b495) · [Paper](https://arxiv.org/abs/2605.09973)

Build the checkout CLI with the Fastino feature and a configured ONNX Runtime. [Build guide](../BUILDING.md)

```sh
cargo build --locked -p anno-cli --no-default-features --features gliner2-fastino,extractor-readability
target/debug/anno extract --model gliner2-pii --format json \
  --text 'Élodie: Email john.smith@acme.com or call +1 415 555 0199.'
```

The email occupies character range `14..33` and the phone number occupies `42..57`. Both cover the original source text, including the leading plus sign on the phone number.

## Download the tested CPU export

The selector resolves the eight ONNX graphs and tokenizer in the pinned community snapshot automatically, subject to Anno’s download settings. For an explicit local copy, download the FP32 directory and set the override to the snapshot root. The first command uses the Hugging Face `hf` CLI.

```sh
hf download jugaadsrl/gliner2-privacy-filter-PII-multi-onnx \
  --revision e594898629d452e8311796f5f329c7edbeda907c \
  --include 'fp32_v2/*' --local-dir models/gliner2-pii
GLINER2_PII_MODEL_DIR=models/gliner2-pii ANNO_NO_DOWNLOADS=1 \
  target/debug/anno extract --model gliner2-pii --extract-types email \
  --text 'Email john.smith@acme.com or call +1 415 555 0199.' --format json
```

The root contains `fp32_v2/tokenizer.json` and `fp32_v2/{encoder,token_gather,span_rep,schema_gather,count_pred_argmax,count_lstm_fixed,scorer,classifier}_fp32.onnx`. These nine files require about 1.25 GB. The Rust constructor is `anno::backends::gliner2_fastino::GLiNER2Fastino::from_pii()`. Use `GLiNER2Fastino::from_pii_local(path)` to select local weights directly in Rust, and `ZeroShotNER::extract_with_types` for custom labels. [Pinned export](https://huggingface.co/jugaadsrl/gliner2-privacy-filter-PII-multi-onnx/tree/e594898629d452e8311796f5f329c7edbeda907c/fp32_v2)

In a Fastino-enabled build of the Python bindings, the same local directory can be passed as `model`. The source build uses `maturin develop --features extension-module,fastino` from `crates/anno-py` inside a Python virtual environment.

```python
from anno_py import Extractor

extractor = Extractor(backend="gliner2-pii", model="models/gliner2-pii", labels=["email"])
text = "Élodie: Email john.smith@acme.com or call +1 415 555 0199."
print([(e.label, e.start, e.end) for e in extractor.extract(text)])
# [('EMAIL', 14, 33)]
```

## Convert the original PyTorch checkpoint

The preparation command downloads Fastino’s pinned PyTorch checkpoint and the upstream V2 converter, checks their SHA-256 digests and validates the eight generated FP32 graphs before installing them with the tokenizer. Conversion also creates temporary FP16 graphs. Use a new output directory with several GB of free working space. For pre-downloaded inputs, pass `--source-dir PATH` and `--converter FILE`.

```sh
python -m venv .venv-export
. .venv-export/bin/activate
python -m pip install 'torch==2.15.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install 'gliner2[local]==2.2.0' 'transformers==5.22.2' \
  'onnxruntime==1.32.0' 'onnx==1.23.2' 'onnxscript==0.7.2' sentencepiece
python scripts/export_gliner2_pii_onnx.py --output models/gliner2-pii-rebuilt
GLINER2_PII_MODEL_DIR=models/gliner2-pii-rebuilt ANNO_NO_DOWNLOADS=1 \
  target/debug/anno extract --model gliner2-pii --format json \
  --text 'Élodie: Email john.smith@acme.com or call +1 415 555 0199.'
```

[Exporter source](https://github.com/dariofinardi/gliner2-rs/blob/7626d237d8c58f0fcfadd094e8360e7d5ae9c8ee/onnx_conversion_scripts/export_gliner2_onnx_fragments_v2.py)

The regenerated graphs and the published export each matched all upstream labeled contact spans on a fixed 75-document multilingual REDACT panel. The named CLI also recovered the two Unicode source ranges above using both exports. Run the checkpoint integration test against a prepared directory to check the default schema and an email-only request.

```sh
GLINER2_PII_MODEL_DIR=models/gliner2-pii ANNO_NO_DOWNLOADS=1 \
  cargo test --locked -p anno --features gliner2-fastino --test gliner2_fastino_integration \
  pii_checkpoint_extracts_contacts_and_honors_a_narrower_schema -- --ignored
```

## Contact-only privacy requests

`anno privacy --types CONTACT` reports emails and phone numbers. With the
`gliner2-fastino` build feature, it adds GLiNER2-PII when a complete export
already exists in `GLINER2_PII_MODEL_DIR` or in the pinned Hub cache. The
automatic selector opens local files directly. When the checkpoint is
unavailable, regular expressions and the structured PII scanner supply the
contacts. A model inference failure also falls back to those rules.

When sources detect overlapping pieces of the same contact, the report spans
their union. Both redaction and pseudonymization replace the entire union,
including a rule-detected prefix or suffix that a model missed.

```sh
anno privacy --types CONTACT --action redact --text 'Email kim@example.org.'
```

An explicit `--model` selects that backend. Requests for all PII types or
a mixture of types use the general stacked model by default.

On a separate REDACT sample of 50 synthetic documents across 25 locales, the
previous default and the local fallback each matched 44 of 55 disclosed
contacts exactly, with 13 extra spans. Adding the prepared checkpoint matched
53 exactly, with 14 extras. Both remaining contacts were partially masked.
Previously 5 contacts were partially masked and 6 were entirely missed.
Unmasked characters inside annotated contacts fell from 121 to 12. Masked
characters outside annotated contacts rose from 147 to 203. Every character
the previous default masked remained covered on this sample. Selection used
two documents per locale, excluded the earlier 75-record panel, and limited
source texts to 900 Unicode characters. These measurements concern email and
phone only. [REDACT sample and license](https://github.com/guneeshvats/REDACT-PII-Benchmark/tree/252232ac29669beeb99cb6474eeb2736761be508)

### Reproduce the CONTACT comparison

The replay driver runs the actual `privacy --types CONTACT` command for each
of the three routes. It verifies
the dataset, the two-stage selection of the 50 documents and the nine pinned
FP32 model files. It reports aggregate exact matches and masked character
counts. It also compares actual redaction output with the reported spans on
every tenth document and on documents with a partially masked contact under
the prepared route. [Replay driver](../../scripts/qa/contact_privacy.py) · [Aggregate receipt](../eval/contact-privacy.json)

From the Anno checkout, build a baseline CLI at the revision preceding the
CONTACT change and the current CLI using the same ONNX Runtime. The commands
use Python 3.11 or newer, the published FP32 snapshot from the download section
above at `models/gliner2-pii`, and a temporary directory for the baseline build
and the REDACT sample. `--jobs 2` can load two copies of the checkpoint at once.

```sh
study_dir=$(mktemp -d)
git worktree add --detach "$study_dir/before" 437a37d864c06704123e8700bdc3565dd3e5448d
cargo build --locked --manifest-path "$study_dir/before/Cargo.toml" --target-dir "$study_dir/before-target" \
  -p anno-cli --no-default-features --features gliner2-fastino,extractor-readability --bin anno
cargo build --locked -p anno-cli --no-default-features --features gliner2-fastino,extractor-readability --bin anno
curl -fL https://media.githubusercontent.com/media/guneeshvats/REDACT-PII-Benchmark/252232ac29669beeb99cb6474eeb2736761be508/data/pii_benchmark_sample1000.json \
  -o "$study_dir/sample.json"
python3 scripts/qa/contact_privacy.py --data "$study_dir/sample.json" --model-dir models/gliner2-pii \
  --before-cli "$study_dir/before-target/debug/anno" --cli target/debug/anno --jobs 2 > "$study_dir/result.json"
diff -u docs/eval/contact-privacy.json "$study_dir/result.json"
git worktree remove "$study_dir/before"
```
