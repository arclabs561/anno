# FCoref word alignment across an encoder boundary

This experimental branch combines arlyon’s document-budget implementation (#141) with word-aligned mention scoring (#150). Both APIs share the document budget check and segmented encoder. The default cap is 512 encoded tokens, including special tokens. A caller can select a larger cap or use `None`. Document-level scoring still uses quadratic memory.

The test starts with 502 copies of “x”, then appends “Alice reviewed the proposal. She approved it. Bob objected. She overruled him.” Punctuation is a separate word. The tokenizer produces 521 tokens, which span two encoder calls. Python FastCoref selects 130 subtoken candidates. Projecting endpoints onto source words produces 126 distinct scored mentions and three clusters.

| Cluster | Source mentions |
| --- | --- |
| 1 | Alice, She, She |
| 2 | the proposal, it |
| 3 | Bob, him |

The Rust regression checks all 126 spans against the FP32 PyTorch reference, with absolute logit tolerance 0.01, and compares every cluster member. It exercises an exact limit of 521, a limit of 520 and an unlimited budget. The expected scores come from the upstream model’s `return_all_outputs=True`, taking the largest selected logit per verbatim source span. The reference split the single tokenization after 512 tokens and used the model’s `leftovers` input for the final nine. [FastCoref model at the tested revision](https://github.com/shon-otmazgin/fastcoref/blob/ae224139196d122b225af5a7e73b5fc0b6e1076d/fastcoref/coref_models/modeling_fcoref.py)

## Reproduce on Linux x86-64

From the root of this branch, use a recent Rust toolchain, Python 3.12, uv, curl and the CPU wheels below. The official checkpoint is 346 MiB. The exporter verifies its SHA-256 before loading it.

```bash
set -eu
repo=$PWD
work=$(mktemp -d)
uv venv --python 3.12 "$work/venv"
py="$work/venv/bin/python"
uv pip install --python "$py" 'torch==2.7.1+cpu' --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
uv pip install --python "$py" 'transformers==4.51.3' 'onnx==1.20.0' 'onnxruntime==1.24.2' 'numpy==2.2.6' 'safetensors==0.8.0'
hf=https://huggingface.co/biu-nlp/f-coref/resolve/e91dfff12879495d882ba9460d0c5d5dd44ade59
mkdir -p "$work/source"
for file in pytorch_model.bin config.json tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt; do
  curl -fLsS --retry 3 "$hf/$file" -o "$work/source/$file"
done
export FCOREF_MODEL_DIR="$work/model"
"$py" scripts/fcoref_word_budget/export_model.py "$work/source" "$FCOREF_MODEL_DIR"

export ORT_DYLIB_PATH=$("$py" -c 'import onnxruntime as o; from pathlib import Path; print(next((Path(o.__file__).parent/"capi").glob("libonnxruntime.so.*")))')
mkdir -p "$work/ort"
ln -s "$ORT_DYLIB_PATH" "$work/ort/libonnxruntime.so"
export ORT_LIB_LOCATION="$work/ort" ORT_SKIP_DOWNLOAD=1 ANNO_NO_DOWNLOADS=1
cargo test -p anno --no-default-features --features 'onnx ort/load-dynamic' --test fcoref_word_budget -- --ignored
```

## Run the Aurelius caller

The Aurelius test below uses its source-aligned nlprule tokenization and `LanguageRuntime::analyze`. Use the same shell and model directory as above. The edits apply to a temporary Aurelius checkout. They select the experimental anno dependency and forward the configured limit.

```bash
git clone --quiet https://github.com/arlyon/aurelius.git "$work/aurelius"
git -C "$work/aurelius" checkout --quiet 1319bcdcee28ee5b668b41a2a4f85dd0722b460f
lanes="$work/aurelius/lanes"
sed -i "s|path = \"vendor/anno\"|path = \"$repo/crates/anno\"|" "$lanes/Cargo.toml"
sed -i 's/anno = { version = "=0.13.0"/anno = { version = "=0.14.0"/' "$lanes/Cargo.toml"
sed -i '/num_threads: config.threads,/a\                    max_doc_tokens: Some(config.max_document_tokens),' "$lanes/src/language.rs"
cp "$repo/scripts/fcoref_word_budget/aurelius_word_budget.rs" "$lanes/tests/aurelius_word_budget.rs"
curl -fLsS --retry 3 https://github.com/bminixhofer/nlprule/releases/download/0.6.4/en_tokenizer.bin.gz -o "$work/en_tokenizer.bin.gz"
echo "b500dd208ace9ba218f6b52f8cdab63d4c09d6f2967e9bd8f917bf5984d4468a  $work/en_tokenizer.bin.gz" | sha256sum -c -
gzip -dc "$work/en_tokenizer.bin.gz" > "$FCOREF_MODEL_DIR/en_tokenizer.bin"
export AURELIUS_TEST_LANGUAGE_MODEL_DIR="$FCOREF_MODEL_DIR"
(cd "$lanes" && cargo test --features 'language postgres' --test language --test aurelius_word_budget -- --test-threads=1)
```
