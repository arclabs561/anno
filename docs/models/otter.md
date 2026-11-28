# Otter multilingual entity extraction

Use `--model otter-cross` or `--model otter-bi` with a list of entity types. Both selectors use the original Otter decoder and emit character spans in the source text. The cross-encoder reads the labels and text together. The bi-encoder encodes the text and type names separately. [Otter paper](https://aclanthology.org/2026.findings-emnlp.2403/) · [Released models and implementation](https://github.com/whoisjones/otter)

## Prepare the local models

The Rust library and CLI use a persistent Python CPU worker. The tested environment used Python 3.12, PyTorch 2.8.0+cpu and Transformers 4.48.3. Python 3.11 or later supports the checkpoint hash verification. Run these commands from the Anno checkout.

```sh
python3 -m venv .venv-otter
export ANNO_OTTER_PYTHON="$PWD/.venv-otter/bin/python"
"$ANNO_OTTER_PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.8.0'
"$ANNO_OTTER_PYTHON" -m pip install 'transformers==4.48.3' 'tokenizers==0.21.4' 'numpy==1.26.4'
export OTTER_CROSS_MODEL_DIR="$PWD/models/otter-cross-mmbert"
export OTTER_BI_MODEL_DIR="$PWD/models/otter-bi-mmbert"
"$ANNO_OTTER_PYTHON" scripts/prepare_otter.py cross "$OTTER_CROSS_MODEL_DIR"
"$ANNO_OTTER_PYTHON" scripts/prepare_otter.py bi "$OTTER_BI_MODEL_DIR"
```

Each selector needs its own snapshot. The preparation script verifies the pinned inference files, including upstream Python model code, tokenizers and safetensors. Rerunning it repairs missing or damaged files from the local Hugging Face cache or the pinned release. The worker verifies them against the bundled manifest before loading the model offline. The cross weights are 1.24 GB and the bi weights are 1.91 GB. Both checkpoints use Apache-2.0. [Cross checkpoint](https://huggingface.co/whoisjones/otter-cross-mmbert/tree/8729188e4f5fc7948d0e9dfd7d7e6d36c2e7270d) · [Bi checkpoint](https://huggingface.co/whoisjones/otter-bi-mmbert/tree/53e10a09bc71a2e45980a7a257233a28305a5777)

## Extract

```sh
anno extract --model otter-cross --extract-types person,location --threshold 0.5 \
  --text '🙂 Angela Merkel besuchte Berlin.' --format json
anno extract --model otter-bi --extract-types person,location --threshold 0.2 \
  --text '🙂 Angela Merkel besuchte Berlin.' --format json
```

At threshold 0.5, both checkpoints identified *Angela Merkel* at Unicode scalar range `[2,15)` in this example. The cross-encoder also identified *Berlin* at `[25,31)`. The default labels are person, organization and location. Each checkpoint uses its own published threshold by default, 0.5 for cross and 0.2 for bi. A single request can use up to 1,024 input tokens, including the schema prefix in the cross-encoder. Oversized inputs return an error. The upstream span head considers entity spans of up to 30 tokens.

In Rust, use `anno::backends::otter::Otter::from_env(OtterVariant::Cross)` or pass the Python executable, checkpoint directory and variant to `Otter::new`. Holding the model open keeps the checkpoint resident across requests. The bi-encoder uses the publisher's standard inference path, which recomputes type representations per request. [Rust API and checkpoint test](../../crates/anno/src/backends/otter/mod.rs)

On Anno's 120-paragraph RED-FM test panel, cross at 0.5 recovered 316 of 436 reference mentions with 191 unmatched predictions (67.02 exact micro F1). Bi at 0.2 recovered 178 with 93 unmatched predictions (50.35). Raising bi's threshold to 0.5 reduced its micro F1 to 26.06. The panel has 12 paragraphs per language and length group. [Corpus, selection and scoring](../eval/ner-language-length.md)
