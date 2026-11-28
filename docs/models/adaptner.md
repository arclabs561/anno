# Arabic AdaptNER

`anno extract --model adaptner` finds Arabic mentions with the 21 Wojood labels. A year can have the label `DATE` while forming part of a longer `EVENT`. JSON preserves both source spans.

The backend uses the released AraBERTv02 development checkpoint at revision `c3096335b1e8d00f51451e267b8034a0cc59876f`. The competition submission combined three other models. [Checkpoint and training details](https://huggingface.co/NAMAA-Space/KnowledgeGraphEval-2026-AdaptNER-AraBERTv02-Nested/tree/c3096335b1e8d00f51451e267b8034a0cc59876f)

## Run locally

From the repository root, export the weights once into a new directory. The script checks the downloaded checkpoint and tokenizer hashes and compares the exported FP32 logits with PyTorch on three sequences of different lengths.

```sh
uv run scripts/export_adaptner.py --output /tmp/adaptner-fp32
export ADAPTNER_MODEL_DIR=/tmp/adaptner-fp32
cargo run --locked -p anno-cli --no-default-features --features onnx,extractor-readability --bin anno -- \
  extract --model adaptner --format json \
  --text 'خارطة فلسطين تبين طرق التهجير التي سلكها الفلسطينيون أثناء حرب 1948 ( النكبة ) .'
```

To export offline, pass `--checkpoint-dir` with a downloaded copy of the pinned Hub snapshot. Inference reads the local graph, manifest and tokenizer. The library entry point is `anno::backends::adaptner::AdaptNer::from_path`.

## How to read the output

Source words are delimited by whitespace, matching the released checkpoint's pretokenized inference example. Shape punctuation as its own word when reproducing a tokenized Wojood sentence. Offsets count Unicode scalar values in the exact input text, including its original whitespace. The first subword supplies the logits for each word. Text longer than 256 subwords is evaluated in consecutive windows of at most 254 complete wordpieces plus two special tokens. A word is kept entirely in one window. Context restarts across windows, and the decoder can continue an active mention across the boundary.

Each type has two independent outputs. A logit above zero activates its tag, equivalent to a sigmoid probability strictly above 0.5. A begin tag opens a new span, including when both begin and inside fire for that type. An inside tag continues the span or opens one if necessary. Anno returns the minimum probability of the selected tags as the span confidence. Distinct Wojood labels survive unchanged, including `PERS`, `GPE`, `LOC`, and `FAC`. The same-type BIO channel can represent only one active span at each word.

Run the native checkpoint regression after exporting:

```sh
ADAPTNER_MODEL_DIR=/tmp/adaptner-fp32 cargo test --locked -p anno --features onnx \
  --lib adaptner::tests -- --include-ignored
```

## Public Wojood sample

The native batch CLI scored 829 true positives, 55 false positives and 41 false negatives across the 357 sentences in SinaLab's public test sample, giving 94.53 exact-span-and-type micro F1. Both spans were recovered for 154 of 176 strictly contained cross-type pairs and zero of five same-type pairs. The scorer reads each ordered nesting layer in the reference before comparing typed source offsets. The test sample is a subset from the Wojood corpus used to develop the checkpoint. Its relationship to training limits conclusions about generalization. [SinaLab sample at the scored revision](https://github.com/SinaLab/ArabicNER/blob/ef3a7f4e806a2d909b9107d757c5f65545ae83dc/data/test.txt)

```sh
curl -fL https://raw.githubusercontent.com/SinaLab/ArabicNER/ef3a7f4e806a2d909b9107d757c5f65545ae83dc/data/test.txt -o /tmp/wojood-sample-test.txt
python scripts/qa/adaptner_wojood.py /tmp/wojood-sample-test.txt target/debug/anno /tmp/adaptner-result.json
```
