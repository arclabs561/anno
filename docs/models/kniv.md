# Kniv v7.1 English extraction

Select `--model kniv` for the fixed-label English BIO model or for within-window relations. This optional backend uses a persistent local Python process and the release's ONNX graphs. Each worker reuses loaded CPU sessions across documents. The Rust library and standard CLI can be built without ONNX Cargo features. Python 3.11+ is required for release verification. [Pinned model and upstream implementation](https://huggingface.co/dragonscale-ai/kniv-v7-large-en/tree/44e6d50774a22ada4fd825e1c48663afb5fd7dc0)

```sh
python3 -m venv .venv-kniv
.venv-kniv/bin/python -m pip install 'onnxruntime==1.32.0' 'tokenizers==0.23.3' 'numpy==2.3.5' huggingface-hub
export KNIV_PYTHON="$PWD/.venv-kniv/bin/python"
export KNIV_MODEL_DIR="$PWD/kniv-v7-large-en"
"$KNIV_PYTHON" scripts/download_kniv.py "$KNIV_MODEL_DIR"
anno extract --model kniv --text 'Ada Lovelace was born in London.'
anno extract --model kniv --extract-relations --relation-types 'place of birth' \
  --text 'Ada Lovelace was born in London.' --format json
```

Rerun the preparation command to repair missing or damaged files. It reuses verified files in the model directory and the Hugging Face cache before downloading from the pinned release.

The fixed-label extractor scores the 18 OntoNotes-style classes. Anno applies its standard tag mapping, including PERSON → PER and both GPE and LOC → LOC. Each mention's confidence is the geometric mean of its decoded token tag scores. For relations, Kniv uses a separate set of grouped entity occurrences. Supply node and predicate names with `--extract-types` and `--relation-types` in the relation command. The standard triple output represents each endpoint by the group's first source occurrence and includes positive, asserted facts. Its score is the sigmoid of the upstream calibrated margin. The upstream runtime rounds the margin to three decimals. Upstream supplies no node membership scores. Anno's generic relation nodes use zero as the unavailable-confidence sentinel.

For complete fact attributes, call `anno::backends::kniv::Kniv::from_env()?.analyze(text, &["place of birth"])`. The document retains all grouped occurrences, including repeated names, and every upstream fact with polarity, modality, holder, time and raw margin. Node indices belong to their individual window. All spans use half-open Unicode scalar offsets into the full source. A fact cannot directly link nodes across windows.

Anno preserves source punctuation and uses the upstream word tokenizer and sentence packing. Oversized sentences and windows are split until each word fits. For relation requests the split reserves enough space for every possible mention marker. A single word that exceeds the encoder budget yields an explicit error.

On a 128-sentence hash sample of CrossNER's CoNLL test copy, the integration returned exactly the upstream Kniv linguistic head's 170 person, organization and location predictions after Anno label mapping. The native checkpoint smoke test also checks the directed place-of-birth example.

```sh
cargo test -p anno --lib backends::kniv::tests -- --include-ignored
```
