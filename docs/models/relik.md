# ReLiK entity linking with a supplied catalog

`anno relik` finds mentions in English text and links them to candidate titles supplied with the document. Its JSON output includes the caller's stable ID, predicted title, exact source text, half-open Unicode character range, and reader probability for that candidate. The reader uses the trained AIDA DeBERTa-v3-base checkpoint and an explicit no-match option. No Wikipedia retrieval index is needed for this mode. [Upstream reader-only example](https://github.com/SapienzaNLP/relik/tree/850da1c62a922e8e1866c935369c518c70f4dccb#usage) · [Reader checkpoint](https://huggingface.co/sapienzanlp/relik-reader-deberta-v3-base-aida/tree/3bed88483cabb40188d7a3f8e79ec895c45b3a28)

Prepare the CPU environment and the model once from an Anno checkout. The commands below use the upstream implementation with the tested CPU PyTorch and Transformers versions. The reader package's original dependency pins target an older environment, so its source installation uses the dependencies installed explicitly here.

```sh
python3.12 -m venv .venv-relik
py="$PWD/.venv-relik/bin/python"
"$py" -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.8.0'
"$py" -m pip install 'transformers==4.48.3' 'numpy==1.26.4' 'spacy==3.7.5' \
  'pytorch-lightning==2.3.3' 'lightning==2.3.3' 'lightning-fabric==2.3.3' 'torchmetrics==1.4.3' \
  'typer==0.12.5' 'click==8.1.8' 'art==6.2' 'rich==13.9.4' 'pprintpp==0.4.0' \
  'colorama==0.4.6' 'termcolor==2.4.0' 'overrides==7.7.0' 'sentencepiece==0.2.1' \
  'hydra-core>=1.3,<1.4' scikit-learn psutil safetensors huggingface-hub
"$py" -m pip install --no-deps 'git+https://github.com/SapienzaNLP/relik@850da1c62a922e8e1866c935369c518c70f4dccb'
"$py" scripts/prepare_relik.py ./relik-aida
```

The acquisition script downloads the pinned reader snapshot and a base encoder configuration, verifies the 747 MB checkpoint's SHA-256, loads tensors with `weights_only=True`, and saves the complete local encoder, reader head, learned extra embeddings and tokenizer. Allow space for both the download and the prepared checkpoint. Inference reads local safetensors and verifies the preparation receipt. All candidate scoring happens on CPU.

For example, save the following as `candidates.json`.

```json
[
  {"id":"player-23","title":"Michael Jordan"},
  {"id":"league-nba","title":"National Basketball Association"},
  {"id":"player-walter","title":"Walter Jordan"},
  {"id":"country-jo","title":"Jordan"},
  {"id":"list-50","title":"50 Greatest Players in NBA History"}
]
```

```sh
anno relik --checkpoint ./relik-aida --candidates candidates.json --python "$py" \
  --text 'Michael Jordan was one of the best players in the NBA.'
```

The pinned model linked “Michael Jordan” at [0, 14) to `player-23` and “NBA” at [50, 53) to `league-nba` on this example. A second run with a Unicode prefix, tabs, paragraph breaks and repeated names recovered both occurrences of each title across overlapping windows. All 220 loaded state tensors matched the released PyTorch weights exactly after preparation.

Omit `--text` to read the document from stdin. The input catalog must contain 1–100 unique title/ID pairs. Titles share the reader's 512-subword budget with each source window. The upstream reader represents each title with at most 22 subwords. Anno fits the entire candidate set and shrinks the source windows as needed. Overlapping windows normally cover 32 words with a stride of 16. For competing predictions with identical source bounds, Anno keeps the title with the higher reader probability. The probability is conditional on the supplied list of titles. An entity missing from the list cannot be linked to its correct ID.
