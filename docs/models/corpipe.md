# CorPipe 26 multilingual coreference

`anno corpipe` accepts one raw UTF-8 document on stdin or with `--text`. It runs a local UDPipe 1 tokenizer and dependency parser for the input language, then invokes the official CorPipe 26 one-stage predictor. Each invocation loads the checkpoint once. The released one-stage model can predict overt mentions and implicit empty nodes. [CorPipe source and model list](https://github.com/ufal/crac2026-corpipe/tree/ec853c3b2d97890f4ef04114b10af6a1ac01d78f) · [System paper](https://aclanthology.org/2026.codi-1.27/)

## Setup

Use a local Python environment with the dependencies from the upstream requirements file and `ufal.udpipe`. The upstream code is MPL-2.0, and the released CorPipe weights use CC BY-NC-SA 4.0. Select a UDPipe model trained for the input language and review its license. The example uses the English EWT model from UD 2.5. [Base model card and license](https://huggingface.co/ufal/corpipe26-onestage-corefud1.4-base-260702/tree/cb62351dd97b58c95bb7e12258a669e471cb577d) · [UDPipe model releases](https://ufal.mff.cuni.cz/udpipe/1/models)

```sh
git clone https://github.com/ufal/crac2026-corpipe
git -C crac2026-corpipe checkout ec853c3b2d97890f4ef04114b10af6a1ac01d78f
python3 -m venv .venv-corpipe
.venv-corpipe/bin/pip install -r crac2026-corpipe/requirements.txt ufal.udpipe
.venv-corpipe/bin/hf download ufal/corpipe26-onestage-corefud1.4-base-260702 \
  --revision cb62351dd97b58c95bb7e12258a669e471cb577d --local-dir corpipe-base
curl -fLsS https://raw.githubusercontent.com/jwijffels/udpipe.models.ud.2.5/master/inst/udpipe-ud-2.5-191206/english-ewt-ud-2.5-191206.udpipe -o english.udpipe
```

The upstream predictor loads the base encoder configuration from `google/umt5-base` and its shared tokenizer from `google/umt5-xl`. To run offline, cache those small assets first. For a different CorPipe size, check the checkpoint's `options.json` and the predictor's tokenizer selection.

```sh
.venv-corpipe/bin/hf download google/umt5-base config.json
.venv-corpipe/bin/hf download google/umt5-xl \
  --include 'config.json' 'tokenizer*' 'spiece.model' 'special_tokens_map.json'
```

The predictor requests both Google models by name. After caching these assets,
set `HF_HUB_OFFLINE=1` for offline inference. The verified base run used
[UMT5 base configuration](https://huggingface.co/google/umt5-base/tree/0de9394d54f8975e71838d309de1cb496c894ab9)
and [UMT5 XL tokenizer](https://huggingface.co/google/umt5-xl/tree/e02ba7fcf3d6286215043111ee2fc83a9d1f18e2).

## Run

```sh
anno corpipe --python .venv-corpipe/bin/python \
  --predictor crac2026-corpipe/corpipe26_onestage.py \
  --checkpoint corpipe-base --tokenizer-model english.udpipe \
  --text 'Élodie met Sarah. She carried the bag.'
```

The base checkpoint with the English EWT tokenizer returned three clusters on that input. It linked `Sarah` (11..16) and `She` (18..21). JSON exposes source text, half-open Unicode character offsets, sentence IDs, UD word IDs, cluster IDs, and mention heads. A word inside a multiword token reports the shared surface-token range with `alignment="multiword_token"`. An implicit token reports its sentence and decimal UD word ID with `alignment="empty_node"`. Use `--format conllu` to retain the original predicted CorefUD annotations and dependency graph. An overlong sentence that upstream truncates causes a command error.

The model requires the Python runtime and weights regardless of Anno's Cargo feature selection. The local command accepts explicitly selected paths for all three artifacts. The example's English UDPipe SHA-256 is `784bd0fa85e3d831fd02a55290d0acfd05c953159dc38cc33d52e1b28add9957` and the base CorPipe model SHA-256 is `023759452da9ffe53c3de06bf517027422d481973388949a2efed937c9638bf5`.

## Matched English GAP replay

On the same 128 English GAP test excerpts used in Anno's FCoref and Maverick comparison, the released CorPipe 26 base checkpoint recovered 86 of 117 marked links and added nine false ones. Pair precision was 0.905, recall 0.735 and F1 0.811. Both named antecedent decisions were correct in 97 of 128 excerpts. On the same inputs, FCoref recovered 73 links with six false links (F1 0.745), and Maverick recovered 76 with one false link (F1 0.784). These scores use model-proposed mentions. The corpus annotates two candidate antecedents for each pronoun. [CorPipe counts and ordered decisions](corpipe-gap.json) · [Panel selection, baselines, and scorer](../eval/coref-panels.md)

The replay used English EWT segmentation and parsing with no gold mention boundaries. It sent one independent CoNLL-U document per passage to the same upstream predictor in a single process, then checked predicted spans with Anno's GAP scorer. The whole replay took 243 s on a shared CPU, including preprocessing and model startup. To reproduce, use the pinned GAP test file from the [existing panel setup](../eval/coref-panels.md#replay):

```sh
.venv-corpipe/bin/python scripts/qa/corpipe_gap.py /path/to/gap-test.tsv \
  crac2026-corpipe/corpipe26_onestage.py corpipe-base english.udpipe /tmp/corpipe-gap.json
```
