# GLiDRE v2 document relations

`anno glidre` scores directed relations between entities in a text document. Supply the original text, the mention occurrences that belong to each entity, and the predicates you want to ask about. Entity IDs and predicates can use your own vocabulary.

## Link entities in your text

Save this invoice note as `note.json`. Its two occurrences of “Ana” share one entity ID. Offsets count Unicode code points from zero, and the end offset is exclusive.

```json
{
  "id": "invoice-47",
  "text": "Zoë called Ana.\nAna sent Café Bleu the invoice.",
  "entities": [
    {"id": "zoe", "mentions": [{"start": 0, "end": 3}]},
    {"id": "ana", "mentions": [{"start": 11, "end": 14}, {"start": 16, "end": 19}]},
    {"id": "cafe", "type": "BUSINESS", "mentions": [{"start": 25, "end": 34}]}
  ]
}
```

Save `{"called":"CALLED","billed":"SENT_INVOICE_TO"}` as `predicates.json`. The keys are your output predicates. The values are distinct model prompts. The authors use uppercase names for zero-shot inference.

```bash
anno glidre --upstream "$work/glidre" --model-dir "$work/model" \
  --python "$work/venv/bin/python" --input note.json --schema predicates.json \
  --threshold 0.3 --device cpu
```

The JSON response contains `id`, the unchanged `text`, the supplied `entities` with exact mention text, and `relations`. Each relation contains `head` and `tail` entity IDs, your `predicate` key and a numeric `score`. Supply an array to process a batch and receive an array of results. Use string or integer IDs and keep them unique within their scope. Mention `text` and entity `type` are optional. When you supply mention text, the adapter checks it against the exact source slice. Each span must begin and end at token boundaries recognized by the checkpoint. Errors identify the document and offending span. The multilingual checkpoint has a 512-word limit. The adapter reports any document that would be clipped.

## Install the multilingual checkpoint

The following commands set up the published multilingual v2 checkpoint for CPU inference. Its Apache-2.0 weights occupy 2.45 GB decimal. Choose a work directory with additional space for Python and PyTorch.

```bash
work="$PWD/glidre-work"
mkdir -p "$work"
git clone https://github.com/robinarmingaud/glidre "$work/glidre"
git -C "$work/glidre" checkout --detach 1a830bb2154fc503c968081e4d583d320fa5c309
python3 -m venv "$work/venv"
source "$work/venv/bin/activate"
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install "$work/glidre" huggingface-hub==0.36.2 sentencepiece protobuf
export GLIDRE_WORK="$work" HF_HOME="$work/hf"
python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

snapshot_download("cea-list-ia/glidre_multi_v2",
                  revision="1881f8126342d47ac4b9b1c7049d4af99424d8f3",
                  local_dir=Path(os.environ["GLIDRE_WORK"]) / "model")
sources = [
    ("microsoft/mdeberta-v3-base", "a0484667b22365f84929a935b5e50a51f71f159d",
     ["config.json"]),
    ("Alibaba-NLP/gte-multilingual-base", "9bbca17d9273fd0d03d5725c7a4b0f6b45142062",
     ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]),
    ("Alibaba-NLP/new-impl", "40ced75c3017eb27626c9d4ea981bde21a2662f4",
     ["configuration.py", "modeling.py"]),
]
for repo, revision, files in sources:
    snapshot = Path(snapshot_download(repo, revision=revision, allow_patterns=files))
    # The authors' loader requests some dependencies by branch. Pin those names
    # in this dedicated cache so inference resolves the same files offline.
    ref = snapshot.parents[1] / "refs" / "main"
    ref.parent.mkdir(exist_ok=True)
    ref.write_text(revision)
PY
export HF_HUB_OFFLINE=1
```

The upstream package pins Transformers 4.51.3, GLiNER 0.2.13 and datasets 4.5.0. Its multilingual predicate encoder imports the two Python files from the pinned Alibaba model repository. [GLiDRE source](https://github.com/robinarmingaud/glidre/tree/1a830bb2154fc503c968081e4d583d320fa5c309) · [Multilingual weights](https://huggingface.co/cea-list-ia/glidre_multi_v2/tree/1881f8126342d47ac4b9b1c7049d4af99424d8f3) · [Predicate encoder implementation](https://huggingface.co/Alibaba-NLP/new-impl/tree/40ced75c3017eb27626c9d4ea981bde21a2662f4)

## Evaluate on DocRED-format documents

The adapter also accepts the DocRED interchange format. Supply `title`, sentence tokens in `sents`, and the mention clusters in `vertexSet`. Use gold vertices to evaluate relation classification or use predicted vertices to evaluate a complete extraction pipeline. Input relation labels and evidence are ignored. For benchmark comparisons, provide the full predicate inventory separately from the test annotations.

```bash
anno glidre --upstream "$work/glidre" --model-dir "$work/model" \
  --python "$work/venv/bin/python" --input documents.json --schema predicates.json \
  --threshold 0.3 --device cpu > predictions.json
python3 crates/anno-eval/examples/docred/score.py gold.json predictions.json --endpoints supplied
```

DocRED's `pos` uses sentence-local, zero-based token indices with an exclusive end. The adapter checks sentence bounds and converts each mention to the document-global inclusive token endpoints used by GLiDRE. The output preserves the original sentence tokens and vertices. Each `labels` entry contains head vertex index `h`, tail vertex index `t`, predicate ID `r`, and model `score`. The decoder permits multiple predicates on an ordered pair. Two vertices sharing a mention retain their input indices.

[GLiDRE paper](https://arxiv.org/abs/2508.00757) · [Anno document relation scorer](../../crates/anno-eval/examples/docred/score.py)

## Verified CPU runs

For the invoice note above, the raw-text path and the equivalent supplied token spans yielded identical scores on all nine directed predictions at threshold 0.1. The model scored Zoë calling Ana at 0.7185 and Ana billing Café Bleu at 0.5470. It also incorrectly predicted that Ana called herself (0.3906).

On a two-sentence concert example with three English predicate prompts, the multilingual v2 model emitted three directed relations at threshold 0.3. All three labels, endpoints and floating-point scores matched the authors' inference code exactly. The supplied-vertex run on six hash-selected Re-DocRED development documents used all 96 predicate prompts at the same threshold. Of 28 predicted relations, 10 matched the 227 annotated relations, giving 35.71% precision, 4.41% recall and 7.84% micro F1. The three-prompt example checks runtime parity. The six-document result measures this checkpoint on a small labeled sample with a much broader vocabulary.
