# GLiNEREL-72m

`anno extract --model glinerel-72m` runs the SauerkrautLM-GLiNEREL-72m checkpoint in FP32 on CPU. The Rust backend retains one local Python worker per model instance, including across documents in a batch. The default entity labels are `person`, `organization` and `location`. Supply a different label list with `--extract-types`. Joint extraction accepts a list of directed predicates and returns character offsets for both arguments. [Checkpoint and license](https://huggingface.co/VAGOsolutions/SauerkrautLM-GLiNEREL-72m/tree/34c024bb2ea85326526f6c5d150b9afe8bbe4808) · [Runtime](https://pypi.org/project/glinerel-fast/0.1.1/)

## Install and run

From the checkout, build the CLI and install the CPU Python runtime in its own environment. The example uses Python 3.11 or 3.12 on Linux.

```sh
cargo build --locked -p anno-cli --no-default-features --features extractor-readability
python3 -m venv .venv-glinerel
.venv-glinerel/bin/python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
.venv-glinerel/bin/python -m pip install glinerel-fast==0.1.1
export ANNO_GLINEREL_PYTHON="$PWD/.venv-glinerel/bin/python"

target/debug/anno extract --model glinerel-72m --format json \
  --extract-types 'person,city' --extract-relations \
  --relation-types 'born in,married to' --relation-threshold 0.4 \
  --text '¡Hola! Marie Curie was born in Warsaw. Pierre Curie married Marie Curie.'
```

The example returns a `born in` relation from the first occurrence of Marie Curie at characters `7..18` to Warsaw at `31..37`. It also returns two `married to` edges to distinct mentions of Marie Curie. Each endpoint is matched to its precise source occurrence. When several entity types share exactly that span, the edge uses the highest-scoring detection. When the relation model supplies an argument missing from the entity detections, Anno adds the span as `RELATION_ARGUMENT`. Its confidence comes from the strongest supporting edge and its producer is `glinerel-72m:relation-argument`. In this example Warsaw is such an argument. The relation and typed mention scores remain the values returned by the upstream runtime.

## Weights and offline runs

The worker reads the publisher’s original `model.safetensors`, `tokenizer.json`, `config.json` and `encoder_config/config.json` directly. Its first run resolves the pinned Hugging Face revision. Setting `ANNO_NO_DOWNLOADS=1` restricts resolution to the local cache. To prepare an explicit directory, run the following in the Python environment with the Hugging Face client installed.

```sh
export ANNO_GLINEREL_MODEL_DIR="$PWD/models/glinerel-72m"
"$ANNO_GLINEREL_PYTHON" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    "VAGOsolutions/SauerkrautLM-GLiNEREL-72m",
    revision="34c024bb2ea85326526f6c5d150b9afe8bbe4808",
    local_dir=os.environ["ANNO_GLINEREL_MODEL_DIR"],
    allow_patterns=["config.json", "encoder_config/config.json", "tokenizer.json", "model.safetensors", "LICENSE"],
)
PY
export ANNO_NO_DOWNLOADS=1
```

The original weight file has SHA-256 `2386dcb222870ea0da86653ebcbfbe85457d5f8d7465b3759299050c409180c2`. The runtime uses the LFM Open License v1.0 checkpoint and the Apache-2.0 glinerel-fast package. Anno’s language and length study records results for the same checkpoint on RED-FM and MultiNERD. [Evaluation and all four file hashes](../eval/ner-language-length.md)

At threshold 0.5, the Anno batch selector reproduced the original runtime's exact typed character spans on all 120 RED-FM test documents and all 48 MultiNERD test documents in the language-and-length study. Against the reference annotations, micro F1 was 60.18 on RED-FM (272 true positives, 196 false positives, 164 false negatives) and 82.44 on MultiNERD (54, 9, 14). These panels score person, organization and location mentions. The directed-relation example above is a separate smoke test.
