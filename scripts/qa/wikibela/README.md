# Full WikiBELA candidate lookup

The smoke test runs `anno mhel retrieve --documents` on a short English source containing an emoji and the name Paris. It verifies five scored Wikidata candidates and preserves the caller’s document ID, entity ID, surface and Unicode character range. The receipt includes the actual candidates so that their order and IDs can be inspected. Its assertion covers candidate retrieval and source alignment; a specific top-ranked Wikidata ID requires a labeled evaluation.

| Publisher asset | Bytes | Runtime use |
|---|---:|---|
| `faiss.index` | 19,765,027,245 | Upstream loads 16,470,856 float32 vectors of dimension 300 for exhaustive inner-product search |
| `knowledge_base.sqlite` | 3,159,400,448 | Disk-backed entity and localized description lookup |
| `model_wiki.ckpt` | 2,243,434,498 | BELA query encoder and task weights |
| `index.txt` | 158,216,234 | Entity catalog loaded into a Python dictionary |
| **WikiBELA total** | **25,326,078,425** | Four independently verified SHA-256 hashes |

The upstream encoder also opens XLM-RoBERTa-large at startup. Its pinned base weights add 2,244,817,354 bytes plus configuration and tokenizer files. The workflow checks for roughly 29.7 GB of available disk after dependency installation and adds working swap if needed. The 28 GiB combined available RAM and swap budget is a provisioning allowance. The index array alone occupies 18.4 GiB. The actual peak depends on allocations in Python, PyTorch and FAISS. [WikiBELA snapshot](https://huggingface.co/sntcristian/WikiBELA/tree/82b71b08c596b821e4a44c29a23d1c5fecd8165c) · [Retriever implementation](https://github.com/sntcristian/MHEL-LLAMO/blob/b262dfb77285ea00993af8d453387fd7bf81228b/src/retriever.py) · [XLM-RoBERTa-large snapshot](https://huggingface.co/FacebookAI/xlm-roberta-large/tree/c23d21b0620b635a76227c604d44e43a9f0ee389)

The hosted workflow runs on requests changing its own paths and supports manual dispatch. A disposable Ubuntu runner removes unused preinstalled SDKs, builds the local CLI, installs the authors' Python 3.10 environment with CPU PyTorch and FAISS, checks capacity, verifies each asset, then runs the pinned models offline. Only the small JSON receipt is uploaded. [Workflow](../../../.github/workflows/wikibela-retrieval.yml)

For a machine with the Python environment and pinned MHEL source already installed, place the checkout at `$work/mhel`, activate the environment and run the following. The capacity probe prints the additional GiB of swap needed for the provisioning allowance. Use an absolute path for a locally built CLI.

~~~sh
python scripts/qa/wikibela/smoke.py budget "$work"
HF_HUB_OFFLINE=1 python scripts/qa/wikibela/smoke.py lookup "$work" --anno "$anno"
~~~
