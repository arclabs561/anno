# Scientific entity types, relations and coreference

Scientific abstracts often refer to a method by its full name and later as
“our approach.” A scientific knowledge graph needs the right endpoints and
types before a relation can be used downstream. We compared independent text
classifiers, a one-way pipeline that conditions predicted edges on predicted
entity types, and joint decoding that also lets edges revise endpoint types.

| On 100 test abstracts | Independent | One-way pipeline | Joint feedback |
|---|---:|---:|---:|
| Mention type accuracy | 73.35% | 73.35% | 72.23% |
| Directed relation F1 | 22.52% | 29.99% | 29.90% |
| Relation F1 requiring both endpoint types | 11.84% | 20.24% | 19.44% |
| Coreferent mention-pair F1 | 34.65% | 37.69% | 38.17% |
| Coreferent pair F1 requiring both endpoint types | 23.47% | 34.16% | 32.55% |

The official test contains 1,685 gold mention boundaries, 974 relations and
534 positive coreference pairs between typed mentions. Joint feedback
recovered three more correct untyped coreference pairs than the pipeline,
while losing 19 correct mention types, five fully correct relations and six
fully correct coreferent pairs. For an annotator proposing typed scientific
facts, the simpler pipeline is the starting baseline.

## Protocol

The 350 training abstracts fit logistic classifiers on sparse mention and
pair lexical features and smoothed relation/coreference compatibility tables
for ordered pairs of endpoint types. The local arms share these lexical
models. All ordered within-sentence pairs are relation candidates. All
unordered abstract-wide pairs of typed mentions are coreference candidates.
Relations must match their serialized argument order, including symmetric
labels. Four duplicate training relation rows were collapsed.

The pipeline uses a log likelihood ratio for observed endpoint types given
the edge class relative to the no-edge class. Joint inference additionally
updates each type to maximize the same objective given current edge decisions,
alternating for at most five rounds. The official 50-document development set
selects the compatibility weight from 0, 0.25, 0.5 and 1 using the equally
weighted sum of typed relation and typed coreference F1. Both arms selected
1. Type and edge labels are hidden during inference. All test features use
gold span boundaries and full abstracts. The pair classifier imposes no
cluster transitivity. This is an inference ablation using linear models,
not a reproduction of the published neural system or an end-to-end span
detector. Results include 2,000 paired abstract bootstrap replicates for
joint minus one-way typed edge F1.

## Reproduce

Use NumPy 2.3.5 and scikit-learn 1.8.0. The original archive is about 695 MB
and includes unused ELMo features. Keep downloaded data in a temporary directory and extract only the JSON files after the checksum passes.

```sh
set -e
corpus_dir=$(mktemp -d)
curl -fL https://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz -o "$corpus_dir/sciERC_processed.tar.gz"
(cd "$corpus_dir" && echo 'bce752fb7ebe4acf570937d76ffb27c239cfc907e477a69075dc7082d0e72e9b  sciERC_processed.tar.gz' | sha256sum -c -)
tar -xzf "$corpus_dir/sciERC_processed.tar.gz" -C "$corpus_dir" processed_data/json
OPENBLAS_NUM_THREADS=1 python run.py "$corpus_dir/processed_data/json"
```

[Original dataset](https://nlp.cs.washington.edu/sciIE/) · [SciERC paper](https://aclanthology.org/D18-1360/)
