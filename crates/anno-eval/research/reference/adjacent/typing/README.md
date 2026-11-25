# UFET typing and WordNet hierarchy

A reviewer deciding whether to assign a mention a type needs a model that can rank the appropriate labels. The same text evidence can support several labels, and a known subtype sometimes provides evidence for a broader label. We ran two controls on the crowd-labeled portion of Ultra-Fine Entity Typing. [Dataset authors](https://github.com/uwnlp/open_type) · [Learned box reference](https://aclanthology.org/2021.acl-long.160/)

## Typing

We kept all 1,998 official test mentions. Using the document portion of each annotation ID, we removed test documents from development, then held-out documents from training. This left 1,897 training and 1,931 development mentions. A model could predict any of the 1,608 labels seen during training. The test set contained 10,733 gold labels, including 763 assignments of labels unseen in training. Unseen assignments count toward recall.

| Model | Test mention-macro F1 × 100 | Correct label assignments |
|---|---:|---:|
| Training-label prior | 16.80 | 1,000 |
| Point centroid | 22.48 | 2,151 |
| Box midpoint | 21.76 | 2,089 |
| Box envelope | 22.28 | 2,126 |
| Lexical ridge regression | 27.72 | 2,681 |

The prior emitted two labels per mention. Development selected five for every other model. The three prototypes share the same unit-normalized, 32-dimensional training-fit SVD space over character TF-IDF features for the mention and word TF-IDF for its context. Box bounds are the 10th and 90th per-coordinate percentiles of training instances bearing each label. Box distance sums squared outside distance and a quarter of squared midpoint distance. Development selected the label-prior weight and prediction count. The ridge control uses the full sparse lexical space and regularization 1. This design tests a low-data prototype with gold spans and bidirectional context. It neither reproduces the learned contextual box architecture nor measures real-time coreference. Missing annotations can penalize valid types.

## Direction and joint decoding

We matched training-label strings to all WordNet noun senses bearing those lemmas and collected direct hypernym and instance-hypernym edges. This lexical mapping admits sense ambiguity. We removed self edges and pairs with both directions. Of 4,522 edges, only 1,413 had coannotated endpoints on a single training mention. On those, empirical conditional probability preferred the correct direction on 970 pairs and tied on 123. A marginal-frequency baseline produced the identical result. The reason is algebraic. Both conditionals share a numerator, and the larger denominator belongs to the more frequent label. Reversing an edge by itself cannot establish learned hierarchy knowledge.

The joint control propagates the strongest positive predicted ridge score one step along an external edge, adds a development-selected multiple to the target score, and decodes top-k. We also fitted the same decoder with all edges reversed. This evaluates a joint decision about mention labels under a fixed external taxonomy. Taxonomy edges are given at inference, and implicit hierarchy discovery is outside this measurement.

Forward propagation selected weight 0.1 on development. On test, it recovered 2,678 correct label assignments, three fewer than independent ridge, and scored 27.65 macro F1. Reversed edges selected weight zero and reproduced 27.72. Keep lexical ridge as the control for any learned Subsume or joint type/coreference experiment. There is no evidence here for promoting a mention solely because of lexical hypernym propagation.

## Reproduce

Run with Python 3.12, NumPy 2.3.5, SciPy 1.17.0 and scikit-learn 1.8.0. The script streams the authors' archive to its three crowd splits and verifies their SHA-256 checksums plus the checksum of [WordNet](https://wordnet.princeton.edu/). The cache holds the data locally. Training uses crowd supervision only. No raw examples or checkpoints accompany these results.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python ufet.py /tmp/ufet-cache > results.json
```
