# Document-local entity-memory pilot

The pilot compares bounding boxes with two retained mention vectors at equal
float32 storage on LitBank. Centroid uses the freed bytes to hold more entities.
Midpoint and diagonal spread are additional controls. All arms use exact
candidate scoring and the same LRU policy. Lookup is capped at eight returned
candidates. The pilot encoder hashes a mention and up to 12 preceding tokens.
It uses no right context or gold cluster information.

Use the original LitBank fold 0 development and test IDs. Run the document
excerpts without filling in source words from elsewhere. The CoNLL corpus
flattens relation subtypes, so the pilot cannot isolate apposition, copulas, or
identity revelations. The canonical CoNLL metric is corpus-level. The harness
reports the mean of document-level CoNLL scores for this pilot.

The prespecified target is 35% links per supplied mention. Select each arm's
confidence threshold on development documents using the fixed grid. On test,
interpret wrong-attachment rates as matched coverage only if both arms are
within three percentage points of the target and of each other. For any matched
comparison, require at least 20% fewer wrong attachments per accepted link
against the strongest non-box baseline, with a paired 95% document-bootstrap
interval for the absolute rate reduction above zero and no more than one
point lower mean document CoNLL F1. Count attachments to a mixed predicted
cluster as wrong and keep first mentions in the denominator for link coverage.

Separately, replay with gold history and report Recall@8 for previously seen
entities with a gap of at least 128 tokens between mention ends. Partition
misses into eviction and ranking. Oracle-history results are explanatory.
Count allocated arrays for memory budgets, and keep model weights, encoder
execution and scorer bookkeeping out of the retained-state comparison.

Treat this as a screening pilot. Held-out contextual-model results, subtype
annotations, caller-specific predictions, runtime and RSS measurements, and
full-book transfer are separate stages. Keep BookCoref gold works away from
development and calibration, including Pride and Prejudice in LitBank.

## Reproduce fold 0

Use the [original LitBank evaluation repository](https://github.com/dbamman/lrec2020-coref/tree/e30de53743d36d1ea2c9e7292c69477fa332713c).
Copy the individual CoNLL files named by `data/litbank_tenfold_splits/0/dev.ids`
and `test.ids` into separate directories. The ID lists use a `.tsv` suffix.
Replace it with `.conll` to find each file under `data/original/conll`.

```bash
OPENBLAS_NUM_THREADS=1 python entity_memory_replay.py \
  --dev /path/to/dev --test /path/to/test > receipt.json
python -m unittest discover -s . -p test_entity_memory_replay.py
```

For a contextual follow-up, add `--features /path/to/features`,
`--encoder-id MODEL@REVISION-PREFIX-RECIPE` and the matching `--dimension`.
Each document gets one `<CoNLL-file-stem>.npz` with integer `spans` in
replay order and a `vectors` array. Spans have zero-based, half-open token
coordinates. Vectors should be unit-normalized or zero. The harness checks
coordinates and shape. The encoder producer must independently establish
that vectors use only text visible at mention completion.

## Fold 0 receipt

The pilot used 10 development and 10 test documents, 64 dimensions, a
12,864-byte retained-array budget and the default thresholds grid. The
centroid could hold 45 entities. Each two-vector arm held 24.

| Policy | Development coverage | Test coverage | Wrong attachments / links | False-merge pairs | Oracle long-gap hits / 191 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Centroid | 32.1% | 32.2% | 691 / 913 | 2,850 | 157 |
| Two exemplars | 33.3% | 32.8% | 684 / 929 | 1,083 | 112 |
| Box | 12.9% | 33.6% | 905 / 952 | 59,441 | 126 |
| Box midpoint | 32.5% | 31.9% | 659 / 904 | 1,004 | 121 |
| Diagonal spread | 32.0% | 31.4% | 691 / 888 | 1,734 | 129 |

Box and two exemplars both retained the correct entity in 148 of 191
long-gap queries with oracle history. The box ranked it among eight for 14
more queries. With closed-loop history, the box had a higher wrong-attachment
rate than exemplars by 21.4 percentage points at similar realized test
coverage. The paired document-bootstrap interval for the increase was
10.5 to 28.6 points. Development coverage for the box missed the target
badly on the fixed grid, so cross-split threshold behavior is itself a
failure of this configuration. The box fails the advance criterion.

This result applies to the lexical pilot and the specified update and
scoring rules. A model-backed comparison needs independently generated
causal vectors, an operational coverage check and further held-out data.
The complete pilot command took 18.64 seconds with 67,024 KiB process peak
RSS in the recorded Linux run. Those numbers include all five arms,
development threshold sweeps and scoring. They are not per-arm latency
or resident-state measurements.

## Ten-fold follow-up

`entity_memory_folds.py` uses the same fixed k, gap, target and per-fold
development threshold grid. Pass `--corpus` pointing at the original
`data/original/conll` and `--splits` pointing at `data/litbank_tenfold_splits`.
The driver holds out the LitBank excerpt of Pride and Prejudice from all
development and test folds so it remains available for a BookCoref check.
Each of the other 99 works contributes one held-out prediction.

The lexical run covered 28,733 gold mentions. Boxes had 9,415 wrong
attachments in 10,041 links (93.8%) versus 6,982 in 9,590 (72.8%) for two
exemplars, at 34.9% and 33.4% realized coverage. The paired interval for the
increase was 18.4 to 23.3 percentage points. Pooled coverage hides
calibration drift. The box met the 32–38% development band in two of ten
folds and its test coverage ranged from 10.3% to 80.0%. Treat the pooled
wrong-attachment comparison as descriptive evidence at similar aggregate
coverage. Oracle-history boxes retrieved
1,156 of 1,834 long-gap entities, compared with 1,048 for two exemplars and
1,367 for centroid. The centroid could hold more entities at the same budget.

### Contextual features

The follow-up uses
[sentence-transformers/all-MiniLM-L6-v2 at revision 1110a243](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41),
`onnx/model_quint8_avx2.onnx`, and that revision's `tokenizer.json`.
Use ONNX Runtime 1.24.4 and tokenizers 0.22.2. Each input contains up to
48 preceding corpus tokens and the mention, ending at its last token.
Mean-pool mention wordpieces and normalize the result. One mention is run
per ONNX call. In a direct check, the quantized model's batched output for
an identical early prefix changed when a later batch row changed. The
single-mention path returned bit-identical early vectors in that fixture.

```bash
python encode_entity_memory.py --corpus /path/to/conll \
  --model /path/to/model_quint8_avx2.onnx --tokenizer /path/to/tokenizer.json \
  --output /path/to/features --threads 4
OPENBLAS_NUM_THREADS=1 python entity_memory_folds.py \
  --corpus /path/to/conll --splits /path/to/litbank_tenfold_splits \
  --features /path/to/features --dimension 384 \
  --encoder-id minilm-l6-v2@1110a243-uint8-avx2-left48-single \
  --jobs 4 > minilm-receipt.json
```

The contextual run holds 24 entities in each two-vector arm at 74,304
retained-array bytes. Its operating point and advancement bar are the same
as the lexical pilot. This encoder is frozen and the memory policies are
untrained.

### Contextual ten-fold result

The MiniLM run covered the same 99 works and 28,733 held-out gold mentions.
Model SHA-256 was
`b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21`.
Tokenizer SHA-256 was
`be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037`.

| Policy | Test coverage | Wrong attachments / links | Wrong rate | False-merge pairs | Oracle long-gap hits / 1,834 | Mean document CoNLL F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Centroid | 30.4% | 3,950 / 8,728 | 45.3% | 23,456 | 1,610 | 46.8 |
| Two exemplars | 37.1% | 4,415 / 10,658 | 41.4% | 15,240 | 1,242 | 48.9 |
| Box | 35.4% | 8,323 / 10,159 | 81.9% | 623,870 | 1,289 | 41.9 |
| Box midpoint | 31.4% | 3,867 / 9,011 | 42.9% | 20,973 | 1,310 | 47.2 |
| Diagonal spread | 39.4% | 4,343 / 11,329 | 38.3% | 15,145 | 1,307 | 50.3 |

Boxes retrieved 47 more long-gap antecedent entities than exemplars with
gold history, but 21 fewer than the midpoint ablation. In closed loop the
box wrong-attachment rate exceeded exemplars by 40.5 percentage points.
The paired interval for the pooled difference was 34.8 to 45.4 points.
The box and exemplars have similar pooled test coverage, but they failed
to maintain a common operating point across folds. Box test coverage
ranged from 13.0% to 56.1%. The box met the development coverage band in
four folds and exemplars met it in none with the fixed grid.

**Decision.** This box update and scoring rule fails the advance gate
with both encoders. The result tests an untrained bounding-box summary.
A learned box scorer could behave differently. BookCoref replication,
the Anno predicted-mention trial, runtime comparisons at caller limits,
and Anno integration were conditional on clearing the gate.
