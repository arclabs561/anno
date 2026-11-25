# Frozen different-text review queue on WikiCoref

This transfer tests whether the box reranking effect seen among different-text
mentions in LitBank and PreCo generalizes to Wikipedia articles. The official
MMAX2 archive contains separate gold projections. We used all 30 OntoNotes
projection files and the corresponding XML word IDs, checked sequential word
IDs, span bounds and unique coordinates, and included every projected mention.
The original release also contains broader annotations including attributive
and copular mentions. [WikiCoref release](https://rali.iro.umontreal.ca/rali/?q=en/wikicoref)
[Corpus paper](https://aclanthology.org/L16-1021/)

## Results

Across 59,857 XML words and 6,991 projected gold mentions, 4,930 proposals
paired different surface forms. The common review budget selected 974 pairs.

| Queue ordering | Wrong displayed pairs | Wrong or impure proposed clusters |
| --- | ---: | ---: |
| Exemplar | 490 | 600 |
| Exemplar and midpoint | 489 | 598 |
| Exemplar and box | 497 | 649 |

Box scoring produced seven additional wrong displayed pairs and 49 additional
wrong or mixed clusters. The 95% paired article interval for the reduction in
displayed-pair errors was −2.29 to +0.90 percentage points. For the cluster
outcome, errors increased by 5.03 points (95% interval +3.60 to +6.58).
The favorable different-text result from LitBank and PreCo failed to replicate
on this untouched article corpus. Prefer the cheaper exemplar ranking for general
use. The query-type breakdown also shows a change in what reviewers see.
Adding box scores displaced 35 nominal-query reviews and admitted 19 extra
named-query reviews and 16 extra pronominal-query reviews.

## Fixed transfer protocol

The model, tokenizer, causal 48-token left context, single-input inference and
24-slot memory per summary come from the prior replay. All arms use the
PreCo development-calibrated exemplar history threshold of −0.159. No article
was used to fit a parameter. Each arm receives the same predicted target with
its latest occurrence as the displayed evidence. After dropping exact
case-folded text matches, each arm selects the top fifth of the remaining
proposals within each article. The box and midpoint controls each add their
within-document confidence rank to the exemplar rank with equal weight. Article
history updates remain causal. The final queue ordering uses all the proposals
in the article. The bootstrap samples articles jointly across arms, 2,000
resamples at seed 7. Gold spans and non-singleton annotation make this a test
of ranking over supplied mentions. It does not establish end-to-end detection
or time saved by human review.

## Reproduce

Download the author archive, the pinned MiniLM AVX2 model, and its tokenizer
to local scratch. The harness verifies SHA-256 checksums of all three. From `crates/anno-eval/research/reference`, set
`PYTHONPATH` to the reference evaluation directory to use the original
queue and encoder implementations. Use ONNX Runtime 1.24.4 and tokenizers
0.22.2. The output includes the counts for each article, and the numbers
selected and correct by the gold query mention type, in that order.
[Model and tokenizer](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41)
[Corpus archive](https://rali.iro.umontreal.ca/rali/sites/default/files/resources/wikicoref/WikiCoref.tar.gz)

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 python adjacent/wikicoref/run.py \
  --archive /tmp/WikiCoref.tar.gz --model /tmp/model_quint8_avx2.onnx \
  --tokenizer /tmp/tokenizer.json
```
