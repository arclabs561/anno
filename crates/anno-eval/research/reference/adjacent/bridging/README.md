# Typed bridging and joint antecedent selection

We used the [GUMBridge v1 release](https://github.com/lauren-lizzy-levine/gumbridge/tree/bfe271bcb554aaa924c697779f2735dc16151554). Each test query is an annotated bridging anaphor. Its candidates are the 32 most recently completed gold mentions before the start of the anaphor. This reads the published TSV links from earlier mention to later anaphor, then reverses the lookup to retrieve an antecedent. Gold mention boundaries and coarse entity types are inputs. The model sees neither subtype annotations, information status, salience, gold links, nor identity clusters at prediction time. Strict completion excludes overlapping mentions. Correctness requires an annotated antecedent mention. A different mention of the same entity can thus be marked wrong. Untyped split antecedents are a separate task and are excluded.

We fitted count-based type-pair link rates, recency rates and subtype frequencies on 4,025 anaphors in 211 training documents. The ordered model has separate counts for (anaphor type, antecedent type), while the symmetric model pools reversed pairs. Both see the same candidate lists. The pipeline maximizes a smoothed type-pair link rate times the recency rate before predicting the most probable subtype. The joint decoder maximizes the product of this retrieval score and that subtype's estimated probability. Any annotated subtype is accepted for a correct antecedent. Multiple subtype labels contribute one training count each. These scores are simple frequency estimates. This count experiment uses no learned boxes. A separate fitted geometry comparison follows below.

| Split | Queries | Reachable | Recency hits | Ordered pipeline hits | Symmetric pipeline hits | Ordered pipeline correct target + subtype | Ordered joint correct target + subtype |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dev | 732 | 663 | 118 | 183 | 168 | 107 | 105 |
| Test | 558 | 465 | 99 | 161 | 162 | 66 | 69 |
| Out-of-domain test2 | 379 | 300 | 52 | 64 | 63 | 29 | 28 |

Joint decoding on test moved correct antecedents from 161 to 147 while gaining three correct antecedent–subtype pairs. Across the eight out-of-domain genres, it moved correct antecedents from 64 to 60 and correct pairs from 29 to 28. Prioritizing a likely subtype can therefore drop a useful antecedent. The symmetric control essentially ties ordered retrieval on both tests. The next question for Subsume is whether it adds relation-specific information beyond coarse types. These numbers offer no evidence that a generic direction score improves the ranking. Annotated earlier candidates cover only 465/558 and 300/379, setting a ceiling on both decoders.

## Learned geometry on the same candidates

We trained 4-dimensional type embeddings with the same 86 parameters for each arm. The point model uses two vectors per coarse type (query and antecedent roles) and negative squared distance. The box models learn one center and positive half-width per coarse type. For smoothed side lengths we use 0.1 × softplus(length / 0.1), including for the intersection. The directional score is log intersection volume minus log anaphor volume. The symmetric control subtracts the mean of the two log volumes. Each model learns six additional recency-bucket intercepts. We minimize binary cross-entropy on the same 122,422 training candidate pairs plus a small L2 penalty, using analytic derivatives checked against a directional finite difference. Model selection and tuning used no test labels. These runs used NumPy 2.3.5 and SciPy 1.17.0 L-BFGS-B on CPU. This implements the scoring operation directly in Python. It does not benchmark Rust crate performance.

| Top-1 antecedent, range over seeds 7, 11, 23 | GUM test, 558 queries | GENTLE test2, 379 queries |
|---|---:|---:|
| Role-specific points | 159 | 64 |
| Symmetric boxes | 149–150 | 59–60 |
| Directional boxes | 155–156 | 61–65 |
| Directional model with inference roles swapped | 143–145 | 55–56 |
| Ordered type counts plus recency | 161 | 64 |

Correct orientation helps within the box model. It still misses more in-domain antecedents than the point and count models, and reaches only 161–162 correct out-of-domain top-five lists versus 169 for points. For a fixed anaphor the denominator in its conditional overlap score is constant across candidates. Its effect enters through training shared representations, while swapping the roles at inference also introduces candidate-volume normalization. This test learns only coarse type geometry; lexical context, entity-specific boxes and learned span encoders are outside its scope. The count receipt also reports subtype breakdowns. For example, the ordered pipeline finds seven of 18 subset antecedents on test, predicts the correct subset label for zero, and finds 13 of 46 member antecedents with zero correct member labels. Predicting the semantic relation from two coarse types remains the bottleneck on these slices.

## DiscoTrack excerpt audit

The [pinned DiscoTrack release](https://github.com/Lanni-ni/DiscoTrack/tree/3266566563173444544cc12f89224384b312f289) supplies extractive bridge antecedent questions in English and Czech. For a document-local source citation, we counted whether the published answer text occurs in the visible prefix before the query marker, using exact contiguous case-folded word and punctuation tokens. This is a string-availability diagnostic. A matching string can denote the wrong entity, and a paraphrase or differently inflected answer can be valid.

| Language | Released rows | Unique document/text/answer triples | Answer in previous 32 tokens | Answer anywhere in prefix |
|---|---:|---:|---:|---:|
| English | 195 | 193 | 136 | 186 |
| Czech | 500 | 500 | 229 | 292 |

All English document IDs were absent from GUMBridge training. English DiscoTrack draws from GUMBridge and does not supply an independent replication of its test scores. The 208 Czech answers without an exact prior string in the supplied excerpt make this a poor benchmark for a strict prefix-only exact-answer extractor unless those cases are reported separately. The analysis does not assume what caused each mismatch.

## Reproduction

Use Python 3.10+ and a checkout of each pinned revision. The two count-based scripts require only the Python standard library. The geometry run requires NumPy and SciPy.

```sh
python bridge_joint.py "$GUMBRIDGE/data" > results.json
OPENBLAS_NUM_THREADS=1 python geometry.py "$GUMBRIDGE/data" > geometry_results.json
python discotrack_audit.py \
  "$DISCOTRACK/bridge/antecedent/bridging_antecedent_benchmark.json" \
  "$DISCOTRACK/bridge_c/bridging_ante_czech_benchmark.json" \
  "$GUMBRIDGE/data/train" > discotrack_results.json
```
