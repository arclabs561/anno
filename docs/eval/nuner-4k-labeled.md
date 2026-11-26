# NuNER Zero-4k FP32 and QUInt8 on labeled text

We scored the same 38 inputs with two ONNX graphs built from NuMind's MIT-licensed `numind/NuNER_Zero-4k` checkpoint at revision `7a7cd8d65af2572c297054dbaf8f25c0d46da55d`. The FP32 graph is the verified Longformer export. The candidate uses ONNX Runtime dynamic QUInt8 weight quantization without calibration or retraining. Both graphs have the exact SHA-256 previously checked on the synthetic FP32 parity cases. The candidate remains a QA artifact. [Model files](https://huggingface.co/numind/NuNER_Zero-4k/tree/7a7cd8d65af2572c297054dbaf8f25c0d46da55d) · [Parity receipt](nuner-4k-quantization.json) · [Labeled receipt](nuner-4k-labeled.json)

## What the comparison measures

The 24 English sentences were selected by the lowest SHA-256 of `nuner-multinerd-v1/en/<zero-based-test-row>`. The next 12 disjoint non-ASCII English sentences in that order form a separate Unicode panel. Two longer inputs concatenate contiguous whole test sentences using newlines. The English pack starts at test row 2000 and the Russian pack at row 1500. Their lengths are 1,772 and 4,033 tokenizer IDs including the prompt. These are constructed stress inputs. The dataset distribution and original document boundaries cannot be inferred from the packs. The checkpoint card lists English. The Russian pack is an out-of-language stress case, so its score cannot isolate the effect of length. No tuning or threshold selection used the test annotations.

Every input uses the same three prompts in order, `person`, `organization`, `location`. We reconstructed tokens with single ASCII spaces and converted the six associated BIO tags to half-open Unicode character spans. Other published classes are outside this three-label task. The two sessions read identical saved tensors. The replay uses the GLiNER processor, the highest-scoring label per word, inclusive sigmoid threshold, adjacent same-label word merging and Anno's trailing punctuation/possessive cleanup. It scores exact span **and** label, micro-averaged within each group. Both model arms retain every selected input, including empty predictions. This is a graph-level comparison through Python ONNX Runtime. Native application startup, chunk scheduling and text ingestion are outside the timing.

The dataset comes from Simone Tedeschi and Roberto Navigli, *MultiNERD: A Multilingual, Multi-Genre and Fine-Grained Dataset for Named Entity Recognition (and Disambiguation)*, Findings of NAACL 2022. The files used here are the published English and Russian test splits at revision `2814b78e7af4b5a1f1886fe7ad49632de4d9dd25`. The dataset card lists CC BY-NC-SA 4.0. The receipt contains selection IDs, text and file digests, and derived offsets. [Dataset](https://huggingface.co/datasets/Babelscape/multinerd/tree/2814b78e7af4b5a1f1886fe7ad49632de4d9dd25) · [Paper](https://aclanthology.org/2022.findings-naacl.60/)

## Results

At the fixed 0.5 threshold, the counts below use one correct prediction per exact span and label. P/R/F1 are percentages.

| Input group | Cases | Gold | FP32 TP/FP/FN | FP32 P/R/F1 | QUInt8 TP/FP/FN | QUInt8 P/R/F1 |
|---|---:|---:|---:|---:|---:|---:|
| English sentences | 24 | 25 | 18/28/7 | 39.1/72.0/50.7 | 18/28/7 | 39.1/72.0/50.7 |
| English, non-ASCII | 12 | 14 | 10/9/4 | 52.6/71.4/60.6 | 10/7/4 | 58.8/71.4/64.5 |
| English, 1,772 IDs | 1 | 64 | 51/12/13 | 81.0/79.7/80.3 | 52/10/12 | 83.9/81.2/82.5 |
| Russian, 4,033 IDs | 1 | 29 | 1/0/28 | 100.0/3.4/6.7 | 1/0/28 | 100.0/3.4/6.7 |

The F1 sweep below pairs the two graphs at the same prespecified threshold. A threshold change can alter both which words qualify and where an adjacent run of words ends.

| Input group | 0.3 FP32 / QUInt8 | 0.5 FP32 / QUInt8 | 0.7 FP32 / QUInt8 |
|---|---:|---:|---:|
| English sentences | 40.0 / 40.5 | 50.7 / 50.7 | 56.2 / 63.5 |
| English, non-ASCII | 55.6 / 51.4 | 60.6 / 64.5 | 66.7 / 71.0 |
| English, 1,772 IDs | 69.6 / 75.4 | 80.3 / 82.5 | 83.2 / 83.2 |
| Russian, 4,033 IDs | 22.2 / 14.7 | 6.7 / 6.7 | 0.0 / 0.0 |

At threshold 0.5, 27 of 8,412 word–label decisions changed. The largest absolute change in sigmoid probability was 0.460. Exact predicted span sets differed on 10 of the 38 inputs. The released FP32 export remains the reference for matching upstream decisions.

Paired sentence bootstrap intervals (10,000 resamples, 95% percentile, QUInt8 minus FP32 in F1 points) span -2.7 to +3.0 on the ordinary English panel and +0.0 to +10.8 on the non-ASCII English panel. Each packed group has only one constructed input. These panels cannot establish a population-wide accuracy ranking.

## Where the remaining errors come from

At threshold 0.5, both graphs missed the **same 50 labeled mentions**. FP32 alone missed two others and QUInt8 alone missed one. Both graphs also produced 32 identical extra span–label triples. Another 17 extras were unique to FP32 and 13 to QUInt8. Changing one boundary can remove one arm's extra and add a different extra to the other arm.

| Disagreement against the three reference classes | FP32 extras | QUInt8 extras | FP32 misses | QUInt8 misses |
|---|---:|---:|---:|---:|
| Exact boundaries, different type | 2 | 5 | 2 | 5 |
| Overlap, same type, different boundaries | 18 | 15 | 18 | 15 |
| Overlap, different boundaries and type | 4 | 2 | 2 | 1 |
| No overlap | 25 | 23 | 30 | 30 |
| Total | 49 | 45 | 52 | 51 |

Each extra is classified against all reference spans in its input. Each miss is classified independently against all predictions. The rows above give the order of precedence when multiple spans overlap. This counts split predictions individually and keeps the published precision and recall denominators. Of the 30 no-overlap misses in each arm, 28 came from the out-of-language Russian pack.

Inspection of the original English BIO rows explains several distinct cases. Row IDs below are zero-based.

- **A boundary convention affected both arms.** At row 8179 both returned “Maestro Ryan Cayabyab” for the reference person “Ryan Cayabyab”. Exact scoring charges one extra and one miss even though both graphs located the person.
- **Several extras are generic mentions.** At row 1103 both graphs labeled “cyclists” and “men” as people. Row 4443 has the FP32-only “He”. These mentions can refer to people, but they are outside this named-entity task.
- **An exact surface can still have the wrong type.** At row 2006 FP32 correctly labeled “Tropicana” as an organization in a sentence about competing juice brands. QUInt8 kept the span and labeled it as a location. At row 736 both called “Valencia” a location. The reference labels it an organization in a football match, where a team and its home city share a name.
- **An overlap can hide a split.** Both arms split the location “San Pablo Bay National Wildlife Refuge” at row 12680. Each produced a shorter location and a separate organization. A one-to-one alignment alone would hide the second extra.
- **Reference scope limits interpretation.** At row 2050 the reference marks “Lydbrook”, while both arms predict “Lydbrook valley” as a location. The ledger establishes a boundary disagreement. Deciding which span is preferable for a geographic index requires an annotation rule for physical features. The benchmark contains no adjudication for that downstream purpose. We retained its labels and scores.

Reproduce every cell and its per-group, per-arm breakdown with `python scripts/qa/nuner_error_analysis.py docs/eval/nuner-4k-labeled.json`. The script checks its FP and FN totals against all eight group/model ledgers. [Paired ledger](nuner-4k-labeled.json) · [MultiNERD test files](https://huggingface.co/datasets/Babelscape/multinerd/tree/2814b78e7af4b5a1f1886fe7ad49632de4d9dd25/test)

## Resource measurements

| Graph | ONNX file (MiB) | Session load (s) | Process peak RSS (MiB) | Median English sentence (s) | English pack (s) | Russian pack (s) |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 1448.7 | 11.27 | 4000 | 5.93 | 25.12 | 42.21 |
| U8 | 387.9 | 7.23 | 3084 | 5.13 | 17.98 | 34.64 |

Measured on one AMD EPYC 9V74 Linux host, ONNX Runtime 1.32.0 CPU provider, FP32 first. Each packed latency is a single observation. Session loading followed graph hashing, which warms the filesystem cache. Peak RSS includes ORT session creation and its working memory across the fixed input sequence. Runtime and memory measurements depend on architecture, allocator, thread count and length.

## Replay

The exporter verifies the source checkpoint in FP32. The separate QA utility produces the candidate. Each graph runs in a fresh process with two intra-op threads, one inter-op thread, and one excluded warm-up on the first sentence. Use the same available CPU affinity for both `run` commands when measuring timings. Input preparation and postprocessing are outside per-call elapsed time. Peak RSS is for the whole Python inference process.

```sh
uv run scripts/export_nuner_to_onnx.py --model numind/NuNER_Zero-4k --output /tmp/nuner-4k
uv run scripts/qa/check_nuner_4k_quantization.py /tmp/nuner-4k/model.onnx /tmp/nuner-u8.onnx --quantize
uv run scripts/qa/nuner_labeled.py --work /tmp/nuner-panel prepare /tmp/nuner-4k
uv run scripts/qa/nuner_labeled.py --work /tmp/nuner-panel run /tmp/nuner-4k/model.onnx fp32
uv run scripts/qa/nuner_labeled.py --work /tmp/nuner-panel run /tmp/nuner-u8.onnx u8
uv run scripts/qa/nuner_labeled.py --work /tmp/nuner-panel report /tmp/labeled.json
```
