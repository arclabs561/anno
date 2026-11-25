# Relation extraction on held-out CrossRE AI sentences

This 32-sentence pilot has a follow-up covering all 431 AI and 416 literature test sentences. [Full-domain results](crossre-full-prompt-comparison.md)

The production-style condition gives the extractor one vocabulary for the entire run. We took the sorted union of annotated entity types and relation types over the 100 CrossRE AI training sentences, yielding 14 entity labels and 15 relation labels. The diagnostic condition supplies only the labels present in each test sentence's gold relations. Both conditions ask the backend to find endpoints in the text. The fixed inventory also reaches sentences with no reference relations. [Fixed schema](crossre-ai-prompt-schema.json) · [CrossRE dataset and split](https://github.com/mainlp/CrossRE/tree/a58885fa760559d2dc9e176e730fe731f020d6f3/crossre_data)

The evaluator selected 32 of 431 test sentences with seed 42. The train and test splits share zero identical token sequences. Strict micro-averaged scoring requires a case-insensitive relation-label match and exact directed head and tail character spans in the same sentence. It ignores the endpoint type strings. Every annotated relation counts toward recall, including labels outside the fixed vocabulary. Gold endpoints were disabled in all runs.

| Effective producer | Prompt | Exact / predicted / gold | Precision | Recall | Strict F1 | Wall (s) | Peak RSS (GiB) |
|---|---|---:|---:|---:|---:|---:|---:|
| GLiNER-Relex | Train inventory | 21 / 61 / 83 | 34.4% | 25.3% | 29.2% | 108.57 | 2.78 |
| GLiNER-Relex | Document gold | 49 / 75 / 83 | 65.3% | 59.0% | 62.0% | 66.79 | 2.83 |
| TPLinker heuristic | Train inventory | 0 / 0 / 83 | — | 0.0% | 0.0% | 0.40 | 0.05 |
| TPLinker heuristic | Document gold | 0 / 0 / 83 | — | 0.0% | 0.0% | 0.28 | 0.05 |

The document-gold prompt increased GLiNER-Relex strict F1 by 32.9 percentage points on this sample. GLiNER-Relex returned no predictions on the 11 sentences without annotated relations in either condition. One of the 83 gold relations had an endpoint or relation label outside the train inventory. TPLinker selected Anno's rule-based fallback on all 32 sentences in both runs. Its general-purpose entity and relation rules have limited coverage of the CrossRE scientific vocabulary. [Machine-readable measurements](crossre-ai-prompt-comparison.json)

Each row runs in a fresh process on the same two logical CPUs, with model and dataset files cached locally. Elapsed time includes artifact fingerprinting, model loading and scoring. RSS is peak resident memory of the benchmark process. These single-run measurements include OS caching and competing system load. The sample covers one scientific domain and one seed. It cannot establish an overall backend ranking, training-set exposure of pretrained weights, or steady-state throughput.

## Replay

Install the joint GLiNER-Relex graph according to the backend guide and seed the dataset cache with the pinned AI test split. Verify that the TPLinker build resolves to the heuristic producer to reproduce that baseline. The JSON results record the effective producer count, prompt vocabulary, endpoint source, sample digest and quality denominators. [Backend guide](../BACKENDS.md) · [Held-out source](https://raw.githubusercontent.com/mainlp/CrossRE/a58885fa760559d2dc9e176e730fe731f020d6f3/crossre_data/ai-test.json)

```bash
export ANNO_RELATION_ORACLE_ENTITIES=0
for backend in gliner_relex tplinker; do
  for prompt in fixed document-gold; do
    schema=()
    if [[ "$prompt" == fixed ]]; then
      schema=(--relation-schema docs/eval/crossre-ai-prompt-schema.json)
    fi
    anno benchmark --tasks relation --datasets CrossREAI --backends "$backend" \
      --max-examples 32 --seed 42 --cached-only "${schema[@]}" \
      --output "$backend-$prompt.md" --output-json "$backend-$prompt.json"
  done
done
```

The SHA-256 digest of the train file is `44d54f2669c2a41f613aac70095aed8a3b8f2e6befde454abb99bf97c77385d3`. The test file digest is `2000a4011676319f9f3f3f6044e7de356b975d81979e080460e82c2273ed55cb`. The graph digest is `084b3873084a4173fc443c8bca691aabdf9f4357972fe032766bf6ec0c76a689`. [Training source](https://raw.githubusercontent.com/mainlp/CrossRE/a58885fa760559d2dc9e176e730fe731f020d6f3/crossre_data/ai-train.json)
