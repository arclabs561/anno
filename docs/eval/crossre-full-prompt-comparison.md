# How much do gold prompts change CrossRE relation extraction?

On all 2,446 official test sentences, GLiNER-Relex scored **29.5% strict F1** when given the entity and predicate vocabularies from each domain’s training split. Supplying the labels from each test sentence’s gold relations raised pooled F1 to **58.2%**, a gain of 28.7 percentage points on the same 9,168 reference relations. Reject 58.2% as an estimate for extraction on unseen text. It gives the model part of the answer. Use 29.5% for that decision. Music scored above news with the fixed prompt (33.1% versus 27.7%). News scored above music with the gold prompt (74.2% versus 59.6%). [Paired measurements](crossre-full-prompt-comparison.json) · [CrossRE paper](https://aclanthology.org/2022.findings-emnlp.263/)

![The inference-time column gives 29.5% strict F1 with a fixed vocabulary. Two diagnostics using test annotations give 32.2% after filtering and 58.2% with gold prompts. The lower panel counts sentences with zero, one or several reference predicate types in each domain.](../figures/crossre-prompts.png)

**Filtered** recalculates the fixed-prompt score after removing its 1,659 predictions on the 620 sentences with no reference relation. All 2,446 sentences and 9,168 reference relations still count. Knowing which sentences are relation-free accounts for 2.7 F1 points in this calculation. Supplying the full gold prompt adds a further 26.0 points. Both diagnostics require the held-out annotations. Removing 315 false alarms adds 8.2 points in news. Removing 341 adds 2.4 points in music, whose reference set is more than six times larger. [Numerators, denominators and prompt policies](crossre-full-prompt-comparison.json)

| Test domain | Gold relations | Fixed exact / emitted | Gold-prompt exact / emitted | Fixed F1 | Filtered F1 | Gold F1 | Gain (pts) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **All domains** | 9,168 | 2,887 / 10,405 | 5,246 / 8,858 | 29.5% | 32.2% | 58.2% | +28.7 |
| AI | 1,164 | 256 / 963 | 680 / 1,251 | 24.1% | 26.0% | 56.3% | +32.2 |
| Literature | 1,623 | 614 / 2,174 | 964 / 1,660 | 32.3% | 34.0% | 58.7% | +26.4 |
| Music | 2,415 | 834 / 2,619 | 1,390 / 2,246 | 33.1% | 35.5% | 59.6% | +26.5 |
| News | 396 | 192 / 989 | 296 / 402 | 27.7% | 35.9% | 74.2% | +46.5 |
| Politics | 2,124 | 623 / 2,202 | 1,140 / 1,881 | 28.8% | 30.4% | 56.9% | +28.1 |
| Science | 1,446 | 368 / 1,458 | 776 / 1,418 | 25.3% | 29.9% | 54.2% | +28.8 |

An exact match requires the directed head and tail character spans and a case-insensitive predicate in the same sentence. Matching is one-to-one, including duplicate occurrences, and ignores endpoint type strings. Every condition predicts its own endpoints. Pooled F1 uses the sum of exact, emitted and gold counts across domains. Music and politics contribute 49.5% of the reference edges from 32.7% of the sentences. [Native scoring](../../crates/anno-eval/src/eval/relation.rs)

The gold prompts add **2,359 net exact matches**. Only 108 reference relations use a predicate missing from that domain’s training inventory. Even if gold prompts recovered all 108, at least **2,251** of the net gain came from relations whose predicate was already available. The experiment changes the entity and predicate lists together. Measuring their separate contributions requires holding one list fixed. An earlier per-edge replay found 19 AI relations and 23 literature relations that only the fixed prompt matched. [Error analysis](backend-residuals.md) · [Training-vocabulary coverage](crossre-prompt-scope.csv)

On the same six splits, Anno’s TPLinker heuristic emitted two relations with the fixed inventories and nine with gold prompts. It matched zero reference edges in either condition. With an empty predicate list the heuristic uses its built-in relation vocabulary. Eight of its nine gold-prompt outputs consequently occurred on sentences with no reference relation. [Effective producers and counts](crossre-full-prompt-comparison.json) · [TPLinker fallback](../../crates/anno/src/backends/tplinker.rs)

## Which labels reach the model?

The fixed prompt uses the entity and predicate inventories from that domain’s publisher training split. The document-gold prompt uses the entity types at the endpoints of its annotated relations and the predicates on those relations. It gives the extractor an empty predicate list on 620 of 2,446 test sentences and a single possible predicate type on another 718. A single predicate can still connect several pairs. [Prompt construction and scoring](../../crates/anno-eval/src/eval/task_evaluator.rs) · [All 18 split measurements](crossre-prompt-scope.csv)

Measured per gold relation, the document prompt offers 2.4 predicate choices on average. The fixed inventories contain 14 to 17. For 2,440 of 9,168 gold relations (26.6%), the prompt supplies exactly one predicate, leaving the extractor to find its directed endpoints. This applies to 196 of 396 news relations (49.5%) and 245 of 1,623 literature relations (15.1%). To reproduce the average, multiply each sentence’s number of distinct predicates by its gold relation count, sum across sentences and divide by 9,168. [Predicate counts weighted by relations](crossre-prompt-scope.csv)

The training vocabulary excludes 108 of 9,168 test relations (1.2%) by predicate. A stricter rule would also require the head and tail types to have occurred in their respective roles for that predicate during training. Those occurrences can come from different examples. That rule excludes 2,006 relations (21.9%), including the 108. Science illustrates the distinction. Its training vocabulary contains every test predicate, yet 466 of 1,446 gold relations fall outside the observed endpoint roles. The GLiNER-Relex runs use the broader inventories. [Split measurements](crossre-prompt-scope.csv) · [JointIE role restriction on an AI sample](joint-edge-loss.md)

## Which other domains did the corpus survey cover?

The corpus reader counts 5,265 sentences across all 18 train, development and test files. The model comparison scores the six official test files. The training files supply vocabularies, and development contributes to the corpus survey. Music and news have almost the same test sentence count. Music has more than five times as many tokens inside gold entity spans and over six times as many gold relations. Each token covered by one or more gold entity spans counts once. [File sizes, token counts and SHA-256 identities](dataset-repo-files.csv)

| Domain | Training sentences | Train entity / predicate types | Test sentences | Tokens in entity spans / all test tokens | Test gold relations | Unseen predicate | Outside training head/tail roles |
|---|---:|---:|---:|---:|---:|---:|---:|
| AI | 100 | 14 / 15 | 431 | 3,782 / 12,991 (29.1%) | 1,164 | 12 | 326 |
| Literature | 100 | 13 / 16 | 416 | 5,318 / 16,157 (32.9%) | 1,623 | 2 | 264 |
| Music | 100 | 14 / 14 | 399 | 6,870 / 17,099 (40.2%) | 2,415 | 31 | 370 |
| News | 164 | 5 / 14 | 400 | 1,344 / 5,134 (26.2%) | 396 | 35 | 87 |
| Politics | 101 | 13 / 15 | 400 | 6,156 / 16,531 (37.2%) | 2,124 | 28 | 493 |
| Science | 103 | 18 / 17 | 400 | 4,882 / 14,304 (34.1%) | 1,446 | 0 | 466 |

Five music test token sequences also occur in its training file. The other five domains have zero exact within-domain train–test token-sequence matches. Repetitions within a test file remain in the score, including 24 repeated news rows by token sequence. The upstream checkpoint’s training exposure to these test texts is unknown. [Corpus identities and duplicate counts](crossre-full-prompt-comparison.json)

## Reproduce the comparison

The Python corpus reader verifies all 18 pinned publisher file digests and their sentence and token counts. Add `--cache-dir PATH` to reuse a publisher-file cache. The Matplotlib generator renders PNG and SVG from the checked-in CSV and model measurements. [Corpus reader](../../scripts/survey_crossre_prompts.py) · [Figure generator](../../scripts/plot_benchmark_examples.py)

```sh
python3 scripts/survey_crossre_prompts.py /tmp/crossre-prompt-scope.csv
cmp /tmp/crossre-prompt-scope.csv docs/eval/crossre-prompt-scope.csv
python3 scripts/plot_benchmark_examples.py .generated/dataset-figures
```

For fresh predictions, build the Linux CLI with `eval` and `onnx` and prepare the joint ONNX export of `knowledgator/gliner-relex-large-v1.0`, revision `4aedc9226a5ac9e2f6b5ea3e91c1ee577c88a290`. The replay verifies the training and test file identities, the graph, tokenizer and configuration, the actual producer on every sentence, the prompt policy, the confidence threshold and the scored denominators. Domains that share Anno’s generic `CrossRE` dataset ID each get a separate cache. The TPLinker control runs with a private model cache to select Anno’s heuristic implementation. [Replay script](../../scripts/qa/crossre_prompt_replay.py) · [Model and runtime setup](../BACKENDS.md#architecture-specific-gotchas) · [Pinned model checkpoint](https://huggingface.co/knowledgator/gliner-relex-large-v1.0/tree/4aedc9226a5ac9e2f6b5ea3e91c1ee577c88a290)

```bash
cargo build --locked -p anno-cli --bin anno --features eval,onnx
python3 scripts/export_gliner_relex_onnx.py --joint \
  --revision 4aedc9226a5ac9e2f6b5ea3e91c1ee577c88a290
replay_dir="$(mktemp -d)"
mkdir "$replay_dir/input"
python3 scripts/qa/check_gliner_relex_cli.py --anno target/debug/anno \
  --receipt "$replay_dir/model-check.json"
base=https://raw.githubusercontent.com/mainlp/CrossRE/a58885fa760559d2dc9e176e730fe731f020d6f3/crossre_data
for domain in ai literature music news politics science; do
  for split in train test; do
    curl -fsSL "$base/$domain-$split.json" -o "$replay_dir/input/$domain-$split.json"
  done
done
cpus=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:2])))')
python3 scripts/qa/crossre_prompt_replay.py --anno target/debug/anno --all-domains \
  --data "$replay_dir/input" --out "$replay_dir/results" --cpus "$cpus"
python3 scripts/plot_benchmark_examples.py "$replay_dir/figures" \
  --crossre-receipt "$replay_dir/results/summary.json"
```

Each condition uses a fresh process, native ONNX Runtime 1.24.2 and a confidence threshold of 0.5. AI and literature retain the earlier published measurements. Music, news, politics and science use Anno revision `f942797` with the generic dataset ID routed through its existing CrossRE relation reader. The rebuilt joint export has the same graph, tokenizer and configuration SHA-256 digests as the earlier trials. On the historical 32-sentence AI panel, the rebuilt CLI reproduced both GLiNER-Relex scores, all associated metrics and the ordered sample digest. A second pass with isolated model caches reproduced all 12 full-test heuristic conditions, including the ordered samples and every metric. [Frozen native evaluator](https://github.com/arclabs561/anno/blob/f9427977f2f27b32f26c4c77a6795a8a4dce596c/crates/anno-eval/src/eval/task_evaluator.rs) · [Pilot receipt](crossre-ai-prompt-comparison.json) · [Native model parity check](../../scripts/qa/check_gliner_relex_cli.py)

The receipt records the train and test file digests for each domain and an ordered sample digest for each condition. The four additional domains also record the file digest reported by each native process. Each condition lists its executable identity, CPU affinity, elapsed and CPU time and peak resident memory. Times include startup and model loading on shared hosts. The CrossRE publisher repository specifies GPL-3.0 and gives no separate annotation terms. [Trial measurements](crossre-full-prompt-comparison.json) · [Earlier AI sample](crossre-ai-prompt-comparison.md) · [Publisher license](https://github.com/mainlp/CrossRE/blob/a58885fa760559d2dc9e176e730fe731f020d6f3/LICENSE)
