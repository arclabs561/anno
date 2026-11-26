# Full-domain relation extraction on CrossRE AI and literature

GLiNER-Relex received either a fixed vocabulary from the domain's 100 training sentences or the entity and relation labels occurring in each test sentence's gold relations. All four producer–prompt combinations predicted their own mention endpoints. Moving from the train vocabulary to the reference-conditioned prompt changed strict F1 by 32.2 percentage points on AI and 26.4 on literature. The comparison uses all 847 sentences and 2,787 gold relations in the two official test splits. [Measurements](crossre-full-prompt-comparison.json) · [Dataset paper](https://aclanthology.org/2022.findings-emnlp.263/)

![Gold prompts disclose zero or one predicate type in 350 of 400 news sentences and 141 of 416 literature sentences. On AI and literature, removing predictions on sentences with no gold relations adds 2.0 and 1.6 F1 points. The full gold prompts add 32.2 and 26.4 points.](../figures/crossre-prompts.png)

Across all six test domains, the reference prompt tells the extractor that **620 of 2,446 sentences have no relation** and gives it exactly one possible predicate type in another **718**. A single predicate can still connect several different pairs. Report which vocabulary the caller will have at prediction time. For an application that knows only the training labels, compare the fixed-prompt scores. [All 18 split measurements](crossre-prompt-scope.csv)

| Test domain | Producer | Prompt | Exact / predicted / gold | Strict P / R / F1 | Wall (s) | CPU (s) | Peak RSS (GiB) |
|---|---|---|---:|---:|---:|---:|---:|
| AI | GLiNER-Relex | Train inventory | 256 / 963 / 1164 | 26.6% / 22.0% / **24.1%** | 806.5 | 1323.7 | 2.80 |
| AI | GLiNER-Relex | Document gold | 680 / 1251 / 1164 | 54.4% / 58.4% / **56.3%** | 563.7 | 930.8 | 2.90 |
| AI | TPLinker heuristic | Train inventory | 0 / 0 / 1164 | — / 0.0% / **0.0%** | 8.4 | 2.1 | 0.05 |
| AI | TPLinker heuristic | Document gold | 0 / 0 / 1164 | — / 0.0% / **0.0%** | 3.1 | 0.8 | 0.05 |
| Literature | GLiNER-Relex | Train inventory | 614 / 2174 / 1623 | 28.2% / 37.8% / **32.3%** | 649.9 | 1215.6 | 2.60 |
| Literature | GLiNER-Relex | Document gold | 964 / 1660 / 1623 | 58.1% / 59.4% / **58.7%** | 410.1 | 769.0 | 2.73 |
| Literature | TPLinker heuristic | Train inventory | 0 / 0 / 1623 | — / 0.0% / **0.0%** | 12.4 | 4.5 | 0.05 |
| Literature | TPLinker heuristic | Document gold | 0 / 1 / 1623 | 0.0% / 0.0% / **0.0%** | 3.3 | 1.6 | 0.05 |

The numerator is the count of one-to-one exact matches of directed head span, tail span and case-insensitive relation label within a sentence. The denominator includes all gold relations, even when the fixed schema lacks their label. Entity type strings do not enter the strict match. A dash is undefined precision because the producer predicted no relations. TPLinker used Anno's heuristic path on every sentence. These two English Wikipedia-derived domains measure sentence-level extraction. They provide no measurement of document-level clustering, pretrained exposure to these texts, or neural TPLinker quality.

## Which labels reach the model?

| Full test split | Training entity / relation labels | Test sentences / gold relations | Sentences without relations | Gold with unseen relation label | Gold with unseen endpoint type |
|---|---:|---:|---:|---:|---:|
| AI | 14 / 15 | 431 / 1,164 | 105 | 12 | 6 |
| Literature | 13 / 16 | 416 / 1,623 | 47 | 2 | 15 |

AI has 11 instances of `opposite` and one of `social` outside the train relation inventory. Literature has two of `usage`. A gold triplet with an unseen endpoint type can still match by its spans and relation label. GLiNER-Relex predicted 160 triplets on relation-free AI sentences and 181 on relation-free literature sentences with the fixed vocabulary. With an empty relation-label list, Relex emits no triplets. Removing only these false positives from the saved fixed runs gives 26.0% strict F1 on AI and 34.0% on literature, keeping every gold relation in the denominator. The remaining gaps to the per-sentence gold runs are 30.3 and 24.8 points. The gold prompt changes both the entity and relation vocabularies on positive sentences, so these runs measure their combined effect. [Paired trial receipt](crossre-full-prompt-comparison.json)

Neither domain has identical token sequences across its train and test files. AI contains one repeated test token sequence and one duplicate annotated triplet within a sentence. The full-split evaluation retains them. Both the native input/reference digest and the original file digests are recorded with the results. The native digest is identical across the four runs within each domain. [Pinned train and test files](https://github.com/mainlp/CrossRE/tree/a58885fa760559d2dc9e176e730fe731f020d6f3/crossre_data)

## Which other domains did the corpus survey cover?

The corpus reader counted 5,265 sentences across train, development and test in all six CrossRE domains. Each token covered by one or more gold entity spans counts once. The news and music test files have nearly the same sentence count, yet music contains over five times as many entity-covered tokens. A reviewer planning a fixed-size panel will also encounter 396 gold relation occurrences in news and 2,415 in music. Only AI and literature have paired full-test model predictions. [All 18 split files and SHA-256 digests](dataset-repo-files.csv)

| Test domain | Sentences | Tokens inside gold entity spans / all tokens | Gold relations | Unseen predicate | Outside training head/tail roles |
|---|---:|---:|---:|---:|---:|
| AI | 431 | 3,782 / 12,991 (29.1%) | 1,164 | 12 | 326 |
| Literature | 416 | 5,318 / 16,157 (32.9%) | 1,623 | 2 | 264 |
| Music | 399 | 6,870 / 17,099 (40.2%) | 2,415 | 31 | 370 |
| News | 400 | 1,344 / 5,134 (26.2%) | 396 | 35 | 87 |
| Politics | 400 | 6,156 / 16,531 (37.2%) | 2,124 | 28 | 493 |
| Science | 400 | 4,882 / 14,304 (34.1%) | 1,446 | 0 | 466 |

A caller restricting predicates to the domain's training list would exclude **108 of 9,168 test relations (1.2%)** by label. A stricter rule also requires the gold head type to have appeared as a head for that predicate during training and the gold tail type to have appeared as a tail. The two roles may come from different training examples. This rule would exclude **2,006 of 9,168 typed relations (21.9%)**, including the 108 with unseen predicates. In science, training supplies every test predicate, yet the role rule would exclude 466 of 1,446 gold relations. These are counts of gold annotations a schema permits. The paired GLiNER-Relex runs above used the broader lists. [Split measurements](crossre-prompt-scope.csv) · [Measured JointIE role restriction on an AI prefix](joint-edge-loss.md)

## Reproduce the comparison

To rebuild the corpus measurements, read all 18 pinned publisher files with the Python standard library. The script verifies each file digest and its sentence and token counts against the ledger. On held-out rows, it uses only the training split from the same domain to construct predicate and role inventories. The chart renders offline from the checked-in CSV and paired model receipt. [Corpus reader](../../scripts/survey_crossre_prompts.py) · [Matplotlib generator](../../scripts/plot_benchmark_examples.py)

```sh
python3 scripts/survey_crossre_prompts.py /tmp/crossre-prompt-scope.csv
cmp /tmp/crossre-prompt-scope.csv docs/eval/crossre-prompt-scope.csv
python3 scripts/plot_benchmark_examples.py .generated/dataset-figures
```

Add `--cache-dir PATH` to the survey command to reuse the publisher file survey's cache.

Cache a verified native joint export for `knowledgator/gliner-relex-large-v1.0` and build the CLI with `eval` and `onnx`. The replay requires Linux and a Python 3 standard library. It verifies dataset checksums before building the prompt, the effective producer for every document, the graph, tokenizer and model configuration fingerprints, identical sample hashes, and quality denominators. The upstream repository declares GPL-3.0 and does not specify separate terms for the annotations. Use the data under the terms applicable to your work. The following commands keep the raw inputs and replay artifacts in a temporary directory outside the checkout. Choose two CPU numbers allowed on your host. [Replay script](../../scripts/qa/crossre_prompt_replay.py) · [Backend setup](../BACKENDS.md)

```bash
replay_dir="$(mktemp -d)"
mkdir "$replay_dir/input"
base=https://raw.githubusercontent.com/mainlp/CrossRE/a58885fa760559d2dc9e176e730fe731f020d6f3/crossre_data
for domain in ai literature; do
  for split in train test; do
    curl -fsSL "$base/$domain-$split.json" -o "$replay_dir/input/$domain-$split.json"
  done
done
python3 scripts/qa/crossre_prompt_replay.py --anno target/debug/anno \
  --data "$replay_dir/input" --out "$replay_dir/results" --cpus 0,1
```

Each row uses a fresh process with the same CPU affinity, ONNX Runtime 1.24.2, threshold 0.5, and cached artifacts. The process times include startup, fingerprinting, model load and scoring. Other tasks shared the host. Wall times are one observation per condition. The fixed AI Relex arm used the earlier native executable from the 32-sentence comparison. The two executables use the same extraction and scoring implementations and the same graph, tokenizer and model configuration. Their binary hashes are recorded. [Earlier 32-sentence comparison](crossre-ai-prompt-comparison.md)
