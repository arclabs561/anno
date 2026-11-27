# What GLiNER-Relex's directed pair head missed

The stock checkpoint produced 85 relations on 32 English CrossRE AI sentences, of which 17 matched 90 annotated directed span-and-predicate occurrences. This run used the actual source weights, the trained export module and a vocabulary obtained from the training split. Every document's predicted edge multiset from that module agreed with upstream `predict_relations`. Anno's native ONNX frontend has separate upstream-parity fixtures. [Export module](../../scripts/export_gliner_relex_joint.py) · [Native parity](../../scripts/qa/check_gliner_relex_cli.py) · [Training vocabulary](crossre-ai-prompt-schema.json)

| First loss on indices 64–95 | Missed relations |
|---|---:|
| Head below mention threshold | 2 |
| Tail below mention threshold | 6 |
| Flat mention decoding suppressed an overlapping endpoint | 16 |
| Both exact endpoints survived, predicate score below the output gate | 49 |
| **Total** | **73** |

The exported pair head scored 3,526 ordered pairs before final filtering. A relation can disappear because a high-scoring overlapping mention replaces its required endpoint, even when the graph scored that pair. Native extraction can expose the word-to-character mapping, each span and label score, flat overlap decisions and each directed pair and predicate score. With an `onnx,instrument` CLI and an installed joint export, limit a developer run to one document, then inspect its candidate log using prompt label order. Full diagnostics can reveal source-derived annotations and should stay with the evaluator.

```sh
RUST_LOG=anno::relex_candidates=trace anno extract --model gliner-relex \
  --extract-types person,organization --extract-relations --relation-types founded \
  --format json --text 'Ada founded Acme.' > "$out/prediction.json" 2> "$out/candidates.log"
```

## Which competing span should survive?

Counting distinct intervals within each sentence gives 26,925 annotated entity spans across all 5,265 sentences in the 18 pinned CrossRE files. Zero sentences contain two distinct overlapping gold spans. Each endpoint of the 19,761 relation annotations matches an annotated entity. The 16 losses to overlap suppression on the AI diagnostic panel therefore involve competition with a predicted boundary. A flat decoder can represent the gold entity layout in these files. [Counts by domain and split](crossre-prompt-scope.csv) · [Validation and counting code](../../scripts/survey_crossre_prompts.py)

The first-loss count includes an edge as soon as either endpoint is suppressed. We replayed the same 32 AI test sentences and inspected the trained predicate score for each of those 16 edges in its annotated direction. Both exact endpoints could coexist in all 16. Only **three** correct predicate scores exceeded 0.50. The other 13 were at or below the gate. Even with gold boundaries to guide the selection, changing span order alone could take the exact count from 17/90 to at most 20/90 on this panel. The stock export reproduced upstream predictions on all 32 sentences. [Paired score counts](../../scripts/qa/joint_baseline/receipts/relex_ai32_suppressed.json) · [Replay](../../scripts/qa/joint_baseline/relex_roles.py)

Inspect predicate ranking first. Across the 73 misses, the trained head scored the exact directed pair but assigned its reference predicate at most 0.50 in 62 cases, including 49 with both endpoints retained. Compare the reference score with the strongest unannotated predicate for the same pair and with its score in the reverse direction, accounting for pairs with multiple valid annotations. This separates correct low-confidence rankings from cases where the model prefers a different label or direction. Test any scoring change on fresh documents and count exact directed edges gained and lost, plus new unmatched outputs. The native decoder exposes the required pair scores before suppressing overlapping mentions. [Native decoder](../../crates/anno/src/backends/gliner_onnx/relex.rs)

## Predicate-only threshold trial

We froze a lower predicate gate of 0.40 and tested it on the next 32 source sentences (indices 96–127). All 32 text hashes were distinct from each other and from the prior 96. The mention gate stayed at 0.50. Both decodes used the same 3,280 ordered pair scores and final mention list, with 77 gold relations. The stock export-path outputs again agreed with the upstream public predictions on every document.

| Predicate gate | Exact / predicted | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Stock 0.50 | 24 / 85 | 28.2% | 31.2% | 29.6% |
| Experimental 0.40 | 29 / 112 | 25.9% | 37.7% | 30.7% |

The lower gate recovered five gold edges, retained all previous matches and added 22 unmatched predictions. A caller that reviews suggestions can evaluate whether those five extra matches justify reviewing 27 additional outputs. A high-precision caller has a different cost. This English AI-domain result does not establish a multilingual default. On these same 32 sentences, the stock Rust CLI matched all 252 upstream mentions and all 85 directed edges. The maximum absolute score differences were 0.00000227 for mentions and 0.00000191 for edges. The replay also checked each grounded edge after reloading the export.

We repeated the predicate-only comparison on indices 128–159, another 32 distinct sentences with no text in common with the first 128. Of 93 annotated relations, the stock gate recovered 10 among 63 predictions. The lower gate recovered 12 among 88 predictions, retaining the original matches and adding 23 unmatched outputs for two additional matches. Both panels support leaving the default threshold in place. Candidate ranking on retained endpoints remains the larger opportunity. On the replication panel, 66 misses had both endpoints but a predicate score below the gate. [Replication counts](relex-predicate-replication.json)

## Reproduce

Use the pinned local `knowledgator/gliner-relex-large-v1.0` checkpoint at revision `4aedc9226a5ac9e2f6b5ea3e91c1ee577c88a290`, `gliner==0.2.30` and the official pinned CrossRE AI test JSONL. The probe checks the file and weight digests and emits aggregate counts. The held-out slice uses the same fixed inventories of 14 entity types and 15 relations. Scoring uses case-sensitive source spans and direction, preserves repeated reference annotations and compares predicates from the published vocabulary. Raw text and reference annotations stay external. [Checkpoint and license](https://huggingface.co/knowledgator/gliner-relex-large-v1.0/tree/4aedc9226a5ac9e2f6b5ea3e91c1ee577c88a290) · [CrossRE data and terms](https://github.com/mainlp/CrossRE/tree/a58885fa760559d2dc9e176e730fe731f020d6f3) · [Probe](../../scripts/qa/joint_baseline/relex_roles.py)

```sh
python scripts/qa/joint_baseline/relex_roles.py "$test" "$relex" --first-index 64 --documents 32
python scripts/qa/joint_baseline/relex_roles.py "$test" "$relex" --first-index 96 --documents 32 --predicate-threshold 0.40
python scripts/qa/joint_baseline/relex_roles.py "$test" "$relex" --first-index 128 --documents 32 --predicate-threshold 0.40
```

The held-out CI replay checks the fixed 32-sentence panel against the saved counts and a digest of every stock mention and directed edge in its original sentence. It exports the trained joint graph, runs the stock Rust CLI on each sentence, and compares every mention and directed relation with upstream PyTorch, including confidence and grounded reloads. The native CLI uses a shared mention and predicate threshold of 0.50. The predicate-only 0.40 comparison uses the original 0.50 mention list in Python. [Workflow](../../.github/workflows/relex-crossre.yml) · [Frozen counts](relex-predicate-heldout.json)

With the joint export installed and the native CLI built, run both sides locally. The temporary case file contains the publisher text and reference predictions.

```sh
trial_dir=$(mktemp -d)
python scripts/qa/joint_baseline/relex_roles.py "$test" "$relex" --first-index 96 --documents 32 \
  --predicate-threshold 0.40 --native-cases "$trial_dir/cases.json" > "$trial_dir/counts.json"
python scripts/qa/check_gliner_relex_cli.py --anno target/debug/anno \
  --crossre-cases "$trial_dir/cases.json" --receipt "$trial_dir/native.json"
```
