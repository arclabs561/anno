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

## Predicate-only threshold trial

We froze a lower predicate gate of 0.40 and tested it on the next 32 source sentences (indices 96–127). All 32 text hashes were distinct from each other and from the prior 96. The mention gate stayed at 0.50. Both decodes used the same 3,280 ordered pair scores and final mention list, with 77 gold relations. The stock export-path outputs again agreed with the upstream public predictions on every document.

| Predicate gate | Exact / predicted | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Stock 0.50 | 24 / 85 | 28.2% | 31.2% | 29.6% |
| Experimental 0.40 | 29 / 112 | 25.9% | 37.7% | 30.7% |

The lower gate recovered five gold edges, retained all previous matches and added 22 unmatched predictions. A caller that reviews suggestions can evaluate whether those five extra matches justify reviewing 27 additional outputs. A high-precision caller has a different cost. This English AI-domain result does not establish a multilingual default. The experiment ran the pinned PyTorch source and trained export module on CPU, checking parity with the public upstream API. The native Rust ONNX observer was exercised separately on a controlled Unicode overlap fixture.

## Reproduce

Use the pinned local `knowledgator/gliner-relex-large-v1.0` checkpoint at revision `4aedc9226a5ac9e2f6b5ea3e91c1ee577c88a290`, `gliner==0.2.30` and the official pinned CrossRE AI test JSONL. The probe checks the file and weight digests and emits aggregate counts. The held-out slice uses the same fixed inventories of 14 entity types and 15 relations. Scoring uses case-sensitive source spans and direction, preserves repeated reference annotations and compares predicates from the published vocabulary. Raw text and reference annotations stay external. [Checkpoint and license](https://huggingface.co/knowledgator/gliner-relex-large-v1.0/tree/4aedc9226a5ac9e2f6b5ea3e91c1ee577c88a290) · [CrossRE data and terms](https://github.com/mainlp/CrossRE/tree/a58885fa760559d2dc9e176e730fe731f020d6f3) · [Probe](../../scripts/qa/joint_baseline/relex_roles.py)

```sh
python scripts/qa/joint_baseline/relex_roles.py "$test" "$relex" --first-index 64 --documents 32
python scripts/qa/joint_baseline/relex_roles.py "$test" "$relex" --first-index 96 --documents 32 --predicate-threshold 0.40
```
