# Joint extraction stage replay

## JEREX on annotated documents

The CPU replay uses the first three records in published order from the authors’ DocRED joint test split. Model input consists only of sentence tokens. Predicted mention and entity indices are local to each document. Gold annotation enters only during scoring. The three documents are “Jeff Prosserman”, “Pettigrew State Park” and “Homi Wadia”. [Source test split](https://lavis.cs.hs-rm.de/storage/jerex/public/datasets/docred_joint/test_joint.json) · [Checkpoint and split digests](../../scripts/qa/joint_baseline/receipts/jerex.json)

| Observation | Gold | Predicted | Matched |
|---|---:|---:|---:|
| Distinct mention spans | 104 | 106 | 99 |
| Unordered mention pairs in the same cluster | 56 | 58 | 52 |
| Exact cluster membership | 79 | 80 | 72 |
| Exact membership and entity type | 79 | 80 | 68 |
| Directed pairs with any relation, exact cluster endpoints | 72 | 56 | 16 |
| Directed labeled relations, exact cluster endpoints | 76 | 58 | 17 |
| Directed labeled relations, exact typed cluster endpoints | 76 | 58 | 16 |

A singleton contributes one mention and one cluster, and contributes zero mention pairs. A directed endpoint pair can have multiple predicates. This accounts for the difference between the directed-pair and labeled-relation counts. Multiple gold vertices with identical mention sets can make relation alignment ambiguous. The scorer reports lower and upper possible counts. Those counts coincide for these documents. The counts describe this fixed sample. [Scorer](../../crates/anno-eval/examples/docred/score_jerex.py)

## Frozen projection fixture

A separate authored source lets the Rust test replay actual model output without a checkpoint download. The source says that Marie Curie was born in Warsaw and later moved to Paris. JEREX recognized “Marie Curie” and “Curie” as one cluster. It predicted one birthplace relation (P19) between that cluster and Warsaw. The scorer supplied probabilities for individual mention detections. Its upstream relation writer supplies directed concept indices and a predicate. The grounded importer therefore leaves the relation score absent and records both endpoint memberships. [Input](../../scripts/qa/joint_baseline/jerex_input.json) · [Frozen prediction](../../scripts/qa/joint_baseline/jerex_prediction.json)

The fixture begins with a pre-source-map grounded JSON containing four heuristic signals. The importer verifies token alignment against the exact supplied text, grounds the five JEREX detections, creates their four tracks, and records the observed edge. The test reloads the document, verifies original source slices, merges the head’s track with an unrelated track, then reloads again. The head no longer projects to an exact current track. The source memberships, all four older signals and the tail projection remain reproducible. [Legacy grounded fixture](../../scripts/qa/joint_baseline/legacy_grounded.json) · [Importer and regression](../../crates/anno/examples/import_jerex.rs)

```bash
cargo test -p anno --no-default-features --example import_jerex
cargo run -p anno --no-default-features --example import_jerex -- \
  scripts/qa/joint_baseline/jerex_input.json scripts/qa/joint_baseline/jerex_prediction.json > grounded.json
anno query relations grounded.json
```

To rerun trained JEREX, use the CPU environment documented in [QA](../QA.md#run-the-jerex-joint-checkpoint). The verifier checks the upstream checkout, five checkpoint files and the original test split, passes the source documents through the predictor, and compares each diagnostic count and the authored fixture’s spans, memberships, directed labels and mention probabilities.

```bash
python scripts/qa/joint_baseline/jerex_stages.py /models/jerex-source /models/docred-joint ./test_joint.json
```

## JointIE window and candidate coverage

The paired runs use the first eight CrossRE AI held-out sentences in published order. They retain empty-relation sentences. The same vocabulary from the training split supplies 14 entity types and 15 directed predicates. Each predicate permits all training entity types at either endpoint. No document gold types or predicates enter the model prompt. Token boundaries in the benchmark are converted to character offsets after joining the published tokens with spaces. [Training vocabulary](crossre-ai-prompt-schema.json) · [Held-out data](https://raw.githubusercontent.com/mainlp/CrossRE/a58885fa760559d2dc9e176e730fe731f020d6f3/crossre_data/ai-test.json)

A gold pair has window support if both complete spans occur in at least one actual model window. A scored gold edge additionally requires the exact endpoint coordinates, both entity types, direction and predicate in the merged sparse candidates. A selected gold edge must also survive the document-wide decoder. Exact model mention proposals can coexist with missing gold edge candidates. The actual candidate count includes low-scoring entries. These quantities distinguish the upper bound imposed by windowing from what the model scored and selected. [Production window and decode path](../../crates/anno-cli/src/cli/commands/joint.py) · [Coverage runner](../../scripts/qa/joint_baseline/jointie_coverage.py)

| Stage | 16 words, overlap 4 | 384 words, overlap 64 |
|---|---:|---:|
| Scored windows | 20 | 8 |
| Gold directed pairs | 12 | 12 |
| Gold pairs with window support | 7 | 12 |
| Gold typed mentions in sparse candidates | 24 | 24 |
| All typed candidate edges | 61,152 | 34,104 |
| Gold labeled edges in sparse candidates | 3 | 3 |
| Decoded mentions | 28 | 23 |
| Gold mentions decoded | 10 | 8 |
| Decoded labeled edges | 3 | 0 |
| Gold labeled edges decoded | 0 | 0 |

[Measured default-window run](../../scripts/qa/joint_baseline/receipts/jointie_384.json)

Under the 16-word ceiling, five gold edges crossed a window boundary that no overlapping window bridged. Four more shared a window but were absent from the typed candidate list. The remaining three were scored and none were selected. All 24 gold mentions were proposed and 10 survived decoding. [Measured 16-word run](../../scripts/qa/joint_baseline/receipts/jointie_16.json)

Use the GLiNER2.5 Python environment with the local pinned Fastino small v1 checkpoint. The replay reads the trained model’s sparse candidates immediately before the production decoder and uses its actual returned window offsets. Both receipts preserve per-document windows, counts, source digests, package versions and graph-file digests.

```bash
python scripts/qa/joint_baseline/jointie_coverage.py ai-test.json /models/gliner2.5-small-v1 run-16.json \
  --window-words 16 --overlap-words 4 --check scripts/qa/joint_baseline/receipts/jointie_16.json
python scripts/qa/joint_baseline/jointie_coverage.py ai-test.json /models/gliner2.5-small-v1 run-384.json \
  --window-words 384 --overlap-words 64 --check scripts/qa/joint_baseline/receipts/jointie_384.json
```
