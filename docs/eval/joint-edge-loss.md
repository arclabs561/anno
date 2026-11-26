# Where joint extraction loses annotated relations

## GLiNER2.5 JointIE

The cohort consists of the first 64 CrossRE AI test sentences in publisher order, including documents with zero annotations. An observation is a directed predicate between two exactly typed source occurrences. Multiple predicates on one pair count separately. Repeated annotation rows for the same edge count once. All three runs use the same local `fastino/gliner2.5-small-v1` checkpoint, frozen training vocabulary, source edition and upstream GLiNER2 2.2.0 optimizer. These are diagnostics on a held-out prefix. [Held-out annotations](https://raw.githubusercontent.com/mainlp/CrossRE/a58885fa760559d2dc9e176e730fe731f020d6f3/crossre_data/ai-test.json) · [Original eight-document replay](joint-stage-baseline.md)

Each gold edge receives one outcome, in pipeline order. The declared predicate and endpoint types must permit it. Both complete spans must share a scored window, and both typed mention proposals must occur in the **same** window. An otherwise eligible edge can still be absent from the ordered pairs returned by the sparse scorer. The upstream sparse output does not identify the particular proposal operation that omitted an absent pair. A surviving sparse edge can fall below the candidate gate, exceed the cap for its predicate, lose a required endpoint, be omitted from the solver’s chosen graph, or be dropped during result construction. The trace compares the observed candidate probabilities, ranks and endpoint membership with the actual optimizer input, and reads both the solver solution and rendered graph. An edge’s utility is recorded when available. A negative utility alone cannot identify which competing graph the optimizer preferred. [Stage instrumentation](../../scripts/qa/joint_baseline/jointie_loss.py)

The first eight documents exactly reproduce the previous counts, document hashes and window geometry under both settings. Each arm scored three gold edges and decoded zero gold edges. Five edges lacked a shared short window and four more were absent from sparse output. All 12 shared a default-size window, where nine were absent from sparse output. In each arm, one `part-of` edge exceeded the per-predicate cap, while the optimizer received and omitted the other `part-of` and `compare` edges. All three exceeded the 0.05 candidate gate. The scorer expands a span-pair probability over multiple allowed typed endpoints. This means a large typed-candidate count can describe many copies of fewer geometric pair scores. [Pinned eight-document receipt](../../scripts/qa/joint_baseline/receipts/jointie_16.json)

### Expanded cohort

All 64 documents scored and returned a feasible solution under each configuration. The cohort contains 245 typed mentions, 155 distinct directed occurrence pairs and 161 labeled edges. Geometry counts include all gold edges, even the two whose predicate is outside the declared schema. The first-loss rows give each edge its earliest observed loss, so those counts sum to 161 including matches.

| Observation | 16 words, overlap 4 | 384 words, overlap 64 | 384 words, training-derived types |
|---|---:|---:|---:|
| Scored windows | 162 | 64 | 64 |
| Gold pairs co-window / 155 | 83 | 155 | 155 |
| Gold edges co-window / 161 | 86 | 161 | 161 |
| Gold mentions proposed / 245 | 244 | 244 | 244 |
| Gold mentions decoded / predicted mentions | 69 / 215 | 64 / 179 | 64 / 180 |
| Distinct predicate and directed span-pair scores | 2,807 | 2,041 | 2,041 |
| Typed candidate edges | 550,172 | 400,036 | 61,261 |
| Exact annotated edges scored / 161 | 23 | 22 | 16 |
| Exact annotated edges decoded / predictions | 1 / 27 | 0 / 2 | 0 / 4 |

| First outcome for each gold edge | 16 words | 384 words | 384 words, training-derived types |
|---|---:|---:|---:|
| Predicate outside declared schema | 2 | 2 | 2 |
| Endpoint types outside declared routes | 0 | 0 | 51 |
| Outside shared window | 73 | 0 | 0 |
| Absent from sparse output despite both typed mentions in a shared window | 63 | 137 | 92 |
| Scored, exceeded per-predicate cap | 9 | 17 | 5 |
| Presented to optimizer, unselected | 13 | 5 | 11 |
| Matched | 1 | 0 | 0 |
| **Total** | **161** | **161** | **161** |

Wider windows covered 75 additional annotated edges, yet produced one fewer exact typed candidate and one fewer exact match in the broad-schema arms. Geometry alone overestimates what the sparse scorer and decoder deliver. The short arm’s micro precision was 1/27 (3.7%) and recall 1/161 (0.6%). The two wider arms had zero exact matches. The observed result concerns this fixed schema, checkpoint and held-out sample. [Short-window totals](../../scripts/qa/joint_baseline/receipts/jointie_ai64_w16.json) · [Default-window totals](../../scripts/qa/joint_baseline/receipts/jointie_ai64_w384.json) · [Restricted-schema totals](../../scripts/qa/joint_baseline/receipts/jointie_ai64_training_routes.json)

### Separately tested change

The 384-word comparison restricts each predicate’s allowed head and tail **types** to the respective marginal sets observed in the pinned AI training split. It permits the Cartesian product of those two sets. This tests whether reducing expansion across endpoint types helps a capped decoder. No test labels are supplied during inference. Of the 161 held-out edges, two use a predicate absent from the fixed training vocabulary, and 51 more fall outside the narrower training-derived routes. All 161 remain in the reported denominator. The default CLI continues to honor the caller’s declared schema. [Training data](https://raw.githubusercontent.com/mainlp/CrossRE/a58885fa760559d2dc9e176e730fe731f020d6f3/crossre_data/ai-train.json)

The restricted arm generated the same 2,041 distinct directed predicate/span pairs as the broad schema, with 61,261 typed candidates in place of 400,036. It reduced annotated edges lost at the candidate cap from 17 to five, and increased annotated edges presented to the solver from five to 11. It also excluded 51 annotated edges at schema declaration and reduced exact scored gold edges from 22 to 16. Neither 384-word arm decoded any annotated relation. The comparison supports recording schema eligibility and candidate rank separately. It supplies no accuracy gain for changing the default schema. [Broad schema](../../scripts/qa/joint_baseline/receipts/jointie_ai64_w384.json) · [Training-derived schema](../../scripts/qa/joint_baseline/receipts/jointie_ai64_training_routes.json)

## JEREX

The JEREX cohort consists of the first 32 documents in the authors’ DocRED joint test split, in published order. Only sentence tokens enter the trained inference path. The observer examines annotations after the forward pass. A scored relation binds directed **clusters**, so correct labels attached to an incorrect partition do not receive exact-membership credit. A singleton contributes to the mention and cluster counts and contributes zero within-cluster mention pairs. Entity typing is scored as a separate condition on exact cluster matches and exact labeled relations. [Authors’ test split](https://lavis.cs.hs-rm.de/storage/jerex/public/datasets/docred_joint/test_joint.json) · [Checkpoint](https://lavis.cs.hs-rm.de/storage/jerex/public/models/docred_joint/joint_multi_instance/)

For each annotated directed edge, the trace checks whether its endpoint mention spans were enumerated and whether the detector retained them. If every required mention survived, it checks each cluster’s exact membership. A cluster mismatch can contain a split across predicted clusters, extraneous members, or both. For a uniquely aligned endpoint pair, the trace inspects the actual pair mask and the gold predicate’s sigmoid probability before masking. It verifies the threshold comparison against the returned classifier tensor. Ambiguous duplicate memberships receive a separate outcome. The writer’s independent endpoint-aware scoring must reproduce the trace’s gold and matched counts for every document. [JEREX observer](../../scripts/qa/joint_baseline/jerex_loss.py) · [Writer scoring](../../crates/anno-eval/examples/docred/score_jerex.py)

All 32 enrolled documents scored (141–458 encoded wordpieces). Of 406 labeled gold edges, 140 matched, 99 reached an exact aligned pair but fell below the 0.60 relation threshold, and 110 lost an endpoint at cluster construction despite every required mention surviving. Of those 110, 54 had extraneous cluster members alone, 28 split across clusters alone, and 28 showed both. An additional 56 edges lost at least one enumerated mention below the 0.85 mention threshold. One edge required a 13-token mention absent from the enumerated spans. All 239 edges with both endpoints uniquely aligned reached their directed pair classifier. The 140 matches included one with incorrect endpoint typing. These are edge-level counts, so a single missing mention can affect multiple relations. [Aggregate edge counts](../../scripts/qa/joint_baseline/receipts/jerex_docred32.json)

| Observation | Gold | Predicted | Matched |
|---|---:|---:|---:|
| Mention spans | 891 | 899 | 831 |
| Within-cluster mention pairs | 378 | 383 | 315 |
| Exact clusters | 686 | 696 | 584 |
| Exact typed clusters | 686 | 696 | 562 |
| Directed cluster pairs with any relation | 383 | 331 | 135 |
| Directed labeled relations | 406 | 344 | 140 |
| Directed labeled relations with typed endpoints | 406 | 344 | 139 |

The first three documents reproduce every published stage count in the earlier receipt, including 17 exact labeled matches out of 58 predictions and 76 annotated edges. The expanded cohort’s exact labeled micro precision is 40.7% and recall is 34.5%. The first-loss counts explain recall on this prefix. They do not establish why the model assigned an individual logit or which intervention would improve precision. [Original three documents](../../scripts/qa/joint_baseline/receipts/jerex.json)

### What the 204 unmatched predictions contain

Of 344 distinct directed labeled predictions, 140 matched and 204 were unmatched against the released annotations. In 121, at least one predicted cluster had no gold cluster with exactly its membership. For 48 of those edges, both endpoints contained only annotated mentions and each touched just one gold cluster. The inexact endpoint contained only part of that cluster. On 29 other edges at least one endpoint combined mentions from multiple gold clusters. On 14 an endpoint included a mention absent from all gold clusters, and on 30 both features occurred. These counts describe edges, so a single problematic cluster can contribute several times. [Aggregate prediction counts](../../scripts/qa/joint_baseline/receipts/jerex_predictions32.json)

The remaining 83 unmatched predictions had two uniquely aligned endpoints. The exact directed pair carried a different gold predicate on two predictions. On the other 81 it had no annotated predicate in that direction. None of the 83 carried a predicate annotated only in reverse. The saved writer output contains neither calibrated truth nor the logits or competing candidates behind these choices. An absent annotation establishes a benchmark mismatch. Determining whether the document actually supports the prediction would require source adjudication. [Classification and checks](../../scripts/qa/joint_baseline/jerex_predictions.py)

## Source terms and replay

Elisa Bassignana and Barbara Plank created the CrossRE annotations. Their repository publishes GPLv3 and names the data release in its README. It gives no separate annotation license. Yao et al. created DocRED from Wikipedia and Wikidata. THUNLP publishes an MIT license for its repository. Eberts and Ulges supply the separately hosted DocRED joint split used here. The JEREX repository's MIT notice covers its code, and the hosted split directory supplies no separate data license. The retained study files contain aggregate counts, selected settings and SHA-256 digests. Source text, document titles, gold annotations, predictions, route inventories and per-document diagnostics from these expanded studies stay in the evaluator's external working directory. [CrossRE release](https://github.com/mainlp/CrossRE/tree/a58885fa760559d2dc9e176e730fe731f020d6f3) · [CrossRE license](https://github.com/mainlp/CrossRE/blob/a58885fa760559d2dc9e176e730fe731f020d6f3/LICENSE) · [DocRED authors and sources](https://github.com/thunlp/DocRED) · [THUNLP license](https://github.com/thunlp/DocRED/blob/master/LICENSE) · [JEREX code and split preparation](https://github.com/lavis-nlp/jerex/tree/85851dfcbb485d9bfa2bf662b8073e826c760d7f) · [Hosted split directory](https://lavis.cs.hs-rm.de/storage/jerex/public/datasets/docred_joint/)

From an Anno checkout, use the authorized external data and pinned model environments documented in the baseline. Set the absolute paths below to your downloaded inputs. Run the JEREX commands with its pinned upstream checkout on `PYTHONPATH`, then switch to the GLiNER2 2.2.0 CPU environment for the three JointIE runs. The checkers use the Python standard library. [Model setup and pins](joint-stage-baseline.md)

```bash
qa="$PWD/scripts/qa/joint_baseline"
out=$(mktemp -d)
gold=/absolute/path/to/test_joint.json
test=/absolute/path/to/ai-test.json
train=/absolute/path/to/ai-train.json
jerex=/absolute/path/to/docred-joint
jointie=/absolute/path/to/gliner2.5-small-v1

python "$qa/jerex_loss.py" "$jerex" "$gold" "$out/jerex.json" --documents 32 --predictions "$out/writer.json"
python "$qa/jerex_predictions.py" "$gold" "$out/writer.json" "$out/jerex.json" "$out/predictions.json"
python "$qa/../audit_jerex_edges.py" "$gold" "$out/writer.json" --first 32 --trace "$out/jerex.json" --output "$out/audit.json"
python "$qa/aggregate_receipt.py" jerex-stages "$out/jerex.json" "$qa/receipts/jerex_docred32.json"
python "$qa/aggregate_receipt.py" jerex-predictions "$out/predictions.json" "$qa/receipts/jerex_predictions32.json"

python "$qa/jointie_coverage.py" "$test" "$jointie" "$out/short.json" --documents 64 --window-words 16 --overlap-words 4 --loss-trace
python "$qa/jointie_coverage.py" "$test" "$jointie" "$out/wide.json" --documents 64 --window-words 384 --overlap-words 64 --loss-trace
python "$qa/jointie_coverage.py" "$test" "$jointie" "$out/typed.json" --documents 64 --window-words 384 --overlap-words 64 --loss-trace --training-routes "$train"
python "$qa/check_jointie_loss.py" "$test" "$out/short.json" --previous "$qa/receipts/jointie_16.json"
python "$qa/check_jointie_loss.py" "$test" "$out/wide.json" --previous "$qa/receipts/jointie_384.json"
python "$qa/check_jointie_loss.py" "$test" "$out/typed.json"
python "$qa/aggregate_receipt.py" jointie "$out/short.json" "$qa/receipts/jointie_ai64_w16.json"
python "$qa/aggregate_receipt.py" jointie "$out/wide.json" "$qa/receipts/jointie_ai64_w384.json"
python "$qa/aggregate_receipt.py" jointie "$out/typed.json" "$qa/receipts/jointie_ai64_training_routes.json"
```

The public `full_receipt_sha256` hashes compact UTF-8 JSON with sorted object keys and preserved array order. A change to an individual annotation or prediction changes this digest even when aggregate counts agree. The check requires the full local receipt. A mismatch on a different model runtime requires inspecting the local evidence before claiming exact reproduction.
