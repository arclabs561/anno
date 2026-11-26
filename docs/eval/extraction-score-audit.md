# Independent checks of extraction scores

Independent scorers rebuild labels from original corpus records and inspect saved model outputs. The source units differ across these checks. CrossRE labels ordered pairs of annotated mentions in one sentence, DocRED labels ordered pairs of document entities, and MultiNERD labels token sequences. A number from one unit cannot be substituted for another.

## Sentence relations and prompts

Eight saved native runs on the full CrossRE AI and literature test files were checked against source JSONL. The 431 AI sentences contain 1,164 reference relations and 105 sentences without a relation. The 416 literature sentences contain 1,623 reference relations and 47 sentences without one. Recomputing micro F1 as twice the exact match count divided by predicted plus gold reproduces all eight results. All arms used predicted endpoints. The four requested TPLinker arms executed the heuristic producer on every sentence. The document-gold prompt disclosed the absence of relations on empty sentences. These checks validate the saved numerators against their published denominators and F1 formula. The native receipts omit individual predicted triplets. [Full native prompt comparison](crossre-full-prompt-comparison.md)

For JointIE, the checker rebuilds every typed directed gold edge from inclusive token indices, maps it to the exact source Unicode string, and recomputes each shared-window test. It verifies document identity and text hash, every recorded edge's window list and the sum of the per-document fields. The checkpoint trace supplies the proposal and decoding observations. The checker does not rerun its optimizer. [Independent checker](../../scripts/qa/audit_joint_candidates.py) · [Checkpoint replay](joint-stage-baseline.md)

Across the first 64 AI test sentences, the short-window policy put 86 of 161 distinct gold edges inside at least one window. The 384-word policy put all 161 inside a window. The larger window still yielded only 22 exact edge candidates and two decoded predictions, both incorrect. Independently checking the returned mention and edge keys against source annotations gave the following counts.

| JointIE policy | Gold edges in a shared window | Exact gold edges in scored candidates | Matched / decoded / gold |
|---|---:|---:|---:|
| 16 words, 4 overlap | 86 | 23 | 1 / 27 / 161 |
| 384 words, unrestricted endpoint types | 161 | 22 | 0 / 2 / 161 |
| 384 words, training-derived endpoint types | 161 | 16 | 0 / 4 / 161 |

For the restricted policy, rebuilding the head and tail types permitted per predicate from the 100 training sentences verified every schema-eligibility decision. The predicate inventory omitted two gold edges and the endpoint-type restrictions excluded 51 more. [JointIE stage receipts](../../scripts/qa/joint_baseline/receipts)

## Document-level relations

The JEREX checker reads the authors' prediction format. It verifies title, token sequence and mention indices, converts document-local entity IDs to exact sets of token spans, and checks typed and untyped matches independently. Duplicate entity memberships make identity ambiguous. The checker records those documents and their indexed edge counts separately. Detected spans count even when no predicted entity contains them. A labeled edge may match its head and tail cluster memberships and predicate while failing on an endpoint entity type. [Independent checker](../../scripts/qa/audit_jerex_edges.py) · [Scorer tests](../../scripts/qa/test_audit_jerex_edges.py)

On the original three documents, all memberships were identifiable. The independent matches / predicted / reference counts were 99/106/104 for mentions, 52/58/56 for within-cluster mention pairs, 72/80/79 for exact clusters, 68/80/79 for typed clusters, 16/56/72 for directed entity pairs, 17/58/76 for predicate-labeled edges and 16/58/76 when the endpoint types also had to match. [Original stage receipt](../../scripts/qa/joint_baseline/receipts/jerex.json)

The expanded 32-document run had 32 identifiable documents. Independent counts from the source annotations and native predictions reproduced all seven measurement rows. All 406 source edges also agreed with the checkpoint trace on whether their endpoint mentions were present and whether the labeled edge was returned.

| DocRED observation | Matched | Predicted | Gold |
|---|---:|---:|---:|
| Detected mentions | 831 | 899 | 891 |
| Within-cluster mention pairs | 315 | 383 | 378 |
| Exact clusters | 584 | 696 | 686 |
| Typed exact clusters | 562 | 696 | 686 |
| Directed entity pairs | 135 | 331 | 383 |
| Directed predicate-labeled edges | 140 | 344 | 406 |
| Labeled edges with typed endpoints | 139 | 344 | 406 |

The execution trace attributed 110 of the 266 missing gold edges to an incorrect cluster membership, 99 to the relation threshold, 56 to the mention threshold and one to mention enumeration. The output-only check establishes the missing mentions and incorrect edges. A claim about a particular threshold uses the separately captured checkpoint logits. [Checkpoint trace and replay](joint-stage-baseline.md)

## Typed entity spans

The NuNER checker rebuilds the three prompted label classes from the MultiNERD BIO tags and original token strings. It takes the best of the three raw word logits, applies the cutoff in logit space, groups consecutive winning words, applies the runtime's documented span-tail cleanup and counts exact half-open character spans with types inside each case. It checks the entire saved prediction ledger at 0.5 and the per-group integer TP, FP and FN at 0.3, 0.5 and 0.7. [Independent checker](../../scripts/qa/audit_nuner_spans.py)

Both exported graphs completed all 38 cases. For each graph the checker reproduced all saved 0.5 span sets and all counts at the three cutoffs. Results below are true positives / false positives / false negatives at 0.5.

| MultiNERD cohort | Cases | FP32 | U8 |
|---|---:|---:|---:|
| English, hash-selected | 24 | 18 / 28 / 7 | 18 / 28 / 7 |
| English, Unicode-enriched | 12 | 10 / 9 / 4 | 10 / 7 / 4 |
| English, 1,772-token pack | 1 | 51 / 12 / 13 | 52 / 10 / 12 |
| Russian, 4,033-token pack | 1 | 1 / 0 / 28 | 1 / 0 / 28 |

The two packed cases join contiguous source sentences. They demonstrate execution at those lengths and describe accuracy on those particular inputs. Language, sentence composition and length vary together.

## Reproduction

Use each benchmark's preparation and inference commands to obtain the raw files. These score checks need only the Python standard library, except the NuNER checker which also requires NumPy.

```sh
python3 scripts/qa/audit_joint_candidates.py ai-test.json jointie_16.json jointie_384.json
python3 scripts/qa/audit_jerex_edges.py test.json predicted.json --first 3 --output counts.json
python3 scripts/qa/audit_nuner_spans.py /path/to/prepared-multinerd /path/to/nuner-receipt.json
```
