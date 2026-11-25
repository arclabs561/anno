# Independent checks of extraction scores

Independent scorers rebuild labels from original corpus records and inspect saved model outputs. The source units differ across these checks. CrossRE labels ordered pairs of annotated mentions in one sentence, DocRED labels ordered pairs of document entities, and MultiNERD labels token sequences. A number from one unit cannot be substituted for another.

## Sentence relations and prompts

Eight saved native runs on the full CrossRE AI and literature test files were checked against source JSONL. The 431 AI sentences contain 1,164 reference relations and 105 sentences without a relation. The 416 literature sentences contain 1,623 reference relations and 47 sentences without one. Recomputing micro F1 as twice the exact match count divided by predicted plus gold reproduces all eight results. All arms used predicted endpoints. The four requested TPLinker arms executed the heuristic producer on every sentence. The document-gold prompt disclosed the absence of relations on empty sentences. These checks validate the saved numerators against their published denominators and F1 formula. The native receipts omit individual predicted triplets. [Full native prompt comparison](crossre-full-prompt-comparison.md)

For JointIE, the checker rebuilds every typed directed gold edge from inclusive token indices, maps it to the exact source Unicode string, and recomputes each shared-window test. It verifies document identity and text hash, every recorded edge's window list and the sum of the per-document fields. The checkpoint trace supplies the proposal and decoding observations. The checker does not rerun its optimizer. [Independent checker](../../scripts/qa/audit_joint_candidates.py) · [Checkpoint replay](joint-stage-baseline.md)

## Document-level relations

The JEREX checker reads the authors' prediction format. It verifies title, token sequence and mention indices, converts document-local entity IDs to exact sets of token spans, and checks typed and untyped matches independently. Duplicate entity memberships make identity ambiguous. The checker records those documents and their indexed edge counts separately. Detected spans count even when no predicted entity contains them. A labeled edge may match its head and tail cluster memberships and predicate while failing on an endpoint entity type. [Independent checker](../../scripts/qa/audit_jerex_edges.py) · [Scorer tests](../../scripts/qa/test_audit_jerex_edges.py)

On the original three documents, all memberships were identifiable. The independent matches / predicted / reference counts were 99/106/104 for mentions, 52/58/56 for within-cluster mention pairs, 72/80/79 for exact clusters, 68/80/79 for typed clusters, 16/56/72 for directed entity pairs, 17/58/76 for predicate-labeled edges and 16/58/76 when the endpoint types also had to match. [Original stage receipt](../../scripts/qa/joint_baseline/receipts/jerex.json)

## Typed entity spans

The NuNER checker rebuilds the three prompted label classes from the MultiNERD BIO tags and original token strings. It takes the best of the three raw word logits, applies the cutoff in logit space, groups consecutive winning words, applies the runtime's documented span-tail cleanup and counts exact half-open character spans with types inside each case. It checks the entire saved prediction ledger at 0.5 and the per-group integer TP, FP and FN at 0.3, 0.5 and 0.7. [Independent checker](../../scripts/qa/audit_nuner_spans.py)

## Reproduction

Use each benchmark's preparation and inference commands to obtain the raw files. These score checks need only the Python standard library, except the NuNER checker which also requires NumPy.

```sh
python3 scripts/qa/audit_joint_candidates.py ai-test.json jointie_16.json jointie_384.json
python3 scripts/qa/audit_jerex_edges.py test.json predicted.json --first 3 --output counts.json
python3 scripts/qa/audit_nuner_spans.py /path/to/prepared-multinerd /path/to/nuner-receipt.json
```
