# Anno backend error analysis

The six reviewed workstreams and their source reports are collected in [Anno PR #375](https://github.com/arclabs561/anno/pull/375). These scripts unpack individual LitBank and CrossRE residuals from replays of the published native evaluations. Their outputs include character offsets, source hashes and error counts. [Results](../../../docs/eval/backend-residuals.md) Original passages and model weights remain in their respective upstream distributions.

## Reading the two ledgers

**LitBank** uses the ten annotated excerpts in fold 0 and the original English text edition. A link here is an unordered pair of distinct mentions in one emitted cluster. It therefore counts every pair implied by a cluster of three or more mentions. These pair counts differ from CoNLL F1 and from the number of decisions a mention ranker makes. Gold singletons still contribute to mention coverage. Rows are keyed by document ID and SHA-256 of the analyzed text. Missing annotated pairs and extra emitted pairs are stored separately with half-open Unicode character offsets. A missing output endpoint can result from mention proposals, output policies, or clustering. An unannotated output span can reflect the corpus's restricted entity categories.

The native, FCoref and Maverick raw-text arms scored the same ten works. Native supplied-mention mode also scored ten. Maverick supplied mode declined *Dear Enemy* because three supplied boundaries split tokens. Its separate ledger records this refusal and scores the other nine. Each replay’s per-document original mention, MUC, B³ and CEAF counts had to match the published run before its residuals were accepted. [Coreference scoring](https://github.com/arclabs561/anno/blob/9973b2a5/docs/eval/coref-panels.md)

**CrossRE** uses the complete AI and literature test splits, the original seeded Anno benchmark selection, and both training-inventory and document-gold prompt policies. Every arm predicts its own endpoints. Each document’s gold and emitted directed relations form multisets keyed by ordered endpoint spans and case-insensitive predicate, preserving duplicate multiplicity. Exact matching follows the benchmark's choice to ignore endpoint type strings. Individual residuals contain head and tail character offsets, predicate, multiplicity, and the first applicable description in the order below. Rows use SHA-256 of the text edition to locate the original. The model's output and annotations may disagree even for a linguistically reasonable prediction.

1. The same exact relation exists beyond its multiplicity on the opposite side.
2. The same predicate connects the exact endpoints in reverse.
3. Another predicate connects the exact endpoints forward, or only in reverse.
4. Both directed endpoint spans overlap, with either the same predicate or a different one. Overlap here means at least one shared character at both endpoints.
5. The two exact spans occur elsewhere in emitted/annotated relations, only one occurs, or neither occurs.
6. The document has no opposite-side relation at all.

These are independent descriptions of each missed or extra relation against the opposite set. An overlapping extra and miss may describe the same boundary disagreement. The descriptions do not constitute a one-to-one replacement matching. The ledger observes final relation endpoints. It cannot identify an unreturned mention proposal, recover a suppressed score, or establish the model's internal reason for omitting a relation. For the heuristic baseline the recorded producer was the heuristic, even though the benchmark selected the TPLinker backend name. [CrossRE scoring and original receipts](https://github.com/arclabs561/anno/blob/9973b2a5/docs/eval/crossre-full-prompt-comparison.md)

## Reproduction

1. Follow Anno's [original coreference replay](https://github.com/arclabs561/anno/blob/9973b2a5/docs/eval/coref-panels.md) for the pinned models, environment, original LitBank Parquet, example driver and five baseline JSON files. Invoke `replay_litbank.py --anno CHECKOUT --assets INPUT_DIRECTORY --driver DRIVER --output litbank-errors.json` in the same offline environment.
2. Check out Anno commit `b1857609`, apply `crossre-row-capture.patch`, and build `anno-cli` with `--features eval` and its normal ONNX features. The patch instruments the native benchmark after inference and before scoring. It writes one JSONL record per sentence when `ANNO_DIAGNOSTIC_RELATION_ROWS` is set.
3. For each native CrossRE arm use the same CLI benchmark command, fixed-schema JSON if applicable, model revision and offline cache as the original run. Set `ANNO_RELATION_ORACLE_ENTITIES=0`, `ANNO_RELATION_TPLINKER_ORACLE_ENTITIES=0`, and seed 42. Set the row destination to `TRACE_DIR/{ai,literature}-{gliner_relex,tplinker}-{fixed,oracle}.rows` as appropriate and save the benchmark JSON at the same basename with suffix `.json`. Each run should request 431 AI or 416 literature examples.
4. Run `python classify_crossre.py TRACE_DIR ORIGINAL_DIR crossre-errors.json`. ORIGINAL_DIR contains the original native benchmark JSON files named in the form `ai-relex-fixed.json` or `literature-tplinker-oracle.json`. The script checks the evaluated sample (including the producer map), the gold, predicted and strict-match counts, and both residual totals for every arm.

The JointIE extra-prediction ledger can be regenerated by passing the pinned joint baseline receipts directory and the desired output path to `classify_jointie_extras.py`. It checks all 27, two, and four emitted relations for the three configurations. A missing reference relation does not adjudicate linguistic truth.

For LitBank the independent checker reconstructed the emitted mention sets and cluster partitions from the residual ledger and the pinned gold corpus. It reproduced the original per-document mention, MUC, B³ and CEAF statistics on all 49 scored documents and the explicit refusal. For CrossRE an independent scorer reassembled source sentences and gold spans from the two upstream splits and verified all 3,535 extra and 8,634 missed occurrences across 3,388 arm–sentence records.
