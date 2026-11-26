# Anno backend error analysis

These scripts compute per-document diagnostics from authorized local inputs. Keep inputs, predictions and the resulting ledgers in an external working directory. The published report contains aggregate results. [Results](../../../docs/eval/backend-residuals.md)

## Reading the two ledgers

**LitBank** uses the ten annotated excerpts in fold 0 and the original English text edition. A link here is an unordered pair of distinct mentions in one emitted cluster. It therefore counts every pair implied by a cluster of three or more mentions. These pair counts differ from CoNLL F1 and from the number of decisions a mention ranker makes. Gold singletons still contribute to mention coverage. The default LitBank receipt contains aggregate counts, asset identities and a digest for each arm. The optional local ledger keys each row by document ID and text-edition SHA-256. It records pair and span residuals using half-open character offsets. A missing output endpoint can result from mention proposals, output policies, or clustering. An unannotated output span can reflect the corpus's restricted entity categories.

The native, FCoref and Maverick raw-text arms scored the same ten works. Native supplied-mention mode also scored ten. Maverick supplied mode declined *Dear Enemy* because three supplied boundaries split tokens. Its separate ledger records this refusal and scores the other nine. Each replay’s per-document original mention, MUC, B³ and CEAF counts had to match the published run before its residuals were accepted. [Coreference scoring](../../../docs/eval/coref-panels.md)

**CrossRE** uses the complete AI and literature test splits, the original seeded Anno benchmark selection, and both training-inventory and document-gold prompt policies. Every arm predicts its own endpoints. Each document’s gold and emitted directed relations form multisets keyed by ordered endpoint spans and case-insensitive predicate, preserving duplicate multiplicity. Exact matching follows the benchmark's choice to ignore endpoint type strings. Individual residuals contain head and tail character offsets, predicate, multiplicity, and the first applicable description in the order below. Rows use SHA-256 of the text edition to locate the original. The model's output and annotations may disagree even for a linguistically reasonable prediction.

1. The same exact relation exists beyond its multiplicity on the opposite side.
2. The same predicate connects the exact endpoints in reverse.
3. Another predicate connects the exact endpoints forward, or only in reverse.
4. Both directed endpoint spans overlap, with either the same predicate or a different one. Overlap here means at least one shared character at both endpoints.
5. The two exact spans occur elsewhere in emitted/annotated relations, only one occurs, or neither occurs.
6. The document has no opposite-side relation at all.

These are independent descriptions of each missed or extra relation against the opposite set. An overlapping extra and miss may describe the same boundary disagreement. The descriptions do not constitute a one-to-one replacement matching. The ledger observes final relation endpoints. It cannot identify an unreturned mention proposal, recover a suppressed score, or establish the model's internal reason for omitting a relation. For the heuristic baseline the recorded producer was the heuristic, even though the benchmark selected the TPLinker backend name. [CrossRE scoring and original receipts](../../../docs/eval/crossre-full-prompt-comparison.md)

## Reproduction

Prepare the pinned LitBank Parquet and model assets from the [coreference replay](../../../docs/eval/coref-panels.md#replay). Build the `coref_gap_panel` example from this checkout and use its restricted Maverick worker with the verified local safetensors. Run `python replay_litbank.py --assets INPUT_DIRECTORY --driver DRIVER --output /tmp/litbank-summary.json`. Add `--local-details /tmp/litbank-errors.json` to keep the character offsets on your own machine. The scorer validates each work against the tracked panel receipt. LitBank annotations are by Bamman, Lewke and Mansoor under CC BY 4.0. Its source works have their own editions and attribution. [Annotations and license](https://github.com/dbamman/litbank/tree/3e50db0ffc033d7ccbb94f4d88f6b99210328ed8)

### CrossRE

From the current Anno checkout, create a separate historical checkout for the observation patch. Use the pinned source files and model cache from the [original full-split study](../../../docs/eval/crossre-full-prompt-comparison.md#reproduce-the-comparison). The upstream CrossRE repository publishes GPLv3 and gives no separate annotation terms. The patched benchmark writes the source text hash, spans and predicates to a new external file for each arm. The classifier checks the ordered sample, producer and counts against the eight published arms. [CrossRE source terms](https://github.com/mainlp/CrossRE/blob/a58885fa760559d2dc9e176e730fe731f020d6f3/LICENSE)

```bash
qa="$PWD/scripts/qa/backend_residuals"
work=$(mktemp -d)
# Four authorized, pinned DOMAIN-{train,test}.json files in an external directory.
data=/absolute/path/to/crossre-inputs
git worktree add --detach "$work/capture" b185760926a3f8b85a449336d9ddd95a259f960b
git -C "$work/capture" apply "$qa/crossre-row-capture.patch"
cargo build --locked --manifest-path "$work/capture/Cargo.toml" -p anno-cli --features eval,onnx --bin anno --target-dir "$work/target"
cpus=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:2])))')
python3 "$work/capture/scripts/qa/crossre_prompt_replay.py" --anno "$work/target/debug/anno" \
  --data "$data" --out "$work/arms" --cpus "$cpus"
python3 "$qa/classify_crossre.py" "$work/arms" "$work/crossre-errors.json"
```

### JointIE

Run the [three JointIE replays](../../../docs/eval/joint-edge-loss.md#source-terms-and-replay) and retain their external output directory as `out`. Pass those full traces to the classifier below. It verifies each against the full-receipt digest in the corresponding public aggregate, then checks all 27, two and four predictions respectively. The resulting private ledger contains source-derived span and predicate annotations. An unmatched edge establishes disagreement with the reference.

```bash
python3 scripts/qa/backend_residuals/classify_jointie_extras.py "$out/short.json" "$out/wide.json" \
  "$out/typed.json" "$out/jointie-extras.json"
```

For LitBank the independent checker reconstructed the emitted mention sets and cluster partitions from the residual ledger and the pinned gold corpus. It reproduced the original per-document mention, MUC, B³ and CEAF statistics on all 49 scored documents and the explicit refusal. For CrossRE an independent scorer reassembled source sentences and gold spans from the two upstream splits and verified all 3,535 extra and 8,634 missed occurrences across 3,388 arm–sentence records.
