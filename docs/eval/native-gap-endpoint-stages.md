# Where the native raw-text ranker loses GAP pronoun links

Across the three official English GAP splits, 3,964 marked pronoun–name pairs refer to the same person. The stock native ranker linked 186. In 2,030 pairs it never proposed the exact name. In another 1,584 it retained both exact spans but left at least one isolated. An application that sees only the returned clusters cannot distinguish these two outcomes. [Aggregate receipt](native-gap-endpoint-stages.json) · [Original 128-excerpt investigation](native-gap-loss.md)

| Split | Excerpts | Gold pairs | Missing exact proposal | Removed by overlap | Isolated endpoint | Separate clusters | Linked |
|---|---:|---:|---:|---:|---:|---:|---:|
| Development | 2,000 | 1,799 | 921 | 0 | 725 | 79 | 74 |
| Test | 2,000 | 1,773 | 928 | 0 | 696 | 64 | 85 |
| Validation | 454 | 392 | 181 | 0 | 163 | 21 | 27 |
| **Total** | **4,454** | **3,964** | **2,030** | **0** | **1,584** | **164** | **186** |

Each gold pair appears once, at its first loss or successful link. Every marked gold pronoun received an exact proposal. Overlap deduplication removed three exact pronouns, one per split. All three paired names were already missing at proposal, which explains the zero in the overlap column. An isolated endpoint has neither a selected earlier antecedent nor a later mention that selects it. The raw-text resolver emits only components with at least two mentions. The receipt also reports both historical 128-excerpt cohorts separately. Each belongs to the official test split.

Among the 2,030 missing exact names, 1,175 were multiword names with only proper subspans proposed. In 250 cases a proposal included the possessive clitic. Another 521 had no overlapping proposal at the start of the excerpt or after terminal punctuation, 11 had none immediately after an opening parenthesis, and 73 had other patterns. Testing a boundary repair requires rerunning the ranker because a new mention can change candidate competition or the overlap decision.

For the 1,934 pairs with both endpoints retained, the later endpoint selected its labeled partner directly in 178 cases. The direct score was at or below its threshold in 1,433, the endpoints were outside the distance cap in 291, and a different antecedent won in 32. Eight of the 186 correct links therefore used an indirect path. Among the 173 links to a labeled wrong name, 163 were direct and ten used an intervening mention. GAP labels two names per excerpt and leaves other relations unjudged.

## Replay and verification

The test-only Rust trace records the production proposal, overlap and antecedent routines. Only the ID and raw text enter the resolver. The replay checks which proposals survive overlap deduplication. For each retained mention, it verifies the scored antecedent indices, the strict threshold and the nearest-first tie rule. It reconstructs components from the selected edges, filters isolated mentions and compares every emitted span and component to the resolver output. We also reran a separately built public worker on all 4,454 excerpts. Every complete partition matched both the worker and the previously frozen fingerprints. The data were already examined in earlier experiments. [Trace](../../crates/anno/src/backends/coref/mention_ranking/algorithm/gap_trace.rs) · [Public worker fingerprints](native-gap-neutral-mention.json)

Use the three pinned TSVs in a local directory named by `GAP`. The receipt contains counts and fingerprints. The synthetic tests use project-authored fixtures, including two traces with identical empty output but different loss stages and a changed direct edge that preserves its component. [Source revision](https://github.com/google-research-datasets/gap-coreference/tree/83135f26025cdd705c04f77e940d0c9201116b80) · [GAP and Wikipedia attribution and rights](gap-decision-audit.md#source-and-rights) · [Synthetic controls](../../scripts/qa/test_native_gap_endpoint_stages.py)

```sh
python3 -m unittest discover -s scripts/qa -p 'test_native_gap_endpoint_stages.py'
python3 scripts/qa/native_gap_endpoint_stages.py "$GAP" /tmp/native-gap-stages.json
cmp /tmp/native-gap-stages.json docs/eval/native-gap-endpoint-stages.json
```

To compare each complete partition with a fresh public worker too, build the existing example with ONNX Runtime available. The native backend does not load model weights.

```sh
cargo build --locked -p anno --no-default-features --features onnx --example coref_gap_panel
python3 scripts/qa/native_gap_endpoint_stages.py "$GAP" /tmp/native-gap-stages.json \
    --stock "${CARGO_TARGET_DIR:-target}/debug/examples/coref_gap_panel"
cmp /tmp/native-gap-stages.json docs/eval/native-gap-endpoint-stages.json
```
