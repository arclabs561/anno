# Native Anno predictions in the memory replay

The [adapter](adapter.py) runs Anno's CLI on the repository's authored person
panel. It uses the heuristic detector and native mention ranker. The receipt
includes the actual coreference run status and backend and rejects a fallback
to any other backend. The input to the memory replay is all contiguous signals
emitted by the pipeline, including detector signals that Anno left untracked
and additional signals emitted by native coreference. Native track IDs and
signal IDs are retained per document. The experimental replay creates a
separate partition whose cluster ID is the original signal ID that opened
that cluster.

The adapter checks every contiguous signal directly against its original
source text using half-open Unicode-scalar offsets. It reports distinct
emitted spans and separately retains every signal ID, including repeated
spans. Noncontiguous signals are identified separately and left out of
retrieval and span scoring. A native track containing an excluded member
keeps that member's ID and cannot count as an exact gold cluster. The
predicted partition uses the shared replay's memory and lexical prefix
features. Mentions enter in source-visible order by end, start and signal
ID. All ranking and cluster updates finish before gold is read for audit.
The same finished trace is scored twice using the two annotation boundaries.
Box support and the link threshold are raw ranking scores.

## Reproduce

Run from the Anno repository root, with the reference replay present in the
parent directory of this adapter and Python with NumPy and SciPy installed.

```sh
cargo build --locked -p anno-cli --no-default-features --features extractor-html2text

python -m unittest discover \
  -s crates/anno-eval/research/reference/anno_adapter -v

python crates/anno-eval/research/reference/anno_adapter/adapter.py \
  --anno target/debug/anno \
  --gold testdata/grounded/dual_span_people.json \
  --summary-only > /tmp/anno-dual-span-receipt.json
```

The committed [summary receipt](dual-span-receipt.json) used Anno
`b57d4c240af5e1e41874f056acfe62c7f817d8b1` (CLI 0.16.0), Rust 1.95.0
and NumPy 2.3.5. The binary hash identifies this particular executable and
can differ across rebuilds. To inspect decisions and verify ID alignment,
drop `--summary-only`. To save the original CLI JSON including source text,
add `--raw-output /tmp/anno-predictions.json`. Raw outputs and full traces
are intended for local inspection. Neither source text nor model assets are
copied into the checked-in receipt.

## What the receipt measures

Across 12 documents, Anno emitted 39 distinct contiguous spans. There were
43 contiguous signal IDs, including four repeated coordinates. Three gold
mentions have unresolved identity, so pairs use only the resolved mentions.

| Boundary | Covered gold mentions | Reachable gold positive pairs | Native joined positive | Native joined negative |
| --- | ---: | ---: | ---: | ---: |
| Compact highlight | 12 / 37 | 3 / 26 | 1 | 1 |
| Accepted reference | 18 / 37 | 6 / 26 | 1 | 1 |

At the fixed smoke threshold of −0.5, each policy uses 1,960 bytes for
seven resident entities. On the accepted-reference boundary:

| Partition | Correct linked pairs | False merges | Covered positive pairs split | Mixed known-identity clusters |
| --- | ---: | ---: | ---: | ---: |
| Native Anno | 1 | 1 | 5 | 1 |
| Exemplar replay | 1 | 3 | 5 | 2 |
| Midpoint replay | 1 | 4 | 5 | 3 |
| Box replay | 0 | 3 | 6 | 3 |

There were 20 positive pairs lost at mention detection under this boundary.
The box trace reports two candidate misses, four ranking misses and two
threshold abstentions among accepted-reference signals with an observed
distinct antecedent. Each signal is evaluated once in the trace; pair
counts are over distinct gold occurrences. A prior alternate span for the
same gold occurrence cannot create an antecedent opportunity.

- A **retention miss** has a previously observed co-referring occurrence,
  but none of its clusters remain resident. A **candidate miss** has a
  resident such cluster outside the returned shortlist. A **ranking miss**
  has one in the shortlist beneath a different top candidate.
- A **threshold abstention** occurs when the top ranked target contains a
  correct earlier occurrence and the score is below threshold. An accepted
  supported target can still contain a known wrong occurrence. The trace
  tallies wrong or impure accepted targets separately.
- **Exact gold cluster** requires every raw member to align to a distinct
  resolved occurrence for one gold identity and all its gold members to
  be present. Explicit negatives, unknown spans, unresolved mentions and
  excluded noncontiguous members disqualify exactness.
- Native counts summarize published tracks. Replay counts include a
  singleton for an unattached signal. The array capacity counts only the
  bounded memory, while the audit keeps an unbounded post-run trace.
  The corpus is an authored alignment diagnostic and the threshold was
  not fit on a development set. The outcome checks the adapter and
  exposes domain errors without establishing a model ranking.

## Audit of the existing track score

The adapter archives the raw wire value and performs no thresholding or
calibration based on it. All nine native tracks in the checked-in receipt
carry 1.0.

- **Producers.** [`Track::new`](../../../../../anno/src/core/grounded/mod.rs)
  writes `Confidence::ONE`. The [native mention-ranking producer](../../../../../anno/src/backends/coref/mention_ranking/types.rs)
  constructs its tracks with that initializer and supplies no measured
  whole-track score. Coreference clusters entering through the
  [CLI projection](../../../../../anno-cli/src/cli/utils.rs) call
  `create_track_from_signals`, which uses the same initializer. Fresh
  tracks created for a multi-track merge also start with the default.
  Tests use the public setter to place other values on the wire.
- **Consumers.** A source search found the public getter, serializable
  field and range tests. The human-output code in
  [`output.rs`](../../../../../anno-cli/src/cli/output.rs) computes a
  formatted string into `_cluster_conf` and never prints it. No decision
  consumer in the pinned Anno tree reads the field for association or
  catalog linking. External consumers are outside this inventory.
- **Legacy records.** The field is a transparent numeric
  [`Confidence`](../../../../../anno/src/core/confidence.rs) without a
  field-level serde default on `Track`. A numeric 1.0 can be authored by a
  caller or emitted by the constructor. These values cannot be distinguished
  from the number alone. Removing the field from a real emitted track
  caused `anno enhance FILE --quiet --format grounded` to fail during
  deserialization with `missing field cluster_confidence`. A future reader
  that supports missing values would need explicit compatibility policy
  and tests. The [track score decision record](../../../../../../docs/decisions/track-association.md)
  covers the separate events and score populations.
