# Where native GAP links disappear

The raw native baseline returned the exact pronoun and correct name for **8 of 117 true GAP pairs**. It linked six. This trace reproduces all 128 published per-excerpt coverage and decision masks, including the four false links, before attributing any loss. It uses default English mention-ranking without a NER model. Only the excerpt ID and original source text reach the Rust resolver. The scorer joins gold boundaries and links afterward. [Original panel and model comparison](coref-panels.md) · [Aggregate trace receipt](native-gap-loss.json)

| First observed outcome for a true pair | Pairs | Evidence |
|---|---:|---|
| Exact name absent at proposal | 65 | All 117 exact pronouns were proposed. |
| Exact proposal removed during overlap deduplication | 0 | Both exact endpoints survived whenever both were proposed. |
| Both exact endpoints retained, at least one absent from returned clusters | 44 | Neither an outgoing nor an incoming selected link touched each omitted endpoint. |
| Both endpoints returned in different clusters | 2 | Their character spans exist in the model output. |
| Both endpoints returned in the same cluster | 6 | Six correct positive link decisions. |

Among the 65 missing exact names, 41 were multiword names for which the detector proposed only subspans. In ten others, the detector included the English possessive clitic in the proposed span. Twelve had no overlapping proposals at sentence starts and two had none immediately after an opening parenthesis. These describe the observed source shapes and spans. A corrected boundary does not establish that the ranker would link the name to the right pronoun. GAP leaves most other spans and relations unlabeled.

The raw default uses a left-to-right linker, a 100-character distance cap and type-specific antecedent limits. Its returned clusters contain only mentions participating in a selected link. The trace records the scored candidates, the effective threshold and the selected antecedent using the same production ranking routine. Looking separately at the direct edge between the two gold endpoints after deduplication yields 35 pairs scored at or below threshold, 11 beyond the distance cap, and six selected. Within the 44 pairs missing a returned endpoint, 34 direct edges scored at or below threshold and ten lay beyond the distance cap. Of the two pairs returned separately, one fell into each category. Either endpoint can be the later mention. An indirect path could also put the endpoints in one cluster. For an omitted endpoint, the observed absence of a qualifying earlier candidate or a selected later link does not identify which linguistic feature ought to change. No threshold adjustment follows from these counts.

To reproduce the baseline, use the version of the driver and native resolver in the commit accompanying this trace, and the pinned GAP test TSV from the [panel replay](coref-panels.md#replay).

```sh
python3 scripts/qa/native_gap_loss.py /path/to/gap-test.tsv /tmp/native-gap-loss.json
cmp /tmp/native-gap-loss.json docs/eval/native-gap-loss.json
```

The public receipt records aggregate counts and SHA-256 digests of the ordered pair outcomes and false-link paths. Add `--local-details /tmp/gap-trace-details.json` to inspect individual decisions and paths locally. [GAP source and rights](gap-decision-audit.md#source-and-rights)

## Which failures are visible in the returned clusters?

On GAP, each decision asks whether one designated name belongs in the pronoun’s cluster. Among missed true links, the saved panel masks distinguish missing exact output boundaries from endpoints returned in separate clusters. Counts below refer to decisions across the same 128 excerpts. [Frozen five-bit output masks](coref-panels.json) · [Reconciliation code](../../scripts/qa/native_gap_loss.py)

| Backend / input | Only name absent | Only pronoun absent | Both absent | Both returned separately | Total missed |
|---|---:|---:|---:|---:|---:|
| Native / raw | 54 | 7 | 48 | 2 | 111 |
| FCoref / raw | 37 | 0 | 4 | 3 | 44 |
| Maverick / raw | 40 | 0 | 0 | 1 | 41 |
| Native / supplied | 0 | 0 | 0 | 117 | 117 |
| Maverick / supplied | 2 | 1 | 1 | 0 | 4 |

FCoref's six false links comprise two wrong-only decisions where the correct name was absent, two where it was visible, one linking both names and one where neither candidate was gold. Maverick’s one raw false link linked the wrong name while the correct name was visible. FCoref also made that error. On supplied spans Maverick emitted 13 false links. Of those, nine linked both names in positive cases, two linked a wrong name while omitting the correct name, and two linked both candidates in the same negative case. These counts describe exact output spans. The historical neural receipts omit internal proposals, antecedent logits, pruning decisions and cluster paths, so the model-internal reasons remain unknown. Maverick’s supplied run received all three spans and emitted only linked mentions.

On native raw text, three of four false links attached the wrong name while omitting the correct name. The fourth linked B in a neither-name case. All four were direct selections of the preceding wrong name. No intermediate cluster member connected the endpoints. The supplied native resolver made one false link, with the correct name visible. Its saved output does not identify a deciding feature or rank margin.

## Removing English possessive clitics from proposed names

Ten of the diagnostic panel's missing name proposals included their following possessive marker. We froze one change to the English proper-name heuristic before scoring new excerpts. After its existing trailing-punctuation cleanup, the candidate strips a final ASCII or curly apostrophe followed by lowercase `s`. It preserves internal apostrophes, as in `O'Neil's`, and leaves ranking, multiword detection and sentence-initial detection alone. The fixture includes a character-boundary regression covering both apostrophe forms and Unicode names. [Frozen candidate](../../scripts/qa/fixtures/native-gap-possessive.patch)

All 454 rows in the upstream validation split qualified for evaluation. They represent 328 distinct article URLs, none shared with the full test split. Every row contains the same original text in each arm. The resolver receives no gold boundaries. [Validation split](https://github.com/google-research-datasets/gap-coreference/blob/83135f26025cdd705c04f77e940d0c9201116b80/gap-validation.tsv) · [Aggregate results and change digest](native-gap-possessives.json)

| Validation measure | Baseline | Possessive candidate |
|---|---:|---:|
| True pairs with both exact boundaries proposed | 211/392 | 238/392 |
| True pairs with both exact boundaries in output clusters | 48/392 | 50/392 |
| Correct pronoun–name links | 27 | 27 |
| Incorrect pronoun–name links | 40 | 41 |
| Distinct returned mentions, summed by excerpt | 2,691 | 2,798 |
| Exact two-label decisions | 80/454 | 80/454 |

On the separately sealed 128-excerpt test cohort with fresh article URLs, returned gold pairs increased from 11 to 14 of 118. Correct links increased from seven to eight and false links stayed at two. The ten LitBank excerpts had identical raw native mention and cluster scores in both arms, including 71 exact mention matches among 118 outputs and CoNLL F1 of 0.314809. The validation panel shows a proposal gain, a smaller gain in observable linked endpoints and no net gain in correct links. GAP cannot classify most additional output mentions as correct or false. Lexicalized names can include a final apostrophe-s. These results support retaining the existing default and keeping clitic handling as a testable candidate.

To replay the candidate, build the native test executable at the baseline trace revision and copy the executable path printed by Cargo to a temporary directory. In a separate worktree at the same revision, apply the [candidate patch](../../scripts/qa/fixtures/native-gap-possessive.patch), run the mention-ranking tests and copy that test executable as well. Run the scorer with the two pinned GAP TSVs and the two binaries.

```sh
cargo test --locked -p anno --lib --no-default-features --no-run
# In the candidate worktree:
git apply scripts/qa/fixtures/native-gap-possessive.patch
cargo test --locked -p anno --lib --no-default-features mention_ranking
python3 scripts/qa/native_gap_possessives.py gap-test.tsv gap-validation.tsv \
  /tmp/baseline-test /tmp/candidate-test /tmp/native-gap-possessives.json
cmp /tmp/native-gap-possessives.json docs/eval/native-gap-possessives.json
```

In the optional local details, the three proposal bits in each changed validation row refer to the pronoun, A and B. The five output bits add the two pronoun–candidate link decisions to those three returned-endpoint bits. A changed row can reflect a change in proposal coverage even when its emitted decisions match exactly.

## FCoref scores for competing mention boundaries

On the 256 previously frozen GAP validation and test excerpts, the raw-text
FCoref CPU replay first reproduced all 256 saved stock cluster partitions.
The model retained (K = ceil(tokens / 4)) candidate spans per document.
Of 67 misses with an absent exact antecedent, 53 had a nested or containing
span in the pronoun's returned cluster. For each of those 53, we took the
best pre-pruning mention logit that mapped to the exact source span and the
best logit of any such linked rival. Multiple subtoken candidates can map
to one trimmed source span. [Frozen panel, stock evidence and cohort IDs](https://github.com/arclabs561/anno/issues/288#issuecomment-7176979503)

| Measured over these 53 pairs | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Exact target's rank below the stock cutoff | 13 | 524 | 2,469 |
| Stock cutoff logit minus exact target logit | 0.61 | 8.65 | 18.82 |
| Best linked rival logit minus exact target logit | 3.16 | 11.88 | 22.38 |

Doubling the rounded stock candidate count admitted just two of those exact
targets. Neither linked to its pronoun. All 512 labeled A/B decisions across
the 256 excerpts stayed identical, with 131 true and 11 false links in each
arm. The competition occurs well before final cluster selection for most of
these misses. These results favor investigating span representation and
training boundary labels before increasing the default candidate budget.

The [aggregate rank receipt](fcoref-boundary-ranks.json) preserves the per-split confusion counts and unrounded rank and score gaps.

The diagnostic used the original FP32 encoder (SHA-256
`190101f8e3b927bc7fb5e053f63fb82a85ef7e92dc2acd3a23a5ad0dd58553f4`),
CPU ONNX Runtime and the Anno Rust mention/antecedent scorer. We enumerated
all legal spans of up to 30 tokens, retained their stable descending score
order to measure rank, restored source order before antecedent scoring, and
used the stock null and union-find decisions for both budgets. The separate
[public runner comparison](../../scripts/qa/fcoref_proposal_budget.py)
uses `ceil(tokens / 2)` on different articles. That differs from exactly
twice the rounded stock count by at most one candidate.

A [follow-up on 256 separate English articles](fcoref-emitted-boundaries.md) tests whether names already emitted elsewhere in a cluster can repair an exact boundary.

### Reproduce the boundary-rank table

The external inputs are GAP's original test and validation TSV files, the
local FP32 export identified above (including its tokenizer and scorer), and a
plain-text list of the ordered GAP IDs. Create that list from the two number
blocks in the [frozen cohort](https://github.com/arclabs561/anno/issues/288#issuecomment-7176979503):
start with `test-1845`, prefix each of the next 96 numbers with `validation-`,
then prefix the final 160 with `test-`. Whitespace separates IDs. The replay
checks both TSV digests and the digest of the ordered IDs. It omits the first
case, which was used to form the hypothesis.

In a disposable checkout at this revision, apply the [study readout patch](../../scripts/qa/fixtures/fcoref-boundary-ranks.patch)
and build the normal Rust example. The patch exposes the pre-pruning rank
table and permits doubling the integer candidate budget. Store input, logits
and output clusters in an external private directory. The final scorer reads
these local files and prints aggregate counts, verifying a commitment to the
256 stock partitions archived in the original run. Install Python
`tokenizers` for the final tokenizer-offset projection.

```sh
git apply scripts/qa/fixtures/fcoref-boundary-ranks.patch
cargo build --locked -p anno --features onnx --example coref_gap_panel
export FCOREF_MODEL_PATH=/path/to/the/verified/fp32/export
qa=scripts/qa/fcoref_boundary_ranks.py
private=/path/to/external/private/replay
runner=target/debug/examples/coref_gap_panel
python3 "$qa" "$private/ids.txt" "$private/test.tsv" "$private/validation.tsv" \
    --prepare-input "$private/input.jsonl"
ANNO_FCOREF_BOUNDARY_RANKS=1 "$runner" fcoref < "$private/input.jsonl" \
    > "$private/stock.jsonl" 2> "$private/ranks.log"
ANNO_FCOREF_DOUBLE_K=1 "$runner" fcoref < "$private/input.jsonl" > "$private/double.jsonl"
python3 "$qa" "$private/ids.txt" "$private/test.tsv" "$private/validation.tsv" \
    --stock "$private/stock.jsonl" --double "$private/double.jsonl" \
    --trace "$private/ranks.log" --tokenizer "$FCOREF_MODEL_PATH/tokenizer.json"
```
