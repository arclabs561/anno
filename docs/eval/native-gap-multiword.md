# Joining two native proper-name proposals

## Frozen comparison

The native GAP trace found 1,175 labeled pairs where the exact multiword name was missing and every overlapping proposal lay inside it. The English fallback detector checks one whitespace-delimited word at a time. This experiment adds one candidate at the proposal stage and runs the full ranker again. [Prior stage audit](native-gap-endpoint-stages.md)

For English input with NER disabled, take the fallback's existing proper-name tokens in source order. A token qualifies when its first character is uppercase and each remaining character is a lowercase letter. Join a maximal run of exactly two qualifying proposals whose source spans are separated by one ASCII space. The new span takes the second token's head and both original proposals stay available for the stock overlap, feature and scoring routines. Sentence-initial exclusions still apply to the constituent proposals. [Candidate and project-authored control](../../scripts/qa/fixtures/native-gap-multiword.patch)

Freeze the rule before scoring any candidate output. Run development first, then test and validation, with the same rule throughout. Count exact proposed, retained and returned gold endpoint pairs, correct and wrong links gained and lost, and additions and removals of emitted spans outside GAP's three marked coordinates. A split passes only if the candidate produces a net gain in correct links and introduces zero newly wrong decisions. Previously wrong decisions removed cannot offset newly wrong ones. All three official splits have been studied previously. Adopting a default would also require evidence on an independent source with annotations relevant to its proposed use.

## Results

The candidate recovered the exact names in 681 of the 1,175 targeted gold pairs. The full resolver gained 92 correct links, lost two, introduced 210 wrong links and removed three. Every split exceeded the frozen budget. Keep the existing default. [Full receipt](native-gap-multiword.json)

| Split | Target names recovered | Gold pairs with both proposals, stock → candidate | Gold pairs with both returned, stock → candidate | Correct links gained / lost | Wrong links gained / lost |
|---|---:|---:|---:|---:|---:|
| Development | 316 / 535 | 878 → 1,194 | 153 → 216 | 44 / 0 | 92 / 0 |
| Test | 303 / 538 | 845 → 1,148 | 149 → 207 | 42 / 2 | 101 / 1 |
| Validation | 62 / 102 | 211 → 273 | 48 → 60 | 6 / 0 | 17 / 2 |

All 92 newly correct decisions came from the targeted partial-name cohort. All 210 newly wrong decisions also involved a marked name whose exact span was missing from stock proposals. In 187 cases the candidate linked only the distractor. In 23 neither marked name was correct. Of the two lost correct links, one had an isolated endpoint and the other ended with both mentions in different clusters.

The additive proposal changed 1,983 of 4,454 emitted partitions. Overlap deduplication prefers the longer of two proposals with the same start. Sixteen previously retained gold pairs lost an exact endpoint. Among coordinates outside GAP's three annotations per excerpt, 2,031 appeared for the first time in the output and 4,190 disappeared. Even among unmarked spans present in both outputs, 22 pairs became co-clustered and 21 were separated. GAP does not label these unmarked changes.

The disposable candidate also fails one existing native ranking test. Its two dotted-I spelling variants produce the same offsets for both pronouns, but the test compares every emitted mention's surface. Joining the name now returns its original spelling in each variant, so those surfaces differ. The stock family passed all 129 active tests and skipped three. The candidate passed 129, including its new control, failed one and skipped three. [Dotted-I regression test](../../crates/anno/src/backends/coref/mention_ranking/algorithm/mod.rs)

## Reproduce

Use a disposable checkout and the three pinned GAP TSVs in a directory named by `GAP`. The test executable uses the native raw-text resolver, including its normal feature extraction, linking and cluster emission. Both arms receive only a document ID and raw text. The scorer checks all input file hashes, reconstructs each complete partition from antecedent decisions, and verifies every split's stock partition fingerprint and targeted cohort size against the prior audit. The public receipt holds aggregate counts and fingerprints. Excerpts, annotations and local per-document output retain their upstream terms. [Scorer](../../scripts/qa/native_gap_multiword.py) · [Pinned GAP files](https://github.com/google-research-datasets/gap-coreference/tree/83135f26025cdd705c04f77e940d0c9201116b80) · [Attribution and rights](gap-decision-audit.md#source-and-rights)

```sh
trial=$(mktemp -d)
build_worker() {
    cargo test --locked -p anno --no-default-features --lib --no-run \
        --message-format=json > "$trial/build.jsonl" || return 1
    worker=$(python3 -c 'import json,sys; print(next(r["executable"] for r in map(json.loads,sys.stdin) if r.get("executable")))' < "$trial/build.jsonl") || return 1
    cp "$worker" "$1"
}
build_worker "$trial/stock"
git apply scripts/qa/fixtures/native-gap-multiword.patch
build_worker "$trial/candidate"
"$trial/candidate" fallback_titlecase_pair_control
git apply -R scripts/qa/fixtures/native-gap-multiword.patch
python3 scripts/qa/native_gap_multiword.py "$GAP" "$trial/stock" "$trial/candidate" "$trial/result.json"
cmp "$trial/result.json" docs/eval/native-gap-multiword.json
```

We also verified the candidate's complete partitions against a separately built public worker. To include that check, build the example with the candidate patch applied in a checkout with ONNX Runtime available, then pass its path to the scorer.

```sh
git apply scripts/qa/fixtures/native-gap-multiword.patch
cargo build --locked -p anno --no-default-features --features onnx --example coref_gap_panel
cp "${CARGO_TARGET_DIR:-target}/debug/examples/coref_gap_panel" "$trial/public-candidate"
git apply -R scripts/qa/fixtures/native-gap-multiword.patch
python3 scripts/qa/native_gap_multiword.py "$GAP" "$trial/stock" "$trial/candidate" \
    "$trial/result.json" --public-candidate "$trial/public-candidate"
cmp "$trial/result.json" docs/eval/native-gap-multiword.json
```

The scorer prints the correct and wrong links gained and lost on each split. For an adoption check, add `--require-gate`. It writes the evidence before returning a failing status if any split misses the frozen rule. A plain reproduction run can succeed for a rejected candidate.

Pass `--local-details /tmp/gap-link-changes.json` to the scorer to inspect each changed A/B decision. The file records the article URL, excerpt ID and digest, half-open character ranges and surface text for both occurrences, the gold label, and the stock and candidate outcomes. For the arm that linked the pair, `linking_edges` follows the path from pronoun to name. Each edge identifies the later mention, its selected antecedent, the score and the applied threshold. Group by gold label and direction of change to reconcile the four link-change counts above. Keep this file with the local GAP inputs.

The separate literary comparison scores exact proper-person boundaries against annotations in 27 books and tracks what happens to existing single-word spans after overlap removal. [LitBank boundary transfer](litbank-name-boundaries.md)
