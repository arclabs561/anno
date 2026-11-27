# Can a margin recover pronoun links at the stock error count?

On 2,000 GAP development excerpts with the three mention spans supplied, the strongest eligible margin recovered the same nine correct answers and made the same eight wrong assignments as stock. The predeclared tie rule retained stock. The alternative also issued the exact same target or abstention as stock on every test and validation excerpt. There is no measured gain for this caller from adopting the candidate. [Development sweep](native-gap-margin-development.json) · [Frozen policy](native-gap-margin-policy.json) · [Test and validation receipts](native-gap-margin-replay.json)

| Official split | Excerpts | Stock correct / wrong / abstained | Neutral type, ungated | Neutral type, margin 0.15 |
|---|---:|---:|---:|---:|
| Development | 2,000 | 9 / 8 / 1,983 | 123 / 138 / 1,739 | 9 / 8 / 1,983 |
| Test | 2,000 | 3 / 14 / 1,983 | 131 / 141 / 1,728 | 3 / 14 / 1,983 |
| Validation | 454 | 0 / 6 / 448 | 37 / 39 / 378 | 0 / 6 / 448 |

## How the reader decides

For an application prepared to accept more wrong assignments in exchange for recovering more correct names, the retention study traces the two margin components separately and reports both costs. [Operating points on GAP and transfer to Winogender](native-gap-link-retention.md)

The native entity-only resolver receives three generic mentions with surfaces and scalar offsets. A reader issues a target when exactly one marked name shares the pronoun's returned group. Two names in the group, or zero, give an abstention. An ignored Rust test records the actual scores, visit order, threshold and winner from the native antecedent comparison. It also invokes the public resolver for its final groups. The replay checks the direct edge and the final membership separately. A later name can join an earlier mention, and a direct winner can belong to a group with several candidate names. The trace uses the same supplied-mention preparation and empty source-text argument as the native entity resolver. Its inputs contain no labels. [Trace](../../crates/anno/src/backends/coref/mention_ranking/algorithm/gap_supplied_trace.rs) · [Earlier neutral-type experiment](native-gap-neutral-mention.md)

The experimental reader also requires the unique returned target to be the pronoun's direct winner under the neutral-type scoring patch. Its lead must reach the selected margin over **both** the effective native acceptance threshold and the next best scored name. With no scored rival, only the threshold matters. Native acceptance itself requires a strict threshold pass, including at margin zero. Equal-scoring candidates retain the native nearer-first visit order. An explicit positive margin can suppress a tied decision. A stock penalty of 0.15 applied to an untyped candidate for certain English pronouns. On these three splits, requiring the neutral winner to clear the threshold by 0.15 already reproduces stock. Zero further queries were vetoed by the competitor condition at that cutoff. In the authored text “Hera told Zoe her news.” the neutral resolver chooses “Zoe”. Its score leads the native threshold by about 0.272 and the other name by only 0.102. The experiment requires the 0.15 comparator to abstain on that real native result. [Decision controls](../../scripts/qa/test_native_gap_margin.py)

Before selecting on development, we fixed a grid from 0 to 0.4, a cap of the stock wrong-answer count, and a preference for the most correct answers, then fewer wrong answers, then the largest margin. At equal correct and wrong counts the production policy stays stock. The closest lower grid point, 0.125, yielded 20 correct and 25 wrong. The frozen selector and its strongest feasible experimental comparator were fixed before transfer scoring. The separate commitment binds the ordered development grid, stock counts, source digest, selected policy digest and comparator. Each scoring run checks that the executable hashes match its local development run. [Comparator commitment](native-gap-margin-comparison.json) · [Runner](../../scripts/qa/native_gap_margin.py)

## Where the correct antecedent was lost

Among the 1,799 development queries with a gold target, 353 gold names followed the pronoun and 382 lay beyond the default distance cap. The neutral scorer directly selected 123 gold targets. Of the remainder, 394 lost to the other name and 547 ranked first but failed the strict native score threshold. The 201 queries with neither candidate correct form their own stratum. Neutral made 12 wrong commitments there and 126 to a distractor where a gold name was present. These stage counts describe the supplied-span path and its inferred mention types. The decoder can also treat a capitalized pronoun differently from its lowercase form. The raw-text detector has a different candidate set. [Raw and supplied study](gap-decision-audit.md)

All official GAP splits had been scored in the earlier type experiment. Test and validation provide previously examined comparison data for a development-only selection. They cannot establish fresh transfer. Across 1,790 independently prepared LitBank occurrences, the 0.15 comparator again matched every stock target and abstention. Ungated neutral scoring raised correct answers from 7 to 72 and wrong assignments from 14 to 64. The transfer panel covers 62 books and has its own selection and licensing rules. [Independent LitBank transfer](litbank-pronoun-transfer.md)

## Reproduce

Use a disposable checkout. Obtain the three GAP TSVs at the pinned upstream revision and keep them in a local directory. The repository stores aggregate counts and fingerprints. The source excerpts are drawn from Wikipedia and carry their respective reuse and attribution terms. [Pinned GAP release](https://github.com/google-research-datasets/gap-coreference/tree/83135f26025cdd705c04f77e940d0c9201116b80) · [Source digests](dataset-github-sources.csv)

Build the stock test executable and a second one with the existing neutral-type patch. Cargo prints its exact test executable path. Copy each executable into the trial directory before the next build. This trace requires the Rust crate's ordinary unit test harness and runs offline with cached build dependencies.

```sh
trial=$(mktemp -d)
build_trace() {
    cargo test --locked -p anno --lib --no-default-features --no-run --message-format=json \
      > "$trial/build.jsonl" || return 1
    python3 - "$trial/build.jsonl" "$1" <<'PY'
import json, shutil, sys
with open(sys.argv[1]) as stream:
    artifacts = [json.loads(line) for line in stream]
path, = [m['executable'] for m in artifacts if m.get('target', {}).get('name') == 'anno' and m.get('executable')]
shutil.copy(path, sys.argv[2])
PY
}
build_trace "$trial/stock"
git apply scripts/qa/fixtures/native-gap-neutral-mention.patch
build_trace "$trial/neutral"
git apply -R scripts/qa/fixtures/native-gap-neutral-mention.patch
qa=scripts/qa/native_gap_margin.py
python3 "$qa" freeze "$GAP/gap-development.tsv" "$trial/stock" "$trial/neutral" "$trial/policy.json" "$trial/dev.json"
cmp "$trial/policy.json" docs/eval/native-gap-margin-policy.json
for part in test validation; do
    python3 "$qa" score "$GAP/gap-$part.tsv" "$trial/stock" "$trial/neutral" \
      "$trial/policy.json" "$trial/$part.json" --development-receipt "$trial/dev.json"
done
```

The command also accepts a local panel with the same TSV columns and whole Unicode-scalar character offsets. Give each occurrence a unique ID and each work a stable URL or key in the URL column. The two binary digests identify the build used and may differ across machines. The exact development policy and its normalized comparison evidence are checked independently of the machine. To compare another run with the published receipts, compare the counts, transition tables, source digest and policy digest.
