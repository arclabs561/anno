# What happens when the native pronoun ranker treats an unknown type as neutral?

On 1,704 GAP test excerpts from previously unexamined articles, removing one type penalty gained **107 correct pronoun links and 107 wrong links**. In 94 of the new errors the ranker chose the competing name. In 13 the pronoun referred to neither candidate. This comparison matters for a caller who knows where two names occur and wants Anno to choose which person a pronoun refers to. [All splits and article counts](native-gap-neutral-mention.json)

## The change in the scorer

The entity-only ranker receives the three supplied spans as `custom("mention", Misc)` when their types are unknown. For an English person pronoun, the stock scorer gives an explicitly typed person a bonus of 0.20 and every other supplied type a penalty of 0.15. Its default pronoun link threshold is 0.225. The candidate changed the contribution of the generic `mention` type to zero. The spans, score for known types, distance cap, threshold, singleton policy and remaining features were identical in each pair of runs. The native entity-only interface receives surfaces and offsets, with no source-text argument. [Production scorer](../../crates/anno/src/backends/coref/mention_ranking/algorithm/scoring.rs) · [Native adapter](../../crates/anno/examples/coref_gap_panel.rs) · [Candidate patch and typed-score tests](../../scripts/qa/fixtures/native-gap-neutral-mention.patch)

| Supplied-span comparison | Excerpts | Gold links | Stock correct / wrong | Candidate correct / wrong | Correct gained / lost | Wrong added |
|---|---:|---:|---:|---:|---:|---:|
| Complete test | 2,000 | 1,773 | 3 / 14 | 131 / 141 | 128 / 0 | 127 |
| Complete development | 2,000 | 1,799 | 9 / 8 | 123 / 138 | 114 / 0 | 130 |
| Complete validation | 454 | 392 | 0 / 6 | 37 / 39 | 37 / 0 | 33 |

The stock result motivated the change on two historical test panels. The table below distinguishes those comparisons from other test excerpts. Of the 1,744 remaining excerpts, 40 share an article URL with a historical panel. The three official splits have disjoint sets of article URLs. All 4,454 English excerpts were included without parameter tuning. The two policies returned every supplied boundary. A separate raw-text control used each worker's ordinary mention detection and produced identical complete cluster memberships on all 4,454 excerpts. [Historical selection and scoring](gap-decision-audit.md) · [Corpus revision and attribution](coref-corpus-sources.md)

| Test subset | Excerpts | Article URLs | Correct gained | Wrong added | Additional wrong links to distractor / neither-name |
|---|---:|---:|---:|---:|---:|
| Original panel | 128 | 127 | 8 | 9 | 8 / 1 |
| Second article-disjoint panel | 128 | 128 | 11 | 6 | 5 / 1 |
| All remaining excerpts | 1,744 | 1,606 | 109 | 112 | 98 / 14 |
| Remaining excerpts on new articles | 1,704 | 1,570 | 107 | 107 | 94 / 13 |

Each excerpt contributes two labeled decisions. Linking both candidates when only A is correct would produce one correct link and one wrong link. Linking neither when neither is correct gives two true negatives. No source row marked both candidates as the answer and neither supplied-span run linked both. Among all test rows, 111 of the 127 additional wrong links went to a distractor and 16 occurred where neither name was correct. Across all three splits the candidate retained every previous correct and incorrect link.

The full-test candidate still missed 1,642 of the 1,773 true links with every endpoint supplied. Keep the automatic assignment policy pending a test of which candidate wins. Trace which names precede the pronoun and fall within the distance cap. Where both compete, measure the winner's margin over the other name and the threshold for making a link. Check whether abstaining on small margins reduces wrong identities while retaining the newly recovered links. The one-for-one tradeoff on new test articles gives that experiment a concrete baseline.

The [follow-up margin trial](native-gap-margin.md) retained stock at the development split's wrong-link budget. The [literary transfer trial](litbank-pronoun-transfer.md) applied that frozen choice to a separate set of works.

## Reproduce

The coreference benchmark workflow replays the current native ranker on the three pinned GAP TSVs. It checks the complete cluster partitions and link counts for supplied-span and raw-text input. Passage text stays in the runner's temporary directory. An intentional output change requires a new measured baseline and a review of the changed decisions.

To check the stock system locally, place the three pinned TSVs in `$GAP` and run:

```sh
cargo build --locked -p anno --no-default-features --example coref_gap_panel
python3 scripts/qa/check_native_gap_stock.py "$GAP" "${CARGO_TARGET_DIR:-target}/debug/examples/coref_gap_panel"
```

For the paired historical study, use a disposable checkout at its recorded revision with ONNX Runtime available for the original example. The native resolver needs no model weights. The runner verifies every file digest, historical per-passage stock decisions and a project-authored example where the workers must differ. The JSON stores aggregate counts and fingerprints. [Runner](../../scripts/qa/native_gap_neutral_type.py) · [Pinned files](https://github.com/google-research-datasets/gap-coreference/tree/83135f26025cdd705c04f77e940d0c9201116b80)

```sh
trial=$(mktemp -d)
cargo build --locked -p anno --no-default-features --features onnx --example coref_gap_panel
cp "${CARGO_TARGET_DIR:-target}/debug/examples/coref_gap_panel" "$trial/stock"
git apply scripts/qa/fixtures/native-gap-neutral-mention.patch
cargo test --locked -p anno --no-default-features --lib backends::coref::mention_ranking
cargo build --locked -p anno --no-default-features --features onnx --example coref_gap_panel
cp "${CARGO_TARGET_DIR:-target}/debug/examples/coref_gap_panel" "$trial/candidate"
git apply -R scripts/qa/fixtures/native-gap-neutral-mention.patch
python3 scripts/qa/native_gap_neutral_type.py "$GAP" "$trial/stock" "$trial/candidate" "$trial/result.json"
```
