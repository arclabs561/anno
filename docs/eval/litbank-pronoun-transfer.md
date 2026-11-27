# Choosing a literary antecedent at the GAP wrong-link budget

The rule selected on GAP development retained the stock resolver. On 1,790 questions from 62 LitBank works, it chose the correct name seven times, chose a wrong name 14 times and abstained 1,769 times. The strongest feasible neutral-type margin from development, 0.15, gave the same answers here. Removing the stock type penalty and issuing every unique component link recovered more correct links and also increased wrong links. [Selection receipt](litbank-pronoun-transfer.json) · [Score receipt](litbank-pronoun-transfer-score.json) · [Development policy](native-gap-margin.md)

| Policy on three supplied spans | Correct | Wrong competing name | Wrong when neither is correct | Abstained |
|---|---:|---:|---:|---:|
| Stock, also selected on development | 7 | 10 | 4 | 1,769 |
| Neutral type, eligible unique links | 72 | 35 | 29 | 1,654 |
| Neutral type, frozen comparison margin 0.15 | 7 | 10 | 4 | 1,769 |

The 0.15 comparator changed zero issued targets. Correct and wrong counts were unchanged within each of the 62 contributing works. Among candidates that cleared its required margin above the stock abstention threshold under neutral scoring, the runner-up vetoed zero further links.

Of the 1,296 answerable questions, 686 correct names lay outside the native antecedent distance limit. In another 546 the correct name ranked first but failed the stock threshold. In 57 a competitor outscored it. The remaining seven were linked correctly. Each prediction used the pronoun and two marked spans with the same generic mention type. Distances and name surfaces matter on this path. The experiment does not measure finding mentions in raw prose. [Ranker and trace protocol](native-gap-margin.md)

## Does widening the search recover links?

Widening the distance limit from 100 to 500 Unicode scalars brought all 686 formerly unreachable correct names into consideration. It changed zero answers across 62 works. Of those newly reachable names, 371 ranked first but missed the stock acceptance threshold, and 315 lost to the other name. The wider search scored 3,580 pronoun–name pairs, up from 1,218. Seven of the 1,296 answerable cases received the correct link at either distance. [Paired distance receipt](litbank-pronoun-distance.json)

| Maximum gap | Eligible correct names | Correct names selected directly | Correct / wrong / abstained |
|---|---:|---:|---:|
| 100 scalars | 610 | 7 | 7 / 14 / 1,769 |
| 500 scalars | 1,296 | 7 | 7 / 14 / 1,769 |

This exploratory replay used the previously inspected panel. We fixed one alternative beforehand, taking 500 from the repository’s book-scale preset and keeping every other stock setting. The panel’s own 500-scalar lookback bounds both offered names, so full eligibility at that cap follows from case selection. The scorer still penalizes longer gaps and rewards proximity. The cap applies alongside the type-specific count limit and includes a gap equal to the cap. Four authored controls record actual scored visits at gaps of 100, 101, 500 and 501 with multibyte padding. [Replay script and controls](../../scripts/qa/litbank_pronoun_distance.py) · [Native scoring](../../crates/anno/src/backends/coref/mention_ranking/algorithm/scoring.rs)

## How the transfer questions were chosen

We froze selection before applying the development rule. We examined 63 works from the pinned LitBank release, the complement of Anno's earlier 27-work competing-name panel and ten-work fold-0 panel. The existing BRAT reader checks source text at the annotated ranges, joins coreference and entity layers, and discards ambiguous chain identifiers. Among the aligned person pronouns, we considered every occurrence of *he*, *she*, *him*, *her*, *his* and *hers*. The selector looks back 500 Unicode scalars for two preceding, nonoverlapping, proper person names with distinct case-folded surfaces. It chooses the most recently ending name, then the nearest qualifying earlier name. A is the earlier position and B the later one. Only after both positions are fixed does it read their relationship to the pronoun's chain. The excerpt starts at the beginning of the lookback and includes up to 80 scalars after the pronoun. [Selector](../../scripts/qa/litbank_pronoun_transfer.py) · [Source reader](../../scripts/qa/coref_corpora/litbank.py)

Of 5,413 candidate pronouns, 1,622 had no prior name in the window and 1,803 had no qualifying second name. In 198 the two names referred to the same person, including 126 where both would have been correct. We excluded these shared identities to preserve the single-answer task. The remaining 1,790 questions comprised 362 earlier-name, 934 later-name and 494 neither-name answers. Neither means both offered names have a different annotated chain from the pronoun. The median was 23.5 questions per contributing work and the maximum was 86. Of 1,790 windows, 1,765 overlapped another window in the same work. The counts describe selected occurrences. Use the work as the unit when comparing systems.

No retained work URL, case-folded whitespace-normalized window, or sequence of 20 contiguous case-folded words matched the three complete GAP TSVs. The transfer text comes from Project Gutenberg editions, while GAP draws on Wikipedia articles. This checks identifiable overlap with the policy-selection sources. Other uses of LitBank and exposure during pretraining remain possible.

## Reproduction and attribution

Obtain the publisher's pinned LitBank release and the three official GAP TSVs. The panel generator checks both text-bearing BRAT layers, both annotation layers, and all three GAP file hashes. Run in a private working directory. The output TSV contains adapted literary excerpts, edition URLs, offsets and derived labels. Keep it with the original work credits, upstream notices and license. The repository stores the selector, original tests, hashes and aggregates. Cite Bamman, Popat and Shen (2019) for the entity annotations and Bamman, Lewke and Mansoor (2020) for literary coreference. The publisher identifies the source editions as public-domain Project Gutenberg texts and licenses LitBank under CC BY 4.0. [Upstream release and citations](https://github.com/dbamman/litbank/tree/3e50db0ffc033d7ccbb94f4d88f6b99210328ed8) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

From the anno checkout, use the stock and neutral workers built by the [development trial](native-gap-margin.md). Supply their paths in `stock` and `neutral`, the two source directories in `litbank` and `gap`, and run:

```sh
private=$(mktemp -d)
python3 scripts/qa/litbank_pronoun_transfer.py "$litbank" "$gap" "$private/panel.tsv" --check
python3 scripts/qa/native_gap_margin.py freeze "$gap/gap-development.tsv" "$stock" "$neutral" "$private/policy.json" "$private/development.json"
cmp "$private/policy.json" docs/eval/native-gap-margin-policy.json
python3 scripts/qa/native_gap_margin.py score "$private/panel.tsv" "$stock" "$neutral" "$private/policy.json" "$private/score.json" --development-receipt "$private/development.json"
```

Run the distance comparison with a stock worker built from a checkout containing both trace entry points:

```sh
python3 scripts/qa/litbank_pronoun_distance.py "$private/panel.tsv" "$stock" "$private/distance.json" --check
```

Rebuilt executables can have different hashes. Compare the selection, policy and score fields and use the local development receipt to bind the scoring run to the same executables.

## Does predicting person types identify the antecedent?

We ran the native heuristic entity recognizer on the complete English excerpt, then passed the three supplied spans through the stock mention ranker. An exact character-span match to a predicted person received Anno's `Person` type. All other supplied spans kept the same generic type as the baseline. The model saw the source and the sorted spans. It received no gold links, role labels or literary chain IDs. We also evaluated a stricter rule that issued a choice only when the native partition selected one name and that name was the sole candidate tagged as a person. Ambiguous partitions abstained in every arm. [Native example](../../crates/anno/examples/coref_gap_panel.rs) · [Scorer](../../scripts/qa/native_pronoun_person_evidence.py)

| Panel | Stock correct / wrong | With predicted person types | Sole supported name required |
|---|---:|---:|---:|
| GAP development, 2,000 | 9 / 8 | 354 / 329 | 178 / 144 |
| GAP validation, 454 | 0 / 6 | 83 / 85 | 41 / 38 |
| GAP test, 2,000 | 3 / 14 | 321 / 344 | 167 / 160 |
| LitBank, 1,790 | 7 / 14 | 257 / 211 | 41 / 54 |

Development selection maximized correct issued choices subject to at most eight wrong choices, then preferred fewer wrong choices and retained stock on an exact tie. Stock won. The same three arms were replayed unchanged on the other two official GAP splits and the previously studied literary panel. This is a transfer check on supplied spans. In particular, the stock result does not estimate end-to-end pronoun resolution on unrestricted text. [Panel digests, grouped changes and decision totals](native-pronoun-person-evidence.json)

Both names received exact person support on 1,170 of the 1,790 literary questions. Another 536 had support for exactly one. Person typing helps the ranker issue links, but a missed person tag on the other name makes the exclusive rule an unreliable identity signal. Even that rule exceeded the development wrong-link budget by 136. No resolver policy was promoted.

The existing coreference benchmark workflow replays all three GAP splits and checks the stored counts. To include the private literary panel, use the panel-generation instructions above and the same pinned GAP release:

```sh
cargo build --locked -p anno --no-default-features --example coref_gap_panel
python3 scripts/qa/native_pronoun_person_evidence.py "$gap" target/debug/examples/coref_gap_panel \
  --litbank-panel "$private/panel.tsv" --check
```
