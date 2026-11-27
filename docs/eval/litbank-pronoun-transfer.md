# Choosing a literary antecedent at the GAP wrong-link budget

The rule selected on GAP development retained the stock resolver. On 1,790 questions from 62 LitBank works, it chose the correct name seven times, chose a wrong name 14 times and abstained 1,769 times. The strongest feasible neutral-type margin from development, 0.15, gave the same answers here. Removing the stock type penalty and issuing every unique component link recovered more correct links and also increased wrong links. [Selection receipt](litbank-pronoun-transfer.json) · [Score receipt](litbank-pronoun-transfer-score.json) · [Development policy](native-gap-margin.md)

| Policy on three supplied spans | Correct | Wrong competing name | Wrong when neither is correct | Abstained |
|---|---:|---:|---:|---:|
| Stock, also selected on development | 7 | 10 | 4 | 1,769 |
| Neutral type, eligible unique links | 72 | 35 | 29 | 1,654 |
| Neutral type, frozen comparison margin 0.15 | 7 | 10 | 4 | 1,769 |

The 0.15 comparator changed zero issued targets. Correct and wrong counts were unchanged within each of the 62 contributing works. Among candidates that cleared its required margin above the stock abstention threshold under neutral scoring, the runner-up vetoed zero further links.

Of the 1,296 answerable questions, 686 correct names lay outside the native antecedent distance limit. In another 546 the correct name ranked first but failed the stock threshold. In 57 a competitor outscored it. The remaining seven were linked correctly. Each prediction used the pronoun and two marked spans with the same generic mention type. Distances and name surfaces matter on this path. The experiment does not measure finding mentions in raw prose. [Ranker and trace protocol](native-gap-margin.md)

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

Rebuilt executables can have different hashes. Compare the selection, policy and score fields and use the local development receipt to bind the scoring run to the same executables.
