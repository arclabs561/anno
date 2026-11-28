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

## Can an overlapping mention hide a valid antecedent?

The three-span transfer selector deliberately avoids overlapping names. To check the rule on inputs where spans can nest, we replayed all 63 source works through the entity-only resolver. For each book we supplied the complete set of aligned, unambiguous spans from the existing ingestion code, in document order, with the same generic entity label used in the original experiment. We compared the original and corrected one-pass candidate searches at the same 100-scalar cap. Each trace emitted the candidate spans, the scored selection and the final clusters. The private input carried no annotated chain or ID beyond the book identifier. [Panel reader](../../scripts/qa/coref_corpora/litbank.py) · [Native trace](../../crates/anno/src/backends/coref/mention_ranking/algorithm/gap_supplied_trace.rs)

We examined the 5,589 annotated person mentions matching the existing pronoun list, which includes *himself* and *herself*. Before the fix, a short span could end outside the search window even though an earlier, longer span ended inside it. Searching the correct set restored 82 candidate pairs for 80 pronouns across 35 books. Nineteen of the 82 pairs had matching annotated chains. The scoring code and thresholds were identical in both runs. Five pronouns went from abstention to a selected antecedent. Two selections matched the annotated chain and three selected a different chain. Seventeen pronouns had a different final set of cluster members, because linking two mentions can also change the cluster of another mention. This test establishes which pairs the eligibility rule must consider. Recovering more gold links at a given error rate is a separate scoring problem. [Pronoun list and corpus conventions](../../scripts/qa/coref_corpora/litbank.py) · [Input validation and scoring](../../crates/anno/src/backends/coref/mention_ranking/algorithm/mod.rs)

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

## How often is the nearer name correct for a literary reflexive?

On the same 63 LitBank works, the aligned person mentions include 176 occurrences of *himself* or *herself*. We changed only the target forms in the preceding-name selector. Forty-three occurrences lacked a prior proper name within 500 Unicode scalars, 67 lacked a second distinct name, and 14 offered two names from the same annotated chain. The resulting 52 occurrences span 25 works and include 27 instances of *himself* and 25 of *herself*. Six refer to the earlier name, 34 to the nearer name and 12 to neither. These target occurrences are disjoint from the six-form study above, but come from the same books. LitBank chain identity supplies the answer even when the matching form is emphatic, as in “the captain himself”. The annotations do not label clause structure or distinguish reflexive and emphatic uses. [Source and attribution](#reproduction-and-attribution) · [Selection and decision counts](litbank-reflexives.json)

| Answer source | Correct unique name | Wrong competing name | Wrong when neither is correct | Both names | No exact named link |
|---|---:|---:|---:|---:|---:|
| Always choose the nearer name | 34 | 6 | 12 | 0 | 0 |
| Native, three supplied spans | 5 | 0 | 0 | 0 | 47 |
| Native, raw text | 3 | 0 | 0 | 0 | 49 |
| FCoref FP32, raw text | 18 | 1 | 0 | 0 | 33 |
| Maverick MES, raw text | 27 | 0 | 0 | 0 | 25 |
| Maverick MES, three supplied spans | 38 | 0 | 1 | 1 | 12 |

FCoref chose the correct earlier name on one of six such cases and the correct nearer name on 17 of 34. Its one incorrect choice picked the earlier name. Native raw output included the exact target span in 19 queries, compared with 50 for FCoref. On 12 of the latter 50, FCoref returned the exact reflexive but neither exact offered name in its public clusters. These counts combine the model's mention and linking behavior. A missing span can have been proposed internally and subsequently omitted from its public output. The nearer-name baseline receives the annotated candidate positions, an advantage over either raw model. Every unique choice is evaluated at exact character coordinates. None of the 52 cases were used to tune a model or a threshold. Among the 31 FCoref cases with an exact reflexive and no exact linked candidate, a longer span in the reflexive's own cluster strictly enclosed exactly one marked name nine times. Eight of those enclosed names matched the annotated chain and one was the wrong competing name. Both names were enclosed in one other case and neither in 21. Checking an enclosing span therefore recovers some correct candidates but can also assign the wrong person. The diagnostic leaves the exact answer counts above unchanged. [Scorer](../../scripts/qa/litbank_reflexives.py)

Maverick's raw output returned all 52 exact reflexives. It made the same 18 correct choices as FCoref, linked eight more correctly and corrected FCoref's one wrong choice. On the remaining 25 cases neither model linked an exact candidate. In Maverick's output, checking larger mentions in the reflexive's cluster enclosed just one name in nine cases, both names in two and neither in 14. All nine single enclosed names were the right candidate. For eight of those nine, the enclosing mention itself had no exact, unambiguous LitBank chain annotation. The ninth belonged to a different annotated chain than its enclosed name. All nine FCoref enclosing mentions for its single-name cases likewise lacked an exact, unambiguous chain annotation. Correctly identifying the name therefore does not establish that the larger model span denotes the same person. [Endpoint and boundary counts](litbank-reflexives.json)

With the three annotated positions supplied, Maverick chose five of the six correct earlier names and 33 of the 34 correct nearer names. It linked both candidates on the remaining nearer-name case and wrongly selected the nearer name once when neither name was correct. The worker accepted every row.

With the same local corpus directories as above and a freshly built `coref_gap_panel` example, run:

```sh
python3 scripts/qa/litbank_reflexives.py "$litbank" "$gap" "$worker" --check
```

For FCoref, build the example with Anno's `onnx` feature, set `FCOREF_MODEL_PATH` to the pinned FP32 export, and add `--fcoref`. For Maverick, set `MAVERICK_PYTHON`, `MAVERICK_MODEL_PATH` and `HF_HUB_CACHE` to the pinned CPU environment, prepared OntoNotes checkpoint and encoder cache, then add `--maverick`. Both switches can run together. The replay checks the source manifest, reference offsets, model assets and ordered decision digests. Only aggregates are stored in the repository. [FCoref setup](../../.github/workflows/fcoref-cli.yml) · [Maverick setup](../../.github/workflows/maverick-coref.yml)
