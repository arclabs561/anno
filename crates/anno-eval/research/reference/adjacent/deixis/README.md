# Dialogue segments and plural reference

A reviewer must sometimes select the earlier utterance that “that” refers to, or several distinct entities that “they” denotes. Both actions require richer output than a pair of mention IDs. We measured the cost of choosing the action and retrieving its argument on the three official CODI-CRAC22 test domains (3 AMI meetings, 33 persuasion dialogues, 38 LIGHT dialogues). Counts are pooled, so the domain breakdown in the receipt matters.

| Outcome | Correct or covered | Gold queries |
|---|---:|---:|
| Previous complete sentence exactly matches a segment | 150 | 351 |
| One of the previous three sentences exactly matches | 225 | 351 |
| All plural antecedents among eight latest distinct entities | 50 | 74 |
| Two latest distinct entities give the exact plural set | 9 | 74 |
| Lexical routing plus previous sentence, both correct | 134 | 351 |
| Lexical routing plus latest two entities, both correct | 7 | 74 |

For deixis, 247 of 351 queries have an earlier, wholly completed discourse segment explicitly annotated in the same chain. The remaining cases stay in the denominator and include composite antecedents, overlaps, and other representations. The query set includes all non-segment marks in the discourse-deixis column. Exact sentence coordinates can be stricter than human judgments. For plural reference we score the first identity mark of groups with at least two distinct entities declared via identity-column ElementOf. Gold entity IDs, gold markable boundaries, and gold sentence boundaries make this a retrieval and routing diagnostic. We exclude pseudo-entities from the plural candidate list. Of 74 groups, 73 have all members mentioned in the past. Simultaneously ending mentions break ties by latest starting position and then source markable ID.

The routing baseline learns each case-folded span's association with ordinary, group, or discourse-segment reference from the official training files. The scores use equal class priors to expose rare references. This yields 304 correctly detected segment queries, plus 1,150 spurious segment classifications (1,148 ordinary mentions and two groups). The initial rankers then select the previous sentence for segment queries or the two most recent distinct entities for groups. No review threshold was tuned.

We also compared two learned decoders over exactly the same lexical evidence. The factorized decoder predicts the reference kind first, then its sentence offset or tuple of entity-recency ranks. The joint decoder chooses the highest-scoring (kind, offsets) combination directly. Both count conditional frequencies by mention surface and use inverse training class frequency as a weight on the kind. Positions cover three sentences and eight recent entities respectively. An unreachable label lets the model abstain. If several recent sentences match an annotated segment, training and scoring use the nearest match. Each requested tuple counts as an offer, including an offset that has no candidate at runtime. Both decoders yielded **112 exact segment predictions in 1,120 offers and 2 exact groups in 561 offers** across 74 test documents. Matching aggregate counts establish no gain from joint decoding with these features. The output is not an official end-to-end coreference score. The experiment did not train box embeddings.

A stronger experiment would represent turns and discourse segments explicitly, retrieve a scored set of participants for plural reference, and compare joint and sequential decisions at the same review budget with predicted mention histories. Compare turn boundaries against sentence boundaries, including partial-utterance antecedents. The 38 LIGHT dialogues provide a useful check because only 38 of 128 queries match the immediately preceding sentence, compared with 69 of 105 in persuasion dialogues.

## Reproduce

Python 3.10+ and Git suffice. Corpus text stays in a separate checkout.

```sh
git clone https://github.com/UniversalAnaphora/CODI-CRAC22-Corpus.git /tmp/dialogue-corpus
git -C /tmp/dialogue-corpus checkout 2f7b2506b87e2fd58da44b4f1d197483deba03ef
python dialogue.py /tmp/dialogue-corpus > reproduced.json
diff -u result.json reproduced.json
```

[Corpus release](https://github.com/UniversalAnaphora/CODI-CRAC22-Corpus/tree/2f7b2506b87e2fd58da44b4f1d197483deba03ef) · [Shared-task introduction and conventions](https://github.com/juntaoy/codi-crac2022_scripts/blob/main/2022_CODI_CRAC_Introduction.md)
