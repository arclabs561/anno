# Can competing predicate scores recover more CrossRE relations?

Keep GLiNER-Relex’s 0.50 predicate gate for automatic extraction. On 192 fresh CrossRE test sentences, the stock decoder matched 211 of 646 gold relations in 737 predictions. Lowering the gate to 0.40 recovered 23 more and added 206 unmatched outputs. Strict F1 fell from 30.5% to 29.0%. The earlier 32-sentence AI threshold trial had seen a 1.1-point gain. [Six-domain counts](crossre-predicate-ranking.json) · [Earlier AI trial](relex-role-trial.md#predicate-only-threshold-trial)

For a caller reviewing at most one extra suggestion per sentence, raw confidence recovered **20 relations for 140 unmatched suggestions**. It won the development tie-break. Adding the margins over both the competing predicate and the reverse direction recovered 24 for 136 unmatched on test. This rule changed 21 suggestions, gained five correct ones and lost one. It had recovered one fewer on development (12 versus 13). Its four net test gains came from music and politics. Both rules recovered zero from 24 news suggestions. [Aggregate and per-domain comparisons](crossre-predicate-ranking.json)

![Correct and unmatched outputs per 100 sentences for stock extraction, one additional suggestion by confidence or both margins, and the 0.40 gate. The 0.40 gate adds the most unmatched outputs in both cohorts.](../figures/crossre-predicate-tradeoff.svg)

## Where does the model lose the relation?

The 192 test sentences contain **288 distinct annotated relations** whose exact endpoints survived mention decoding but whose relation score fell at or below 0.50. In 145 cases a stronger unannotated predicate won on the same directed pair. In 16 more, only another valid predicate scored higher. The reverse direction scored higher on 41 cases, including 36 that also had a stronger wrong label. [Rank counts and predicate confusions](crossre-predicate-ranking.json)

| Low-scoring annotated relations | Keys | Ranked first | Only other valid labels ahead | Unannotated label ahead | Same predicate higher in unannotated reverse |
|---|---:|---:|---:|---:|---:|
| Original 32 AI sentences, retained or suppressed endpoints | 62 | 23 | 0 | 39 | 4 |
| Fresh test, both endpoints retained | 288 | 127 | 16 | 145 | 41 |
| Fresh test, at least one endpoint suppressed | 58 | 36 | 2 | 20 | 3 |

The first three rank columns partition the keys. The direction column can overlap them. All these keys had reverse scores and none tied for first. Across the 603 distinct directed pairs in the test reference, 43 had multiple valid predicates. Nine pairs were annotated in both directions, each with different predicates.

Only **53 of the 288** low-scoring retained gold relations entered the `(0.30, 0.50]` suggestion pool. They appeared in **36 of the 160** sentences that offered a suggestion. Even a reviewer with the answer key could recover at most 36 relations at one per sentence. Among confidence’s 140 unmatched outputs, 86 used at least one span missing from the entity annotations. A further 32 joined two annotated entities with no reference relation in either direction. The other 22 selected a different predicate or direction on a linked pair.

The next development trial should rank or withhold suggestions using the two endpoint mention scores. Compare that rule with confidence and label margins at equal review volume, then measure how many exact relations it retains on unused test sentences. This tests whether Anno can spare reviewers the largest observed class of unmatched suggestions.

## What was scored

The replay read all 18 pinned English CrossRE files. Each domain’s training split supplied its entity and predicate prompts. We selected 16 development and 32 test sentences per domain by ascending SHA-256 of the space-joined UTF-8 sentence. Selection excluded training texts, the first 128 AI test texts from earlier diagnostics and previously selected texts. Test also excluded every development text. The historical replay added AI test indices 64–95. In all, the model scored 320 inputs. The new panels cover 96 of 2,151 development sentences and 192 of 2,446 test sentences, including 47 test sentences without a gold relation. They contain 397 of 8,199 development and 646 of 9,168 test relation occurrences. Full-release measurements are available for the stock decoder. [Input identities and panel digests](crossre-predicate-ranking.json) · [Full-test stock results](crossre-full-prompt-comparison.md)

The upstream PyTorch `predict_relations` and Anno’s trained export module produced the same stock directed edge multiset for all 320 sentences. Exact matching uses both character spans and the predicate in the annotated direction, preserving duplicate reference occurrences. Ranking counts distinct keys and records their multiplicity. All 646 held-out occurrences had distinct keys. An unmatched prediction disagrees with the published annotations, which may omit valid relations. [Checkpoint revision](https://huggingface.co/knowledgator/gliner-relex-large-v1.0/tree/4aedc9226a5ac9e2f6b5ea3e91c1ee577c88a290) · [Export module](../../scripts/export_gliner_relex_joint.py)

All review rules keep stock outputs and choose at most one extra on retained endpoints. The four rankings used raw confidence, confidence plus half its lead over the strongest other prompted label, confidence plus half its lead over the same label in reverse, or both terms. Missing reverse scores contribute zero margin. Ties use raw confidence then lexical edge order. Development selected the most exact additions, breaking ties in favor of the simpler rule before evaluating fresh test predictions. The 0.40 comparison returns every retained-pair predicate above that gate. [Replay and scoring code](../../scripts/qa/joint_baseline/relex_predicate_ranking.py)

## Reproduce

Set `CROSSRE` to the directory with the 18 pinned publisher files and `RELEX` to the local model snapshot above. The replay uses `gliner==0.2.30` and verifies source, weight and tokenizer hashes. The renderer reads the aggregate chart specification offline. [Pinned CrossRE files and terms](https://github.com/mainlp/CrossRE/tree/a58885fa760559d2dc9e176e730fe731f020d6f3) · [Portable chart specification](crossre-predicate-tradeoff.json)

```sh
trial=$(mktemp -d)
python scripts/qa/joint_baseline/relex_predicate_ranking.py "$CROSSRE" "$RELEX" \
  --output "$trial/crossre-predicate-ranking.json" \
  --tradeoff-output "$trial/crossre-predicate-tradeoff.json"
anno viz tradeoff "$trial/crossre-predicate-tradeoff.json" -o "$trial/crossre-predicate-tradeoff.svg"
```

For a larger run, `--all-eligible` expands the Python replay to 2,121 development and 2,239 test sentences after the same exclusions, plus the historical 32.
