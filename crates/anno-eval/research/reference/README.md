# Reference resolution

Use this suite to trace a reference from its source span to a proposed antecedent and the track that would receive it. The same correct mention pair can point into a track that also contains another person. Score those outcomes separately. [Anno track semantics](../../../../docs/decisions/track-association.md)

| Question | Start here |
|---|---|
| Did Anno detect the mention and put the right antecedent in reach? | [Predicted mentions and tracks](anno_adapter/README.md) uses actual Anno output and checks the alignment to the source. |
| Did bounded memory forget the entity, or did the ranker miss it? | [Book-scale retrieval](bookscale/README.md) supplies gold character history to isolate the two failures. [LitBank](entity_memory_protocol.md) compares summary policies under an equal byte budget. |
| Is a suggestion useful as a displayed pair or as a whole-track attachment? | [LitBank and PreCo](region_reasoning_followup.md) compare a fixed review quota. [WikiCoref](adjacent/wikicoref/README.md) transfers the frozen scorer to 30 previously unused articles. |
| What if the answer is a group, a discourse segment, a speaker, or a typed relation? | [Joint and adjacent tasks](adjacent/README.md) lists the prediction target, baseline, supplied annotations, and result for each corpus. |
| Does a related entity satisfy a graph query? | [Heyting’s witnessed-query experiment](https://github.com/arclabs561/heyting/tree/main/docs/experiments/typed-film-query) checks the returned entity IDs against an independent set join. |

## Run the checks

From the Anno checkout, run the command below for the offline regressions. The replay tests need NumPy and SciPy. The BookCoref test uses the Python standard library. Larger reproductions are opt-in. Each corpus page lists its inputs, versions, invocation and results. Use the reference directory for the LitBank, PreCo and WikiCoref commands. Other corpus scripts run from their own directories unless indicated.

```sh
cd crates/anno-eval/research/reference
python -m unittest discover -s bookscale -p test_replay.py
python -m unittest test_entity_memory_replay
```

## Reading the evidence

Start with the denominator. A gold-span experiment can isolate bad ranking even when a detector would never propose the target. A correct proposal under a verified catalog history measures reuse of a known link. A hit under predicted history also reflects earlier assignment errors. Compare systems with the same document split, evidence available at decision time, candidate budget and output type. [Related systems and limitations](adjacent/related_systems.md)

The historical replays originated at commit `b5854bbe8dffbdc11e67bb2d62f2cd9f6f706816` in `arclabs561/box-coref` (`experiments/` and `tests/test_entity_memory_replay.py`). Its project manifest declares MIT OR Apache-2.0. The port includes first-party scripts, aggregate measurements and methodological notes. Corpus texts, annotations, model weights and encoded features are fetched separately under the upstream terms referenced by each protocol. [Source revision](https://github.com/arclabs561/box-coref/tree/b5854bbe8dffbdc11e67bb2d62f2cd9f6f706816)
