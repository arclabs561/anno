# Multilingual antecedent retrieval on CorefUD 1.4

Three deterministic rules select an earlier annotated source mention. Each
query must have a strictly earlier, non-overlapping mention of the same entity.
We count one hit if the selected earlier span has the right entity ID, and give
the ranker no access to those IDs. This measures source-pair retrieval on
returning mentions with gold boundaries. It does not measure detection,
abstention, cluster quality, or the error rate of automatic merges.

Across 1,751 development documents from 29 corpora in 19 languages, the
three rules produced the following counts of correct suggestions. The
surface rule prefers the most recent exact case-folded source text, backing
off to the most recent eligible mention. The head rule additionally tries
matching the annotated head's surface text before falling back to recency.

| Query group | Queries | Most recent | Exact surface | Surface then head |
|---|---:|---:|---:|---:|
| All returning mentions | 91,178 | 24,232 | 43,358 | 50,336 |
| Current form previously used by the target | 32,943 | 7,577 | 30,445 | 30,445 |
| Current form never previously used by the target | 58,235 | 16,655 | 12,913 | 19,891 |
| At least 128 intervening tokens since latest gold antecedent | 8,796 | 0 | 3,481 | 5,027 |

**Consequence for Anno.** Among the 58,235 new-form mentions, exact-surface
priority costs 3,742 correct pairs relative to recency. This occurs when
another entity used the same form previously. The diagnosis uses gold target
history and is unavailable at prediction time. An annotated-head fallback
recovers 6,978 correct pairs overall compared with surface plus recency.
That advantage is conditional on the supplied head boundaries. A predicted
dependency-head baseline is a necessary comparator before crediting learned
geometry with better alias retrieval. The English counts are 7,317 and 8,169
correct of 11,456 for surface and head fallback. Czech gives 12,204 and
14,815 of 31,998, Korean 1,420 and 1,532 of 3,365, and Latin 80 and 79 of
146. The pooled counts weight queries equally. Corpora within each language
differ in genre and annotation; scores do not isolate language effects.

**Scope.** The input shard contains 175,492 mentions. We excluded 1,545
invalid or non-alphanumeric spans, 4,452 mentions with duplicate observable
coordinates and 78,317 without an eligible gold antecedent. Most recent
means greatest end offset, with ties broken by later start. Groups for new
and repeated forms partition returning mentions. The long-gap group cuts
across both. Surface matches use Unicode case folding and normalized
whitespace. This development set includes previously studied English LitBank
and GUM material. Cross-corpus and cross-language performance on the
non-public test data remains unknown.

**Reproduce.** The script verifies the pinned public development shard's
SHA-256 before reading it. Python 3.12 and PyArrow 20.0.0 were used. Raw
corpus text stays in the supplied local cache. See the original corpus for
individual resource licenses and attribution. [CorefUD 1.4](https://hdl.handle.net/11234/1-6108) · [Pinned reformatted data](https://huggingface.co/datasets/lattice-nlp/corefud-1-4/tree/3ded5a1638e43d89b81ee24bf5b98f66943e1c60)

```bash
python -m pip install pyarrow==20.0.0
python corefud_retrieval.py /tmp/corefud-dev.parquet results.json
```
