# Joint document identity, typing, Wikipedia linking and relations on DWIE

The practical question is whether predicted document context helps an annotator
choose the right Wikipedia entry or recover a typed fact, and how often using
that choice as identity introduces a false merge. This CPU experiment uses the
official 702 training and 100 test news documents at the pinned DWIE revision.
All 5,433 annotated test mention spans are supplied. Test-time features use
their text and released link candidates and scores. Gold concept IDs, relations,
types, Wikipedia links, and representative entity names enter only evaluation
on the test split. [DWIE release](https://github.com/klimzaporojets/DWIE/tree/575e0be3e6f980b77152f587b26530082f94a6c1)

| Arm | Identity pair F1 | Wikipedia link F1 | Type F1 | Directed relation F1 | Incorrect identity pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact surface pipeline | 90.02 | 82.24 | 79.66 | 47.16 | 88 |
| Merge shared predicted page | 95.45 | 82.24 | 79.66 | 47.16 | 182 |
| Rerank with aliases and types | 95.68 | 83.22 | 79.91 | 47.77 | 204 |
| Also use known relations | 95.74 | 83.61 | 80.11 | 48.44 | 216 |

All scores are micro F1 percentages. Identity compares unordered within-document
mention pairs. Typing compares mention–type pairs for tags beginning with
`type::`. Linking compares non-NIL mention–Wikipedia page predictions.
Relations expand each directed entity-level fact over its subject and object
mentions and exclude same-mention pairs. Repeated entities consequently carry
more weight. Gold totals are 10,881 identity pairs, 3,959 linked mentions,
16,725 mention/type pairs and 30,198 mention-pair/label triples. These are
diagnostics on supplied mentions, distinct from the official entity-centric
metrics and end-to-end leaderboard.

## What couples the tasks

The pipeline groups normalized identical surface forms and averages their
released Wikipedia candidate priors. Where training data contain the surface,
its empirical distribution over pages, including NIL, is added. The basic
coupling merges different surfaces that selected the same non-NIL page. The
next arm does one synchronous page reranking pass using other surfaces' initial
page predictions and agreement between type sets estimated from training.
The full arm adds a bonus for connections to pages assigned elsewhere in the
document, using directed page-pair facts observed in training. Returned relation
labels always preserve their training direction. The reranking coefficients
were fixed before running the test. All arms use the identical lookup decoder
for types and relation labels. The decoder can recover known page-pair facts
and cannot generate genuinely new relations. The method uses full-document
evidence and no learned geometric model.

Compared with merging shared pages alone, the full pass increased link F1 by
1.38 points (95% document-bootstrap interval +0.58 to +2.24) and relation F1
by 1.29 points (+0.37 to +2.45). Its extra 0.28 points of identity F1 had an
interval from −0.03 to +0.69. Its 89 additional correct coreference pairs came
with 34 additional incorrect pairs. The result favors offering enriched page
and fact suggestions for review. It supplies a control for a future Ricci graph
or Subsume type compatibility scorer. This benchmark includes externally linked
named entities, gold mention boundaries and training fact reuse. Decisions about
automatic merges in open-world text need separate evidence.

## Reproduce

With Python 3.10 or newer, no third-party packages are needed. Clone the
[release](https://github.com/klimzaporojets/DWIE), check out the SHA in the
command below, and run the experiment. The results file contains full counts
and intervals from 2,000 paired resamples of held-out documents, seed 7.

```bash
git clone https://github.com/klimzaporojets/DWIE.git /tmp/dwie-data
git -C /tmp/dwie-data checkout 575e0be3e6f980b77152f587b26530082f94a6c1
python run.py /tmp/dwie-data/data/annos
```
