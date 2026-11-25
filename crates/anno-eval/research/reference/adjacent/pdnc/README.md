# Joint quotation attribution on PDNC

An Anno reviewer who sees a quote may want a correct speaker and at least one
correct addressee in one suggestion. We compared paired and separate choices
on all 28 books at the [pinned PDNC revision](https://github.com/Priya22/project-dialogism-novel-corpus/tree/e507a27b89a85c1e9710a897ad0b3f5a9ad73dc4).
There were 37,131 quotes. The 36,446 with a speaker and at least one addressee
formed the joint evaluation. All quote spans matched the annotated fragments
after trimming outer whitespace. All methods used gold quote boundaries and
per-book character names/alias rosters, including special character labels.
None consumed gold speaker, addressee, quote type or internal character
mention labels as features. Quote type was used only for evaluation.

| Decoder | Speaker right | One addressee right | Both right | Both right, implicit |
|---|---:|---:|---:|---:|
| Separate choices from pipeline history | 40.86% | 51.02% | 19.03% | 10.63% |
| Soft penalty for identical roles | 43.30% | 49.71% | 29.70% | 18.53% |
| Previous addressee as evidence for speaker | 43.15% | 51.02% | 20.27% | 12.32% |
| Both cues, frozen pipeline history | 43.11% | 48.98% | 29.07% | 20.40% |
| Both cues, own predicted history | 41.94% | 49.80% | 28.50% | 21.25% |

The soft penalty produced 3,891 additional correct pairs and 477 fewer correct
addressee-only predictions. It improved pair counts in each of the 28 books.
The simple penalty provides a useful comparator before testing learned
consistency or Sheaf. Implicit quotations are a harder slice (16,609
jointly labeled). The feedback model recovered the most correct pairs there.

Each arm predicts one addressee, so the main pair metric counts a hit when
that person belongs to the annotated set. Only 21,798 examples have singleton
audiences. Requiring the full audience and the speaker to be right gave
3,427 hits for the baseline and 5,547 for the penalty across all 36,446.
Speaker attribution and full-audience recovery both have considerable headroom.

The experiment uses transparent fixed heuristics. Contextual name matches
near reporting verbs and earlier predicted speakers score potential speakers.
Vocative-like comma-adjacent names and the previously predicted speaker score
addressees. Speaker scoring may use immediate post-quote narration and
interrupted reporting clauses. Three arms hold the history fixed at the
pipeline's own predictions, and one propagates its own paired predictions.
Candidate inventory uses the full annotated book roster, so a streaming system
with predicted entities would face a harder problem. This is an exploratory
whole-corpus comparison including titles shared with earlier LitBank work.
No model was trained and no geometry or human-review time was measured.

Run with Python 3.10+ and network access for the first download. A populated
cache permits an offline repeat. The JSON includes per-book counts and a
2,000-resample paired book bootstrap for the penalty's gain in pair accuracy.

```bash
python pdnc_joint.py /tmp/pdnc-joint results.json
```
