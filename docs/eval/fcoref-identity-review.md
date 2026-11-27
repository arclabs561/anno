# When a wider FCoref mention refers to the right person

On 43 previously identified English GAP pairs, both blinded agent reviewers judged **42 wider mentions to denote the marked person**. They agreed that the remaining expression denoted a writing class associated with the named person. All 43 missed the exact annotated name span. Wider mentions extended by a median of 14 source characters (range 1–86), to the left in 16 cases, to the right in 19, and on both sides in eight. [Aggregate review](fcoref-identity-review.json)

These cases were selected from the [256-article stage replay](fcoref-endpoint-stages.md). For each positive pair, the exact name fell below the stock mention cutoff and the pronoun's returned cluster contained a strictly larger mention at the name occurrence. If several mentions met that definition, we chose the shortest, then the earliest start and end. This is a diagnostic of those 43 known misses. A population accuracy estimate would also need independently sampled correct links, wrong links and missing mentions.

## Two questions with different answers

GAP identifies which marked name refers to the pronoun and supplies its character offsets. It has no label for the referent of the model's longer phrase. Two agent reviewers in separate fresh contexts independently read the source context, name and enclosing phrase. They received shuffled opaque codes and no GAP link labels, model scores, peer judgments or answer key. They classified the referent of the entire phrase as the same person, a different referent, or uncertain. Both answered all eight interleaved authored controls correctly, including a coordinated group, a relative and a possessed object. Neither abstained. These are model-assisted judgments from the same model family. Independent human adjudication would strengthen the semantic result.

For a reader following *who* the pronoun refers to, the wider mention appears useful in 42 cases. For a tool highlighting *exactly the name*, all 43 boundaries are wrong. A rule that always credits a name inside a linked phrase would have credited the writing-class case too. The existing [compact display comparison](../QA.md#compare-compact-name-displays) includes a separate attribution control for the same risk. Keep the linked referring expression and any proposed display-name span separately identifiable.

## How the model made the link

The [native pair-score replay](fcoref-rival-pair-scores.md) evaluated the same 43 cases at the original candidate budget, giving the exact name a slot. Every direct edge between the pronoun and exact name scored below the zero threshold for linking. The enclosing mention had the higher mention score in all 43, and the higher pair-specific term in 41. It linked directly with the pronoun in 38 cases and through other cluster members in five. Those are properties of model inference. The referent judgments and the character-overhang counts measure two separate consequences for the reader.

## Does a single contained person span make a usable highlight?

We tested the existing [single contained person span rule](../QA.md#compare-compact-name-displays) on the same reviewed expressions. Built at `75f961e5`, the offline Anno CLI received each complete passage as plain text. For each original FCoref expression, the display rule offered a highlight only when exactly one distinct stacked NER person span fell inside it. The resolver's expression and cluster membership remained available in every case. The reader judgments and annotated names were used only for scoring.

| Cases | Exact name offered | Other span offered | No suggestion |
|---|---:|---:|---:|
| 42 GAP expressions judged to denote the person | 11 | 11 | 20 |
| 4 authored expressions denoting the person | 3 | 1 | 0 |

For the one GAP expression judged to denote something else, the rule still offered a person highlight. It also made that error on all four authored controls with a different referent. The display is meant to identify the marked person as the expression’s referent, so offering a name on a different-referent expression fails the attribution check. On the 42 same-person cases, “other span” means a boundary mismatch; these data do not establish that the suggested span denotes a different person. The standalone person detector emitted 16 exact names on the 42 same-person cases, of which the one-span rule offered 11. The 43 passages were selected for a known boundary problem, so the counts apply to that set.

**Decision.** Retain the full linked expression. Keep this shortening rule out of the public display options. Even on the same-person cases only half of its 22 suggestions matched the marked name, and it failed the writing-class attribution control. [Aggregate result](fcoref-optional-highlight.json)

To replay, use the private packet and the two independent reviews described below. Build the text-only study configuration and run the [scorer](../../scripts/qa/fcoref_optional_highlight.py). It verifies the requested and observed backend, source digests, and character offsets before counting a suggestion.

```sh
cargo build --locked -p anno-cli --no-default-features --features extractor-html2text --bin anno
python3 scripts/qa/fcoref_optional_highlight.py \
  "$private/packet.jsonl" "$private/reviewer-a.jsonl" "$private/reviewer-b.jsonl" target/debug/anno
```

## Replay and review

Use the GAP files and model pins from the [stage replay](fcoref-endpoint-stages.md#replay-the-stage-counts). In a disposable checkout apply the rank-readout patch and build the native runner as described there. Create a private output directory outside Git. The script verifies the stock cluster fingerprint, all model and corpus pins, and the final packet fingerprint. Preparing the packet requires Python `tokenizers`. Scoring the reviews uses the standard library.

```sh
python3 scripts/qa/fcoref_identity_review.py prepare \
  "$GAP/gap-development.tsv" "$GAP/gap-test.tsv" "$GAP/gap-validation.tsv" \
  "$FCOREF_MODEL_PATH" target/debug/examples/coref_gap_panel "$private/packet.jsonl"
python3 scripts/qa/fcoref_identity_review.py score \
  "$private/packet.jsonl" "$private/reviewer-a.jsonl" "$private/reviewer-b.jsonl"
```

Give each reviewer the packet alone and the following question. “Does the entire phrase denote the same individual as the name at this occurrence? Use `same`, `different`, or `uncertain`. A longer name, title or appositive can still denote the person. A group, relative, possessed item or work may denote something else. Judge the phrase in context and use uncertain when the referent is genuinely ambiguous.” The name and phrase intervals use half-open Unicode character offsets. Collect one JSON object per line with `code` and `label`. Reviewers can keep a private reason alongside each label. Fresh reviews can yield different judgments. Only the aggregate and fingerprints belong in this public study. [Scorer and project-authored controls](../../scripts/qa/fcoref_identity_review.py) · [GAP rights and sources](coref-corpus-sources.md)
