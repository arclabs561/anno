# Reviewing source pairs and track assignments

## Who would use this

The first user is an Anno operator reviewing source-backed groups of mentions
in long reports or transcripts. Missing the intended antecedent costs review
time. Merging two people can attribute a statement to the wrong person across
an entire document. A useful experimental output is a short, source-linked
list of possible antecedents for an uncertain mention, with the option to leave
it unlinked. Measure the number of suggestions a reviewer inspects to recover
a valid antecedent and the number of known-distinct pairs incorrectly linked.
A benchmark win would still need a labeled caller workload and a review-time
comparison. Autonomous identity assignment and catalog linking are outside
this pilot.

Ontology maintainers have a different use. Directional scores can suggest a
missing parent for a type, check a proposed subsumption, or prioritize an
inconsistent axiom. Query authors have another. A reasoning layer can explain
which atomic edges supported a conjunction and which are missing. Neither use
should inherit results from the mention clustering pilot.

## Fixed-budget evaluation

The LitBank pilot found that boxes could recover additional candidates under
gold history and then accumulate many wrong merges in closed loop. Its
development thresholds also drifted across folds. The follow-up retains that
closed-loop measurement and adds an exact-coverage diagnostic on a frozen
gold-history trajectory. Per document, accept the most confident
`floor(0.35 * supplied_mentions)` scored proposals, with source order breaking
ties. Judge correctness after selection. This controls the review budget
without fitting a threshold to held-out labels. Selecting a quota over the
whole document makes this retrospective. A streaming deployment must use
development-calibrated decisions and measure realized coverage.

The same frozen MiniLM feature recipe is used on LitBank and a new PreCo
sample. The latter comes from Hugging Face revision
`02c8ffedb9d0b49168809a572ff553c2f6f9f7ca`. From its 500 public development
documents, deterministic hash order selects 40 for calibration and the next
80 for audit. This is a disjoint audit of public development data rather than
a claim on PreCo's official test. The input SHA-256 is
`a2c5dc47aef6d1c0ee7d3b684efc7b55158f530a01f1f846c97866c84e719c5e`.
PreCo marks singletons and its source offsets
are sentence-relative half-open token spans. The adapter checks coordinates
and preserves each occurrence. Dataset text, weights and features are local
inputs. [PreCo on Hugging Face](https://huggingface.co/datasets/coref-data/preco_raw)

| MiniLM gold-history score diagnostic | LitBank wrong proposals / 10,005 | PreCo wrong proposals / 2,800 |
| --- | ---: | ---: |
| Centroid | 2,679 | 1,527 |
| Two exemplars | 2,559 | 1,590 |
| Box | 2,106 | 1,447 |
| Box midpoint | 2,935 | 1,613 |
| Diagonal spread | 3,204 | 1,758 |

On PreCo closed-loop audit, boxes made **3,063 wrong attachments in 3,483
links** (87.9%) versus **2,223 in 3,223** (69.0%) for two exemplars. Coverage
was 43.0% and 39.8% respectively, above the intended 35%. Boxes made 95,271
false-merge pairs versus 4,045. Mean document CoNLL F1 was 49.8 versus 59.6.
At the 128-token gap, boxes retrieved 10 of 139 returning entities with gold
history, exemplars 11 and the larger-capacity centroid 100. Both two-vector arms
retained 11 correct entities, while centroid retained 103. Here the gap
between centroid and the two-vector summaries is largely retention.

The gold-history result suggests testing boxes as a proposal feature for a
review budget. It uses correct prior clusters, so it leaves open how to
obtain that history. The closed-loop result still fails the merge gate.
A next comparison should hold history to the same independently produced
clusters across arms, add a vector score-margin ranking baseline, and measure
reviewer utility on namesakes and aliases. Train and calibrate only on
non-overlapping works. Use exact span reachability with predicted mentions and
report the number of documents, mentions, returned candidates and failed
alignments. Coreference CoNLL folds flatten relation subtypes, so typed
diagnostics require the original annotations or a separate labeled corpus.
[Anno detector coverage](https://github.com/arclabs561/anno/pull/283)

## Library responsibilities

| Library | Job in a follow-up |
| --- | --- |
| **Subsume** | Use its hard/Gumbel region operations for directional type entailment, and test soft scores against held-out labels before interpreting them as probabilities. Compare point, hard-box, Gumbel-box and diagonal-spread heads on frozen features. For identity, score `same_entity` symmetrically and keep its labels separate from containment triples. |
| **Heyting** | Compose explicit typed atomic scores for bounded queries such as a two-hop affiliation plus a type constraint. Evaluate exact symbolic answers on a tiny complete graph, then compare dense and pruned paths on a larger snapshot. Use an explicit unknown-edge policy. Missing triples in an incomplete graph do not establish a negative fact. |
| **Tuplet** | Train the point-scoring control with matched positives and hard negatives, particularly same-name different-entity pairs. Freeze document splits before mining. Compare a learned symmetric point scorer with a learned symmetric region scorer at a matched budget. |
| **Ricci** | Test graph context only after the text-only comparison. Message-pass over observable relations, with no gold identity or later-document edges in the online condition. Compare the same scorer with and without graph features. |
| **Precinct and Vicinity** | Add approximate candidate search after the exact-scoring result. Quantify candidate recall and latency separately from score quality. Precinct's extent-aware candidates are relevant to point-to-region retrieval. |
| **Anno and anno-eval** | Anno owns source spans, document-local clusters and application decisions. The reference-resolution suite in anno-eval owns the replay and comparison. The accepted source-backed document graph is the input to later query experiments. |

[Subsume](https://github.com/arclabs561/subsume) · [Heyting](https://github.com/arclabs561/heyting) · [Tuplet](https://github.com/arclabs561/tuplet) · [Ricci](https://github.com/arclabs561/ricci) · [Anno identity boundaries](https://github.com/arclabs561/anno/pull/224)

### Other crates to reach for when the data warrants them

- **Lattix** can represent typed, provenance-bearing observation and relation
  edges once the task needs an actual graph. Anno's source occurrence IDs
  remain the join keys. Use **Tnorms** to make Heyting's connective choices
  explicit when comparing Gödel, product or other soft semantics.
- **Sheaf** is an option when evidence arrives in views with a known
  correspondence, such as sentence windows, quoted speaker attributions and
  structured transcript metadata. Put a comparable identity score or latent
  state at each view, define what adjacent views must agree on, and rank
  contradictions for review. Start with independent scores and a weighted
  consensus or ordinary graph Laplacian. Compare a cellular sheaf only if
  typed restrictions explain a failure those baselines miss. Measure wrong
  cross-view merges and downstream review burden. Sheaf's tree reconciliation
  also needs a genuine additive hierarchy, which entity mentions generally
  do not provide. Conformal experiments need document-level held-out
  calibration, a named scoring unit, and empirical coverage by domain.
- **Rankops** is useful once the output is a ranked review queue. Score valid
  antecedent retrieval at a fixed review budget and disclose missing or
  partial judgments. Ranking conventions belong in the receipt.

[Sheaf](https://github.com/arclabs561/sheaf) · [Lattix](https://github.com/arclabs561/lattix) · [Tnorms](https://github.com/arclabs561/tnorms) · [Rankops](https://github.com/arclabs561/rankops)

## Dataset ladder

| Task | Dataset and holdout |
| --- | --- |
| In-document identity and transfer | **LitBank** and **PreCo** for the two current domains. Add **WikiCoref** for an encyclopedia transfer and **CorefUD 1.4** for multilingual exact-span testing. Audit singleton policy and annotation equivalence for each release. [CRAC 2026](https://aclanthology.org/2026.codi-1.21/) |
| Long-distance identity | **BookCoref gold** as a per-book scale check. Keep overlapping LitBank works out of development. **NovelCR** adds English and Chinese long-span material. The authors link their Chinese Hugging Face release. Run Chinese only with a suitable pinned multilingual encoder. [BookCoref](https://huggingface.co/datasets/sapienzanlp/bookcoref) · [NovelCR](https://aclanthology.org/2025.findings-acl.268/) |
| Typed identity and bridging | Use **GUMBridge** and **DiscoTrack** to test differences among identity, part-whole, affiliation and other discourse relations. Predict labeled links and use relation-specific precision and recall. [GUMBridge](https://aclanthology.org/2026.lrec-1.543/) · [DiscoTrack](https://aclanthology.org/2026.emnlp-main.1020/) |
| Entity typing and taxonomy expansion | **UFET** provides the closest published box-typing comparison and a short-pair coreference transfer probe. Use **WordNet** or a TaxoBell taxonomy with a branch-disjoint split for hierarchy expansion. Track parent ranking and consistency separately from calibration. [Box entity typing](https://aclanthology.org/2021.acl-long.160/) · [TaxoBell](https://arxiv.org/abs/2601.09633) |
| Logical query answering | Use an exact small typed graph first, then **FB15k-237/Query2Box-BetaE** easy/hard splits or Subsume's **GALEN** ontology route with separate axiom losses. Graph splits must withhold queried evidence and guard against closure or inverse-edge leakage. [Heyting examples](https://github.com/arclabs561/heyting) · [Subsume benchmarks](https://github.com/arclabs561/subsume/blob/main/docs/benchmarks.md) |
| Cross-document linking and event identity | Test separately with entity-only **ZELDA** disambiguation and **ECB+** event co-reference. Keep document/topic identities and time anchors intact. Use gold NIL labels where available. [Selective entity disambiguation](https://aclanthology.org/2026.uncertainlp-main.9/) · [Cross-document dataset analysis](https://aclanthology.org/2026.lrec-1.328/) |

The August hierarchy-aware semantic-loss paper tests an especially relevant
Ricci–Subsume design. Compare a fixed GNN with plain loss, subclass edges and
a box hierarchy loss. Its graph link-prediction result motivates a separate
test and does not establish a mention identity benefit.
[Hierarchy-aware semantic losses](https://arxiv.org/abs/2608.22981)

## Burn compatibility

Upstream stable is **Burn 0.22.1**. Keep older Rust dependencies out of the
new training probe. The causal replay uses NumPy and ONNX.
Tuplet's 0.5.1 candidate admits 0.22.1. Cargo could not resolve 0.5.1 from
crates.io during this check, so the compatibility probe uses its Git source. Subsume 0.19's Burn
backends currently use the 0.20 line, and Ricci main uses a pinned Git revision
while its published 0.10 uses 0.21. A shared training graph needs one verified
Burn dependency and compatible adapters before passing tensors across these
crate boundaries. Keep the first learned scorer isolated until that proof
exists. A standalone Rust 1.95 Flex probe verified finite positive triplet
loss (0.358579) and completed backward on exact Burn 0.22.1 with Tuplet at
`b415204bf5962057f0752103e9e3d25c812da603`. This verifies that point baseline's
CPU training path. [Burn release](https://github.com/tracel-ai/burn/releases/tag/v0.22.1)

## Reproduce the PreCo audit

Download the pinned `dev.jsonl` into scratch and run:

```bash
python prepare_preco_replay.py /scratch/dev.jsonl /scratch/preco
```

Generate causal features for the two resulting directories as in the
[LitBank protocol](entity_memory_protocol.md). Use a common feature output
directory. Then run:

```bash
OPENBLAS_NUM_THREADS=1 python entity_memory_replay.py \
  --dev /scratch/preco/calibration --test /scratch/preco/audit \
  --features /scratch/preco-features --dimension 384 --budget 74304 \
  --encoder-id minilm-l6-v2@1110a243-uint8-avx2-left48-single
```

## Does a box score earn a place in the review queue?

A reviewer sees a proposed pair of source spans. Accepting the whole
predicted cluster behind the earlier span is a different decision. We scored
both on the same predicted histories across 99 LitBank excerpts and 80 PreCo
development documents. All methods received the same proposed pair at every
step. The exemplar scorer chose the cluster and its latest prior occurrence
supplied the visible evidence. Each fold reused its original development-fit
exemplar threshold for history updates. We checked accepted and wrong attachment
counts against the original closed-loop replay for every document.

At a 20% per-document review budget, the entries below count incorrect
suggestions. The two numbers in each cell count wrong mention pairs and
clusters with at least one member of the wrong gold entity, respectively.

| Queue ordering | LitBank, 5,714 reviews | PreCo, 1,587 reviews |
|---|---:|---:|
| Exemplar distance and margin | 1,348 / 2,308 | 962 / 1,090 |
| Add midpoint support | 1,356 / 2,313 | 946 / 1,063 |
| Add box support | 1,269 / 2,430 | 923 / 1,081 |
| Exact text first, then exemplar | **719 / 1,605** | 544 / **712** |
| Exact text first, then exemplar plus box | 723 / 1,706 | **536** / 713 |

Exact text means the current and proposed prior span have the same sequence
of case-folded tokens, including pronouns. It prioritizes candidates for
inspection. Repeated names and pronouns can still refer to different people.
The simple cue avoided 629 wrong pair reviews on LitBank and 418 on PreCo
relative to exemplar scoring alone. Adding boxes after that cue incurred four
extra wrong pair reviews on LitBank and saved eight on PreCo. The paired
document-bootstrap 95% intervals for the reduction in wrong pair reviews
were −0.40 to +0.27 percentage points and −0.12 to +1.11 points respectively.
On LitBank, adding boxes also exposed 101 extra wrong or mixed clusters
(1.23 to 2.30 points higher). The 10% and 35% budgets likewise showed no
consistent incremental pair benefit over exact text. The existing practical
gate of 20% fewer errors remains unmet for adding boxes.

**Different-text pairs.** In an exploratory follow-up we restricted the queue
to pairs with different case-folded tokens, then reviewed the top fifth of
those pairs per document. Boxes reduced incorrect visible pairs from 1,575
to 1,490 in 3,287 LitBank reviews and from 1,101 to 1,080 in 1,313 PreCo
reviews. Absolute reductions were 2.59 points (paired 95% interval 1.69 to
3.48) and 1.60 points (0.15 to 3.05). The midpoint control produced 1,546
and 1,103 errors. The boxes still surfaced slightly more mixed or wrong
clusters in both corpora. Different text includes pronouns and spelling
variants, so these results do not establish an alias-specific gain.

**Decision for Anno.** Use exact text plus exemplar confidence as the baseline
for any reviewer study. Keep a source-pair acceptance separate from a cluster
merge. The case for a learned geometry experiment now rests on finding extra
valid links among *different* surface forms, such as aliases and descriptions.
Measure these cases separately and use predicted spans before claiming a
workflow benefit. This replay measured annotated identity across all mention
types. It measured neither person-specific attribution accuracy nor human time.

**Protocol.** We ranked completed documents, taking the top floor(quota ×
mention count) suggestions, with stable source-order ties. For the augmented
arms, average-tie ranks of exemplar and geometry confidence receive equal
weight. Each scorer compares the proposed target's distance to its best
competitor, including when it prefers a different target. Histories, candidates
and input embeddings are causal. Queue quotas and ranks use the full document.
Each summary uses the same 24 retained cluster IDs. Augmented arms additionally
store bounds and perform extra scoring. The surface control was added after
inspecting the geometry results, and both evaluation sets have been used in
earlier rounds. An untouched corpus would be needed for confirmation.

Run the following once per corpus using its prior replay JSON (the ten-fold
receipt for LitBank), original CoNLL directory and causal single-example
MiniLM features. The [aggregate receipts](runs/predicted_review_queue.json)
include all quotas, denominators and paired intervals (2,000 document
resamples, seed 7). The command also emits per-document counts.

```bash
OPENBLAS_NUM_THREADS=1 python predicted_review_queue.py \
  --receipt "$REPLAY_JSON" --corpus "$CORPUS" --features "$FEATURES"
```

## Adjacent and joint tasks

The [parallel corpus experiments](adjacent/README.md) test which pairs of decisions
benefit from sharing evidence. The [related-systems review](adjacent/related_systems.md)
identifies reference implementations and evaluation gaps for the next comparison.
