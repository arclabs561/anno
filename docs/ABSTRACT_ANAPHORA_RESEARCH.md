# Abstract Anaphora Resolution: A Research Gap Analysis

## The Problem

Standard coreference resolution links **noun phrases** to their antecedents:

```
"John went to the store. He bought milk."
       ^^^^                ^^
       antecedent          pronoun (links to "John")
```

**Abstract anaphora** is fundamentally different - it links pronouns like "this", "that", "it" to **events, propositions, or situations**, not noun phrases:

```
"Russia invaded Ukraine in 2022. This caused inflation to spike."
        ^^^^^^^                  ^^^^
        EVENT                    abstract anaphor (links to the INVASION EVENT)
```

The referent of "This" is not "Russia" or "Ukraine" - it's the entire **invasion event**.

## Why Current Methods Fail

### 1. Mention-Ranking Architecture Mismatch

Standard neural coref (Lee et al. 2017, SpanBERT-coref) enumerates **noun phrase spans** as candidate mentions. Events are not noun phrases - they're triggered by verbs or nominalized forms that don't fit the span enumeration heuristics.

### 2. Coreference Training Data Excludes Events

OntoNotes - the dominant coref benchmark - does **not annotate event coreference**. Systems trained on OntoNotes learn to ignore event-pronoun links entirely.

### 3. The Scope Problem

"This" can refer to:
- A single clause: "He fell. This hurt."
- A sentence: "The merger failed spectacularly. This surprised analysts."
- Multiple sentences: "First A happened. Then B happened. This cascade led to C."
- An implicit consequence: "Oil prices rose. This [the economic impact] affected consumers."

The scope is **pragmatically determined** - there's no syntactic signal for where the antecedent ends.

## Research Literature

### Key Papers (Verified)

1. **Kolhatkar & Hirst (2012)**: "Resolving 'this-issue' anaphors"
   - First systematic study of abstract anaphora in biomedical text
   - Candidate ranking model with 43 features
   - Found correct antecedent contained in candidates 92% of the time
   - Average of 23.80 candidates per anaphor - high ambiguity
   - Key finding: General abstract-anaphora features > issue-specific features
   - Source: http://www.cs.toronto.edu/pub/gh/Kolhatkar+Hirst-2012.pdf

2. **Marasović et al. (2017)**: "A Mention-Ranking Model for Abstract Anaphora Resolution"
   - EMNLP 2017, pages 221-232
   - LSTM-Siamese network architecture for comparing anaphor to candidates
   - Generated artificial training data to overcome data scarcity
   - Outperformed SOTA on shell noun resolution
   - Struggled with pronominal anaphors on ARRAU corpus
   - Code: https://github.com/amarasovic/neural-abstract-anaphora
   - Source: https://aclanthology.org/D17-1021/

3. **Moosavi & Strube (2016)**: "Which Coreference Evaluation Metric Do You Trust?"
   - Introduced LEA (Link-Based Entity-Aware) metric
   - Identified "mention identification effect" flaw in B³, CEAF, BLANC
   - These metrics reward mention identification even without correct links
   - LEA formula: Recall = Σ(|entity| × resolved_links/total_links) / Σ|entity|
   - Source: https://aclanthology.org/P16-1060/

4. **ARRAU Corpus** (Poesio et al.)
   - Multi-genre corpus with bridging references and discourse deixis
   - Annotates ALL NPs (including non-referring) as markables
   - Handles ambiguity with multiple antecedents
   - Used in CRAC shared tasks
   - Source: https://aclanthology.org/W18-0702/

5. **Shell Nouns** (Schmid 2000, Simonjetz & Roussel 2016)
   - ~670 English nouns that encapsulate propositions
   - Categories: factual, linguistic, mental, modal, eventive, circumstantial
   - Examples: "this problem", "the issue", "the fact that..."
   - Source: https://aclanthology.org/D13-1030.pdf

6. **Li & Ng (2022)**: "End-to-End Neural Discourse Deixis Resolution in Dialogue"
   - EMNLP 2022 - **Current State-of-the-Art**
   - Adapts Lee et al.'s span-based entity coreference model
   - Model name: **dd-utt** (discourse deixis utterance)
   - SOTA on 4 datasets from CODI-CRAC 2021 shared task
   - Key insight: Task-specific extensions that exploit discourse deixis characteristics
   - Code: https://github.com/samlee946/EMNLP22-dd-utt
   - Source: https://arxiv.org/abs/2211.15980

7. **ARRAU 3.0** (Poesio et al., 2024)
   - Latest version released in 2024
   - Available via LDC: LDC2013T22
   - Free subsets: GNOME, Pear Stories (downloadable)
   - Full corpus: https://sites.google.com/view/arrau/corpus

## Taxonomy of Abstract Referents

| Type | Example | Referent |
|------|---------|----------|
| **Event** | "The crash happened. This shocked everyone." | The crash event |
| **Fact** | "He won. This is undeniable." | The fact that he won |
| **Proposition** | "She might leave. This worries me." | The possibility of leaving |
| **Situation** | "Prices rose while wages fell. This was unsustainable." | The combined state |
| **Manner** | "He spoke rudely. This annoyed her." | The manner of speaking |

## Current System Evaluation

### Test Setup

We evaluate `anno::eval::coref_resolver::SimpleCorefResolver` on:
1. **Nominal coref** (baseline): "John" → "he"
2. **Abstract anaphora**: "The invasion" → "This"

### Expected Results

- Nominal coref: ~70-80% accuracy (rule-based systems do well on simple cases)
- Abstract anaphora: ~0-10% accuracy (no mechanism to even detect events)

### Failure Modes

1. **No Event Detection**: Current resolver only considers entities from NER
2. **Pronoun Heuristics Fail**: "This" matches to nearest person/org, not to events
3. **No Discourse Structure**: No representation of clause/sentence boundaries

## Proposed Research Direction

### Phase 1: Quantify the Gap (This Document)

- Create evaluation dataset mixing nominal + abstract cases
- Run current resolver
- Generate failure analysis report

### Phase 2: Event Extraction Integration

- Add event triggers to `EntityCategory` (already has `Relation`)
- Extend `Mention` type to include event mentions
- Create `EventMention` struct parallel to entity mentions

### Phase 3: Discourse-Aware Resolution

- Implement Discourse Representation Structure (DRS) parsing
- Add "sentence" and "clause" level scope tracking
- Create `AbstractAntecedent` type that can point to discourse units

### Phase 4: Evaluation on Benchmark

- Evaluate on ARRAU corpus (has abstract anaphora annotations)
- Compare to neural baselines (SpanBERT-coref, LingMess)

## Benchmark Data Formats

### Marasović et al. JSON Format

The standard format for abstract anaphora data (from `abstract-anaphora-data` repo):

```json
{
  "anaphor_derived_from": "since-prp",      // Syntactic pattern source
  "anaphor_head": "that",                   // The anaphor word
  "anaphor_phrase": "because of that",      // Full anaphoric NP
  "antecedents": ["S clause text", ...],    // Positive candidates (gold)
  "antecedent_nodes": ["S", "VP"],          // Syntactic tags of antecedents
  "candidates": ["neg candidate 1", ...],   // Negative candidates (same sentence)
  "candidates_0": [...],                    // Neg candidates from anaphoric sentence
  "candidates_1": [...],                    // Neg candidates from sentence-1
  "prev_context": ["sent-1", "sent-2", ...] // 4 preceding sentences
}
```

### CODI-CRAC 2021 Evaluation

The shared task used:
- **Input**: Dialogue with utterance boundaries
- **Task**: Identify discourse deixis spans and link to antecedents
- **Metrics**: MUC, B³, CEAF, LEA (LEA preferred)
- **Baseline**: Lee et al. (2018) span-based coref
- **SOTA**: dd-utt model by Li & Ng (2022)

### Shell Noun Categories (Schmid 2000)

| Class | Examples | Typical Antecedent |
|-------|----------|-------------------|
| Factual | fact, reason, evidence, result | Event/Fact |
| Linguistic | claim, argument, answer, question | Proposition |
| Mental | idea, belief, thought, view | Proposition |
| Modal | possibility, chance, need, risk | Proposition |
| Eventive | event, incident, action, decision | Event |
| Circumstantial | situation, problem, case, issue | Situation |

## Files in This Research Track

- `docs/ABSTRACT_ANAPHORA_RESEARCH.md` - This document
- `src/discourse.rs` - Core discourse module (EventMention, DiscourseReferent, ShellNoun)
- `src/eval/abstract_anaphora.rs` - Evaluation infrastructure
- `examples/abstract_anaphora_eval.rs` - Run evaluation
- `abstract_anaphora_analysis.html` - Generated failure report

## References

```bibtex
@inproceedings{kolhatkar2012resolving,
  title={Resolving "this-issue" anaphors},
  author={Kolhatkar, Varada and Hirst, Graeme},
  booktitle={EMNLP},
  year={2012}
}

@inproceedings{marasovic2017mention,
  title={A mention-ranking model for abstract anaphora resolution},
  author={Marasovi{\'c}, Ana and Born, Leo and Opitz, Juri and Frank, Anette},
  booktitle={EMNLP},
  year={2017}
}

@article{thalken2024rethinking,
  title={Rethinking Coreference Evaluation: A Case for Stratified Analysis},
  author={Thalken, Rosamond and others},
  journal={arXiv:2401.00238},
  year={2024}
}
```

