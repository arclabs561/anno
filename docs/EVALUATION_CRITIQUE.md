# Evaluation Critiques and Limitations

A research-grounded perspective on the limitations of NER and coreference evaluation.

## TL;DR

- **47% of "errors" on CoNLL-03 are actually correct** — annotation noise, not model mistakes (CleanCoNLL, EMNLP 2023)
- **7-10% of annotations in standard benchmarks are wrong** — CoNLL-03 (Rücker 2023), OntoNotes (Bernier-Colborne 2024)
- **Single scores hide everything important** — aggregate F1 conflates boundary errors, type errors, and dataset artifacts
- **Cross-dataset evaluation is unreliable** — scores measure *definition differences*, not generalization (Porada et al., ACL Findings 2024)

---

## Part 1: NER Evaluation Problems

### The Benchmark Noise Problem

The most cited NER benchmarks contain significant annotation errors:

| Dataset | Error Rate | Source |
|---------|-----------|--------|
| CoNLL-03 | 7.0% of labels | CleanCoNLL (Rücker & Akbik, EMNLP 2023) |
| OntoNotes 5.0 | ~8% of entities | Bernier-Colborne & Vajjala (2024) |
| WikiNER | >10% (semi-supervised) | WikiNER-fr-gold (Cao et al., 2024) |

**Consequence**: State-of-the-art models now exceed the dataset's noise floor. On CoNLL-03, **47% of "errors" scored by traditional metrics were actually correct predictions** penalized by annotation mistakes. After cleaning: F1 jumped from ~94% to 97.1%.

*Implication for `anno`*: Our synthetic datasets have verified annotations and should outperform real benchmarks for error analysis. We track annotation provenance.

### The Glass Ceiling Problem

> "Is there a glass ceiling? Do we know which types of errors are still hard or even impossible to correct?"
> — Stanislawek et al., 2019

Error analysis of SOTA models reveals:
- **Seen vs unseen entities**: Models memorize training entities; unseen names cause most errors
- **Type-confusable mentions**: "Washington" (person, location, organization)
- **Boundary ambiguity**: "New York Times" vs "New York"

*`anno` addresses this via*: `dataset_quality.rs` tracks unseen entity ratio, entity ambiguity, and type distribution skew.

### The F1 Tunnel Vision Problem

F1 score obscures actionable information:

| What F1 hides | Why it matters |
|--------------|----------------|
| Boundary vs type errors | Different fixes needed |
| Rare entity performance | Macro vs micro averages differ wildly |
| Error severity | Missing "John" ≠ missing "Dr. John Smith, CEO" |
| Cross-document consistency | Same entity tagged differently in different sentences |

*`anno` addresses this via*:
- `ner_metrics.rs`: Separate strict/exact/partial/type scores
- `error_analysis.rs`: Fine-grained error taxonomy
- `backend_eval.rs`: Per-domain and per-difficulty breakdowns

### The TMR (Tough Mentions Recall) Problem

Traditional evaluation weights all mentions equally. But models struggle specifically with:

1. **Unseen mentions**: Tokens/types not in training
2. **Type-confusable mentions**: Could be multiple entity types
3. **Rare types**: Low-frequency entity classes

*`anno` addresses this via*: `dataset_quality.rs` computes unseen ratio and type confusion metrics.

---

## Part 2: Coreference Evaluation Problems

### The Measurement Validity Problem

> "Measurements intended to reflect CR model generalization are often correlated with differences in both how coreference is defined and how it is operationalized."
> — Porada et al., ACL Findings 2024

When Model A outperforms Model B across datasets, this might reflect:
- Model A is genuinely better (what we want to measure)
- Model A was trained on data closer to the test definition
- Test datasets happen to include phenomena Model A handles well

**Example**: A model trained on OntoNotes fails on "trees" coreference in PreCo — not because it can't generalize, but because OntoNotes doesn't annotate generic nouns as coreferring.

### The Chain-Length Problem

> "A unique score cannot represent the full complexity of the problem at stake."
> — Duron-Tejedor et al., CHR 2023

Standard metrics (MUC, B³, CEAF) average over all chains. But:
- **Long chains** (main characters): Lots of redundant signal
- **Short chains** (secondary characters): Sparse, harder
- **Singletons**: Often ignored entirely

*Proposed*: Disaggregate by chain length. Report performance on main characters separately from secondary ones.

### The Boundary Dependency Problem

> "The current CoNLL score calculation heavily relies on accurate boundary of the coreference mention... failing to fully reflect the LLMs' understanding of coreference."
> — Wang et al., LREC 2024

LLMs understand "Elon Musk" and "the Tesla CEO" are coreferent even if they can't produce exact span boundaries. Traditional metrics penalize this.

*`anno` addresses this via*: Partial matching modes in evaluation.

### The Cross-Document Problem

> "Common evaluation practices for cross-document coreference resolution have been unrealistically permissive."
> — Cattan et al., 2021

Cross-document coreference is harder than within-document:
- No syntactic cues across documents
- Entity disambiguation required
- Evaluation protocols often don't properly test this

---

## Part 3: LLM-Era Evaluation Gaps

### Hallucination Detection

LLMs introduce a new failure mode: generating entity mentions that don't exist in the source text at all.

| Problem | Traditional metrics | Modern approach |
|---------|-------------------|-----------------|
| Hallucinated spans | Counted as spurious (SPU) | Need source verification |
| Wrong positions | Conflated with boundary error | Distinguish position vs boundary |
| Confident errors | Not measured | Calibration analysis |

*`anno` addresses this via*: `calibration.rs` for confidence analysis.

### Semantic Similarity

LLM outputs may be semantically correct but textually different:
- Gold: "Barack Obama"
- Prediction: "the 44th U.S. President"

Both correct, but strict matching fails.

*Emerging approaches*:
- LLM-as-judge evaluation
- Embedding similarity thresholds
- Entity linking verification

---

## Part 4: What `anno` Does About This

### Modern Metrics Available

| Module | What it measures | Research basis |
|--------|-----------------|----------------|
| `dataset_quality.rs` | Unseen ratio, ambiguity, imbalance | TMR (Tu & Lignos, 2021) |
| `ner_metrics.rs` | Strict/Exact/Partial/Type | SemEval-2013 (legacy, for comparison) |
| `error_analysis.rs` | Error taxonomy | SeqScore (Palen-Michel et al., 2021) |
| `calibration.rs` | Confidence calibration | Standard ML practice |
| `coref_resolver.rs` | MUC/B³/CEAF/LEA/BLANC | CoNLL-2012 standard |

### Synthetic Dataset Advantages

Our synthetic datasets avoid benchmark contamination issues:
- Verified annotations (human-checked offsets)
- Known difficulty levels
- Domain-specific test cases
- No train/test leakage

### Recommended Evaluation Workflow

```
1. Quick validation: Run on synthetic datasets (fast, free, no network)
2. Error analysis: Use error taxonomy to understand failure modes
3. Cross-domain: Test on multiple synthetic domains
4. Dataset quality: Check for unseen entities, ambiguity
5. Benchmark: Run on CoNLL/OntoNotes for paper comparisons (use legacy metrics)
6. Clean benchmark: Use CleanCoNLL for true error analysis
```

---

## Part 5: Coreference Definition Inconsistencies

### The Core Problem

> "Measurements intended to reflect CR model generalization are often correlated with differences in both how coreference is defined and how it is operationalized."
> — Porada et al., ACL Findings 2024

Different datasets define "coreference" differently:

| Dataset | Singletons | Generic NPs | Event Coref | Bridging |
|---------|------------|-------------|-------------|----------|
| **OntoNotes** | ✗ Excluded | ✗ Excluded | ✗ Excluded | ✗ Excluded |
| **PreCo** | ✓ Included | ✓ Included | ✗ Excluded | ✗ Excluded |
| **LitBank** | ✓ Included | Partial | ✗ Excluded | ✗ Excluded |
| **ECB+** | ✓ Included | ✗ Excluded | ✓ Included | ✗ Excluded |
| **ARRAU** | ✓ Included | ✓ Included | ✗ Excluded | ✓ Included |

### Practical Implications

1. **Cross-dataset evaluation is unreliable**: A model trained on OntoNotes may fail on PreCo not because it's worse, but because PreCo annotates singletons.

2. **Metric choice matters by definition**:
   - MUC: Ignores singletons → favors OntoNotes-style
   - B³: Counts singletons → inflates scores on PreCo-style
   - BLANC: Rewards non-links → best for comparing across definitions

3. **Generic NP handling differs**: "Dogs are loyal" - is "Dogs" coreferent across sentences? OntoNotes says no, PreCo says yes.

### `anno` Approach

We handle definition inconsistencies via:

1. **Dataset-aware evaluation**: `DatasetId::type_mapper()` normalizes labels per dataset
2. **Configurable singleton handling**: `CorefConfig::include_singletons`
3. **Multiple metrics reported**: Always report MUC, B³, CEAF, LEA, BLANC
4. **Chain-length stratification**: `CorefChainStats` for per-length analysis

### Recommendations

```rust
// Always report multiple metrics
let eval = CorefEvaluation::compute(&predicted, &gold);
println!("MUC F1: {:.1}%", eval.muc.f1 * 100.0);
println!("B³ F1: {:.1}%", eval.b_cubed.f1 * 100.0);
println!("BLANC F1: {:.1}%", eval.blanc.f1 * 100.0);

// For cross-dataset: prefer BLANC (handles all edge cases)
// For OntoNotes compat: use CoNLL F1
// For detailed analysis: use chain-length stratified metrics
```

---

## Part 6: Document-Level Context

### The Context Window Problem

Most NER models process fixed-length windows (512 tokens). But:

- **Long documents**: Entity mentions may be separated by thousands of tokens
- **Coreference**: Pronouns may refer to entities from previous paragraphs
- **Events**: Multi-sentence event descriptions require cross-sentence context

### Current `anno` Support

1. **Streaming extraction**: `Model::extract_entities_streaming()` for chunked processing
2. **Offset preservation**: Character offsets maintained across chunks
3. **CDCR**: Cross-document coreference via `CDCRResolver`

### Best Practices

```rust
// For long documents: use streaming extraction
let entities = model.extract_entities_streaming(&long_text, 512, 64)?;
// 512 = chunk size, 64 = overlap

// For cross-document: use CDCR
let resolver = CDCRResolver::with_config(CDCRConfig {
    use_lsh: true,  // Efficient blocking
    min_similarity: 0.5,
    ..Default::default()
});
let clusters = resolver.resolve(&documents);
```

---

## Research References

### NER Dataset Quality

- **CleanCoNLL** (2023): "7.0% of all labels in English CoNLL-03" corrected. [arXiv:2310.16225](https://arxiv.org/abs/2310.16225)
- **OntoNotes Errors** (2024): "~8% of mentions" corrected. [arXiv:2406.19172](https://arxiv.org/abs/2406.19172)
- **CoNLL#** (2024): Fine-grained error analysis. [arXiv:2405.11865](https://arxiv.org/abs/2405.11865)
- **NoiseBench** (2024): Real label noise impact. [arXiv:2405.07609](https://arxiv.org/abs/2405.07609)

### NER Evaluation Methodology

- **TMR** (2021): Tough Mentions Recall. [arXiv:2103.12312](https://arxiv.org/abs/2103.12312)
- **SeqScore** (2021): Reproducibility crisis. [arXiv:2107.14154](https://arxiv.org/abs/2107.14154)
- **Glass Ceiling** (2019): Error type analysis. [arXiv:1910.02403](https://arxiv.org/abs/1910.02403)
- **Human Label Variation** (2024): Annotation disagreement sources. [arXiv:2402.01423](https://arxiv.org/abs/2402.01423)

### Coreference Evaluation

- **Measurement Modeling** (2024): Generalization validity. [arXiv:2303.09092](https://arxiv.org/abs/2303.09092)
- **Literary Coref** (2023): Chain-length disaggregation. [arXiv:2401.00238](https://arxiv.org/abs/2401.00238)
- **Cross-Document** (2021): Realistic evaluation. [arXiv:2106.04192](https://arxiv.org/abs/2106.04192)
- **OntoNotes Scope** (2021): Definition problems. [arXiv:2112.09742](https://arxiv.org/abs/2112.09742)

### LLM Evaluation

- **Span Hallucination** (SemEval-2025): Task 3 Mu-SHROOM
- **LLM Coreference** (2024): Boundary dependency. LREC-2024.

---

## Key Takeaways

1. **Report multiple metrics**: Strict F1 alone is insufficient
2. **Analyze errors, not just scores**: Error taxonomy > aggregate numbers
3. **Know your dataset's noise level**: Many benchmarks are ~7-10% wrong
4. **Test generalization explicitly**: Same-corpus F1 ≠ real-world performance
5. **Consider chain/entity properties**: Disaggregate by difficulty
6. **Document definitional choices**: What counts as an entity? As coreference?

