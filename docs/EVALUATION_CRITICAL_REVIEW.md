# Critical Evaluation Review: Research-Aligned Assessment

**Date**: 2025-01-XX  
**Context**: Review of evaluation implementation against latest research on NER/NED/Coref evaluation pitfalls

> **Related documents**:
> - [EVALUATION_CRITIQUE.md](EVALUATION_CRITIQUE.md) - Research-based evaluation limitations
> - [EVAL_CRITIQUE.md](EVAL_CRITIQUE.md) - Critique of specific evaluation results

## Executive Summary

The codebase has **strong architectural foundations** with awareness of many research findings, but several critical features are **defined but not computed** in actual evaluations. This review identifies:

1. ✅ **What's working well** (research-aware design)
2. ⚠️ **What's defined but unused** (missing implementation)
3. ❌ **What's missing entirely** (critical gaps)
4. 🔧 **What needs fixing** (implementation issues)

---

## 1. Coreference Metrics: Awareness vs. Implementation

### ✅ What's Good

- **All major metrics implemented**: MUC, B³, CEAF-e/m, LEA, BLANC, CoNLL F1
- **Research awareness**: Comments reference arXiv:2401.00238 (Thalken 2024) about CoNLL F1 being "uninformative"
- **Stratified structure defined**: `CorefChainStats` exists for per-chain-length breakdown

### ❌ Critical Gap: Chain-Length Stratification Not Computed

**Problem**: `CorefChainStats` is defined in `src/eval/types.rs` but **never computed** in actual evaluations.

**Research Context**: Thalken et al. (2024) show that single CoNLL F1 hides:
- Long chains (protagonists): Models excel (92% F1)
- Short chains (secondary): Models struggle (71% F1)  
- Singletons: Often ignored (45% F1)

**Current State**:
```rust
// Defined but never used:
pub struct CorefChainStats {
    pub long_chain_count: usize,
    pub short_chain_count: usize,
    pub singleton_count: usize,
    pub long_chain_f1: f64,
    pub short_chain_f1: f64,
    pub singleton_f1: f64,
}
```

**Impact**: All coreference evaluations report only aggregate CoNLL F1, hiding performance differences that matter for real applications.

**Recommendation**: 
1. Add `compute_chain_length_stratified()` function to `coref_metrics.rs`
2. Integrate into `CorefEvaluation::compute()` or as separate method
3. Report in evaluation outputs (not just aggregate)

---

## 2. Label Shift / Familiarity: Placeholder Implementation

### ✅ What's Good

- **Type defined**: `LabelShift` struct exists with research context (arXiv:2412.10121)
- **Documentation**: Excellent explanation of the problem (80%+ overlap inflates F1)

### ❌ Critical Gap: Familiarity Not Actually Computed

**Problem**: `LabelShift::from_type_sets()` uses **placeholder** familiarity calculation:

```rust
// Simple familiarity heuristic (proper version needs embeddings)
let familiarity = overlap_ratio; // Placeholder
```

**Research Context**: Golde et al. (2024) show that:
- Models trained on NuNER/PileNER have >88% Familiarity with common benchmarks
- "Zero-shot" claims are inflated by label overlap
- True familiarity = semantic similarity × frequency weighting (requires embeddings)

**Current State**: Only string-match overlap is computed, not semantic similarity.

**Impact**: Cannot detect when "zero-shot" evaluations are actually high-overlap transfer tasks.

**Recommendation**:
1. Implement proper familiarity using embeddings (e.g., sentence-transformers for label names)
2. Compute familiarity in `TaskEvaluator::evaluate()` before running evaluation
3. Report alongside F1 scores to contextualize results
4. Add warning when `familiarity > 0.85` (threshold from paper)

---

## 3. Boundary Error Double-Penalty Issue

### ✅ What's Good

- **Boundary error detection**: Correctly identifies overlapping but inexact spans
- **Error categorization**: Distinguishes boundary vs. type errors

### ⚠️ Potential Issue: Double Penalty

**Research Context**: AIDA-CoNLL analysis (D13-1027) shows systems are **double-penalized** for boundary mistakes:
- Boundary mismatch = FP + FN (two errors)
- Complete miss = FN only (one error)
- This penalizes "almost right" more than "completely wrong"

**Current Implementation** (`src/eval/ner_metrics.rs:300-415`):
```rust
// Uses greedy matching - each gold can match one pred, each pred can match one gold
// If boundary is wrong but type is right, it's counted as:
// - incorrect (strict mode)
// - partial (partial mode)
```

**Analysis**: The current implementation uses **greedy matching**, which means:
- A boundary error creates **one** incorrect match (not double-penalty)
- BUT: The unmatched gold becomes FN, unmatched pred becomes FP
- So boundary errors ARE effectively double-penalized

**Recommendation**:
1. Add "weak annotation matching" mode (requires URI match + position overlap, not exact boundaries)
2. Document the double-penalty behavior explicitly
3. Consider IoU-based partial credit for boundary errors

---

## 4. Inter vs. Intra-Document Coreference

### ✅ What's Good

- **Separation exists**: `inter_doc_coref.rs` vs. `coref.rs` (intra-doc)
- **Architecture documented**: `INTER_INTRA_DOC_ABSTRACTIONS.md` explains the distinction
- **Different complexity**: Recognizes O(n²) in documents vs. mentions

### ⚠️ Missing: Explicit Evaluation Separation

**Research Context**: Cross-document coref is fundamentally different:
- One-entity-per-name assumption fails (many "John Smiths")
- Requires more sophisticated approaches than surface matching
- Systems tend to overspecialize on target corpora

**Current State**: Both use same metrics (MUC, B³, CEAF), but:
- No explicit evaluation mode distinction
- No cross-document specific metrics
- No validation that inter-doc systems aren't overfitting

**Recommendation**:
1. Add `evaluate_inter_doc_coref()` with cross-document specific checks
2. Report generalization metrics (train/test domain split)
3. Add cross-document entity linking evaluation (NED integration)

---

## 5. Temporal Drift and Entity Evolution

### ❌ Missing Entirely

**Research Context**: 
- NER models become "stale" over time (temporal drift)
- TempEL dataset shows 3.1% accuracy drop for continual entities
- New entities (didn't exist in training) cause larger drops
- KB staleness: Systems using 2014 Wikipedia perform poorly on 2024 data

**Current State**: No temporal stratification or temporal evaluation.

**Recommendation**:
1. Add `temporal_stratification` to evaluation config
2. Track entity creation dates (if available in datasets)
3. Report performance by temporal stratum (pre/post KB cutoff)
4. Add temporal drift detection (performance degradation over time)

---

## 6. Robustness Testing: Exists But Not Integrated

### ✅ What's Good

- **Robustness module exists**: `src/eval/robustness.rs` with multiple perturbation types
- **Research awareness**: References Pacific AI (2024) on robustness testing

### ⚠️ Not Integrated into Main Evaluation

**Current State**: `RobustnessEvaluator` exists but:
- Not called in `TaskEvaluator::evaluate()`
- Not part of standard evaluation pipeline
- No automatic robustness testing in benchmarks

**Recommendation**:
1. Add `robustness: bool` flag to `TaskEvalConfig`
2. Integrate robustness testing into main evaluation loop
3. Report robustness scores alongside standard metrics
4. Add RUIE-Bench style LLM-generated perturbations (not just handcrafted)

---

## 7. Stratified Analysis: Partial Implementation

### ✅ What's Good

- **Per-type metrics**: `by_entity_type` in `BackendResults`
- **Stratified sampling**: `stratified_sample_ner()` exists
- **Domain breakdown**: `by_domain` in evaluation results

### ⚠️ Missing Dimensions

**Research Context**: Should stratify by:
- Entity type frequency (seen/rare/unseen in training)
- Temporal stratum (pre/post KB cutoff)
- Surface form type (proper noun/common noun/pronoun)
- Mention characteristics (capitalized vs. lowercased, partial vs. full names)

**Current State**: Only entity type and domain are stratified.

**Recommendation**:
1. Add `StratifiedMetrics` struct with all dimensions
2. Compute familiarity-based stratification (seen/rare/unseen)
3. Add surface form analysis (proper noun detection)
4. Report per-stratum metrics in evaluation outputs

---

## 8. Metric Reporting: Aggregate vs. Stratified

### ⚠️ Issue: Single Aggregate Scores

**Research Context**: Single F1 scores hide everything important:
- Class imbalance (90% accuracy by ignoring rare types)
- Boundary vs. type errors (different fixes needed)
- Chain-length differences (protagonists vs. background)

**Current State**: 
- `TaskEvalResult` stores single `metrics: HashMap<String, f64>`
- Reports aggregate F1, precision, recall
- Per-type breakdown exists but not always reported

**Recommendation**:
1. Always report per-type metrics (not just aggregate)
2. Add `StratifiedResults` to `TaskEvalResult`
3. Report macro AND micro F1 (currently only micro for NER)
4. Add confidence intervals for all metrics

---

## 9. KB Version Mismatch

### ❌ Missing

**Research Context**: GERBIL framework addresses:
- KB-agnostic matching via `owl:sameAs` links
- Deprecation handling (check if gold URIs still exist)
- Emerging entity classification (EE/NIL performance separate from in-KB)

**Current State**: No KB version tracking or URI validation.

**Recommendation**:
1. Add KB version metadata to datasets
2. Validate gold URIs against current KB
3. Separate emerging entity (NIL) performance from in-KB performance
4. Use URI set expansion for matching (handle DBpedia 2014 vs. 2019 URIs)

---

## 10. Zero-Shot Evaluation: Binary vs. Spectrum

### ⚠️ Issue: Treating Zero-Shot as Binary

**Research Context**: Zero-shot is a **spectrum**, not binary:
- If training has "Person", "Human", "Individual" and test has "Person" → not zero-shot
- Correlation between training entity type frequency and per-type F1 is consistently positive

**Current State**: 
- Zero-shot backends (NuNER, GLiNER) are identified
- But no familiarity computation to validate "true" zero-shot

**Recommendation**:
1. Compute familiarity for all "zero-shot" evaluations
2. Report familiarity score alongside F1
3. Warn when familiarity > 0.8 (high overlap, not true zero-shot)
4. Generate test splits at varying difficulty (low/medium/high label shift)

---

## Priority Recommendations

### High Priority (Critical Gaps)

1. **Compute chain-length stratification** for coreference
   - Add `compute_chain_length_stratified()` 
   - Integrate into evaluation pipeline
   - Report in outputs

2. **Implement proper familiarity computation**
   - Use embeddings for semantic similarity
   - Compute in `TaskEvaluator::evaluate()`
   - Report alongside F1 scores

3. **Add temporal stratification**
   - Track entity creation dates
   - Report by temporal stratum
   - Detect temporal drift

### Medium Priority (Important Improvements)

4. **Integrate robustness testing** into main evaluation
   - Add flag to `TaskEvalConfig`
   - Run automatically in benchmarks
   - Report robustness scores

5. **Expand stratified analysis**
   - Add familiarity-based stratification
   - Surface form analysis
   - Mention characteristics

6. **Fix boundary error double-penalty**
   - Document current behavior
   - Add weak annotation matching option
   - Consider IoU-based partial credit

### Low Priority (Nice to Have)

7. **KB version tracking**
   - URI validation
   - Emerging entity separation
   - URI set expansion

8. **Cross-document specific evaluation**
   - Inter-doc specific metrics
   - Generalization validation
   - NED integration

---

## Implementation Checklist

- [ ] Add `compute_chain_length_stratified()` to `coref_metrics.rs`
- [ ] Implement proper familiarity using embeddings
- [ ] Add temporal stratification to evaluation
- [ ] Integrate robustness testing into `TaskEvaluator`
- [ ] Expand `StratifiedMetrics` with all dimensions
- [ ] Add familiarity computation to zero-shot evaluations
- [ ] Document boundary error double-penalty behavior
- [ ] Add KB version tracking and URI validation
- [ ] Report stratified metrics in all evaluation outputs
- [ ] Add confidence intervals to all metrics

---

## References

- Thalken et al. (2024): "How to Evaluate Coreference in Literary Texts?" arXiv:2401.00238
- Golde et al. (2024): "Familiarity" - Label overlap bias in zero-shot eval. arXiv:2412.10121
- AIDA-CoNLL analysis: D13-1027 (boundary errors, domain bias)
- GERBIL framework: Semantic Web Journal (KB version handling)
- RUIE-Bench: Realistic perturbations for unified IE evaluation
- TempEL: Temporal entity linking evaluation

