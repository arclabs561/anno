# Original Goals and Motivations Review

**Date**: 2025-01-XX  
**Purpose**: Review the original research findings, motivations, and goals to ensure implementation aligns with intent

## Original Research Findings

### 1. Evaluation Metric Problems

**Finding**: Standard coreference metrics (MUC, B³, CEAF, LEA) have significant flaws:
- MUC is least discriminative, can give non-zero scores even when no correct relations exist
- B³ suffers from "mention identification effect" - singletons inflate scores
- CEAFe shows unreasonable drops when correct link ratios change
- CoNLL F1 (average) doesn't eliminate individual metric biases

**Implementation Status**: ✅ **Addressed**
- Added chain-length stratification to identify where metrics fail
- `CorefChainStats` breaks down performance by chain length (long/short/singleton)
- Helps identify if aggregate CoNLL F1 is misleading due to singleton inflation

### 2. Benchmark Artifacts and Biases

**Finding**: Entity linking benchmarks suffer from:
- Named entity focus (missing professions, diseases, genres)
- Inconsistent annotation guidelines
- Ambiguity handling (multiple valid links penalized)
- Domain bias (e.g., AIDA-CoNLL 44% sports)
- Span boundary errors (double-penalization)

**Implementation Status**: ⚠️ **Partially Addressed**
- ✅ Documented boundary error handling (greedy matching, no double-penalization)
- ⚠️ Familiarity computation detects label shift but doesn't address domain bias
- ❌ No ambiguity handling (multiple valid links still penalized)
- ❌ No domain stratification

### 3. Intra vs. Cross-Document Coreference

**Finding**: 
- Intra-doc: One-entity-per-name assumption often works
- Cross-doc: One-entity-per-name unreasonable, requires sophisticated approaches
- Cross-doc systems overspecialize on target corpora

**Implementation Status**: ❌ **Not Addressed**
- Current implementation treats all coref the same
- No distinction between intra-doc and cross-doc evaluation
- No generalization validation (train/test domain split)

### 4. Temporal Drift and Entity Evolution

**Finding**:
- NER models become "stale" over time (temporal drift)
- Entity linking temporal decay (3.1% accuracy drop over time)
- Context drift (vocabulary changes faster than entity descriptions)
- KB staleness (older Wikipedia dumps perform poorly on newer data)

**Implementation Status**: ⚠️ **Structure Ready, No Data**
- `StratifiedMetrics` has `by_temporal_stratum` field
- No temporal metadata in datasets
- No KB version tracking
- Structure exists but cannot compute without temporal data

### 5. Entity Identity Changes Across Dimensions

**Finding**:
- Temporal evolution (job changes, relocations, name changes)
- Metonymy ("France" → team vs country) - 30-100% error rates
- Partial names ("Biden" vs "President Biden") - 16-65% error rates
- Generic vs. specific mentions ("the president" ambiguity)

**Implementation Status**: ❌ **Not Addressed**
- No metonymy detection
- No partial name handling
- No generic vs. specific distinction
- These are model capabilities, not evaluation features

### 6. Misinformation, Noise, and Adversarial Robustness

**Finding**:
- Adversarial attacks (character-level perturbations, attention manipulation)
- Social media NER degradation
- Defense frameworks (DINA-style dual defense)

**Implementation Status**: ✅ **Addressed**
- `RobustnessEvaluator` tests perturbations (typos, case, whitespace, punctuation, unicode)
- Integrated into evaluation pipeline
- Computes robustness scores (baseline vs perturbed F1)

### 7. NER → Knowledge Graph Boundary

**Finding**: Continuum from NER → NED → RE → KG population

**Implementation Status**: ⚠️ **Partially Addressed**
- NER evaluation: ✅ Complete
- NED evaluation: ❌ Not implemented
- RE evaluation: ✅ Basic implementation
- KG population: ❌ Not addressed

### 8. Joint Extraction and LLM-Based Unified IE

**Finding**: Shift toward unified generative frameworks (ReasoningNER, Reamend, Code4UIE)

**Implementation Status**: ❌ **Not Addressed**
- Current evaluation treats NER, RE, Coref as separate tasks
- No joint evaluation framework
- No reasoning chain evaluation

### 9. Multimodal Entity Recognition and Linking

**Finding**: GMNER, VP-MEL, GEMEL for visual entity recognition

**Implementation Status**: ❌ **Not Addressed**
- Text-only evaluation
- No multimodal support

## Original Request: "How would we want to design it? How would we want to avoid doing it?"

### Design Principles (What We Did)

1. **Stratified Evaluation**: Break down performance across dimensions
   - ✅ Chain-length stratification for coref
   - ✅ Entity type stratification (structure ready)
   - ⚠️ Temporal stratification (structure ready, no data)

2. **Familiarity Computation**: Detect label shift and zero-shot inflation
   - ✅ String-based familiarity
   - ✅ True zero-shot type detection
   - ⚠️ Embedding-based familiarity (function exists, not integrated)

3. **Robustness Testing**: Evaluate under perturbations
   - ✅ Perturbation types (typos, case, whitespace, punctuation, unicode)
   - ✅ Robustness score computation
   - ✅ Integration into evaluation pipeline

4. **Confidence Intervals**: Quantify uncertainty
   - ✅ Per-example score computation (sampled)
   - ✅ CI computation from variance
   - ⚠️ Could be more efficient (recomputes predictions)

5. **Boundary Error Documentation**: Clarify evaluation behavior
   - ✅ Documented greedy matching strategy
   - ✅ Explained no double-penalization

### Anti-Patterns (What We Avoided)

1. **Averaging Flawed Metrics**: ✅ Addressed
   - Chain-length stratification reveals where aggregate metrics fail
   - Don't just report CoNLL F1, break it down

2. **Ignoring Label Shift**: ✅ Addressed
   - Familiarity computation detects when "zero-shot" isn't truly zero-shot
   - Warns about inflated performance

3. **No Robustness Testing**: ✅ Addressed
   - Perturbation testing reveals brittleness
   - Models that fail on typos are not production-ready

4. **No Confidence Intervals**: ✅ Addressed
   - CI computation quantifies uncertainty
   - Helps distinguish real improvements from noise

5. **Ignoring Temporal Drift**: ⚠️ Structure Ready
   - Temporal stratification structure exists
   - Needs temporal metadata in datasets

## Implementation Completeness

### Fully Implemented ✅
1. Chain-length stratification for coreference
2. Familiarity computation (string-based)
3. Robustness testing integration
4. Confidence interval computation (sampled)
5. Boundary error documentation
6. Test coverage for new features

### Partially Implemented ⚠️
1. Stratified metrics (structure ready, computation placeholder)
2. Temporal stratification (structure ready, no data source)
3. Embedding-based familiarity (function exists, not integrated)
4. Reporting stratified metrics (computed but not fully reported)

### Not Implemented ❌
1. KB version tracking and URI validation
2. Inter-document coreference specific evaluation
3. Metonymy and partial name handling
4. Joint extraction evaluation
5. Multimodal evaluation
6. Domain bias detection
7. Ambiguity handling (multiple valid links)

## Alignment with Original Goals

**Primary Goal**: Design evaluation system that avoids common pitfalls and provides comprehensive insights.

**Achievement**: ✅ **Core Goals Met**
- Stratified evaluation reveals where models fail
- Familiarity computation detects inflated zero-shot claims
- Robustness testing identifies brittleness
- Confidence intervals quantify uncertainty
- Documentation clarifies evaluation behavior

**Gaps**: ⚠️ **Some Advanced Features Missing**
- Temporal stratification needs data
- Inter-doc coref needs specialized evaluation
- Some research findings (metonymy, partial names) are model capabilities, not evaluation features

## Recommendations

### High Priority
1. **Add temporal metadata to datasets** - Enable temporal stratification
2. **Complete stratified metrics computation** - Use per-example scores
3. **Report all stratified metrics** - Include in markdown outputs

### Medium Priority
4. **Integrate embedding-based familiarity** - Use encoder backends
5. **Add KB version tracking** - Track KB staleness
6. **Inter-doc coref evaluation** - Specialized metrics

### Low Priority
7. **Joint extraction evaluation** - Unified framework
8. **Multimodal evaluation** - Visual entity recognition
9. **Domain bias detection** - Stratify by domain

## Conclusion

The implementation successfully addresses the **core evaluation pitfalls** identified in the research:
- ✅ Metric flaws (chain-length stratification)
- ✅ Label shift (familiarity computation)
- ✅ Robustness (perturbation testing)
- ✅ Uncertainty (confidence intervals)
- ✅ Boundary errors (documentation)

Some **advanced features** remain incomplete due to:
- Missing data sources (temporal metadata)
- Model capability requirements (metonymy, partial names)
- Scope limitations (multimodal, joint extraction)

The architecture is sound and extensible - remaining work is primarily integration and data collection.

