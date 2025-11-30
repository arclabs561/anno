# Comprehensive Benchmark Analysis

**Date**: 2025-01-25  
**Scope**: All NER backends across 5 datasets (WikiGold, Wnut17, CoNLL2003Sample, OntoNotesSample, MultiNERD)

## Executive Summary

### Performance Ranking

| Rank | Backend | Avg F1 | Datasets | Status |
|------|---------|--------|----------|--------|
| 1 | `bert_onnx` | **55.4%** | 10 | ✅ Production-ready |
| 2 | `gliner_onnx` | **42.0%** | 10 | ✅ Zero-shot capable |
| 3 | `heuristic` | **33.3%** | 1 | ✅ Baseline |
| 4 | `stacked` | **31.6%** | 6 | ✅ No ML dependencies |
| 5 | `nuner` | **0.0%** | 1 | ❌ **Critical issue** |
| 6 | `pattern` | **0.0%** | 1 | ℹ️ Expected (structured only) |

## Key Findings

### 1. bert_onnx: Best Overall Performance

**Strengths:**
- Highest average F1 (55.4%) across all datasets
- Excellent on CoNLL2003 (74.9%) and OntoNotes (71.7%)
- Fast inference (~50ms)
- Reliable, production-tested model

**Weaknesses:**
- Fixed entity types (PER/ORG/LOC/MISC) - no zero-shot capability
- Struggles on domain-mismatched datasets (Wnut17: 27.1%)
- Cannot extract custom entity types

**Use Case**: Standard NER tasks with PER/ORG/LOC/MISC entities

### 2. gliner_onnx: Zero-Shot Champion

**Strengths:**
- Zero-shot capability - can extract any entity type at runtime
- Consistent performance across datasets (39-50% range)
- Best on MultiNERD (48.7%) - outperforms bert_onnx
- Flexible for custom entity types

**Weaknesses:**
- Lower average F1 than bert_onnx (42.0% vs 55.4%)
- Slower inference (~100ms)
- Requires ONNX feature

**Use Case**: Custom entity types, cross-domain NER, when flexibility > raw performance

### 3. stacked: Solid Baseline

**Strengths:**
- No ML dependencies - works everywhere
- Combines pattern + heuristic backends
- Reasonable performance (31.6% avg F1)
- Good fallback when ML unavailable

**Weaknesses:**
- Lower accuracy than ML backends
- Limited to rule-based patterns

**Use Case**: Fallback when ML unavailable, offline scenarios, baseline comparisons

### 4. nuner: Critical Issue (0% F1)

**Root Cause:**
- Model loads successfully (verified manually)
- Manual extraction works (tested: "Barack Obama was born in Hawaii" → finds entities)
- Evaluation returns 0% F1 due to **label mismatch**

**Technical Details:**
- `Model::extract_entities()` uses default labels: `["person", "organization", "location"]`
- Datasets use different labels:
  - WikiGold/CoNLL2003: `["PER", "LOC", "ORG", "MISC"]`
  - Wnut17: `["person", "location", "corporation", "product", ...]`
  - OntoNotes: `["PERSON", "ORG", "GPE", "LOC", ...]`
- NuNER is zero-shot but evaluation doesn't pass dataset labels

**Fix Required:**
1. Extract entity types from dataset (via `dataset.entity_types()`)
2. For zero-shot models, call `NuNER::extract(text, labels, threshold)` instead of `extract_entities()`
3. Map dataset labels to NuNER-compatible labels (e.g., "PER" → "person")

## Dataset-Specific Observations

### CoNLL2003Sample: Best Performance

All backends perform best on this dataset:
- `bert_onnx`: 74.9% (trained on CoNLL)
- `gliner_onnx`: 50.2% (zero-shot)
- `stacked`: 35.9% (baseline)

**Implication**: CoNLL2003 is well-represented in training data for most models.

### Wnut17: Most Challenging

All backends struggle on this social media dataset:
- `bert_onnx`: 27.1% (domain mismatch)
- `gliner_onnx`: 29.5% (zero-shot helps slightly)
- `stacked`: 14.1% (baseline struggles)

**Implication**: Social media text is significantly different from news/Wikipedia training data.

### MultiNERD: Balanced Performance

Interesting case where zero-shot outperforms fixed-type:
- `gliner_onnx`: 48.7% (best!)
- `bert_onnx`: 46.5%
- `stacked`: 39.5%

**Implication**: Zero-shot models can adapt better to diverse entity types.

## Architectural Implications

### 1. Model Trait Limitation

**Current State:**
- `Model::extract_entities(text, language)` doesn't accept entity type labels
- Zero-shot models (NuNER, GLiNER) need label specification
- Evaluation uses default labels, causing mismatches

**Proposed Solution:**
```rust
trait Model {
    // Existing method (backward compatible)
    fn extract_entities(&self, text: &str, language: Option<&str>) -> Result<Vec<Entity>>;
    
    // New method for zero-shot models
    fn extract_entities_with_labels(
        &self,
        text: &str,
        labels: &[&str],
        language: Option<&str>
    ) -> Result<Vec<Entity>> {
        // Default: ignore labels, use extract_entities()
        self.extract_entities(text, language)
    }
}
```

**Alternative**: Handle in evaluation framework by detecting zero-shot backends and calling their specific `extract()` methods.

### 2. Evaluation Framework Enhancement

**Current State:**
- Calls `backend.extract_entities()` for all backends
- Doesn't extract dataset entity types
- Doesn't map labels between datasets and models

**Required Changes:**
1. Extract entity types from dataset: `dataset.entity_types()`
2. Detect zero-shot backends (NuNER, GLiNER)
3. For zero-shot: call `extract(text, labels, threshold)`
4. For fixed-type: continue using `extract_entities()`
5. Map dataset labels to model labels (e.g., "PER" → "person")

### 3. Label Mapping Strategy

**Dataset Labels → Model Labels:**
- WikiGold/CoNLL2003: `PER` → `person`, `ORG` → `organization`, `LOC` → `location`
- Wnut17: `person` → `person` (direct), `corporation` → `organization`
- OntoNotes: `PERSON` → `person`, `GPE` → `location`, etc.

**Implementation:**
- Create label mapping function in evaluation framework
- Support both exact matches and fuzzy matching
- Document mapping requirements per backend

## Recommended Fixes

### Priority 1: Fix NuNER Evaluation (Critical)

**Location**: `src/eval/task_evaluator.rs::evaluate_ner_task()`

**Changes:**
1. Extract entity types from dataset:
   ```rust
   let dataset_labels: Vec<String> = dataset.entity_types()
       .iter()
       .map(|s| s.to_string())
       .collect();
   ```

2. Detect zero-shot backends and call appropriate method:
   ```rust
   let entities = if backend_name == "nuner" || backend_name == "gliner_onnx" {
       // For zero-shot models, use extract() with dataset labels
       if let Ok(nuner) = backend.downcast_ref::<NuNER>() {
           nuner.extract(&text, &dataset_labels, 0.5)?
       } else {
           backend.extract_entities(&text, None)?
       }
   } else {
       backend.extract_entities(&text, None)?
   };
   ```

3. Map dataset labels to model-compatible labels before calling `extract()`

### Priority 2: Enhance Model Trait (Optional)

**Location**: `src/lib.rs::Model`

**Changes:**
- Add `extract_entities_with_labels()` method
- Zero-shot models implement both methods
- Fixed-type models use default implementation

**Benefits:**
- Cleaner API for zero-shot models
- Better type safety
- More explicit about label requirements

### Priority 3: Update Documentation

**Location**: `README.md`, `docs/`, backend-specific docs

**Changes:**
- Document label requirements per backend
- Add examples showing zero-shot usage
- Explain label mapping for evaluation

## Performance Benchmarks

### By Dataset

| Dataset | bert_onnx | gliner_onnx | stacked | Best |
|---------|-----------|-------------|---------|------|
| CoNLL2003Sample | **74.9%** | 50.2% | 35.9% | bert_onnx |
| OntoNotesSample | **71.7%** | 46.6% | 34.6% | bert_onnx |
| WikiGold | **53.4%** | 39.5% | 32.7% | bert_onnx |
| MultiNERD | 46.5% | **48.7%** | 39.5% | gliner_onnx |
| Wnut17 | 27.1% | **29.5%** | 14.1% | gliner_onnx |

### By Use Case

**Standard NER (PER/ORG/LOC/MISC):**
- Best: `bert_onnx` (55.4% avg)
- Fallback: `stacked` (31.6% avg)

**Custom Entity Types:**
- Best: `gliner_onnx` (42.0% avg, zero-shot)
- Note: `nuner` should work but needs evaluation fix

**No ML Dependencies:**
- Best: `stacked` (31.6% avg)
- Alternative: `heuristic` (33.3% on WikiGold)

**Cross-Domain NER:**
- Best: `gliner_onnx` (zero-shot adapts better)
- Example: MultiNERD (48.7% vs 46.5% for bert_onnx)

## Conclusion

1. **bert_onnx** is the best choice for standard NER tasks with fixed entity types
2. **gliner_onnx** is the best choice for custom entity types and cross-domain scenarios
3. **stacked** provides a solid baseline without ML dependencies
4. **nuner** has a critical evaluation bug that needs fixing (0% F1 is incorrect)

**Next Steps:**
1. Fix NuNER evaluation (Priority 1)
2. Consider Model trait enhancement (Priority 2)
3. Update documentation (Priority 3)

