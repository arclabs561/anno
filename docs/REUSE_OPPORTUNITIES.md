# Reuse Opportunities: Accidental Optimizations

## The User's Insight

Instead of adding new caching infrastructure, look for places where we're **already computing or storing intermediate representations** that could be reused.

## Key Finding: Evaluation Loop Structure

Looking at `evaluate_all`:
```rust
for task in tasks {
    for dataset in datasets {
        for backend in backends {
            evaluate_combination(task, dataset, backend, config)
        }
    }
}
```

**The same text is processed by multiple backends!**

Example:
- Text: "Apple Inc. was founded by Steve Jobs in 1976."
- PatternNER processes it
- HeuristicNER processes it  
- GLiNER processes it
- All tokenize/encode the same text independently

## Concrete Reuse Opportunities

### 1. `per_example_scores_cache` - Already Has Text

**Location**: `src/eval/task_evaluator.rs:218-219`

**Current**: Stores `(Vec<Entity>, Vec<Entity>, String)` - gold, predicted, text

**Opportunity**: When computing stratified metrics or confidence intervals, we iterate over `per_example_scores` multiple times. If we stored intermediate representations alongside:

```rust
struct PerExampleResult {
    gold: Vec<Entity>,
    predicted: Vec<Entity>,
    text: String,
    // NEW: Intermediate representations
    text_embeddings: Option<Vec<f32>>,  // Token embeddings (if available)
    prompt_encoding: Option<PromptEncoding>,  // GLiNER prompt encoding (if available)
}
```

**When to reuse**:
- Re-computing metrics from cached results
- Robustness testing (same text, different perturbations)
- Re-running evaluation with different thresholds

**Challenge**: Memory cost - embeddings are large (~768 * seq_len * 4 bytes per example)

### 2. Robustness Testing - Same Text, Multiple Perturbations

**Location**: `src/eval/task_evaluator.rs:compute_robustness`

**What happens**:
```rust
// For each example
let text = example.text();
let perturbed = apply_perturbation(text, Perturbation::Typo);
let entities = backend.extract_entities(&perturbed, None)?;
```

**Opportunity**: If the perturbation doesn't change tokenization significantly (e.g., case change, whitespace), we could reuse token embeddings.

**Example**:
- Original: "Apple Inc. was founded..."
- Perturbed: "apple inc. was founded..." (case change)
- Tokenization might be identical → can reuse embeddings

**Challenge**: Need to detect when perturbation affects tokenization vs. when it doesn't.

### 3. Evaluation Across Backends - Shared Text Embeddings

**Location**: `src/eval/task_evaluator.rs:evaluate_all`

**What happens**:
```rust
for backend in backends {
    // Each backend processes the same dataset
    evaluate_combination(task, dataset, backend, config)
}
```

**Opportunity**: If multiple backends use the same tokenizer/encoder, we could share embeddings at the `TaskEvaluator` level:

```rust
pub struct TaskEvaluator {
    // ... existing fields ...
    shared_text_cache: Arc<Mutex<HashMap<String, TextEmbeddings>>>,  // text -> embeddings
}
```

**When to use**:
- Same text processed by multiple backends with same tokenizer
- GLiNER and GLiNER2 might share tokenizer
- BERT-based backends might share tokenizer

**When NOT to use**:
- Different tokenizers (PatternNER vs. GLiNER)
- Different encoders (BERT vs. ModernBERT)

### 4. GLiNER - Same (text, entity_types) with Different Thresholds

**Location**: `src/backends/gliner_onnx.rs:extract`

**What happens**:
```rust
// First call
let e1 = gliner.extract(text, &["person", "org"], 0.5)?;

// Second call (same text, same types, different threshold)
let e2 = gliner.extract(text, &["person", "org"], 0.7)?;
```

**Opportunity**: The prompt encoding (`encode_prompt`) is identical - only the threshold changes. We could cache the prompt encoding and reuse it.

**Implementation**: Cache `encode_prompt` output keyed by `(text_hash, entity_types_hash)`.

### 5. `SemanticRegistry` - Already Caches Label Embeddings

**Location**: `src/backends/inference.rs:368-379`

**What it does**: Pre-computes label embeddings once, reuses forever. ✅

**Gap**: Text embeddings are NOT cached - only label embeddings are.

**Opportunity**: Mirror the `SemanticRegistry` pattern for text embeddings:
- `SemanticRegistry` = pre-computed label embeddings
- `TextEmbeddingCache` = cached text embeddings (same pattern)

## Recommended Implementation Order

### Phase 1: GLiNER Prompt Encoding Cache (Highest Value)

**Why**: 
- Clear benefit: `encode_prompt` is expensive
- Low memory cost: Input IDs are small
- High hit rate: Evaluation often queries same (text, entity_types)
- Easy to measure: Can benchmark cache hit vs miss

**Implementation**: Add LRU cache to `GLiNEROnnx` for `encode_prompt` output.

### Phase 2: Robustness Testing Reuse

**Why**:
- Robustness testing calls `extract_entities` multiple times on similar text
- Many perturbations don't change tokenization (case, whitespace)
- Could reuse embeddings when tokenization is identical

**Implementation**: 
- Detect when perturbation doesn't affect tokenization
- Reuse embeddings for those cases
- Fall back to recompute when tokenization changes

### Phase 3: Cross-Backend Text Embedding Cache (If Valuable)

**Why**:
- Multiple backends process same text
- If they share tokenizers, could share embeddings

**Implementation**:
- Add `shared_text_cache` to `TaskEvaluator`
- Key: `(text, tokenizer_id)`
- Value: Token embeddings
- Check tokenizer compatibility before sharing

### Phase 4: Store Embeddings in `per_example_scores` (If Memory Allows)

**Why**:
- Already iterating over `per_example_scores` multiple times
- Could reuse embeddings when re-computing metrics

**Implementation**:
- Extend `per_example_scores` to include optional embeddings
- Only store if memory budget allows
- Reuse when computing stratified metrics or confidence intervals

## Implementation Plan

1. ✅ **Profile first** - Created `benches/gliner_profiling.rs`
2. ⏳ **Run benchmarks** - Measure actual bottlenecks
3. ⏳ **Implement Phase 1** - GLiNER prompt encoding cache (if profiling shows it's worth it)
4. ⏳ **Measure impact** - Cache hit rate, performance improvement
5. ⏳ **Expand if valuable** - Add to other backends or implement Phase 2/3

## Questions to Answer

1. **What's the cache hit rate?** (How often is same (text, entity_types) queried?)
2. **What's the memory cost?** (Average prompt size * cache size)
3. **Do robustness perturbations change tokenization?** (If not, can reuse embeddings)
4. **Do backends share tokenizers?** (If yes, can share embeddings across backends)

