# Accidental Optimization Opportunities

## The User's Insight

The user is thinking about **accidental optimizations** - places where we're already computing or storing intermediate representations that could be reused, rather than adding new caching infrastructure.

## Existing Intermediate Representations

### 1. `per_example_scores_cache` in `TaskEvaluator`

**Location**: `src/eval/task_evaluator.rs:218-219`

```rust
per_example_scores_cache: std::sync::Mutex<Option<Vec<(Vec<Entity>, Vec<Entity>, String)>>>
```

**What it stores**:
- Gold entities
- Predicted entities  
- **Original text** (String)

**Opportunity**: When evaluating the same text with different backends or entity types, we're:
1. Re-tokenizing the same text
2. Re-encoding the same text
3. Only the entity types change

**Potential reuse**: If we store `(text, token_embeddings)` in the cache, we could reuse token embeddings when:
- Same text evaluated with different backends
- Same text evaluated with different entity types (for zero-shot backends)

### 2. `SemanticRegistry` - Pre-computed Label Embeddings

**Location**: `src/backends/inference.rs:368-379`

```rust
pub struct SemanticRegistry {
    pub embeddings: Vec<f32>,  // Pre-computed label embeddings
    pub hidden_dim: usize,
    pub labels: Vec<LabelDefinition>,
    pub label_index: HashMap<String, usize>,
}
```

**What it does**: Already caches label embeddings (entity type descriptions).

**Opportunity**: This is already optimized! ✅

**Gap**: Text embeddings are NOT cached - only label embeddings are.

### 3. `GLiNERPipeline` - Encoder is `Arc<E>`

**Location**: `src/backends/gliner_pipeline.rs:310-311`

```rust
pub struct GLiNERPipeline<E> {
    encoder: Arc<E>,  // Shared encoder
    registry: SemanticRegistry,  // Pre-computed labels
    // ...
}
```

**What it does**: Encoder is shared via `Arc`, but each call to `extract()` still:
1. Calls `encoder.encode(text)` - **recomputes token embeddings**
2. Generates span candidates
3. Computes span embeddings
4. Computes similarity

**Opportunity**: If the same text is processed multiple times (e.g., in evaluation with different entity types), we could cache `encoder.encode(text)` result.

### 4. Evaluation Loop - Same Text, Different Backends

**Location**: `src/eval/task_evaluator.rs:evaluate_combination`

**What happens**:
```rust
for backend in backends {
    for example in dataset {
        let entities = backend.extract_entities(&example.text, None)?;
        // ... evaluate ...
    }
}
```

**Opportunity**: If multiple backends process the same text:
- PatternNER, HeuristicNER, StackedNER all process "Apple Inc. was founded..."
- They all tokenize/parse the same text
- Could share tokenization results (though PatternNER doesn't use ML tokenization)

**Challenge**: Different backends use different tokenizers/encoders, so sharing is limited.

### 5. Zero-Shot Evaluation - Same Text, Different Entity Types

**Location**: `src/eval/task_evaluator.rs:evaluate_ner_task`

**What happens**:
```rust
// For zero-shot backends
let entity_types = dataset.entity_types;  // e.g., ["person", "organization"]
let entities = backend.extract_with_types(&text, &entity_types, 0.5)?;
```

**Opportunity**: If we evaluate the same text with different entity type sets:
- First: `extract_with_types(text, ["person", "org"], 0.5)`
- Second: `extract_with_types(text, ["person", "org", "location"], 0.5)`

For GLiNER, the prompt encoding includes entity types:
- `[START] <<ENT>> person <<ENT>> org <<SEP>> text...`
- `[START] <<ENT>> person <<ENT>> org <<ENT>> location <<SEP>> text...`

**Can't reuse**: Different entity types = different prompts = different encodings.

**BUT**: If we evaluate with the same entity types but different thresholds, we could reuse the prompt encoding!

### 6. Batch Processing - Already Aggregated

**Location**: `src/eval/task_evaluator.rs:per_example_scores`

**What it stores**: `Vec<(Vec<Entity>, Vec<Entity>, String)>` - per-example results.

**Opportunity**: When computing stratified metrics or confidence intervals, we iterate over `per_example_scores` multiple times. If we stored intermediate representations (token embeddings, span embeddings) alongside the results, we could reuse them.

**Challenge**: Memory cost - storing embeddings for all examples could be large.

## Concrete Opportunities

### Opportunity 1: Text Embedding Cache in Evaluation

**Scenario**: Evaluating same dataset with multiple backends.

**Current**: Each backend re-tokenizes and re-encodes the same text.

**Optimization**: Store `(text_hash, token_embeddings)` in `TaskEvaluator`:
```rust
pub struct TaskEvaluator {
    // ... existing fields ...
    text_embedding_cache: Arc<Mutex<LruCache<u64, Vec<f32>>>>,  // text_hash -> embeddings
}
```

**When to use**:
- Same text processed by multiple backends
- Same text processed with different entity types (if entity types don't affect text encoding)

**When NOT to use**:
- GLiNER: Entity types are in the prompt, so different types = different encoding
- Different tokenizers: Can't share embeddings across tokenizers

### Opportunity 2: Prompt Encoding Cache (GLiNER-specific)

**Scenario**: Evaluating same (text, entity_types) combination multiple times.

**Current**: `encode_prompt` is called every time, even if (text, entity_types) is the same.

**Optimization**: Cache `encode_prompt` output:
```rust
struct PromptCacheKey {
    text_hash: u64,
    entity_types_hash: u64,  // Hash of sorted entity types
}

struct PromptCacheValue {
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    words_mask: Vec<i64>,
    // ...
}
```

**When to use**:
- Same text + same entity types queried multiple times
- Evaluation with multiple thresholds (same prompt, different threshold)

**When NOT to use**:
- Different entity types = different prompt = can't reuse

### Opportunity 3: Reuse `per_example_scores` Text Embeddings

**Scenario**: Computing stratified metrics or confidence intervals from `per_example_scores`.

**Current**: We store `(gold, predicted, text)` but not embeddings.

**Optimization**: Store embeddings alongside:
```rust
struct PerExampleResult {
    gold: Vec<Entity>,
    predicted: Vec<Entity>,
    text: String,
    text_embeddings: Option<Vec<f32>>,  // Optional, cached if available
}
```

**When to use**:
- Re-computing metrics from cached results
- Re-running evaluation with different thresholds

**Challenge**: Memory cost - embeddings are large.

## Recommendation

**Start with Opportunity 2** (Prompt Encoding Cache) because:
1. **Clear benefit**: `encode_prompt` is expensive (tokenization + encoding)
2. **Low memory cost**: Input IDs are small (Vec<i64>, ~100-500 elements)
3. **High hit rate**: Evaluation often queries same (text, entity_types) combinations
4. **Easy to measure**: Can benchmark cache hit vs miss

**Then consider Opportunity 1** (Text Embedding Cache) if:
- Profiling shows text encoding is a bottleneck
- Multiple backends process the same text
- Memory cost is acceptable

**Skip Opportunity 3** (Storing embeddings in per_example_scores) unless:
- Memory is not a concern
- We frequently re-compute metrics from cached results

## Implementation Plan

1. **Profile first** (already done ✅)
2. **Implement prompt encoding cache** in `GLiNEROnnx` (if profiling shows it's worth it)
3. **Measure cache hit rate** in evaluation scenarios
4. **Expand to other backends** if valuable

