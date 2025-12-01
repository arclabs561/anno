# Feature Cache Design Critique

## Current Design Issues

### 1. **Problem Statement is Incomplete**

The design document assumes text embeddings are the bottleneck, but doesn't analyze:
- **Where is time actually spent?** (tokenization, encoding, span generation, similarity computation)
- **What's the cost breakdown?** (encoding might be 30ms, span computation 20ms, similarity 50ms)
- **Is caching even worth it?** (if encoding is 10% of total time, caching saves little)

**Missing**: Performance profiling data showing actual bottlenecks.

### 2. **Cache Key Design is Flawed**

Current proposal uses `text_hash` (u64) as cache key:
```rust
text_cache: Arc<Mutex<HashMap<u64, Vec<f32>>>>
```

**Problems**:
- **Hash collisions**: Two different texts could hash to same value
- **No model versioning**: Same text with different models should have different embeddings
- **No label dependency**: GLiNER embeddings depend on entity types being queried (prompt includes labels)
- **Memory bloat**: Storing full `Vec<f32>` for every text (could be 768 * seq_len * 4 bytes)

**Better approach**: Cache key should include:
- Text content (or hash with collision handling)
- Model ID/version
- Entity types (for GLiNER-style backends)
- Sequence length (for variable-length encoders)

### 3. **Option 1 (Shared Context) Has API Problems**

Adding `context: Option<&SharedContext>` to `extract_entities`:
- **Breaking change**: Would require updating all backends
- **Trait pollution**: Not all backends benefit from caching (PatternNER doesn't need it)
- **Complexity**: Context must be passed through call chains

**Better**: Make caching opt-in via a separate trait or builder pattern.

### 4. **Option 2 (Backend-Level) Misses Cross-Backend Sharing**

The user's question: "what if extract_entities is called then another related project"

This suggests they want **cross-backend sharing**, which Option 2 doesn't provide.

**Missing**: A way to share embeddings between:
- Different backend instances (GLiNEROnnx vs GLiNERCandle)
- Different projects/processes (if possible)
- Different entity type queries on same text

### 5. **Span Embedding Caching is Premature**

Caching span embeddings `(text_hash, span_start, span_end)` assumes:
- Same text will query same spans (not true - entity types change span candidates)
- Span computation is expensive (might be cheap relative to encoding)
- Memory cost is acceptable (could be huge for long texts)

**Reality**: Span embeddings are cheap to compute from token embeddings. Cache token embeddings, not spans.

### 6. **No Cache Invalidation Strategy**

The design doesn't address:
- **Memory limits**: What if cache grows to 10GB?
- **Staleness**: What if model weights change?
- **Thread safety**: Multiple threads accessing cache simultaneously
- **Cache metrics**: Hit rate, memory usage, eviction stats

### 7. **GLiNER-Specific: Prompt Encoding is the Real Bottleneck**

Looking at the code, GLiNER does:
1. **Prompt encoding**: `[START] <<ENT>> type1 <<ENT>> type2 <<SEP>> word1 word2 ... [END]`
2. **Tokenization**: Split into tokens
3. **ONNX inference**: Run model
4. **Span generation**: Create span candidates
5. **Similarity computation**: Match spans to labels

**Key insight**: The prompt includes entity types, so:
- Same text + different entity types = different embeddings
- Caching by text alone won't help for zero-shot queries
- Need to cache `(text, entity_types)` → embeddings

## What Actually Needs Caching

### High Value (Worth Implementing)
1. **Token embeddings** (after encoding, before span computation)
   - Key: `(text_hash, model_id, entity_types_hash)`
   - Value: `Vec<f32>` (token embeddings)
   - Benefit: Saves 30-50ms per call if text+types match

2. **Label embeddings** (already done via `SemanticRegistry` ✅)
   - Pre-computed once, reused forever
   - Already implemented correctly

### Low Value (Skip for Now)
1. **Span embeddings**: Cheap to compute, depends on entity types
2. **Similarity scores**: Very cheap, depends on both spans and labels
3. **Final entities**: No point caching (output is small)

## Revised Recommendation

### Phase 1: Backend-Level Token Embedding Cache (Simple)

Add LRU cache to GLiNER backends for token embeddings:

```rust
pub struct GLiNEROnnx {
    // ... existing fields ...
    token_cache: Arc<Mutex<LruCache<CacheKey, Vec<f32>>>>,
}

struct CacheKey {
    text_hash: u64,
    entity_types_hash: u64,  // Hash of sorted entity types
    model_id: String,
}
```

**Why this works**:
- Simple to implement (one backend at a time)
- No API changes needed
- Handles the common case (same text + same types)
- LRU prevents memory bloat

### Phase 2: Shared Context (If Needed)

Only if profiling shows cross-backend sharing is valuable:

```rust
pub trait CachedModel: Model {
    fn extract_with_cache(
        &self,
        text: &str,
        language: Option<&str>,
        cache: Option<&EmbeddingCache>,
    ) -> Result<Vec<Entity>>;
}
```

**Why separate trait**:
- Doesn't pollute base `Model` trait
- Opt-in for backends that benefit
- Backward compatible

## Implementation Plan

1. **Profile first**: Measure where time is spent in `extract_entities`
2. **Start small**: Add token embedding cache to `GLiNEROnnx` only
3. **Measure impact**: Compare cached vs uncached performance
4. **Expand if valuable**: Add to other backends if cache hit rate is high
5. **Add metrics**: Track cache hit rate, memory usage, eviction rate

## Questions to Answer Before Implementing

1. **What's the actual bottleneck?** (Profile `extract_entities` with `criterion` or `perf`)
2. **What's the cache hit rate?** (How often is same text+types queried?)
3. **What's the memory cost?** (Average text length * hidden_dim * 4 bytes)
4. **Is cross-backend sharing needed?** (Or is single-backend cache enough?)

## Conclusion

The current design is **premature optimization** without profiling data. Start with:
1. Performance profiling to identify bottlenecks
2. Simple backend-level cache for token embeddings
3. Measure impact before adding complexity

