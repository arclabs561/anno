# Feature Cache Implementation Plan

## Critique Summary

The original design had several issues:
1. **No profiling data** - Assumed caching would help without measuring
2. **Flawed cache key** - Didn't account for entity types in GLiNER prompts
3. **API breaking** - Option 1 would require changing all backends
4. **Premature optimization** - Span caching is cheap, not worth it
5. **Missing insight** - GLiNER prompt includes entity types, so `(text, entity_types)` is the real cache key

## Key Insight from Code Analysis

Looking at `GLiNEROnnx::extract()`:
- Line 247: `self.encode_prompt(&text_words, entity_types)?` - **Entity types are part of the prompt**
- The prompt format: `[START] <<ENT>> type1 <<ENT>> type2 <<SEP>> word1 word2 ... [END]`
- **Same text + different entity types = different embeddings**

**Conclusion**: Cache key must be `(text_hash, entity_types_hash, model_id)`.

## Implementation Plan

### Phase 1: Simple Token Embedding Cache (Recommended Start)

**Goal**: Add LRU cache for `encode_prompt` output (input_ids) in `GLiNEROnnx` only.

**Why this works**:
- `encode_prompt` is expensive (tokenization + encoding)
- Input IDs are small (Vec<i64>, ~100-500 elements)
- LRU prevents memory bloat
- No API changes needed
- Easy to measure impact

**Implementation**:

1. Add `lru` dependency to `Cargo.toml`:
```toml
lru = { version = "0.12", optional = true }
```

2. Add cache to `GLiNEROnnx`:
```rust
use lru::LruCache;
use std::num::NonZeroUsize;

pub struct GLiNEROnnx {
    // ... existing fields ...
    prompt_cache: Arc<Mutex<LruCache<PromptCacheKey, PromptCacheValue>>>,
}

#[derive(Hash, PartialEq, Eq, Clone)]
struct PromptCacheKey {
    text_hash: u64,           // Hash of text
    entity_types_hash: u64,   // Hash of sorted entity types
    model_id: String,         // Model identifier
}

struct PromptCacheValue {
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    words_mask: Vec<i64>,
    text_lengths: i64,
    entity_count: usize,
}
```

3. Modify `encode_prompt` to check cache first:
```rust
fn encode_prompt_cached(
    &self,
    text_words: &[&str],
    entity_types: &[&str],
) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>, i64, usize)> {
    // Build cache key
    let text = text_words.join(" ");
    let text_hash = hash_text(&text);
    let entity_types_hash = hash_entity_types(entity_types);
    let key = PromptCacheKey {
        text_hash,
        entity_types_hash,
        model_id: self.model_name.clone(),
    };
    
    // Check cache
    let mut cache = self.prompt_cache.lock().unwrap();
    if let Some(cached) = cache.get(&key) {
        return Ok((
            cached.input_ids.clone(),
            cached.attention_mask.clone(),
            cached.words_mask.clone(),
            cached.text_lengths,
            cached.entity_count,
        ));
    }
    
    // Compute (existing encode_prompt logic)
    let result = self.encode_prompt(text_words, entity_types)?;
    
    // Cache result
    cache.put(key, PromptCacheValue {
        input_ids: result.0.clone(),
        attention_mask: result.1.clone(),
        words_mask: result.2.clone(),
        text_lengths: result.3,
        entity_count: result.4,
    });
    
    Ok(result)
}
```

4. Update `extract` to use cached version:
```rust
let (input_ids, attention_mask, words_mask, text_lengths, entity_count) =
    self.encode_prompt_cached(&text_words, entity_types)?;
```

**Cache size**: Default 100 entries (configurable via `GLiNERConfig`).

### Phase 2: Measure Impact

Add benchmarking to measure cache hit rate and performance:

```rust
#[cfg(test)]
mod benches {
    use criterion::{black_box, criterion_group, criterion_main, Criterion};
    
    fn bench_gliner_with_cache(c: &mut Criterion) {
        let model = GLiNEROnnx::new("onnx-community/gliner_small-v2.1")?;
        let text = "Apple Inc. was founded by Steve Jobs in California.";
        let types = &["person", "organization", "location"];
        
        // First call (cache miss)
        c.bench_function("gliner_extract_cache_miss", |b| {
            b.iter(|| model.extract(black_box(text), types, 0.5))
        });
        
        // Second call (cache hit)
        c.bench_function("gliner_extract_cache_hit", |b| {
            b.iter(|| model.extract(black_box(text), types, 0.5))
        });
    }
}
```

**Success criteria**:
- Cache hit should be 2-5x faster than cache miss
- Cache hit rate > 30% in realistic workloads
- Memory usage < 10MB for 100 entries

### Phase 3: Expand if Valuable

If Phase 1 shows benefit:
1. Add to `GLiNERCandle` (same pattern)
2. Add to `GLiNER2` (if it benefits)
3. Add cache metrics (hit rate, memory usage)
4. Make cache size configurable

### Phase 4: Cross-Backend Sharing (If Needed)

Only if profiling shows cross-backend sharing is valuable:

```rust
pub struct SharedEmbeddingCache {
    prompt_cache: Arc<Mutex<LruCache<PromptCacheKey, PromptCacheValue>>>,
}

impl SharedEmbeddingCache {
    pub fn new(size: usize) -> Self {
        Self {
            prompt_cache: Arc::new(Mutex::new(LruCache::new(
                NonZeroUsize::new(size).unwrap()
            ))),
        }
    }
    
    pub fn get_or_compute<F>(
        &self,
        key: PromptCacheKey,
        compute: F,
    ) -> Result<PromptCacheValue>
    where
        F: FnOnce() -> Result<PromptCacheValue>,
    {
        let mut cache = self.prompt_cache.lock().unwrap();
        if let Some(cached) = cache.get(&key) {
            return Ok(cached.clone());
        }
        
        let value = compute()?;
        cache.put(key, value.clone());
        Ok(value)
    }
}
```

**Usage**:
```rust
let shared_cache = SharedEmbeddingCache::new(200);
let gliner1 = GLiNEROnnx::with_cache("model1", shared_cache.clone())?;
let gliner2 = GLiNEROnnx::with_cache("model2", shared_cache.clone())?;
// Both share the same cache
```

## Implementation Steps

1. ✅ **Critique complete** - Identified 7 issues with original design
2. ✅ **Add `lru` dependency** - Added to `Cargo.toml` with `onnx` feature
3. ✅ **Create profiling benchmark** - `benches/gliner_profiling.rs` to measure actual bottlenecks
4. ⏳ **Run benchmarks** - `cargo bench --bench gliner_profiling --features onnx`
5. ⏳ **Analyze results** - Determine if `encode_prompt` is the bottleneck
6. ⏳ **Implement cache if valuable** - Only if profiling shows it's worth it
7. ⏳ **Measure cache impact** - Compare cached vs uncached performance
8. ⏳ **Expand** - Add to other backends if valuable

## Questions to Answer

1. **What's the actual cost of `encode_prompt`?** (Profile with `perf` or `criterion`)
2. **What's the cache hit rate?** (Log cache hits/misses in dev mode)
3. **What's the memory cost?** (Average prompt size * cache size)
4. **Is cross-backend sharing needed?** (Measure if same text queried across backends)

## Next Steps

Start with **Phase 1** (simple cache in `GLiNEROnnx`). This is:
- Low risk (no API changes)
- Easy to measure (clear before/after)
- Easy to remove (if not valuable)
- Addresses the user's question (caching between calls)

