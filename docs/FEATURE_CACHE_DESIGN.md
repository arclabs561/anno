# Shared Feature Cache Design

> **Note**: This document has been critiqued. See [FEATURE_CACHE_CRITIQUE.md](FEATURE_CACHE_CRITIQUE.md) for issues identified, and [FEATURE_CACHE_IMPLEMENTATION_PLAN.md](FEATURE_CACHE_IMPLEMENTATION_PLAN.md) for the revised implementation plan.

## Problem

When `extract_entities` is called multiple times (or by different projects), text embeddings are recomputed each time. For backends like GLiNER that use bi-encoder architectures, we can cache:

1. **Text embeddings**: Token-level embeddings for input text
2. **Span embeddings**: Pre-computed span representations
3. **Label embeddings**: Already cached in `SemanticRegistry` ✅

## Proposed Solution

### Option 1: Shared Context (Recommended)

Add an optional `SharedContext` parameter to `extract_entities`:

```rust
pub trait Model {
    fn extract_entities(
        &self,
        text: &str,
        language: Option<&str>,
        context: Option<&SharedContext>,  // NEW
    ) -> Result<Vec<Entity>>;
}

pub struct SharedContext {
    // Text embeddings cache: text_hash -> token_embeddings
    text_cache: Arc<Mutex<HashMap<u64, Vec<f32>>>>,
    // Span embeddings cache: (text_hash, span_start, span_end) -> span_embedding
    span_cache: Arc<Mutex<HashMap<(u64, usize, usize), Vec<f32>>>>,
    // Encoder shared across backends
    encoder: Option<Arc<dyn TextEncoder>>,
}
```

**Pros:**
- Explicit control over caching
- Can be shared across multiple backend instances
- Backward compatible (context is optional)

**Cons:**
- Requires updating all backend implementations
- Adds complexity to the trait

### Option 2: Backend-Level Cache

Add caching directly to backends that support it:

```rust
pub struct GLiNEROnnx {
    // ... existing fields ...
    text_cache: Arc<Mutex<LruCache<u64, Vec<f32>>>>,  // LRU cache for text embeddings
}
```

**Pros:**
- Simpler API (no context parameter)
- Backend-specific optimization

**Cons:**
- Cache not shared between backend instances
- Each backend needs its own cache implementation

### Option 3: Global Cache Manager (Advanced)

A singleton cache manager that can be shared across the entire process:

```rust
pub struct GlobalFeatureCache {
    text_embeddings: Arc<Mutex<LruCache<u64, Vec<f32>>>>,
    span_embeddings: Arc<Mutex<LruCache<(u64, usize, usize), Vec<f32>>>>,
}

impl GlobalFeatureCache {
    pub fn global() -> &'static GlobalFeatureCache { /* ... */ }
    
    pub fn get_text_embedding(&self, text: &str) -> Option<Vec<f32>> {
        let hash = self.hash_text(text);
        self.text_embeddings.lock().unwrap().get(&hash).cloned()
    }
    
    pub fn cache_text_embedding(&self, text: &str, embedding: Vec<f32>) {
        let hash = self.hash_text(text);
        self.text_embeddings.lock().unwrap().put(hash, embedding);
    }
}
```

**Pros:**
- Shared across all backends and projects
- Maximum efficiency

**Cons:**
- Global state (harder to test, potential memory issues)
- Requires careful cache invalidation strategy

## Recommendation

Start with **Option 2** (backend-level cache) for immediate benefit, then consider **Option 1** (shared context) if cross-backend sharing is needed.

### Implementation Plan

1. Add `LruCache` to GLiNER backends (ONNX and Candle)
2. Cache text embeddings keyed by text hash
3. Add cache size configuration (default: 100 entries)
4. Document cache behavior in backend docs

### Example Usage

```rust
// Backend automatically caches embeddings
let ner = GLiNEROnnx::new("model")?;

// First call: computes embeddings
let e1 = ner.extract_entities("Apple Inc. was founded by Steve Jobs", None)?;

// Second call with same text: uses cached embeddings
let e2 = ner.extract_entities("Apple Inc. was founded by Steve Jobs", None)?;

// Different text: computes new embeddings
let e3 = ner.extract_entities("Microsoft was founded by Bill Gates", None)?;
```

## Future Enhancements

- Cross-backend sharing via `SharedContext`
- Cache warming for known texts
- Cache statistics/metrics
- Configurable cache eviction policies

