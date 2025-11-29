# NER/Coref Codebase Audit Report (2025 Edition)

> **Audit Date**: November 2025  
> **Codebase**: anno (Rust NER library)  
> **Auditor**: Technical Assessment per 2025 NER/Coref Guide

## Executive Summary

This codebase is **well-architected for modern NER/IE patterns**. Unlike many legacy systems, it:

1. **Already implements GLiNER bi-encoder** (span-based, zero-shot)
2. **Has proper offset handling** (`TextSpan`, `TokenSpan`, `OffsetMapping`)
3. **Supports entity-centric coref** via `canonical_id` field
4. **Includes relation extraction** (`RelationExtractor` trait, `RelationTriple`)
5. **Explicitly deprecates legacy patterns** (gazetteers in `rule.rs`)

**Risk Level**: Low - architecture is sound, minor refactoring opportunities exist.

---

## Phase 1: Code Smell Audit Results

### 1. Fixed Label Head (`id2label`, `num_labels`)

| File | Status | Notes |
|------|--------|-------|
| `backends/candle.rs` | Expected | Traditional BERT NER - fixed head is correct design |
| `backends/onnx.rs` | Expected | Same - legacy token classification |
| `backends/gliner_candle.rs` | Clean | Uses dynamic label embeddings |
| `backends/gliner_onnx.rs` | Clean | Uses dynamic label embeddings |
| `backends/inference.rs` | Clean | `SemanticRegistry` decouples labels from model |

**Assessment**: The `num_labels` usage is confined to traditional BERT backends (`CandleNER`, `BertNEROnnx`), which is correct. The primary GLiNER implementations use the modern bi-encoder pattern where labels are passed at runtime.

```rust
// Modern pattern (already implemented in gliner_candle.rs)
let entities = model.extract(
    "Steve Jobs founded Apple",
    &["person", "organization", "location"], // Runtime labels!
    0.5,
)?;
```

### 2. BIO/BILOU Tagging

| Component | Status | Notes |
|-----------|--------|-------|
| `eval/bio_adapter.rs` | Adapter | Explicitly for legacy format conversion |
| `eval/loader.rs` | Input | Parses CoNLL/BIO format datasets |
| `backends/nuner.rs` | Intentional | NuNER uses BIO for arbitrary-length entities |
| `backends/candle.rs` | Legacy | Traditional BERT - expected |
| `backends/gliner_*.rs` | Clean | Span-based extraction |

**Assessment**: BIO exists in two legitimate contexts:
1. **Dataset loading** (`loader.rs`) - necessary to load CoNLL-format datasets
2. **Bio adapter** (`bio_adapter.rs`) - explicitly converts to/from span format

The primary inference paths (GLiNER, W2NER) output `(start, end, label)` tuples directly.

```rust
// From bio_adapter.rs - this is an ADAPTER, not the primary path
/// Convert BIO-tagged tokens to entity spans.
pub fn bio_to_entities<S: AsRef<str>>(
    tokens: &[S],
    tags: &[S],
    scheme: BioScheme,
) -> Result<Vec<Entity>>
```

### 3. Heuristic Coreference

| Component | Status | Notes |
|-----------|--------|-------|
| `eval/coref_resolver.rs` | Documented | Explicitly "simple, not production" |
| `eval/coref_resolver.rs` | Good | Gender-debiased, neopronoun support |
| `discourse/` | Good | Event extraction for abstract anaphora |

**Assessment**: The codebase acknowledges this limitation explicitly:

```rust
// From coref_resolver.rs
//! For production coreference, use a dedicated system like:
//! - Stanford CoreNLP
//! - AllenNLP coref
//! - Hugging Face neuralcoref
```

The `DiscourseAwareResolver` (feature-gated) adds event extraction for better abstract anaphora handling, which is more modern than pure heuristics.

**Recommendation**: Add a `Seq2SeqCorefResolver` trait implementable by external T5-Coref models.

### 4. Token Classification vs Span-Based

| Backend | Architecture | Status |
|---------|--------------|--------|
| `GLiNEROnnx` | Span-based bi-encoder | Modern |
| `GLiNERCandle` | Span-based bi-encoder | Modern |
| `GLiNER2` | Multi-task span-based | Modern |
| `W2NER` | Word-word relation matrix | Modern |
| `CandleNER` | Token classification | Legacy (documented) |
| `BertNEROnnx` | Token classification | Legacy (documented) |
| `NuNER` | Token classification | Intentional design |

**Assessment**: The architecture provides both:
- **Span-based** (GLiNER, W2NER) for zero-shot and nested entities
- **Token classification** (BERT, NuNER) for fixed-type production

The architecture document (`backends/ARCHITECTURE.md`) correctly explains when to use each.

### 5. Manual Rules/Gazetteers

| Component | Status | Notes |
|-----------|--------|-------|
| `backends/rule.rs` | Deprecated | `#[deprecated]` attribute applied |
| `backends/pattern.rs` | Clean | Only regex for structured data |
| `backends/pattern_config.rs` | Clean | Configurable regex patterns |

**Assessment**: Correctly handled. The deprecated module includes clear documentation:

```rust
#[deprecated(
    since = "0.1.0",
    note = "Use PatternNER (no gazetteers) or ML backends (BERT ONNX). Will be removed in 1.0."
)]
pub struct RuleBasedNER { ... }
```

### 6. Tokenizer Alignment

| Component | Status | Notes |
|-----------|--------|-------|
| `offset.rs` | Excellent | Comprehensive byte/char/token mapping |
| `TextSpan` | Good | Stores both byte and char offsets |
| `TokenSpan` | Good | Links to `TextSpan` |
| `OffsetMapping` | Good | HuggingFace-compatible |
| `SpanConverter` | Good | Batch conversion with pre-computed tables |

**Assessment**: This is one of the strongest parts of the codebase. The `offset.rs` module explicitly addresses the three-coordinate problem:

```text
BYTE INDEX (what regex/file I/O returns)
CHAR INDEX (what humans count, what eval tools expect)
TOKEN INDEX (what BERT/transformers return)
```

The `SpanConverter` pre-computes mapping tables for O(1) conversion:

```rust
let conv = SpanConverter::new(text);
let span = conv.from_bytes(6, 11); // Returns TextSpan with both offsets
```

---

## Phase 2: Architecture Assessment

### Current State (Good)

```text
┌─────────────────────────────────────────────────────────────────┐
│                    anno Architecture                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ZERO-SHOT (GLiNER bi-encoder)                                 │
│  ─────────────────────────────                                 │
│  • Labels passed at runtime                                    │
│  • Span-based extraction (start, end, label)                   │
│  • No BIO decoding needed                                      │
│                                                                 │
│  NESTED/DISCONTINUOUS (W2NER)                                  │
│  ────────────────────────────                                  │
│  • Word-word relation matrix                                   │
│  • Handles overlapping spans                                   │
│                                                                 │
│  MULTI-TASK (GLiNER2)                                          │
│  ────────────────────                                          │
│  • NER + Classification + Questions                            │
│  • Unified schema                                              │
│                                                                 │
│  RELATION EXTRACTION                                           │
│  ────────────────────                                          │
│  • RelationExtractor trait                                     │
│  • RelationTriple output                                       │
│  • Integration with entity extraction                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Graph RAG Integration Readiness

| Requirement | Status | Component |
|-------------|--------|-----------|
| Entity spans with offsets | ✓ | `Entity`, `TextSpan` |
| Entity types as labels | ✓ | `EntityType::Custom` |
| Canonical ID for coref | ✓ | `Entity.canonical_id` |
| KB linking | ✓ | `Entity.kb_id` |
| Relation triples | ✓ | `RelationTriple`, `Relation` |
| Entity viewports | ✓ | `EntityViewport` |
| Graph export format | Partial | Needs explicit converter |

**Missing**: Explicit `to_graph_document()` function for Neo4j/NetworkX integration.

---

## Phase 3: Recommendations

### Priority 1: Graph RAG Export (Low Effort, High Value)

Add a graph document converter:

```rust
// Suggested addition to entity.rs or new graph.rs module
pub struct GraphDocument {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

pub struct GraphNode {
    pub id: String,           // canonical_id or kb_id
    pub node_type: String,    // EntityType label
    pub properties: HashMap<String, serde_json::Value>,
}

pub struct GraphEdge {
    pub source: String,
    pub target: String,
    pub relation: String,
    pub properties: HashMap<String, serde_json::Value>,
}

impl GraphDocument {
    pub fn from_extraction(
        entities: &[Entity],
        relations: &[Relation],
        coref_chains: Option<&[CorefChain]>,
    ) -> Self {
        // Use canonical_mention from coref chains as node IDs
        // ...
    }
    
    pub fn to_cypher(&self) -> String { ... }
    pub fn to_networkx_json(&self) -> String { ... }
}
```

### Priority 2: Seq2Seq Coref Trait (Medium Effort)

Define a trait for entity-centric coref models:

```rust
pub trait EntityCentricCoref: Send + Sync {
    /// Resolve coreference using entity-centric state.
    /// 
    /// Input: Document text + extracted entities
    /// Output: Entity clusters in compressed format
    fn resolve(&self, text: &str, entities: &[Entity]) -> Vec<CorefCluster>;
}

pub struct CorefCluster {
    pub cluster_id: u64,
    pub entity_type: EntityType,
    pub canonical_mention: String,
    pub mentions: Vec<MentionSpan>,
}
```

### Priority 3: Memory Safety for Long Documents (Low Priority)

The codebase handles long documents reasonably, but could add explicit windowing:

```rust
// Already somewhat handled by MAX_SPAN_WIDTH in gliner_candle.rs
const MAX_SPAN_WIDTH: usize = 12;

// Could add document windowing for coref
pub struct WindowedCoref {
    window_size: usize,      // tokens per window
    overlap: usize,          // tokens overlap between windows
}
```

---

## Gotchas Check

### 1. BIO Sentence Boundary Bug

**Status**: Not applicable - primary inference uses span-based extraction.

The BIO adapter (`bio_adapter.rs`) handles this for dataset loading but is not in the inference path.

### 2. Tokenizer Misalignment

**Status**: Addressed via `OffsetMapping`.

```rust
// From offset.rs - HuggingFace compatible
pub fn char_span_to_tokens(&self, char_start: usize, char_end: usize) -> Option<(usize, usize)>
```

### 3. Memory Leaks in Inference

**Status**: N/A (Rust) - no GC overhead, no Python runtime.

For ONNX models, the Rust `ort` crate handles memory management automatically.

---

## Summary

| Audit Category | Status | Action |
|----------------|--------|--------|
| Fixed Label Head | ✓ Modern | None needed |
| BIO Tagging | ✓ Adapter only | None needed |
| Heuristic Coref | Documented | Optional: Seq2Seq trait |
| Token Classification | ✓ Both available | None needed |
| Manual Rules | ✓ Deprecated | None needed |
| Offset Handling | ✓ Excellent | None needed |
| Graph RAG | Partial | Add export format |

**Overall Assessment**: This codebase is well-prepared for 2025 NER/IE patterns. The primary refactoring opportunity is adding explicit Graph RAG export functions to connect the existing relation extraction with downstream graph databases.

---

## References

1. GLiNER: arXiv:2311.08526 (NAACL 2024)
2. W2NER: arXiv:2112.10070 (AAAI 2022)
3. ModernBERT: Dec 2024 release
4. Entity-Centric Coref: ACL 2023 trends

