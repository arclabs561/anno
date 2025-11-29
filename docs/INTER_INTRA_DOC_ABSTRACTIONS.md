# Inter/Intra-Doc NER, Coref, NED: Abstraction Mapping

## Executive Summary

Your existing `Signal → Track → Identity` hierarchy is **architecturally sound**, but there's a **semantic gap** between the evaluation layer (`CrossDocCluster` in `cdcr.rs`) and the core representation (`Identity` in `grounded.rs`). The abstraction hierarchy correctly separates concerns, but the **operations** that bridge these levels need clearer type-level distinctions.

## The Operation Taxonomy

### Level 1: Signal Detection (NER)

**Intra-doc NER** and **Inter-doc NER** are operationally identical—both produce `Signal<Location>`. The distinction is only in **scope** (one document vs. multiple documents), not in the abstraction.

```rust
// Intra-doc: Single document
fn extract_entities(doc: &str) -> Vec<Signal<Location>>

// Inter-doc: Multiple documents (same operation, different scope)
fn extract_entities_batch(docs: &[&str]) -> Vec<(DocId, Vec<Signal<Location>>)>
```

**Key Insight**: NER is **stateless** at the signal level. Each document's signals are independent until you form tracks.

### Level 2: Track Formation (Intra-Doc Coreference)

**Intra-doc coreference** is the **only** operation that creates `Track` instances. Tracks are inherently document-scoped.

```rust
// This is the ONLY way tracks are created
fn resolve_coreference(signals: &[Signal<Location>]) -> Vec<Track>
```

**Design Note**: Your `Track` struct correctly captures this:
- `signals: Vec<SignalRef>` - all signals are from the same document
- `identity_id: Option<IdentityId>` - optional link to global identity
- `canonical_surface: String` - best name from this document's context

**The Gap**: There's no explicit operation that says "form tracks across documents." This is intentional—tracks are document-local. Cross-document linking happens at Level 3.

### Level 3: Identity Resolution (Inter-Doc Coref + NED/Entity Linking)

This is where the nuance matters. There are **two distinct operations** that both produce `Identity`:

#### 3a. Inter-Document Coreference (CDCR)

**Operation**: Cluster tracks from multiple documents without KB linking.

```rust
// Input: Multiple documents, each with tracks
// Output: Identity clusters (no KB link yet)
fn resolve_cross_doc_coref(
    documents: &[GroundedDocument]
) -> Vec<Identity>
```

**Characteristics**:
- **Input**: `Vec<(DocId, TrackId)>` - tracks from different documents
- **Output**: `Identity` with `kb_id: None`
- **Method**: Embedding similarity, string matching, LSH blocking
- **Result**: "These tracks refer to the same real-world entity"

**Current State**: Your `cdcr.rs` module does this, but it uses `CrossDocCluster` (evaluation format) rather than `Identity` directly. The conversion exists (`Identity::from_cross_doc_cluster`), but the operation isn't first-class in the `GroundedDocument` API.

#### 3b. Named Entity Disambiguation / Entity Linking (NED)

**Operation**: Link a track (or identity) to a knowledge base entry.

```rust
// Input: Track or Identity
// Output: Identity with KB link
fn link_to_kb(
    track: &Track,
    kb: &KnowledgeBase
) -> Option<Identity>
```

**Characteristics**:
- **Input**: Single `Track` or `Identity`
- **Output**: `Identity` with `kb_id: Some(...)`
- **Method**: Candidate generation → re-ranking → validation
- **Result**: "This entity is Q7186 in Wikidata"

**Current State**: Your `Identity` struct supports this (`kb_id`, `kb_name`), but there's no explicit `link_to_kb` operation in the `GroundedDocument` API.

## The Abstraction Correctness

### What's Right

1. **Signal is stateless**: Correctly represents raw detection without identity assumptions.
2. **Track is document-scoped**: Correctly captures intra-doc coreference.
3. **Identity is global**: Correctly represents cross-doc and KB-linked entities.
4. **Separation of concerns**: Each level has distinct responsibilities.

### What Needs Clarification

1. **Inter-doc coref without KB**: Currently, `Identity` can have `kb_id: None`, which is correct. But the operation to create such identities from multiple tracks isn't first-class.

2. **KB linking without inter-doc coref**: A single track can link to KB without needing other documents. This should be a distinct operation.

3. **The CDCR ↔ Identity gap**: `CrossDocCluster` is an evaluation format. The core representation should use `Identity` directly, with conversion only at the evaluation boundary.

## Recommended Abstraction Refinements

### 1. Explicit Operation Types

Add operation types that make the distinctions clear:

```rust
// In grounded.rs

/// Operation that creates identities from tracks across documents
pub fn resolve_inter_doc_coref(
    documents: &[GroundedDocument],
    config: &CDCRConfig,
) -> Vec<Identity> {
    // 1. Extract all tracks from all documents
    // 2. Compute track embeddings
    // 3. Cluster tracks (LSH blocking + similarity)
    // 4. Create Identity for each cluster (kb_id: None)
    // 5. Return identities
}

/// Operation that links a track/identity to a knowledge base
pub fn link_entity_to_kb(
    track: &Track,
    kb: &KnowledgeBase,
) -> Option<Identity> {
    // 1. Generate candidates from KB
    // 2. Re-rank candidates
    // 3. Validate top candidate
    // 4. Return Identity with kb_id
}
```

### 2. Track Provenance

Add document provenance to `Track` so inter-doc operations know which document each track came from:

```rust
pub struct Track {
    // ... existing fields ...
    /// Document ID this track belongs to
    pub doc_id: String,  // Add this
}
```

**Alternative**: Keep `Track` document-scoped, but add a `TrackRef` type for cross-document operations:

```rust
/// Reference to a track in a specific document
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TrackRef {
    pub doc_id: String,
    pub track_id: TrackId,
}
```

### 3. Identity Formation Modes

Make the distinction between "clustered from tracks" vs "linked from KB" explicit:

```rust
pub enum IdentitySource {
    /// Created from cross-document track clustering
    CrossDocCoref { track_refs: Vec<TrackRef> },
    /// Linked from knowledge base
    KnowledgeBase { kb_name: String, kb_id: String },
    /// Both: clustered AND linked
    Hybrid { track_refs: Vec<TrackRef>, kb_name: String, kb_id: String },
}
```

## The Complete Operation Map

| Operation | Input | Output | Level | Current State |
|-----------|-------|--------|-------|---------------|
| **NER (intra-doc)** | `&str` | `Vec<Signal>` | 1 | ✅ Implemented |
| **NER (inter-doc)** | `&[&str]` | `Vec<(DocId, Vec<Signal>)>` | 1 | ✅ Just batch NER |
| **Intra-doc coref** | `Vec<Signal>` | `Vec<Track>` | 2 | ✅ Implemented |
| **Inter-doc coref** | `Vec<GroundedDocument>` | `Vec<Identity>` | 3 | ⚠️ Uses `CrossDocCluster` |
| **Entity Linking (NED)** | `Track` or `Identity` | `Identity` | 3 | ⚠️ Not first-class |
| **Stream coref** | `Stream<Signal>` | `Stream<Track>` | 2 | ⚠️ Not implemented |

## The Vision Analogy (Revisited)

| Vision Operation | NLP Operation | Your Abstraction |
|------------------|---------------|------------------|
| **Object Detection** | NER | `Signal` |
| **Single-Camera Tracking** | Intra-doc coref | `Track` |
| **Multi-Camera Tracking** | Inter-doc coref | `Identity` (from tracks) |
| **Face Recognition** | Entity Linking | `Identity` (from KB) |
| **Re-identification** | Cross-doc coref | `Identity` (clustered) |

## Recommendations

1. **Keep the hierarchy**: Signal → Track → Identity is correct.

2. **Add explicit operations**: Make inter-doc coref and KB linking first-class operations in `GroundedDocument` or a new `Corpus` type.

3. **Unify CDCR and Identity**: Use `Identity` as the core representation, convert to `CrossDocCluster` only for evaluation.

4. **Add TrackRef**: For cross-document operations, use `TrackRef` to reference tracks without copying.

5. **Document the distinction**: Add clear docs explaining:
   - Inter-doc coref = clustering tracks → Identity (no KB)
   - Entity linking = linking track → Identity (with KB)
   - They can be combined (cluster tracks, then link cluster to KB)

## Implementation Path

1. **Phase 1**: Add `TrackRef` and `resolve_inter_doc_coref` to `grounded.rs`
2. **Phase 2**: Add `link_entity_to_kb` operation
3. **Phase 3**: Refactor `cdcr.rs` to use `Identity` internally, convert to `CrossDocCluster` only for evaluation
4. **Phase 4**: Add streaming support for incremental track formation

The abstractions are sound. The gap is in making the **operations** that bridge levels explicit and type-safe.

