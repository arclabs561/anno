# Missed Opportunities in CLI Implementation

After comprehensive review of the repository, here are significant capabilities we're not leveraging:

## 1. Graph Export Functionality ⚠️ **HIGH PRIORITY**

**What exists**: `GraphDocument` with export to Neo4j Cypher, NetworkX JSON, and other formats

**Location**: `src/graph.rs`

**What we're missing**:
- No CLI command to export GroundedDocuments to graph formats
- Graph export is a key feature for RAG applications
- Could be added to `extract`, `debug`, or `pipeline` commands

**Example API**:
```rust
use anno::graph::{GraphDocument, GraphExportFormat};

let graph = GraphDocument::from_grounded_document(&doc);
println!("{}", graph.to_cypher());  // Neo4j
println!("{}", graph.to_networkx_json());  // NetworkX
```

**Proposed CLI**:
```bash
anno extract "text" --export-graph neo4j
anno debug "text" --coref --export-graph networkx
anno pipeline --dir ./docs --export-graph cypher
```

## 2. Corpus Management ⚠️ **HIGH PRIORITY**

**What exists**: `Corpus` type with `resolve_inter_doc_coref()` method

**Location**: `src/grounded.rs:3159-3214`

**What we're missing**:
- `Corpus` provides better inter-document coreference than our manual implementation
- Has built-in identity management across documents
- Could replace/improve `cross-doc` command

**Example API**:
```rust
use anno::grounded::Corpus;

let mut corpus = Corpus::new();
corpus.add_document(doc1);
corpus.add_document(doc2);
corpus.resolve_inter_doc_coref(0.7);  // Automatic!
```

**Proposed CLI**:
```bash
anno corpus --dir ./docs --resolve-coref --threshold 0.7
anno corpus --import doc1.json doc2.json --resolve-coref
```

## 3. Validation Utilities ⚠️ **MEDIUM PRIORITY**

**What exists**: 
- `GroundedDocument::validate()` - Returns validation errors
- `GroundedDocument::add_signal_validated()` - Validates before adding
- `Signal::validate_against()` - Per-signal validation

**Location**: `src/grounded.rs:1619-1679`

**What we're missing**:
- Not using validation in `extract`/`enhance` commands
- Could add `--validate` flag to check signal offsets
- Could use `add_signal_validated()` instead of `add_signal()`

**Proposed CLI**:
```bash
anno extract "text" --validate  # Check all signals are valid
anno enhance doc.json --validate --coref  # Validate before/after
```

## 4. Process Text Utilities ⚠️ **MEDIUM PRIORITY**

**What exists**: 
- `process_text()` - Convenience function for extraction + validation
- `process_with_gold()` - Extraction with gold comparison

**Location**: `src/grounded.rs:2951-3038`

**What we're missing**:
- Could simplify `extract` command implementation
- `process_with_gold()` could enhance `eval` command

**Current**: Manual entity extraction + GroundedDocument building
**Could use**: `process_text()` helper

## 5. Spatial Index for Range Queries ⚠️ **LOW PRIORITY**

**What exists**:
- `build_text_index()` - Spatial index for efficient range queries
- `query_signals_in_range_indexed()` - Fast range queries
- `query_overlapping_signals_indexed()` - Fast overlap queries

**Location**: `src/grounded.rs:2118-2167`

**What we're missing**:
- Query command could use spatial index for range queries
- Could add `--range START:END` filter to query command

**Proposed CLI**:
```bash
anno query doc.json --range 0:100  # Signals in first 100 chars
anno query doc.json --overlap 50:150  # Signals overlapping range
```

## 6. Statistics Helper ⚠️ **LOW PRIORITY**

**What exists**: `GroundedDocument::stats()` - Comprehensive document statistics

**Location**: `src/grounded.rs:1687-1750`

**What we're missing**:
- Manually computing stats in some commands
- Could use `doc.stats()` instead

**Current**: Manual stat computation
**Could use**: `doc.stats()` which includes:
- Signal/track/identity counts
- Average track size
- Singleton count
- Average confidence
- Negated/quantified counts
- Modality breakdown

## 7. Additional Query Helpers ⚠️ **LOW PRIORITY**

**What exists**:
- `negated_signals()` - Get negated entities
- `quantified_signals()` - Get quantified entities
- `overlapping_signals()` - Get overlapping signals
- `signals_in_range()` - Get signals in text range
- `untracked_signals()` - Get signals not in tracks

**Location**: `src/grounded.rs:1559-1597`

**What we're missing**:
- Query command could support these filters
- Could add `--negated`, `--quantified`, `--untracked` flags

**Proposed CLI**:
```bash
anno query doc.json --negated  # Only negated entities
anno query doc.json --untracked  # Signals not in tracks
anno query doc.json --quantified universal  # Universally quantified
```

## 8. HTML Rendering Consistency ⚠️ **LOW PRIORITY**

**What exists**: `render_document_html()` - Brutalist HTML visualization

**Location**: `src/grounded.rs:2174-2915`

**What we're missing**:
- `debug` command uses custom HTML rendering
- Could standardize on `render_document_html()` for consistency

## 9. Corpus Inter-Doc Coref vs. Cross-Doc Command

**What exists**: `Corpus::resolve_inter_doc_coref()` - Built-in inter-doc coreference

**Location**: `src/grounded.rs:3192-3214`

**What we're missing**:
- Our `cross-doc` command manually implements clustering
- `Corpus` already has this functionality with better integration
- Could refactor `cross-doc` to use `Corpus` internally

**Benefits**:
- Better identity management
- Automatic track-to-identity linking
- Consistent with library patterns

## 10. Evaluation Framework Integration

**What exists**: 
- `TaskEvaluator` - Comprehensive evaluation
- `EvalSystem` - Unified evaluation API
- `BackendEvaluator` - Backend-specific evaluation

**Location**: `src/eval/`

**What we're missing**:
- Pipeline/batch commands don't use evaluation framework for metrics
- Could add `--evaluate` flag to pipeline to get F1/precision/recall
- Could use `process_with_gold()` for eval command

## Priority Recommendations

### High Priority (Should Implement)
1. **Graph Export** - Key feature for RAG, easy to add
2. **Corpus Management** - Better than manual cross-doc implementation

### Medium Priority (Nice to Have)
3. **Validation Utilities** - Improve robustness
4. **Process Text Utilities** - Simplify code

### Low Priority (Future Enhancements)
5. **Spatial Index** - Performance optimization
6. **Statistics Helper** - Code simplification
7. **Additional Query Helpers** - Feature completeness
8. **HTML Rendering Consistency** - Code quality

## Implementation Strategy

### Phase 1: Graph Export (1-2 hours)
- Add `--export-graph FORMAT` to extract/debug/pipeline
- Support: neo4j, networkx, json
- Use `GraphDocument::from_grounded_document()`

### Phase 2: Corpus Integration (2-3 hours)
- Refactor `cross-doc` to use `Corpus`
- Add `corpus` command for corpus management
- Use `Corpus::resolve_inter_doc_coref()`

### Phase 3: Validation & Utilities (1-2 hours)
- Add `--validate` flag to extract/enhance
- Use `process_text()` in extract command
- Use `add_signal_validated()` where appropriate

### Phase 4: Query Enhancements (1-2 hours)
- Add spatial index support to query
- Add negated/quantified/untracked filters
- Use `doc.stats()` for statistics

## Code Locations

- **Graph Export**: `src/graph.rs:633-850`
- **Corpus**: `src/grounded.rs:3159-3214`
- **Validation**: `src/grounded.rs:1619-1679`
- **Process Text**: `src/grounded.rs:2951-3038`
- **Spatial Index**: `src/grounded.rs:2118-2167`
- **Stats**: `src/grounded.rs:1687-1750`
- **Query Helpers**: `src/grounded.rs:1559-1597`

## Implementation Notes

### Graph Export from GroundedDocument

`GraphDocument` has `from_extraction(entities, relations, coref_chains)` but no direct `from_grounded_document()`. To implement:

1. Convert `Signal` → `Entity` (signals already have location, label, confidence)
2. Extract relations from tracks/identities (if available)
3. Convert tracks → `CorefChain` (use `Track::to_coref_chain()`)
4. Call `GraphDocument::from_extraction()`

**Helper function needed**:
```rust
impl GraphDocument {
    pub fn from_grounded_document(doc: &GroundedDocument) -> Self {
        // Convert signals to entities
        let entities: Vec<Entity> = doc.signals().iter().map(|s| {
            let (start, end) = s.text_offsets().unwrap_or((0, 0));
            Entity::new(s.surface(), EntityType::from_label(s.label()), start, end, s.confidence)
                .with_canonical_id(/* from track if linked */)
        }).collect();
        
        // Get coref chains from tracks
        let coref_chains = Some(doc.to_coref_chains());
        
        // Extract relations (if available in doc)
        let relations = vec![]; // TODO: Extract from doc if relations stored
        
        Self::from_extraction(&entities, &relations, coref_chains.as_deref())
    }
}
```

### Corpus vs. Cross-Doc Command

Current `cross-doc` command manually implements clustering. `Corpus::resolve_inter_doc_coref()` provides:
- Better identity management
- Automatic track-to-identity linking
- Consistent with library patterns
- Built-in similarity thresholding

**Refactoring approach**:
1. Load documents into `Corpus`
2. Call `corpus.resolve_inter_doc_coref(threshold, require_type_match)`
3. Export identities as cross-doc clusters

This would simplify `cmd_crossdoc` significantly.

