# Test Coverage Gap Analysis

Comprehensive review of implementation vs test coverage, identifying missing tests that should exist.

## Executive Summary

**Total Test Files**: 43 test files with ~1,633 test cases  
**Coverage**: Strong for core NER backends, evaluation framework, and integration tests  
**Gaps**: Missing tests for utility modules, schema mapping, graph export, language detection, and some advanced features

---

## Critical Gaps (High Priority)

### 1. **Graph Export Module** (`src/graph.rs`)
**Status**: ❌ **NO TESTS FOUND**

Missing tests for:
- `GraphNode::new()`, `with_property()`, `with_mentions_count()`, `with_first_seen()`
- `GraphEdge::new()`, `with_confidence()`, `with_property()`, `with_trigger()`
- `GraphDocument::new()`, `from_extraction()`, `to_cypher()`, `to_networkx_json()`, `to_json_ld()`
- `GraphDocument::with_metadata()`, `node_count()`, `edge_count()`, `is_empty()`
- Edge cases: empty graphs, duplicate nodes, missing relations, invalid canonical_ids
- Export format validation (Cypher syntax, JSON-LD structure, NetworkX format)

**Recommended**: Create `tests/graph_export_tests.rs` with 20+ tests

---

### 2. **Schema Mapping Module** (`src/schema.rs`)
**Status**: ⚠️ **PARTIAL** (only GLiNER2 schema builder tested, not the core schema mapper)

Missing tests for:
- `CanonicalType::name()`, `category()`, `to_entity_type()`
- `DatasetSchema::labels()` for all variants (CoNLL, OntoNotes, MultiNERD, etc.)
- `SchemaMapper::for_dataset()`, `to_canonical()`, `information_loss()`, `to_entity_type()`
- `SchemaMapper::all_losses()`, `label_overlap()`
- `CoarseType::from_canonical()`, `from_label()`
- `map_to_canonical()` utility function
- Round-trip mapping: dataset label → canonical → entity type
- Information loss detection and reporting
- Schema overlap calculations

**Recommended**: Create `tests/schema_mapping_tests.rs` with 30+ tests

---

### 3. **Language Detection** (`src/lang.rs`)
**Status**: ❌ **NO TESTS FOUND**

Missing tests for:
- `Language::is_cjk()` for Chinese, Japanese, Korean
- `Language::is_rtl()` for Arabic, Hebrew
- `detect_language()` with:
  - Pure CJK text (Chinese, Japanese, Korean)
  - RTL text (Arabic, Hebrew)
  - Latin-based languages (English, German, French, Spanish, Italian, Portuguese)
  - Cyrillic (Russian)
  - Mixed scripts
  - Empty text (should default to English)
  - Edge cases: emoji, numbers, punctuation-only

**Recommended**: Create `tests/lang_detection_tests.rs` with 25+ tests

---

### 4. **Auto Backend Selection** (`src/lib.rs`)
**Status**: ❌ **NO TESTS FOUND**

Missing tests for:
- `auto()` function - should return best available backend
- `auto_for(UseCase::BestQuality)` - should prefer GLiNER > BERT > Candle > Stacked
- `auto_for(UseCase::Fast)` - should return StackedNER
- `auto_for(UseCase::ZeroShot)` - should return GLiNER or error
- `auto_for(UseCase::NestedEntities)` - should return W2NER
- `auto_for(UseCase::Production)` - should prefer stable backends
- `available_backends()` - should list all backends with availability status
- Feature-gated behavior (what happens when `onnx` or `candle` features disabled)
- Fallback behavior when preferred backend unavailable

**Recommended**: Add to `tests/integration.rs` or create `tests/auto_backend_tests.rs` with 15+ tests

---

### 5. **EntityViewport** (`src/entity.rs`)
**Status**: ⚠️ **MINIMAL** (only 1 test in `new_features_integration.rs`)

Missing tests for:
- `EntityViewport::as_str()` for all variants
- `EntityViewport::is_professional()` for Business, Legal, Technical, Academic, Political, Media
- `EntityViewport::is_personal()` for Personal
- `EntityViewport::from_str()` parsing (all variants, case-insensitive, custom)
- `EntityViewport::Display` trait implementation
- `Entity::set_viewport()`, `has_viewport()`, `viewport_or_default()`, `matches_viewport()`
- Viewport filtering in entity collections
- Custom viewport creation and usage

**Recommended**: Add to `tests/entity.rs` or create `tests/entity_viewport_tests.rs` with 15+ tests

---

### 6. **EntityCategory Methods** (`src/entity.rs`)
**Status**: ⚠️ **PARTIAL** (some usage in tests, but no dedicated tests)

Missing tests for:
- `EntityCategory::requires_ml()` for all categories
- `EntityCategory::pattern_detectable()` for all categories
- `EntityCategory::is_relation()` for Relation category
- `EntityCategory::as_str()` for all variants
- `EntityCategory::Display` trait implementation
- Category-based filtering logic

**Recommended**: Add to `tests/entity.rs` with 10+ tests

---

## Medium Priority Gaps

### 7. **MockModel Validation** (`src/lib.rs`)
**Status**: ⚠️ **PARTIAL** (basic usage tested, but not all validation paths)

Missing tests for:
- `MockModel::without_validation()` - should skip offset validation
- `MockModel::validate_entities()` error cases:
  - Entity end > text length
  - Entity text mismatch with source
  - Character offset vs byte offset handling
- `MockModel` with invalid entities (start >= end, confidence out of range)
- `MockModel::with_types()` and `supported_types()` behavior

**Recommended**: Add to `tests/integration.rs` or create `tests/mock_model_tests.rs` with 10+ tests

---

### 8. **Offset Conversion Utilities** (`src/offset.rs`)
**Status**: ⚠️ **PARTIAL** (some usage, but no dedicated tests)

Missing tests for:
- `bytes_to_chars()` with:
  - ASCII-only text (should be 1:1)
  - UTF-8 multi-byte sequences
  - Emoji and surrogate pairs
  - Edge cases: empty string, single char, boundary conditions
- `chars_to_bytes()` with same cases
- `is_ascii()` helper
- `OffsetMapping` trait implementations
- `SpanConverter` for text/visual/token spans
- Round-trip conversions (bytes → chars → bytes)

**Recommended**: Create `tests/offset_conversion_tests.rs` with 20+ tests

---

### 9. **Similarity Module** (`src/similarity.rs`)
**Status**: ⚠️ **UNKNOWN** (need to check if tests exist)

Missing tests for:
- String similarity functions (if any)
- Entity similarity calculations
- Threshold-based matching
- Edge cases: empty strings, identical strings, completely different strings

**Recommended**: Review `src/similarity.rs` and create tests if missing

---

### 10. **GroundedDocument Advanced Methods** (`src/grounded.rs`)
**Status**: ⚠️ **PARTIAL** (basic operations tested, but not all methods)

Missing tests for:
- `GroundedDocument::track_ref()` - get TrackRef for a signal
- `GroundedDocument::identity_ref()` - get IdentityRef for a track
- `GroundedDocument::to_coref_document()` - conversion to evaluation format
- `GroundedDocument::from_entities()` - construction from Entity list
- Edge cases: orphaned signals, tracks without identities, invalid references
- Multi-document operations (if any)

**Recommended**: Add to `tests/corpus_tests.rs` or `tests/grounded_multimodal.rs` with 10+ tests

---

### 11. **Corpus Advanced Operations** (`src/grounded.rs`)
**Status**: ⚠️ **PARTIAL** (basic operations tested, but advanced methods need more coverage)

Missing tests for:
- `Corpus::resolve_inter_doc_coref()` with:
  - Various similarity thresholds
  - `require_type_match` flag behavior
  - Empty corpus, single document, no tracks
  - Large corpora (performance)
- `Corpus::link_track_to_kb()` with:
  - Valid KB IDs
  - Invalid/missing KB IDs
  - Identity creation vs reuse
  - Track reference updates
- Error handling: invalid TrackRefs, missing documents, missing identities
- Concurrent access (if supported)

**Recommended**: Add to `tests/corpus_tests.rs` with 15+ tests

---

### 12. **IdentitySource Enum** (`src/grounded.rs`)
**Status**: ⚠️ **PARTIAL** (usage tested, but enum methods not explicitly tested)

Missing tests for:
- All `IdentitySource` variants (KB, InterDocCoref, Manual, CrossDocCluster)
- `IdentitySource::as_str()` if it exists
- `IdentitySource::Display` if implemented
- Conversion between sources
- Source-based filtering

**Recommended**: Add to `tests/corpus_tests.rs` with 5+ tests

---

## Lower Priority Gaps (Nice to Have)

### 13. **TypeMapper** (`src/entity.rs`)
**Status**: ⚠️ **UNKNOWN** (need to check if tests exist)

Missing tests for:
- Custom type creation and mapping
- Type normalization
- Type equality and hashing
- Integration with EntityType

**Recommended**: Review and add tests if missing

---

### 14. **Provenance Serialization** (`src/entity.rs`)
**Status**: ⚠️ **PARTIAL** (basic usage tested)

Missing tests for:
- `Provenance::ml()`, `pattern()`, `heuristic()` creation
- Serialization round-trips (JSON)
- Provenance comparison and equality
- Provenance-based filtering

**Recommended**: Add to `tests/serialization_tests.rs` with 5+ tests

---

### 15. **HierarchicalConfidence** (`src/types/confidence.rs`)
**Status**: ⚠️ **PARTIAL** (some usage in tests)

Missing tests for:
- All constructor methods
- Confidence clamping and validation
- Conversion to/from f64
- Arithmetic operations (if any)
- Edge cases: NaN, infinity, negative values

**Recommended**: Add to `tests/types_tests.rs` or create new file with 10+ tests

---

### 16. **DiscontinuousSpan** (`src/entity.rs`)
**Status**: ⚠️ **PARTIAL** (some tests in `discontinuous_span_tests.rs`)

Missing tests for:
- All constructor methods
- Span merging and splitting
- Overlap detection
- Boundary validation
- Conversion to/from continuous spans

**Recommended**: Review `tests/discontinuous_span_tests.rs` and add missing cases

---

### 17. **RaggedBatch** (`src/entity.rs`)
**Status**: ⚠️ **PARTIAL** (some property tests exist)

Missing tests for:
- Construction from sequences
- Padding behavior
- Token preservation
- Document range calculations
- Edge cases: empty batch, single document, very large batches

**Recommended**: Add to `tests/entity.rs` with 10+ tests

---

### 18. **Error Types** (`src/error.rs`)
**Status**: ⚠️ **UNKNOWN** (need to check)

Missing tests for:
- All error variants
- Error message formatting
- Error conversion (From implementations)
- Error context and chaining

**Recommended**: Review `src/error.rs` and create `tests/error_tests.rs` if missing

---

## Integration Test Gaps

### 19. **End-to-End Workflows**
Missing integration tests for:
- Full pipeline: Text → Entities → Graph Export
- Full pipeline: Text → Entities → Corpus → Inter-Doc Coref → Graph Export
- Full pipeline: Text → Entities → Schema Mapping → Evaluation
- Multi-backend comparison workflows
- Error propagation through pipeline

**Recommended**: Add to `tests/integration_full_pipeline.rs` with 5+ tests

---

### 20. **CLI Integration**
**Status**: ⚠️ **PARTIAL** (some tests in `cli_integration.rs`)

Missing tests for:
- All CLI subcommands
- Error handling in CLI
- Output format validation
- Feature-gated commands

**Recommended**: Review `tests/cli_integration.rs` and add missing cases

---

## Property-Based Test Gaps

### 21. **Schema Mapping Properties**
Missing property tests for:
- Schema mapping is idempotent (canonical → canonical = same)
- Schema mapping preserves category (Agent → Agent)
- Information loss is documented for all mappings
- Label overlap is symmetric

**Recommended**: Add to `tests/schema_mapping_tests.rs` with proptest

---

### 22. **Graph Export Properties**
Missing property tests for:
- Graph export is reversible (entities → graph → entities, with some loss)
- Node deduplication works correctly
- Edge creation preserves relation types
- Export formats are valid (Cypher syntax, JSON structure)

**Recommended**: Add to `tests/graph_export_tests.rs` with proptest

---

### 23. **Language Detection Properties**
Missing property tests for:
- Language detection is consistent (same text → same language)
- Language detection handles mixed scripts
- Language detection never panics on arbitrary input

**Recommended**: Add to `tests/lang_detection_tests.rs` with proptest

---

## Performance Test Gaps

### 24. **Graph Export Performance**
Missing performance tests for:
- Large graphs (1000+ nodes, 5000+ edges)
- Export format generation time
- Memory usage for large graphs

**Recommended**: Add to `tests/slow_benchmarks.rs` or create `benches/graph_export.rs`

---

### 25. **Schema Mapping Performance**
Missing performance tests for:
- Large dataset mapping (10000+ labels)
- Information loss calculation time
- Label overlap calculation time

**Recommended**: Add to `tests/slow_benchmarks.rs`

---

## Summary by Priority

### Critical (Should exist now)
1. Graph Export Module (0% coverage)
2. Schema Mapping Module (partial coverage)
3. Language Detection (0% coverage)
4. Auto Backend Selection (0% coverage)
5. EntityViewport (minimal coverage)
6. EntityCategory Methods (partial coverage)

### High Priority (Should add soon)
7. MockModel Validation
8. Offset Conversion Utilities
9. Similarity Module
10. GroundedDocument Advanced Methods
11. Corpus Advanced Operations
12. IdentitySource Enum

### Medium Priority (Nice to have)
13-18. Various utility types and modules

### Integration & Property Tests
19-23. End-to-end workflows and property-based tests

### Performance Tests
24-25. Large-scale performance validation

---

## Recommended Action Plan

1. **Immediate**: Create tests for Graph Export, Schema Mapping, Language Detection (3 new test files)
2. **Short-term**: Add tests for Auto Backend Selection, EntityViewport, EntityCategory (extend existing files)
3. **Medium-term**: Fill gaps in GroundedDocument, Corpus, MockModel (extend existing files)
4. **Long-term**: Add property-based tests and performance tests

**Estimated effort**: 
- Critical gaps: ~100 new tests, 3-4 new test files
- High priority: ~50 new tests, extend existing files
- Medium priority: ~30 new tests, extend existing files

