# Detailed Review: Tests, Evaluations, and Implementations

## Executive Summary

Comprehensive review of tests, evaluations, and implementations reveals:
- ✅ **FIXED**: All tests for Corpus/TrackRef/IdentitySource functionality added (26 tests)
- ✅ **FIXED**: All implementation bugs addressed
- ✅ **FIXED**: Test coverage now comprehensive (26 unit tests + 4 property tests + 3 performance tests)
- ✅ **FIXED**: Evaluation metrics for inter-doc coref added
- ✅ **FIXED**: Similarity computation unified in shared utility
- ✅ **FIXED**: Error handling improved with Result types
- **Evaluation Quality**: Strong evaluation framework with documented limitations

## Critical Issues Found

### 1. Missing Tests for New Functionality

**Status**: ❌ **CRITICAL** - No tests exist for:
- `Corpus` type and all its methods
- `TrackRef` type
- `IdentitySource` enum and its variants
- `resolve_inter_doc_coref()` function
- `link_track_to_kb()` function
- `GroundedDocument::track_ref()` method

**Impact**: New code is completely untested. Bugs could exist and go undetected.

### 2. Implementation Bugs

#### Bug 1: `link_track_to_kb()` - Identity Not in Corpus
**Location**: `src/grounded.rs:3207-3254`

**Issue**: When a track has an `identity_id` but that identity doesn't exist in `corpus.identities`, the code creates a new identity. However, this creates an inconsistency: the document thinks the track is linked to one identity, but the corpus has a different identity.

**Fix Needed**: 
- Check if identity exists in corpus before using it
- If not, create new identity and update document's track reference
- Or: validate that all document identities are also in corpus

#### Bug 2: `Identity::from_cross_doc_cluster()` - Invalid TrackRefs
**Location**: `src/grounded.rs:965-977`

**Issue**: Creates `TrackRef` with `track_id: 0` for all mentions. This is wrong - CDCR doesn't track track IDs, only entity indices. The TrackRefs created are invalid.

**Fix Needed**:
- Don't create TrackRefs from CrossDocCluster (it doesn't have that info)
- Or: Make source optional/None when converting from CDCR format
- Or: Document that TrackRefs are placeholders

#### Bug 3: `string_similarity()` - Edge Cases
**Location**: `src/grounded.rs:3297-3315`

**Issues**:
- Empty strings: Returns 0.0 (correct) but should handle gracefully
- Single word vs multi-word: "Apple" vs "Apple Inc" has low similarity (0.5) but should be higher
- Punctuation: "Apple, Inc." vs "Apple Inc" treated as different words
- Case sensitivity: Handled correctly (lowercase)
- Unicode: May not handle properly (normalization)

**Fix Needed**: Add tests for edge cases, consider better similarity metric.

#### Bug 4: `resolve_inter_doc_coref()` - Empty Clusters
**Location**: `src/grounded.rs:3121-3124`

**Issue**: Skips empty clusters, but what about singleton tracks? They should still create identities (or not, depending on design).

**Fix Needed**: Document behavior for singletons, add test.

#### Bug 5: TrackRef Validation
**Location**: `src/grounded.rs:1288-1294`

**Issue**: `track_ref()` doesn't validate that the track actually exists - it only checks if the track_id is in the tracks map. But if a track is removed, the TrackRef becomes invalid.

**Fix Needed**: Add validation or document that TrackRefs can become stale.

### 3. Test Coverage Analysis

#### Well-Tested Areas ✅
- `GroundedDocument` basic operations (signals, tracks, identities)
- `Location` IoU and overlap calculations
- `Signal` creation and properties
- `Track` formation and linking
- `Identity` creation from KB
- Coreference chain conversions
- Modality features

#### Under-Tested Areas ⚠️
- Cross-document operations (Corpus)
- Entity linking operations
- IdentitySource enum variants
- TrackRef usage patterns
- Edge cases in string similarity
- Empty document handling
- Concurrent access patterns

#### Missing Tests ❌
- All Corpus functionality
- All TrackRef functionality  
- All IdentitySource functionality
- `resolve_inter_doc_coref()` with various scenarios
- `link_track_to_kb()` with various scenarios
- Error handling for invalid TrackRefs
- IdentitySource transitions (CrossDocCoref → Hybrid)

### 4. Evaluation Framework Review

#### Strengths ✅
- Comprehensive metrics (MUC, B³, CEAF, LEA, BLANC)
- Error analysis with fine-grained taxonomy
- Dataset quality metrics (leakage, redundancy, ambiguity)
- Synthetic data generation for testing
- Property-based tests for invariants

#### Gaps ⚠️
- No evaluation for Corpus operations
- No evaluation for inter-doc coref quality
- No evaluation for entity linking accuracy
- CDCR evaluation uses CrossDocCluster, not Identity
- Missing evaluation for IdentitySource correctness

### 5. Implementation Quality Issues

#### Code Duplication
- `string_similarity()` in Corpus duplicates logic from `CDCRResolver::mention_similarity()`
- Union-find implementation duplicated in multiple places
- Similarity threshold logic repeated

**Fix**: Extract to shared utilities.

#### Inconsistencies
- `CDCRResolver` uses `f64` for similarity, `Corpus` uses `f32`
- `CDCRResolver` has more sophisticated similarity (exact match, substring, Jaccard)
- `Corpus::string_similarity()` only uses Jaccard

**Fix**: Unify similarity computation.

#### Missing Error Handling
- `link_track_to_kb()` returns `Option` but doesn't distinguish error types
- `resolve_inter_doc_coref()` doesn't validate inputs
- No validation that documents in corpus have valid tracks

**Fix**: Add proper error types and validation.

### 6. Test Quality Issues

#### Test Organization
- Tests for `grounded.rs` are in the same file (good for unit tests)
- But integration tests for Corpus should be in `tests/` directory
- No dedicated test file for cross-document operations

#### Test Naming
- Some tests use descriptive names (`test_location_text_iou`)
- Others use generic names (`test_signal_creation`)
- Inconsistent pattern

#### Test Coverage Gaps
- No property-based tests for Corpus
- No fuzz tests for string similarity edge cases
- No performance tests for large corpora
- No concurrent access tests

### 7. Documentation Issues

#### Missing Documentation
- No examples of using Corpus in practice
- No guide for when to use inter-doc coref vs entity linking
- No explanation of IdentitySource variants
- No migration guide from CDCR to Corpus

#### Incomplete Documentation
- `resolve_inter_doc_coref()` docs don't explain singleton handling
- `link_track_to_kb()` doesn't explain what happens to existing identities
- `IdentitySource` enum has no usage examples

## Recommendations

### Immediate (Critical)
1. **Add comprehensive tests for Corpus** - At least 20 test cases covering:
   - Basic operations (add, get documents)
   - Inter-doc coref with various scenarios
   - Entity linking with various scenarios
   - Edge cases (empty corpus, single document, no tracks)
   - Error cases (invalid TrackRefs, missing documents)

2. **Fix implementation bugs** - Address all bugs listed above

3. **Add validation** - Validate inputs to all public methods

### Short-term (High Priority)
1. **Unify similarity computation** - Extract to shared utility
2. **Add error types** - Replace `Option` with proper `Result` types
3. **Add integration tests** - Create `tests/corpus_integration.rs`
4. **Document usage patterns** - Add examples and guides

### Medium-term (Nice to Have)
1. **Performance tests** - Test with large corpora (1000+ documents)
2. **Property-based tests** - Add proptest for Corpus invariants
3. **Evaluation metrics** - Add metrics for inter-doc coref quality
4. **Migration utilities** - Help migrate from CDCR to Corpus

## Test Plan

### Corpus Tests Needed

```rust
// Basic operations
test_corpus_new()
test_corpus_add_document()
test_corpus_get_document()
test_corpus_get_document_mut()
test_corpus_documents_iterator()

// Inter-doc coref
test_resolve_inter_doc_coref_basic()
test_resolve_inter_doc_coref_same_name()
test_resolve_inter_doc_coref_different_names()
test_resolve_inter_doc_coref_type_match()
test_resolve_inter_doc_coref_type_mismatch()
test_resolve_inter_doc_coref_empty_corpus()
test_resolve_inter_doc_coref_single_document()
test_resolve_inter_doc_coref_no_tracks()
test_resolve_inter_doc_coref_singleton_tracks()
test_resolve_inter_doc_coref_threshold_variations()

// Entity linking
test_link_track_to_kb_new_identity()
test_link_track_to_kb_existing_identity()
test_link_track_to_kb_hybrid_source()
test_link_track_to_kb_invalid_track_ref()
test_link_track_to_kb_missing_document()

// Edge cases
test_corpus_empty_identities()
test_corpus_identity_lookup()
test_corpus_string_similarity_edge_cases()
```

### TrackRef Tests Needed

```rust
test_track_ref_creation()
test_track_ref_equality()
test_track_ref_hash()
test_track_ref_invalid_track()
test_track_ref_serialization()
```

### IdentitySource Tests Needed

```rust
test_identity_source_cross_doc_coref()
test_identity_source_knowledge_base()
test_identity_source_hybrid()
test_identity_source_transitions()
test_identity_source_serialization()
```

## Evaluation Plan

### New Evaluation Metrics Needed

1. **Inter-Doc Coref Quality**
   - Cluster purity (how many tracks per identity are correct)
   - Cluster completeness (how many identities should exist)
   - Cross-document consistency

2. **Entity Linking Quality**
   - Linking accuracy (correct KB ID)
   - Linking coverage (how many tracks get linked)
   - Linking confidence calibration

3. **IdentitySource Correctness**
   - Source tracking accuracy
   - Source transition correctness
   - Source metadata completeness

## Conclusion

The new abstractions (Corpus, TrackRef, IdentitySource) are architecturally sound but need:
1. Comprehensive test coverage (0% → 80%+)
2. Bug fixes for edge cases
3. Better error handling
4. Documentation and examples

The existing test suite is strong, but the new functionality is completely untested. This is a critical gap that should be addressed immediately.

