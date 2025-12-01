# Backend and Interface Review

**Date**: 2025-01-XX  
**Scope**: All backend implementations and interface definitions in `src/backends/`

## Executive Summary

The backend architecture is well-designed with a sealed `Model` trait providing a consistent interface. The codebase implements 12+ backends ranging from zero-dependency pattern matching to state-of-the-art zero-shot NER models. The design uses capability marker traits effectively, but there are some inconsistencies in trait implementation coverage and a few architectural improvements that could be made.

**Overall Assessment**: ✅ **Strong** - Well-architected with room for incremental improvements.

---

## 1. Interface Design

### 1.1 Core `Model` Trait

**Location**: `src/lib.rs:734-760`

**Strengths**:
- ✅ Sealed trait pattern prevents external implementations (good for API stability)
- ✅ Minimal, focused interface: `extract_entities()`, `supported_types()`, `is_available()`, `name()`, `description()`
- ✅ All backends consistently implement the trait
- ✅ Good documentation with examples

**Issues**:
- ⚠️ `language` parameter is `Option<&str>` but most backends ignore it (only `AutoNER` uses it)
- ⚠️ No standardized way to pass backend-specific configuration (thresholds, batch size, etc.)

**Recommendations**:
1. Consider making `language` more prominent or documenting which backends use it
2. Add a `Config` associated type pattern for backend-specific options:
   ```rust
   trait Model: sealed::Sealed + Send + Sync {
       type Config: Default;
       fn with_config(config: Self::Config) -> Result<Self>;
       // ... existing methods
   }
   ```

### 1.2 Capability Marker Traits

**Location**: `src/lib.rs:784-928`

**Implemented Traits**:
- `BatchCapable` - Efficient batch processing
- `GpuCapable` - GPU acceleration support
- `StreamingCapable` - Chunked/streaming extraction
- `NamedEntityCapable` - Named entity extraction (Person/Org/Location)
- `StructuredEntityCapable` - Pattern-based entities (Date/Money/Email)
- `RelationCapable` - Joint entity-relation extraction
- `DynamicLabels` - Zero-shot/custom entity types
- `CalibratedConfidence` - Calibrated confidence scores
- `VisualCapable` - Image/multimodal support

**Strengths**:
- ✅ Clear separation of concerns
- ✅ Default implementations where appropriate (e.g., `BatchCapable::extract_entities_batch()`)
- ✅ Good documentation explaining when to use each trait

**Issues**:
- ⚠️ **Inconsistent implementation**: Not all backends that could implement these traits do so
- ⚠️ `GpuCapable` is defined but **no backend implements it** (GLiNERCandle, CandleNER could)
- ⚠️ `CalibratedConfidence` and `VisualCapable` are defined but **never implemented**
- ⚠️ `NamedEntityCapable` and `StructuredEntityCapable` are marker traits with no methods (could add helper methods)

**Implementation Coverage**:

| Backend | BatchCapable | StreamingCapable | DynamicLabels | RelationCapable | GpuCapable |
|---------|--------------|------------------|---------------|-----------------|------------|
| RegexNER | ✅ | ✅ | ❌ | ❌ | ❌ |
| HeuristicNER | ❌ | ❌ | ❌ | ❌ | ❌ |
| StackedNER | ✅ | ✅ | ❌ | ❌ | ❌ |
| GLiNEROnnx | ✅ | ✅ | ✅ (via ZeroShotNER) | ❌ | ❌ |
| GLiNERCandle | ✅ | ✅ | ✅ (via ZeroShotNER) | ❌ | ❌ |
| GLiNER2Onnx | ✅ | ✅ | ✅ | ✅ | ❌ |
| GLiNER2Candle | ✅ | ✅ | ✅ | ✅ | ❌ |
| NuNER | ❌ | ❌ | ✅ (via ZeroShotNER) | ❌ | ❌ |
| W2NER | ❌ | ❌ | ❌ | ✅ | ❌ |
| BertNEROnnx | ❌ | ❌ | ❌ | ❌ | ❌ |
| CandleNER | ❌ | ❌ | ❌ | ❌ | ❌ |

**Recommendations**:
1. Implement `GpuCapable` for Candle backends (GLiNERCandle, CandleNER, GLiNER2Candle)
2. Add `BatchCapable` to NuNER, W2NER, BertNEROnnx, CandleNER (they can batch)
3. Add `StreamingCapable` to more backends (most can support it)
4. Consider removing or implementing `CalibratedConfidence` and `VisualCapable` (currently dead code)

### 1.3 Specialized Traits

**Location**: `src/backends/inference.rs`

**Traits Defined**:
- `TextEncoder` - Text → embeddings
- `LabelEncoder` - Labels → embeddings  
- `BiEncoder` - Combined text + label encoding
- `ZeroShotNER` - Custom entity types at runtime
- `RelationExtractor` - Joint entity-relation extraction
- `DiscontinuousNER` - Non-contiguous entity spans
- `LateInteraction` - Span-label similarity computation

**Strengths**:
- ✅ Excellent documentation explaining bi-encoder architecture
- ✅ Research-backed design (GLiNER, W2NER papers)
- ✅ Clear separation: encoding vs. matching vs. extraction

**Issues**:
- ⚠️ `DiscontinuousNER` and `LateInteraction` are defined but **never used** in backend implementations
- ⚠️ `TextEncoder` and `LabelEncoder` are duplicated in `encoder_candle.rs` (different trait definitions)

**Recommendations**:
1. Consolidate encoder traits (either use `inference.rs` versions or create a unified trait)
2. Implement `DiscontinuousNER` for W2NER (it supports discontinuous entities)
3. Document why `LateInteraction` exists if it's not used

---

## 2. Backend Implementations

### 2.1 Zero-Dependency Backends

#### RegexNER
**File**: `src/backends/pattern.rs`

**Strengths**:
- ✅ Fast (~400ns per call)
- ✅ High precision on structured entities
- ✅ Proper Unicode handling (byte → char offset conversion)
- ✅ Comprehensive pattern coverage (dates, money, emails, URLs, phones, mentions, hashtags)
- ✅ Implements `BatchCapable` and `StreamingCapable`

**Issues**:
- ⚠️ Hardcoded patterns (not configurable)
- ⚠️ No language-specific patterns (all patterns are English-centric)

**Recommendations**:
- Consider making patterns configurable via `PatternConfig`
- Add language-specific date/time patterns

#### HeuristicNER
**File**: `src/backends/heuristic.rs`

**Strengths**:
- ✅ Zero dependencies
- ✅ Handles CJK text
- ✅ Capitalization-based heuristics work reasonably well

**Issues**:
- ⚠️ Does not implement `BatchCapable` or `StreamingCapable` (could easily do so)
- ⚠️ Hardcoded entity lists (KNOWN_ORGS, KNOWN_PERSONS, etc.)
- ⚠️ No confidence calibration (always returns 0.5)

**Recommendations**:
- Add `BatchCapable` and `StreamingCapable` implementations
- Make entity lists configurable or loadable from files
- Implement confidence scoring based on capitalization patterns

#### StackedNER
**File**: `src/backends/stacked.rs`

**Strengths**:
- ✅ Composable architecture (layers can be combined)
- ✅ Conflict resolution strategies (LongestSpan, HighestConfidence, etc.)
- ✅ Implements `BatchCapable` and `StreamingCapable`
- ✅ Good builder pattern

**Issues**:
- ⚠️ No validation that layers don't conflict (e.g., two RegexNER layers)
- ⚠️ Conflict resolution happens per-entity, not globally (could be optimized)

**Recommendations**:
- Add layer validation in builder
- Consider global conflict resolution optimization

### 2.2 ONNX Backends

#### GLiNEROnnx
**File**: `src/backends/gliner_onnx.rs`

**Strengths**:
- ✅ Manual ONNX implementation (full control)
- ✅ Implements `ZeroShotNER`, `BatchCapable`, `StreamingCapable`
- ✅ Good error handling
- ✅ Supports quantized models

**Issues**:
- ⚠️ Complex code (~800 lines)
- ⚠️ Manual prompt formatting (could drift from reference)
- ⚠️ No `GpuCapable` implementation (ONNX supports GPU)

**Recommendations**:
- Add `GpuCapable` implementation (check ONNX execution provider)
- Consider extracting prompt formatting to shared module

#### GLiNER2Onnx / GLiNER2Candle
**File**: `src/backends/gliner2.rs`

**Strengths**:
- ✅ Multi-task extraction (NER + classification + structure)
- ✅ Comprehensive `TaskSchema` API
- ✅ Implements `ZeroShotNER`, `RelationExtractor`, `BatchCapable`, `StreamingCapable`
- ✅ Well-documented with examples

**Issues**:
- ⚠️ Very large file (~2700 lines) - could be split
- ⚠️ Duplicate code between ONNX and Candle implementations
- ⚠️ No `GpuCapable` for Candle version

**Recommendations**:
- Split into multiple files (core, onnx, candle)
- Extract shared logic to reduce duplication
- Add `GpuCapable` for Candle version

#### NuNER
**File**: `src/backends/nuner.rs`

**Strengths**:
- ✅ Zero-shot token classifier
- ✅ Handles arbitrary-length entities
- ✅ Dynamic span tensor generation (handles different model variants)
- ✅ Implements `ZeroShotNER`

**Issues**:
- ⚠️ Does not implement `BatchCapable` or `StreamingCapable` (could easily do so)
- ⚠️ Limited documentation compared to GLiNER

**Recommendations**:
- Add `BatchCapable` and `StreamingCapable` implementations
- Expand documentation with examples

#### W2NER
**File**: `src/backends/w2ner.rs`

**Strengths**:
- ✅ Handles nested and discontinuous entities
- ✅ Implements `RelationCapable`
- ✅ Good error handling for authentication issues

**Issues**:
- ⚠️ Does not implement `BatchCapable` or `StreamingCapable`
- ⚠️ Does not implement `DiscontinuousNER` trait (even though it supports it)
- ⚠️ Model requires authentication (documented but limits usability)

**Recommendations**:
- Add `BatchCapable` and `StreamingCapable` implementations
- Implement `DiscontinuousNER` trait
- Consider alternative nested NER models that don't require auth

#### BertNEROnnx
**File**: `src/backends/onnx.rs`

**Strengths**:
- ✅ Reliable, well-tested
- ✅ Good error handling
- ✅ Supports quantized models

**Issues**:
- ⚠️ Does not implement `BatchCapable` or `StreamingCapable` (ONNX can batch)
- ⚠️ Fixed entity types (PER/ORG/LOC/MISC) - not extensible
- ⚠️ No `GpuCapable` implementation

**Recommendations**:
- Add `BatchCapable`, `StreamingCapable`, and `GpuCapable` implementations
- Document that this is for fixed types only (vs. zero-shot backends)

### 2.3 Candle Backends

#### GLiNERCandle
**File**: `src/backends/gliner_candle.rs`

**Strengths**:
- ✅ Pure Rust implementation
- ✅ Automatic PyTorch → Safetensors conversion
- ✅ Implements `ZeroShotNER`, `BatchCapable`, `StreamingCapable`
- ✅ Metal/CUDA support (via Candle)

**Issues**:
- ⚠️ Does not implement `GpuCapable` (even though it supports GPU)
- ⚠️ Status marked as "WIP" in catalog but seems complete
- ⚠️ Large file (~1000 lines)

**Recommendations**:
- Add `GpuCapable` implementation (check Candle device)
- Update catalog status to "Beta" or "Stable"
- Consider splitting into multiple files

#### CandleNER
**File**: `src/backends/candle.rs`

**Strengths**:
- ✅ Pure Rust BERT implementation
- ✅ Automatic PyTorch → Safetensors conversion
- ✅ Metal/CUDA support

**Issues**:
- ⚠️ Does not implement `BatchCapable`, `StreamingCapable`, or `GpuCapable`
- ⚠️ Lower F1 than ONNX version (~74% vs ~86%)

**Recommendations**:
- Add `BatchCapable`, `StreamingCapable`, and `GpuCapable` implementations
- Investigate F1 gap (could be model weights or implementation difference)

### 2.4 Composite Backends

#### NERExtractor
**File**: `src/backends/extractor.rs`

**Strengths**:
- ✅ Unified interface with fallback
- ✅ Hybrid mode (ML + patterns)
- ✅ Good backend selection logic (`best_available()`, `fast()`, `best_quality()`)
- ✅ Implements `Model` trait

**Issues**:
- ⚠️ Fallback logic is simple (tries primary, then fallback) - could be smarter
- ⚠️ No way to configure which backends to try

**Recommendations**:
- Add configuration for backend priority
- Consider more sophisticated fallback (e.g., try multiple backends in parallel)

#### AutoNER
**File**: `src/backends/router.rs`

**Strengths**:
- ✅ Language-aware routing
- ✅ Automatic backend selection

**Issues**:
- ⚠️ Limited implementation (only 28 lines)
- ⚠️ No documentation on how language detection works

**Recommendations**:
- Expand documentation
- Consider using a language detection library

---

## 3. Architecture & Design Patterns

### 3.1 Sealed Trait Pattern

**Implementation**: `src/lib.rs:734` (trait is sealed via `sealed::Sealed`)

**Strengths**:
- ✅ Prevents external implementations (API stability)
- ✅ Allows adding methods in minor versions

**Recommendations**:
- Document the sealed trait pattern in architecture docs

### 3.2 Error Handling

**Pattern**: All backends return `Result<Vec<Entity>>` with `crate::Error`

**Strengths**:
- ✅ Consistent error types
- ✅ Good error messages (especially in W2NER for auth errors)

**Issues**:
- ⚠️ Some backends have backend-specific errors (e.g., `ParseError` in `llm_prompt.rs`) that don't integrate with main `Error` type
- ⚠️ Error conversion could be more systematic

**Recommendations**:
- Standardize error types across all backends
- Use `thiserror` or `anyhow` for better error context

### 3.3 Configuration

**Current State**: Each backend has its own config struct (e.g., `GLiNERConfig`, `BertNERConfig`, `W2NERConfig`)

**Issues**:
- ⚠️ No unified configuration interface
- ⚠️ Some backends have no config (RegexNER, HeuristicNER)

**Recommendations**:
- Consider a unified `BackendConfig` trait or enum
- Add configuration to zero-dependency backends (e.g., pattern thresholds)

### 3.4 Feature Flags

**Current State**: Backends are feature-gated (`onnx`, `candle`, etc.)

**Strengths**:
- ✅ Good separation of optional dependencies
- ✅ Clear feature documentation

**Issues**:
- ⚠️ Some backends are marked as requiring features but are always available (e.g., NuNER is in catalog as requiring "nuner" feature but uses "onnx")

**Recommendations**:
- Review feature flags for accuracy
- Document which features enable which backends

---

## 4. Documentation

### 4.1 Strengths

- ✅ Excellent module-level documentation in `inference.rs` explaining bi-encoder architecture
- ✅ Good examples in trait documentation
- ✅ `ARCHITECTURE.md` provides comprehensive overview
- ✅ `catalog.rs` documents all backends with status

### 4.2 Issues

- ⚠️ Some backends have minimal documentation (e.g., `AutoNER`, `RuleBasedNER`)
- ⚠️ No comparison table showing which backends implement which traits
- ⚠️ `PROBLEMS.md` documents fixed issues but could be archived

**Recommendations**:
- Add trait implementation matrix to `ARCHITECTURE.md`
- Expand documentation for less-documented backends
- Archive `PROBLEMS.md` to `docs/archive/` since all issues are fixed

---

## 5. Testing

### 5.1 Coverage

**Observation**: Backends have varying levels of test coverage

**Recommendations**:
- Add property-based tests for all backends (invariants: confidence in [0,1], valid offsets, etc.)
- Add integration tests for trait implementations
- Test fallback behavior in `NERExtractor`

---

## 6. Recommendations Summary

### High Priority

1. **Implement missing trait implementations**:
   - Add `GpuCapable` to Candle backends
   - Add `BatchCapable` to NuNER, W2NER, BertNEROnnx, CandleNER
   - Add `StreamingCapable` to more backends
   - Implement `DiscontinuousNER` for W2NER

2. **Remove or implement dead traits**:
   - Either implement `CalibratedConfidence` and `VisualCapable` or remove them

3. **Consolidate encoder traits**:
   - Unify `TextEncoder` and `LabelEncoder` definitions

### Medium Priority

4. **Improve configuration**:
   - Add unified configuration interface
   - Make RegexNER and HeuristicNER configurable

5. **Split large files**:
   - Split `gliner2.rs` (2700 lines) into multiple files
   - Consider splitting `gliner_candle.rs` (1000 lines)

6. **Update catalog**:
   - Mark GLiNERCandle as "Beta" or "Stable" (not "WIP")
   - Fix feature flags (NuNER uses "onnx", not "nuner")

### Low Priority

7. **Documentation improvements**:
   - Add trait implementation matrix
   - Expand documentation for less-documented backends
   - Archive `PROBLEMS.md`

8. **Error handling**:
   - Standardize error types
   - Use better error context libraries

---

## 7. Conclusion

The backend architecture is **well-designed** with a clear separation of concerns, good use of traits, and comprehensive coverage of NER approaches (regex-based, heuristic, traditional ML, zero-shot, nested). The main areas for improvement are:

1. **Completeness**: Many backends could implement more capability traits
2. **Consistency**: Some traits are defined but never used
3. **Organization**: Some files are very large and could be split

Overall, this is a **production-ready** codebase with room for incremental improvements. The sealed trait pattern and capability markers provide a solid foundation for future extensions.

