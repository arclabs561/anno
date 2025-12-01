# Backend Harmonization Plan

**Date**: 2025-01-XX  
**Goal**: Harmonize all backend implementations to ensure consistent trait coverage, fix inconsistencies, and improve test coverage.

## Design Principles

1. **Consistency**: All backends that can implement a trait should implement it
2. **Completeness**: No dead code - either implement or remove unused traits
3. **Testability**: Every trait implementation must have property-based tests
4. **Backward Compatibility**: Changes must not break existing APIs

## Implementation Strategy

### Phase 1: Catalog & Metadata Fixes
- Fix GLiNERCandle status (WIP → Beta)
- Fix NuNER feature flag (nuner → onnx)
- Update documentation

### Phase 2: Core Trait Implementations
- Implement `GpuCapable` for Candle backends
- Add `BatchCapable` to backends that support batching
- Add `StreamingCapable` to all backends (default impl works for most)

### Phase 3: Specialized Traits
- Implement `DiscontinuousNER` for W2NER
- Consolidate encoder traits
- Remove or implement dead traits

### Phase 4: Testing
- Add property-based tests for all new implementations
- Add integration tests for trait combinations
- Verify backward compatibility

## Trait Implementation Matrix (Target State)

| Backend | Model | BatchCapable | StreamingCapable | GpuCapable | DynamicLabels | RelationCapable | DiscontinuousNER |
|---------|-------|--------------|-----------------|------------|---------------|-----------------|------------------|
| PatternNER | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| HeuristicNER | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| StackedNER | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| GLiNEROnnx | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| GLiNERCandle | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| GLiNER2Onnx | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| GLiNER2Candle | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| NuNER | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| W2NER | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| BertNEROnnx | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| CandleNER | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |

## Test Strategy

### Property-Based Tests (proptest)
- **Confidence bounds**: All entities have confidence in [0, 1]
- **Offset validity**: start <= end <= text.len()
- **Batch consistency**: batch results match sequential results
- **Streaming consistency**: streaming results match non-streaming (with offset adjustment)
- **GPU fallback**: GPU-capable backends work on CPU

### Integration Tests
- Trait combinations (e.g., BatchCapable + StreamingCapable)
- Backend selection (NERExtractor)
- Fallback behavior

### Regression Tests
- Existing functionality continues to work
- Performance doesn't regress

## Backward Compatibility

All changes maintain backward compatibility:
- No breaking changes to `Model` trait
- New trait implementations are additive
- Default implementations preserve existing behavior

