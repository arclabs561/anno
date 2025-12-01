# Backward Review: What We Missed

**Date**: 2025-01-XX  
**Purpose**: Review implementation backward to identify missed items from research findings

## Review Process

1. ✅ Checked original research findings
2. ✅ Verified implementations match research recommendations
3. ⚠️ Identified gaps and partial implementations

## Missed or Incomplete Items

### 1. Per-Example Score Tracking for Confidence Intervals

**Issue**: `compute_confidence_intervals_improved()` recomputes predictions on a sample, which is expensive and duplicates work.

**Better Approach**: Track per-example F1/precision/recall during main evaluation loop, then compute CI from those scores.

**Status**: Partially implemented (samples and recomputes)

**Recommendation**: Modify `evaluate_ner_task()` to return per-example scores alongside aggregate metrics.

### 2. Proper Stratified Metrics Computation

**Issue**: `compute_stratified_metrics()` uses aggregate metrics as placeholder for per-type metrics. Doesn't compute surface form or mention characteristics.

**Missing**:
- Per-type F1/precision/recall from actual per-example scores
- Surface form detection (proper noun vs common noun vs pronoun)
- Mention characteristics (capitalized, partial name, metonym detection)

**Status**: Structure exists, computation is placeholder

**Recommendation**: Add per-example tracking in evaluation loop, then stratify by dimensions.

### 3. Temporal Stratification

**Issue**: Structure exists but no temporal metadata in datasets.

**Missing**:
- Entity creation date tracking
- KB version metadata
- Temporal stratum assignment (pre/post cutoff)

**Status**: Structure ready, no data source

**Recommendation**: Add temporal metadata to dataset loaders or use dataset-specific heuristics.

### 4. Robustness Testing Integration

**Issue**: Robustness testing is computed but not integrated into main evaluation flow properly.

**Missing**:
- Robustness results not stored in TaskEvalResult when computed
- No reporting of robustness scores in outputs

**Status**: Function exists, integration incomplete

**Recommendation**: Ensure robustness results are properly stored and reported.

### 5. KB Version Tracking and URI Validation

**Issue**: Not implemented at all.

**Missing**:
- KB version metadata
- URI validation against current KB
- Emerging entity (NIL) separation
- URI set expansion (owl:sameAs links)

**Status**: Not started

**Recommendation**: Add KB version tracking to datasets, validate URIs, separate NIL performance.

### 6. Inter-Document Coreference Specific Evaluation

**Issue**: Inter-doc coref uses same metrics as intra-doc, but research shows they need different evaluation.

**Missing**:
- Cross-document specific metrics
- Generalization validation (train/test domain split)
- NED integration for cross-doc evaluation

**Status**: Not implemented

**Recommendation**: Add inter-doc specific evaluation mode with generalization checks.

### 7. Familiarity with Embeddings

**Issue**: `from_type_sets_with_embeddings()` exists but no embedding function provided.

**Missing**:
- Integration with encoder backends for label embeddings
- Automatic embedding computation for familiarity

**Status**: Function exists, no integration

**Recommendation**: Add embedding computation using existing encoder infrastructure.

### 8. Reporting Stratified Metrics

**Issue**: Stratified metrics are computed but not reported in evaluation outputs.

**Missing**:
- Chain-length stats in markdown reports
- Familiarity scores in outputs
- Stratified metrics in summary tables

**Status**: Computed but not reported

**Recommendation**: Update report generation to include stratified metrics.

### 9. Weak Annotation Matching Option

**Issue**: Boundary error documentation explains greedy matching, but no weak matching option exists.

**Missing**:
- IoU-based partial credit for boundary errors
- Weak annotation matching mode (URI match + position overlap)

**Status**: Documented but not implemented

**Recommendation**: Add weak matching mode as optional evaluation mode.

### 10. Per-Chain-Length Metrics in Coref Reports

**Issue**: Chain-length stats computed but not included in markdown reports.

**Missing**:
- Chain-length breakdown in evaluation summaries
- Per-chain-length F1 in tables

**Status**: Computed but not reported

**Recommendation**: Update `to_markdown()` to include chain-length stratification.

## Priority Fixes

### High Priority
1. **Report stratified metrics** - Computed but not shown
2. **Fix confidence interval computation** - Avoid recomputation, use per-example scores
3. **Complete stratified metrics** - Compute actual per-type/surface-form metrics

### Medium Priority
4. **Integrate robustness reporting** - Show robustness scores in outputs
5. **Add familiarity to reports** - Display familiarity scores for zero-shot evaluations
6. **KB version tracking** - Add metadata support

### Low Priority
7. **Inter-doc specific evaluation** - Add specialized metrics
8. **Weak annotation matching** - Add optional mode
9. **Embedding-based familiarity** - Integrate with encoders

## Implementation Notes

Most core functionality is implemented. The main gaps are:
- **Reporting**: Metrics computed but not displayed
- **Efficiency**: Some recomputation that could be avoided
- **Completeness**: Some stratified dimensions not fully computed

The architecture is sound - these are mostly integration and reporting tasks.

