# Repo-Specific Static Analysis

This document describes static analysis tools and rules tailored specifically for the anno NLP/ML evaluation framework.

## Unique Aspects of This Repository

### 1. NLP/ML Evaluation Framework
- Task-dataset-backend evaluation combinations
- Per-example score caching
- Confidence interval computation
- Stratified metrics (by entity type, temporal, etc.)
- Bias analysis (gender, demographic, length, temporal)

### 2. ML Backend Integration
- ONNX Runtime for transformer models
- Candle for pure Rust inference
- HuggingFace model downloads
- Session pooling for performance
- Thread-local backend caching for parallel evaluation

### 3. Text Processing
- Character vs byte offset handling (Unicode)
- Entity span validation
- Text extraction with Unicode correctness
- Multi-modal spans (text + visual)

### 4. Statistical Correctness
- Bessel's correction for variance (n-1)
- Confidence interval computation
- Metric calculations (F1, precision, recall)
- Edge case handling (n=0, n=1, division by zero)

## Custom OpenGrep Rules

### 1. NLP/ML Patterns (`rust-nlp-ml-patterns.yaml`)
Catches issues specific to NLP and ML code:
- Text offset validation (start <= end)
- Character vs byte offset usage
- Confidence score range validation (0.0-1.0)
- Model download error handling
- HuggingFace authentication checks
- ONNX session management
- Tokenizer error context
- Sequence length validation

### 2. Evaluation Framework (`rust-evaluation-framework.yaml`)
Catches evaluation-specific issues:
- Backend recreation in loops
- Per-example score caching patterns
- Confidence interval recomputation
- Stratified metrics computation
- Task-dataset-backend mapping
- Bias stratification edge cases
- Robustness testing limits
- Coreference chain validation

### 3. Anno-Specific (`rust-anno-specific.yaml`)
Catches project-specific patterns:
- Entity offset validation
- Session pool resource management
- Confidence score ranges
- Graph node validation

## Custom Analysis Scripts

### 1. `check-nlp-patterns.sh`
Validates NLP-specific patterns:
- Text offset validation
- Confidence score ranges
- Variance calculation (Bessel's correction)
- Model download error handling
- Unicode handling

### 2. `analyze-evaluation-patterns.sh`
Analyzes evaluation framework:
- Backend reuse patterns
- Per-example score caching
- Parallel evaluation support
- Confidence interval computation
- Task-dataset validation
- Metric computation patterns

### 3. `check-ml-backend-patterns.sh`
Validates ML backend code:
- HuggingFace authentication handling
- Model download error context
- ONNX session pooling
- Tokenizer error handling
- Sequence length validation
- Unsafe code documentation

### 4. `check-evaluation-invariants.sh`
Checks statistical correctness:
- Bessel's correction in variance
- Confidence interval edge cases
- F1/precision/recall zero-checks
- Per-example score reuse
- Stratified metrics computation

### 5. `generate-repo-specific-report.sh`
Combines all repo-specific checks into unified report.

## Usage

### Quick Checks
```bash
# NLP/ML patterns
just check-nlp-patterns

# Evaluation framework
just analyze-eval-patterns

# ML backends
just check-ml-backends

# Evaluation invariants
just check-eval-invariants
```

### Comprehensive Analysis
```bash
# All repo-specific checks
just analysis-nlp-ml

# Generate unified report
just repo-analysis
```

## CI Integration

The `nlp-ml-patterns` job in CI runs:
- NLP pattern checks
- Evaluation framework analysis
- ML backend pattern validation

Results are uploaded as artifacts for review.

## Key Patterns Detected

### ✅ Good Patterns (Already in Code)
- Character offset handling for Unicode (`text.chars().skip().take()`)
- Entity validation with start <= end checks
- Confidence score range validation (0.0-1.0)
- Per-example score caching
- Thread-local backend caching for parallel evaluation
- Bessel's correction in variance calculations

### ⚠️ Patterns to Watch
- Backend recreation in loops (performance)
- Confidence interval recomputation (should use cache)
- Model download error context (could be more helpful)
- Sequence length validation (may be missing in some backends)

## Integration with General Tools

Repo-specific rules complement general static analysis:
- **clippy**: Rust idioms and style
- **cargo-deny**: Dependency security
- **OpenGrep (general)**: Security patterns
- **OpenGrep (repo-specific)**: NLP/ML/evaluation patterns
- **Miri**: Unsafe code validation

## Examples

### Example 1: Text Offset Validation
```rust
// ❌ Bad (caught by rule)
let entity = Entity::new("text", EntityType::Person, 10, 5, 0.9);
// start > end, invalid span

// ✅ Good
if start > end {
    return Err(Error::InvalidSpan { start, end });
}
let entity = Entity::new("text", EntityType::Person, start, end, 0.9);
```

### Example 2: Variance Calculation
```rust
// ❌ Bad (caught by rule)
let variance = scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / scores.len() as f64;
// Population variance (biased)

// ✅ Good
let n = scores.len() as f64;
let variance = if n > 1.0 {
    scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
} else {
    0.0
};
// Sample variance with Bessel's correction
```

### Example 3: Model Download Error Handling
```rust
// ❌ Bad (caught by rule)
let model = repo.get("model.onnx").map_err(|e| Error::Retrieval(format!("Failed: {}", e)))?;

// ✅ Good
let model = repo.get("model.onnx").map_err(|e| {
    let error_msg = format!("{}", e);
    if error_msg.contains("401") || error_msg.contains("Unauthorized") {
        Error::Retrieval(format!(
            "Model '{}' requires HuggingFace authentication. Set HF_TOKEN environment variable.",
            model_id
        ))
    } else {
        Error::Retrieval(format!("Failed to download model '{}': {}", model_id, e))
    }
})?;
```

## Continuous Improvement

As the codebase evolves, update rules to catch new patterns:
1. Identify recurring issues in code reviews
2. Add rules to catch them early
3. Update analysis scripts to validate fixes
4. Document patterns in this file

## References

- [General Static Analysis Setup](STATIC_ANALYSIS_SETUP.md)
- [Creative Uses](STATIC_ANALYSIS_CREATIVE_USES.md)
- [Quick Reference](STATIC_ANALYSIS_QUICK_REFERENCE.md)

