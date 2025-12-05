# Math Documentation Guide: Selective Mathematical Explanations

## Philosophy

Add mathematical explanations only where:
1. **Relevant to the project's core purpose** (e.g., evaluation metrics, clustering algorithms)
2. **Aligned with implementation complexity** (complex algorithms deserve more explanation)
3. **Necessary for correct usage** (formulas that affect behavior or parameters)

Avoid over-explaining simple utility functions or well-known concepts.

## Current State Assessment

### What We Have
- ✅ Modularity formula in `strata/src/leiden.rs` (concise, appropriate)
- ✅ Box embedding formulas in `anno/src/backends/box_embeddings_training.rs` (research-focused)
- ✅ Late interaction scoring in `anno/src/backends/inference.rs` (brief, technical)
- ✅ Similarity functions (basic, sufficient)

### When to Add More Detail
- Complex algorithms (Leiden, box embeddings) → formula + brief intuition
- Evaluation metrics (P/R/F1) → formula + when to use each
- Calibration metrics (ECE, Brier) → formula + interpretation guidance
- **Not needed**: Simple utilities, well-known concepts, obvious formulas

## Best Practices for Math in Rust Docs

### 1. Markup Options

**Recommended: ASCII Math (primary)**

```rust
/// Formula: `similarity(a, b) = (a · b) / (||a|| × ||b||)`
///
/// Where `a · b` is dot product and `||a||` is L2 norm.
```

**Why ASCII?**
- Works everywhere (offline docs, GitHub, docs.rs)
- No external dependencies
- Simple and portable

**Optional: KaTeX for docs.rs** (only if formula is complex and visual rendering adds value)

### 2. When to Add Mathematical Detail

**Add formula + brief explanation:**
- Complex algorithms (Leiden modularity, box embeddings)
- Evaluation metrics that affect interpretation (ECE, Brier)
- Parameters that need tuning guidance (resolution, thresholds)

**Keep minimal:**
- Well-known formulas (cosine similarity, F1 score)
- Simple utilities (string similarity, basic math)
- Obvious calculations

**Structure for complex cases:**
```rust
/// [Brief description]
///
/// Formula: `[ASCII formula]`
///
/// [One sentence intuition if non-obvious]
///
/// # Example
/// [Code example]
```

## Areas for Selective Enhancement

### 1. Evaluation Metrics (`anno/src/eval/metrics.rs`)

**Current:** Mentions P/R/F1 but no formulas
**Enhancement:** Add concise formulas where metrics are defined:
- `Precision = TP / (TP + FP)`
- `Recall = TP / (TP + FN)`
- `F1 = 2 × (P × R) / (P + R)`

**Rationale:** Core evaluation metrics, users need to understand what they mean.

### 2. Clustering Algorithms (`strata/src/leiden.rs`)

**Current:** Has modularity formula, good as-is
**Enhancement:** Add one-sentence intuition about resolution parameter if missing

**Rationale:** Algorithm is complex, but current explanation level is appropriate.

### 3. Confidence Calibration (`anno/src/eval/calibration.rs`)

**Current:** Lists metrics but no formulas
**Enhancement:** Add formulas for ECE and Brier where they're computed

**Rationale:** These are less well-known metrics, formulas help interpretation.

### 4. Similarity Metrics (`coalesce/src/resolver.rs`)

**Current:** Basic description (sufficient)
**Enhancement:** None needed - well-known concept, current level is appropriate

## Implementation Approach

**Selective enhancement only:**
- Add formulas where metrics are computed (not everywhere they're used)
- Add brief intuition only for non-obvious concepts
- Keep explanations concise and technical, not pedagogical

## Example: Appropriate Level of Detail

**For F1 score (well-known metric):**
```rust
/// Calculate F1 score: `F1 = 2 × (P × R) / (P + R)`
///
/// Harmonic mean of precision and recall. Returns 0.0 if both are 0.
```

**For ECE (less well-known metric):**
```rust
/// Expected Calibration Error (ECE).
///
/// Formula: `ECE = Σ(n_i / N) × |acc_i - conf_i|`
///
/// Where bins are confidence intervals [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0],
/// `n_i` is count in bin i, `acc_i` is accuracy in bin i, `conf_i` is mean confidence.
/// Lower is better (0 = perfectly calibrated).
```

**For simple utilities:**
```rust
/// Compute cosine similarity between embeddings.
///
/// Returns normalized value in [0.0, 1.0].
```

## References

- [math-in-rust-doc](https://docs.rs/math-in-rust-doc): KaTeX integration example
- [Rustdoc documentation](https://doc.rust-lang.org/rustdoc/): Official rustdoc guide
- Martin Gardner's style: Clear, concrete, step-by-step, puzzle-like

