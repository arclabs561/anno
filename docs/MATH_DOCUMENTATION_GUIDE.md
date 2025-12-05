# Math Documentation Guide: Gardner-Style Explanations

## Current State Assessment

### What We Have
- ✅ Modularity formula in `strata/src/leiden.rs` (ASCII math)
- ✅ Box embedding formulas in `anno/src/backends/box_embeddings_training.rs` (ASCII)
- ✅ Late interaction scoring in `anno/src/backends/inference.rs` (ASCII)
- ✅ Some similarity explanations (basic)

### What's Missing (Gardner-Style)
- ❌ Step-by-step derivations with intuition
- ❌ Concrete numerical examples
- ❌ Visual diagrams for formulas
- ❌ "Why this works" explanations
- ❌ Puzzle-like examples that build understanding

## Best Practices for Math in Rust Docs

### 1. Markup Options

**Recommended: ASCII Math + KaTeX (for docs.rs)**

```rust
/// # The Formula
///
/// ```text
/// similarity(a, b) = (a · b) / (||a|| × ||b||)
/// ```
///
/// Where:
/// - `a · b` = dot product (sum of element-wise products)
/// - `||a||` = L2 norm (Euclidean length)
///
/// For docs.rs rendering, we can add KaTeX:
/// \[ \text{similarity}(a, b) = \frac{a \cdot b}{\|a\| \times \|b\|} \]
```

**Why ASCII + KaTeX?**
- ASCII works everywhere (offline docs, GitHub)
- KaTeX enhances docs.rs with proper rendering
- No external dependencies for basic viewing
- Progressive enhancement approach

### 2. Gardner-Style Structure

**Template:**
```rust
/// # The Problem
///
/// [Concrete example that motivates the formula]
///
/// # The Intuition
///
/// [Why this formula makes sense, step by step]
///
/// # The Formula
///
/// ```text
/// [ASCII version]
/// ```
///
/// # Worked Example
///
/// [Concrete numbers showing each step]
///
/// # Why It Works
///
/// [Connection to underlying principles]
```

## Areas Needing Improvement

### 1. Similarity Metrics (`coalesce/src/resolver.rs`)

**Current:** Basic description
**Needed:** Gardner-style explanation with:
- Visual diagram of cosine similarity
- Step-by-step calculation example
- Intuition: "Why cosine measures angle, not magnitude"

### 2. Evaluation Metrics (`anno/src/eval/metrics.rs`)

**Current:** Mentions P/R/F1 but no formulas
**Needed:**
- Precision = TP / (TP + FP) with concrete example
- Recall = TP / (TP + FN) with concrete example
- F1 = 2 × (P × R) / (P + R) with harmonic mean intuition

### 3. Clustering Algorithms (`strata/src/leiden.rs`)

**Current:** Has modularity formula but lacks intuition
**Needed:**
- "What does modularity measure?" (community quality)
- Step-by-step calculation example
- Why resolution parameter matters

### 4. Confidence Calibration (`anno/src/eval/calibration.rs`)

**Current:** Lists metrics but no formulas
**Needed:**
- ECE formula with binning explanation
- Brier score with probability interpretation
- Concrete example of well-calibrated vs poorly-calibrated

## Implementation Plan

### Phase 1: Add Formulas (ASCII)
- Add P/R/F1 formulas to `metrics.rs`
- Add cosine similarity formula to `resolver.rs`
- Add ECE/Brier formulas to `calibration.rs`

### Phase 2: Add Gardner-Style Explanations
- Worked examples with concrete numbers
- Step-by-step derivations
- Intuition sections

### Phase 3: Add KaTeX (Optional)
- Create `katex.html` header
- Add to `Cargo.toml` metadata for docs.rs
- Enhance formulas with LaTeX rendering

## Example: Gardner-Style F1 Explanation

```rust
/// Calculate F1 score (harmonic mean of precision and recall).
///
/// # The Problem
///
/// Imagine you're evaluating a NER system. You predicted 10 entities,
/// and 8 were correct. But there were actually 12 entities in the text.
///
/// - Precision: 8/10 = 0.8 (80% of predictions were correct)
/// - Recall: 8/12 = 0.67 (67% of actual entities found)
///
/// Which is more important? Both! F1 balances them.
///
/// # The Formula
///
/// ```text
/// F1 = 2 × (Precision × Recall) / (Precision + Recall)
/// ```
///
/// # Why Harmonic Mean?
///
/// The harmonic mean punishes extreme imbalances. If precision is 1.0
/// but recall is 0.1, the arithmetic mean is 0.55, but F1 is only 0.18.
/// This reflects reality: a system that finds only 10% of entities
/// isn't very useful, even if it's always right when it guesses.
///
/// # Worked Example
///
/// ```text
/// Precision = 0.8, Recall = 0.67
/// F1 = 2 × (0.8 × 0.67) / (0.8 + 0.67)
///    = 2 × 0.536 / 1.47
///    = 1.072 / 1.47
///    = 0.729
/// ```
///
/// # Intuition
///
/// F1 is always between precision and recall, closer to the smaller value.
/// This makes it a conservative metric: you need both high precision
/// AND high recall to get a good F1 score.
```

## References

- [math-in-rust-doc](https://docs.rs/math-in-rust-doc): KaTeX integration example
- [Rustdoc documentation](https://doc.rust-lang.org/rustdoc/): Official rustdoc guide
- Martin Gardner's style: Clear, concrete, step-by-step, puzzle-like

