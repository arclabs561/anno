# Box Embedding Training: Mathematical Details

## Overview

This document describes the complete mathematical framework for training box embeddings for coreference resolution, based on research from BERE (2022), BoxTE (2022), and UKGE (2021).

## Mathematical Foundation

### 1. Box Parameterization

We use **reparameterization** to ensure boxes are always valid (min ≤ max):

```
For each dimension i:
  μᵢ = center position (learnable)
  δᵢ = log(width) (learnable)
  
  min[i] = μᵢ - exp(δᵢ) / 2
  max[i] = μᵢ + exp(δᵢ) / 2
```

This guarantees `min[i] ≤ max[i]` for all dimensions without constraints.

### 2. Loss Function

#### Positive Pairs (Should Corefer)

For entities `i` and `j` that corefer:

```
L_pos(i,j) = -log(P(box_i | box_j)) - log(P(box_j | box_i))
            + λ_reg * (Vol(box_i) + Vol(box_j))
```

Where:
- `P(box_i | box_j) = Vol(box_i ∩ box_j) / Vol(box_j)` (conditional probability)
- `λ_reg` = regularization weight (prevents boxes from growing too large)

**Intuition**: Maximize mutual conditional probability (high overlap).

#### Negative Pairs (Shouldn't Corefer)

For entities `i` and `j` that don't corefer:

```
L_neg(i,j) = max(0, margin - P(box_i | box_j)) * λ_neg
```

Where:
- `margin` = minimum separation threshold (e.g., 0.3)
- `λ_neg` = negative sampling weight

**Intuition**: Enforce separation (low overlap).

#### Total Loss

```
L = (1/|P|) * Σ_{(i,j) ∈ P} L_pos(i,j) 
  + (1/|N|) * Σ_{(i,j) ∈ N} L_neg(i,j)
```

Where:
- `P` = set of positive pairs (same chain)
- `N` = set of negative pairs (different chains)

### 3. Gradient Computation

We use **finite differences** for gradient computation (simple but slow):

```
∂L/∂μᵢ ≈ (L(μᵢ + ε) - L(μᵢ)) / ε
∂L/∂δᵢ ≈ (L(δᵢ + ε) - L(δᵢ)) / ε
```

Where `ε = 1e-5` (small perturbation).

**Note**: For production, use automatic differentiation (e.g., via a Rust autograd library).

### 4. Optimization

**Stochastic Gradient Descent (SGD)**:

```
μᵢ ← μᵢ - lr * ∂L/∂μᵢ
δᵢ ← δᵢ - lr * ∂L/∂δᵢ
```

Where `lr` = learning rate (e.g., 0.01-0.1).

## Implementation Details

### Training Loop

1. **Initialize boxes**:
   - From vector embeddings (if available): `BoxEmbedding::from_vector()`
   - Random: center in [-1, 1], width = 0.2

2. **For each epoch**:
   - For each training example:
     - Build positive pairs (entities in same chain)
     - Build negative pairs (entities in different chains)
     - Compute gradients for all pairs
     - Accumulate gradients
     - Update boxes

3. **Evaluation**:
   - Convert trainable boxes to `BoxEmbedding`
   - Use `BoxCorefResolver` for inference

### Performance Considerations

**Finite differences are slow**:
- For each parameter, requires 2 forward passes (perturb + compute loss)
- For a box with dimension `d`, each pair requires `4d` forward passes
- **Solution**: Use automatic differentiation for production

**Optimization strategies**:
- Mini-batch training (process multiple examples at once)
- Gradient accumulation (batch gradients before update)
- Learning rate scheduling (reduce LR over time)

## Example Usage

```rust
use anno::backends::box_embeddings_training::{BoxEmbeddingTrainer, TrainingConfig, TrainingExample};
use anno::eval::coref::{CorefChain, Mention};
use anno::{Entity, EntityType};

// Create training data
let example = TrainingExample {
    entities: vec![
        Entity::new("John", EntityType::Person, 0, 4, 0.9),
        Entity::new("he", EntityType::Person, 20, 22, 0.8),
    ],
    chains: vec![CorefChain::new(vec![
        Mention::new("John", 0, 4),
        Mention::new("he", 20, 22),
    ])],
};

// Configure training
let config = TrainingConfig {
    learning_rate: 0.01,
    negative_weight: 1.0,
    margin: 0.3,
    regularization: 0.001,
    epochs: 100,
    batch_size: 32,
};

// Train
let dim = 50;
let mut trainer = BoxEmbeddingTrainer::new(config, dim, None);
trainer.initialize_boxes(&[example.clone()], None);
let losses = trainer.train(&[example]);

// Get trained boxes
let trained_boxes = trainer.get_boxes();
```

## Research References

1. **BERE (2022)**: "Box Embeddings for Event-Event Relation Extraction"
   - Conditional probability for asymmetric relations
   - Margin-based loss for negative pairs

2. **BoxTE (2022)**: "Temporal Knowledge Graph Completion with Box Embeddings"
   - Temporal box evolution
   - Time-slice training

3. **UKGE (2021)**: "Uncertainty-Aware Knowledge Graph Embeddings"
   - Volume = confidence
   - Conflict detection

## Future Improvements

1. **Automatic Differentiation**: Replace finite differences with autograd
2. **Mini-batch Training**: Process multiple examples simultaneously
3. **Learning Rate Scheduling**: Adaptive LR (e.g., cosine annealing)
4. **Initialization**: Better initialization from pre-trained vectors
5. **Regularization**: L2 on box centers, L1 on box sizes
6. **Early Stopping**: Stop when validation loss plateaus

