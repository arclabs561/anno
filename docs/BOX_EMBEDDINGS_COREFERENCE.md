# Box Embeddings for Coreference Resolution

## Overview

This document explores how **box embeddings** and related geometric representations can encode the logical invariants of coreference resolution, addressing the limitations of current vector-based approaches.

## Current State

The codebase currently uses:
- **Vector embeddings** (cosine similarity) in `resolve_coreferences()` 
- **Rule-based** coreference in `SimpleCorefResolver`
- **Union-find** clustering with similarity thresholds

**Limitations:**
- Transitivity is enforced via union-find, but similarity scores don't guarantee logical consistency
- No explicit encoding of syntactic constraints (Principle B/C)
- No temporal modeling (entities change over time)
- No uncertainty quantification (misinformation/noise)

## Logical Invariants → Box Geometry

### 1. Equivalence Relations (Transitivity)

**Current**: Union-find enforces transitivity, but similarity scores can be inconsistent.

**Box Embedding Solution**: Coreference as high mutual overlap.

```rust
// Box representation
struct Box {
    min: Vec<f32>,  // Lower bound in each dimension
    max: Vec<f32>,  // Upper bound in each dimension
}

// Coreference = high mutual conditional probability
fn coreference_probability(box_a: &Box, box_b: &Box) -> f32 {
    let vol_intersection = intersection_volume(box_a, box_b);
    let vol_a = volume(box_a);
    let vol_b = volume(box_b);
    
    // P(A|B) and P(B|A) both high → coreference
    let p_a_given_b = vol_intersection / vol_b;
    let p_b_given_a = vol_intersection / vol_a;
    
    // Coreference requires mutual high overlap
    (p_a_given_b + p_b_given_a) / 2.0
}
```

**Invariant**: If `coreference_probability(A, B) ≥ δ` and `coreference_probability(B, C) ≥ δ`, then by box containment transitivity, `coreference_probability(A, C) ≥ δ'` where `δ'` is guaranteed to be high (though not necessarily ≥ δ).

### 2. Syntactic Constraints (Principle B/C)

**Current**: No explicit syntactic constraints.

**Box Embedding Solution**: Enforce disjointness via box separation.

```rust
// Principle B: Pronominal must be free in local domain
fn enforce_principle_b(
    pronoun_box: &Box,
    local_entities: &[Box],
    threshold: f32,
) -> bool {
    for entity_box in local_entities {
        // Pronoun cannot corefer with local entity (unless reflexive)
        let overlap = intersection_volume(pronoun_box, entity_box);
        let vol_pronoun = volume(pronoun_box);
        let p_pronoun_given_entity = overlap / vol_pronoun;
        
        if p_pronoun_given_entity > threshold {
            return false; // Violates Principle B
        }
    }
    true
}

// Principle C: R-expression must be free everywhere
fn enforce_principle_c(
    rexpression_box: &Box,
    ccommanding_entities: &[Box],
    threshold: f32,
) -> bool {
    for entity_box in ccommanding_entities {
        let overlap = intersection_volume(rexpression_box, entity_box);
        if overlap > threshold {
            return false; // Violates Principle C
        }
    }
    true
}
```

### 3. Temporal Invariants (Near-Identity)

**Current**: No temporal modeling.

**Box Embedding Solution**: Temporal boxes (BoxTE-style).

```rust
// Temporal box: box that evolves over time
struct TemporalBox {
    // Base box at time t=0
    base: Box,
    // Velocity: how box moves/resizes per time unit
    velocity: BoxVelocity,
    // Time range where this box is valid
    time_range: (f64, f64),
}

struct BoxVelocity {
    min_delta: Vec<f32>,  // Change in min per time unit
    max_delta: Vec<f32>,  // Change in max per time unit
}

// Coreference at time t
fn temporal_coreference(
    box_a: &TemporalBox,
    box_b: &TemporalBox,
    time: f64,
) -> f32 {
    // Get boxes at time t
    let box_a_t = box_a.at_time(time);
    let box_b_t = box_b.at_time(time);
    
    // Only corefer if both boxes are valid at time t
    if !box_a_t.is_valid_at(time) || !box_b_t.is_valid_at(time) {
        return 0.0;
    }
    
    coreference_probability(&box_a_t, &box_b_t)
}
```

**Example**: "The President (Obama)" and "The President (Bush)" are disjoint boxes at different times, preventing false coreference.

### 4. Interaction Constraints

**Current**: No explicit interaction modeling.

**Box Embedding Solution**: Conditional probability encodes asymmetric relations.

```rust
// Interaction = conditional probability
fn interaction_strength(
    actor_box: &Box,
    action_box: &Box,
    target_box: &Box,
) -> f32 {
    // Interaction volume: where all three boxes overlap
    let interaction_vol = triple_intersection_volume(
        actor_box, action_box, target_box
    );
    
    // P(action, target | actor) = how much of actor's space contains the interaction
    let vol_actor = volume(actor_box);
    interaction_vol / vol_actor
}

// Asymmetry by design: P(A|B) ≠ P(B|A) for interactions
fn acquisition_relation(
    buyer_box: &Box,
    seller_box: &Box,
    company_box: &Box,
) -> (f32, f32) {
    let acquisition_box = create_relation_box("acquired");
    
    let buyer_role = interaction_strength(buyer_box, &acquisition_box, company_box);
    let seller_role = interaction_strength(seller_box, &acquisition_box, company_box);
    
    // Asymmetric: buyer and seller have different conditional probabilities
    (buyer_role, seller_role)
}
```

### 5. Misinformation & Uncertainty

**Current**: No uncertainty quantification.

**Box Embedding Solution**: Box volume = confidence (UKGE-style).

```rust
// Uncertainty-aware box
struct UncertainBox {
    box: Box,
    // Volume = confidence: small box = high confidence, large box = low confidence
    confidence: f32,  // Derived from volume
    source_trust: f32,  // Trust in source
}

// Conflict detection: disjoint boxes with high confidence = contradiction
fn detect_conflict(
    claim_a: &UncertainBox,
    claim_b: &UncertainBox,
) -> Option<Conflict> {
    let overlap = intersection_volume(&claim_a.box, &claim_b.box);
    let min_vol = volume(&claim_a.box).min(volume(&claim_b.box));
    
    // If both are high-confidence (small volume) but disjoint, conflict
    if overlap < min_vol * 0.1 && 
       claim_a.confidence > 0.8 && 
       claim_b.confidence > 0.8 {
        Some(Conflict {
            claim_a: claim_a.source_trust,
            claim_b: claim_b.source_trust,
            severity: (1.0 - overlap / min_vol) * 
                      (claim_a.confidence + claim_b.confidence) / 2.0,
        })
    } else {
        None
    }
}
```

### 6. Noise Robustness

**Current**: Similarity thresholds are brittle.

**Box Embedding Solution**: Gumbel-soft boxes with fuzzy boundaries.

```rust
// Gumbel box: soft boundaries via Gumbel distribution
struct GumbelBox {
    // Mean box boundaries
    mean_min: Vec<f32>,
    mean_max: Vec<f32>,
    // Temperature: controls fuzziness (higher = more fuzzy)
    temperature: f32,
}

// Membership probability (not binary)
fn membership_probability(
    point: &[f32],
    gumbel_box: &GumbelBox,
) -> f32 {
    // For each dimension, compute Gumbel CDF
    let mut prob = 1.0;
    for (i, &coord) in point.iter().enumerate() {
        let min_prob = gumbel_cdf(
            coord,
            gumbel_box.mean_min[i],
            gumbel_box.temperature,
        );
        let max_prob = 1.0 - gumbel_cdf(
            coord,
            gumbel_box.mean_max[i],
            gumbel_box.temperature,
        );
        prob *= min_prob * max_prob;
    }
    prob
}

// Robust coreference: tolerate slight misalignments
fn robust_coreference(
    box_a: &GumbelBox,
    box_b: &GumbelBox,
    samples: usize,
) -> f32 {
    // Sample points from box_a, check membership in box_b
    let mut total_prob = 0.0;
    for _ in 0..samples {
        let point = sample_from_gumbel_box(box_a);
        total_prob += membership_probability(&point, box_b);
    }
    total_prob / samples as f32
}
```

## Implementation Plan

### Phase 1: Core Box Infrastructure ✅

1. **Add `BoxEmbedding` type** to `src/backends/box_embeddings.rs`:
   ```rust
   pub struct BoxEmbedding {
       min: Vec<f32>,
       max: Vec<f32>,
       confidence: f32,  // Derived from volume
   }
   ```

2. **Box operations** (implemented in `BoxEmbedding`):
   - `intersection_volume()` ✅
   - `conditional_probability()` ✅
   - `coreference_score()` ✅

### Phase 2: Box-Based Coreference Resolver ✅

1. **Create `BoxCorefResolver`** in `src/eval/coref_resolver.rs`:
   ```rust
   pub struct BoxCorefResolver {
       config: BoxCorefConfig,
   }
   
   impl CoreferenceResolver for BoxCorefResolver {
       fn resolve(&self, entities: &[Entity]) -> Vec<Entity> {
           // 1. Convert entities to boxes (if not already)
           // 2. Compute pairwise conditional probabilities
           // 3. Cluster via transitive closure (box containment)
           // 4. Enforce syntactic constraints (Principle B/C)
       }
   }
   ```

2. **Integration with existing resolvers**:
   - `BoxCorefResolver` can be used alongside `SimpleCorefResolver`
   - Hybrid approach: use boxes for uncertain cases, rules for clear cases

### Phase 3: Temporal & Interaction Support

1. **Temporal boxes** (`src/box_temporal.rs`):
   - `TemporalBox` type
   - `at_time()` method
   - Integration with `GroundedDocument` for time-aware coreference

2. **Interaction modeling** (`src/box_interaction.rs`):
   - Relation boxes (e.g., "acquired", "founded")
   - Triple intersection for actor-action-target
   - Asymmetric role detection

### Phase 4: Uncertainty & Noise

1. **Uncertainty-aware boxes** (`src/box_uncertainty.rs`):
   - `UncertainBox` with source trust
   - Conflict detection
   - Alignment with fact-checking systems

2. **Gumbel boxes** (`src/box_gumbel.rs`):
   - Soft boundaries for noise tolerance
   - Probabilistic membership
   - Robust coreference scoring

## Research Questions

1. **Box dimension**: What's the optimal embedding dimension? (Literature suggests 50-200)
2. **Training**: How to learn box embeddings from coreference annotations?
3. **Hybrid**: When to use boxes vs. vectors? (Boxes for logical constraints, vectors for semantic similarity)
4. **Performance**: Box operations are more expensive than cosine similarity. Can we optimize?

## References

- **Box Embeddings**: Vilnis et al. (2018), "Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures"
- **BERE**: Lee et al. (2022), "Box Embeddings for Event-Event Relation Extraction"
- **BoxTE**: Messner et al. (2022), "Temporal Knowledge Graph Completion with Box Embeddings"
- **UKGE**: Chen et al. (2021), "Uncertainty-Aware Knowledge Graph Embeddings"

## Implementation Status

### ✅ Completed

1. **Core Box Infrastructure** (`src/backends/box_embeddings.rs`):
   - `BoxEmbedding` type with volume, intersection, conditional probability
   - `BoxCorefConfig` for configuration
   - All box operations implemented and tested

2. **Box-Based Coreference Resolver** (`src/eval/coref_resolver.rs`):
   - `BoxCorefResolver` implementing `CoreferenceResolver` trait
   - Syntactic constraint enforcement (Principle B/C)
   - Integration with existing resolver infrastructure

3. **Temporal Boxes** (BoxTE-style):
   - `TemporalBox` and `BoxVelocity` types
   - `at_time()` method for time-slice operations
   - Coreference at specific times

4. **Uncertainty-Aware Boxes** (UKGE-style):
   - `UncertainBox` with confidence derived from volume
   - `Conflict` detection for contradictory claims
   - Source trust modeling

5. **Gumbel Boxes** (Noise Robustness):
   - `GumbelBox` with soft, probabilistic boundaries
   - `membership_probability()` for fuzzy membership
   - `robust_coreference()` for noise-tolerant scoring

6. **Identity Integration** (`src/grounded.rs`):
   - `Identity.box_embedding` field added
   - Serialization support

### 🔄 Next Steps (Research & Evaluation)

1. **Learning Box Embeddings**: How to learn box parameters from coreference annotations?
2. **Evaluation**: Compare box-based vs. vector-based coreference on CoNLL-2012
3. **Hybrid Approach**: When to use boxes vs. vectors? (Boxes for constraints, vectors for semantics)
4. **Performance Optimization**: Box operations are more expensive than cosine similarity
5. **Temporal Integration**: Full BoxTE-style training for time-varying entities
6. **Interaction Modeling**: Triple intersection for actor-action-target relations

