# Box Embeddings Usage Guide

## Quick Start

### Basic Coreference Resolution

```rust
use anno::backends::box_embeddings::{BoxCorefConfig, BoxEmbedding, BoxCorefResolver};
use anno::eval::coref_resolver::CoreferenceResolver;
use anno::{Entity, EntityType};

// Entities to resolve
let entities = vec![
    Entity::new("John Smith", EntityType::Person, 0, 10, 0.9),
    Entity::new("he", EntityType::Person, 50, 52, 0.8),
];

// Create box embeddings (in practice, learned from data)
let boxes = vec![
    BoxEmbedding::new(vec![0.0, 0.0], vec![1.0, 1.0]),      // John Smith
    BoxEmbedding::new(vec![0.1, 0.1], vec![0.9, 0.9]),      // he (overlaps)
];

// Resolve
let config = BoxCorefConfig::default();
let resolver = BoxCorefResolver::new(config);
let resolved = resolver.resolve_with_boxes(&entities, &boxes);

// Check coreference
assert_eq!(resolved[0].canonical_id, resolved[1].canonical_id);
```

### Converting Vector Embeddings to Boxes

```rust
use anno::eval::coref_resolver::vectors_to_boxes;
use anno::backends::box_embeddings::BoxEmbedding;

// Vector embeddings from a text encoder
let embeddings = vec![
    0.1, 0.2, 0.3,  // Entity 0
    0.15, 0.25, 0.35,  // Entity 1
];
let hidden_dim = 3;

// Convert to boxes with fixed radius
let boxes = vectors_to_boxes(&embeddings, hidden_dim, Some(0.1));

// Or use adaptive radius (proportional to vector magnitude)
let boxes_adaptive = vectors_to_boxes(&embeddings, hidden_dim, None);
```

### Temporal Boxes (Time-Varying Entities)

```rust
use anno::backends::box_embeddings::{TemporalBox, BoxVelocity, BoxEmbedding};

// "The President" in 2012 (Obama)
let obama_base = BoxEmbedding::new(vec![0.0, 0.0], vec![1.0, 1.0]);
let velocity = BoxVelocity::static_velocity(2);
let obama_presidency = TemporalBox::new(obama_base, velocity, (2012.0, 2016.0));

// "The President" in 2017 (Trump) - different box
let trump_base = BoxEmbedding::new(vec![5.0, 5.0], vec![6.0, 6.0]);
let trump_presidency = TemporalBox::new(trump_base, velocity, (2017.0, 2021.0));

// Check coreference at specific times
let score_2015 = obama_presidency.coreference_at_time(&trump_presidency, 2015.0);
assert_eq!(score_2015, 0.0); // Should not corefer (different times)
```

### Uncertainty-Aware Boxes (Misinformation Detection)

```rust
use anno::backends::box_embeddings::{UncertainBox, BoxEmbedding};

// High-confidence claim: "Trump is in NY" (small, precise box)
let claim_a = UncertainBox::new(
    BoxEmbedding::new(vec![0.0, 0.0], vec![0.1, 0.1]), // Small = high confidence
    0.95, // Source trust
);

// Contradictory claim: "Trump is in FL" (disjoint, high confidence)
let claim_b = UncertainBox::new(
    BoxEmbedding::new(vec![5.0, 5.0], vec![5.1, 5.1]), // Disjoint
    0.90,
);

// Detect conflict
if let Some(conflict) = claim_a.detect_conflict(&claim_b) {
    println!("Conflict detected! Severity: {:.3}", conflict.severity);
}
```

### Gumbel Boxes (Noise Robustness)

```rust
use anno::backends::box_embeddings::{GumbelBox, BoxEmbedding};

let mean_box = BoxEmbedding::new(vec![0.0, 0.0], vec![1.0, 1.0]);
let gumbel = GumbelBox::new(mean_box, 0.1); // Temperature = 0.1 (sharp)

// Membership is probabilistic, not binary
let point = vec![0.5, 0.5];
let prob = gumbel.membership_probability(&point);
assert!(prob > 0.5); // High probability inside box

// Robust coreference tolerates slight misalignments
let box2 = BoxEmbedding::new(vec![0.05, 0.05], vec![0.95, 0.95]);
let gumbel2 = GumbelBox::new(box2, 0.1);
let score = gumbel.robust_coreference(&gumbel2, 100);
assert!(score > 0.3);
```

### Interaction Modeling

```rust
use anno::backends::box_embeddings::{interaction_strength, acquisition_roles, BoxEmbedding};

// Actor-action-target triple
let buyer = BoxEmbedding::new(vec![0.0, 0.0], vec![1.0, 1.0]);
let seller = BoxEmbedding::new(vec![0.5, 0.5], vec![1.5, 1.5]);
let acquisition = BoxEmbedding::new(vec![0.2, 0.2], vec![0.8, 0.8]);

// Compute interaction strength
let strength = interaction_strength(&buyer, &acquisition, &seller);

// Determine roles (asymmetric)
let (buyer_role, seller_role) = acquisition_roles(&buyer, &seller, &acquisition);
```

## Integration with Existing Code

### Using with Identity Type

```rust
use anno::grounded::{Identity, IdentityId};
use anno::backends::box_embeddings::BoxEmbedding;

let mut identity = Identity::new(0, "Marie Curie");
identity.box_embedding = Some(BoxEmbedding::new(
    vec![0.0, 0.0],
    vec![1.0, 1.0],
));
```

### Combining with Vector Embeddings

Box embeddings can be used alongside vector embeddings:

- **Vectors**: Fast semantic similarity (cosine similarity)
- **Boxes**: Logical constraints (transitivity, syntactic rules)

Hybrid approach: Use vectors for initial filtering, boxes for final resolution.

## Performance Considerations

- **Box operations** are more expensive than cosine similarity (O(d) vs O(d) but with more operations)
- **Temporal boxes** require time-slice computation (cache `at_time()` results)
- **Gumbel boxes** use grid sampling (adjust sample count for speed vs accuracy)

## Research References

- **Box Embeddings**: Vilnis et al. (2018) - Probabilistic embedding of knowledge graphs
- **BERE**: Lee et al. (2022) - Event-event relation extraction
- **BoxTE**: Messner et al. (2022) - Temporal knowledge graphs
- **UKGE**: Chen et al. (2021) - Uncertainty-aware embeddings

