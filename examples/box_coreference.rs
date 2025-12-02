//! Example: Box embeddings for coreference resolution.
//!
//! This example demonstrates how to use box embeddings to resolve coreference
//! with explicit encoding of logical invariants.

use anno::backends::box_embeddings::{BoxCorefConfig, BoxEmbedding};
use anno::eval::coref_resolver::BoxCorefResolver;
use anno::{Entity, EntityType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example: "Marie Curie won the Nobel Prize. She was a physicist."
    let entities = vec![
        Entity::new("Marie Curie", EntityType::Person, 0, 11, 0.95),
        Entity::new("She", EntityType::Person, 35, 38, 0.85),
    ];

    // Create box embeddings (in practice, these would be learned from data)
    // "Marie Curie" and "She" should have high overlap
    let boxes = vec![
        BoxEmbedding::new(vec![0.0, 0.0], vec![1.0, 1.0]), // Marie Curie
        BoxEmbedding::new(vec![0.1, 0.1], vec![0.9, 0.9]), // She (overlaps)
    ];

    // Configure resolver
    let config = BoxCorefConfig {
        coreference_threshold: 0.6,
        enforce_syntactic_constraints: true,
        max_local_distance: 5,
        vector_to_box_radius: Some(0.1),
    };

    let resolver = BoxCorefResolver::new(config);
    let resolved = resolver.resolve_with_boxes(&entities, &boxes);

    // Check that they corefer
    assert_eq!(
        resolved[0].canonical_id, resolved[1].canonical_id,
        "Marie Curie and She should corefer"
    );

    println!("✓ Coreference resolved successfully!");
    println!(
        "  Entity 0: {} -> cluster {:?}",
        entities[0].text, resolved[0].canonical_id
    );
    println!(
        "  Entity 1: {} -> cluster {:?}",
        entities[1].text, resolved[1].canonical_id
    );

    // Example: Temporal boxes (preventing false coreference)
    use anno::backends::box_embeddings::{BoxVelocity, TemporalBox};

    let obama_base = BoxEmbedding::new(vec![0.0, 0.0], vec![1.0, 1.0]);
    let trump_base = BoxEmbedding::new(vec![5.0, 5.0], vec![6.0, 6.0]);
    let velocity = BoxVelocity::static_velocity(2);

    let obama_presidency = TemporalBox::new(obama_base, velocity.clone(), (2012.0, 2016.0));
    let trump_presidency = TemporalBox::new(trump_base, velocity, (2017.0, 2021.0));

    // Should not corefer at different times
    let score_2015 = obama_presidency.coreference_at_time(&trump_presidency, 2015.0);
    let score_2018 = obama_presidency.coreference_at_time(&trump_presidency, 2018.0);

    println!("\n✓ Temporal boxes prevent false coreference:");
    println!("  Score at 2015 (Obama's term): {:.3}", score_2015);
    println!("  Score at 2018 (Trump's term): {:.3}", score_2018);
    assert_eq!(score_2015, 0.0, "Should not corefer at different times");
    assert_eq!(score_2018, 0.0, "Should not corefer at different times");

    // Example: Uncertainty-aware boxes (conflict detection)
    use anno::backends::box_embeddings::UncertainBox;

    let claim_a = UncertainBox::new(
        BoxEmbedding::new(vec![0.0, 0.0], vec![0.1, 0.1]), // Small = high confidence
        0.95,                                              // High source trust
    );
    let claim_b = UncertainBox::new(
        BoxEmbedding::new(vec![5.0, 5.0], vec![5.1, 5.1]), // Disjoint, high confidence
        0.90,
    );

    if let Some(conflict) = claim_a.detect_conflict(&claim_b) {
        println!("\n✓ Conflict detected:");
        println!("  Severity: {:.3}", conflict.severity);
        println!("  Claim A trust: {:.2}", conflict.claim_a_trust);
        println!("  Claim B trust: {:.2}", conflict.claim_b_trust);
    }

    // Example: Converting vector embeddings to boxes
    let vector_embedding = vec![0.5, 0.5, 0.5];
    let box_from_vector = BoxEmbedding::from_vector(&vector_embedding, 0.1);
    println!("\n✓ Vector to box conversion:");
    println!("  Vector: {:?}", vector_embedding);
    println!("  Box center: {:?}", box_from_vector.center());
    println!("  Box volume: {:.6}", box_from_vector.volume());

    Ok(())
}
