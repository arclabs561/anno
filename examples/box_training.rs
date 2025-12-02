//! Example: Training box embeddings for coreference resolution.
//!
//! This demonstrates the complete training pipeline:
//! 1. Create synthetic coreference data
//! 2. Initialize trainable boxes
//! 3. Train using gradient descent
//! 4. Evaluate on test set

use anno::backends::box_embeddings::{BoxCorefConfig, BoxEmbedding};
use anno::backends::box_embeddings_training::{
    BoxEmbeddingTrainer, TrainingConfig, TrainingExample,
};
use anno::eval::coref::{CorefChain, Mention};
use anno::eval::coref_resolver::BoxCorefResolver;
use anno::{Entity, EntityType};

fn create_synthetic_data() -> Vec<TrainingExample> {
    // Example 1: "John went to the store. He bought milk."
    let ex1 = TrainingExample {
        entities: vec![
            Entity::new("John", EntityType::Person, 0, 4, 0.9),
            Entity::new("He", EntityType::Person, 25, 27, 0.8),
        ],
        chains: vec![CorefChain::new(vec![
            Mention::new("John", 0, 4),
            Mention::new("He", 25, 27),
        ])],
    };

    // Example 2: "Mary and John met. They went to the park. She was happy."
    let ex2 = TrainingExample {
        entities: vec![
            Entity::new("Mary", EntityType::Person, 0, 4, 0.9),
            Entity::new("John", EntityType::Person, 9, 13, 0.9),
            Entity::new("They", EntityType::Person, 19, 23, 0.8),
            Entity::new("She", EntityType::Person, 40, 43, 0.8),
        ],
        chains: vec![
            CorefChain::new(vec![
                Mention::new("Mary", 0, 4),
                Mention::new("She", 40, 43),
            ]),
            CorefChain::new(vec![
                Mention::new("John", 9, 13),
                Mention::new("They", 19, 23), // Note: "They" refers to both, simplified
            ]),
        ],
    };

    // Example 3: "Apple Inc. announced earnings. The company reported growth."
    let ex3 = TrainingExample {
        entities: vec![
            Entity::new("Apple Inc.", EntityType::Organization, 0, 10, 0.95),
            Entity::new("The company", EntityType::Organization, 32, 43, 0.85),
        ],
        chains: vec![CorefChain::new(vec![
            Mention::new("Apple Inc.", 0, 10),
            Mention::new("The company", 32, 43),
        ])],
    };

    vec![ex1, ex2, ex3]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Box Embedding Training Example ===\n");

    // Create synthetic training data
    let training_data = create_synthetic_data();
    println!("Created {} training examples", training_data.len());

    // Training configuration
    let config = TrainingConfig {
        learning_rate: 0.001, // BERE standard (with warmup)
        negative_weight: 0.5, // Less weight on negatives
        margin: 0.3,
        regularization: 0.0001, // L2 regularization
        epochs: 100,            // More epochs
        batch_size: 2,          // Small batch for small dataset
        warmup_epochs: 10,      // 10% of epochs
        use_self_adversarial: true,
        adversarial_temperature: 1.0,
        early_stopping_patience: Some(15), // Stop if no improvement
        early_stopping_min_delta: 0.001,
    };

    // Initialize trainer
    let dim = 10; // Small dimension for quick training
    let mut trainer = BoxEmbeddingTrainer::new(config.clone(), dim, None);

    // Initialize boxes from entities
    trainer.initialize_boxes(&training_data, None);

    println!("\nInitialized {} boxes", trainer.get_boxes().len());

    // Train
    println!("\nTraining...");
    let losses = trainer.train(&training_data);

    println!("\nTraining complete!");
    println!("Initial loss: {:.4}", losses[0]);
    println!("Final loss: {:.4}", losses[losses.len() - 1]);
    println!(
        "Loss reduction: {:.1}%",
        (1.0 - losses[losses.len() - 1] / losses[0]) * 100.0
    );

    // Evaluate on training set with different thresholds
    println!("\n=== Evaluation ===");
    for threshold in [0.3, 0.4, 0.5, 0.6] {
        let (accuracy, precision, recall, f1) = trainer.evaluate(&training_data, threshold);
        println!(
            "Threshold {:.1}: Accuracy={:.1}%, Precision={:.1}%, Recall={:.1}%, F1={:.1}%",
            threshold,
            accuracy * 100.0,
            precision * 100.0,
            recall * 100.0,
            f1 * 100.0
        );
    }

    // Get trained boxes
    let trained_boxes = trainer.get_boxes();

    // Test: Use entities from training data (they have trained boxes)
    println!("\n=== Testing on Training Example ===");
    let test_example = &training_data[0]; // Use first training example
    let test_entities = &test_example.entities;

    // Get boxes for test entities (should have trained boxes)
    let mut test_boxes = Vec::new();
    for entity in test_entities {
        if let Some(box_embedding) = trained_boxes.get(&entity.start) {
            test_boxes.push(box_embedding.clone());
        } else {
            // Fallback: create a small box
            let center = vec![0.0; dim];
            let box_embedding = BoxEmbedding::from_vector(&center, 0.1);
            test_boxes.push(box_embedding);
        }
    }

    println!("Using {} entities from training data", test_entities.len());

    // Show box overlap scores
    if test_entities.len() >= 2 {
        let box_a = &test_boxes[0];
        let box_b = &test_boxes[1];
        let score = box_a.coreference_score(box_b);
        let p_a_b = box_a.conditional_probability(box_b);
        let p_b_a = box_b.conditional_probability(box_a);
        println!("Coreference score: {:.4}", score);
        println!("  P(A|B): {:.4}", p_a_b);
        println!("  P(B|A): {:.4}", p_b_a);
    }

    // Find optimal threshold based on evaluation
    let mut best_threshold = 0.3;
    let mut best_f1 = 0.0;
    for threshold in [0.2, 0.25, 0.3, 0.35, 0.4] {
        let (_, _, _, f1) = trainer.evaluate(&training_data, threshold);
        if f1 > best_f1 {
            best_f1 = f1;
            best_threshold = threshold;
        }
    }

    // Resolve coreference with optimal threshold
    let mut config = BoxCorefConfig::default();
    config.coreference_threshold = best_threshold;
    println!(
        "Using optimal threshold: {:.2} (F1: {:.1}%)",
        best_threshold,
        best_f1 * 100.0
    );
    let resolver = BoxCorefResolver::new(config);
    let resolved = resolver.resolve_with_boxes(&test_entities, &test_boxes);

    println!("\nResolved entities:");
    for (i, entity) in test_entities.iter().enumerate() {
        println!(
            "  {}: '{}' -> cluster {:?}",
            i, entity.text, resolved[i].canonical_id
        );
    }

    // Check if they corefer
    if test_entities.len() >= 2 && resolved[0].canonical_id == resolved[1].canonical_id {
        println!("\n✓ Coreference resolved correctly!");
    } else if test_entities.len() >= 2 {
        println!("\n✗ Coreference not resolved (may need more training or threshold adjustment)");
    }

    Ok(())
}
