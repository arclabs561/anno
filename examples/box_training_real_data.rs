//! Train box embeddings on real coreference datasets (GAP, PreCo, etc.)
//!
//! This example demonstrates training box embeddings on real-world coreference
//! datasets. It loads GAP dataset, converts it to training examples, and
//! trains a box embedding model for coreference resolution.

use anno::backends::box_embeddings::BoxCorefConfig;
use anno::backends::box_embeddings::BoxEmbedding;
use anno::backends::box_embeddings_training::{
    coref_documents_to_training_examples, split_train_val, BoxEmbeddingTrainer, TrainingConfig,
};
use anno::eval::coref_loader::CorefLoader;
use anno::eval::coref_resolver::BoxCorefResolver;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Box Embedding Training on Real Datasets ===\n");

    // Load GAP dataset
    println!("Loading GAP dataset...");
    let loader = CorefLoader::new()?;

    // Try to load GAP (will download if not cached)
    let gap_docs = match loader.load_gap() {
        Ok(docs) => {
            println!("Loaded {} GAP documents", docs.len());
            docs
        }
        Err(e) => {
            println!("Failed to load GAP dataset: {}", e);
            println!("Using synthetic data instead...");
            use anno::eval::coref_loader::synthetic_coref_dataset;
            synthetic_coref_dataset(10)
        }
    };

    // Convert to training examples
    println!("Converting to training examples...");
    let mut all_data = coref_documents_to_training_examples(&gap_docs);

    // Use more data for better generalization (increased to 300 for experiments)
    if all_data.len() > 300 {
        println!("Limiting to first 300 examples for faster training");
        all_data.truncate(300);
    }

    // Split into train/validation sets (80/20 split)
    let (training_data, validation_data) = split_train_val(&all_data, 0.2);
    println!(
        "Created {} training examples, {} validation examples",
        training_data.len(),
        validation_data.len()
    );

    // Count entities and chains
    let total_entities: usize = training_data.iter().map(|e| e.entities.len()).sum();
    let total_chains: usize = training_data.iter().map(|e| e.chains.len()).sum();
    println!(
        "Total entities: {}, Total chains: {}",
        total_entities, total_chains
    );

    // Training configuration - Experiment: Slower learning rate for stability
    let config = TrainingConfig {
        learning_rate: 0.0008,   // Slightly slower for more stable training
        negative_weight: 0.6,    // Balanced
        margin: 0.04,            // Moderate margin
        regularization: 0.00001, // Very light regularization
        epochs: 150,             // More epochs
        batch_size: 8,
        warmup_epochs: 10,
        use_self_adversarial: true,
        adversarial_temperature: 1.0,
        early_stopping_patience: Some(40), // More patience
        early_stopping_min_delta: 0.0001,
        positive_focus_epochs: Some(35), // Longer positive focus
    };

    // Initialize trainer
    let dim = 32; // Standard dimension - proven stable
    let mut trainer = BoxEmbeddingTrainer::new(config.clone(), dim, None);

    // Initialize boxes from entities
    trainer.initialize_boxes(&training_data, None);

    println!("\nInitialized {} boxes", trainer.get_boxes().len());

    // Show initial overlap statistics
    let (avg_pos, avg_neg, overlap_rate) = trainer.get_overlap_stats(&training_data);
    println!("Initial overlap stats:");
    println!("  Avg positive pair score: {:.4}", avg_pos);
    println!("  Avg negative pair score: {:.4}", avg_neg);
    println!("  Overlap rate: {:.1}%", overlap_rate * 100.0);

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

    // Show final overlap statistics
    let (avg_pos, avg_neg, overlap_rate) = trainer.get_overlap_stats(&training_data);
    println!("\nFinal overlap stats:");
    println!("  Avg positive pair score: {:.4}", avg_pos);
    println!("  Avg negative pair score: {:.4}", avg_neg);
    println!("  Overlap rate: {:.1}%", overlap_rate * 100.0);

    // Evaluate on training and validation sets with lower thresholds
    println!("\n=== Evaluation (Training Set) ===");
    for threshold in [0.01, 0.02, 0.03, 0.05, 0.1, 0.15] {
        let (accuracy, precision, recall, f1) = trainer.evaluate(&training_data, threshold);
        println!(
            "Threshold {:.2}: Accuracy={:.1}%, Precision={:.1}%, Recall={:.1}%, F1={:.1}%",
            threshold,
            accuracy * 100.0,
            precision * 100.0,
            recall * 100.0,
            f1 * 100.0
        );
    }

    println!("\n=== Evaluation (Validation Set) - Pair-wise ===");
    let mut best_threshold = 0.01;
    let mut best_f1 = 0.0;
    for threshold in [0.01, 0.02, 0.03, 0.05, 0.1, 0.15] {
        let (accuracy, precision, recall, f1) = trainer.evaluate(&validation_data, threshold);
        println!(
            "Threshold {:.2}: Accuracy={:.1}%, Precision={:.1}%, Recall={:.1}%, F1={:.1}%",
            threshold,
            accuracy * 100.0,
            precision * 100.0,
            recall * 100.0,
            f1 * 100.0
        );
        if f1 > best_f1 {
            best_f1 = f1;
            best_threshold = threshold;
        }
    }
    println!(
        "Best threshold (pair-wise): {:.3} (F1: {:.1}%)",
        best_threshold,
        best_f1 * 100.0
    );

    // Evaluate using standard coreference metrics
    println!("\n=== Standard Coreference Metrics (Validation Set) ===");
    let standard_eval = trainer.evaluate_standard_metrics(&validation_data, best_threshold);
    println!(
        "CoNLL F1: {:.1}% (standard benchmark metric)",
        standard_eval.conll_f1 * 100.0
    );
    println!(
        "MUC: P={:.1}%, R={:.1}%, F1={:.1}%",
        standard_eval.muc.precision * 100.0,
        standard_eval.muc.recall * 100.0,
        standard_eval.muc.f1 * 100.0
    );
    println!(
        "B³: P={:.1}%, R={:.1}%, F1={:.1}%",
        standard_eval.b_cubed.precision * 100.0,
        standard_eval.b_cubed.recall * 100.0,
        standard_eval.b_cubed.f1 * 100.0
    );
    println!(
        "CEAF-e: P={:.1}%, R={:.1}%, F1={:.1}%",
        standard_eval.ceaf_e.precision * 100.0,
        standard_eval.ceaf_e.recall * 100.0,
        standard_eval.ceaf_e.f1 * 100.0
    );
    println!(
        "LEA: P={:.1}%, R={:.1}%, F1={:.1}%",
        standard_eval.lea.precision * 100.0,
        standard_eval.lea.recall * 100.0,
        standard_eval.lea.f1 * 100.0
    );
    println!(
        "BLANC: P={:.1}%, R={:.1}%, F1={:.1}%",
        standard_eval.blanc.precision * 100.0,
        standard_eval.blanc.recall * 100.0,
        standard_eval.blanc.f1 * 100.0
    );

    // Chain-length stratified metrics (if available)
    if let Some(stats) = &standard_eval.chain_stats {
        println!("\nChain-length stratified metrics:");
        println!(
            "  Long chains (>10): {} chains, F1={:.1}%",
            stats.long_chain_count,
            stats.long_chain_f1 * 100.0
        );
        println!(
            "  Short chains (2-10): {} chains, F1={:.1}%",
            stats.short_chain_count,
            stats.short_chain_f1 * 100.0
        );
        println!(
            "  Singletons (1): {} chains, F1={:.1}%",
            stats.singleton_count,
            stats.singleton_f1 * 100.0
        );
    }

    // Test on a sample document
    if !training_data.is_empty() {
        println!("\n=== Testing on Sample Document ===");
        let test_example = &training_data[0];
        let test_entities = &test_example.entities;

        // Get boxes for test entities
        let trained_boxes = trainer.get_boxes();
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

        println!("Document: {} entities", test_entities.len());
        if test_entities.len() >= 2 {
            let box_a = &test_boxes[0];
            let box_b = &test_boxes[1];
            let score = box_a.coreference_score(box_b);
            let p_a_b = box_a.conditional_probability(box_b);
            let p_b_a = box_b.conditional_probability(box_a);
            println!("Coreference score (first two entities): {:.4}", score);
            println!("  P(A|B): {:.4}, P(B|A): {:.4}", p_a_b, p_b_a);
        }

        // Find best threshold on validation set (using more granular search)
        let mut best_threshold = 0.01;
        let mut best_f1 = 0.0;
        let mut best_metrics = (0.0, 0.0, 0.0, 0.0);

        // Search more granularly around promising thresholds
        let thresholds = vec![
            0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2,
        ];

        for threshold in thresholds {
            let (accuracy, precision, recall, f1) = trainer.evaluate(&validation_data, threshold);
            // Prefer thresholds with better F1, but also consider precision-recall balance
            let pr_product: f32 = precision * recall;
            let score = f1 + pr_product.sqrt() * 0.1_f32; // Slight bonus for balanced precision/recall
            let best_pr_product: f32 = best_metrics.1 * best_metrics.2;
            let best_score = best_f1 + best_pr_product.sqrt() * 0.1_f32;
            if score > best_score {
                best_f1 = f1;
                best_threshold = threshold;
                best_metrics = (accuracy, precision, recall, f1);
            }
        }
        println!("Best threshold on validation set: {:.3} (F1: {:.1}%, Precision: {:.1}%, Recall: {:.1}%)", 
            best_threshold, best_f1 * 100.0, best_metrics.1 * 100.0, best_metrics.2 * 100.0);

        let mut config = BoxCorefConfig::default();
        config.coreference_threshold = best_threshold;
        println!(
            "Using optimal threshold: {:.2} (Pair-wise F1: {:.1}%)",
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

        // Show coreference clusters
        let mut clusters: std::collections::HashMap<Option<u64>, Vec<usize>> =
            std::collections::HashMap::new();
        for (i, entity) in resolved.iter().enumerate() {
            clusters.entry(entity.canonical_id).or_default().push(i);
        }

        println!("\nCoreference clusters:");
        for (cluster_id, entity_indices) in &clusters {
            if entity_indices.len() > 1 {
                println!(
                    "  Cluster {:?}: {}",
                    cluster_id,
                    entity_indices
                        .iter()
                        .map(|&i| format!("'{}'", test_entities[i].text))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
        }

        // Show gold chains for comparison
        println!("\nGold coreference chains:");
        for (i, chain) in test_example.chains.iter().enumerate() {
            let mentions: Vec<String> = chain
                .mentions
                .iter()
                .map(|m| format!("'{}'", m.text))
                .collect();
            println!("  Chain {}: {}", i, mentions.join(", "));
        }

        // Verify correctness
        if test_example.chains.len() == 1 && clusters.len() == 1 {
            let gold_chain = &test_example.chains[0];
            let resolved_cluster = clusters.keys().next().unwrap();
            let gold_mentions: Vec<usize> = gold_chain.mentions.iter().map(|m| m.start).collect();
            let resolved_entity_ids: Vec<usize> = resolved
                .iter()
                .enumerate()
                .filter(|(_, e)| e.canonical_id == *resolved_cluster)
                .map(|(i, _)| test_entities[i].start)
                .collect();

            let all_gold_in_resolved = gold_mentions
                .iter()
                .all(|&id| resolved_entity_ids.contains(&id));
            let all_resolved_in_gold = resolved_entity_ids
                .iter()
                .all(|&id| gold_mentions.contains(&id));

            if all_gold_in_resolved && all_resolved_in_gold {
                println!("\n✓ PERFECT MATCH: All gold mentions correctly resolved!");
            } else {
                println!(
                    "\n⚠ Partial match: Gold mentions: {:?}, Resolved: {:?}",
                    gold_mentions, resolved_entity_ids
                );
            }
        }
    }

    Ok(())
}
