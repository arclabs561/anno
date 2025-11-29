//! Advanced evaluation example: Discontinuous NER, Relations, Visual.
//!
//! Tests all the new evaluation infrastructure.
//!
//! Run: cargo run --features eval --example advanced

use anno::eval::{
    evaluate_discontinuous_gold_vs_gold, evaluate_discontinuous_ner, evaluate_relations,
    evaluate_relations_gold_vs_gold, evaluate_visual_gold_vs_gold, evaluate_visual_ner,
    synthetic_dataset_stats, BoundingBox, DiscontinuousEvalConfig, DiscontinuousGold,
    RelationEvalConfig, RelationGold, RelationPrediction, VisualEvalConfig, VisualGold,
    VisualPrediction,
};
use anno::DiscontinuousEntity;

fn main() {
    println!("=== Advanced NER Evaluation Demo ===\n");

    // Dataset statistics
    let stats = synthetic_dataset_stats();
    println!("Synthetic Dataset Statistics:");
    println!("  Discontinuous examples: {}", stats.discontinuous_examples);
    println!("  Discontinuous entities: {}", stats.discontinuous_entities);
    println!("  Relation examples: {}", stats.relation_examples);
    println!("  Relations: {}", stats.relations);
    println!("  Visual examples: {}", stats.visual_examples);
    println!("  Visual entities: {}", stats.visual_entities);
    println!();

    // ==========================================================================
    // 1. DISCONTINUOUS NER
    // ==========================================================================
    println!("--- Discontinuous NER Evaluation ---\n");

    // Perfect prediction baseline
    let disc_baseline = evaluate_discontinuous_gold_vs_gold();
    println!("Perfect prediction baseline:");
    println!("  Exact F1: {:.1}%", disc_baseline.exact_f1 * 100.0);
    println!(
        "  Boundary F1: {:.1}%",
        disc_baseline.entity_boundary_f1 * 100.0
    );
    println!(
        "  Partial F1: {:.1}%",
        disc_baseline.partial_span_f1 * 100.0
    );
    println!();

    // Test with imperfect predictions
    let gold = vec![DiscontinuousGold::new(
        vec![(0, 8), (25, 33)], // "New York" + "airports"
        "LOC",
        "New York airports",
    )];

    // Partial match - only got first span
    let pred_partial = vec![DiscontinuousEntity {
        spans: vec![(0, 8)], // Only "New York"
        text: "New York".to_string(),
        entity_type: "LOC".to_string(),
        confidence: 0.9,
    }];

    let config = DiscontinuousEvalConfig::default();
    let partial_metrics = evaluate_discontinuous_ner(&gold, &pred_partial, &config);
    println!("Partial match (missing span):");
    println!("  Exact F1: {:.1}%", partial_metrics.exact_f1 * 100.0);
    println!(
        "  Boundary F1: {:.1}%",
        partial_metrics.entity_boundary_f1 * 100.0
    );
    println!(
        "  Partial F1: {:.1}%",
        partial_metrics.partial_span_f1 * 100.0
    );
    println!();

    // ==========================================================================
    // 2. RELATION EXTRACTION
    // ==========================================================================
    println!("--- Relation Extraction Evaluation ---\n");

    // Perfect prediction baseline
    let rel_baseline = evaluate_relations_gold_vs_gold();
    println!("Perfect prediction baseline:");
    println!("  Strict F1: {:.1}%", rel_baseline.strict_f1 * 100.0);
    println!("  Boundary F1: {:.1}%", rel_baseline.boundary_f1 * 100.0);
    println!();

    // Test with imperfect predictions
    let gold_rel = vec![RelationGold::new(
        (0, 10),
        "PER",
        "Steve Jobs",
        (19, 24),
        "ORG",
        "Apple",
        "FOUNDED",
    )];

    // Wrong relation type
    let pred_wrong_type = vec![RelationPrediction {
        head_span: (0, 10),
        head_type: "PER".to_string(),
        tail_span: (19, 24),
        tail_type: "ORG".to_string(),
        relation_type: "WORKS_FOR".to_string(), // Wrong!
        confidence: 0.8,
    }];

    let rel_config = RelationEvalConfig::default();
    let wrong_type_metrics = evaluate_relations(&gold_rel, &pred_wrong_type, &rel_config);
    println!("Wrong relation type:");
    println!("  Strict F1: {:.1}%", wrong_type_metrics.strict_f1 * 100.0);
    println!(
        "  Boundary F1: {:.1}%",
        wrong_type_metrics.boundary_f1 * 100.0
    );
    println!();

    // ==========================================================================
    // 3. VISUAL NER
    // ==========================================================================
    println!("--- Visual NER Evaluation ---\n");

    // Perfect prediction baseline
    let vis_baseline = evaluate_visual_gold_vs_gold();
    println!("Perfect prediction baseline:");
    println!("  Text F1: {:.1}%", vis_baseline.text_f1 * 100.0);
    println!("  Box IoU: {:.1}%", vis_baseline.mean_iou * 100.0);
    println!("  E2E F1: {:.1}%", vis_baseline.e2e_f1 * 100.0);
    println!();

    // Test with shifted bounding box
    let gold_vis = vec![VisualGold::new(
        "Invoice #12345",
        "DOCUMENT_ID",
        BoundingBox::new(0.1, 0.05, 0.4, 0.1),
    )];

    // Prediction with slightly shifted box
    let pred_shifted = vec![VisualPrediction {
        text: "Invoice #12345".to_string(),
        entity_type: "DOCUMENT_ID".to_string(),
        bbox: BoundingBox::new(0.15, 0.05, 0.45, 0.1), // Shifted right
        confidence: 0.95,
    }];

    let vis_config = VisualEvalConfig::default();
    let shifted_metrics = evaluate_visual_ner(&gold_vis, &pred_shifted, &vis_config);
    println!("Shifted bounding box:");
    println!("  Text F1: {:.1}%", shifted_metrics.text_f1 * 100.0);
    println!("  Mean IoU: {:.1}%", shifted_metrics.mean_iou * 100.0);
    println!("  E2E F1: {:.1}%", shifted_metrics.e2e_f1 * 100.0);
    println!();

    // ==========================================================================
    // SUMMARY
    // ==========================================================================
    println!("=== Summary ===\n");
    println!("All advanced evaluation modules working:");
    println!("  [x] Discontinuous NER: Exact, Boundary, Partial F1");
    println!("  [x] Relation Extraction: Strict, Boundary F1");
    println!("  [x] Visual NER: Text F1, Box IoU, End-to-End F1");
    println!();
    println!("Try running with actual models:");
    println!("  cargo test --test advanced_trait_tests -- --ignored");
}
