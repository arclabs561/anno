//! Combined evaluation example: Standard + Bias evaluations.
//!
//! Demonstrates how to run both standard task evaluations (on real datasets)
//! and bias evaluations (on synthetic datasets) in the same workflow.
//!
//! Run: cargo run --example combined_evaluation --features eval-bias

use anno::eval::bias_config::BiasDatasetConfig;
use anno::eval::coref_resolver::SimpleCorefResolver;
use anno::eval::demographic_bias::{create_diverse_name_dataset, DemographicBiasEvaluator};
use anno::eval::gender_bias::{create_winobias_templates, GenderBiasEvaluator};
use anno::eval::length_bias::{create_length_varied_dataset, EntityLengthEvaluator};
use anno::eval::task_evaluator::{TaskEvalConfig, TaskEvaluator};
use anno::eval::task_mapping::Task;
use anno::eval::temporal_bias::{create_temporal_name_dataset, TemporalBiasEvaluator};
use anno::RegexNER;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Combined Evaluation: Standard + Bias ===\n");

    // Create a model to evaluate
    let model = RegexNER::new();

    // ========================================================================
    // Part 1: Standard Task Evaluation (Real Datasets)
    // ========================================================================
    println!("--- Part 1: Standard Task Evaluation ---\n");
    println!("Evaluating on real datasets via TaskEvaluator...\n");

    let task_evaluator = TaskEvaluator::new()?;

    // Configure standard evaluation
    let config = TaskEvalConfig {
        tasks: vec![Task::NER],
        datasets: vec![],                      // Use all suitable datasets for NER
        backends: vec!["pattern".to_string()], // Use pattern backend
        max_examples: Some(100),               // Limit for quick demo
        seed: Some(42),
        require_cached: false,
        ..Default::default()
    };

    // Run standard evaluation
    match task_evaluator.evaluate_all(config) {
        Ok(standard_results) => {
            println!("Standard evaluation completed:");
            println!(
                "  Total combinations: {}",
                standard_results.summary.total_combinations
            );
            println!("  Successful: {}", standard_results.summary.successful);
            println!("  Failed: {}", standard_results.summary.failed);

            if !standard_results.results.is_empty() {
                let first_result = &standard_results.results[0];
                if first_result.success {
                    println!("\nSample result:");
                    println!("  Dataset: {:?}", first_result.dataset);
                    println!("  Backend: {}", first_result.backend);
                    if let Some(f1) = first_result.metrics.get("f1") {
                        println!("  F1: {:.1}%", f1 * 100.0);
                    }
                }
            }
        }
        Err(e) => {
            println!("Standard evaluation error: {}", e);
            println!("(This is expected if datasets aren't cached)");
        }
    }

    // ========================================================================
    // Part 2: Bias Evaluation (Synthetic Datasets)
    // ========================================================================
    println!("\n--- Part 2: Bias Evaluation ---\n");
    println!("Evaluating on synthetic bias datasets...\n");

    // Configure bias evaluation
    let bias_config = BiasDatasetConfig::default()
        .with_frequency_weighting()
        .with_validation()
        .with_detailed(true);

    // 2a. Demographic Bias
    println!("2a. Demographic Bias (Names):");
    let names = create_diverse_name_dataset();
    println!("  Dataset size: {} names", names.len());

    let demo_evaluator = DemographicBiasEvaluator::with_config(true, bias_config.clone());
    let demo_results = demo_evaluator.evaluate_ner(&model, &names);

    println!(
        "  Overall recognition: {:.1}%",
        demo_results.overall_recognition_rate * 100.0
    );
    println!(
        "  Ethnicity parity gap: {:.1}%",
        demo_results.ethnicity_parity_gap * 100.0
    );
    println!(
        "  Script bias gap: {:.1}%",
        demo_results.script_bias_gap * 100.0
    );

    if let Some(freq) = &demo_results.frequency_weighted {
        println!(
            "  Frequency-weighted rate: {:.1}%",
            freq.weighted_rate * 100.0
        );
    }

    if let Some(validation) = &demo_results.distribution_validation {
        println!("  Distribution valid: {}", validation.is_valid);
    }

    // 2b. Gender Bias (Coreference)
    println!("\n2b. Gender Bias (Coreference):");
    let resolver = SimpleCorefResolver::default();
    let templates = create_winobias_templates();
    println!("  Dataset size: {} examples", templates.len());

    let gender_evaluator = GenderBiasEvaluator::new(true);
    let gender_results = gender_evaluator.evaluate_resolver(&resolver, &templates);

    println!(
        "  Pro-stereotypical accuracy: {:.1}%",
        gender_results.pro_stereotype_accuracy * 100.0
    );
    println!(
        "  Anti-stereotypical accuracy: {:.1}%",
        gender_results.anti_stereotype_accuracy * 100.0
    );
    println!("  Bias gap: {:.1}%", gender_results.bias_gap * 100.0);

    // 2c. Temporal Bias
    println!("\n2c. Temporal Bias (Names by Decade):");
    let temporal_names = create_temporal_name_dataset();
    println!("  Dataset size: {} names", temporal_names.len());

    let temporal_evaluator = TemporalBiasEvaluator::new(true);
    let temporal_results = temporal_evaluator.evaluate(&model, &temporal_names);

    println!(
        "  Historical (pre-1950): {:.1}%",
        temporal_results.historical_rate * 100.0
    );
    println!(
        "  Modern (post-2000): {:.1}%",
        temporal_results.modern_rate * 100.0
    );
    println!(
        "  Temporal gap: {:.1}%",
        temporal_results.historical_modern_gap * 100.0
    );

    // 2d. Length Bias
    println!("\n2d. Length Bias (Entity Length):");
    let length_examples = create_length_varied_dataset();
    println!("  Dataset size: {} examples", length_examples.len());

    let length_evaluator = EntityLengthEvaluator::new(true);
    let length_results = length_evaluator.evaluate(&model, &length_examples);

    println!(
        "  Short vs long gap: {:.1}%",
        length_results.short_vs_long_gap * 100.0
    );

    // ========================================================================
    // Part 3: Combined Analysis
    // ========================================================================
    println!("\n--- Part 3: Combined Analysis ---\n");

    println!("Summary:");
    println!("  Standard evaluation: Real datasets via TaskEvaluator");
    println!("  Bias evaluation: Synthetic datasets via bias evaluators");
    println!("\nKey Differences:");
    println!("  - Standard: Uses DatasetLoader for real datasets");
    println!("  - Bias: Uses create_*_dataset() for synthetic datasets");
    println!("  - Standard: Task → Dataset → Backend pipeline");
    println!("  - Bias: Direct evaluator → model pipeline");
    println!("\nBoth are complementary:");
    println!("  - Standard: Measures accuracy/performance");
    println!("  - Bias: Measures fairness/parity");

    Ok(())
}
