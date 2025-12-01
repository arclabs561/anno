//! Coreference evaluation with chain-length stratification.
//!
//! Demonstrates:
//! - Chain-length stratified metrics (long, short, singleton)
//! - Coreference evaluation across different datasets
//! - Comparison of coreference metrics
//!
//! Run with:
//!   cargo run --example eval_coref_analysis --features eval-advanced

use anno::eval::loader::DatasetId;
use anno::eval::task_evaluator::{TaskEvalConfig, TaskEvaluator};
use anno::eval::task_mapping::Task;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Coreference Evaluation with Chain-Length Stratification ===\n");

    let evaluator = TaskEvaluator::new()?;

    // Test coreference evaluation with chain-length stratification
    let config = TaskEvalConfig {
        tasks: vec![Task::IntraDocCoref],
        datasets: vec![], // Use all compatible coreference datasets
        backends: vec![], // Use all compatible backends
        max_examples: Some(50),
        require_cached: false,
        relation_threshold: 0.5,
        seed: Some(42),
        robustness: false,
        compute_familiarity: false,
        temporal_stratification: false,
        confidence_intervals: false,
        confidence_intervals: true,
    };

    println!("Running coreference evaluation...\n");
    let results = evaluator.evaluate_all(config)?;

    println!("=== Results Summary ===");
    println!("Total combinations: {}", results.summary.total_combinations);
    println!("Successful: {}", results.summary.successful);
    println!("Failed: {}", results.summary.failed);
    println!();

    // Analyze chain-length stratification
    println!("=== Chain-Length Stratification Analysis ===\n");
    for result in &results.results {
        if result.success {
            println!("{} on {}:", result.backend, result.dataset);

            // Extract chain-length metrics from result.metrics
            if let Some(long_f1) = result.metrics.get("chain_long_f1") {
                let long_count = result
                    .metrics
                    .get("chain_long_count")
                    .copied()
                    .unwrap_or(0.0) as usize;
                let short_f1 = result.metrics.get("chain_short_f1").copied().unwrap_or(0.0);
                let short_count = result
                    .metrics
                    .get("chain_short_count")
                    .copied()
                    .unwrap_or(0.0) as usize;
                let singleton_f1 = result
                    .metrics
                    .get("chain_singleton_f1")
                    .copied()
                    .unwrap_or(0.0);
                let singleton_count = result
                    .metrics
                    .get("chain_singleton_count")
                    .copied()
                    .unwrap_or(0.0) as usize;

                println!(
                    "  Long chains (>10): F1 = {:.3}, Count = {}",
                    long_f1, long_count
                );
                println!(
                    "  Short chains (2-10): F1 = {:.3}, Count = {}",
                    short_f1, short_count
                );
                println!(
                    "  Singletons (1): F1 = {:.3}, Count = {}",
                    singleton_f1, singleton_count
                );

                // Overall CoNLL F1
                if let Some(conll_f1) = result.metrics.get("conll_f1") {
                    println!("  Overall CoNLL F1: {:.3}", conll_f1);
                }
            } else {
                println!("  No chain-length stratification available");
            }
            println!();
        }
    }

    Ok(())
}
