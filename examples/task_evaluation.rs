//! Example: Comprehensive task-dataset-backend evaluation.
//!
//! This example demonstrates how to:
//! 1. Map tasks to suitable datasets
//! 2. Map datasets to compatible backends
//! 3. Run evaluations across all valid combinations
//! 4. Generate comprehensive reports

use anno::eval::task_evaluator::{TaskEvalConfig, TaskEvaluator};
use anno::eval::task_mapping::Task;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Comprehensive Task-Dataset-Backend Evaluation ===\n");

    // Create evaluator
    let evaluator = TaskEvaluator::new()?;

    // Configure evaluation
    let config = TaskEvalConfig {
        tasks: vec![Task::NER, Task::RelationExtraction, Task::IntraDocCoref],
        datasets: vec![],        // Empty = use all suitable datasets
        backends: vec![],        // Empty = use all compatible backends
        max_examples: Some(100), // Limit for quick testing
        require_cached: false,   // Download if needed
    };

    println!("Running comprehensive evaluation...");
    println!("Tasks: {:?}", config.tasks);
    println!("Max examples per dataset: {:?}\n", config.max_examples);

    // Run evaluation
    let results = evaluator.evaluate_all(config)?;

    // Print summary
    println!("=== Evaluation Summary ===");
    println!("Total combinations: {}", results.summary.total_combinations);
    println!("Successful: {}", results.summary.successful);
    println!("Failed: {}", results.summary.failed);
    println!("\nTasks evaluated: {}", results.summary.tasks.len());
    println!("Datasets used: {}", results.summary.datasets.len());
    println!("Backends tested: {}", results.summary.backends.len());

    // Generate markdown report
    let report = results.to_markdown();
    println!("\n=== Markdown Report ===");
    println!("{}", report);

    // Save report to file
    std::fs::write("task_evaluation_report.md", &report)?;
    println!("\nReport saved to task_evaluation_report.md");

    Ok(())
}
