//! Example: Comprehensive task-dataset-backend evaluation.
//!
//! This example demonstrates how to:
//! 1. Use the unified EvalSystem (recommended)
//! 2. Use configuration builders for cleaner code
//! 3. Run evaluations across all valid combinations
//! 4. Generate comprehensive reports
//!
//! This example shows both the new unified API and the legacy TaskEvaluator API.

#[cfg(feature = "eval-advanced")]
use anno::eval::config_builder::TaskEvalConfigBuilder;
#[cfg(feature = "eval-advanced")]
use anno::eval::task_evaluator::{TaskEvalConfig, TaskEvaluator};
#[cfg(feature = "eval-advanced")]
use anno::eval::task_mapping::Task;
#[cfg(feature = "eval-advanced")]
use anno::eval::BackendName;
#[cfg(feature = "eval-advanced")]
use anno::eval::EvalSystem;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Comprehensive Task-Dataset-Backend Evaluation ===\n");

    #[cfg(feature = "eval-advanced")]
    {
        // === NEW: Using Unified EvalSystem (Recommended) ===
        println!("--- Using Unified EvalSystem (Recommended) ---\n");

        let results = EvalSystem::new()
            .with_tasks(vec![
                Task::NER,
                Task::RelationExtraction,
                Task::IntraDocCoref,
            ])
            .with_max_examples(Some(100))
            .run()?;

        if let Some(standard) = &results.standard {
            println!("Standard Evaluation Results:");
            println!("  Overall F1: {:.1}%", standard.f1 * 100.0);
            println!("  Tasks evaluated: {}", standard.per_task.len());
            println!("  Datasets used: {}", standard.per_dataset.len());
            println!("  Backends tested: {}", standard.per_backend.len());
        }

        println!("\n--- Using Configuration Builder (Alternative) ---\n");

        // === NEW: Using Configuration Builder with BackendName ===
        let config = TaskEvalConfigBuilder::new()
            .with_tasks(vec![
                Task::NER,
                Task::RelationExtraction,
                Task::IntraDocCoref,
            ])
            .with_backends(vec![
                BackendName::Stacked.as_str().to_string(),
                #[cfg(feature = "onnx")]
                BackendName::TPLinker.as_str().to_string(),
                #[cfg(not(feature = "onnx"))]
                "stacked".to_string(), // Fallback
            ])
            .with_max_examples(Some(100))
            .with_confidence_intervals(true)
            .with_familiarity(true)
            .build();

        println!("Configuration:");
        println!("  Tasks: {:?}", config.tasks);
        println!("  Max examples: {:?}", config.max_examples);
        println!("  Confidence intervals: {}", config.confidence_intervals);
        println!();

        // === Legacy: Using TaskEvaluator (Still works) ===
        let evaluator = TaskEvaluator::new()?;
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
    }

    #[cfg(not(feature = "eval-advanced"))]
    {
        println!("This example requires the 'eval-advanced' feature.");
        println!("Run with: cargo run --example task_evaluation --features eval-advanced");
    }

    Ok(())
}
