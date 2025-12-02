//! Example: Using the Unified Evaluation System
//!
//! This demonstrates the new single entry point for all evaluation types,
//! replacing the multiple entry points (TaskEvaluator, EvalHarness, etc.)

#[cfg(feature = "eval-advanced")]
use anno::eval::task_mapping::Task;
use anno::eval::EvalSystem;
use anno::StackedNER;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Unified Evaluation System Example ===\n");

    // Create a model to evaluate
    let model = Box::new(StackedNER::default());
    let model_name = "stacked".to_string();

    // Build unified evaluation system
    #[cfg(feature = "eval-advanced")]
    use anno::eval::task_mapping::Task;

    let mut eval_system = EvalSystem::new();

    // Standard evaluation
    #[cfg(feature = "eval-advanced")]
    {
        eval_system = eval_system
            .with_tasks(vec![Task::NER])
            .with_backends(vec!["stacked".to_string()])
            .with_max_examples(Some(100)); // Quick test
    }

    // Bias evaluation
    #[cfg(feature = "eval-bias")]
    {
        eval_system = eval_system.with_bias_analysis(true);
    }

    // Provide model for bias/calibration
    let results = eval_system.with_model(model, Some(model_name)).run()?;

    // Display results
    println!("Evaluation Results:\n");

    // Standard results
    #[cfg(feature = "eval-advanced")]
    if let Some(standard) = &results.standard {
        println!("Standard Evaluation:");
        println!("  Overall F1: {:.1}%", standard.f1 * 100.0);
        println!("  Precision: {:.1}%", standard.precision * 100.0);
        println!("  Recall: {:.1}%", standard.recall * 100.0);
        println!("  Tasks evaluated: {}", standard.per_task.len());
        println!("  Datasets used: {}", standard.per_dataset.len());
        println!("  Backends tested: {}", standard.per_backend.len());
        println!();
    }

    // Bias results
    #[cfg(feature = "eval-bias")]
    if let Some(bias) = &results.bias {
        println!("Bias Evaluation:");

        if let Some(gender) = &bias.gender {
            println!("  Gender Bias:");
            println!("    Bias Gap: {:.1}%", gender.bias_gap * 100.0);
            println!(
                "    Pro-stereotype Accuracy: {:.1}%",
                gender.pro_stereotype_accuracy * 100.0
            );
            println!(
                "    Anti-stereotype Accuracy: {:.1}%",
                gender.anti_stereotype_accuracy * 100.0
            );
        }

        if let Some(demo) = &bias.demographic {
            println!("  Demographic Bias:");
            println!(
                "    Ethnicity Parity Gap: {:.1}%",
                demo.ethnicity_parity_gap * 100.0
            );
            println!("    Script Bias Gap: {:.1}%", demo.script_bias_gap * 100.0);
            println!(
                "    Overall Recognition Rate: {:.1}%",
                demo.overall_recognition_rate * 100.0
            );
        }

        if let Some(temporal) = &bias.temporal {
            println!("  Temporal Bias:");
            println!(
                "    Historical-Modern Gap: {:.1}%",
                temporal.historical_modern_gap * 100.0
            );
            println!(
                "    Historical Rate: {:.1}%",
                temporal.historical_rate * 100.0
            );
            println!("    Modern Rate: {:.1}%", temporal.modern_rate * 100.0);
        }

        if let Some(length) = &bias.length {
            println!("  Length Bias:");
            println!(
                "    Short vs Long Gap: {:.1}%",
                length.short_vs_long_gap * 100.0
            );
            println!(
                "    Short Entity F1: {:.1}%",
                length.short_entity_f1 * 100.0
            );
            println!("    Long Entity F1: {:.1}%", length.long_entity_f1 * 100.0);
        }
        println!();
    }

    // Metadata
    println!("Metadata:");
    println!("  Timestamp: {}", results.metadata.timestamp);
    if let Some(name) = &results.metadata.model_name {
        println!("  Model: {}", name);
    }
    if let Some(duration) = results.metadata.total_duration_ms {
        println!("  Duration: {:.1}ms", duration);
    }
    println!("  Examples: {}", results.metadata.num_examples);

    // Warnings
    if !results.warnings.is_empty() {
        println!("\nWarnings:");
        for warning in &results.warnings {
            println!("  - {}", warning);
        }
    }

    Ok(())
}
