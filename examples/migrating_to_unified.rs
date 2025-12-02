//! Example: Migrating from Old APIs to Unified EvalSystem
//!
//! This example shows how to migrate from the old multiple-entry-point APIs
//! to the new unified EvalSystem.

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
use anno::StackedNER;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Migrating to Unified EvalSystem ===\n");

    #[cfg(feature = "eval-advanced")]
    {
        // ====================================================================
        // Migration Pattern 1: TaskEvaluator → EvalSystem
        // ====================================================================
        println!("--- Pattern 1: TaskEvaluator → EvalSystem ---\n");

        // OLD WAY:
        println!("OLD WAY (TaskEvaluator):");
        let evaluator = TaskEvaluator::new()?;
        let old_config = TaskEvalConfig {
            tasks: vec![Task::NER],
            datasets: vec![],
            backends: vec!["stacked".to_string()],
            max_examples: Some(100),
            seed: Some(42),
            require_cached: false,
            relation_threshold: 0.5,
            robustness: false,
            compute_familiarity: true,
            temporal_stratification: false,
            confidence_intervals: true,
        };
        let old_results = evaluator.evaluate_all(old_config)?;
        println!(
            "  Results: {} combinations",
            old_results.summary.total_combinations
        );

        // NEW WAY:
        println!("\nNEW WAY (EvalSystem):");
        let model = Box::new(StackedNER::default());
        let new_results = EvalSystem::new()
            .with_tasks(vec![Task::NER])
            .with_backends(vec!["stacked".to_string()])
            .with_max_examples(Some(100))
            .with_seed(42)
            .with_model(model, Some("stacked".to_string()))
            .run()?;
        if let Some(standard) = &new_results.standard {
            println!("  Results: F1 = {:.1}%", standard.f1 * 100.0);
        }

        // ====================================================================
        // Migration Pattern 2: Using Configuration Builder
        // ====================================================================
        println!("\n--- Pattern 2: Configuration Builder ---\n");

        // OLD WAY (manual struct):
        println!("OLD WAY (manual struct):");
        let _old_config = TaskEvalConfig {
            tasks: vec![Task::NER],
            datasets: vec![],
            backends: vec!["stacked".to_string()],
            max_examples: Some(100),
            seed: Some(42),
            require_cached: false,
            relation_threshold: 0.5,
            robustness: false,
            compute_familiarity: true,
            temporal_stratification: false,
            confidence_intervals: true,
        };

        // NEW WAY (builder):
        println!("NEW WAY (builder):");
        let new_config = TaskEvalConfigBuilder::new()
            .with_tasks(vec![Task::NER])
            .with_backends(vec!["stacked".to_string()])
            .with_max_examples(Some(100))
            .with_seed(42)
            .with_confidence_intervals(true)
            .with_familiarity(true)
            .build();
        println!("  Config built with {} tasks", new_config.tasks.len());

        // ====================================================================
        // Migration Pattern 3: Type-Safe Backend Names
        // ====================================================================
        println!("\n--- Pattern 3: Type-Safe Backend Names ---\n");

        // OLD WAY (strings):
        println!("OLD WAY (strings):");
        let _old_backend = "gliner2"; // Typo-prone!
        let _old_backend2 = "gliner_onnx"; // Easy to misspell

        // NEW WAY (enum):
        println!("NEW WAY (enum):");
        let new_backend = BackendName::Stacked; // Compile-time checked!
        #[cfg(feature = "onnx")]
        let new_backend2 = BackendName::GLiNEROnnx; // IDE autocomplete!
        #[cfg(not(feature = "onnx"))]
        let new_backend2 = BackendName::Stacked; // Fallback if onnx not enabled
        println!("  Backend 1: {} ({})", new_backend.as_str(), new_backend);
        println!("  Backend 2: {} ({})", new_backend2.as_str(), new_backend2);

        #[cfg(feature = "onnx")]
        {
            let gliner2 = BackendName::GLiNER2;
            println!("  Backend 3 (GLiNER2): {} ({})", gliner2.as_str(), gliner2);
        }

        // Parse from string (backward compatible):
        if let Some(backend) = BackendName::try_parse("stacked") {
            println!("  Parsed 'stacked' → {:?}", backend);
        }

        // Or use FromStr trait:
        match "stacked".parse::<BackendName>() {
            Ok(backend) => println!("  Parsed via FromStr: {:?}", backend),
            Err(e) => println!("  Parse error: {}", e),
        }

        // ====================================================================
        // Migration Pattern 4: Combined Evaluation
        // ====================================================================
        println!("\n--- Pattern 4: Combined Standard + Bias ---\n");

        // OLD WAY (separate calls):
        println!("OLD WAY (separate calls):");
        println!("  // Run standard evaluation");
        println!("  let standard = task_evaluator.evaluate_all(...)?;");
        println!("  // Run bias evaluation separately");
        println!("  let bias = bias_evaluator.evaluate_ner(...)?;");

        // NEW WAY (unified):
        println!("\nNEW WAY (unified):");
        let model = Box::new(StackedNER::default());
        let combined_results = EvalSystem::new()
            .with_tasks(vec![Task::NER])
            .with_backends(vec!["stacked".to_string()])
            .with_bias_analysis(true) // Just enable it!
            .with_model(model, Some("stacked".to_string()))
            .run()?;

        if let Some(standard) = &combined_results.standard {
            println!("  Standard F1: {:.1}%", standard.f1 * 100.0);
        }
        if let Some(bias) = &combined_results.bias {
            if let Some(demo) = &bias.demographic {
                println!("  Bias gap: {:.1}%", demo.ethnicity_parity_gap * 100.0);
            }
        }

        println!("\n=== Migration Complete ===");
        println!("All examples show the new unified approach!");
    }

    #[cfg(not(feature = "eval-advanced"))]
    {
        println!("This example requires the 'eval-advanced' feature.");
    }

    Ok(())
}
