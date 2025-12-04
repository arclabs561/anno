//! Benchmark command - Comprehensive evaluation across all task-dataset-backend combinations

use std::fs;

use clap::Parser;

#[cfg(feature = "eval-advanced")]
use crate::eval::loader::DatasetId;
#[cfg(feature = "eval-advanced")]
use crate::eval::task_evaluator::TaskEvaluator;
#[cfg(feature = "eval-advanced")]
use crate::eval::task_mapping::Task;

/// Comprehensive evaluation across all task-dataset-backend combinations
#[derive(Parser, Debug)]
pub struct BenchmarkArgs {
    /// Tasks to evaluate (comma-separated: ner,coref,relation). Default: all
    #[arg(short, long, value_delimiter = ',')]
    pub tasks: Option<Vec<String>>,

    /// Datasets to use (comma-separated). Default: all suitable datasets
    #[arg(short, long, value_delimiter = ',')]
    pub datasets: Option<Vec<String>>,

    /// Backends to test (comma-separated). Default: all compatible backends
    #[arg(short, long, value_delimiter = ',')]
    pub backends: Option<Vec<String>>,

    /// Maximum examples per dataset (for quick testing)
    #[arg(short, long)]
    pub max_examples: Option<usize>,

    /// Random seed for sampling (for reproducibility and varied testing)
    #[arg(long)]
    pub seed: Option<u64>,

    /// Only use cached datasets (skip downloads)
    #[arg(long)]
    pub cached_only: bool,

    /// Output file for markdown report (default: stdout)
    #[arg(short, long)]
    pub output: Option<String>,
}

pub fn run(args: BenchmarkArgs) -> Result<(), String> {
    #[cfg(not(feature = "eval-advanced"))]
    {
        let _ = args;
        return Err("Benchmark command requires --features eval-advanced".to_string());
    }

    #[cfg(feature = "eval-advanced")]
    {
        println!("=== Comprehensive Task-Dataset-Backend Evaluation ===\n");

        // Parse tasks
        let tasks = if let Some(task_strs) = args.tasks {
            let mut parsed = Vec::new();
            for t in task_strs {
                match t.to_lowercase().as_str() {
                    "ner" | "ner_task" => parsed.push(Task::NER),
                    "coref" | "coreference" | "intradoc_coref" => parsed.push(Task::IntraDocCoref),
                    "relation" | "relation_extraction" => parsed.push(Task::RelationExtraction),
                    other => {
                        return Err(format!(
                            "Unknown task: {}. Use: ner, coref, relation",
                            other
                        ));
                    }
                }
            }
            parsed
        } else {
            Task::all().to_vec()
        };

        // Parse datasets
        let datasets = if let Some(dataset_strs) = args.datasets {
            let mut parsed = Vec::new();
            for d in dataset_strs {
                let dataset_id: DatasetId = d
                    .parse()
                    .map_err(|e| format!("Invalid dataset '{}': {}", d, e))?;
                parsed.push(dataset_id);
            }
            parsed
        } else {
            vec![] // Empty = use all suitable datasets
        };

        // Parse backends
        let backends = args.backends.unwrap_or_default();

        // Create evaluator
        let evaluator =
            TaskEvaluator::new().map_err(|e| format!("Failed to create evaluator: {}", e))?;

        // Configure evaluation using builder pattern
        use crate::eval::config_builder::TaskEvalConfigBuilder;
        let mut builder = TaskEvalConfigBuilder::new()
            .with_tasks(tasks)
            .with_datasets(datasets)
            .with_backends(backends)
            .require_cached(args.cached_only)
            .with_confidence_intervals(true)
            .with_familiarity(true);

        // Set max_examples (None means "all examples", 0 also means "all examples")
        if let Some(max) = args.max_examples {
            if max > 0 {
                builder = builder.with_max_examples(max);
            }
            // If max == 0, don't set it (None = unlimited)
        }

        // Only set seed if provided (default is 42 in builder)
        if let Some(seed) = args.seed {
            builder = builder.with_seed(seed);
        }

        let config = builder.build();

        println!("Running comprehensive evaluation...");
        println!("Tasks: {:?}", config.tasks);
        if !config.datasets.is_empty() {
            println!("Datasets: {:?}", config.datasets);
        } else {
            println!("Datasets: all suitable datasets");
        }
        if !config.backends.is_empty() {
            println!("Backends: {:?}", config.backends);
        } else {
            println!("Backends: all compatible backends");
        }
        if let Some(max) = config.max_examples {
            println!("Max examples per dataset: {}", max);
        }
        if let Some(seed) = config.seed {
            println!("Random seed: {}", seed);
        }
        println!();

        // Run evaluation
        let results = evaluator
            .evaluate_all(config)
            .map_err(|e| format!("Evaluation failed: {}", e))?;

        // Print summary
        println!("=== Evaluation Summary ===");
        println!("Total combinations: {}", results.summary.total_combinations);
        println!("Successful: {}", results.summary.successful);
        println!(
            "Skipped (feature not available): {}",
            results.summary.skipped
        );
        println!("Failed (actual errors): {}", results.summary.failed);
        println!("\nTasks evaluated: {}", results.summary.tasks.len());
        println!("Datasets used: {}", results.summary.datasets.len());
        println!("Backends tested: {}", results.summary.backends.len());
        println!();

        // Generate markdown report
        let report = results.to_markdown();

        // Output report
        if let Some(output_path) = &args.output {
            fs::write(output_path, &report)
                .map_err(|e| format!("Failed to write report to {}: {}", output_path, e))?;
            println!("Report saved to: {}", output_path);
        } else {
            println!("=== Markdown Report ===");
            println!("{}", report);
        }

        Ok(())
    }
}
