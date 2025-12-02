//! Example: Using Configuration Builders
//!
//! Demonstrates the builder pattern for evaluation configurations.

#[cfg(feature = "eval-advanced")]
use anno::eval::config_builder::TaskEvalConfigBuilder;
#[cfg(feature = "eval-advanced")]
use anno::eval::loader::DatasetId;
#[cfg(feature = "eval-advanced")]
use anno::eval::task_mapping::Task;

#[cfg(feature = "eval-bias")]
use anno::eval::config_builder::BiasDatasetConfigBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Configuration Builder Example ===\n");

    // TaskEvalConfig builder
    #[cfg(feature = "eval-advanced")]
    {
        let config = TaskEvalConfigBuilder::new()
            .with_tasks(vec![Task::NER, Task::IntraDocCoref])
            .add_dataset(DatasetId::WikiGold)
            .add_backend("stacked".to_string())
            .add_backend("gliner2".to_string())
            .with_max_examples(1000)
            .with_seed(42)
            .with_confidence_intervals(true)
            .with_robustness(false)
            .build();

        println!("TaskEvalConfig:");
        println!("  Tasks: {:?}", config.tasks);
        println!("  Datasets: {:?}", config.datasets);
        println!("  Backends: {:?}", config.backends);
        println!("  Max examples: {:?}", config.max_examples);
        println!("  Seed: {:?}", config.seed);
        println!();
    }

    // BiasDatasetConfig builder
    #[cfg(feature = "eval-bias")]
    {
        let bias_config = BiasDatasetConfigBuilder::new()
            .with_frequency_weighting(true)
            .with_validation(true)
            .with_min_samples(20)
            .add_seed(42)
            .add_seed(123)
            .add_seed(456)
            .with_confidence_level(0.95)
            .with_detailed(true)
            .build();

        println!("BiasDatasetConfig:");
        println!("  Frequency weighted: {}", bias_config.frequency_weighted);
        println!(
            "  Validate distributions: {}",
            bias_config.validate_distributions
        );
        println!("  Min samples: {}", bias_config.min_samples_per_category);
        println!("  Seeds: {:?}", bias_config.evaluation_seeds);
        println!("  Confidence level: {:.2}", bias_config.confidence_level);
        println!("  Detailed: {}", bias_config.detailed);
    }

    Ok(())
}
