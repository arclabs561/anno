//! Comprehensive evaluation script demonstrating all evaluation features.
//!
//! Tests:
//! - Per-example score integration
//! - Stratified metrics computation
//! - Confidence intervals
//! - Temporal stratification (when metadata available)
//! - KB version tracking
//! - Familiarity computation
//! - Robustness testing
//!
//! Run with:
//!   cargo run --example comprehensive_evaluation --features eval-advanced

#[cfg(feature = "eval-advanced")]
use anno::eval::config_builder::TaskEvalConfigBuilder;
use anno::eval::loader::DatasetId;
use anno::eval::task_evaluator::{TaskEvalConfig, TaskEvaluator};
use anno::eval::task_mapping::Task;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Comprehensive Evaluation with Advanced Features ===\n");

    let evaluator = TaskEvaluator::new()?;

    // Test configuration with all new features enabled
    // Use datasets with temporal metadata (TweetNER7, BroadTwitterCorpus) and others
    // === NEW: Using Configuration Builder ===
    #[cfg(feature = "eval-advanced")]
    let config = TaskEvalConfigBuilder::new()
        .with_tasks(vec![Task::NER])
        .add_dataset(DatasetId::TweetNER7) // Has temporal metadata
        .add_dataset(DatasetId::BroadTwitterCorpus) // Has temporal metadata
        .add_dataset(DatasetId::WikiGold) // Standard dataset
        .with_max_examples(Some(50))
        .with_seed(42)
        .with_confidence_intervals(true)
        .with_familiarity(true)
        .with_temporal_stratification(true)
        .with_robustness(true)
        .build();

    #[cfg(not(feature = "eval-advanced"))]
    let config = TaskEvalConfig {
        tasks: vec![Task::NER],
        datasets: vec![
            DatasetId::TweetNER7,
            DatasetId::BroadTwitterCorpus,
            DatasetId::WikiGold,
        ],
        backends: vec![],
        max_examples: Some(50),
        require_cached: false,
        relation_threshold: 0.5,
        confidence_intervals: true,
        compute_familiarity: true,
        temporal_stratification: true,
        robustness: true,
        seed: Some(42),
    };

    println!("Configuration:");
    println!("  - Confidence intervals: {}", config.confidence_intervals);
    println!(
        "  - Familiarity computation: {}",
        config.compute_familiarity
    );
    println!(
        "  - Temporal stratification: {}",
        config.temporal_stratification
    );
    println!("  - Robustness testing: {}", config.robustness);
    println!("  - Max examples: {:?}", config.max_examples);
    println!("  - Seed: {:?}\n", config.seed);

    println!("Running evaluation with new features...\n");
    let results = evaluator.evaluate_all(config)?;

    println!("=== Results Summary ===");
    println!("Total combinations: {}", results.summary.total_combinations);
    println!("Successful: {}", results.summary.successful);
    println!("Failed: {}", results.summary.failed);
    println!();

    // Check for new features in results
    let mut has_stratified = 0;
    let mut has_confidence_intervals = 0;
    let mut has_familiarity = 0;
    let mut has_kb_version = 0;
    let mut has_temporal = 0;

    for result in &results.results {
        if result.stratified.is_some() {
            has_stratified += 1;
            if let Some(ref stratified) = result.stratified {
                if stratified.by_temporal_stratum.is_some() {
                    has_temporal += 1;
                }
            }
        }
        if result.confidence_intervals.is_some() {
            has_confidence_intervals += 1;
        }
        if result.label_shift.is_some() {
            has_familiarity += 1;
        }
        if result.kb_version.is_some() {
            has_kb_version += 1;
        }
    }

    println!("=== Feature Verification ===");
    println!(
        "Results with stratified metrics: {}/{}",
        has_stratified,
        results.results.len()
    );
    println!(
        "Results with confidence intervals: {}/{}",
        has_confidence_intervals,
        results.results.len()
    );
    println!(
        "Results with familiarity: {}/{}",
        has_familiarity,
        results.results.len()
    );
    println!(
        "Results with KB version: {}/{}",
        has_kb_version,
        results.results.len()
    );
    println!(
        "Results with temporal stratification: {}/{}",
        has_temporal,
        results.results.len()
    );
    println!();

    // Show detailed example
    if let Some(result) = results.results.iter().find(|r| r.success) {
        println!("=== Example Result ===");
        println!("Task: {:?}", result.task);
        println!("Dataset: {:?}", result.dataset);
        println!("Backend: {}", result.backend);
        println!("Success: {}", result.success);
        println!("Examples: {}", result.num_examples);

        if let Some(ref ci) = result.confidence_intervals {
            println!("\nConfidence Intervals (95%):");
            println!("  F1: [{:.3}, {:.3}]", ci.f1_ci.0, ci.f1_ci.1);
            println!(
                "  Precision: [{:.3}, {:.3}]",
                ci.precision_ci.0, ci.precision_ci.1
            );
            println!("  Recall: [{:.3}, {:.3}]", ci.recall_ci.0, ci.recall_ci.1);
        }

        if let Some(ref stratified) = result.stratified {
            println!("\nStratified Metrics:");
            println!("  Entity types: {}", stratified.by_entity_type.len());
            if let Some(ref temporal) = stratified.by_temporal_stratum {
                println!("  Temporal strata: {}", temporal.len());
            }
        }

        if let Some(ref label_shift) = result.label_shift {
            println!("\nFamiliarity:");
            println!("  Overlap ratio: {:.2}%", label_shift.overlap_ratio * 100.0);
            println!("  Familiarity: {:.2}", label_shift.familiarity);
            println!(
                "  Zero-shot types: {}",
                label_shift.true_zero_shot_types.len()
            );
        }

        if let Some(ref kb_version) = result.kb_version {
            println!("\nKB Version: {}", kb_version);
        }
    }

    // Generate and save report
    let report = results.to_markdown();
    std::fs::write("comprehensive_evaluation_report.md", &report)?;
    println!("\n=== Report Generated ===");
    println!("Saved to: comprehensive_evaluation_report.md");
    println!("\nReport preview (first 1000 chars):");
    println!("{}", &report.chars().take(1000).collect::<String>());

    Ok(())
}
