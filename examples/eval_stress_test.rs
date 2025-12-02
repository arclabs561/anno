//! Stress test for evaluation system.
//!
//! Tests:
//! - Large sample sizes
//! - Multiple datasets simultaneously
//! - All backends
//! - Edge cases (empty results, single examples, etc.)
//!
//! Run with:
//!   cargo run --example eval_stress_test --features eval-advanced

#[cfg(feature = "eval-advanced")]
use anno::eval::config_builder::TaskEvalConfigBuilder;
use anno::eval::loader::DatasetId;
use anno::eval::task_evaluator::{TaskEvalConfig, TaskEvaluator};
use anno::eval::task_mapping::Task;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Evaluation System Stress Test ===\n");

    let evaluator = TaskEvaluator::new()?;

    // Test 1: Multiple datasets with all features enabled
    println!("Test 1: Multi-Dataset Evaluation with All Features\n");

    // === NEW: Using Configuration Builder ===
    #[cfg(feature = "eval-advanced")]
    let config = TaskEvalConfigBuilder::new()
        .with_tasks(vec![Task::NER])
        .add_dataset(DatasetId::WikiGold)
        .add_dataset(DatasetId::TweetNER7)
        .add_dataset(DatasetId::BroadTwitterCorpus)
        .add_backend("stacked".to_string())
        .add_backend("tplinker".to_string())
        .with_max_examples(Some(100))
        .with_seed(42)
        .with_robustness(true)
        .with_familiarity(true)
        .with_temporal_stratification(true)
        .with_confidence_intervals(true)
        .build();

    #[cfg(not(feature = "eval-advanced"))]
    let config = TaskEvalConfig {
        tasks: vec![Task::NER],
        datasets: vec![
            DatasetId::WikiGold,
            DatasetId::TweetNER7,
            DatasetId::BroadTwitterCorpus,
        ],
        backends: vec!["stacked".to_string(), "tplinker".to_string()],
        max_examples: Some(100),
        require_cached: false,
        relation_threshold: 0.5,
        seed: Some(42),
        robustness: true,
        compute_familiarity: true,
        temporal_stratification: true,
        confidence_intervals: true,
    };

    let results = evaluator.evaluate_all(config)?;
    println!(
        "  Total combinations: {}",
        results.summary.total_combinations
    );
    println!("  Successful: {}", results.summary.successful);
    println!(
        "  With temporal stratification: {}",
        results
            .results
            .iter()
            .filter(|r| r.success && r.stratified.is_some())
            .filter(|r| r
                .stratified
                .as_ref()
                .and_then(|s| s.by_temporal_stratum.as_ref())
                .map(|t| !t.is_empty())
                .unwrap_or(false))
            .count()
    );
    println!(
        "  With confidence intervals: {}",
        results
            .results
            .iter()
            .filter(|r| r.success && r.confidence_intervals.is_some())
            .count()
    );
    println!(
        "  With stratified metrics: {}",
        results
            .results
            .iter()
            .filter(|r| r.success && r.stratified.is_some())
            .count()
    );

    // Test 2: Small sample sizes (edge case)
    println!("\nTest 2: Small Sample Size Edge Cases\n");
    for max_examples in [1, 5, 10] {
        let config = TaskEvalConfig {
            tasks: vec![Task::NER],
            datasets: vec![DatasetId::WikiGold],
            backends: vec!["stacked".to_string()],
            max_examples: Some(max_examples),
            require_cached: false,
            relation_threshold: 0.5,
            seed: Some(42),
            robustness: false,
            compute_familiarity: false,
            temporal_stratification: false,
            confidence_intervals: true,
        };

        let results = evaluator.evaluate_all(config)?;
        for result in &results.results {
            if result.success {
                println!(
                    "  N={}: Examples={}, Has CI={}, Has Stratified={}",
                    max_examples,
                    result.num_examples,
                    result.confidence_intervals.is_some(),
                    result.stratified.is_some()
                );
            }
        }
    }

    // Test 3: Different seeds (reproducibility)
    println!("\nTest 3: Reproducibility with Different Seeds\n");
    for seed in [42, 123, 999] {
        let config = TaskEvalConfig {
            tasks: vec![Task::NER],
            datasets: vec![DatasetId::WikiGold],
            backends: vec!["stacked".to_string()],
            max_examples: Some(50),
            require_cached: false,
            relation_threshold: 0.5,
            seed: Some(seed),
            robustness: false,
            compute_familiarity: false,
            temporal_stratification: false,
            confidence_intervals: true,
        };

        let results = evaluator.evaluate_all(config)?;
        for result in &results.results {
            if result.success {
                if let Some(f1) = result.metrics.get("f1") {
                    println!(
                        "  Seed {}: F1 = {:.3}, Examples = {}",
                        seed, f1, result.num_examples
                    );
                }
            }
        }
    }

    // Test 4: Feature combinations
    println!("\nTest 4: Feature Combination Testing\n");
    let feature_combos = vec![
        (false, false, false, false, "No features"),
        (true, false, false, false, "CI only"),
        (false, true, false, false, "Temporal only"),
        (false, false, true, false, "Familiarity only"),
        (true, true, false, false, "CI + Temporal"),
        (true, true, true, true, "All features"),
    ];

    for (ci, temporal, familiarity, robustness, name) in feature_combos {
        let config = TaskEvalConfig {
            tasks: vec![Task::NER],
            datasets: vec![DatasetId::TweetNER7],
            backends: vec!["stacked".to_string()],
            max_examples: Some(50),
            require_cached: false,
            relation_threshold: 0.5,
            seed: Some(42),
            robustness,
            compute_familiarity: familiarity,
            temporal_stratification: temporal,
            confidence_intervals: ci,
        };

        let results = evaluator.evaluate_all(config)?;
        let successful = results.results.iter().filter(|r| r.success).count();
        println!("  {}: {} successful", name, successful);
    }

    // Test 5: Performance timing
    println!("\nTest 5: Performance Analysis\n");
    use std::time::Instant;
    let start = Instant::now();

    let config = TaskEvalConfig {
        tasks: vec![Task::NER],
        datasets: vec![DatasetId::WikiGold, DatasetId::TweetNER7],
        backends: vec!["stacked".to_string(), "tplinker".to_string()],
        max_examples: Some(100),
        require_cached: false,
        relation_threshold: 0.5,
        seed: Some(42),
        robustness: false,
        compute_familiarity: false,
        temporal_stratification: true,
        confidence_intervals: true,
    };

    let results = evaluator.evaluate_all(config)?;
    let elapsed = start.elapsed();

    println!("  Total time: {:.2}s", elapsed.as_secs_f64());
    println!("  Combinations: {}", results.summary.total_combinations);
    println!(
        "  Avg time per combination: {:.2}ms",
        elapsed.as_secs_f64() * 1000.0 / results.summary.total_combinations as f64
    );

    let total_examples: usize = results
        .results
        .iter()
        .filter(|r| r.success)
        .map(|r| r.num_examples)
        .sum();
    println!("  Total examples processed: {}", total_examples);
    if total_examples > 0 {
        println!(
            "  Examples per second: {:.1}",
            total_examples as f64 / elapsed.as_secs_f64()
        );
    }

    // Test 6: Report generation quality
    println!("\nTest 6: Report Generation Quality\n");
    let report = results.to_markdown();
    let report_len = report.len();
    let has_temporal = report.contains("Temporal Stratification");
    let has_ci = report.contains("Confidence Intervals");
    let has_stratified = report.contains("Stratified by Entity Type");

    println!("  Report length: {} characters", report_len);
    println!("  Contains temporal stratification: {}", has_temporal);
    println!("  Contains confidence intervals: {}", has_ci);
    println!("  Contains stratified metrics: {}", has_stratified);

    // Save detailed report
    std::fs::write("stress_test_report.md", &report)?;
    println!("  Report saved to: stress_test_report.md");

    println!("\n=== All Stress Tests Complete ===\n");
    Ok(())
}
