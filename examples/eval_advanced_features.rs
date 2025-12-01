//! Advanced evaluation features demonstration.
//!
//! This example demonstrates:
//! - Temporal stratification across different datasets
//! - Confidence intervals with different sample sizes
//! - Stratified metrics by entity type
//! - Familiarity computation for zero-shot backends
//! - Robustness testing
//!
//! Run with:
//!   cargo run --example eval_advanced_features --features eval-advanced

use anno::eval::loader::DatasetId;
use anno::eval::task_evaluator::{TaskEvalConfig, TaskEvaluator};
use anno::eval::task_mapping::Task;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Advanced Evaluation Features Demonstration ===\n");

    let evaluator = TaskEvaluator::new()?;

    // Test 1: Temporal stratification with different sample sizes
    println!("Test 1: Temporal Stratification Analysis\n");
    for max_examples in [25, 50, 100] {
        let config = TaskEvalConfig {
            tasks: vec![Task::NER],
            datasets: vec![DatasetId::TweetNER7, DatasetId::BroadTwitterCorpus],
            backends: vec!["stacked".to_string()],
            max_examples: Some(max_examples),
            require_cached: false,
            relation_threshold: 0.5,
            seed: Some(42),
            robustness: false,
            compute_familiarity: false,
            temporal_stratification: true,
            confidence_intervals: true,
        };

        let results = evaluator.evaluate_all(config)?;
        let successful = results.results.iter().filter(|r| r.success).count();
        let with_temporal = results
            .results
            .iter()
            .filter(|r| r.success && r.stratified.is_some())
            .filter(|r| {
                r.stratified
                    .as_ref()
                    .and_then(|s| s.by_temporal_stratum.as_ref())
                    .map(|t| !t.is_empty())
                    .unwrap_or(false)
            })
            .count();

        println!(
            "  Max examples: {} -> {} successful, {} with temporal stratification",
            max_examples, successful, with_temporal
        );
    }

    // Test 2: Confidence intervals with different configurations
    println!("\nTest 2: Confidence Interval Analysis\n");
    let config = TaskEvalConfig {
        tasks: vec![Task::NER],
        datasets: vec![DatasetId::WikiGold],
        backends: vec!["stacked".to_string(), "tplinker".to_string()],
        max_examples: Some(100),
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
        if result.success && result.confidence_intervals.is_some() {
            let ci = result.confidence_intervals.as_ref().unwrap();
            let f1_width = ci.f1_ci.1 - ci.f1_ci.0;
            println!(
                "  {} on {}: F1 CI width = {:.3} [{:.3}, {:.3}]",
                result.backend, result.dataset, f1_width, ci.f1_ci.0, ci.f1_ci.1
            );
        }
    }

    // Test 3: Stratified metrics by entity type
    println!("\nTest 3: Entity Type Stratification\n");
    let config = TaskEvalConfig {
        tasks: vec![Task::NER],
        datasets: vec![DatasetId::WikiGold],
        backends: vec!["stacked".to_string()],
        max_examples: Some(100),
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
        if result.success && result.stratified.is_some() {
            let stratified = result.stratified.as_ref().unwrap();
            println!(
                "  {} on {}: {} entity types analyzed",
                result.backend,
                result.dataset,
                stratified.by_entity_type.len()
            );

            // Show top 3 entity types by F1
            let mut types: Vec<_> = stratified.by_entity_type.iter().collect();
            types.sort_by(|a, b| b.1.mean.partial_cmp(&a.1.mean).unwrap());
            for (type_name, metric) in types.iter().take(3) {
                println!(
                    "    {}: F1 = {:.3} (CI: [{:.3}, {:.3}], N={})",
                    type_name, metric.mean, metric.ci_95.0, metric.ci_95.1, metric.n
                );
            }
        }
    }

    // Test 4: Compare temporal stratification across datasets
    println!("\nTest 4: Temporal Stratification Comparison\n");
    let config = TaskEvalConfig {
        tasks: vec![Task::NER],
        datasets: vec![DatasetId::TweetNER7, DatasetId::BroadTwitterCorpus],
        backends: vec!["stacked".to_string()],
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
    for result in &results.results {
        if result.success && result.stratified.is_some() {
            if let Some(temporal) = result
                .stratified
                .as_ref()
                .and_then(|s| s.by_temporal_stratum.as_ref())
            {
                println!("  {} on {}:", result.backend, result.dataset);
                for (stratum, metric) in temporal {
                    println!(
                        "    {}: F1 = {:.3} (CI: [{:.3}, {:.3}], N={})",
                        stratum, metric.mean, metric.ci_95.0, metric.ci_95.1, metric.n
                    );
                }

                // Calculate temporal drift (difference between pre and post cutoff)
                if let (Some(pre), Some(post)) =
                    (temporal.get("pre_cutoff"), temporal.get("post_cutoff"))
                {
                    let drift = pre.mean - post.mean;
                    println!(
                        "    Temporal drift: {:.3} (pre_cutoff - post_cutoff)",
                        drift
                    );
                }
            }
        }
    }

    // Test 5: Full feature demonstration
    println!("\nTest 5: Full Feature Demonstration\n");
    let config = TaskEvalConfig {
        tasks: vec![Task::NER],
        datasets: vec![DatasetId::TweetNER7],
        backends: vec!["stacked".to_string()],
        max_examples: Some(50),
        require_cached: false,
        relation_threshold: 0.5,
        seed: Some(42),
        robustness: true,
        compute_familiarity: true,
        temporal_stratification: true,
        confidence_intervals: true,
    };

    let results = evaluator.evaluate_all(config)?;
    for result in &results.results {
        if result.success {
            println!("  {} on {}:", result.backend, result.dataset);
            println!("    Examples: {}", result.num_examples);

            if result.confidence_intervals.is_some() {
                println!("    ✓ Confidence intervals computed");
            }
            if result.stratified.is_some() {
                let s = result.stratified.as_ref().unwrap();
                println!("    ✓ Stratified metrics: {} types", s.by_entity_type.len());
                if s.by_temporal_stratum.is_some() {
                    println!(
                        "    ✓ Temporal stratification: {} strata",
                        s.by_temporal_stratum.as_ref().unwrap().len()
                    );
                }
            }
            if result.label_shift.is_some() {
                println!("    ✓ Familiarity computed");
            }
            #[cfg(feature = "eval-advanced")]
            if result.robustness.is_some() {
                println!("    ✓ Robustness testing completed");
            }
            if result.kb_version.is_some() {
                println!("    ✓ KB version: {}", result.kb_version.as_ref().unwrap());
            }
        }
    }

    println!("\n=== All Tests Complete ===\n");
    Ok(())
}
