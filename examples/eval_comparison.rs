//! Compare evaluation results across different configurations.
//!
//! Demonstrates:
//! - Comparing metrics with and without advanced features
//! - Analyzing impact of sample size on confidence intervals
//! - Comparing temporal stratification across datasets
//!
//! Run with:
//!   cargo run --example eval_comparison --features eval-advanced

use anno::eval::loader::DatasetId;
use anno::eval::task_evaluator::{TaskEvalConfig, TaskEvaluator};
use anno::eval::task_mapping::Task;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Evaluation Comparison Analysis ===\n");

    let evaluator = TaskEvaluator::new()?;

    // Comparison 1: With vs Without Confidence Intervals
    println!("Comparison 1: Impact of Confidence Intervals\n");

    let config_without_ci = TaskEvalConfig {
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
        confidence_intervals: false,
        custom_coref_resolver: None,
    };

    let config_with_ci = TaskEvalConfig {
        confidence_intervals: true,
        custom_coref_resolver: None,
        ..config_without_ci.clone()
    };

    let results_without = evaluator.evaluate_all(config_without_ci)?;
    let results_with = evaluator.evaluate_all(config_with_ci)?;

    for (result_without, result_with) in results_without
        .results
        .iter()
        .zip(results_with.results.iter())
    {
        if result_without.success && result_with.success {
            let f1_without = result_without.metrics.get("f1").copied().unwrap_or(0.0);
            let f1_with = result_with.metrics.get("f1").copied().unwrap_or(0.0);

            println!(
                "  {} on {}:",
                result_without.backend, result_without.dataset
            );
            println!("    F1 without CI: {:.3}", f1_without);
            println!("    F1 with CI: {:.3}", f1_with);

            if let Some(ci) = result_with.confidence_intervals.as_ref() {
                let ci_width = ci.f1_ci.1 - ci.f1_ci.0;
                println!(
                    "    CI width: {:.3} [{:.3}, {:.3}]",
                    ci_width, ci.f1_ci.0, ci.f1_ci.1
                );
            }
        }
    }

    // Comparison 2: Sample Size Impact on CI Width
    println!("\nComparison 2: Sample Size vs CI Width\n");
    for max_examples in [25, 50, 100, 200] {
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
            custom_coref_resolver: None,
        };

        let results = evaluator.evaluate_all(config)?;
        for result in &results.results {
            if result.success {
                if let Some(ci) = result.confidence_intervals.as_ref() {
                    let ci_width = ci.f1_ci.1 - ci.f1_ci.0;
                    println!(
                        "  N={}: CI width = {:.3}, Examples = {}",
                        max_examples, ci_width, result.num_examples
                    );
                }
            }
        }
    }

    // Comparison 3: Temporal Stratification Across Datasets
    println!("\nComparison 3: Temporal Stratification Comparison\n");
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
        custom_coref_resolver: None,
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

                if let (Some(pre), Some(post)) =
                    (temporal.get("pre_cutoff"), temporal.get("post_cutoff"))
                {
                    let drift = pre.mean - post.mean;
                    let drift_pct = (drift / pre.mean.max(0.001)) * 100.0;
                    println!(
                        "    Pre-cutoff F1: {:.3} (CI: [{:.3}, {:.3}])",
                        pre.mean, pre.ci_95.0, pre.ci_95.1
                    );
                    println!(
                        "    Post-cutoff F1: {:.3} (CI: [{:.3}, {:.3}])",
                        post.mean, post.ci_95.0, post.ci_95.1
                    );
                    println!("    Temporal drift: {:.3} ({:.1}%)", drift, drift_pct);
                }
            }
        }
    }

    // Comparison 4: Entity Type Performance Ranking
    println!("\nComparison 4: Entity Type Performance Ranking\n");
    let config = TaskEvalConfig {
        tasks: vec![Task::NER],
        datasets: vec![DatasetId::WikiGold, DatasetId::TweetNER7],
        backends: vec!["stacked".to_string()],
        max_examples: Some(100),
        require_cached: false,
        relation_threshold: 0.5,
        seed: Some(42),
        robustness: false,
        compute_familiarity: false,
        temporal_stratification: false,
        confidence_intervals: true,
        custom_coref_resolver: None,
    };

    let results = evaluator.evaluate_all(config)?;
    for result in &results.results {
        if result.success && result.stratified.is_some() {
            let stratified = result.stratified.as_ref().unwrap();
            println!("  {} on {}:", result.backend, result.dataset);

            let mut types: Vec<_> = stratified.by_entity_type.iter().collect();
            types.sort_by(|a, b| b.1.mean.partial_cmp(&a.1.mean).unwrap());

            println!("    Top 3 entity types:");
            for (idx, (type_name, metric)) in types.iter().take(3).enumerate() {
                println!(
                    "      {}. {}: F1 = {:.3} (CI: [{:.3}, {:.3}], N={})",
                    idx + 1,
                    type_name,
                    metric.mean,
                    metric.ci_95.0,
                    metric.ci_95.1,
                    metric.n
                );
            }

            if types.len() > 3 {
                println!("    Bottom entity type:");
                if let Some((type_name, metric)) = types.last() {
                    println!(
                        "      {}. {}: F1 = {:.3} (CI: [{:.3}, {:.3}], N={})",
                        types.len(),
                        type_name,
                        metric.mean,
                        metric.ci_95.0,
                        metric.ci_95.1,
                        metric.n
                    );
                }
            }
        }
    }

    println!("\n=== All Comparisons Complete ===\n");
    Ok(())
}
