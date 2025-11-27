//! Quality benchmark comparing NER backends on synthetic datasets.
//!
//! Run with:
//!   cargo run --example quality_bench                    # Zero-dep backends only
//!   cargo run --example quality_bench --features onnx    # Include BERT ONNX
//!
//! Shows:
//! - Per-backend quality metrics (F1, Precision, Recall)
//! - Per-difficulty breakdown (Easy/Medium/Hard/Adversarial)
//! - Per-domain breakdown (News/Financial/Technical/etc.)
//! - Variance across domains using MetricWithVariance

use anno::eval::harness::{EvalConfig, EvalHarness};
use anno::eval::synthetic::dataset_stats;
use anno::eval::MetricWithVariance;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== NER Quality Benchmark ===\n");

    // Dataset overview
    let stats = dataset_stats();
    println!(
        "Dataset: {} examples, {} entities",
        stats.total_examples, stats.total_entities
    );
    println!("Domains: {:?}", stats.domains);
    println!("Difficulties: {:?}\n", stats.difficulties);

    // Configure evaluation
    let config = EvalConfig {
        breakdown_by_difficulty: true,
        breakdown_by_domain: true,
        warmup: true,
        warmup_iterations: 3,
        ..EvalConfig::default()
    };

    // Create harness with custom config and default backends
    let harness = EvalHarness::with_config(config)?;

    // Print registered backends
    println!("Backends: {}", harness.backend_count());
    for (name, desc, _) in harness.registry().iter() {
        println!("  - {}: {}", name, desc);
    }
    println!();

    // Run evaluation
    println!("Evaluating on synthetic data...\n");
    let results = harness.run_synthetic()?;

    // === Overall Results ===
    println!("=== Overall Results ===\n");
    println!(
        "{:<16} {:>8} {:>10} {:>8} {:>10} {:>12}",
        "Backend", "F1", "Precision", "Recall", "Found/Exp", "Time"
    );
    println!("{}", "-".repeat(70));

    for backend in &results.backends {
        println!(
            "{:<16} {:>7.1}% {:>9.1}% {:>7.1}% {:>5}/{:<5} {:>10.1}ms",
            backend.backend_name,
            backend.f1.mean * 100.0,
            backend.precision.mean * 100.0,
            backend.recall.mean * 100.0,
            backend.total_found,
            backend.total_expected,
            backend.total_duration_ms
        );
    }

    // === Per-type metrics for best backend ===
    // Find StackedNER and show per-type breakdown from its per_dataset results
    if let Some(stacked) = results.backends.iter().find(|b| b.backend_name == "StackedNER") {
        if let Some(dataset_result) = stacked.per_dataset.first() {
            println!("\n=== Per-Type Metrics (StackedNER) ===\n");
            print_per_type_metrics(&dataset_result.per_type);
        }
    }

    // === Breakdown by difficulty with variance ===
    if let Some(by_difficulty) = &results.by_difficulty {
        println!("\n=== Results by Difficulty ===\n");

        // Pick StackedNER for detailed analysis
        let mut difficulty_f1s: Vec<f64> = Vec::new();

        println!(
            "{:<14} {:>8} {:>10} {:>8} {:>8}",
            "Difficulty", "F1", "Precision", "Recall", "Count"
        );
        println!("{}", "-".repeat(52));

        for difficulty in &["Easy", "Medium", "Hard", "Adversarial"] {
            if let Some(results_list) = by_difficulty.get(*difficulty) {
                // Find StackedNER result
                if let Some(result) = results_list.iter().find(|r| r.backend_name == "StackedNER") {
                    println!(
                        "{:<14} {:>7.1}% {:>9.1}% {:>7.1}% {:>8}",
                        difficulty,
                        result.f1 * 100.0,
                        result.precision * 100.0,
                        result.recall * 100.0,
                        result.num_examples
                    );
                    difficulty_f1s.push(result.f1);
                }
            }
        }

        // Show variance across difficulties
        if !difficulty_f1s.is_empty() {
            let diff_variance = MetricWithVariance::from_samples(&difficulty_f1s);
            println!("\nVariance across difficulties: {}", diff_variance);
            println!(
                "  Range: {:.1}% - {:.1}%",
                diff_variance.min * 100.0,
                diff_variance.max * 100.0
            );
        }
    }

    // === Breakdown by domain with variance ===
    if let Some(by_domain) = &results.by_domain {
        println!("\n=== Results by Domain (StackedNER) ===\n");

        let mut domain_f1s: Vec<f64> = Vec::new();

        println!(
            "{:<16} {:>8} {:>10} {:>8} {:>8}",
            "Domain", "F1", "Precision", "Recall", "Count"
        );
        println!("{}", "-".repeat(54));

        // Sort domains by F1 score for readability
        let mut domain_results: Vec<_> = by_domain
            .iter()
            .filter_map(|(domain, results_list)| {
                results_list
                    .iter()
                    .find(|r| r.backend_name == "StackedNER")
                    .map(|r| (domain, r))
            })
            .collect();
        domain_results.sort_by(|a, b| b.1.f1.partial_cmp(&a.1.f1).unwrap_or(std::cmp::Ordering::Equal));

        for (domain, result) in &domain_results {
            println!(
                "{:<16} {:>7.1}% {:>9.1}% {:>7.1}% {:>8}",
                domain,
                result.f1 * 100.0,
                result.precision * 100.0,
                result.recall * 100.0,
                result.num_examples
            );
            domain_f1s.push(result.f1);
        }

        // Show variance across domains
        if !domain_f1s.is_empty() {
            let domain_variance = MetricWithVariance::from_samples(&domain_f1s);
            println!("\nVariance across domains: {}", domain_variance);
            println!(
                "  Range: {:.1}% - {:.1}%",
                domain_variance.min * 100.0,
                domain_variance.max * 100.0
            );
            println!(
                "  CV (coefficient of variation): {:.1}%",
                domain_variance.coefficient_of_variation() * 100.0
            );
        }
    }

    // === Entity type distribution ===
    println!("\n=== Entity Type Distribution in Dataset ===\n");
    println!("{:<20} {:>8} {:>10}", "Type", "Count", "Percent");
    println!("{}", "-".repeat(40));

    let mut sorted_types: Vec<_> = results.dataset_stats.entity_type_distribution.iter().collect();
    sorted_types.sort_by(|a, b| b.1.cmp(a.1));

    let total: usize = sorted_types.iter().map(|(_, c)| **c).sum();
    for (type_name, count) in sorted_types {
        println!(
            "{:<20} {:>8} {:>9.1}%",
            type_name,
            count,
            (*count as f64 / total as f64) * 100.0
        );
    }

    // === Summary ===
    println!("\n=== Summary ===\n");

    // Find best backend
    if let Some(best) = results.backends.iter().max_by(|a, b| {
        a.f1.mean.partial_cmp(&b.f1.mean).unwrap_or(std::cmp::Ordering::Equal)
    }) {
        println!("Best backend: {} (F1: {:.1}%)", best.backend_name, best.f1.mean * 100.0);
    }

    println!("\nKey observations:");
    println!("  - PatternNER: High precision on DATE/MONEY/PERCENT/EMAIL/URL/PHONE");
    println!("  - StatisticalNER: Baseline for PER/ORG/LOC (heuristic-based)");
    println!("  - StackedNER: Best zero-dependency option");

    #[cfg(not(feature = "onnx"))]
    {
        println!("\nTo test ML backends with higher accuracy:");
        println!("  cargo run --example quality_bench --features onnx");
    }

    #[cfg(feature = "onnx")]
    {
        println!("\nML backend (BertNEROnnx) provides significant improvement on named entities.");
    }

    // Optionally save HTML report
    if std::env::var("SAVE_HTML").is_ok() {
        let html = results.to_html();
        std::fs::write("eval_results.html", &html)?;
        println!("\nHTML report saved to eval_results.html");
    }

    Ok(())
}

fn print_per_type_metrics(per_type: &HashMap<String, anno::eval::TypeMetrics>) {
    let mut sorted_types: Vec<_> = per_type.iter().collect();
    sorted_types.sort_by(|a, b| b.1.f1.partial_cmp(&a.1.f1).unwrap_or(std::cmp::Ordering::Equal));

    println!(
        "{:<3} {:<14} {:>8} {:>10} {:>8} {:>12}",
        "", "Type", "F1", "Precision", "Recall", "Correct/Exp"
    );
    println!("{}", "-".repeat(60));

    for (entity_type, metrics) in sorted_types {
        let status = if metrics.f1 > 0.9 {
            "[+]" // Excellent
        } else if metrics.f1 > 0.7 {
            "[~]" // Good
        } else if metrics.f1 > 0.3 {
            "[?]" // Moderate
        } else if metrics.expected > 0 {
            "[-]" // Poor
        } else {
            "   " // N/A
        };

        println!(
            "{:<3} {:<14} {:>7.1}% {:>9.1}% {:>7.1}% {:>6}/{:<5}",
            status,
            entity_type,
            metrics.f1 * 100.0,
            metrics.precision * 100.0,
            metrics.recall * 100.0,
            metrics.correct,
            metrics.expected
        );
    }
}
