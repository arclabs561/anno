//! Slow benchmarks that test against large datasets.
//!
//! These tests are marked with `#[ignore]` and should be run with:
//! ```bash
//! cargo test --test slow_benchmarks -- --ignored
//! ```

use anno::eval::benchmark::{generate_large_dataset, BenchmarkStats, EdgeCaseType};
use anno::eval::synthetic::{all_datasets, AnnotatedExample};
use anno::eval::{evaluate_ner_model, GoldEntity};
use anno::{Model, PatternNER};
use std::time::Instant;

/// Convert AnnotatedExample to the format expected by evaluate_ner_model.
/// Filters out empty texts which would cause evaluation to fail.
fn to_test_cases(examples: &[AnnotatedExample]) -> Vec<(String, Vec<GoldEntity>)> {
    examples
        .iter()
        .filter(|ex| !ex.text.is_empty())
        .map(|ex| (ex.text.clone(), ex.entities.clone()))
        .collect()
}

/// Evaluate model and print detailed stats
fn evaluate_with_stats(model: &dyn Model, test_cases: &[(String, Vec<GoldEntity>)], name: &str) {
    let start = Instant::now();
    let results = evaluate_ner_model(model, test_cases).expect("Evaluation failed");
    let elapsed = start.elapsed();

    println!("\n=== {} ===", name);
    println!("  F1:  {:.1}%  Precision:  {:.1}%  Recall:  {:.1}%",
        results.f1 * 100.0, results.precision * 100.0, results.recall * 100.0);
    println!("  Found: {} / Expected: {}  ({:.2}s)",
        results.found, results.expected, elapsed.as_secs_f64());
    println!("  Throughput: {:.0} tok/sec", results.tokens_per_second);

    // Per-type breakdown
    if !results.per_type.is_empty() {
        println!("\n  Per-Type:");
        let mut types: Vec<_> = results.per_type.iter().collect();
        types.sort_by(|a, b| b.1.f1.partial_cmp(&a.1.f1).unwrap());
        for (type_name, metrics) in types {
            let status = if metrics.f1 > 0.9 { "+" }
                else if metrics.f1 > 0.5 { "~" }
                else { "-" };
            println!("    {} {:15} F1={:5.1}% ({}/{})",
                status, type_name, metrics.f1 * 100.0, metrics.correct, metrics.expected);
        }
    }
}

#[test]
#[ignore]
fn test_large_benchmark_500_examples() {
    println!("\n=== Large Benchmark (500 examples) ===");
    let dataset = generate_large_dataset(500, EdgeCaseType::All);
    let stats = BenchmarkStats::from_dataset(&dataset);
    println!("Generated {} examples with {} entities", stats.total_examples, stats.total_entities);
    println!("Examples with no entities: {}", stats.examples_with_no_entities);

    let test_cases = to_test_cases(&dataset);
    let pattern_ner = PatternNER::new();
    evaluate_with_stats(&pattern_ner, &test_cases, "PatternNER on 500 hard examples");

    // PatternNER should perform ~50% or less on mixed hard examples
    let results = evaluate_ner_model(&pattern_ner, &test_cases).unwrap();
    assert!(results.f1 < 0.8, "Expected F1 < 80% on hard examples, got {:.1}%", results.f1 * 100.0);
}

#[test]
#[ignore]
fn test_large_benchmark_1000_examples() {
    println!("\n=== Large Benchmark (1000 examples) ===");
    let dataset = generate_large_dataset(1000, EdgeCaseType::All);
    let stats = BenchmarkStats::from_dataset(&dataset);
    println!("Generated {} examples with {} entities", stats.total_examples, stats.total_entities);

    let test_cases = to_test_cases(&dataset);
    let pattern_ner = PatternNER::new();
    evaluate_with_stats(&pattern_ner, &test_cases, "PatternNER on 1000 hard examples");
}

#[test]
#[ignore]
fn test_ambiguous_cases_only() {
    println!("\n=== Ambiguous Cases (500 examples) ===");
    let dataset = generate_large_dataset(500, EdgeCaseType::Ambiguous);
    let test_cases = to_test_cases(&dataset);
    let pattern_ner = PatternNER::new();
    evaluate_with_stats(&pattern_ner, &test_cases, "PatternNER on ambiguous examples");

    // Ambiguous cases are genuinely hard - models should struggle
    let _results = evaluate_ner_model(&pattern_ner, &test_cases).unwrap();
    println!("\nNote: Low scores expected on ambiguous cases (Apple company vs fruit, etc.)");
}

#[test]
#[ignore]
fn test_unicode_edge_cases() {
    println!("\n=== Unicode Edge Cases (500 examples) ===");
    let dataset = generate_large_dataset(500, EdgeCaseType::Unicode);
    let stats = BenchmarkStats::from_dataset(&dataset);
    println!("Generated {} examples with {} entities", stats.total_examples, stats.total_entities);

    let test_cases = to_test_cases(&dataset);
    let pattern_ner = PatternNER::new();
    evaluate_with_stats(&pattern_ner, &test_cases, "PatternNER on Unicode examples");
}

#[test]
#[ignore]
fn test_numeric_edge_cases() {
    println!("\n=== Numeric Edge Cases (500 examples) ===");
    let dataset = generate_large_dataset(500, EdgeCaseType::NumericEdge);
    let stats = BenchmarkStats::from_dataset(&dataset);
    println!("Generated {} examples with {} entities", stats.total_examples, stats.total_entities);

    let test_cases = to_test_cases(&dataset);
    let pattern_ner = PatternNER::new();
    evaluate_with_stats(&pattern_ner, &test_cases, "PatternNER on numeric edge cases");

    // PatternNER should do well on numeric patterns
    let results = evaluate_ner_model(&pattern_ner, &test_cases).unwrap();
    assert!(results.f1 > 0.3, "Expected F1 > 30% on numeric edge cases, got {:.1}%", results.f1 * 100.0);
}

#[test]
#[ignore]
fn test_boundary_edge_cases() {
    println!("\n=== Boundary Edge Cases (500 examples) ===");
    let dataset = generate_large_dataset(500, EdgeCaseType::Boundary);
    let test_cases = to_test_cases(&dataset);
    let pattern_ner = PatternNER::new();
    evaluate_with_stats(&pattern_ner, &test_cases, "PatternNER on boundary cases");
}

#[test]
#[ignore]
fn test_dense_text() {
    println!("\n=== Dense Text (500 examples) ===");
    let dataset = generate_large_dataset(500, EdgeCaseType::Dense);
    let stats = BenchmarkStats::from_dataset(&dataset);
    println!("Generated {} examples with {} entities ({:.1} avg/ex)",
        stats.total_examples, stats.total_entities, stats.avg_entities_per_example);

    let test_cases = to_test_cases(&dataset);
    let pattern_ner = PatternNER::new();
    evaluate_with_stats(&pattern_ner, &test_cases, "PatternNER on dense text");
}

#[test]
#[ignore]
fn test_comprehensive_slow_benchmark() {
    println!("\n========================================");
    println!("  COMPREHENSIVE SLOW BENCHMARK");
    println!("========================================");

    // Test on full synthetic dataset first
    let synthetic = all_datasets();
    let test_cases = to_test_cases(&synthetic);
    let pattern_ner = PatternNER::new();

    println!("\n--- Synthetic Dataset ({} examples) ---", synthetic.len());
    evaluate_with_stats(&pattern_ner, &test_cases, "PatternNER");

    // Then test on each edge case type
    for edge_type in [
        EdgeCaseType::Ambiguous,
        EdgeCaseType::Unicode,
        EdgeCaseType::Dense,
        EdgeCaseType::Sparse,
        EdgeCaseType::Nested,
        EdgeCaseType::Casing,
        EdgeCaseType::Boundary,
        EdgeCaseType::MultiWord,
        EdgeCaseType::NumericEdge,
        EdgeCaseType::Jargon,
    ] {
        let dataset = generate_large_dataset(100, edge_type);
        let test_cases = to_test_cases(&dataset);
        evaluate_with_stats(&pattern_ner, &test_cases, &format!("{:?}", edge_type));
    }

    // Large combined benchmark
    println!("\n--- Large Combined Benchmark (5000 examples) ---");
    let large_dataset = generate_large_dataset(5000, EdgeCaseType::All);
    let stats = BenchmarkStats::from_dataset(&large_dataset);
    println!("Total: {} examples, {} entities, {} negative",
        stats.total_examples, stats.total_entities, stats.examples_with_no_entities);

    let test_cases = to_test_cases(&large_dataset);
    evaluate_with_stats(&pattern_ner, &test_cases, "PatternNER on 5000 hard examples");

    println!("\n========================================");
    println!("  BENCHMARK COMPLETE");
    println!("========================================");
}

