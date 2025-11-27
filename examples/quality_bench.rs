//! Quality benchmark comparing NER backends on synthetic datasets.
//!
//! Run with: `cargo run --example quality_bench`

use anno::eval::synthetic::{all_datasets, dataset_stats, Difficulty, Domain};
use anno::eval::{evaluate_ner_model, NEREvaluationResults};
use anno::{HybridNER, PatternNER};
use std::collections::HashMap;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== NER Quality Benchmark ===\n");

    let stats = dataset_stats();
    println!("Dataset: {} examples, {} entities", stats.total_examples, stats.total_entities);
    println!("Domains: {:?}", stats.domains);
    println!("Difficulties: {:?}", stats.difficulties);

    let all_examples = all_datasets();
    
    // Filter out empty texts
    let test_cases: Vec<_> = all_examples
        .iter()
        .filter(|ex| !ex.text.is_empty())
        .map(|ex| (ex.text.clone(), ex.entities.clone()))
        .collect();

    println!("\nEvaluating {} test cases...\n", test_cases.len());

    // === PatternNER ===
    let pattern_ner = PatternNER::new();
    let start = Instant::now();
    let pattern_results = evaluate_ner_model(&pattern_ner, &test_cases)?;
    let pattern_time = start.elapsed();
    
    print_results("PatternNER", &pattern_results, pattern_time);
    print_per_type_metrics(&pattern_results);

    // === HybridNER (pattern only) ===
    let hybrid_ner = HybridNER::pattern_only();
    let start = Instant::now();
    let hybrid_results = evaluate_ner_model(&hybrid_ner, &test_cases)?;
    let hybrid_time = start.elapsed();
    
    print_results("HybridNER (pattern-only)", &hybrid_results, hybrid_time);

    // === Breakdown by difficulty ===
    println!("\n=== Results by Difficulty ===");
    for difficulty in [Difficulty::Easy, Difficulty::Medium, Difficulty::Hard, Difficulty::Adversarial] {
        let subset: Vec<_> = all_examples
            .iter()
            .filter(|ex| ex.difficulty == difficulty && !ex.text.is_empty())
            .map(|ex| (ex.text.clone(), ex.entities.clone()))
            .collect();
        
        if subset.is_empty() {
            continue;
        }
        
        if let Ok(results) = evaluate_ner_model(&pattern_ner, &subset) {
            println!(
                "{:12} F1={:5.1}% P={:5.1}% R={:5.1}% (n={})",
                format!("{:?}", difficulty),
                results.f1 * 100.0,
                results.precision * 100.0,
                results.recall * 100.0,
                subset.len()
            );
        }
    }

    // === Breakdown by domain ===
    println!("\n=== Results by Domain ===");
    let domains = [
        Domain::News, Domain::Financial, Domain::Technical, Domain::Sports,
        Domain::Entertainment, Domain::Politics, Domain::Ecommerce, Domain::Travel,
        Domain::Weather, Domain::Academic, Domain::Historical, Domain::Food,
        Domain::RealEstate, Domain::Conversational, Domain::SocialMedia,
        Domain::Biomedical, Domain::Legal, Domain::Scientific,
    ];
    
    for domain in domains {
        let subset: Vec<_> = all_examples
            .iter()
            .filter(|ex| ex.domain == domain && !ex.text.is_empty())
            .map(|ex| (ex.text.clone(), ex.entities.clone()))
            .collect();
        
        if subset.is_empty() {
            continue;
        }
        
        if let Ok(results) = evaluate_ner_model(&pattern_ner, &subset) {
            println!(
                "{:14} F1={:5.1}% P={:5.1}% R={:5.1}% (n={})",
                format!("{:?}", domain),
                results.f1 * 100.0,
                results.precision * 100.0,
                results.recall * 100.0,
                subset.len()
            );
        }
    }

    // === Entity type coverage ===
    println!("\n=== Entity Type Distribution in Dataset ===");
    let mut type_counts: HashMap<String, usize> = HashMap::new();
    for ex in &all_examples {
        for entity in &ex.entities {
            let type_name = format!("{:?}", entity.entity_type);
            *type_counts.entry(type_name).or_insert(0) += 1;
        }
    }
    let mut sorted_types: Vec<_> = type_counts.into_iter().collect();
    sorted_types.sort_by(|a, b| b.1.cmp(&a.1));
    for (type_name, count) in sorted_types {
        println!("  {:<20} {}", type_name, count);
    }

    // === Summary ===
    println!("\n=== Summary ===");
    println!("PatternNER excels at structured entities (DATE, MONEY, PERCENT, EMAIL, URL, PHONE)");
    println!("For PER/ORG/LOC, enable ONNX/Candle backends with: --features onnx");
    println!("\nTo run with ML backend:");
    println!("  cargo run --example quality_bench --features onnx");

    Ok(())
}

fn print_results(name: &str, results: &NEREvaluationResults, elapsed: std::time::Duration) {
    println!("=== {} ===", name);
    println!(
        "  F1: {:5.1}%  Precision: {:5.1}%  Recall: {:5.1}%",
        results.f1 * 100.0,
        results.precision * 100.0,
        results.recall * 100.0
    );
    println!(
        "  Found: {} / Expected: {}  ({:.0} tok/sec, {:.2}ms total)",
        results.found,
        results.expected,
        results.tokens_per_second,
        elapsed.as_secs_f64() * 1000.0
    );
}

fn print_per_type_metrics(results: &NEREvaluationResults) {
    println!("\n  Per-Type Metrics:");
    let mut sorted_types: Vec<_> = results.per_type.iter().collect();
    sorted_types.sort_by_key(|(name, _)| *name);
    
    for (entity_type, metrics) in sorted_types {
        let status = if metrics.f1 > 0.9 {
            "+"  // Good
        } else if metrics.f1 > 0.5 {
            "~"  // Moderate
        } else if metrics.expected > 0 {
            "-"  // Poor
        } else {
            " "  // N/A
        };
        
        println!(
            "  {} {:<12} F1={:5.1}% P={:5.1}% R={:5.1}% ({}/{})",
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
