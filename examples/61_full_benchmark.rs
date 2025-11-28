//! Full benchmark: Compare all available backends.
//!
//! Runs evaluation harness with all registered backends and generates HTML report.
//!
//! Run: cargo run --features "eval,onnx,network" --example 61_full_benchmark

use anno::eval::{EvalConfig, EvalHarness};
use std::time::Instant;

#[cfg(feature = "onnx")]
use anno::BertNEROnnx;

fn main() -> anno::Result<()> {
    println!("=== Full Backend Benchmark ===\n");

    // Create harness with quick config for demo
    let config = EvalConfig {
        max_examples_per_dataset: 50, // Limit for speed
        breakdown_by_difficulty: true,
        breakdown_by_domain: true,
        breakdown_by_type: true,
        warmup: true,
        warmup_iterations: 1,
        ..Default::default()
    };

    let mut harness = EvalHarness::new(config)?;

    // Register default backends (Pattern, Statistical, Stacked, Hybrid)
    harness.register_defaults();
    println!("Registered {} default backends", harness.backend_count());

    // Add BERT NER if available
    #[cfg(feature = "onnx")]
    {
        match BertNEROnnx::new("protectai/bert-base-NER-onnx") {
            Ok(bert) => {
                harness.register(
                    "BertNER-ONNX",
                    "BERT-base fine-tuned for NER (ONNX runtime)",
                    Box::new(bert),
                );
                println!("Added BERT NER backend");
            }
            Err(e) => {
                println!("BERT NER not available: {}", e);
            }
        }
    }

    println!("\nRunning evaluation on synthetic data...\n");
    let start = Instant::now();
    let results = harness.run_synthetic()?;
    let elapsed = start.elapsed();

    // Print results
    println!("=== Results ({:.1}s) ===\n", elapsed.as_secs_f64());

    for backend in &results.backends {
        println!(
            "{}: P={:.1}% R={:.1}% F1={:.1}%",
            backend.backend_name,
            backend.precision.mean * 100.0,
            backend.recall.mean * 100.0,
            backend.f1.mean * 100.0
        );
    }

    // Generate and save HTML report
    let html = results.to_html();
    let report_path = "eval_report.html";
    std::fs::write(report_path, &html)?;
    println!("\nHTML report saved to: {}", report_path);

    // Show leaderboard
    println!("\n=== Leaderboard (by F1) ===\n");
    let mut sorted: Vec<_> = results.backends.iter().collect();
    sorted.sort_by(|a, b| {
        b.f1.mean.partial_cmp(&a.f1.mean).unwrap_or(std::cmp::Ordering::Equal)
    });

    for (i, backend) in sorted.iter().enumerate() {
        println!(
            "{}. {} - F1: {:.1}%{}",
            i + 1,
            backend.backend_name,
            backend.f1.mean * 100.0,
            if i == 0 { " <- Best" } else { "" }
        );
    }

    Ok(())
}

