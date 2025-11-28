//! Comprehensive NER Benchmark
//!
//! Benchmarks ALL available backends against ALL datasets.
//!
//! ## Running
//!
//! ```bash
//! # Pattern-only (no ML)
//! cargo run --example comprehensive_bench
//!
//! # With ONNX backends (BERT, GLiNER manual)
//! cargo run --example comprehensive_bench --features onnx
//!
//! # With gline-rs GLiNER
//! cargo run --example comprehensive_bench --features gliner
//!
//! # Everything
//! cargo run --example comprehensive_bench --features "onnx,gliner"
//! ```

use anno::eval::benchmark::{generate_large_dataset, EdgeCaseType};
use anno::eval::synthetic::all_datasets;
use anno::{Entity, Model, NERExtractor, PatternNER};
use std::time::{Duration, Instant};

// ============================================================================
// Backend Registry
// ============================================================================

/// Wrapper to make any NER backend work with evaluation
struct BackendWrapper {
    name: &'static str,
    description: &'static str,
    extractor: Box<dyn Fn(&str) -> Vec<Entity> + Send + Sync>,
}

impl BackendWrapper {
    fn extract(&self, text: &str) -> Vec<Entity> {
        (self.extractor)(text)
    }
}

fn create_backends() -> Vec<BackendWrapper> {
    let mut backends = Vec::new();

    // 1. PatternNER (always available)
    backends.push(BackendWrapper {
        name: "pattern",
        description: "Pattern-based (regex)",
        extractor: Box::new(|text| {
            let ner = PatternNER::new();
            ner.extract_entities(text, None).unwrap_or_default()
        }),
    });

    // 2. NERExtractor::best_available
    backends.push(BackendWrapper {
        name: "best_available",
        description: "Best available backend",
        extractor: Box::new(|text| {
            let extractor = NERExtractor::best_available();
            extractor.extract(text, None).unwrap_or_default()
        }),
    });

    // 3. Hybrid mode
    backends.push(BackendWrapper {
        name: "hybrid",
        description: "Hybrid (ML + patterns)",
        extractor: Box::new(|text| {
            let extractor = NERExtractor::best_available();
            extractor.extract_hybrid(text, None).unwrap_or_default()
        }),
    });

    // 4. BERT ONNX (if onnx feature enabled)
    #[cfg(feature = "onnx")]
    {
        if let Ok(bert) = anno::BertNEROnnx::new(anno::DEFAULT_BERT_ONNX_MODEL) {
            let bert = std::sync::Arc::new(bert);
            backends.push(BackendWrapper {
                name: "bert_onnx",
                description: "BERT NER (ONNX)",
                extractor: Box::new(move |text| {
                    bert.extract_entities(text, None).unwrap_or_default()
                }),
            });
        }
    }

    // 5. GLiNER via ONNX (manual, if onnx feature enabled)
    #[cfg(feature = "onnx")]
    {
        if let Ok(gliner) = anno::GLiNEROnnx::new(anno::DEFAULT_GLINER_MODEL) {
            let gliner = std::sync::Arc::new(gliner);
            backends.push(BackendWrapper {
                name: "gliner_onnx",
                description: "GLiNER (manual ONNX)",
                extractor: Box::new(move |text| {
                    // GLiNER is zero-shot - provide common entity types
                    let entity_types = &["person", "organization", "location", "date", "money"];
                    gliner.extract(text, entity_types, 0.5).unwrap_or_default()
                }),
            });
        }
    }

    backends
}

// ============================================================================
// Benchmark Results
// ============================================================================

#[derive(Debug)]
#[allow(dead_code)]
struct BenchmarkResult {
    backend: String,
    dataset: String,
    num_examples: usize,
    total_entities_found: usize,
    elapsed: Duration,
    throughput_docs_sec: f64,
    throughput_ents_sec: f64,
}

#[derive(Debug)]
#[allow(dead_code)]
struct QualityResult {
    backend: String,
    dataset: String,
    precision: f64,
    recall: f64,
    f1: f64,
}

// ============================================================================
// Datasets
// ============================================================================

fn get_synthetic_texts() -> Vec<String> {
    all_datasets().into_iter().map(|ex| ex.text).collect()
}

fn get_benchmark_texts(size: usize) -> Vec<String> {
    // Use Ambiguous edge case type for a mix of challenging examples
    generate_large_dataset(size, EdgeCaseType::Ambiguous)
        .into_iter()
        .filter(|ex| !ex.text.is_empty())
        .map(|ex| ex.text)
        .collect()
}

fn get_challenge_texts() -> Vec<String> {
    vec![
        // Named entities (ML-required)
        "Steve Jobs founded Apple in Cupertino, California in 1976.".to_string(),
        "Elon Musk is the CEO of Tesla and SpaceX.".to_string(),
        "The United Nations headquarters is in New York City.".to_string(),
        "Microsoft acquired GitHub for $7.5 billion in 2018.".to_string(),
        "Dr. Jane Smith works at Harvard Medical School.".to_string(),
        
        // Structured entities (pattern-detectable)
        "Meeting scheduled for January 15, 2025 at 3:30 PM.".to_string(),
        "Contact: support@example.com or +1 (555) 123-4567.".to_string(),
        "Revenue increased by 25% to $1.5 billion in Q4 2024.".to_string(),
        "Visit https://example.com/api/v2 for documentation.".to_string(),
        "The event runs from 9:00 AM to 5:00 PM on 2025-03-15.".to_string(),
        
        // Mixed entities
        "Amazon's Jeff Bezos invested $500 million (15% stake) on March 1, 2024.".to_string(),
        "Email john.doe@acme.com before December 31st for 20% discount.".to_string(),
        "Tesla stock rose 8.5% after Elon Musk's announcement at 2pm EST.".to_string(),
        
        // Edge cases
        "".to_string(), // empty
        "No entities here just plain text".to_string(), // no entities
        "100% of $0 is still $0".to_string(), // numeric edge
        "日本語テキスト with English and $100".to_string(), // unicode
    ]
}

// ============================================================================
// Benchmarking Functions
// ============================================================================

fn benchmark_backend_speed(backend: &BackendWrapper, texts: &[String], dataset_name: &str) -> BenchmarkResult {
    let start = Instant::now();
    let mut total_entities = 0;

    for text in texts {
        let entities = backend.extract(text);
        total_entities += entities.len();
    }

    let elapsed = start.elapsed();
    let secs = elapsed.as_secs_f64();
    
    BenchmarkResult {
        backend: backend.name.to_string(),
        dataset: dataset_name.to_string(),
        num_examples: texts.len(),
        total_entities_found: total_entities,
        elapsed,
        throughput_docs_sec: if secs > 0.0 { texts.len() as f64 / secs } else { 0.0 },
        throughput_ents_sec: if secs > 0.0 { total_entities as f64 / secs } else { 0.0 },
    }
}

fn evaluate_backend_quality(
    backend: &BackendWrapper,
    dataset_name: &str,
) -> Option<QualityResult> {
    // Get annotated examples for quality evaluation
    let examples = all_datasets();
    if examples.is_empty() {
        return None;
    }

    // Convert to test cases
    let test_cases: Vec<_> = examples
        .iter()
        .filter(|ex| !ex.text.is_empty())
        .map(|ex| {
            let gold: Vec<_> = ex
                .entities
                .iter()
                .map(|g| anno::eval::GoldEntity {
                    text: g.text.clone(),
                    entity_type: g.entity_type.clone(),
                    start: g.start,
                    end: g.end,
                    original_label: format!("{:?}", g.entity_type),
                })
                .collect();
            (ex.text.clone(), gold)
        })
        .collect();

    if test_cases.is_empty() {
        return None;
    }

    // Evaluate using anno's evaluator
    let mut tp = 0;
    let mut fp = 0;
    let mut fn_count = 0;

    for (text, gold) in &test_cases {
        let predicted = backend.extract(text);
        
        // Simple span-based matching
        for pred in &predicted {
            let matched = gold.iter().any(|g| {
                pred.start == g.start && pred.end == g.end && 
                format!("{:?}", pred.entity_type) == format!("{:?}", g.entity_type)
            });
            if matched {
                tp += 1;
            } else {
                fp += 1;
            }
        }
        
        for g in gold {
            let matched = predicted.iter().any(|p| {
                p.start == g.start && p.end == g.end &&
                format!("{:?}", p.entity_type) == format!("{:?}", g.entity_type)
            });
            if !matched {
                fn_count += 1;
            }
        }
    }

    let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
    let recall = if tp + fn_count > 0 { tp as f64 / (tp + fn_count) as f64 } else { 0.0 };
    let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };

    Some(QualityResult {
        backend: backend.name.to_string(),
        dataset: dataset_name.to_string(),
        precision,
        recall,
        f1,
    })
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("{}", "=".repeat(80));
    println!(" COMPREHENSIVE NER BENCHMARK");
    println!("{}", "=".repeat(80));
    println!();

    // Create all available backends
    let backends = create_backends();
    
    println!("Available backends: {}", backends.len());
    for backend in &backends {
        println!("  - {} ({})", backend.name, backend.description);
    }
    println!();

    // Prepare datasets
    let datasets: Vec<(&str, Vec<String>)> = vec![
        ("synthetic", get_synthetic_texts()),
        ("benchmark_100", get_benchmark_texts(100)),
        ("benchmark_500", get_benchmark_texts(500)),
        ("challenge", get_challenge_texts()),
    ];

    // ========================================================================
    // SPEED BENCHMARKS
    // ========================================================================
    
    println!("{}", "=".repeat(80));
    println!(" SPEED BENCHMARKS");
    println!("{}", "=".repeat(80));
    println!();

    let mut speed_results = Vec::new();

    for (dataset_name, texts) in &datasets {
        println!("Dataset: {} ({} examples)", dataset_name, texts.len());
        println!("{:-<70}", "");
        
        for backend in &backends {
            let result = benchmark_backend_speed(backend, texts, dataset_name);
            
            println!(
                "  {:<20} | {:>6} entities | {:>8.2} ms | {:>8.1} docs/s | {:>8.1} ents/s",
                backend.name,
                result.total_entities_found,
                result.elapsed.as_secs_f64() * 1000.0,
                result.throughput_docs_sec,
                result.throughput_ents_sec,
            );
            
            speed_results.push(result);
        }
        println!();
    }

    // ========================================================================
    // QUALITY BENCHMARKS
    // ========================================================================
    
    println!("{}", "=".repeat(80));
    println!(" QUALITY BENCHMARKS (on synthetic dataset with ground truth)");
    println!("{}", "=".repeat(80));
    println!();

    println!("{:<20} | {:>10} | {:>10} | {:>10}", "Backend", "Precision", "Recall", "F1");
    println!("{:-<60}", "");

    for backend in &backends {
        if let Some(result) = evaluate_backend_quality(backend, "synthetic") {
            println!(
                "{:<20} | {:>10.1}% | {:>10.1}% | {:>10.1}%",
                backend.name,
                result.precision * 100.0,
                result.recall * 100.0,
                result.f1 * 100.0,
            );
        } else {
            println!("{:<20} | {:>10} | {:>10} | {:>10}", backend.name, "N/A", "N/A", "N/A");
        }
    }
    println!();

    // ========================================================================
    // ENTITY TYPE BREAKDOWN
    // ========================================================================
    
    println!("{}", "=".repeat(80));
    println!(" ENTITY TYPE BREAKDOWN (challenge texts)");
    println!("{}", "=".repeat(80));
    println!();

    let challenge_texts = get_challenge_texts();
    
    for backend in &backends {
        println!("Backend: {} ({})", backend.name, backend.description);
        println!("{:-<70}", "");
        
        let mut type_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        
        for text in &challenge_texts {
            let entities = backend.extract(text);
            for entity in &entities {
                let type_name = format!("{:?}", entity.entity_type);
                *type_counts.entry(type_name).or_insert(0) += 1;
            }
        }
        
        if type_counts.is_empty() {
            println!("  No entities found");
        } else {
            let mut counts: Vec<_> = type_counts.into_iter().collect();
            counts.sort_by(|a, b| b.1.cmp(&a.1));
            
            for (entity_type, count) in counts {
                println!("  {:>4}x {}", count, entity_type);
            }
        }
        println!();
    }

    // ========================================================================
    // SAMPLE OUTPUTS
    // ========================================================================
    
    println!("{}", "=".repeat(80));
    println!(" SAMPLE OUTPUTS");
    println!("{}", "=".repeat(80));
    println!();

    let sample_texts = vec![
        "Steve Jobs founded Apple in Cupertino in 1976.",
        "Meeting at 3:30 PM on Jan 15. Cost: $500.",
        "Contact: support@example.com or (555) 123-4567.",
    ];

    for text in sample_texts {
        println!("Input: \"{}\"", text);
        println!();
        
        for backend in &backends {
            let entities = backend.extract(text);
            println!("  {} ({} entities):", backend.name, entities.len());
            
            if entities.is_empty() {
                println!("    (none)");
            } else {
                for entity in &entities {
                    println!(
                        "    - \"{}\" [{}-{}] {:?} ({:.0}%)",
                        entity.text,
                        entity.start,
                        entity.end,
                        entity.entity_type,
                        entity.confidence * 100.0
                    );
                }
            }
        }
        println!();
    }

    // ========================================================================
    // SUMMARY
    // ========================================================================
    
    println!("{}", "=".repeat(80));
    println!(" SUMMARY");
    println!("{}", "=".repeat(80));
    println!();
    
    println!("Backends tested: {}", backends.len());
    println!("Datasets tested: {}", datasets.len());
    
    // Find fastest backend on benchmark_500
    if let Some(fastest) = speed_results
        .iter()
        .filter(|r| r.dataset == "benchmark_500")
        .max_by(|a, b| a.throughput_docs_sec.partial_cmp(&b.throughput_docs_sec).unwrap())
    {
        println!("Fastest backend: {} ({:.0} docs/sec)", fastest.backend, fastest.throughput_docs_sec);
    }
    
    println!();
    println!("To test ML backends, run with features:");
    println!("  cargo run --example comprehensive_bench --features onnx");
    println!("  cargo run --example comprehensive_bench --features gliner");
    println!("  cargo run --example comprehensive_bench --features \"onnx,gliner\"");
}

