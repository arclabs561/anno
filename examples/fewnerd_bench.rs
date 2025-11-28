//! FewNERD benchmark for comparing bi-encoder NER approaches.
//!
//! Evaluates different NER backends on the FewNERD dataset to compare:
//! - Traditional pattern/heuristic approaches (baseline)
//! - ML token classification (BertNEROnnx)
//! - Zero-shot bi-encoder (GLiNEROnnx, NuNER)
//!
//! Run with:
//!   cargo run --example fewnerd_bench --features "onnx,network"

use anno::eval::loader::{DatasetId, DatasetLoader};
use anno::eval::{evaluate_ner_model_with_mapper, GoldEntity};
use anno::{Model, PatternNER, StackedNER, StatisticalNER, TypeMapper};
use std::time::Instant;

/// Backend configuration
struct Backend {
    name: &'static str,
    approach: &'static str,
    model: Box<dyn Model>,
}

/// Results for a single backend
#[derive(Debug)]
struct BenchResult {
    name: String,
    approach: String,
    f1: f64,
    precision: f64,
    recall: f64,
    latency_ms: f64,
    samples: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== FewNERD Benchmark: Bi-Encoder vs Traditional NER ===\n");

    // Build backends
    let mut backends: Vec<Backend> = vec![
        Backend {
            name: "PatternNER",
            approach: "Rule-based",
            model: Box::new(PatternNER::new()),
        },
        Backend {
            name: "StatisticalNER",
            approach: "Heuristic",
            model: Box::new(StatisticalNER::new()),
        },
        Backend {
            name: "StackedNER",
            approach: "Hybrid",
            model: Box::new(StackedNER::new()),
        },
    ];

    // Add ONNX backends
    #[cfg(feature = "onnx")]
    {
        // Token classification (traditional ML)
        if let Ok(bert) = anno::BertNEROnnx::new(anno::DEFAULT_BERT_ONNX_MODEL) {
            backends.push(Backend {
                name: "BertNEROnnx",
                approach: "Token-Classification",
                model: Box::new(bert),
            });
            println!("Loaded: BertNEROnnx (token classification)");
        }

        // Bi-encoder zero-shot
        if let Ok(gliner) = anno::GLiNEROnnx::new(anno::DEFAULT_GLINER_MODEL) {
            backends.push(Backend {
                name: "GLiNEROnnx",
                approach: "Bi-Encoder (zero-shot)",
                model: Box::new(gliner),
            });
            println!("Loaded: GLiNEROnnx (bi-encoder, zero-shot)");
        }

        // NuNER (alternative bi-encoder)
        let nuner = anno::NuNER::new();
        if nuner.is_available() {
            backends.push(Backend {
                name: "NuNER",
                approach: "Bi-Encoder (zero-shot)",
                model: Box::new(nuner),
            });
            println!("Loaded: NuNER (bi-encoder, zero-shot)");
        }
    }

    println!("\nBackends: {}", backends.len());

    // Load FewNERD dataset
    let loader = DatasetLoader::new()?;
    println!("\nLoading FewNERD dataset...");

    let start = Instant::now();

    #[cfg(feature = "network")]
    let loaded = loader.load_or_download(DatasetId::FewNERD);

    #[cfg(not(feature = "network"))]
    let loaded = loader.load(DatasetId::FewNERD);

    let dataset = match loaded {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Could not load FewNERD: {}", e);
            eprintln!("Falling back to synthetic evaluation.");
            return run_synthetic_benchmark(backends);
        }
    };

    println!(
        "Loaded {} sentences in {:.2?}",
        dataset.sentences.len(),
        start.elapsed()
    );

    // Convert to test cases
    let test_cases: Vec<(String, Vec<GoldEntity>)> = dataset
        .sentences
        .iter()
        .filter(|s| !s.tokens.is_empty())
        .map(|s| {
            let text = s.text();
            let entities = s.entities();
            (text, entities)
        })
        .collect();

    // Limit samples for reasonable benchmark time
    let max_samples = 500;
    let test_cases: Vec<_> = test_cases.into_iter().take(max_samples).collect();
    println!("Evaluating on {} samples\n", test_cases.len());

    // Type mapper for FewNERD's fine-grained types
    let mut mapper = TypeMapper::new();
    mapper.add("person", anno::EntityType::Person);
    mapper.add("organization", anno::EntityType::Organization);
    mapper.add("location", anno::EntityType::Location);
    mapper.add("building", anno::EntityType::Location);
    mapper.add("event", anno::EntityType::Other("MISC".into()));
    mapper.add("product", anno::EntityType::Other("MISC".into()));
    mapper.add("art", anno::EntityType::Other("MISC".into()));

    // Evaluate each backend
    let mut results: Vec<BenchResult> = Vec::new();

    for backend in &backends {
        print!("Evaluating {}... ", backend.name);
        std::io::Write::flush(&mut std::io::stdout())?;

        let start = Instant::now();

        match evaluate_ner_model_with_mapper(backend.model.as_ref(), &test_cases, Some(&mapper)) {
            Ok(eval_result) => {
                let elapsed = start.elapsed();
                let latency_ms = elapsed.as_secs_f64() * 1000.0 / test_cases.len() as f64;

                println!(
                    "F1: {:.1}%, P: {:.1}%, R: {:.1}%, {:.1}ms/sample",
                    eval_result.f1 * 100.0,
                    eval_result.precision * 100.0,
                    eval_result.recall * 100.0,
                    latency_ms
                );

                results.push(BenchResult {
                    name: backend.name.to_string(),
                    approach: backend.approach.to_string(),
                    f1: eval_result.f1,
                    precision: eval_result.precision,
                    recall: eval_result.recall,
                    latency_ms,
                    samples: test_cases.len(),
                });
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }
    }

    // Print summary table
    println!("\n=== Summary ===\n");
    println!(
        "{:<15} {:<22} {:>8} {:>8} {:>8} {:>10}",
        "Backend", "Approach", "F1", "Prec", "Recall", "ms/sample"
    );
    println!("{}", "-".repeat(77));

    // Sort by F1
    results.sort_by(|a, b| b.f1.partial_cmp(&a.f1).unwrap());

    for r in &results {
        println!(
            "{:<15} {:<22} {:>7.1}% {:>7.1}% {:>7.1}% {:>10.1}",
            r.name,
            r.approach,
            r.f1 * 100.0,
            r.precision * 100.0,
            r.recall * 100.0,
            r.latency_ms
        );
    }

    // Analysis
    println!("\n=== Analysis ===\n");

    if let Some(best) = results.first() {
        println!(
            "Best F1: {} ({}) at {:.1}%",
            best.name, best.approach, best.f1 * 100.0
        );
    }

    // Compare bi-encoder vs token-classification
    let biencoder: Vec<_> = results
        .iter()
        .filter(|r| r.approach.contains("Bi-Encoder"))
        .collect();
    let tokclass: Vec<_> = results
        .iter()
        .filter(|r| r.approach.contains("Token-Classification"))
        .collect();

    if !biencoder.is_empty() && !tokclass.is_empty() {
        let bi_avg_f1: f64 = biencoder.iter().map(|r| r.f1).sum::<f64>() / biencoder.len() as f64;
        let tc_avg_f1: f64 = tokclass.iter().map(|r| r.f1).sum::<f64>() / tokclass.len() as f64;

        println!(
            "\nBi-Encoder avg F1: {:.1}% vs Token-Classification avg F1: {:.1}%",
            bi_avg_f1 * 100.0,
            tc_avg_f1 * 100.0
        );

        if bi_avg_f1 > tc_avg_f1 {
            println!(
                "Bi-encoder outperforms by {:.1}% (absolute)",
                (bi_avg_f1 - tc_avg_f1) * 100.0
            );
        } else {
            println!(
                "Token-classification outperforms by {:.1}% (absolute)",
                (tc_avg_f1 - bi_avg_f1) * 100.0
            );
        }
    }

    Ok(())
}

/// Fallback to synthetic benchmark if dataset unavailable
fn run_synthetic_benchmark(backends: Vec<Backend>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Synthetic Benchmark (FewNERD unavailable) ===\n");

    let test_texts = [
        "Steve Jobs founded Apple in Cupertino, California in 1976.",
        "Microsoft CEO Satya Nadella announced Azure AI updates.",
        "The European Union met in Brussels to discuss trade policy.",
        "Dr. Jane Smith published research at Stanford University.",
        "Amazon Web Services reported $25B quarterly revenue.",
    ];

    for backend in &backends {
        println!("Testing {}...", backend.name);
        let start = Instant::now();
        let mut total = 0;

        for text in &test_texts {
            if let Ok(entities) = backend.model.extract_entities(text, None) {
                total += entities.len();
                for e in &entities {
                    println!(
                        "  {}: {} ({:.2})",
                        text.get(0..30).unwrap_or(text),
                        e.text,
                        e.confidence
                    );
                }
            }
        }

        println!("  Found {} entities in {:.2?}\n", total, start.elapsed());
    }

    Ok(())
}
