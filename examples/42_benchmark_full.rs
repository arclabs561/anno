//! Full NER Benchmark Suite
//!
//! Tests ALL available backends against ALL available datasets.
//! Provides comprehensive quality and speed metrics.
//!
//! ## Usage
//!
//! ```bash
//! # Quick benchmark (synthetic + pattern-only)
//! cargo run --example full_benchmark
//!
//! # With real datasets (requires network)
//! cargo run --example full_benchmark --features network
//!
//! # With ML backends
//! cargo run --example full_benchmark --features "onnx,gliner,network"
//! ```
//!
//! ## GLiNER Model Catalog
//!
//! Models are available with different encoders:
//! - DeBERTa-v3 (512 tokens): Standard, well-tested
//! - ModernBERT (8192 tokens): SOTA, ~3% better accuracy

use anno::eval::benchmark::{generate_large_dataset, EdgeCaseType};
use anno::eval::synthetic::all_datasets;
use anno::{Entity, EntityType, Model, NERExtractor, PatternNER};
use anno::backends::encoder::{GLiNERModel, GLINER_MODELS};
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[cfg(feature = "network")]
use anno::eval::loader::{DatasetId, DatasetLoader, LoadedDataset};

// ============================================================================
// Backend Definition
// ============================================================================

struct Backend {
    name: &'static str,
    description: &'static str,
    available: bool,
    extractor: Box<dyn Fn(&str) -> Vec<Entity> + Send + Sync>,
}

fn create_all_backends() -> Vec<Backend> {
    let mut backends = Vec::new();

    // PatternNER (always available)
    backends.push(Backend {
        name: "pattern",
        description: "Pattern-based (regex) - structured entities only",
        available: true,
        extractor: Box::new(|text| {
            let ner = PatternNER::new();
            ner.extract_entities(text, None).unwrap_or_default()
        }),
    });

    // NERExtractor best_available
    backends.push(Backend {
        name: "best_available",
        description: "Best available (auto-selects optimal backend)",
        available: true,
        extractor: Box::new(|text| {
            let extractor = NERExtractor::best_available();
            extractor.extract(text, None).unwrap_or_default()
        }),
    });

    // Hybrid mode
    backends.push(Backend {
        name: "hybrid",
        description: "Hybrid (combines ML and patterns)",
        available: true,
        extractor: Box::new(|text| {
            let extractor = NERExtractor::best_available();
            extractor.extract_hybrid(text, None).unwrap_or_default()
        }),
    });

    // BERT ONNX
    #[cfg(feature = "onnx")]
    {
        let available = anno::backends::onnx::BertNEROnnx::new(anno::DEFAULT_BERT_ONNX_MODEL).is_ok();
        if available {
            if let Ok(bert) = anno::backends::onnx::BertNEROnnx::new(anno::DEFAULT_BERT_ONNX_MODEL) {
                let bert = std::sync::Arc::new(bert);
                backends.push(Backend {
                    name: "bert_onnx",
                    description: "BERT NER via ONNX Runtime",
                    available: true,
                    extractor: Box::new(move |text| {
                        bert.extract_entities(text, None).unwrap_or_default()
                    }),
                });
            }
        } else {
            backends.push(Backend {
                name: "bert_onnx",
                description: "BERT NER (model not loaded)",
                available: false,
                extractor: Box::new(|_| Vec::new()),
            });
        }
    }

    // GLiNER via ONNX (manual implementation)
    #[cfg(feature = "onnx")]
    {
        let available = anno::backends::gliner_onnx::GLiNEROnnx::new(anno::DEFAULT_GLINER_MODEL).is_ok();
        if available {
            if let Ok(gliner) = anno::backends::gliner_onnx::GLiNEROnnx::new(anno::DEFAULT_GLINER_MODEL) {
                let gliner = std::sync::Arc::new(gliner);
                backends.push(Backend {
                    name: "gliner_onnx",
                    description: "GLiNER (manual ONNX implementation)",
                    available: true,
                    extractor: Box::new(move |text| {
                        // GLiNER is zero-shot - provide common entity types
                        let entity_types = &["person", "organization", "location", "date", "money"];
                        gliner.extract(text, entity_types, 0.5).unwrap_or_default()
                    }),
                });
            }
        } else {
            backends.push(Backend {
                name: "gliner_onnx",
                description: "GLiNER ONNX (model not loaded)",
                available: false,
                extractor: Box::new(|_| Vec::new()),
            });
        }
    }

    // Candle NER
    #[cfg(feature = "candle")]
    {
        backends.push(Backend {
            name: "candle",
            description: "Candle (pure Rust ML)",
            available: false, // Typically needs model setup
            extractor: Box::new(|_| Vec::new()),
        });
    }

    backends
}

// ============================================================================
// Dataset Definition
// ============================================================================

struct TestDataset {
    name: &'static str,
    texts: Vec<String>,
    /// Optional ground truth for quality evaluation
    ground_truth: Option<Vec<Vec<GoldEntity>>>,
}

#[derive(Clone)]
#[allow(dead_code)]
struct GoldEntity {
    text: String,
    entity_type: EntityType,
    start: usize,
    end: usize,
}

fn get_synthetic_dataset() -> TestDataset {
    let examples = all_datasets();
    let texts: Vec<_> = examples.iter().map(|e| e.text.clone()).collect();
    let ground_truth: Vec<Vec<GoldEntity>> = examples
        .iter()
        .map(|e| {
            e.entities
                .iter()
                .map(|g| GoldEntity {
                    text: g.text.clone(),
                    entity_type: g.entity_type.clone(),
                    start: g.start,
                    end: g.end,
                })
                .collect()
        })
        .collect();

    TestDataset {
        name: "synthetic",
        texts,
        ground_truth: Some(ground_truth),
    }
}

fn get_benchmark_dataset(size: usize, edge_case: EdgeCaseType) -> TestDataset {
    let examples = generate_large_dataset(size, edge_case);
    let texts: Vec<_> = examples
        .iter()
        .filter(|e| !e.text.is_empty())
        .map(|e| e.text.clone())
        .collect();
    let ground_truth: Vec<Vec<GoldEntity>> = examples
        .iter()
        .filter(|e| !e.text.is_empty())
        .map(|e| {
            e.entities
                .iter()
                .map(|g| GoldEntity {
                    text: g.text.clone(),
                    entity_type: g.entity_type.clone(),
                    start: g.start,
                    end: g.end,
                })
                .collect()
        })
        .collect();

    TestDataset {
        name: match edge_case {
            EdgeCaseType::All => "benchmark_all",
            EdgeCaseType::Ambiguous => "benchmark_ambiguous",
            EdgeCaseType::Unicode => "benchmark_unicode",
            EdgeCaseType::Dense => "benchmark_dense",
            EdgeCaseType::Sparse => "benchmark_sparse",
            EdgeCaseType::Nested => "benchmark_nested",
            EdgeCaseType::Casing => "benchmark_casing",
            EdgeCaseType::Boundary => "benchmark_boundary",
            EdgeCaseType::MultiWord => "benchmark_multiword",
            EdgeCaseType::NumericEdge => "benchmark_numeric",
            EdgeCaseType::Jargon => "benchmark_jargon",
        },
        texts,
        ground_truth: Some(ground_truth),
    }
}

fn get_challenge_dataset() -> TestDataset {
    TestDataset {
        name: "challenge",
        texts: vec![
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
            // Edge cases
            "100% of $0 is still $0".to_string(),
            "日本語テキスト with English and $100".to_string(),
        ],
        ground_truth: None,
    }
}

#[cfg(feature = "network")]
fn get_real_dataset(loader: &DatasetLoader, dataset_id: DatasetId) -> Option<TestDataset> {
    match loader.load(dataset_id) {
        Ok(loaded) => {
            let texts: Vec<_> = loaded.sentences.iter().map(|s| s.text()).collect();
            let ground_truth: Vec<Vec<GoldEntity>> = loaded
                .sentences
                .iter()
                .map(|s| {
                    s.entities()
                        .iter()
                        .map(|e| GoldEntity {
                            text: e.text.clone(),
                            entity_type: e.entity_type.clone(),
                            start: e.start,
                            end: e.end,
                        })
                        .collect()
                })
                .collect();
            
            Some(TestDataset {
                name: dataset_id.name(),
                texts,
                ground_truth: Some(ground_truth),
            })
        }
        Err(e) => {
            eprintln!("Warning: Failed to load {:?}: {}", dataset_id, e);
            None
        }
    }
}

// ============================================================================
// Evaluation
// ============================================================================

#[derive(Default)]
struct EvalResults {
    true_positives: usize,
    false_positives: usize,
    false_negatives: usize,
    elapsed: Duration,
    entity_count: usize,
}

impl EvalResults {
    fn precision(&self) -> f64 {
        let denom = self.true_positives + self.false_positives;
        if denom == 0 { 0.0 } else { self.true_positives as f64 / denom as f64 }
    }

    fn recall(&self) -> f64 {
        let denom = self.true_positives + self.false_negatives;
        if denom == 0 { 0.0 } else { self.true_positives as f64 / denom as f64 }
    }

    fn f1(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) }
    }

    fn throughput(&self, n_docs: usize) -> f64 {
        let secs = self.elapsed.as_secs_f64();
        if secs == 0.0 { 0.0 } else { n_docs as f64 / secs }
    }
}

fn evaluate_backend(backend: &Backend, dataset: &TestDataset) -> EvalResults {
    if !backend.available {
        return EvalResults::default();
    }

    let start = Instant::now();
    let mut results = EvalResults::default();

    for (i, text) in dataset.texts.iter().enumerate() {
        let predicted = (backend.extractor)(text);
        results.entity_count += predicted.len();

        // Quality evaluation if ground truth available
        if let Some(ref gt) = dataset.ground_truth {
            if let Some(gold) = gt.get(i) {
                for pred in &predicted {
                    let matched = gold.iter().any(|g| {
                        pred.start == g.start
                            && pred.end == g.end
                            && entity_type_matches(&pred.entity_type, &g.entity_type)
                    });
                    if matched {
                        results.true_positives += 1;
                    } else {
                        results.false_positives += 1;
                    }
                }

                for g in gold {
                    let matched = predicted.iter().any(|p| {
                        p.start == g.start
                            && p.end == g.end
                            && entity_type_matches(&p.entity_type, &g.entity_type)
                    });
                    if !matched {
                        results.false_negatives += 1;
                    }
                }
            }
        }
    }

    results.elapsed = start.elapsed();
    results
}

fn entity_type_matches(a: &EntityType, b: &EntityType) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}

// ============================================================================
// Reporting
// ============================================================================

fn print_header(title: &str) {
    println!();
    println!("{}", "=".repeat(90));
    println!(" {}", title);
    println!("{}", "=".repeat(90));
    println!();
}

fn print_results_table(
    backends: &[Backend],
    datasets: &[TestDataset],
    results: &HashMap<(&str, &str), EvalResults>,
) {
    // Print header
    print!("{:<20}", "Backend");
    for dataset in datasets {
        print!(" | {:>12}", dataset.name);
    }
    println!(" | {:>12}", "Avg F1");
    println!("{}", "-".repeat(20 + (datasets.len() + 1) * 15));

    // Print rows
    for backend in backends {
        if !backend.available {
            continue;
        }
        print!("{:<20}", backend.name);
        let mut f1_sum = 0.0;
        let mut count = 0;
        for dataset in datasets {
            if let Some(r) = results.get(&(backend.name, dataset.name)) {
                print!(" | {:>11.1}%", r.f1() * 100.0);
                f1_sum += r.f1();
                count += 1;
            } else {
                print!(" | {:>12}", "N/A");
            }
        }
        if count > 0 {
            println!(" | {:>11.1}%", (f1_sum / count as f64) * 100.0);
        } else {
            println!(" | {:>12}", "N/A");
        }
    }
}

fn print_speed_table(
    backends: &[Backend],
    datasets: &[TestDataset],
    results: &HashMap<(&str, &str), EvalResults>,
) {
    print!("{:<20}", "Backend");
    for dataset in datasets {
        print!(" | {:>12}", dataset.name);
    }
    println!();
    println!("{}", "-".repeat(20 + datasets.len() * 15));

    for backend in backends {
        if !backend.available {
            continue;
        }
        print!("{:<20}", backend.name);
        for dataset in datasets {
            if let Some(r) = results.get(&(backend.name, dataset.name)) {
                let throughput = r.throughput(dataset.texts.len());
                print!(" | {:>10.0} d/s", throughput);
            } else {
                print!(" | {:>12}", "N/A");
            }
        }
        println!();
    }
}

fn print_entity_breakdown(backend: &Backend, texts: &[String]) {
    if !backend.available {
        return;
    }
    
    let mut type_counts: HashMap<String, usize> = HashMap::new();
    for text in texts {
        for entity in (backend.extractor)(text) {
            let type_name = format!("{:?}", entity.entity_type);
            *type_counts.entry(type_name).or_insert(0) += 1;
        }
    }

    println!("{}: {} ({} total entities)", 
        backend.name, 
        backend.description,
        type_counts.values().sum::<usize>()
    );
    
    let mut counts: Vec<_> = type_counts.into_iter().collect();
    counts.sort_by(|a, b| b.1.cmp(&a.1));
    for (t, c) in counts.iter().take(10) {
        println!("  {:>4}x {}", c, t);
    }
    println!();
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    print_header("FULL NER BENCHMARK SUITE");

    // Show available GLiNER models
    println!("GLiNER Model Catalog:");
    println!("  {:50} {:12} {:5} {}", "Model ID", "Encoder", "Size", "Notes");
    println!("  {}", "-".repeat(90));
    for model in GLINER_MODELS {
        println!("  {:50} {:12} {:5} {}",
            model.model_id, model.encoder.to_string(), model.size.to_string(), model.notes);
    }
    println!("\n  Best for speed: {}", GLiNERModel::fastest().model_id);
    println!("  Best for accuracy: {}", GLiNERModel::most_accurate().model_id);
    println!();

    // Create backends
    let backends = create_all_backends();
    println!("Available backends:");
    for b in &backends {
        let status = if b.available { "[OK]" } else { "[--]" };
        println!("  {} {}: {}", status, b.name, b.description);
    }

    // Create datasets
    #[allow(unused_mut)]
    let mut datasets: Vec<TestDataset> = vec![
        get_synthetic_dataset(),
        get_challenge_dataset(),
        get_benchmark_dataset(100, EdgeCaseType::Ambiguous),
        get_benchmark_dataset(100, EdgeCaseType::Dense),
    ];

    // Try to load real datasets
    #[cfg(feature = "network")]
    {
        println!("\nLoading real-world datasets...");
        if let Ok(loader) = DatasetLoader::new() {
            for dataset_id in &[
                DatasetId::WikiGold,
                DatasetId::Wnut17,
                DatasetId::MitMovie,
                DatasetId::MitRestaurant,
            ] {
                print!("  Loading {:?}... ", dataset_id);
                if let Some(ds) = get_real_dataset(&loader, *dataset_id) {
                    println!("{} examples", ds.texts.len());
                    datasets.push(ds);
                } else {
                    println!("failed");
                }
            }
        }
    }

    println!("\nDatasets:");
    for ds in &datasets {
        let has_gt = if ds.ground_truth.is_some() { "with GT" } else { "no GT" };
        println!("  {}: {} examples ({})", ds.name, ds.texts.len(), has_gt);
    }

    // Run evaluations
    print_header("QUALITY METRICS (F1 Score)");
    
    let mut results: HashMap<(&str, &str), EvalResults> = HashMap::new();
    
    for backend in &backends {
        if !backend.available {
            continue;
        }
        print!("Evaluating {}... ", backend.name);
        std::io::Write::flush(&mut std::io::stdout()).ok();
        
        for dataset in &datasets {
            let r = evaluate_backend(backend, dataset);
            results.insert((backend.name, dataset.name), r);
        }
        println!("done");
    }

    // Quality table (only datasets with ground truth)
    let gt_datasets: Vec<_> = datasets.iter().filter(|d| d.ground_truth.is_some()).collect();
    print_results_table(
        &backends,
        &gt_datasets.iter().map(|d| (*d).clone()).collect::<Vec<_>>(),
        &results,
    );

    // Speed table
    print_header("SPEED METRICS (docs/sec)");
    print_speed_table(&backends, &datasets, &results);

    // Entity breakdown for challenge texts
    print_header("ENTITY TYPE BREAKDOWN (challenge dataset)");
    let challenge = get_challenge_dataset();
    for backend in &backends {
        print_entity_breakdown(backend, &challenge.texts);
    }

    // Summary
    print_header("SUMMARY");
    
    let available_count = backends.iter().filter(|b| b.available).count();
    println!("Backends tested: {} / {}", available_count, backends.len());
    println!("Datasets tested: {}", datasets.len());
    
    // Find best backend
    let mut best_backend = "";
    let mut best_f1 = 0.0;
    for backend in &backends {
        if !backend.available {
            continue;
        }
        let mut f1_sum = 0.0;
        let mut count = 0;
        for dataset in &gt_datasets {
            if let Some(r) = results.get(&(backend.name, dataset.name)) {
                f1_sum += r.f1();
                count += 1;
            }
        }
        if count > 0 {
            let avg_f1 = f1_sum / count as f64;
            if avg_f1 > best_f1 {
                best_f1 = avg_f1;
                best_backend = backend.name;
            }
        }
    }
    
    if !best_backend.is_empty() {
        println!("Best backend (avg F1): {} ({:.1}%)", best_backend, best_f1 * 100.0);
    }

    // Find fastest backend
    let mut fastest_backend = "";
    let mut best_speed = 0.0;
    for backend in &backends {
        if !backend.available {
            continue;
        }
        if let Some(r) = results.get(&(backend.name, "synthetic")) {
            let speed = r.throughput(get_synthetic_dataset().texts.len());
            if speed > best_speed {
                best_speed = speed;
                fastest_backend = backend.name;
            }
        }
    }
    
    if !fastest_backend.is_empty() {
        println!("Fastest backend: {} ({:.0} docs/sec)", fastest_backend, best_speed);
    }

    println!();
    println!("To test ML backends:");
    println!("  cargo run --example full_benchmark --features onnx");
    println!("  cargo run --example full_benchmark --features gliner");
    println!("  cargo run --example full_benchmark --features \"onnx,gliner,network\"");
}

// Allow Clone for TestDataset
impl Clone for TestDataset {
    fn clone(&self) -> Self {
        TestDataset {
            name: self.name,
            texts: self.texts.clone(),
            ground_truth: self.ground_truth.clone(),
        }
    }
}

