//! NER Evaluation Suite
//!
//! Comprehensive evaluation of NER backends on both synthetic and real datasets.
//!
//! ## Quick Start
//!
//! ```bash
//! # Synthetic only (fast, no downloads)
//! cargo run --example eval
//!
//! # Include real datasets (requires eval-advanced)
//! cargo run --example eval --features eval-advanced
//!
//! # With ML backends
//! cargo run --example eval --features "onnx,eval-advanced"
//! ```
//!
//! ## Output
//!
//! - Per-backend F1/Precision/Recall
//! - Per-entity-type breakdown
//! - Statistical significance testing
//! - Error analysis (confusion matrix)

use anno::eval::analysis::{compare_ner_systems, ConfusionMatrix, ErrorAnalysis};
#[cfg(feature = "eval-advanced")]
use anno::eval::loader::{DatasetId, DatasetLoader};
use anno::eval::synthetic::{all_datasets, Difficulty};
use anno::eval::{evaluate_ner_model, GoldEntity};
use anno::{HeuristicNER, Model, RegexNER, StackedNER};
use std::collections::HashMap;
use std::time::Instant;

// =============================================================================
// Backend Registry
// =============================================================================

struct Backend {
    name: &'static str,
    model: Box<dyn Model>,
}

fn create_backends() -> Vec<Backend> {
    #[allow(unused_mut)]
    let mut backends = vec![
        Backend {
            name: "RegexNER",
            model: Box::new(RegexNER::new()),
        },
        Backend {
            name: "HeuristicNER",
            model: Box::new(HeuristicNER::new()),
        },
        Backend {
            name: "StackedNER",
            model: Box::new(StackedNER::new()),
        },
    ];

    #[cfg(feature = "onnx")]
    {
        use anno::BertNEROnnx;
        if let Ok(bert) = BertNEROnnx::new(anno::DEFAULT_BERT_ONNX_MODEL) {
            backends.push(Backend {
                name: "BertNEROnnx",
                model: Box::new(bert),
            });
        }
    }

    backends
}

// =============================================================================
// Synthetic Evaluation
// =============================================================================

fn evaluate_synthetic(backends: &[Backend]) -> HashMap<String, Vec<f64>> {
    println!("\n=== Synthetic Dataset Evaluation ===\n");

    let datasets = all_datasets();
    let test_cases: Vec<(String, Vec<GoldEntity>)> = datasets
        .iter()
        .filter(|ex| !ex.text.is_empty())
        .map(|ex| (ex.text.clone(), ex.entities.clone()))
        .collect();

    println!("Dataset: {} examples", test_cases.len());

    // Group by difficulty
    let by_difficulty: HashMap<Difficulty, Vec<_>> =
        datasets.iter().fold(HashMap::new(), |mut acc, ex| {
            acc.entry(ex.difficulty).or_default().push(ex);
            acc
        });

    println!(
        "  Easy: {}",
        by_difficulty.get(&Difficulty::Easy).map_or(0, |v| v.len())
    );
    println!(
        "  Medium: {}",
        by_difficulty
            .get(&Difficulty::Medium)
            .map_or(0, |v| v.len())
    );
    println!(
        "  Hard: {}",
        by_difficulty.get(&Difficulty::Hard).map_or(0, |v| v.len())
    );
    println!(
        "  Adversarial: {}",
        by_difficulty
            .get(&Difficulty::Adversarial)
            .map_or(0, |v| v.len())
    );
    println!();

    let mut all_f1_scores: HashMap<String, Vec<f64>> = HashMap::new();

    for backend in backends {
        let start = Instant::now();
        let result = evaluate_ner_model(backend.model.as_ref(), &test_cases);
        let elapsed = start.elapsed();

        match result {
            Ok(metrics) => {
                println!(
                    "{:15} F1={:5.1}%  P={:5.1}%  R={:5.1}%  ({:.0}ms)",
                    backend.name,
                    metrics.f1 * 100.0,
                    metrics.precision * 100.0,
                    metrics.recall * 100.0,
                    elapsed.as_millis()
                );

                // Compute per-example F1 for significance testing
                // (simplified: use per-type F1s as proxy for variance)
                let per_type_f1s: Vec<f64> = metrics
                    .per_type
                    .values()
                    .map(|t| t.f1)
                    .filter(|&f| f > 0.0)
                    .collect();

                if per_type_f1s.is_empty() {
                    all_f1_scores
                        .entry(backend.name.to_string())
                        .or_default()
                        .push(metrics.f1);
                } else {
                    all_f1_scores
                        .entry(backend.name.to_string())
                        .or_default()
                        .extend(per_type_f1s);
                }
            }
            Err(e) => {
                println!("{:15} ERROR: {}", backend.name, e);
            }
        }
    }

    all_f1_scores
}

// =============================================================================
// Real Dataset Evaluation
// =============================================================================

#[cfg(feature = "eval-advanced")]
fn evaluate_real_datasets(backends: &[Backend]) -> HashMap<String, Vec<f64>> {
    println!("\n=== Real Dataset Evaluation ===\n");

    let loader = match DatasetLoader::new() {
        Ok(l) => l,
        Err(e) => {
            println!("Failed to create loader: {}", e);
            return HashMap::new();
        }
    };

    let dataset_ids = DatasetId::all_ner();
    let mut all_f1_scores: HashMap<String, Vec<f64>> = HashMap::new();

    for &id in dataset_ids {
        print!("Loading {:20} ... ", id.name());

        let dataset = match loader.load_or_download(id) {
            Ok(d) => d,
            Err(e) => {
                println!("SKIP ({})", e);
                continue;
            }
        };

        println!(
            "{} sentences, {} entities",
            dataset.len(),
            dataset.entity_count()
        );

        let test_cases = dataset.to_test_cases();

        for backend in backends {
            match evaluate_ner_model(backend.model.as_ref(), &test_cases) {
                Ok(metrics) => {
                    println!(
                        "  {:15} F1={:5.1}%  P={:5.1}%  R={:5.1}%",
                        backend.name,
                        metrics.f1 * 100.0,
                        metrics.precision * 100.0,
                        metrics.recall * 100.0
                    );

                    all_f1_scores
                        .entry(backend.name.to_string())
                        .or_default()
                        .push(metrics.f1);
                }
                Err(e) => {
                    println!("  {:15} ERROR: {}", backend.name, e);
                }
            }
        }
        println!();
    }

    all_f1_scores
}

#[cfg(not(feature = "eval-advanced"))]
fn evaluate_real_datasets(_backends: &[Backend]) -> HashMap<String, Vec<f64>> {
    println!("\n=== Real Dataset Evaluation ===\n");
    println!("Skipped (enable 'eval-advanced' feature to download datasets)\n");
    HashMap::new()
}

// =============================================================================
// Statistical Analysis
// =============================================================================

fn significance_analysis(f1_scores: &HashMap<String, Vec<f64>>) {
    println!("\n=== Statistical Significance ===\n");

    let names: Vec<&String> = f1_scores.keys().collect();
    if names.len() < 2 {
        println!("Need at least 2 backends for comparison\n");
        return;
    }

    // Compare each pair
    for i in 0..names.len() {
        for j in (i + 1)..names.len() {
            let name_a = names[i];
            let name_b = names[j];

            let scores_a = &f1_scores[name_a];
            let scores_b = &f1_scores[name_b];

            // Align lengths (take min)
            let n = scores_a.len().min(scores_b.len());
            if n < 2 {
                continue;
            }

            let test = compare_ner_systems(name_a, &scores_a[..n], name_b, &scores_b[..n]);

            let sig = if test.significant_01 {
                "**"
            } else if test.significant_05 {
                "*"
            } else {
                ""
            };

            println!(
                "{} vs {}: diff={:+.1}% {}",
                name_a,
                name_b,
                test.difference * 100.0,
                sig
            );
        }
    }
    println!("\n* p<0.05, ** p<0.01");
}

// =============================================================================
// Error Analysis Demo
// =============================================================================

fn error_analysis_demo(backends: &[Backend]) {
    println!("\n=== Error Analysis Sample ===\n");

    let sample_texts = vec![
        "Dr. Sarah Johnson, CEO of Microsoft, announced the $5 billion deal in New York.",
        "The meeting is scheduled for January 15, 2025 at 3:00 PM.",
        "Contact john.doe@example.com or call +1-555-123-4567 for details.",
    ];

    // Create ground truth for sample
    let sample_gold: Vec<(String, Vec<GoldEntity>)> = sample_texts
        .iter()
        .map(|&text| {
            let mut entities = vec![];
            // Add known entities (simplified for demo)
            if text.contains("Sarah Johnson") {
                entities.push(GoldEntity::new(
                    "Sarah Johnson",
                    anno::EntityType::Person,
                    4,
                ));
            }
            if text.contains("Microsoft") {
                entities.push(GoldEntity::new(
                    "Microsoft",
                    anno::EntityType::Organization,
                    29,
                ));
            }
            if text.contains("New York") {
                entities.push(GoldEntity::new("New York", anno::EntityType::Location, 71));
            }
            (text.to_string(), entities)
        })
        .collect();

    // Pick one backend for demo
    if let Some(backend) = backends.iter().find(|b| b.name == "StackedNER") {
        let mut confusion = ConfusionMatrix::new();

        for (text, gold) in &sample_gold {
            if let Ok(predicted) = backend.model.extract_entities(text, None) {
                let analysis = ErrorAnalysis::analyze(text, &predicted, gold);

                // Build confusion matrix
                for pred in &predicted {
                    let pred_type = pred.entity_type.as_label();
                    if let Some(g) = gold
                        .iter()
                        .find(|g| pred.start < g.end && pred.end > g.start)
                    {
                        let gold_type = g.entity_type.as_label();
                        confusion.add(pred_type, gold_type);
                    }
                }

                if !analysis.errors.is_empty() {
                    println!("Text: \"{}...\"", &text[..text.len().min(50)]);
                    println!("{}", analysis.summary());
                }
            }
        }

        if !confusion.types().is_empty() {
            println!("Confusion Matrix:\n{}", confusion);

            let confused = confusion.most_confused(3);
            if !confused.is_empty() {
                println!("Most confused pairs:");
                for (pred, actual, count) in confused {
                    println!("  {} -> {} ({}x)", pred, actual, count);
                }
            }
        }
    }
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    println!("NER Evaluation Suite");
    println!("====================");

    let backends = create_backends();
    println!(
        "\nBackends: {}",
        backends
            .iter()
            .map(|b| b.name)
            .collect::<Vec<_>>()
            .join(", ")
    );

    // Synthetic evaluation (always runs)
    let mut all_scores = evaluate_synthetic(&backends);

    // Real dataset evaluation (requires eval-advanced feature)
    let real_scores = evaluate_real_datasets(&backends);
    for (name, scores) in real_scores {
        all_scores.entry(name).or_default().extend(scores);
    }

    // Statistical analysis
    significance_analysis(&all_scores);

    // Error analysis demo
    error_analysis_demo(&backends);

    println!("\n=== Done ===");
}
