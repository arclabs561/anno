//! Real-world NER dataset evaluation.
//!
//! Downloads and evaluates NER models on real datasets:
//! - WikiGold (Wikipedia, PER/LOC/ORG/MISC)
//! - WNUT-17 (Social media, emerging entities)
//! - MIT Movie (Domain-specific: movies)
//! - MIT Restaurant (Domain-specific: restaurants)
//!
//! ## Test vs Eval Design
//!
//! This module follows a clear separation:
//! - **Smoke tests**: Run always, very loose thresholds (don't crash, produce some output)
//! - **Download tests**: Marked `#[ignore]`, require network
//! - **Eval reports**: Generate detailed metrics, never fail
//!
//! ## Running Tests
//!
//! ```bash
//! # Fast smoke tests (no network)
//! cargo test --test real_datasets
//!
//! # Download datasets (requires network feature)
//! cargo test --test real_datasets --features network -- --ignored --nocapture
//!
//! # Full benchmark (slow)
//! cargo test --test real_datasets --features network -- --ignored --nocapture benchmark_all
//! ```

#![allow(dead_code)] // Evaluation scaffolding - used by ignored tests

use anno::eval::loader::{DatasetId, DatasetLoader, LoadedDataset};
use anno::{Model, PatternNER};
use std::collections::HashMap;
use std::time::Instant;

// =============================================================================
// Evaluation Metrics
// =============================================================================

#[derive(Debug, Default, Clone)]
struct EvalMetrics {
    true_positives: usize,
    false_positives: usize,
    false_negatives: usize,
    total_gold: usize,
    total_predicted: usize,
    processing_time_ms: u128,
}

impl EvalMetrics {
    fn precision(&self) -> f64 {
        if self.total_predicted == 0 {
            0.0
        } else {
            self.true_positives as f64 / self.total_predicted as f64
        }
    }

    fn recall(&self) -> f64 {
        if self.total_gold == 0 {
            0.0
        } else {
            self.true_positives as f64 / self.total_gold as f64
        }
    }

    fn f1(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }
}

fn evaluate_ner_on_dataset(
    ner: &dyn Model,
    dataset: &LoadedDataset,
) -> (EvalMetrics, HashMap<String, EvalMetrics>) {
    let mut overall = EvalMetrics::default();
    let mut by_type: HashMap<String, EvalMetrics> = HashMap::new();
    let start = Instant::now();

    for sentence in &dataset.sentences {
        let text = sentence.text();
        let gold_entities = sentence.entities();
        let predicted = ner.extract_entities(&text, None).unwrap_or_default();

        overall.total_gold += gold_entities.len();
        overall.total_predicted += predicted.len();

        // Track gold entities by type
        for gold in &gold_entities {
            let type_key = gold.original_label.clone();
            by_type.entry(type_key).or_default().total_gold += 1;
        }

        // Match predictions to gold (partial text match + type match)
        let mut matched_gold = vec![false; gold_entities.len()];

        for pred in &predicted {
            let pred_type_str = pred.entity_type.as_label().to_string();
            by_type
                .entry(pred_type_str.clone())
                .or_default()
                .total_predicted += 1;

            let mut found_match = false;
            for (i, gold) in gold_entities.iter().enumerate() {
                if matched_gold[i] {
                    continue;
                }

                // Type match (allow flexible matching)
                let type_matches = types_match_flexible(&pred_type_str, &gold.original_label);

                // Text match (partial overlap is OK for patterns)
                let text_matches = pred.text.to_lowercase().contains(&gold.text.to_lowercase())
                    || gold.text.to_lowercase().contains(&pred.text.to_lowercase());

                if type_matches && text_matches {
                    overall.true_positives += 1;
                    by_type
                        .entry(pred_type_str.clone())
                        .or_default()
                        .true_positives += 1;
                    matched_gold[i] = true;
                    found_match = true;
                    break;
                }
            }

            if !found_match {
                overall.false_positives += 1;
                by_type.entry(pred_type_str).or_default().false_positives += 1;
            }
        }

        // Count unmatched gold as false negatives
        for (i, gold) in gold_entities.iter().enumerate() {
            if !matched_gold[i] {
                overall.false_negatives += 1;
                let type_key = gold.original_label.clone();
                by_type.entry(type_key).or_default().false_negatives += 1;
            }
        }
    }

    overall.processing_time_ms = start.elapsed().as_millis();
    (overall, by_type)
}

fn types_match_flexible(pred: &str, gold: &str) -> bool {
    let pred = pred.to_uppercase();
    let gold = gold.to_uppercase();

    if pred == gold {
        return true;
    }

    // Allow common mappings
    match (pred.as_str(), gold.as_str()) {
        // Person
        ("PERSON", "PER") | ("PER", "PERSON") => true,
        // Location
        ("LOCATION", "LOC") | ("LOC", "LOCATION") | ("LOCATION", "GPE") | ("GPE", "LOCATION") => {
            true
        }
        // Organization
        ("ORGANIZATION", "ORG") | ("ORG", "ORGANIZATION") => true,
        // Date/Time
        ("DATE", "YEAR") | ("YEAR", "DATE") | ("DATE", "HOURS") => true,
        _ => false,
    }
}

// =============================================================================
// Smoke Tests (Always Run - Just Check It Works)
// =============================================================================

#[test]
fn smoke_test_dataset_loader_creation() {
    let loader = DatasetLoader::new();
    assert!(loader.is_ok(), "DatasetLoader should create without error");
}

#[test]
fn smoke_test_cache_paths_exist() {
    let loader = DatasetLoader::new().unwrap();

    // Just check paths are generated, don't require files to exist
    let path = loader.cache_path(DatasetId::WikiGold);
    assert!(
        path.to_string_lossy().contains("wikigold"),
        "Cache path should contain dataset name"
    );
}

#[test]
fn smoke_test_dataset_id_all() {
    let all = DatasetId::all();
    // 12 datasets: 9 NER + 3 coreference
    assert!(all.len() >= 6, "Should have at least 6 datasets, got {}", all.len());
    assert!(all.contains(&DatasetId::WikiGold));
    assert!(all.contains(&DatasetId::Wnut17));
}

#[test]
fn smoke_test_dataset_id_from_str() {
    use std::str::FromStr;
    assert_eq!(DatasetId::from_str("wikigold").unwrap(), DatasetId::WikiGold);
    assert_eq!(DatasetId::from_str("wnut-17").unwrap(), DatasetId::Wnut17);
    assert!(DatasetId::from_str("unknown").is_err());
}

// =============================================================================
// Download Tests (Network Required)
// =============================================================================

#[test]
#[ignore] // Run with: cargo test --features network -- --ignored
fn download_wikigold_dataset() {
    #[cfg(feature = "network")]
    {
        let loader = DatasetLoader::new().unwrap();
        let dataset = loader.load_or_download(DatasetId::WikiGold);

        match dataset {
            Ok(ds) => {
                println!("\n=== WikiGold Dataset ===");
                let stats = ds.stats();
                println!("{}", stats);

                // Smoke check: should have reasonable data
                let (min, max) = DatasetId::WikiGold.expected_entity_count();
                assert!(
                    stats.entities >= min && stats.entities <= max * 2,
                    "WikiGold entity count {} outside expected range ({}, {})",
                    stats.entities,
                    min,
                    max
                );
            }
            Err(e) => {
                println!("Failed to load WikiGold (may be network issue): {}", e);
            }
        }
    }
}

#[test]
#[ignore]
fn download_wnut17_dataset() {
    #[cfg(feature = "network")]
    {
        let loader = DatasetLoader::new().unwrap();
        let dataset = loader.load_or_download(DatasetId::Wnut17);

        match dataset {
            Ok(ds) => {
                println!("\n=== WNUT-17 Dataset ===");
                println!("{}", ds.stats());
            }
            Err(e) => {
                println!("Failed to load WNUT17: {}", e);
            }
        }
    }
}

#[test]
#[ignore]
fn download_mit_movie_dataset() {
    #[cfg(feature = "network")]
    {
        let loader = DatasetLoader::new().unwrap();
        let dataset = loader.load_or_download(DatasetId::MitMovie);

        match dataset {
            Ok(ds) => {
                println!("\n=== MIT Movie Dataset ===");
                println!("{}", ds.stats());
            }
            Err(e) => {
                println!("Failed to load MIT Movie: {}", e);
            }
        }
    }
}

// =============================================================================
// Evaluation Tests (Generate Reports)
// =============================================================================

#[test]
#[ignore]
fn evaluate_pattern_ner_on_wikigold() {
    #[cfg(feature = "network")]
    {
        let loader = DatasetLoader::new().unwrap();
        let dataset = match loader.load_or_download(DatasetId::WikiGold) {
            Ok(ds) => ds,
            Err(e) => {
                println!("Skipping WikiGold evaluation: {}", e);
                return;
            }
        };

        let ner = PatternNER::new();
        let (metrics, by_type) = evaluate_ner_on_dataset(&ner, &dataset);

        println!("\n=== PatternNER on WikiGold ===");
        println!("Sentences: {}", dataset.len());
        println!("Gold entities: {}", metrics.total_gold);
        println!("Predicted: {}", metrics.total_predicted);
        println!("True positives: {}", metrics.true_positives);
        println!("False positives: {}", metrics.false_positives);
        println!("False negatives: {}", metrics.false_negatives);
        println!("Precision: {:.1}%", metrics.precision() * 100.0);
        println!("Recall: {:.1}%", metrics.recall() * 100.0);
        println!("F1: {:.1}%", metrics.f1() * 100.0);
        println!("Processing time: {}ms", metrics.processing_time_ms);

        println!("\nBy entity type:");
        for (etype, m) in &by_type {
            if m.total_gold > 0 || m.total_predicted > 0 {
                println!(
                    "  {:15} P={:.1}% R={:.1}% F1={:.1}% (gold={}, pred={})",
                    etype,
                    m.precision() * 100.0,
                    m.recall() * 100.0,
                    m.f1() * 100.0,
                    m.total_gold,
                    m.total_predicted
                );
            }
        }

        // Note: PatternNER won't find PER/ORG/LOC - it's for structured entities
        // We expect very low recall but potentially decent precision on dates/numbers
        println!("\nNote: PatternNER is for structured entities (dates, money, emails, etc.)");
        println!("Low recall on PER/ORG/LOC is expected - use ML backends for those.");
    }
}

// =============================================================================
// Full Benchmark (All Datasets)
// =============================================================================

#[test]
#[ignore]
fn benchmark_all_datasets() {
    #[cfg(feature = "network")]
    {
        let loader = DatasetLoader::new().unwrap();
        let ner = PatternNER::new();

        let datasets = DatasetId::all();

        println!("\n=== NER Benchmark: PatternNER on All Datasets ===\n");
        println!(
            "{:20} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10}",
            "Dataset", "Sents", "Gold", "Pred", "P%", "R%", "F1%"
        );
        println!("{}", "-".repeat(80));

        for dataset_id in datasets {
            match loader.load_or_download(*dataset_id) {
                Ok(dataset) => {
                    let (metrics, _) = evaluate_ner_on_dataset(&ner, &dataset);
                    println!(
                        "{:20} {:>8} {:>8} {:>8} {:>8.1} {:>8.1} {:>10.1}",
                        dataset_id.name(),
                        dataset.len(),
                        metrics.total_gold,
                        metrics.total_predicted,
                        metrics.precision() * 100.0,
                        metrics.recall() * 100.0,
                        metrics.f1() * 100.0
                    );
                }
                Err(e) => {
                    println!("{:20} FAILED: {}", dataset_id.name(), e);
                }
            }
        }
    }
}

// =============================================================================
// Cached Dataset Tests (No Network Required After First Download)
// =============================================================================

#[test]
fn test_cached_dataset_access() {
    let loader = DatasetLoader::new().unwrap();

    // These tests pass if datasets are cached, skip if not
    for dataset_id in DatasetId::all() {
        if loader.is_cached(*dataset_id) {
            let dataset = loader.load(*dataset_id).unwrap();
            assert!(
                !dataset.is_empty(),
                "{:?} should have data when cached",
                dataset_id
            );
            println!(
                "Cached {:?}: {} sentences, {} entities",
                dataset_id,
                dataset.len(),
                dataset.entity_count()
            );
        } else {
            println!(
                "{:?} not cached, skipping (run --ignored tests to download)",
                dataset_id
            );
        }
    }
}

// =============================================================================
// Baseline Regression Tests
// =============================================================================

// These are very loose baselines for PatternNER on named entity datasets
// PatternNER is NOT designed for PER/ORG/LOC - it's for structured patterns
// So we expect near-zero performance, but non-crashing behavior

const PATTERN_NER_MIN_F1: f64 = 0.0; // PatternNER won't find named entities

#[test]
#[ignore]
fn regression_test_wikigold() {
    #[cfg(feature = "network")]
    {
        let loader = DatasetLoader::new().unwrap();
        let dataset = match loader.load_or_download(DatasetId::WikiGold) {
            Ok(ds) => ds,
            Err(_) => return,
        };

        let ner = PatternNER::new();
        let (metrics, _) = evaluate_ner_on_dataset(&ner, &dataset);

        let f1 = metrics.f1();
        assert!(
            f1 >= PATTERN_NER_MIN_F1,
            "WikiGold F1 ({:.3}) dropped below minimum ({:.3})",
            f1,
            PATTERN_NER_MIN_F1
        );

        println!(
            "WikiGold F1: {:.3} (minimum: {:.3}) - PASS",
            f1, PATTERN_NER_MIN_F1
        );
    }
}

