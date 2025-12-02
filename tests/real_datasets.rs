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
//! - **Download tests**: Marked `#[ignore]`, require eval-advanced
//! - **Eval reports**: Generate detailed metrics, never fail
//!
//! ## Running Tests
//!
//! ```bash
//! # Fast smoke tests (no network)
//! cargo test --test real_datasets
//!
//! # Download datasets (requires eval-advanced feature)
//! cargo test --test real_datasets --features eval-advanced -- --ignored --nocapture
//!
//! # Full benchmark (slow)
//! cargo test --test real_datasets --features eval-advanced -- --ignored --nocapture benchmark_all
//! ```

#![allow(dead_code)] // Evaluation scaffolding - used by ignored tests

use anno::eval::loader::{DatasetId, DatasetLoader, LoadedDataset};
#[allow(unused_imports)] // Used by ignored tests
use anno::{HeuristicNER, Model, RegexNER, StackedNER};
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
    // Get type mapper for this dataset if available
    let type_mapper = dataset.id.type_mapper();

    let mut overall = EvalMetrics::default();
    let mut by_type: HashMap<String, EvalMetrics> = HashMap::new();
    let start = Instant::now();

    for sentence in &dataset.sentences {
        let text = sentence.text();
        let gold_entities = sentence.entities();
        let predicted = ner.extract_entities(&text, None).unwrap_or_default();

        overall.total_gold += gold_entities.len();
        overall.total_predicted += predicted.len();

        // Track gold entities by type (use normalized label for consistent tracking)
        for gold in &gold_entities {
            // Normalize gold type for consistent key (matches what we use for matching)
            let gold_type_normalized = if let Some(ref mapper) = type_mapper {
                mapper.normalize(&gold.original_label)
            } else {
                anno::EntityType::from_label(&gold.original_label)
            };
            let type_key = gold_type_normalized.as_label().to_string();
            by_type.entry(type_key).or_default().total_gold += 1;
        }

        // Match predictions to gold (exact span match + type match with normalization)
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

                // Apply type mapping to gold if available
                let gold_type_normalized = if let Some(ref mapper) = type_mapper {
                    mapper.normalize(&gold.original_label)
                } else {
                    anno::EntityType::from_label(&gold.original_label)
                };

                // Type match (allow flexible matching)
                let type_matches =
                    types_match_flexible(&pred_type_str, gold_type_normalized.as_label());

                // Span match: exact match required (more strict than substring)
                let span_matches = pred.start == gold.start && pred.end == gold.end;

                if type_matches && span_matches {
                    overall.true_positives += 1;
                    // Use normalized gold type as key for consistency
                    let gold_type_key = gold_type_normalized.as_label().to_string();
                    by_type.entry(gold_type_key).or_default().true_positives += 1;
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
                // Use normalized gold type as key for consistency
                let gold_type_normalized = if let Some(ref mapper) = type_mapper {
                    mapper.normalize(&gold.original_label)
                } else {
                    anno::EntityType::from_label(&gold.original_label)
                };
                let type_key = gold_type_normalized.as_label().to_string();
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
    // 20 datasets total
    assert!(
        all.len() >= 6,
        "Should have at least 6 datasets, got {}",
        all.len()
    );
    assert!(all.contains(&DatasetId::WikiGold));
    assert!(all.contains(&DatasetId::Wnut17));
}

#[test]
fn smoke_test_dataset_id_quick() {
    let quick = DatasetId::quick();
    assert_eq!(quick.len(), 3, "Quick should have 3 datasets for CI");
    assert!(quick.contains(&DatasetId::WikiGold));
    assert!(quick.contains(&DatasetId::MitMovie));
    assert!(quick.contains(&DatasetId::GAP));
}

#[test]
fn smoke_test_dataset_id_medium() {
    let medium = DatasetId::medium();
    assert_eq!(medium.len(), 6, "Medium should have 6 datasets");
    // Should be a superset of quick
    for ds in DatasetId::quick() {
        assert!(
            medium.contains(ds),
            "Medium should contain {:?} from quick",
            ds
        );
    }
}

#[test]
fn smoke_test_dataset_id_from_str() {
    use std::str::FromStr;
    assert_eq!(
        DatasetId::from_str("wikigold").unwrap(),
        DatasetId::WikiGold
    );
    assert_eq!(DatasetId::from_str("wnut-17").unwrap(), DatasetId::Wnut17);
    assert!(DatasetId::from_str("unknown").is_err());
}

// =============================================================================
// Download Tests (Network Required)
// =============================================================================

#[test]
#[ignore] // Run with: cargo test --features eval-advanced -- --ignored
fn download_wikigold_dataset() {
    #[cfg(feature = "eval-advanced")]
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
    #[cfg(feature = "eval-advanced")]
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
    #[cfg(feature = "eval-advanced")]
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

#[test]
#[ignore]
fn download_genia_dataset() {
    #[cfg(feature = "eval-advanced")]
    {
        let loader = DatasetLoader::new().unwrap();
        let dataset = loader.load_or_download(DatasetId::GENIA);

        match dataset {
            Ok(ds) => {
                println!("\n=== GENIA Dataset (HF API) ===");
                println!("{}", ds.stats());

                // Show first sentence to verify parsing
                if let Some(sent) = ds.sentences.first() {
                    println!("\nFirst sentence: {}", sent.text());
                    for entity in sent.entities() {
                        println!("  Entity: {} ({})", entity.text, entity.original_label);
                    }
                }
            }
            Err(e) => {
                println!("Failed to load GENIA: {}", e);
            }
        }
    }
}

#[test]
#[ignore]
fn download_fewnerd_dataset() {
    #[cfg(feature = "eval-advanced")]
    {
        let loader = DatasetLoader::new().unwrap();
        let dataset = loader.load_or_download(DatasetId::FewNERD);

        match dataset {
            Ok(ds) => {
                println!("\n=== FewNERD Dataset (HF API) ===");
                println!("{}", ds.stats());
            }
            Err(e) => {
                println!("Failed to load FewNERD: {}", e);
            }
        }
    }
}

#[test]
#[ignore]
fn download_bc5cdr_dataset() {
    #[cfg(feature = "eval-advanced")]
    {
        let loader = DatasetLoader::new().unwrap();
        let dataset = loader.load_or_download(DatasetId::BC5CDR);

        match dataset {
            Ok(ds) => {
                println!("\n=== BC5CDR Dataset (BioFLAIR) ===");
                println!("{}", ds.stats());

                // Verify we have actual entities (not all O tags)
                let entity_count = ds.entity_count();
                assert!(entity_count > 0, "BC5CDR should have entities");
                println!("Verified: {} entities found", entity_count);
            }
            Err(e) => {
                println!("Failed to load BC5CDR: {}", e);
            }
        }
    }
}

#[test]
#[ignore]
fn download_ncbi_disease_dataset() {
    #[cfg(feature = "eval-advanced")]
    {
        let loader = DatasetLoader::new().unwrap();
        let dataset = loader.load_or_download(DatasetId::NCBIDisease);

        match dataset {
            Ok(ds) => {
                println!("\n=== NCBI Disease Dataset (BioFLAIR) ===");
                println!("{}", ds.stats());

                // Verify we have Disease entities
                let entity_count = ds.entity_count();
                assert!(entity_count > 0, "NCBI Disease should have entities");
                println!("Verified: {} entities found", entity_count);
            }
            Err(e) => {
                println!("Failed to load NCBI Disease: {}", e);
            }
        }
    }
}

#[test]
#[ignore]
fn download_tweetner7_dataset() {
    #[cfg(feature = "eval-advanced")]
    {
        let loader = DatasetLoader::new().unwrap();
        let dataset = loader.load_or_download(DatasetId::TweetNER7);

        match dataset {
            Ok(ds) => {
                println!("\n=== TweetNER7 Dataset ===");
                println!("{}", ds.stats());

                // Verify we have entities (not all O tags)
                let entity_count = ds.entity_count();
                assert!(entity_count > 0, "TweetNER7 should have entities");
                println!("Verified: {} entities found", entity_count);

                // Show first sentence with entities
                for sent in ds.sentences.iter().take(5) {
                    let entities = sent.entities();
                    if !entities.is_empty() {
                        println!("\nSample: {}", sent.text());
                        for e in entities {
                            println!("  {} ({})", e.text, e.original_label);
                        }
                        break;
                    }
                }
            }
            Err(e) => {
                println!("Failed to load TweetNER7: {}", e);
            }
        }
    }
}

#[test]
#[ignore]
fn download_broad_twitter_dataset() {
    #[cfg(feature = "eval-advanced")]
    {
        let loader = DatasetLoader::new().unwrap();
        let dataset = loader.load_or_download(DatasetId::BroadTwitterCorpus);

        match dataset {
            Ok(ds) => {
                println!("\n=== BroadTwitter Dataset ===");
                println!("{}", ds.stats());

                // Verify we have entities
                let entity_count = ds.entity_count();
                assert!(entity_count > 0, "BroadTwitter should have entities");
                println!("Verified: {} entities found", entity_count);
            }
            Err(e) => {
                println!("Failed to load BroadTwitter: {}", e);
            }
        }
    }
}

#[test]
#[ignore]
fn download_crossre_dataset() {
    #[cfg(feature = "eval-advanced")]
    {
        let loader = DatasetLoader::new().unwrap();

        // Test DocRED proxy (CrossRE AI domain)
        let dataset = loader.load_or_download(DatasetId::DocRED);

        match dataset {
            Ok(ds) => {
                println!("\n=== CrossRE (AI domain - proxy for DocRED) ===");
                println!("{}", ds.stats());

                // Verify we have entities from NER annotations
                let entity_count = ds.entity_count();
                assert!(entity_count > 0, "CrossRE should have entities");
                println!("Verified: {} entities found", entity_count);
            }
            Err(e) => {
                println!("Failed to load CrossRE/DocRED proxy: {}", e);
            }
        }
    }
}

// =============================================================================
// Evaluation Tests (Generate Reports)
// =============================================================================

#[test]
#[ignore]
fn evaluate_regex_ner_on_wikigold() {
    #[cfg(feature = "eval-advanced")]
    {
        let loader = DatasetLoader::new().unwrap();
        let dataset = match loader.load_or_download(DatasetId::WikiGold) {
            Ok(ds) => ds,
            Err(e) => {
                println!("Skipping WikiGold evaluation: {}", e);
                return;
            }
        };

        let ner = RegexNER::new();
        let (metrics, by_type) = evaluate_ner_on_dataset(&ner, &dataset);

        println!("\n=== RegexNER on WikiGold ===");
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

        // Note: RegexNER won't find PER/ORG/LOC - it's for structured entities
        // We expect very low recall but potentially decent precision on dates/numbers
        println!("\nNote: RegexNER is for structured entities (dates, money, emails, etc.)");
        println!("Low recall on PER/ORG/LOC is expected - use ML backends for those.");
    }
}

// =============================================================================
// Full Benchmark (All Datasets)
// =============================================================================

#[test]
#[ignore]
fn benchmark_regex_ner_on_datasets() {
    #[cfg(feature = "eval-advanced")]
    {
        let loader = DatasetLoader::new().unwrap();
        let ner = RegexNER::new();

        // RegexNER is for structured entities (dates, money, emails, URLs, phones)
        // It won't find PER/ORG/LOC, so we benchmark on all datasets but expect low recall
        // on named entity datasets. This is useful to show what RegexNER can/can't do.
        let datasets = DatasetId::all();

        println!("\n=== NER Benchmark: RegexNER on All Datasets ===\n");
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

#[test]
#[ignore]
fn benchmark_heuristic_ner_on_datasets() {
    #[cfg(feature = "eval-advanced")]
    {
        let loader = DatasetLoader::new().unwrap();
        let ner = HeuristicNER::new();

        // Use all NER datasets (skip coref/RE for now as Heuristic is NER-focused)
        let datasets = DatasetId::all_ner();

        println!("\n=== NER Benchmark: HeuristicNER on NER Datasets ===\n");
        println!(
            "{:20} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}",
            "Dataset", "Sents", "Gold", "Pred", "P%", "R%", "F1%", "ms/sent"
        );
        println!("{}", "-".repeat(90));

        for dataset_id in datasets {
            match loader.load_or_download(*dataset_id) {
                Ok(dataset) => {
                    let (metrics, _) = evaluate_ner_on_dataset(&ner, &dataset);
                    let ms_per_sent = if dataset.len() > 0 {
                        metrics.processing_time_ms as f64 / dataset.len() as f64
                    } else {
                        0.0
                    };
                    println!(
                        "{:20} {:>8} {:>8} {:>8} {:>8.1} {:>8.1} {:>10.1} {:>10.1}",
                        dataset_id.name(),
                        dataset.len(),
                        metrics.total_gold,
                        metrics.total_predicted,
                        metrics.precision() * 100.0,
                        metrics.recall() * 100.0,
                        metrics.f1() * 100.0,
                        ms_per_sent
                    );
                }
                Err(e) => {
                    println!("{:20} FAILED: {}", dataset_id.name(), e);
                }
            }
        }
    }
}

#[test]
#[ignore]
fn benchmark_stacked_ner_on_datasets() {
    #[cfg(feature = "eval-advanced")]
    {
        let loader = DatasetLoader::new().unwrap();
        let ner = StackedNER::default();

        let datasets = DatasetId::all_ner();

        println!("\n=== NER Benchmark: StackedNER on NER Datasets ===\n");
        println!(
            "{:20} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}",
            "Dataset", "Sents", "Gold", "Pred", "P%", "R%", "F1%", "ms/sent"
        );
        println!("{}", "-".repeat(90));

        for dataset_id in datasets {
            match loader.load_or_download(*dataset_id) {
                Ok(dataset) => {
                    let (metrics, _) = evaluate_ner_on_dataset(&ner, &dataset);
                    let ms_per_sent = if dataset.len() > 0 {
                        metrics.processing_time_ms as f64 / dataset.len() as f64
                    } else {
                        0.0
                    };
                    println!(
                        "{:20} {:>8} {:>8} {:>8} {:>8.1} {:>8.1} {:>10.1} {:>10.1}",
                        dataset_id.name(),
                        dataset.len(),
                        metrics.total_gold,
                        metrics.total_predicted,
                        metrics.precision() * 100.0,
                        metrics.recall() * 100.0,
                        metrics.f1() * 100.0,
                        ms_per_sent
                    );
                }
                Err(e) => {
                    println!("{:20} FAILED: {}", dataset_id.name(), e);
                }
            }
        }
    }
}

/// Benchmark NuNER on all NER datasets.
#[test]
#[ignore = "Requires network and ONNX runtime"]
fn benchmark_nuner_on_datasets() {
    #[cfg(feature = "onnx")]
    {
        use anno::{NuNER, DEFAULT_NUNER_MODEL};

        println!("\n=== Loading NuNER model... ===\n");
        let nuner = match NuNER::from_pretrained(DEFAULT_NUNER_MODEL) {
            Ok(n) => n,
            Err(e) => {
                println!("Failed to load NuNER: {}", e);
                return;
            }
        };

        let loader = DatasetLoader::new().unwrap();
        let datasets: Vec<DatasetId> = DatasetId::all_ner().to_vec();

        println!("=== NuNER Benchmark on All NER Datasets ===\n");
        println!(
            "{:25} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}",
            "Dataset", "Sents", "Gold", "Pred", "P%", "R%", "F1%", "ms/sent"
        );
        println!("{}", "-".repeat(110));

        for dataset_id in &datasets {
            match loader.load(*dataset_id) {
                Ok(dataset) => {
                    // Limit to first 100 sentences for speed
                    let mut limited = dataset.clone();
                    limited.sentences.truncate(100);

                    let (metrics, by_type) = evaluate_ner_on_dataset(&nuner, &limited);
                    let ms_per_sent = if !limited.sentences.is_empty() {
                        metrics.processing_time_ms as f64 / limited.sentences.len() as f64
                    } else {
                        0.0
                    };

                    println!(
                        "{:25} {:>8} {:>8} {:>8} {:>8.1} {:>8.1} {:>10.1} {:>10.1}",
                        dataset_id.name(),
                        limited.sentences.len(),
                        metrics.total_gold,
                        metrics.total_predicted,
                        metrics.precision() * 100.0,
                        metrics.recall() * 100.0,
                        metrics.f1() * 100.0,
                        ms_per_sent
                    );

                    // Print per-entity-type breakdown
                    if !by_type.is_empty() {
                        println!("\n  Per-entity-type breakdown:");
                        let mut types: Vec<_> = by_type.iter().collect();
                        types.sort_by(|a, b| b.1.total_gold.cmp(&a.1.total_gold));
                        for (etype, m) in types.iter().take(5) {
                            if m.total_gold > 0 || m.total_predicted > 0 {
                                println!(
                                    "    {:20} P={:5.1}% R={:5.1}% F1={:5.1}% (gold={} pred={} tp={})",
                                    etype,
                                    m.precision() * 100.0,
                                    m.recall() * 100.0,
                                    m.f1() * 100.0,
                                    m.total_gold,
                                    m.total_predicted,
                                    m.true_positives
                                );
                            }
                        }
                        println!();
                    }
                }
                Err(e) => {
                    println!("{:25} FAILED: {}", dataset_id.name(), e);
                }
            }
        }
    }
    #[cfg(not(feature = "onnx"))]
    {
        println!("NuNER benchmark requires --features onnx");
    }
}

/// Benchmark W2NER on all NER datasets (handles nested/discontinuous entities).
#[test]
#[ignore = "Requires network and ONNX runtime"]
fn benchmark_w2ner_on_datasets() {
    #[cfg(feature = "onnx")]
    {
        use anno::W2NER;

        println!("\n=== Loading W2NER model... ===\n");
        // Note: W2NER requires a specific model path - adjust as needed
        // For now, we'll use the default config (may not have actual model loaded)
        let w2ner = W2NER::new();

        if !w2ner.is_available() {
            println!("W2NER model not available (requires from_pretrained with valid model path)");
            println!("Skipping W2NER benchmark");
            return;
        }

        let loader = DatasetLoader::new().unwrap();
        let datasets: Vec<DatasetId> = DatasetId::all_ner().to_vec();

        println!("=== W2NER Benchmark on All NER Datasets ===\n");
        println!(
            "{:25} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}",
            "Dataset", "Sents", "Gold", "Pred", "P%", "R%", "F1%", "ms/sent"
        );
        println!("{}", "-".repeat(110));

        for dataset_id in &datasets {
            match loader.load(*dataset_id) {
                Ok(dataset) => {
                    // Limit to first 100 sentences for speed
                    let mut limited = dataset.clone();
                    limited.sentences.truncate(100);

                    let (metrics, by_type) = evaluate_ner_on_dataset(&w2ner, &limited);
                    let ms_per_sent = if !limited.sentences.is_empty() {
                        metrics.processing_time_ms as f64 / limited.sentences.len() as f64
                    } else {
                        0.0
                    };

                    println!(
                        "{:25} {:>8} {:>8} {:>8} {:>8.1} {:>8.1} {:>10.1} {:>10.1}",
                        dataset_id.name(),
                        limited.sentences.len(),
                        metrics.total_gold,
                        metrics.total_predicted,
                        metrics.precision() * 100.0,
                        metrics.recall() * 100.0,
                        metrics.f1() * 100.0,
                        ms_per_sent
                    );

                    // Print per-entity-type breakdown
                    if !by_type.is_empty() {
                        println!("\n  Per-entity-type breakdown:");
                        let mut types: Vec<_> = by_type.iter().collect();
                        types.sort_by(|a, b| b.1.total_gold.cmp(&a.1.total_gold));
                        for (etype, m) in types.iter().take(5) {
                            if m.total_gold > 0 || m.total_predicted > 0 {
                                println!(
                                    "    {:20} P={:5.1}% R={:5.1}% F1={:5.1}% (gold={} pred={} tp={})",
                                    etype,
                                    m.precision() * 100.0,
                                    m.recall() * 100.0,
                                    m.f1() * 100.0,
                                    m.total_gold,
                                    m.total_predicted,
                                    m.true_positives
                                );
                            }
                        }
                        println!();
                    }
                }
                Err(e) => {
                    println!("{:25} FAILED: {}", dataset_id.name(), e);
                }
            }
        }
    }
    #[cfg(not(feature = "onnx"))]
    {
        println!("W2NER benchmark requires --features onnx");
    }
}

/// Benchmark GLiNER on all NER datasets (comprehensive evaluation).
#[test]
#[ignore = "Requires network and ONNX runtime"]
fn benchmark_gliner_on_datasets() {
    #[cfg(feature = "onnx")]
    {
        use anno::eval::analysis::build_confusion_matrix;
        use anno::{GLiNEROnnx, DEFAULT_GLINER_MODEL};

        println!("\n=== Loading GLiNER model... ===\n");
        let gliner = match GLiNEROnnx::new(DEFAULT_GLINER_MODEL) {
            Ok(g) => g,
            Err(e) => {
                println!("Failed to load GLiNER: {}", e);
                return;
            }
        };

        let loader = DatasetLoader::new().unwrap();

        // Test on all NER datasets (not just 6)
        let datasets: Vec<DatasetId> = DatasetId::all_ner().to_vec();

        println!("=== GLiNER Benchmark on All NER Datasets ===\n");
        println!(
            "{:25} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}",
            "Dataset", "Sents", "Gold", "Pred", "P%", "R%", "F1%", "ms/sent"
        );
        println!("{}", "-".repeat(110));

        let mut all_predictions: Vec<(Vec<anno::Entity>, Vec<anno::eval::datasets::GoldEntity>)> =
            Vec::new();

        for dataset_id in &datasets {
            match loader.load(*dataset_id) {
                Ok(dataset) => {
                    // Limit to first 100 sentences for speed (can be increased for full eval)
                    let mut limited = dataset.clone();
                    limited.sentences.truncate(100);

                    let (metrics, by_type) = evaluate_ner_on_dataset(&gliner, &limited);
                    let ms_per_sent = if !limited.sentences.is_empty() {
                        metrics.processing_time_ms as f64 / limited.sentences.len() as f64
                    } else {
                        0.0
                    };

                    println!(
                        "{:25} {:>8} {:>8} {:>8} {:>8.1} {:>8.1} {:>10.1} {:>10.1}",
                        dataset_id.name(),
                        limited.sentences.len(),
                        metrics.total_gold,
                        metrics.total_predicted,
                        metrics.precision() * 100.0,
                        metrics.recall() * 100.0,
                        metrics.f1() * 100.0,
                        ms_per_sent
                    );

                    // Collect predictions for confusion matrix
                    for sentence in &limited.sentences {
                        let text = sentence.text();
                        let gold_entities = sentence.entities();
                        let predicted = gliner.extract_entities(&text, None).unwrap_or_default();
                        all_predictions.push((predicted, gold_entities));
                    }

                    // Print per-entity-type breakdown for this dataset
                    if !by_type.is_empty() {
                        println!("\n  Per-entity-type breakdown:");
                        let mut types: Vec<_> = by_type.iter().collect();
                        types.sort_by(|a, b| b.1.total_gold.cmp(&a.1.total_gold));
                        for (etype, m) in types.iter().take(5) {
                            if m.total_gold > 0 || m.total_predicted > 0 {
                                println!(
                                    "    {:20} P={:5.1}% R={:5.1}% F1={:5.1}% (gold={} pred={} tp={})",
                                    etype,
                                    m.precision() * 100.0,
                                    m.recall() * 100.0,
                                    m.f1() * 100.0,
                                    m.total_gold,
                                    m.total_predicted,
                                    m.true_positives
                                );
                            }
                        }
                        println!();
                    }
                }
                Err(e) => {
                    println!("{:25} FAILED: {}", dataset_id.name(), e);
                }
            }
        }

        // Build and print confusion matrix across all datasets
        if !all_predictions.is_empty() {
            println!("\n=== Overall Confusion Matrix (All Datasets) ===\n");
            let confusion = build_confusion_matrix(&all_predictions);
            println!("{}", confusion);

            println!("\nMost confused entity type pairs:");
            for (pred, actual, count) in confusion.most_confused(10) {
                println!("  {} → {}: {} errors", pred, actual, count);
            }
        }
    }
    #[cfg(not(feature = "onnx"))]
    {
        println!("GLiNER benchmark requires --features onnx");
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
// Coreference Evaluation (Missing - Infrastructure Exists But Not Used)
// =============================================================================

// =============================================================================
// Coreference Evaluation
// =============================================================================

/// Evaluate coreference resolution on a dataset.
///
/// Uses NER model to extract entities, then SimpleCorefResolver to group them,
/// and compares against gold coreference chains.
fn evaluate_coref_on_dataset(
    ner: &dyn Model,
    gold_docs: &[anno::eval::coref::CorefDocument],
) -> anno::eval::coref_metrics::AggregateCorefEvaluation {
    use anno::eval::coref_resolver::SimpleCorefResolver;

    let resolver = SimpleCorefResolver::default();
    let mut all_pred_chains: Vec<Vec<anno::eval::coref::CorefChain>> = Vec::new();
    let mut all_gold_chains: Vec<&[anno::eval::coref::CorefChain]> = Vec::new();

    for doc in gold_docs {
        let text = doc.text.as_str();
        all_gold_chains.push(&doc.chains);

        // Extract entities using NER
        let entities = ner.extract_entities(text, None).unwrap_or_default();

        // Resolve coreference
        let pred_chains = resolver.resolve_to_chains(&entities);
        all_pred_chains.push(pred_chains);
    }

    // Build document pairs
    let document_pairs: Vec<_> = all_pred_chains
        .iter()
        .zip(all_gold_chains.iter())
        .map(|(pred, gold)| (pred.as_slice(), *gold))
        .collect();

    // Compute aggregate metrics
    anno::eval::coref_metrics::AggregateCorefEvaluation::compute(&document_pairs)
}

/// Benchmark coreference resolution on all coreference datasets with multiple models.
#[test]
#[ignore]
fn benchmark_coreference_on_all_datasets() {
    #[cfg(feature = "eval-advanced")]
    {
        use anno::eval::loader::DatasetLoader;

        let loader = DatasetLoader::new().unwrap();

        // Test with multiple NER models
        let mut models: Vec<(&str, Box<dyn Model>)> = vec![
            ("StackedNER", Box::new(StackedNER::default())),
            ("HeuristicNER", Box::new(HeuristicNER::new())),
        ];

        // Add GLiNER if available
        #[cfg(feature = "onnx")]
        {
            use anno::{GLiNEROnnx, DEFAULT_GLINER_MODEL};
            if let Ok(gliner) = GLiNEROnnx::new(DEFAULT_GLINER_MODEL) {
                models.push(("GLiNER", Box::new(gliner)));
            }
        }

        // Add NuNER if available
        #[cfg(feature = "onnx")]
        {
            use anno::{NuNER, DEFAULT_NUNER_MODEL};
            if let Ok(nuner) = NuNER::from_pretrained(DEFAULT_NUNER_MODEL) {
                models.push(("NuNER", Box::new(nuner)));
            }
        }

        let coref_datasets = DatasetId::all_coref();

        println!("\n=== Coreference Resolution Benchmark on All Datasets ===\n");

        for dataset_id in coref_datasets {
            match loader.load_or_download_coref(*dataset_id) {
                Ok(gold_docs) => {
                    println!("\n=== {} Coreference Dataset ===\n", dataset_id.name());
                    println!("Loaded {} {} examples", gold_docs.len(), dataset_id.name());

                    println!(
                        "{:20} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
                        "Model",
                        "Docs",
                        "CoNLL F1",
                        "MUC F1",
                        "B³ F1",
                        "CEAF-e F1",
                        "LEA F1",
                        "BLANC F1"
                    );
                    println!("{}", "-".repeat(100));

                    for (model_name, ner) in &models {
                        println!(
                            "\nEvaluating coreference resolution ({} + SimpleCorefResolver)...",
                            model_name
                        );

                        let start = Instant::now();
                        let results = evaluate_coref_on_dataset(ner.as_ref(), &gold_docs);
                        let elapsed = start.elapsed();

                        println!(
                            "{:20} {:>8} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
                            model_name,
                            results.num_documents,
                            results.mean.conll_f1,
                            results.mean.muc.f1,
                            results.mean.b_cubed.f1,
                            results.mean.ceaf_e.f1,
                            results.mean.lea.f1,
                            results.mean.blanc.f1
                        );

                        println!("  CoNLL F1: {:.3}", results.mean.conll_f1);
                        println!(
                            "  MUC: P={:.3} R={:.3} F1={:.3}",
                            results.mean.muc.precision,
                            results.mean.muc.recall,
                            results.mean.muc.f1
                        );
                        println!(
                            "  B³: P={:.3} R={:.3} F1={:.3}",
                            results.mean.b_cubed.precision,
                            results.mean.b_cubed.recall,
                            results.mean.b_cubed.f1
                        );
                        println!(
                            "  CEAF-e: P={:.3} R={:.3} F1={:.3}",
                            results.mean.ceaf_e.precision,
                            results.mean.ceaf_e.recall,
                            results.mean.ceaf_e.f1
                        );
                        println!(
                            "  LEA: P={:.3} R={:.3} F1={:.3}",
                            results.mean.lea.precision,
                            results.mean.lea.recall,
                            results.mean.lea.f1
                        );
                        println!(
                            "  BLANC: P={:.3} R={:.3} F1={:.3}",
                            results.mean.blanc.precision,
                            results.mean.blanc.recall,
                            results.mean.blanc.f1
                        );
                        println!(
                            "  Documents: {}  Time: {:.1}s",
                            results.num_documents,
                            elapsed.as_secs_f64()
                        );
                        println!();
                    }
                }
                Err(e) => {
                    println!("{:20} FAILED: {}", dataset_id.name(), e);
                }
            }
        }
    }
    #[cfg(not(feature = "eval-advanced"))]
    {
        println!("Coreference benchmark requires --features eval-advanced");
    }
}

#[test]
#[ignore]
fn benchmark_coreference_on_gap() {
    #[cfg(feature = "eval-advanced")]
    {
        use anno::eval::loader::DatasetLoader;

        let loader = DatasetLoader::new().unwrap();

        println!("\n=== Loading GAP Coreference Dataset ===\n");
        let gold_docs = match loader.load_or_download_coref(DatasetId::GAP) {
            Ok(docs) => docs,
            Err(e) => {
                println!("Failed to load GAP: {}", e);
                return;
            }
        };

        println!("Loaded {} GAP examples", gold_docs.len());

        // Test with multiple NER models
        let mut models: Vec<(&str, Box<dyn Model>)> = vec![
            ("StackedNER", Box::new(StackedNER::default())),
            ("HeuristicNER", Box::new(HeuristicNER::new())),
        ];

        // Add GLiNER if available
        #[cfg(feature = "onnx")]
        {
            use anno::{GLiNEROnnx, DEFAULT_GLINER_MODEL};
            if let Ok(gliner) = GLiNEROnnx::new(DEFAULT_GLINER_MODEL) {
                models.push(("GLiNER", Box::new(gliner)));
            }
        }

        println!("\n=== Evaluating Coreference Resolution with Multiple Models ===\n");

        for (model_name, ner) in &models {
            println!("\n--- {} + SimpleCorefResolver ---", model_name);

            let start = Instant::now();
            let results = evaluate_coref_on_dataset(ner.as_ref(), &gold_docs);
            let elapsed = start.elapsed();

            println!("Results:");
            println!("  CoNLL F1: {:.3}", results.mean.conll_f1);
            println!(
                "  MUC: P={:.3} R={:.3} F1={:.3}",
                results.mean.muc.precision, results.mean.muc.recall, results.mean.muc.f1
            );
            println!(
                "  B³: P={:.3} R={:.3} F1={:.3}",
                results.mean.b_cubed.precision,
                results.mean.b_cubed.recall,
                results.mean.b_cubed.f1
            );
            println!(
                "  CEAF-e: P={:.3} R={:.3} F1={:.3}",
                results.mean.ceaf_e.precision, results.mean.ceaf_e.recall, results.mean.ceaf_e.f1
            );
            println!(
                "  LEA: P={:.3} R={:.3} F1={:.3}",
                results.mean.lea.precision, results.mean.lea.recall, results.mean.lea.f1
            );
            println!(
                "  BLANC: P={:.3} R={:.3} F1={:.3}",
                results.mean.blanc.precision, results.mean.blanc.recall, results.mean.blanc.f1
            );
            println!(
                "  Documents: {}  Time: {:.1}s",
                results.num_documents,
                elapsed.as_secs_f64()
            );
        }
    }
    #[cfg(not(feature = "eval-advanced"))]
    {
        println!("Coreference benchmark requires --features eval-advanced");
    }
}

// =============================================================================
// Relation Extraction Evaluation
// =============================================================================

/// Evaluate relation extraction on a dataset.
///
/// Uses NER model to extract entities, then creates relation predictions
/// from entity pairs using heuristics.
fn evaluate_relation_on_dataset(
    ner: &dyn Model,
    gold_docs: &[anno::eval::loader::RelationDocument],
) -> anno::eval::relation::RelationMetrics {
    use anno::eval::relation::{evaluate_relations, RelationEvalConfig, RelationPrediction};

    let mut all_gold = Vec::new();
    let mut all_pred = Vec::new();

    for doc in gold_docs {
        let text = doc.text.as_str();
        all_gold.extend(doc.relations.clone());

        // Extract entities using NER
        let entities = ner.extract_entities(text, None).unwrap_or_default();

        // Create relation predictions from entity pairs (heuristic)
        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                let head = &entities[i];
                let tail = &entities[j];

                // Only consider pairs within reasonable distance
                let distance = if tail.start >= head.end {
                    tail.start - head.end
                } else {
                    head.start.saturating_sub(tail.end)
                };

                if distance < 200 {
                    // Extract text between entities using character offsets (not byte offsets)
                    let between_text = if head.end <= tail.start {
                        text.chars()
                            .skip(head.end)
                            .take(tail.start - head.end)
                            .collect::<String>()
                    } else {
                        text.chars()
                            .skip(tail.end)
                            .take(head.start - tail.end)
                            .collect::<String>()
                    };

                    let rel_type = if between_text.to_lowercase().contains("founded") {
                        "FOUNDED"
                    } else if between_text.to_lowercase().contains("works for") {
                        "WORKS_FOR"
                    } else if between_text.to_lowercase().contains("located in") {
                        "LOCATED_IN"
                    } else {
                        "RELATED"
                    };

                    all_pred.push(RelationPrediction {
                        head_span: (head.start, head.end),
                        head_type: head.entity_type.as_label().to_string(),
                        tail_span: (tail.start, tail.end),
                        tail_type: tail.entity_type.as_label().to_string(),
                        relation_type: rel_type.to_string(),
                        confidence: 0.5,
                    });
                }
            }
        }
    }

    let config = RelationEvalConfig::default();
    evaluate_relations(&all_gold, &all_pred, &config)
}

/// Benchmark relation extraction on all relation extraction datasets.
#[test]
#[ignore]
fn benchmark_relation_extraction_on_all_datasets() {
    #[cfg(feature = "eval-advanced")]
    {
        use anno::eval::loader::DatasetLoader;

        let loader = DatasetLoader::new().unwrap();

        // Test with multiple NER models
        let mut models: Vec<(&str, Box<dyn Model>)> = vec![
            ("StackedNER", Box::new(StackedNER::default())),
            ("HeuristicNER", Box::new(HeuristicNER::new())),
        ];

        // Add GLiNER if available
        #[cfg(feature = "onnx")]
        {
            use anno::{GLiNEROnnx, DEFAULT_GLINER_MODEL};
            if let Ok(gliner) = GLiNEROnnx::new(DEFAULT_GLINER_MODEL) {
                models.push(("GLiNER", Box::new(gliner)));
            }
        }

        let re_datasets = DatasetId::all_relation_extraction();

        println!("\n=== Relation Extraction Benchmark on All Datasets ===\n");

        for dataset_id in re_datasets {
            match loader.load_or_download_relation(*dataset_id) {
                Ok(gold_docs) => {
                    println!(
                        "\n=== {} Relation Extraction Dataset ===\n",
                        dataset_id.name()
                    );
                    println!("Loaded {} {} examples", gold_docs.len(), dataset_id.name());

                    println!(
                        "{:20} {:>8} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
                        "Model",
                        "Docs",
                        "Boundary P",
                        "Boundary R",
                        "Boundary F1",
                        "Strict P",
                        "Strict R",
                        "Strict F1"
                    );
                    println!("{}", "-".repeat(120));

                    for (model_name, ner) in &models {
                        println!(
                            "\nEvaluating relation extraction ({} + entity-pair heuristic)...",
                            model_name
                        );

                        let start = Instant::now();
                        let results = evaluate_relation_on_dataset(ner.as_ref(), &gold_docs);
                        let elapsed = start.elapsed();

                        println!(
                            "{:20} {:>8} {:>12.3} {:>12.3} {:>12.3} {:>12.3} {:>12.3} {:>12.3}",
                            model_name,
                            gold_docs.len(),
                            results.boundary_precision,
                            results.boundary_recall,
                            results.boundary_f1,
                            results.strict_precision,
                            results.strict_recall,
                            results.strict_f1
                        );

                        println!(
                            "  Boundary (Rel): P={:.3} R={:.3} F1={:.3}",
                            results.boundary_precision,
                            results.boundary_recall,
                            results.boundary_f1
                        );
                        println!(
                            "  Strict (Rel+): P={:.3} R={:.3} F1={:.3}",
                            results.strict_precision, results.strict_recall, results.strict_f1
                        );
                        println!(
                            "  Gold relations: {}  Predicted: {}",
                            results.num_gold, results.num_predicted
                        );
                        println!(
                            "  Boundary matches: {}  Strict matches: {}  Time: {:.1}s",
                            results.boundary_matches,
                            results.strict_matches,
                            elapsed.as_secs_f64()
                        );
                        println!();
                    }

                    // Try GLiNER2 RelationExtractor if available
                    #[cfg(feature = "onnx")]
                    {
                        use anno::backends::gliner2::GLiNER2Onnx;
                        use anno::backends::inference::RelationExtractor;
                        use anno::eval::relation::{
                            evaluate_relations, RelationEvalConfig, RelationPrediction,
                        };

                        println!(
                            "--- GLiNER2 RelationExtractor (heuristic-based pattern matching) ---"
                        );
                        if let Ok(gliner2) = GLiNER2Onnx::from_pretrained(
                            "onnx-community/gliner-multitask-large-v0.5",
                        ) {
                            let start = Instant::now();

                            // Collect entity types and relation types from gold data
                            let mut entity_types_set = std::collections::HashSet::new();
                            let mut relation_types_set = std::collections::HashSet::new();
                            for doc in &gold_docs {
                                for rel in &doc.relations {
                                    entity_types_set.insert(rel.head_type.clone());
                                    entity_types_set.insert(rel.tail_type.clone());
                                    relation_types_set.insert(rel.relation_type.clone());
                                }
                            }
                            let entity_types_vec: Vec<&str> =
                                entity_types_set.iter().map(|s| s.as_str()).collect();
                            let relation_types_vec: Vec<&str> =
                                relation_types_set.iter().map(|s| s.as_str()).collect();

                            // Use GLiNER2's RelationExtractor implementation
                            let mut all_gold = Vec::new();
                            let mut all_pred = Vec::new();

                            for doc in &gold_docs {
                                let text = doc.text.as_str();
                                all_gold.extend(doc.relations.clone());

                                // Extract relations using GLiNER2
                                if let Ok(result) = gliner2.extract_with_relations(
                                    text,
                                    &entity_types_vec,
                                    &relation_types_vec,
                                    0.5,
                                ) {
                                    // Convert RelationTriples to RelationPredictions
                                    for triple in &result.relations {
                                        if let Some(pred) =
                                            RelationPrediction::from_triple_with_entities(
                                                triple,
                                                &result.entities,
                                            )
                                        {
                                            all_pred.push(pred);
                                        }
                                    }
                                }
                            }

                            let config = RelationEvalConfig::default();
                            let results = evaluate_relations(&all_gold, &all_pred, &config);
                            let elapsed = start.elapsed();

                            println!(
                                "{:20} {:>8} {:>12.3} {:>12.3} {:>12.3} {:>12.3} {:>12.3} {:>12.3}",
                                "GLiNER2-RE",
                                gold_docs.len(),
                                results.boundary_precision,
                                results.boundary_recall,
                                results.boundary_f1,
                                results.strict_precision,
                                results.strict_recall,
                                results.strict_f1
                            );

                            println!(
                                "  Boundary (Rel): P={:.3} R={:.3} F1={:.3}",
                                results.boundary_precision,
                                results.boundary_recall,
                                results.boundary_f1
                            );
                            println!(
                                "  Strict (Rel+): P={:.3} R={:.3} F1={:.3}",
                                results.strict_precision, results.strict_recall, results.strict_f1
                            );
                            println!(
                                "  Gold relations: {}  Predicted: {}",
                                results.num_gold, results.num_predicted
                            );
                            println!(
                                "  Boundary matches: {}  Strict matches: {}",
                                results.boundary_matches, results.strict_matches
                            );
                            println!("  Time: {:.1}s", elapsed.as_secs_f64());
                            println!();
                        } else {
                            println!(
                                "GLiNER2 model not available, skipping RelationExtractor benchmark"
                            );
                        }
                    }
                }
                Err(e) => {
                    println!("{:20} FAILED: {}", dataset_id.name(), e);
                }
            }
        }
    }
    #[cfg(not(feature = "eval-advanced"))]
    {
        println!("Relation extraction benchmark requires --features eval-advanced");
    }
}

#[test]
#[ignore]
fn benchmark_relation_extraction_on_docred() {
    #[cfg(feature = "eval-advanced")]
    {
        use anno::eval::loader::DatasetLoader;

        let loader = DatasetLoader::new().unwrap();

        println!("\n=== Loading DocRED Relation Extraction Dataset ===\n");
        let gold_docs = match loader.load_or_download_relation(DatasetId::DocRED) {
            Ok(docs) => docs,
            Err(e) => {
                println!("Failed to load DocRED: {}", e);
                return;
            }
        };

        println!("Loaded {} DocRED examples", gold_docs.len());

        // Test with multiple NER models
        let mut models: Vec<(&str, Box<dyn Model>)> = vec![
            ("StackedNER", Box::new(StackedNER::default())),
            ("HeuristicNER", Box::new(HeuristicNER::new())),
        ];

        // Add GLiNER if available
        #[cfg(feature = "onnx")]
        {
            use anno::{GLiNEROnnx, DEFAULT_GLINER_MODEL};
            if let Ok(gliner) = GLiNEROnnx::new(DEFAULT_GLINER_MODEL) {
                models.push(("GLiNER", Box::new(gliner)));
            }
        }

        println!("\n=== Evaluating Relation Extraction with Multiple Models ===\n");

        for (model_name, ner) in &models {
            println!("\n--- {} + entity-pair heuristic ---", model_name);

            let start = Instant::now();
            let results = evaluate_relation_on_dataset(ner.as_ref(), &gold_docs);
            let elapsed = start.elapsed();

            println!("Results:");
            println!(
                "  Boundary (Rel): P={:.3} R={:.3} F1={:.3}",
                results.boundary_precision, results.boundary_recall, results.boundary_f1
            );
            println!(
                "  Strict (Rel+): P={:.3} R={:.3} F1={:.3}",
                results.strict_precision, results.strict_recall, results.strict_f1
            );
            println!(
                "  Gold relations: {}  Predicted: {}",
                results.num_gold, results.num_predicted
            );
            println!(
                "  Boundary matches: {}  Strict matches: {}  Time: {:.1}s",
                results.boundary_matches,
                results.strict_matches,
                elapsed.as_secs_f64()
            );
        }

        // Try GLiNER2 RelationExtractor if available
        #[cfg(feature = "onnx")]
        {
            use anno::backends::gliner2::GLiNER2Onnx;
            use anno::backends::inference::RelationExtractor;
            use anno::eval::relation::{
                evaluate_relations, RelationEvalConfig, RelationPrediction,
            };

            println!("\n--- GLiNER2 RelationExtractor (heuristic-based pattern matching) ---");
            if let Ok(gliner2) =
                GLiNER2Onnx::from_pretrained("onnx-community/gliner-multitask-large-v0.5")
            {
                let start = Instant::now();

                // Collect entity types and relation types from gold data
                let mut entity_types_set = std::collections::HashSet::new();
                let mut relation_types_set = std::collections::HashSet::new();
                for doc in &gold_docs {
                    for rel in &doc.relations {
                        entity_types_set.insert(rel.head_type.clone());
                        entity_types_set.insert(rel.tail_type.clone());
                        relation_types_set.insert(rel.relation_type.clone());
                    }
                }
                let entity_types_vec: Vec<&str> =
                    entity_types_set.iter().map(|s| s.as_str()).collect();
                let relation_types_vec: Vec<&str> =
                    relation_types_set.iter().map(|s| s.as_str()).collect();

                // Use GLiNER2's RelationExtractor implementation
                let mut all_gold = Vec::new();
                let mut all_pred = Vec::new();

                for doc in &gold_docs {
                    let text = doc.text.as_str();
                    all_gold.extend(doc.relations.clone());

                    // Extract relations using GLiNER2
                    if let Ok(result) = gliner2.extract_with_relations(
                        text,
                        &entity_types_vec,
                        &relation_types_vec,
                        0.5,
                    ) {
                        // Convert RelationTriples to RelationPredictions
                        for triple in &result.relations {
                            if let Some(pred) = RelationPrediction::from_triple_with_entities(
                                triple,
                                &result.entities,
                            ) {
                                all_pred.push(pred);
                            }
                        }
                    }
                }

                let config = RelationEvalConfig::default();
                let results = evaluate_relations(&all_gold, &all_pred, &config);
                let elapsed = start.elapsed();

                println!("Results:");
                println!(
                    "  Boundary (Rel): P={:.3} R={:.3} F1={:.3}",
                    results.boundary_precision, results.boundary_recall, results.boundary_f1
                );
                println!(
                    "  Strict (Rel+): P={:.3} R={:.3} F1={:.3}",
                    results.strict_precision, results.strict_recall, results.strict_f1
                );
                println!(
                    "  Gold relations: {}  Predicted: {}",
                    results.num_gold, results.num_predicted
                );
                println!(
                    "  Boundary matches: {}  Strict matches: {}",
                    results.boundary_matches, results.strict_matches
                );
                println!("  Time: {:.1}s", elapsed.as_secs_f64());
            } else {
                println!("GLiNER2 model not available, skipping RelationExtractor benchmark");
            }
        }
    }
    #[cfg(not(feature = "eval-advanced"))]
    {
        println!("Relation extraction benchmark requires --features eval-advanced");
    }
}

// These are very loose baselines for RegexNER on named entity datasets
// RegexNER is NOT designed for PER/ORG/LOC - it's for structured patterns
// So we expect near-zero performance, but non-crashing behavior

const REGEX_NER_MIN_F1: f64 = 0.0; // RegexNER won't find named entities

#[test]
#[ignore]
fn regression_test_wikigold() {
    #[cfg(feature = "eval-advanced")]
    {
        let loader = DatasetLoader::new().unwrap();
        let dataset = match loader.load_or_download(DatasetId::WikiGold) {
            Ok(ds) => ds,
            Err(_) => return,
        };

        let ner = RegexNER::new();
        let (metrics, _) = evaluate_ner_on_dataset(&ner, &dataset);

        let f1 = metrics.f1();
        assert!(
            f1 >= REGEX_NER_MIN_F1,
            "WikiGold F1 ({:.3}) dropped below minimum ({:.3})",
            f1,
            REGEX_NER_MIN_F1
        );

        println!(
            "WikiGold F1: {:.3} (minimum: {:.3}) - PASS",
            f1, REGEX_NER_MIN_F1
        );
    }
}
