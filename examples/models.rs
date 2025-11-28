//! Comprehensive model showcase: Tests all model capabilities.
//!
//! Demonstrates:
//! - Standard NER via Model trait
//! - Zero-shot NER with custom types
//! - Zero-shot NER with natural language descriptions
//! - Batch processing
//! - Confidence calibration
//!
//! Run: cargo run --features "onnx" --example models

use anno::{Model, PatternNER, StatisticalNER, StackedNER};
use std::time::Instant;

#[cfg(feature = "onnx")]
use anno::{BertNEROnnx, GLiNEROnnx, ZeroShotNER};

fn main() -> anno::Result<()> {
    println!("Model Capability Showcase");
    println!("=========================\n");

    let test_texts = [
        "Elon Musk founded SpaceX in 2002 and later acquired Twitter for $44 billion.",
        "Dr. Sarah Chen at MIT published groundbreaking research on CRISPR gene editing.",
        "The Treaty of Versailles was signed on June 28, 1919 at the Palace of Versailles.",
        "Amazon's AWS revenue reached $80 billion, surpassing Microsoft Azure.",
        "Patient presented with severe headache and was prescribed 400mg ibuprofen.",
    ];

    // PART 1: Standard Model Trait
    println!("PART 1: Standard Model Trait (extract_entities)");
    println!("------------------------------------------------\n");

    // Pattern NER
    println!("[PatternNER]");
    let pattern = PatternNER::new();
    run_model_test(&pattern, &test_texts[0..2]);

    // Statistical NER
    println!("[StatisticalNER]");
    let statistical = StatisticalNER::new();
    run_model_test(&statistical, &test_texts[0..2]);

    // Stacked NER
    println!("[StackedNER]");
    let stacked = StackedNER::default();
    run_model_test(&stacked, &test_texts[0..2]);

    #[cfg(feature = "onnx")]
    {
        // BERT NER
        println!("[BertNER-ONNX]");
        match BertNEROnnx::new("protectai/bert-base-NER-onnx") {
            Ok(bert) => run_model_test(&bert, &test_texts[0..2]),
            Err(e) => println!("  Skipped: {}\n", e),
        }

        // GLiNER (via Model trait - uses default labels)
        println!("[GLiNER-ONNX (Model trait)]");
        match GLiNEROnnx::new("onnx-community/gliner_small-v2.1") {
            Ok(gliner) => run_model_test(&gliner, &test_texts[0..2]),
            Err(e) => println!("  Skipped: {}\n", e),
        }
    }

    // PART 2: Zero-Shot NER with Custom Types
    #[cfg(feature = "onnx")]
    {
        println!("\nPART 2: Zero-Shot NER (Custom Entity Types)");
        println!("--------------------------------------------\n");

        match GLiNEROnnx::new("onnx-community/gliner_small-v2.1") {
            Ok(gliner) => {
                // Business domain
                println!("[Business Domain]");
                let business_types = ["company", "person", "money", "product"];
                run_zero_shot_test(&gliner, test_texts[0], &business_types);
                run_zero_shot_test(&gliner, test_texts[3], &business_types);

                // Academic domain
                println!("[Academic Domain]");
                let academic_types = ["researcher", "institution", "technology", "publication"];
                run_zero_shot_test(&gliner, test_texts[1], &academic_types);

                // Historical domain
                println!("[Historical Domain]");
                let historical_types = ["treaty", "date", "location", "historical event"];
                run_zero_shot_test(&gliner, test_texts[2], &historical_types);

                // Medical domain
                println!("[Medical Domain]");
                let medical_types = ["symptom", "medication", "dosage", "patient"];
                run_zero_shot_test(&gliner, test_texts[4], &medical_types);
            }
            Err(e) => println!("GLiNER not available: {}\n", e),
        }
    }

    // PART 3: Zero-Shot NER with Natural Language Descriptions
    #[cfg(feature = "onnx")]
    {
        println!("\nPART 3: Zero-Shot NER (Natural Language Descriptions)");
        println!("------------------------------------------------------\n");

        match GLiNEROnnx::new("onnx-community/gliner_small-v2.1") {
            Ok(gliner) => {
                let text = "Dr. Sarah Chen at MIT published research on CRISPR.";

                // Using descriptions instead of labels
                let descriptions = [
                    "a person who conducts scientific research",
                    "an educational or research institution",
                    "a scientific technology or method",
                ];

                println!("Text: {}", text);
                println!("Descriptions:");
                for (i, d) in descriptions.iter().enumerate() {
                    println!("  {}: {}", i + 1, d);
                }

                match gliner.extract_with_descriptions(text, &descriptions, 0.3) {
                    Ok(entities) => {
                        println!("\nEntities found:");
                        for e in &entities {
                            println!(
                                "  \"{}\" [{}] ({:.0}%)",
                                e.text,
                                e.entity_type.as_label(),
                                e.confidence * 100.0
                            );
                        }
                    }
                    Err(e) => println!("  Error: {}", e),
                }
                println!();
            }
            Err(e) => println!("GLiNER not available: {}\n", e),
        }
    }

    // PART 4: Performance Comparison
    #[cfg(feature = "onnx")]
    {
        println!("\nPART 4: Performance Comparison");
        println!("-------------------------------\n");

        let benchmark_text = "Apple Inc. CEO Tim Cook announced new products at the Cupertino headquarters.";
        let iterations = 10;

        println!("Benchmark: {} iterations on text ({} chars)\n", iterations, benchmark_text.len());

        // Pattern NER
        let pattern = PatternNER::new();
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = pattern.extract_entities(benchmark_text, None);
        }
        let pattern_time = start.elapsed();
        println!("  PatternNER:     {:>6.2}ms avg", pattern_time.as_secs_f64() * 1000.0 / iterations as f64);

        // Statistical NER
        let statistical = StatisticalNER::new();
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = statistical.extract_entities(benchmark_text, None);
        }
        let stat_time = start.elapsed();
        println!("  StatisticalNER: {:>6.2}ms avg", stat_time.as_secs_f64() * 1000.0 / iterations as f64);

        // Stacked NER
        let stacked = StackedNER::default();
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = stacked.extract_entities(benchmark_text, None);
        }
        let stacked_time = start.elapsed();
        println!("  StackedNER:     {:>6.2}ms avg", stacked_time.as_secs_f64() * 1000.0 / iterations as f64);

        // BERT NER
        if let Ok(bert) = BertNEROnnx::new("protectai/bert-base-NER-onnx") {
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = bert.extract_entities(benchmark_text, None);
            }
            let bert_time = start.elapsed();
            println!("  BertNER-ONNX:   {:>6.2}ms avg", bert_time.as_secs_f64() * 1000.0 / iterations as f64);
        }

        // GLiNER
        if let Ok(gliner) = GLiNEROnnx::new("onnx-community/gliner_small-v2.1") {
            let labels = ["person", "organization", "location"];
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = gliner.extract(benchmark_text, &labels, 0.5);
            }
            let gliner_time = start.elapsed();
            println!("  GLiNER-ONNX:    {:>6.2}ms avg", gliner_time.as_secs_f64() * 1000.0 / iterations as f64);
        }
    }

    // PART 5: Entity Type Coverage
    println!("\nPART 5: Entity Type Coverage");
    println!("----------------------------\n");

    println!("[PatternNER Types]");
    for t in PatternNER::new().supported_types() {
        println!("  {}", t.as_label());
    }

    println!("\n[StatisticalNER Types]");
    for t in StatisticalNER::new().supported_types() {
        println!("  {}", t.as_label());
    }

    #[cfg(feature = "onnx")]
    {
        if let Ok(bert) = BertNEROnnx::new("protectai/bert-base-NER-onnx") {
            println!("\n[BertNER-ONNX Types]");
            for t in bert.supported_types() {
                println!("  {}", t.as_label());
            }
        }

        if let Ok(gliner) = GLiNEROnnx::new("onnx-community/gliner_small-v2.1") {
            println!("\n[GLiNER-ONNX Default Types]");
            for t in gliner.supported_types().iter().take(7) {
                println!("  {}", t.as_label());
            }
            println!("  ... and any custom type via zero-shot!");
        }
    }

    println!("\nSummary");
    println!("-------\n");

    println!("Available backends:");
    println!("  [x] PatternNER     - Regex patterns (dates, emails, money, etc.)");
    println!("  [x] StatisticalNER - Capitalization heuristics (PER, ORG, LOC)");
    println!("  [x] StackedNER     - Combined Pattern + Statistical");
    #[cfg(feature = "onnx")]
    println!("  [x] BertNER-ONNX   - BERT fine-tuned for NER");
    #[cfg(feature = "onnx")]
    println!("  [x] GLiNER-ONNX    - Zero-shot with any entity type");

    println!("\nFor zero-shot NER with custom types, use:");
    println!("  gliner.extract_with_types(text, &[\"custom\", \"types\"], threshold)");
    println!("  gliner.extract_with_descriptions(text, &[\"natural language\"], threshold)");

    Ok(())
}

fn run_model_test<M: Model>(model: &M, texts: &[&str]) {
    let start = Instant::now();
    for text in texts {
        println!("  Text: {}", text);
        match model.extract_entities(text, None) {
            Ok(entities) => {
                if entities.is_empty() {
                    println!("    (no entities found)");
                }
                for e in entities {
                    println!(
                        "    {} [{}] ({:.0}%)",
                        e.text,
                        e.entity_type.as_label(),
                        e.confidence * 100.0
                    );
                }
            }
            Err(e) => println!("    Error: {}", e),
        }
    }
    println!("  Time: {:.1}ms\n", start.elapsed().as_secs_f64() * 1000.0);
}

#[cfg(feature = "onnx")]
fn run_zero_shot_test(model: &GLiNEROnnx, text: &str, types: &[&str]) {
    println!("  Text: {}", text);
    println!("  Types: {:?}", types);
    match model.extract_with_types(text, types, 0.4) {
        Ok(entities) => {
            if entities.is_empty() {
                println!("    (no entities found)");
            }
            for e in entities {
                println!(
                    "    \"{}\" [{}] ({:.0}%)",
                    e.text,
                    e.entity_type.as_label(),
                    e.confidence * 100.0
                );
            }
        }
        Err(e) => println!("    Error: {}", e),
    }
    println!();
}

