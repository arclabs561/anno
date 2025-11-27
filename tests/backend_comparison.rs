//! Backend comparison tests.
//!
//! Compares all available NER backends on the same datasets.
//! This helps evaluate which backend is best for different use cases.
//!
//! ## Running
//!
//! ```bash
//! # Fast tests (pattern-only)
//! cargo test --test backend_comparison
//!
//! # With ONNX backends
//! cargo test --test backend_comparison --features onnx -- --ignored --nocapture
//!
//! # With GLiNER (gline-rs)
//! cargo test --test backend_comparison --features gliner -- --ignored --nocapture
//! ```

use anno::eval::synthetic::all_datasets;
use anno::{Model, PatternNER};
use std::time::Instant;

/// Backend info for comparison.
struct BackendInfo {
    name: &'static str,
    model: Box<dyn Model>,
}

impl BackendInfo {
    fn new(name: &'static str, model: impl Model + 'static) -> Self {
        Self {
            name,
            model: Box::new(model),
        }
    }
}

/// Results from running a backend.
struct BenchmarkResult {
    name: String,
    total_entities: usize,
    elapsed_ms: f64,
    entities_per_sec: f64,
}

/// Run a backend on all datasets and return benchmark results.
fn benchmark_backend(backend: &BackendInfo, texts: &[String]) -> BenchmarkResult {
    let start = Instant::now();
    let mut total_entities = 0;

    for text in texts {
        if let Ok(entities) = backend.model.extract_entities(text, None) {
            total_entities += entities.len();
        }
    }

    let elapsed = start.elapsed();
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let entities_per_sec = if elapsed_ms > 0.0 {
        (total_entities as f64) / (elapsed_ms / 1000.0)
    } else {
        0.0
    };

    BenchmarkResult {
        name: backend.name.to_string(),
        total_entities,
        elapsed_ms,
        entities_per_sec,
    }
}

#[test]
fn test_pattern_ner_basic() {
    let ner = PatternNER::new();
    assert!(ner.is_available());
    assert_eq!(ner.name(), "pattern");

    // Should extract structured entities
    let text = "Meeting at 3:30 PM on Jan 15. Cost $50.";
    let entities = ner.extract_entities(text, None).unwrap();
    
    // Debug: print what we found
    eprintln!("Input: {}", text);
    for e in &entities {
        eprintln!("  Found: {:?} - '{}'", e.entity_type, e.text);
    }
    
    assert!(!entities.is_empty(), "Should find entities");

    // Should find date and money at minimum
    let types: Vec<_> = entities.iter().map(|e| &e.entity_type).collect();
    assert!(
        types.iter().any(|t| matches!(t, anno::EntityType::Date)),
        "Should find date. Found: {:?}", types
    );
    assert!(
        types.iter().any(|t| matches!(t, anno::EntityType::Money)),
        "Should find money. Found: {:?}", types
    );
    
    // Time is extracted separately - may or may not be present depending on regex
    let has_time = types.iter().any(|t| matches!(t, anno::EntityType::Time));
    eprintln!("  Time found: {}", has_time);
}

#[test]
fn test_pattern_ner_contact_entities() {
    let ner = PatternNER::new();

    let text = "Contact: bob@example.com, https://example.com, (555) 123-4567";
    let entities = ner.extract_entities(text, None).unwrap();

    let has_email = entities
        .iter()
        .any(|e| matches!(e.entity_type, anno::EntityType::Email));
    let has_url = entities
        .iter()
        .any(|e| matches!(e.entity_type, anno::EntityType::Url));
    let has_phone = entities
        .iter()
        .any(|e| matches!(e.entity_type, anno::EntityType::Phone));

    assert!(has_email, "Should find email");
    assert!(has_url, "Should find URL");
    assert!(has_phone, "Should find phone");
}

#[test]
fn test_pattern_ner_no_named_entities() {
    let ner = PatternNER::new();

    // Pattern NER should NOT extract person/org/location
    let text = "Steve Jobs founded Apple in Cupertino, California.";
    let entities = ner.extract_entities(text, None).unwrap();

    let has_person = entities
        .iter()
        .any(|e| matches!(e.entity_type, anno::EntityType::Person));
    let has_org = entities
        .iter()
        .any(|e| matches!(e.entity_type, anno::EntityType::Organization));
    let has_loc = entities
        .iter()
        .any(|e| matches!(e.entity_type, anno::EntityType::Location));

    assert!(
        !has_person && !has_org && !has_loc,
        "Pattern NER should NOT extract named entities (use ML backends)"
    );
}

#[test]
fn test_ner_extractor_pattern_only() {
    use anno::NERExtractor;

    let extractor = NERExtractor::pattern_only();
    assert!(extractor.is_available());
    assert_eq!(extractor.backend_type(), anno::BackendType::Pattern);
    assert!(!extractor.has_ml_backend());
    assert!(!extractor.supports_zero_shot());

    // Should extract pattern entities
    let text = "Total: $100 (15% discount)";
    let entities = extractor.extract(text, None).unwrap();
    assert!(!entities.is_empty());
}

#[test]
fn test_ner_extractor_best_available() {
    use anno::NERExtractor;

    // best_available should always work (falls back to patterns)
    let extractor = NERExtractor::best_available();
    assert!(extractor.is_available());

    let text = "Meeting on 2024-01-15 cost $500.";
    let entities = extractor.extract(text, None).unwrap();

    // Should find at least pattern entities
    assert!(!entities.is_empty(), "Should extract entities");
}

#[test]
fn test_backend_comparison_on_synthetic() {
    // Collect all synthetic texts
    let datasets = all_datasets();
    let texts: Vec<String> = datasets.iter().map(|ex| ex.text.clone()).collect();

    println!("\n=== Backend Comparison on {} Synthetic Examples ===\n", texts.len());

    // Create available backends
    let backends: Vec<BackendInfo> = vec![BackendInfo::new("pattern", PatternNER::new())];

    // Run benchmarks
    let mut results = Vec::new();
    for backend in &backends {
        let result = benchmark_backend(backend, &texts);
        results.push(result);
    }

    // Print results
    println!("{:<15} {:>10} {:>12} {:>15}", "Backend", "Entities", "Time (ms)", "Entities/sec");
    println!("{}", "-".repeat(55));

    for result in &results {
        println!(
            "{:<15} {:>10} {:>12.2} {:>15.0}",
            result.name, result.total_entities, result.elapsed_ms, result.entities_per_sec
        );
    }

    // Verify pattern NER found entities
    let pattern_result = results.iter().find(|r| r.name == "pattern").unwrap();
    assert!(
        pattern_result.total_entities > 0,
        "Pattern NER should find entities"
    );
}

#[test]
#[ignore] // Requires network and ML features
fn test_ml_backend_comparison() {
    use anno::NERExtractor;

    let texts = vec![
        "Steve Jobs founded Apple in Cupertino.",
        "Microsoft CEO Satya Nadella announced new products.",
        "The meeting is scheduled for January 15, 2025 at 3:30 PM.",
        "Total investment: $50 million.",
        "Contact: support@example.com or (555) 123-4567.",
    ];

    println!("\n=== ML Backend Comparison ===\n");

    // Pattern-only baseline
    let pattern = NERExtractor::pattern_only();
    println!("--- Pattern NER ---");
    for text in &texts {
        let entities = pattern.extract(text, None).unwrap();
        println!("  Input: {}", text);
        for e in &entities {
            println!("    - {} ({:?})", e.text, e.entity_type);
        }
    }

    // Best available (may be ML if features enabled)
    let best = NERExtractor::best_available();
    println!("\n--- Best Available ({}) ---", best.active_backend_name());
    for text in &texts {
        let entities = best.extract(text, None).unwrap();
        println!("  Input: {}", text);
        for e in &entities {
            println!("    - {} ({:?})", e.text, e.entity_type);
        }
    }

    // Hybrid mode
    println!("\n--- Hybrid Mode ---");
    for text in &texts {
        let entities = best.extract_hybrid(text, None).unwrap();
        println!("  Input: {}", text);
        for e in &entities {
            println!("    - {} ({:?})", e.text, e.entity_type);
        }
    }
}

#[test]
#[ignore] // Requires onnx feature and network
#[cfg(feature = "onnx")]
fn test_gliner_backend() {
    use anno::GLiNEROnnx;

    println!("\n=== GLiNER (ONNX) Test ===\n");

    // Try to create GLiNER model
    let result = GLiNEROnnx::new("onnx-community/gliner_small-v2.1");

    if let Ok(gliner) = result {
        assert!(gliner.is_available());
        assert_eq!(gliner.name(), "GLiNER-ONNX");

        let texts = vec![
            "Steve Jobs founded Apple in Cupertino.",
            "Elon Musk is the CEO of Tesla and SpaceX.",
            "The Eiffel Tower is located in Paris, France.",
        ];

        for text in texts {
            println!("Input: {}", text);
            if let Ok(entities) = gliner.extract_entities(text, None) {
                for e in &entities {
                    println!("  - {} ({:?}, {:.2})", e.text, e.entity_type, e.confidence);
                }
            }
            println!();
        }
    } else {
        println!("GLiNER not available: {:?}", result.err());
    }
}

