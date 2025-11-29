//! Production deployment example for anno NER.
//!
//! This example demonstrates best practices for deploying anno in production:
//!
//! 1. **Async-safe inference** - Using `spawn_blocking` to avoid executor starvation
//! 2. **Session pooling** - Parallel inference with multiple ONNX sessions
//! 3. **Quantized models** - INT8 for 2-4x CPU speedup
//! 4. **Global initialization** - One-time model loading with `once_cell`
//! 5. **Entity validation** - Verify extraction quality before downstream use
//!
//! # Run
//!
//! ```bash
//! cargo run --example production --features "production"
//! ```
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │  Web Server (tokio runtime)                                 │
//! │                                                             │
//! │  Request 1 ──► [spawn_blocking] ──► Session 1 ──► Response 1│
//! │  Request 2 ──► [spawn_blocking] ──► Session 2 ──► Response 2│
//! │  Request 3 ──► [spawn_blocking] ──► Session 3 ──► Response 3│
//! │                                                             │
//! │  (Parallel inference, no executor blocking)                 │
//! └─────────────────────────────────────────────────────────────┘
//! ```

#![allow(dead_code)]

use std::sync::Arc;
use std::time::Instant;

// =============================================================================
// Conditional compilation for different feature sets
// =============================================================================

#[cfg(all(feature = "async-inference", feature = "session-pool"))]
mod production_setup {
    use super::*;
    use anno::{
        backends::{
            batch_extract_limited, AsyncNER, GLiNERConfig, GLiNERPool, IntoAsync, PoolConfig,
        },
        Entity, GLiNEROnnx,
    };

    /// Global model instance (initialized once).
    ///
    /// Using `once_cell` ensures the model is loaded exactly once,
    /// avoiding cold-start latency on every request.
    static GLOBAL_MODEL: once_cell::sync::Lazy<Arc<AsyncNER<GLiNEROnnx>>> =
        once_cell::sync::Lazy::new(|| {
            eprintln!("[init] Loading GLiNER model (this happens once)...");
            let start = Instant::now();

            let config = GLiNERConfig {
                prefer_quantized: true, // Use INT8 for faster CPU inference
                optimization_level: 3,  // Maximum ONNX optimization
                num_threads: 4,         // Threads per inference call
            };

            let model = GLiNEROnnx::with_config("onnx-community/gliner_small-v2.1", config)
                .expect("Failed to load model");

            eprintln!(
                "[init] Model loaded in {:?} (quantized: {})",
                start.elapsed(),
                model.is_quantized()
            );

            Arc::new(model.into_async())
        });

    /// Global session pool for high-throughput scenarios.
    ///
    /// Each session can process requests independently, enabling true
    /// parallel inference.
    static GLOBAL_POOL: once_cell::sync::Lazy<GLiNERPool> = once_cell::sync::Lazy::new(|| {
        eprintln!("[init] Creating session pool...");
        let start = Instant::now();

        let config = PoolConfig::with_size(4) // 4 parallel sessions
            .with_timeout(5000) // 5s acquire timeout
            .prefer_quantized(true); // Use INT8 models

        let pool = GLiNERPool::new("onnx-community/gliner_small-v2.1", config)
            .expect("Failed to create pool");

        eprintln!(
            "[init] Pool created in {:?} ({} sessions)",
            start.elapsed(),
            pool.pool().pool_size()
        );

        pool
    });

    /// Extract entities using the global async model.
    ///
    /// Safe to call from async handlers - uses spawn_blocking internally.
    pub async fn extract_async(text: &str) -> Result<Vec<Entity>, anno::Error> {
        GLOBAL_MODEL.extract_entities(text).await
    }

    /// Extract entities using the session pool.
    ///
    /// More efficient for high-throughput workloads as it avoids
    /// the overhead of spawn_blocking per request.
    pub fn extract_pooled(
        text: &str,
        entity_types: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>, anno::Error> {
        GLOBAL_POOL.extract(text, entity_types, threshold)
    }

    /// Batch extract with concurrency limit.
    ///
    /// Processes many texts efficiently while controlling memory usage.
    pub async fn extract_batch(
        texts: &[&str],
        max_concurrent: usize,
    ) -> Result<Vec<Vec<Entity>>, anno::Error> {
        batch_extract_limited(&GLOBAL_MODEL, texts, max_concurrent).await
    }

    /// Demonstrate async extraction.
    pub async fn demo_async() {
        println!("\n=== Async Extraction Demo ===\n");

        let texts = [
            "Elon Musk founded SpaceX in 2002.",
            "Marie Curie won the Nobel Prize in 1903.",
            "Apple was started by Steve Jobs in Cupertino.",
        ];

        for text in &texts {
            let start = Instant::now();
            match extract_async(text).await {
                Ok(entities) => {
                    println!("Text: {}", text);
                    println!(
                        "  Entities: {} found in {:?}",
                        entities.len(),
                        start.elapsed()
                    );
                    for e in &entities {
                        println!(
                            "    - {} ({:?}, {:.2})",
                            e.text, e.entity_type, e.confidence
                        );
                    }
                }
                Err(e) => eprintln!("  Error: {}", e),
            }
            println!();
        }
    }

    /// Demonstrate pooled extraction.
    pub fn demo_pooled() {
        println!("\n=== Pooled Extraction Demo ===\n");

        let entity_types = ["person", "organization", "location"];
        let threshold = 0.5;

        let texts = [
            "Jeff Bezos started Amazon in Seattle.",
            "Google was founded by Larry Page at Stanford.",
        ];

        for text in &texts {
            let start = Instant::now();
            match extract_pooled(text, &entity_types, threshold) {
                Ok(entities) => {
                    println!("Text: {}", text);
                    println!(
                        "  Entities: {} found in {:?}",
                        entities.len(),
                        start.elapsed()
                    );
                    for e in &entities {
                        println!(
                            "    - {} ({:?}, {:.2})",
                            e.text, e.entity_type, e.confidence
                        );
                    }
                }
                Err(e) => eprintln!("  Error: {}", e),
            }
            println!();
        }
    }

    /// Demonstrate batch extraction with validation.
    pub async fn demo_batch_with_validation() {
        println!("\n=== Batch Extraction with Validation Demo ===\n");

        let texts: Vec<&str> = vec![
            "Tim Cook leads Apple Inc.",
            "Sundar Pichai is the CEO of Google.",
            "Satya Nadella runs Microsoft in Redmond.",
            "Mark Zuckerberg founded Meta (formerly Facebook).",
            "Jensen Huang is the CEO of NVIDIA.",
        ];

        let start = Instant::now();
        let results = extract_batch(&texts, 2).await; // Process 2 at a time
        let total_time = start.elapsed();

        println!("Processed {} texts in {:?}", texts.len(), total_time);
        println!("Average: {:?} per text\n", total_time / texts.len() as u32);

        if let Ok(all_entities) = results {
            for (text, entities) in texts.iter().zip(all_entities.iter()) {
                println!("Text: {}", text);

                // Validate entities against source text
                let issues = anno::Entity::validate_batch(entities, text);
                if issues.is_empty() {
                    println!("  All {} entities valid", entities.len());
                } else {
                    println!("  WARNING: {} entities have issues", issues.len());
                    for (idx, errs) in &issues {
                        for err in errs {
                            println!("    Entity {}: {}", idx, err);
                        }
                    }
                }

                for e in entities {
                    println!(
                        "    - {} [{}-{}] ({:?})",
                        e.text, e.start, e.end, e.entity_type
                    );
                }
                println!();
            }
        }
    }

    /// Demonstrate throughput measurement.
    pub async fn demo_throughput() {
        println!("\n=== Throughput Measurement ===\n");

        // Create test data
        let texts: Vec<String> = (0..100)
            .map(|i| format!("Person {} works at Company {} in City {}.", i, i * 2, i * 3))
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

        // Measure sequential async
        let start = Instant::now();
        let mut count = 0;
        for text in &text_refs[..20] {
            if let Ok(entities) = extract_async(text).await {
                count += entities.len();
            }
        }
        let seq_time = start.elapsed();
        println!(
            "Sequential (20 texts): {:?} ({:.1} texts/sec)",
            seq_time,
            20.0 / seq_time.as_secs_f64()
        );

        // Measure batch with concurrency
        let start = Instant::now();
        if let Ok(results) = extract_batch(&text_refs[..20], 4).await {
            let batch_count: usize = results.iter().map(|r| r.len()).sum();
            let batch_time = start.elapsed();
            println!(
                "Batch (20 texts, 4 concurrent): {:?} ({:.1} texts/sec)",
                batch_time,
                20.0 / batch_time.as_secs_f64()
            );
            println!("Entities found: {} (seq) vs {} (batch)", count, batch_count);
        }

        // Measure pool throughput (sync)
        let entity_types = ["person", "organization", "location"];
        let start = Instant::now();
        let mut pool_count = 0;
        for text in &text_refs[..20] {
            if let Ok(entities) = extract_pooled(text, &entity_types, 0.5) {
                pool_count += entities.len();
            }
        }
        let pool_time = start.elapsed();
        println!(
            "Pool (20 texts): {:?} ({:.1} texts/sec)",
            pool_time,
            20.0 / pool_time.as_secs_f64()
        );
        println!("Pool entities: {}", pool_count);
    }

    /// Run all demos.
    pub async fn run_all() {
        // Trigger global model initialization
        let _ = &*GLOBAL_MODEL;
        let _ = &*GLOBAL_POOL;

        demo_async().await;
        demo_pooled();
        demo_batch_with_validation().await;
        demo_throughput().await;
    }
}

// =============================================================================
// Fallback for missing features
// =============================================================================

#[cfg(not(all(feature = "async-inference", feature = "session-pool")))]
mod production_setup {
    pub async fn run_all() {
        eprintln!("This example requires the 'production' feature:");
        eprintln!("  cargo run --example production --features production");
        eprintln!();
        eprintln!("Or individually:");
        eprintln!("  cargo run --example production --features \"async-inference,session-pool\"");
    }
}

// =============================================================================
// Main
// =============================================================================

#[tokio::main]
async fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  anno Production Deployment Example                          ║");
    println!("║                                                              ║");
    println!("║  Features demonstrated:                                      ║");
    println!("║  • Async-safe inference (spawn_blocking)                     ║");
    println!("║  • Session pooling (parallel ONNX sessions)                  ║");
    println!("║  • Quantized models (INT8 for CPU speedup)                   ║");
    println!("║  • Global initialization (once_cell)                         ║");
    println!("║  • Entity validation                                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    production_setup::run_all().await;
}
