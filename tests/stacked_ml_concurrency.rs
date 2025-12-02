//! Concurrency tests for StackedNER with ML backends.
//!
//! These tests verify:
//! - Thread safety with ML backends
//! - Concurrent extraction with ML stacks
//! - Race condition detection
//! - Deadlock prevention

#[cfg(feature = "onnx")]
mod concurrency_tests {
    use anno::{Model, StackedNER};
    use std::sync::Arc;
    use std::thread;

    // Helper to create GLiNER with graceful failure handling
    fn create_gliner() -> Option<anno::GLiNEROnnx> {
        anno::GLiNEROnnx::new("onnx-community/gliner_small-v2.1").ok()
    }

    #[test]
    fn test_ml_stacked_thread_safety() {
        // Test that ML StackedNER is thread-safe
        if let Some(gliner) = create_gliner() {
            let ner = Arc::new(StackedNER::with_ml_first(Box::new(gliner)));
            let text = "Apple Inc. was founded by Steve Jobs in 1976.";

            let handles: Vec<_> = (0..10)
                .map(|_| {
                    let ner = Arc::clone(&ner);
                    let text = text.to_string();
                    thread::spawn(move || ner.extract_entities(&text, None).unwrap())
                })
                .collect();

            for handle in handles {
                let entities = handle.join().unwrap();
                // Verify results are valid
                for entity in entities {
                    assert!(entity.start < entity.end);
                    assert!(entity.confidence >= 0.0 && entity.confidence <= 1.0);
                }
            }
        }
    }

    #[test]
    fn test_ml_stacked_concurrent_different_texts() {
        // Test concurrent extraction with different texts
        if let Some(gliner) = create_gliner() {
            let ner = Arc::new(StackedNER::with_ml_first(Box::new(gliner)));

            let texts = vec![
                "Apple Inc. was founded by Steve Jobs.",
                "Microsoft was founded by Bill Gates.",
                "Google was founded by Larry Page.",
            ];

            let handles: Vec<_> = texts
                .iter()
                .map(|text| {
                    let ner = Arc::clone(&ner);
                    let text = text.to_string();
                    thread::spawn(move || ner.extract_entities(&text, None).unwrap())
                })
                .collect();

            for handle in handles {
                let entities = handle.join().unwrap();
                assert!(!entities.is_empty());
                for entity in entities {
                    assert!(entity.start < entity.end);
                }
            }
        }
    }

    #[test]
    fn test_ml_stacked_concurrent_same_text() {
        // Test concurrent extraction with same text (stress test)
        if let Some(gliner) = create_gliner() {
            let ner = Arc::new(StackedNER::with_ml_first(Box::new(gliner)));
            let text = "Apple Inc. was founded by Steve Jobs in 1976.";

            let num_threads = 20;
            let handles: Vec<_> = (0..num_threads)
                .map(|_| {
                    let ner = Arc::clone(&ner);
                    let text = text.to_string();
                    thread::spawn(move || ner.extract_entities(&text, None).unwrap())
                })
                .collect();

            let mut all_results = Vec::new();
            for handle in handles {
                let entities = handle.join().unwrap();
                all_results.push(entities);
            }

            // All results should be valid
            for entities in all_results {
                for entity in entities {
                    assert!(entity.start < entity.end);
                    assert!(entity.confidence >= 0.0 && entity.confidence <= 1.0);
                }
            }
        }
    }

    #[test]
    fn test_ml_stacked_no_deadlocks() {
        // Test that ML StackedNER doesn't deadlock under concurrent access
        if let Some(gliner) = create_gliner() {
            let ner = Arc::new(StackedNER::with_ml_first(Box::new(gliner)));

            let handles: Vec<_> = (0..5)
                .map(|i| {
                    let ner = Arc::clone(&ner);
                    let text = format!("Text {} with entities.", i);
                    thread::spawn(move || {
                        // Run multiple extractions in each thread
                        for _ in 0..10 {
                            let _ = ner.extract_entities(&text, None).unwrap();
                        }
                    })
                })
                .collect();

            // All threads should complete without deadlock
            for handle in handles {
                handle.join().unwrap();
            }
        }
    }
}
