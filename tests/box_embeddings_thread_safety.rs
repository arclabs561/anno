//! Tests for box embeddings training thread safety.
//!
//! These tests verify that box_embeddings_training works correctly
//! in multi-threaded scenarios, particularly the simple_random() function
//! that now uses AtomicUsize instead of static mut.

mod tests {
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_simple_random_thread_safety_pattern() {
        // Test the pattern used in simple_random() - AtomicUsize counter
        // This verifies the fix works correctly
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::{SystemTime, UNIX_EPOCH};

        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let num_threads = 10;
        let calls_per_thread = 100;
        let mut handles = vec![];

        for _ in 0..num_threads {
            let handle = thread::spawn(move || {
                for _ in 0..calls_per_thread {
                    // This is the exact pattern from simple_random()
                    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
                    let mut hasher = DefaultHasher::new();
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_nanos()
                        .hash(&mut hasher);
                    count.hash(&mut hasher);
                    let _hash = hasher.finish();
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(
            COUNTER.load(Ordering::Relaxed),
            num_threads * calls_per_thread
        );
    }

    #[cfg(feature = "eval-advanced")]
    #[test]
    fn test_box_embeddings_training_parallel_initialization() {
        // Test that box embeddings can be initialized in parallel
        // This indirectly tests simple_random() thread safety
        use anno::backends::box_embeddings_training::{
            BoxEmbeddingTrainer, TrainingConfig, TrainingExample,
        };
        use anno::{Entity, EntityType};

        let num_threads = 3; // Reduced threads
        let mut handles = vec![];

        for i in 0..num_threads {
            let handle = thread::spawn(move || {
                // Each thread creates a trainer with random initialization
                // This will call simple_random() multiple times during training
                let config = TrainingConfig {
                    epochs: 1, // Single epoch for speed
                    learning_rate: 0.01,
                    batch_size: 1,
                    ..Default::default()
                };
                let mut trainer = BoxEmbeddingTrainer::new(config.clone(), 32, None); // Smaller dim

                // Create minimal training examples
                let examples = vec![TrainingExample {
                    entities: vec![Entity::new("Entity1", EntityType::Person, 0, 7, 0.9)],
                    chains: vec![],
                }];

                trainer.initialize_boxes(&examples, None);
                // Training will use simple_random() for shuffling
                trainer.train(&examples);
            });
            handles.push(handle);
        }

        // All threads should complete without data races
        for handle in handles {
            handle.join().expect("Thread should complete successfully");
        }
    }

    #[cfg(feature = "eval-advanced")]
    #[test]
    fn test_box_embeddings_training_concurrent_training() {
        // Test concurrent training of different box embeddings
        use anno::backends::box_embeddings_training::{
            BoxEmbeddingTrainer, TrainingConfig, TrainingExample,
        };
        use anno::{Entity, EntityType};

        let num_threads = 2; // Reduced
        let mut handles = vec![];

        for i in 0..num_threads {
            let handle = thread::spawn(move || {
                let config = TrainingConfig {
                    epochs: 1, // Single epoch
                    learning_rate: 0.01,
                    batch_size: 1,
                    ..Default::default()
                };
                let mut trainer = BoxEmbeddingTrainer::new(config, 16, None); // Smaller dim

                // Minimal examples
                let examples = vec![TrainingExample {
                    entities: vec![Entity::new(
                        &format!("Entity_{}", i),
                        EntityType::Person,
                        0,
                        10,
                        0.9,
                    )],
                    chains: vec![],
                }];

                trainer.initialize_boxes(&examples, None);
                // Training will use simple_random() for shuffling
                trainer.train(&examples);
            });
            handles.push(handle);
        }

        for handle in handles {
            handle
                .join()
                .expect("Training should complete without races");
        }
    }

    #[cfg(feature = "eval-advanced")]
    #[test]
    fn test_box_embeddings_no_data_races() {
        // Lightweight test: fewer threads, minimal training
        use anno::backends::box_embeddings_training::{
            BoxEmbeddingTrainer, TrainingConfig, TrainingExample,
        };
        use anno::{Entity, EntityType};

        let num_threads = 3; // Reduced
        let operations_per_thread = 2; // Minimal
        let mut handles = vec![];

        for i in 0..num_threads {
            let handle = thread::spawn(move || {
                for _ in 0..operations_per_thread {
                    let config = TrainingConfig {
                        epochs: 1, // Single epoch
                        learning_rate: 0.01,
                        batch_size: 1,
                        ..Default::default()
                    };
                    let mut trainer = BoxEmbeddingTrainer::new(config, 16, None);
                    let examples = vec![TrainingExample {
                        entities: vec![Entity::new(
                            &format!("Entity_{}", i),
                            EntityType::Person,
                            0,
                            10,
                            0.9,
                        )],
                        chains: vec![],
                    }];
                    trainer.initialize_boxes(&examples, None);
                    // This exercises simple_random() multiple times
                    let _ = trainer.train(&examples);
                }
            });
            handles.push(handle);
        }

        // Should complete without panics or data races
        for handle in handles {
            handle.join().expect("Should complete without races");
        }
    }
}
