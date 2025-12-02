//! Integration tests for sync module with actual TaskEvaluator usage.
//!
//! These tests verify that the sync module works correctly when used
//! in real evaluation scenarios with TaskEvaluator.

#[cfg(all(feature = "eval", feature = "eval-parallel"))]
mod tests {
    use anno::eval::task_evaluator::{TaskEvalConfig, TaskEvaluator};
    use anno::eval::task_mapping::Task;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_task_evaluator_with_sync_module() {
        // Test that TaskEvaluator uses sync module correctly in parallel evaluation
        let evaluator = TaskEvaluator::new().unwrap();

        let config = TaskEvalConfig {
            tasks: vec![Task::NER],
            datasets: vec![],
            backends: vec!["pattern".to_string()],
            max_examples: Some(10),
            seed: Some(42),
            require_cached: false,
            relation_threshold: 0.5,
            robustness: false,
            compute_familiarity: false,
            temporal_stratification: false,
            confidence_intervals: false,
        };

        // Should not panic or deadlock with sync module
        let result = evaluator.evaluate_all(config);
        assert!(result.is_ok(), "Evaluation should succeed with sync module");
    }

    #[test]
    fn test_per_example_scores_cache_concurrent_access() {
        // Test that per_example_scores_cache works correctly with concurrent access
        let evaluator = Arc::new(TaskEvaluator::new().unwrap());
        let num_threads = 5;
        let mut handles = vec![];

        for _ in 0..num_threads {
            let evaluator_clone = Arc::clone(&evaluator);
            let handle = thread::spawn(move || {
                // Simulate cache access pattern
                // The cache is accessed during evaluation
                let config = TaskEvalConfig {
                    tasks: vec![Task::NER],
                    datasets: vec![],
                    backends: vec!["pattern".to_string()],
                    max_examples: Some(5),
                    seed: Some(42),
                    require_cached: false,
                    relation_threshold: 0.5,
                    robustness: false,
                    compute_familiarity: false,
                    temporal_stratification: false,
                    confidence_intervals: true, // This triggers cache usage
                };
                evaluator_clone.evaluate_all(config)
            });
            handles.push(handle);
        }

        // All threads should complete without deadlock
        for handle in handles {
            let result = handle.join().unwrap();
            assert!(result.is_ok(), "Evaluation should succeed concurrently");
        }
    }
}

#[cfg(feature = "fast-lock")]
mod fast_lock_tests {
    use anno::sync::{lock, Mutex};
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_fast_lock_feature_works() {
        // Verify that fast-lock feature actually uses parking_lot
        let data = Arc::new(Mutex::new(0));
        let num_threads = 10;
        let increments = 100;
        let mut handles = vec![];

        for _ in 0..num_threads {
            let data_clone = Arc::clone(&data);
            let handle = thread::spawn(move || {
                for _ in 0..increments {
                    let mut guard = lock(&data_clone);
                    *guard += 1;
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(*lock(&data), num_threads * increments);
    }

    #[test]
    fn test_fast_lock_no_poisoning() {
        // parking_lot doesn't poison, verify this behavior
        let mutex = Arc::new(Mutex::new(42));
        let mutex_clone = Arc::clone(&mutex);

        let handle = thread::spawn(move || {
            let _guard = lock(&mutex_clone);
            panic!("Intentional panic");
        });

        let _ = handle.join();

        // Should still be able to lock (no poisoning)
        let guard = lock(&mutex);
        assert_eq!(*guard, 42);
    }
}

#[cfg(not(feature = "fast-lock"))]
mod std_mutex_tests {
    use anno::sync::{lock, Mutex};
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_std_mutex_poisoning_recovery() {
        // Verify that std::sync::Mutex poisoning is handled correctly
        let mutex: Arc<Mutex<Option<i32>>> = Arc::new(Mutex::new(Some(42)));
        let mutex_clone = Arc::clone(&mutex);

        let handle = thread::spawn(move || {
            let _guard = mutex_clone.lock().unwrap();
            panic!("Intentional panic to poison");
        });

        let _ = handle.join();

        // Should recover from poisoning
        let guard = lock(&mutex);
        assert!(guard.is_some());
        assert_eq!(guard.unwrap(), 42);
    }
}
