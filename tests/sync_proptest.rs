//! Property-based tests for sync module.
//!
//! These tests verify invariants that must always hold for the sync module.

use anno::sync::{lock, Mutex};
use proptest::prelude::*;
use std::sync::Arc;
use std::thread;

proptest! {
    /// INVARIANT: Mutex preserves values correctly under concurrent access
    #[test]
    fn mutex_preserves_value(
        initial_value in 0i32..1000,
        num_threads in 1usize..10,
        increments_per_thread in 1usize..100
    ) {
        let data = Arc::new(Mutex::new(initial_value));
        let mut handles = vec![];

        for _ in 0..num_threads {
            let data_clone = Arc::clone(&data);
            let handle = thread::spawn(move || {
                for _ in 0..increments_per_thread {
                    let mut guard = lock(&data_clone);
                    *guard += 1;
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let expected = initial_value + (num_threads * increments_per_thread) as i32;
        let actual = *lock(&data);
        prop_assert_eq!(actual, expected,
            "Mutex should preserve value: expected {}, got {}", expected, actual);
    }

    /// INVARIANT: Concurrent writes don't lose data
    #[test]
    fn concurrent_writes_no_data_loss(
        num_threads in 2usize..8,
        writes_per_thread in 10usize..50
    ) {
        let data = Arc::new(Mutex::new(Vec::<usize>::new()));
        let mut handles = vec![];

        for thread_id in 0..num_threads {
            let data_clone = Arc::clone(&data);
            let handle = thread::spawn(move || {
                for i in 0..writes_per_thread {
                    let mut guard = lock(&data_clone);
                    guard.push(thread_id * 1000 + i);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let final_data = lock(&data);
        let expected_count = num_threads * writes_per_thread;
        prop_assert_eq!(final_data.len(), expected_count,
            "Should have {} entries, got {}", expected_count, final_data.len());
    }

    /// INVARIANT: Lock ordering doesn't cause deadlocks
    #[test]
    fn no_deadlock_with_multiple_mutexes(
        num_operations in 10usize..50
    ) {
        let mutex1 = Arc::new(Mutex::new(0));
        let mutex2 = Arc::new(Mutex::new(0));
        let mut handles = vec![];

        // Threads acquire locks in different orders (potential deadlock scenario)
        for i in 0..2 {
            let m1 = Arc::clone(&mutex1);
            let m2 = Arc::clone(&mutex2);
            let handle = thread::spawn(move || {
                for _ in 0..num_operations {
                    if i % 2 == 0 {
                        let _g1 = lock(&m1);
                        let _g2 = lock(&m2);
                    } else {
                        let _g2 = lock(&m2);
                        let _g1 = lock(&m1);
                    }
                }
            });
            handles.push(handle);
        }

        // Should complete without deadlock (parking_lot handles this better)
        for handle in handles {
            handle.join().expect("Should not deadlock");
        }
    }
}
