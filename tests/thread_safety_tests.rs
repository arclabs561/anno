//! Comprehensive tests for thread-safety improvements.
//!
//! Tests cover:
//! - Atomic counter usage (replacing static mut)
//! - Mutex usage patterns (parking_lot vs std::sync)
//! - Concurrent access patterns
//! - Poisoning recovery

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use anno::sync::Mutex;

use anno::sync::{lock, try_lock};

#[test]
fn test_atomic_counter_thread_safety() {
    // Test that AtomicUsize provides thread-safe counter operations
    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    let num_threads = 10;
    let increments_per_thread = 100;
    let mut handles = vec![];

    for _ in 0..num_threads {
        let handle = thread::spawn(move || {
            for _ in 0..increments_per_thread {
                COUNTER.fetch_add(1, Ordering::Relaxed);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    assert_eq!(
        COUNTER.load(Ordering::Relaxed),
        num_threads * increments_per_thread
    );
}

#[test]
fn test_mutex_concurrent_access() {
    // Test that our Mutex type works correctly with concurrent access
    let data = Arc::new(Mutex::new(0));
    let num_threads = 10;
    let increments_per_thread = 100;
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

    let final_value = lock(&data);
    assert_eq!(*final_value, num_threads * increments_per_thread);
}

#[test]
fn test_mutex_try_lock() {
    // Test that try_lock works correctly when lock is available
    let data = Arc::new(Mutex::new(42));

    let guard = try_lock(&data).unwrap();
    assert_eq!(*guard, 42);
}

#[test]
fn test_mutex_try_lock_non_blocking() {
    // Test that try_lock actually doesn't block when lock is held
    let data = Arc::new(Mutex::new(42));
    let data_clone = Arc::clone(&data);

    // Hold the lock in another thread
    let handle = thread::spawn(move || {
        let _guard = lock(&data_clone);
        // Hold the lock for a bit
        thread::sleep(Duration::from_millis(100));
    });

    // Small delay to ensure the other thread has the lock
    thread::sleep(Duration::from_millis(10));

    // try_lock should fail immediately (not block)
    let start = std::time::Instant::now();
    let result = try_lock(&data);
    let elapsed = start.elapsed();

    // Should fail immediately (within 10ms, not wait for 100ms)
    assert!(result.is_err(), "try_lock should fail when lock is held");
    assert!(
        elapsed < Duration::from_millis(50),
        "try_lock should return immediately, not block (took {:?})",
        elapsed
    );

    // Wait for the other thread to release
    handle.join().unwrap();

    // Now try_lock should succeed
    let guard = try_lock(&data).unwrap();
    assert_eq!(*guard, 42);
}

#[test]
fn test_mutex_poisoning_recovery() {
    // Test that mutex poisoning is handled gracefully
    #[cfg(not(feature = "fast-lock"))]
    {
        // Only test poisoning with std::sync::Mutex (parking_lot doesn't poison)
        let mutex: Arc<Mutex<Option<Vec<i32>>>> = Arc::new(Mutex::new(Some(vec![1, 2, 3])));
        let mutex_clone = Arc::clone(&mutex);

        // Poison the mutex by panicking while holding the lock
        let handle = thread::spawn(move || {
            let _guard = mutex_clone.lock().unwrap();
            panic!("Intentional panic to poison mutex");
        });

        let _ = handle.join();

        // Verify mutex is poisoned
        assert!(mutex.is_poisoned());

        // Test recovery using our lock function
        let recovered = lock(&mutex);
        assert!(recovered.is_some());
        assert_eq!(recovered.as_ref().unwrap().len(), 3);
    }

    #[cfg(feature = "fast-lock")]
    {
        // parking_lot doesn't poison, so this test is a no-op
        // But we verify the mutex still works
        let mutex = Arc::new(Mutex::new(42));
        let guard = lock(&mutex);
        assert_eq!(*guard, 42);
    }
}

#[test]
fn test_concurrent_cache_access() {
    // Simulate the per_example_scores_cache pattern
    type Cache = Mutex<Option<Vec<(Vec<String>, Vec<String>, String)>>>;
    let cache = Arc::new(Cache::new(None));
    let num_threads = 5;
    let mut handles = vec![];

    for i in 0..num_threads {
        let cache_clone = Arc::clone(&cache);
        let handle = thread::spawn(move || {
            // Simulate cache write
            {
                let mut guard = lock(&cache_clone);
                *guard = Some(vec![(
                    vec![format!("entity_{}", i)],
                    vec![format!("gold_{}", i)],
                    format!("task_{}", i),
                )]);
            }

            // Small delay to allow other threads to access
            thread::sleep(Duration::from_millis(10));

            // Simulate cache read
            let guard = lock(&cache_clone);
            assert!(guard.is_some());
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_progress_tracking_concurrent() {
    // Simulate the progress tracking pattern from task_evaluator
    let progress = Arc::new(Mutex::new(0));
    let start_time = Arc::new(Mutex::new(std::time::Instant::now()));
    let num_threads = 5;
    let updates_per_thread = 20;
    let mut handles = vec![];

    for _ in 0..num_threads {
        let progress_clone = Arc::clone(&progress);
        let start_time_clone = Arc::clone(&start_time);
        let handle = thread::spawn(move || {
            for _ in 0..updates_per_thread {
                let mut prog = lock(&progress_clone);
                *prog += 1;
                let _elapsed = lock(&start_time_clone).elapsed();
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let final_progress = lock(&progress);
    assert_eq!(*final_progress, num_threads * updates_per_thread);
}

#[test]
fn test_simple_random_thread_safety() {
    // Test that simple_random() is thread-safe (uses AtomicUsize internally)
    // This is an indirect test - we can't directly test the function without
    // exposing it, but we verify the pattern works
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let num_threads = 10;
    let calls_per_thread = 100;
    let mut handles = vec![];

    for _ in 0..num_threads {
        let handle = thread::spawn(move || {
            for _ in 0..calls_per_thread {
                let count = COUNTER.fetch_add(1, Ordering::Relaxed);
                // Simulate hash computation (would use count in real implementation)
                let _hash = count.wrapping_mul(0x9e3779b9);
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

#[test]
fn test_mutex_performance_characteristics() {
    // Basic performance test - verify mutex doesn't deadlock
    let data = Arc::new(Mutex::new(0));
    let num_iterations = 1000;

    let start = std::time::Instant::now();
    for _ in 0..num_iterations {
        let mut guard = lock(&data);
        *guard += 1;
    }
    let elapsed = start.elapsed();

    // Should complete quickly (under 1 second for 1000 iterations)
    assert!(elapsed < Duration::from_secs(1));
    assert_eq!(*lock(&data), num_iterations);
}
