//! Synchronization primitives with conditional compilation.
//!
//! Provides a unified mutex interface that uses `parking_lot::Mutex` when
//! the `fast-lock` feature is enabled, falling back to `std::sync::Mutex` otherwise.

#[cfg(feature = "fast-lock")]
use parking_lot::Mutex as ParkingLotMutex;

#[cfg(not(feature = "fast-lock"))]
use std::sync::Mutex as StdMutex;

/// Mutex type that conditionally uses parking_lot or std::sync::Mutex.
///
/// When `fast-lock` feature is enabled, uses `parking_lot::Mutex` for better
/// performance (1.5-3x faster on uncontended locks). Otherwise uses `std::sync::Mutex`.
///
/// # Example
///
/// ```rust
/// use anno::sync::Mutex;
///
/// let data = Mutex::new(42);
/// *data.lock() = 100;
/// ```
#[cfg(feature = "fast-lock")]
pub type Mutex<T> = ParkingLotMutex<T>;

/// Mutex type using std::sync::Mutex (default, no fast-lock feature).
#[cfg(not(feature = "fast-lock"))]
pub type Mutex<T> = StdMutex<T>;

/// Lock a mutex and return the guard, handling poisoning gracefully.
///
/// For `parking_lot::Mutex`, this is just `mutex.lock()`.
/// For `std::sync::Mutex`, this handles poisoning by recovering the guard.
///
/// # Example
///
/// ```rust
/// use anno::sync::{Mutex, lock};
///
/// let mutex = Mutex::new(42);
/// let guard = lock(&mutex);
/// ```
#[cfg(feature = "fast-lock")]
pub fn lock<T>(mutex: &Mutex<T>) -> parking_lot::MutexGuard<'_, T> {
    mutex.lock()
}

/// Lock a mutex using std::sync::Mutex, recovering from poisoning.
#[cfg(not(feature = "fast-lock"))]
pub fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// Attempt to acquire a mutex lock without blocking.
///
/// Returns `Ok(guard)` if the lock was acquired immediately, or `Err` if the lock
/// is currently held by another thread or if poisoning occurred.
///
/// For `parking_lot::Mutex`, this uses `try_lock()` which returns `None` if the lock
/// is held, converting it to an error.
/// For `std::sync::Mutex`, this uses `try_lock()` and converts `PoisonError` to our `Error` type.
///
/// # Example
///
/// ```rust
/// use anno::sync::{Mutex, try_lock};
/// use anno::Result;
///
/// let mutex = Mutex::new(42);
/// match try_lock(&mutex) {
///     Ok(guard) => println!("Lock acquired: {}", *guard),
///     Err(e) => println!("Lock failed: {}", e),
/// }
/// ```
#[cfg(feature = "fast-lock")]
pub fn try_lock<T>(mutex: &Mutex<T>) -> crate::Result<parking_lot::MutexGuard<'_, T>> {
    mutex
        .try_lock()
        .ok_or_else(|| crate::Error::Retrieval("Mutex lock failed: would block".to_string()))
}

/// Try to lock a mutex using std::sync::Mutex without blocking.
#[cfg(not(feature = "fast-lock"))]
pub fn try_lock<T>(mutex: &Mutex<T>) -> crate::Result<std::sync::MutexGuard<'_, T>> {
    mutex.try_lock().map_err(|e| match e {
        std::sync::TryLockError::Poisoned(poison) => {
            crate::Error::Retrieval(format!("Mutex lock failed: poisoned - {}", poison))
        }
        std::sync::TryLockError::WouldBlock => {
            crate::Error::Retrieval("Mutex lock failed: would block".to_string())
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutex_new_and_lock() {
        let mutex = Mutex::new(42);
        let guard = lock(&mutex);
        assert_eq!(*guard, 42);
    }

    #[test]
    fn test_mutex_modify_value() {
        let mutex = Mutex::new(0);
        {
            let mut guard = lock(&mutex);
            *guard = 100;
        }
        let guard = lock(&mutex);
        assert_eq!(*guard, 100);
    }

    #[test]
    fn test_try_lock_success() {
        let mutex = Mutex::new("test");
        let result = try_lock(&mutex);
        assert!(result.is_ok());
        assert_eq!(*result.unwrap(), "test");
    }

    #[test]
    fn test_mutex_with_struct() {
        #[derive(Debug, PartialEq)]
        struct Data {
            value: i32,
            name: String,
        }

        let mutex = Mutex::new(Data {
            value: 42,
            name: "test".to_string(),
        });

        let guard = lock(&mutex);
        assert_eq!(guard.value, 42);
        assert_eq!(guard.name, "test");
    }

    #[test]
    fn test_multiple_locks_sequential() {
        let mutex = Mutex::new(0);

        for i in 0..10 {
            let mut guard = lock(&mutex);
            *guard = i;
        }

        let guard = lock(&mutex);
        assert_eq!(*guard, 9);
    }

    #[test]
    fn test_try_lock_returns_correct_value() {
        let mutex = Mutex::new(vec![1, 2, 3]);

        // Must bind the guard explicitly to control its lifetime
        let result = try_lock(&mutex);
        match result {
            Ok(guard) => {
                assert_eq!(guard.len(), 3);
                assert_eq!(guard[0], 1);
            }
            Err(_) => {
                panic!("try_lock should succeed when not contested");
            }
        }
    }
}
