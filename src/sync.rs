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
