//! Profiling utilities for performance analysis.
//!
//! Provides timing instrumentation to identify bottlenecks in evaluation.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Profiling timer that tracks time spent in different operations.
#[derive(Debug, Clone)]
pub struct Profiler {
    timings: HashMap<String, Vec<Duration>>,
    current_timers: HashMap<String, Instant>,
}

impl Profiler {
    /// Create a new profiler.
    pub fn new() -> Self {
        Self {
            timings: HashMap::new(),
            current_timers: HashMap::new(),
        }
    }

    /// Start timing an operation.
    pub fn start(&mut self, operation: &str) {
        self.current_timers
            .insert(operation.to_string(), Instant::now());
    }

    /// Stop timing an operation and record the duration.
    pub fn stop(&mut self, operation: &str) {
        if let Some(start) = self.current_timers.remove(operation) {
            let duration = start.elapsed();
            self.timings
                .entry(operation.to_string())
                .or_insert_with(Vec::new)
                .push(duration);
        }
    }

    /// Time a closure and record the duration.
    pub fn time<F, R>(&mut self, operation: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        self.start(operation);
        let result = f();
        self.stop(operation);
        result
    }

    /// Get summary statistics for all timed operations.
    pub fn summary(&self) -> HashMap<String, TimingStats> {
        self.timings
            .iter()
            .map(|(name, durations)| {
                let total: Duration = durations.iter().sum();
                let count = durations.len();
                let avg = if count > 0 {
                    total / count as u32
                } else {
                    Duration::ZERO
                };
                let min = durations.iter().min().copied().unwrap_or(Duration::ZERO);
                let max = durations.iter().max().copied().unwrap_or(Duration::ZERO);

                (
                    name.clone(),
                    TimingStats {
                        total,
                        count,
                        avg,
                        min,
                        max,
                    },
                )
            })
            .collect()
    }

    /// Print a human-readable summary to stderr.
    pub fn print_summary(&self) {
        let summary = self.summary();
        eprintln!("\n=== Profiling Summary ===");
        eprintln!(
            "{:<30} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "Operation", "Count", "Total (ms)", "Avg (ms)", "Min (ms)", "Max (ms)"
        );
        eprintln!("{}", "-".repeat(90));

        let mut sorted: Vec<_> = summary.iter().collect();
        sorted.sort_by(|a, b| b.1.total.cmp(&a.1.total)); // Sort by total time descending

        for (name, stats) in sorted {
            eprintln!(
                "{:<30} {:>10} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
                name,
                stats.count,
                stats.total.as_secs_f64() * 1000.0,
                stats.avg.as_secs_f64() * 1000.0,
                stats.min.as_secs_f64() * 1000.0,
                stats.max.as_secs_f64() * 1000.0,
            );
        }
        eprintln!();
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for a timed operation.
#[derive(Debug, Clone)]
pub struct TimingStats {
    /// Total time spent in this operation
    pub total: Duration,
    /// Number of times this operation was called
    pub count: usize,
    /// Average time per call
    pub avg: Duration,
    /// Minimum time for a single call
    pub min: Duration,
    /// Maximum time for a single call
    pub max: Duration,
}

/// Global profiler instance (thread-local for thread safety).
#[cfg(feature = "eval-profiling")]
thread_local! {
    static PROFILER: std::cell::RefCell<Profiler> = std::cell::RefCell::new(Profiler::new());
}

/// Start timing an operation (if profiling is enabled).
#[cfg(feature = "eval-profiling")]
pub fn start(operation: &str) {
    PROFILER.with(|p| p.borrow_mut().start(operation));
}

/// Stop timing an operation (if profiling is enabled).
#[cfg(feature = "eval-profiling")]
pub fn stop(operation: &str) {
    PROFILER.with(|p| p.borrow_mut().stop(operation));
}

/// Time a closure (if profiling is enabled).
#[cfg(feature = "eval-profiling")]
pub fn time<F, R>(operation: &str, f: F) -> R
where
    F: FnOnce() -> R,
{
    PROFILER.with(|p| p.borrow_mut().time(operation, f))
}

/// Print profiling summary (if profiling is enabled).
#[cfg(feature = "eval-profiling")]
pub fn print_summary() {
    PROFILER.with(|p| p.borrow().print_summary());
}

/// No-op implementations when profiling is disabled.
#[cfg(not(feature = "eval-profiling"))]
pub fn start(_operation: &str) {}

#[cfg(not(feature = "eval-profiling"))]
pub fn stop(_operation: &str) {}

#[cfg(not(feature = "eval-profiling"))]
pub fn time<F, R>(_operation: &str, f: F) -> R
where
    F: FnOnce() -> R,
{
    f()
}

#[cfg(not(feature = "eval-profiling"))]
pub fn print_summary() {}
