//! Model warmup utilities for cold-start mitigation.
//!
//! In serverless environments (AWS Lambda, Cloud Functions), the first
//! inference call is significantly slower due to:
//!
//! 1. ONNX graph optimization (done on first inference)
//! 2. Memory allocation and page faults
//! 3. CPU cache warming
//!
//! This module provides utilities to "warm up" models before serving traffic.
//!
//! # Usage
//!
//! ```rust,ignore
//! use anno::backends::warmup::{warmup_model, WarmupConfig};
//!
//! // During initialization (before serving traffic)
//! let model = GLiNEROnnx::new("onnx-community/gliner_small-v2.1")?;
//! warmup_model(&model, WarmupConfig::default())?;
//!
//! // Now ready for production traffic
//! ```
//!
//! # Timing Example
//!
//! ```text
//! Without warmup:
//!   Request 1: 450ms (cold start)
//!   Request 2: 85ms
//!   Request 3: 82ms
//!
//! With warmup:
//!   Warmup: 450ms (during init, not user-facing)
//!   Request 1: 83ms
//!   Request 2: 82ms
//!   Request 3: 84ms
//! ```

use crate::{Model, Result};
use std::time::{Duration, Instant};

/// Configuration for model warmup.
#[derive(Debug, Clone)]
pub struct WarmupConfig {
    /// Number of warmup inference calls.
    pub iterations: usize,
    /// Sample texts for warmup (various lengths).
    pub sample_texts: Vec<String>,
    /// Whether to log warmup progress.
    pub verbose: bool,
    /// Target warmup duration (stops early if reached).
    pub max_duration: Option<Duration>,
}

impl Default for WarmupConfig {
    fn default() -> Self {
        Self {
            iterations: 3,
            sample_texts: vec![
                // Short text
                "John Smith".to_string(),
                // Medium text
                "Marie Curie was a physicist who won the Nobel Prize.".to_string(),
                // Longer text with multiple entities
                "Apple Inc. was founded by Steve Jobs and Steve Wozniak in Cupertino, \
                 California on April 1, 1976. The company went public on December 12, 1980."
                    .to_string(),
            ],
            verbose: true,
            max_duration: Some(Duration::from_secs(30)),
        }
    }
}

impl WarmupConfig {
    /// Create config with specific iteration count.
    #[must_use]
    pub fn with_iterations(mut self, n: usize) -> Self {
        self.iterations = n;
        self
    }

    /// Add a custom sample text.
    #[must_use]
    pub fn with_sample(mut self, text: impl Into<String>) -> Self {
        self.sample_texts.push(text.into());
        self
    }

    /// Set maximum warmup duration.
    #[must_use]
    pub fn with_max_duration(mut self, duration: Duration) -> Self {
        self.max_duration = Some(duration);
        self
    }

    /// Disable verbose logging.
    #[must_use]
    pub fn quiet(mut self) -> Self {
        self.verbose = false;
        self
    }
}

/// Warmup result with timing information.
#[derive(Debug, Clone)]
pub struct WarmupResult {
    /// Total warmup duration.
    pub total_duration: Duration,
    /// Number of inference calls made.
    pub inference_count: usize,
    /// First inference duration (coldest).
    pub first_duration: Duration,
    /// Last inference duration (warmest).
    pub last_duration: Duration,
    /// Average duration after warmup.
    pub average_warm: Duration,
    /// Speedup ratio (first / last).
    pub speedup: f64,
}

impl WarmupResult {
    /// Check if warmup achieved significant speedup.
    #[must_use]
    pub fn is_effective(&self) -> bool {
        self.speedup > 1.5
    }
}

/// Warm up a model by running sample inferences.
///
/// # Arguments
///
/// * `model` - The model to warm up
/// * `config` - Warmup configuration
///
/// # Returns
///
/// `WarmupResult` with timing information.
///
/// # Example
///
/// ```rust,ignore
/// use anno::{GLiNEROnnx, backends::warmup::{warmup_model, WarmupConfig}};
///
/// let model = GLiNEROnnx::new("onnx-community/gliner_small-v2.1")?;
///
/// let result = warmup_model(&model, WarmupConfig::default())?;
/// println!("Warmup speedup: {:.2}x", result.speedup);
/// ```
pub fn warmup_model<M: Model>(model: &M, config: WarmupConfig) -> Result<WarmupResult> {
    let start = Instant::now();
    let mut durations: Vec<Duration> = Vec::new();
    let mut first_duration = Duration::ZERO;
    let mut inference_count = 0;

    if config.verbose {
        log::info!(
            "[warmup] Starting warmup: {} iterations, {} sample texts",
            config.iterations,
            config.sample_texts.len()
        );
    }

    'outer: for iter in 0..config.iterations {
        for text in &config.sample_texts {
            // Check timeout
            if let Some(max) = config.max_duration {
                if start.elapsed() > max {
                    if config.verbose {
                        log::info!("[warmup] Reached max duration, stopping early");
                    }
                    break 'outer;
                }
            }

            let call_start = Instant::now();
            let _ = model.extract_entities(text, None)?;
            let call_duration = call_start.elapsed();

            if inference_count == 0 {
                first_duration = call_duration;
            }
            durations.push(call_duration);
            inference_count += 1;

            if config.verbose && iter == 0 {
                log::debug!(
                    "[warmup] Sample {}: {:?} (text len: {})",
                    inference_count,
                    call_duration,
                    text.len()
                );
            }
        }
    }

    let total_duration = start.elapsed();
    let last_duration = durations.last().copied().unwrap_or(Duration::ZERO);

    // Calculate average of last half (warmed up)
    let warm_count = durations.len() / 2;
    let average_warm = if warm_count > 0 {
        let warm_sum: Duration = durations.iter().skip(durations.len() - warm_count).sum();
        warm_sum / warm_count as u32
    } else {
        last_duration
    };

    let speedup = if last_duration.as_nanos() > 0 {
        first_duration.as_secs_f64() / last_duration.as_secs_f64()
    } else {
        1.0
    };

    let result = WarmupResult {
        total_duration,
        inference_count,
        first_duration,
        last_duration,
        average_warm,
        speedup,
    };

    if config.verbose {
        log::info!(
            "[warmup] Complete: {} inferences in {:?}",
            inference_count,
            total_duration
        );
        log::info!(
            "[warmup] First: {:?}, Last: {:?}, Speedup: {:.2}x",
            first_duration,
            last_duration,
            speedup
        );
    }

    Ok(result)
}

/// Warmup with progress callback.
///
/// Useful for showing progress in CLI tools or updating health checks.
pub fn warmup_with_callback<M: Model, F>(
    model: &M,
    config: WarmupConfig,
    mut callback: F,
) -> Result<WarmupResult>
where
    F: FnMut(usize, usize, Duration),
{
    let start = Instant::now();
    let total_calls = config.iterations * config.sample_texts.len();
    let mut durations: Vec<Duration> = Vec::new();
    let mut first_duration = Duration::ZERO;
    let mut inference_count = 0;

    'outer: for _iter in 0..config.iterations {
        for text in &config.sample_texts {
            if let Some(max) = config.max_duration {
                if start.elapsed() > max {
                    break 'outer;
                }
            }

            let call_start = Instant::now();
            let _ = model.extract_entities(text, None)?;
            let call_duration = call_start.elapsed();

            if inference_count == 0 {
                first_duration = call_duration;
            }
            durations.push(call_duration);
            inference_count += 1;

            // Call progress callback
            callback(inference_count, total_calls, call_duration);
        }
    }

    let total_duration = start.elapsed();
    let last_duration = durations.last().copied().unwrap_or(Duration::ZERO);
    let warm_count = durations.len() / 2;
    let average_warm = if warm_count > 0 {
        let warm_sum: Duration = durations.iter().skip(durations.len() - warm_count).sum();
        warm_sum / warm_count as u32
    } else {
        last_duration
    };

    let speedup = if last_duration.as_nanos() > 0 {
        first_duration.as_secs_f64() / last_duration.as_secs_f64()
    } else {
        1.0
    };

    Ok(WarmupResult {
        total_duration,
        inference_count,
        first_duration,
        last_duration,
        average_warm,
        speedup,
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warmup_config_default() {
        let config = WarmupConfig::default();
        assert_eq!(config.iterations, 3);
        assert!(!config.sample_texts.is_empty());
        assert!(config.verbose);
    }

    #[test]
    fn test_warmup_config_builder() {
        let config = WarmupConfig::default()
            .with_iterations(5)
            .with_sample("Custom text")
            .with_max_duration(Duration::from_secs(10))
            .quiet();

        assert_eq!(config.iterations, 5);
        assert!(config.sample_texts.iter().any(|t| t == "Custom text"));
        assert_eq!(config.max_duration, Some(Duration::from_secs(10)));
        assert!(!config.verbose);
    }

    #[test]
    fn test_warmup_result_effective() {
        let effective = WarmupResult {
            total_duration: Duration::from_secs(1),
            inference_count: 9,
            first_duration: Duration::from_millis(300),
            last_duration: Duration::from_millis(100),
            average_warm: Duration::from_millis(110),
            speedup: 3.0,
        };
        assert!(effective.is_effective());

        let not_effective = WarmupResult {
            speedup: 1.1,
            ..effective.clone()
        };
        assert!(!not_effective.is_effective());
    }
}
