//! Evaluation types: MetricValue, GoalCheckResult, etc.
//!
//! These are shared primitives for evaluation that can be reused
//! across NER evaluation and other evaluation tasks.

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A type-safe metric value bounded to [0.0, 1.0].
///
/// Ensures metrics like precision, recall, and F1 are always valid.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, PartialOrd)]
#[serde(transparent)]
pub struct MetricValue(f64);

impl MetricValue {
    /// Create a new MetricValue, clamping to [0.0, 1.0].
    ///
    /// # Example
    /// ```
    /// use anno::eval::MetricValue;
    /// let v = MetricValue::new(0.95);
    /// assert!((v.get() - 0.95).abs() < 1e-6);
    /// ```
    pub fn new(value: f64) -> Self {
        MetricValue(value.clamp(0.0, 1.0))
    }

    /// Try to create a MetricValue, returning error if out of bounds.
    pub fn try_new(value: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&value) {
            return Err(Error::InvalidInput(format!(
                "MetricValue must be in [0.0, 1.0], got {}",
                value
            )));
        }
        Ok(MetricValue(value))
    }

    /// Get the underlying value.
    #[inline]
    pub fn get(&self) -> f64 {
        self.0
    }
}

impl Default for MetricValue {
    fn default() -> Self {
        MetricValue(0.0)
    }
}

impl std::fmt::Display for MetricValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.4}", self.0)
    }
}

impl From<f64> for MetricValue {
    fn from(value: f64) -> Self {
        MetricValue::new(value)
    }
}

/// Result of checking evaluation goals.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GoalCheckResult {
    /// Whether all goals were met.
    pub passed: bool,
    /// Individual goal check results.
    pub checks: HashMap<String, GoalCheck>,
    /// Summary message.
    pub summary: Option<String>,
}

impl GoalCheckResult {
    /// Create a new GoalCheckResult (defaults to passed = true).
    pub fn new() -> Self {
        Self {
            passed: true,
            checks: HashMap::new(),
            summary: None,
        }
    }

    /// Add a goal check result.
    pub fn add_check(&mut self, name: impl Into<String>, check: GoalCheck) {
        if !check.passed {
            self.passed = false;
        }
        self.checks.insert(name.into(), check);
    }

    /// Add a failure (convenience method for add_check with fail).
    pub fn add_failure(&mut self, name: impl Into<String>, actual: f64, threshold: f64) {
        self.add_check(name, GoalCheck::fail(threshold, actual));
    }

    /// Add a success (convenience method for add_check with pass).
    pub fn add_success(&mut self, name: impl Into<String>, actual: f64, threshold: f64) {
        self.add_check(name, GoalCheck::pass(threshold, actual));
    }

    /// Get number of passed checks.
    pub fn passed_count(&self) -> usize {
        self.checks.values().filter(|c| c.passed).count()
    }

    /// Get number of failed checks.
    pub fn failed_count(&self) -> usize {
        self.checks.values().filter(|c| !c.passed).count()
    }
}

/// Individual goal check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalCheck {
    /// Whether this goal was met.
    pub passed: bool,
    /// Expected threshold.
    pub threshold: f64,
    /// Actual value achieved.
    pub actual: f64,
    /// Optional message.
    pub message: Option<String>,
}

impl GoalCheck {
    /// Create a new goal check.
    pub fn new(passed: bool, threshold: f64, actual: f64) -> Self {
        Self {
            passed,
            threshold,
            actual,
            message: None,
        }
    }

    /// Create a passing check.
    pub fn pass(threshold: f64, actual: f64) -> Self {
        Self::new(true, threshold, actual)
    }

    /// Create a failing check.
    pub fn fail(threshold: f64, actual: f64) -> Self {
        Self::new(false, threshold, actual)
    }

    /// Add a message to the check.
    pub fn with_message(mut self, msg: impl Into<String>) -> Self {
        self.message = Some(msg.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_value_clamping() {
        assert_eq!(MetricValue::new(0.5).get(), 0.5);
        assert_eq!(MetricValue::new(-0.5).get(), 0.0);
        assert_eq!(MetricValue::new(1.5).get(), 1.0);
    }

    #[test]
    fn test_metric_value_try_new() {
        assert!(MetricValue::try_new(0.5).is_ok());
        assert!(MetricValue::try_new(-0.1).is_err());
        assert!(MetricValue::try_new(1.1).is_err());
    }

    #[test]
    fn test_goal_check_result() {
        let mut result = GoalCheckResult::new();
        assert!(result.passed);

        result.add_check("precision", GoalCheck::pass(0.8, 0.85));
        assert!(result.passed);

        result.add_check("recall", GoalCheck::fail(0.9, 0.75));
        assert!(!result.passed);

        assert_eq!(result.passed_count(), 1);
        assert_eq!(result.failed_count(), 1);
    }
}
