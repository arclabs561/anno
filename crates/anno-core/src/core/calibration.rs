//! Confidence calibration utilities for NER systems.
//!
//! Model confidence scores are often miscalibrated - a 0.9 confidence doesn't mean
//! 90% of such predictions are correct. This module provides tools to calibrate
//! confidence scores using held-out validation data.
//!
//! # Background
//!
//! GLiNER and similar models output confidence scores that:
//! - Are not true probabilities
//! - May be systematically over- or under-confident
//! - Vary in calibration across entity types
//!
//! # Calibration Methods
//!
//! - **Isotonic regression**: Non-parametric, monotonic transformation
//! - **Platt scaling**: Logistic regression on logits
//! - **Temperature scaling**: Single parameter adjustment
//!
//! # Usage
//!
//! ```rust,ignore
//! use anno_core::calibration::{IsotonicCalibrator, Calibrator};
//!
//! // Collect (predicted_confidence, was_correct) pairs from validation set
//! let validation_data = vec![
//!     (0.9, true), (0.8, true), (0.7, false), (0.6, true), ...
//! ];
//!
//! let calibrator = IsotonicCalibrator::fit(&validation_data);
//!
//! // Now calibrate new predictions
//! let raw_confidence = 0.85;
//! let calibrated = calibrator.calibrate(raw_confidence);
//! ```

use serde::{Deserialize, Serialize};

/// Trait for confidence calibration methods.
pub trait Calibrator: Send + Sync {
    /// Calibrate a raw confidence score to a calibrated probability.
    fn calibrate(&self, raw_confidence: f32) -> f32;

    /// Calibrate a batch of confidence scores.
    fn calibrate_batch(&self, raw_confidences: &[f32]) -> Vec<f32> {
        raw_confidences.iter().map(|c| self.calibrate(*c)).collect()
    }

    /// Get the name of this calibration method.
    fn method_name(&self) -> &str;
}

/// Identity calibrator (no calibration, returns raw scores).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IdentityCalibrator;

impl Calibrator for IdentityCalibrator {
    fn calibrate(&self, raw_confidence: f32) -> f32 {
        raw_confidence
    }

    fn method_name(&self) -> &str {
        "identity"
    }
}

/// Temperature scaling calibrator.
///
/// Applies a single temperature parameter to logits:
/// `calibrated = sigmoid(logit(raw) / temperature)`
///
/// Temperature > 1 → less confident
/// Temperature < 1 → more confident
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureCalibrator {
    /// Temperature parameter (1.0 = no change)
    pub temperature: f32,
}

impl Default for TemperatureCalibrator {
    fn default() -> Self {
        Self { temperature: 1.0 } // Identity calibration
    }
}

impl TemperatureCalibrator {
    /// Create a new temperature calibrator.
    #[must_use]
    pub fn new(temperature: f32) -> Self {
        Self {
            temperature: temperature.max(0.001), // Prevent division by zero
        }
    }

    /// Fit temperature using grid search on validation data.
    ///
    /// # Arguments
    /// * `validation_data` - Pairs of (predicted_confidence, was_correct)
    #[must_use]
    pub fn fit(validation_data: &[(f32, bool)]) -> Self {
        if validation_data.is_empty() {
            return Self::new(1.0);
        }

        // Grid search for best temperature
        let mut best_temp = 1.0;
        let mut best_loss = f32::MAX;

        for temp_idx in 1..=100 {
            let temp = temp_idx as f32 * 0.1; // 0.1 to 10.0
            let calibrator = Self::new(temp);

            let loss = calibrator.cross_entropy_loss(validation_data);
            if loss < best_loss {
                best_loss = loss;
                best_temp = temp;
            }
        }

        Self::new(best_temp)
    }

    /// Compute cross-entropy loss for calibration evaluation.
    fn cross_entropy_loss(&self, data: &[(f32, bool)]) -> f32 {
        let eps = 1e-7;
        let n = data.len() as f32;

        data.iter()
            .map(|(conf, correct)| {
                let p = self.calibrate(*conf).clamp(eps, 1.0 - eps);
                if *correct {
                    -p.ln()
                } else {
                    -(1.0 - p).ln()
                }
            })
            .sum::<f32>()
            / n
    }
}

impl Calibrator for TemperatureCalibrator {
    fn calibrate(&self, raw_confidence: f32) -> f32 {
        // Convert confidence to logit, scale, convert back
        let raw_clamped = raw_confidence.clamp(1e-7, 1.0 - 1e-7);
        let logit = (raw_clamped / (1.0 - raw_clamped)).ln();
        let scaled_logit = logit / self.temperature;
        1.0 / (1.0 + (-scaled_logit).exp())
    }

    fn method_name(&self) -> &str {
        "temperature_scaling"
    }
}

/// Isotonic regression calibrator.
///
/// Non-parametric calibration that learns a monotonic mapping from
/// raw confidences to calibrated probabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsotonicCalibrator {
    /// Sorted thresholds (raw confidence values)
    thresholds: Vec<f32>,
    /// Corresponding calibrated values
    calibrated_values: Vec<f32>,
}

impl IsotonicCalibrator {
    /// Fit isotonic regression on validation data.
    ///
    /// # Arguments
    /// * `validation_data` - Pairs of (predicted_confidence, was_correct)
    #[must_use]
    pub fn fit(validation_data: &[(f32, bool)]) -> Self {
        if validation_data.is_empty() {
            return Self {
                thresholds: vec![0.0, 1.0],
                calibrated_values: vec![0.0, 1.0],
            };
        }

        // Sort by confidence
        let mut sorted: Vec<_> = validation_data.to_vec();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Bin into ~20 bins for stability
        let bin_size = (sorted.len() / 20).max(1);
        let mut thresholds = Vec::new();
        let mut calibrated_values = Vec::new();

        for chunk in sorted.chunks(bin_size) {
            let avg_conf: f32 = chunk.iter().map(|(c, _)| c).sum::<f32>() / chunk.len() as f32;
            let accuracy: f32 =
                chunk.iter().filter(|(_, correct)| *correct).count() as f32 / chunk.len() as f32;

            thresholds.push(avg_conf);
            calibrated_values.push(accuracy);
        }

        // Apply isotonic constraint (monotonically increasing)
        Self::make_isotonic(&mut calibrated_values);

        Self {
            thresholds,
            calibrated_values,
        }
    }

    /// Ensure values are monotonically increasing.
    fn make_isotonic(values: &mut [f32]) {
        if values.len() < 2 {
            return;
        }

        // Pool Adjacent Violators algorithm
        let mut i = 0;
        while i < values.len() - 1 {
            if values[i] > values[i + 1] {
                // Violation found - merge with next
                let merged = (values[i] + values[i + 1]) / 2.0;
                values[i] = merged;
                values[i + 1] = merged;

                // Check backwards for new violations
                while i > 0 && values[i - 1] > values[i] {
                    let merged = (values[i - 1] + values[i]) / 2.0;
                    values[i - 1] = merged;
                    values[i] = merged;
                    i -= 1;
                }
            }
            i += 1;
        }
    }
}

impl Calibrator for IsotonicCalibrator {
    fn calibrate(&self, raw_confidence: f32) -> f32 {
        if self.thresholds.is_empty() {
            return raw_confidence;
        }

        // Binary search for the right bin
        let idx = self
            .thresholds
            .binary_search_by(|t| {
                t.partial_cmp(&raw_confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or_else(|i| i.saturating_sub(1).min(self.thresholds.len() - 1));

        // Linear interpolation between bins
        if idx >= self.thresholds.len() - 1 {
            return *self.calibrated_values.last().unwrap_or(&raw_confidence);
        }

        let t0 = self.thresholds[idx];
        let t1 = self.thresholds[idx + 1];
        let v0 = self.calibrated_values[idx];
        let v1 = self.calibrated_values[idx + 1];

        if (t1 - t0).abs() < 1e-7 {
            return v0;
        }

        let alpha = (raw_confidence - t0) / (t1 - t0);
        v0 + alpha * (v1 - v0)
    }

    fn method_name(&self) -> &str {
        "isotonic_regression"
    }
}

/// Per-type calibrator that uses different calibration for each entity type.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerTypeCalibrator<C: Calibrator + Clone> {
    /// Calibrators for each entity type
    type_calibrators: std::collections::HashMap<String, C>,
    /// Default calibrator for unknown types
    default_calibrator: Option<C>,
}

impl<C: Calibrator + Clone + Default> PerTypeCalibrator<C> {
    /// Create a new per-type calibrator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            type_calibrators: std::collections::HashMap::new(),
            default_calibrator: None,
        }
    }

    /// Add a calibrator for a specific entity type.
    pub fn add_type_calibrator(&mut self, entity_type: impl Into<String>, calibrator: C) {
        self.type_calibrators.insert(entity_type.into(), calibrator);
    }

    /// Set the default calibrator.
    pub fn set_default(&mut self, calibrator: C) {
        self.default_calibrator = Some(calibrator);
    }

    /// Calibrate a confidence for a specific entity type.
    pub fn calibrate_typed(&self, raw_confidence: f32, entity_type: &str) -> f32 {
        if let Some(cal) = self.type_calibrators.get(entity_type) {
            cal.calibrate(raw_confidence)
        } else if let Some(cal) = &self.default_calibrator {
            cal.calibrate(raw_confidence)
        } else {
            raw_confidence
        }
    }
}

/// Calibration metrics for evaluating calibration quality.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationMetrics {
    /// Expected Calibration Error (lower is better)
    pub ece: f32,
    /// Maximum Calibration Error
    pub mce: f32,
    /// Brier score (combines calibration + discrimination)
    pub brier_score: f32,
    /// Number of samples
    pub n_samples: usize,
}

impl CalibrationMetrics {
    /// Compute calibration metrics from predictions.
    ///
    /// # Arguments
    /// * `predictions` - Pairs of (confidence, was_correct)
    /// * `n_bins` - Number of bins for ECE/MCE computation
    #[must_use]
    pub fn compute(predictions: &[(f32, bool)], n_bins: usize) -> Self {
        if predictions.is_empty() {
            return Self {
                ece: 0.0,
                mce: 0.0,
                brier_score: 0.0,
                n_samples: 0,
            };
        }

        let n = predictions.len() as f32;

        // Brier score
        let brier_score: f32 = predictions
            .iter()
            .map(|(conf, correct)| {
                let target = if *correct { 1.0 } else { 0.0 };
                (conf - target).powi(2)
            })
            .sum::<f32>()
            / n;

        // Bin predictions for ECE/MCE
        let bin_size = 1.0 / n_bins as f32;
        let mut bin_counts = vec![0usize; n_bins];
        let mut bin_conf_sum = vec![0.0f32; n_bins];
        let mut bin_acc_sum = vec![0.0f32; n_bins];

        for (conf, correct) in predictions {
            let bin_idx = ((conf / bin_size).floor() as usize).min(n_bins - 1);
            bin_counts[bin_idx] += 1;
            bin_conf_sum[bin_idx] += conf;
            bin_acc_sum[bin_idx] += if *correct { 1.0 } else { 0.0 };
        }

        // ECE and MCE
        let mut ece = 0.0f32;
        let mut mce = 0.0f32;

        for i in 0..n_bins {
            if bin_counts[i] > 0 {
                let avg_conf = bin_conf_sum[i] / bin_counts[i] as f32;
                let avg_acc = bin_acc_sum[i] / bin_counts[i] as f32;
                let gap = (avg_conf - avg_acc).abs();

                ece += (bin_counts[i] as f32 / n) * gap;
                mce = mce.max(gap);
            }
        }

        Self {
            ece,
            mce,
            brier_score,
            n_samples: predictions.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Identity Calibrator
    // =========================================================================

    #[test]
    fn test_identity_calibrator() {
        let cal = IdentityCalibrator;
        assert_eq!(cal.calibrate(0.0), 0.0);
        assert_eq!(cal.calibrate(0.5), 0.5);
        assert_eq!(cal.calibrate(1.0), 1.0);
        assert_eq!(cal.method_name(), "identity");
    }

    #[test]
    fn test_identity_calibrator_batch() {
        let cal = IdentityCalibrator;
        let input = vec![0.1, 0.5, 0.9];
        let output = cal.calibrate_batch(&input);
        assert_eq!(input, output);
    }

    // =========================================================================
    // Temperature Calibrator
    // =========================================================================

    #[test]
    fn test_temperature_calibrator_identity() {
        // Temperature 1.0 should be identity
        let cal = TemperatureCalibrator::new(1.0);
        assert!((cal.calibrate(0.5) - 0.5).abs() < 0.01);
        assert!((cal.calibrate(0.1) - 0.1).abs() < 0.02);
        assert!((cal.calibrate(0.9) - 0.9).abs() < 0.02);
    }

    #[test]
    fn test_temperature_calibrator_softening() {
        // Higher temperature should reduce confidence (move toward 0.5)
        let cal_high = TemperatureCalibrator::new(2.0);
        assert!(cal_high.calibrate(0.9) < 0.9);
        assert!(cal_high.calibrate(0.9) > 0.5); // Still above 0.5
        assert!(cal_high.calibrate(0.1) > 0.1);
        assert!(cal_high.calibrate(0.1) < 0.5); // Still below 0.5
    }

    #[test]
    fn test_temperature_calibrator_sharpening() {
        // Lower temperature should increase confidence (move away from 0.5)
        let cal_low = TemperatureCalibrator::new(0.5);
        assert!(cal_low.calibrate(0.7) > 0.7);
        assert!(cal_low.calibrate(0.3) < 0.3);
    }

    #[test]
    fn test_temperature_calibrator_fit() {
        // Overconfident predictions (high confidence, low accuracy)
        let overconfident = vec![
            (0.9, false),
            (0.9, false),
            (0.9, true),
            (0.8, false),
            (0.8, true),
            (0.8, false),
            (0.7, true),
            (0.7, false),
            (0.7, true),
        ];
        let cal = TemperatureCalibrator::fit(&overconfident);

        // Fit should find temperature > 1 to reduce confidence
        assert!(
            cal.temperature > 1.0,
            "Expected temp > 1 for overconfident, got {}",
            cal.temperature
        );
    }

    #[test]
    fn test_temperature_calibrator_fit_empty() {
        let cal = TemperatureCalibrator::fit(&[]);
        assert_eq!(cal.temperature, 1.0);
    }

    // =========================================================================
    // Isotonic Calibrator
    // =========================================================================

    #[test]
    fn test_isotonic_calibrator_monotonicity() {
        let data = vec![
            (0.1, false),
            (0.2, false),
            (0.3, true),
            (0.4, false),
            (0.5, true),
            (0.6, true),
            (0.7, true),
            (0.8, true),
            (0.9, true),
        ];
        let cal = IsotonicCalibrator::fit(&data);

        // Test monotonicity across entire range
        let mut prev = cal.calibrate(0.0);
        for i in 1..=100 {
            let conf = i as f32 / 100.0;
            let calibrated = cal.calibrate(conf);
            assert!(
                calibrated >= prev - 1e-6,
                "Non-monotonic at {}: {} < {}",
                conf,
                calibrated,
                prev
            );
            prev = calibrated;
        }
    }

    #[test]
    fn test_isotonic_calibrator_empty() {
        let cal = IsotonicCalibrator::fit(&[]);
        // Should return identity for empty fit
        assert_eq!(cal.thresholds, vec![0.0, 1.0]);
    }

    #[test]
    fn test_isotonic_calibrator_single_sample() {
        let cal = IsotonicCalibrator::fit(&[(0.5, true)]);
        // Should handle gracefully
        let _ = cal.calibrate(0.5);
    }

    // =========================================================================
    // Per-Type Calibrator
    // =========================================================================

    #[test]
    fn test_per_type_calibrator() {
        let mut cal = PerTypeCalibrator::<TemperatureCalibrator>::new();

        // Different temperatures for different types
        cal.add_type_calibrator("PER", TemperatureCalibrator::new(0.5)); // Sharpen
        cal.add_type_calibrator("ORG", TemperatureCalibrator::new(2.0)); // Soften
        cal.set_default(TemperatureCalibrator::new(1.0));

        let raw = 0.7;
        let per_cal = cal.calibrate_typed(raw, "PER");
        let org_cal = cal.calibrate_typed(raw, "ORG");
        let unknown = cal.calibrate_typed(raw, "UNKNOWN");

        // PER should be sharpened (higher)
        assert!(per_cal > raw, "PER not sharpened");
        // ORG should be softened (closer to 0.5)
        assert!(org_cal < raw, "ORG not softened");
        // Unknown should use default (identity at temp=1)
        assert!((unknown - raw).abs() < 0.05, "Unknown not using default");
    }

    #[test]
    fn test_per_type_calibrator_no_default() {
        let mut cal = PerTypeCalibrator::<TemperatureCalibrator>::new();
        cal.add_type_calibrator("PER", TemperatureCalibrator::new(2.0));

        // Unknown type with no default should return raw
        let raw = 0.7;
        let result = cal.calibrate_typed(raw, "UNKNOWN");
        assert_eq!(result, raw);
    }

    // =========================================================================
    // Calibration Metrics
    // =========================================================================

    #[test]
    fn test_calibration_metrics_perfect() {
        // Perfectly calibrated: confidence equals empirical accuracy
        let perfect = vec![(0.5, true), (0.5, false), (0.5, true), (0.5, false)];
        let metrics = CalibrationMetrics::compute(&perfect, 10);

        assert!(
            metrics.ece < 0.01,
            "Perfect calibration should have ECE ~0, got {}",
            metrics.ece
        );
        assert!(
            metrics.mce < 0.01,
            "Perfect calibration should have MCE ~0, got {}",
            metrics.mce
        );
        assert_eq!(metrics.n_samples, 4);
    }

    #[test]
    fn test_calibration_metrics_overconfident() {
        // Overconfident: high confidence, low accuracy
        let overconfident = vec![(0.9, false), (0.9, false), (0.9, false), (0.9, true)];
        let metrics = CalibrationMetrics::compute(&overconfident, 10);

        // ECE should reflect the gap between confidence (0.9) and accuracy (0.25)
        assert!(
            metrics.ece > 0.5,
            "Overconfident should have high ECE, got {}",
            metrics.ece
        );
    }

    #[test]
    fn test_calibration_metrics_empty() {
        let metrics = CalibrationMetrics::compute(&[], 10);
        assert_eq!(metrics.n_samples, 0);
        assert_eq!(metrics.ece, 0.0);
        assert_eq!(metrics.brier_score, 0.0);
    }

    #[test]
    fn test_brier_score_bounds() {
        // All correct with high confidence -> low Brier
        let good = vec![(0.9, true), (0.9, true), (0.9, true)];
        let good_metrics = CalibrationMetrics::compute(&good, 10);

        // All wrong with high confidence -> high Brier
        let bad = vec![(0.9, false), (0.9, false), (0.9, false)];
        let bad_metrics = CalibrationMetrics::compute(&bad, 10);

        assert!(good_metrics.brier_score < bad_metrics.brier_score);
        assert!(good_metrics.brier_score < 0.1);
        assert!(bad_metrics.brier_score > 0.5);
    }

    // =========================================================================
    // Serialization
    // =========================================================================

    #[test]
    fn test_calibrator_serialization() {
        let cal = TemperatureCalibrator::new(1.5);
        let json = serde_json::to_string(&cal).expect("serialize TemperatureCalibrator");
        let recovered: TemperatureCalibrator =
            serde_json::from_str(&json).expect("deserialize TemperatureCalibrator");
        assert!((cal.temperature - recovered.temperature).abs() < 0.001);
    }

    #[test]
    fn test_isotonic_serialization() {
        let data = vec![(0.3, false), (0.5, true), (0.7, true)];
        let cal = IsotonicCalibrator::fit(&data);
        let json = serde_json::to_string(&cal).expect("serialize IsotonicCalibrator");
        let recovered: IsotonicCalibrator =
            serde_json::from_str(&json).expect("deserialize IsotonicCalibrator");

        // Should produce same calibration
        assert!((cal.calibrate(0.5) - recovered.calibrate(0.5)).abs() < 0.01);
    }
}
