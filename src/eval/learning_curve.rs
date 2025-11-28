//! Learning curve analysis for NER evaluation.
//!
//! Tracks how model performance changes with training data size,
//! enabling data efficiency analysis and optimal data budgeting.
//!
//! # Key Insights
//!
//! - **Sample Efficiency**: How many examples needed for target F1?
//! - **Diminishing Returns**: Where does adding data stop helping?
//! - **Per-Entity Curves**: Which entity types need more data?
//!
//! # Example
//!
//! ```rust
//! use anno::eval::learning_curve::{LearningCurveAnalyzer, DataPoint};
//!
//! let points = vec![
//!     DataPoint { train_size: 100, f1: 0.65, precision: 0.70, recall: 0.60 },
//!     DataPoint { train_size: 500, f1: 0.80, precision: 0.82, recall: 0.78 },
//!     DataPoint { train_size: 1000, f1: 0.85, precision: 0.86, recall: 0.84 },
//! ];
//!
//! let analyzer = LearningCurveAnalyzer::new(points);
//! let analysis = analyzer.analyze();
//!
//! if let Some(samples) = analysis.samples_for_target(0.90) {
//!     println!("Estimated samples for 90% F1: {}", samples);
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Data Structures
// =============================================================================

/// A single point on the learning curve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    /// Number of training samples
    pub train_size: usize,
    /// F1 score at this training size
    pub f1: f64,
    /// Precision at this training size
    pub precision: f64,
    /// Recall at this training size
    pub recall: f64,
}

/// Learning curve analysis results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningCurveAnalysis {
    /// Raw data points
    pub data_points: Vec<DataPoint>,
    /// Per-entity-type learning curves (if available)
    pub per_entity_curves: HashMap<String, Vec<DataPoint>>,
    /// Sample efficiency metrics
    pub efficiency: SampleEfficiencyMetrics,
    /// Fitted curve parameters (power law: y = a * x^b + c)
    pub curve_fit: Option<CurveFitParams>,
    /// Recommendations based on curve shape
    pub recommendations: Vec<String>,
}

/// Sample efficiency metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleEfficiencyMetrics {
    /// F1 improvement per 100 samples (average)
    pub f1_per_100_samples: f64,
    /// Estimated samples needed for various F1 targets
    pub samples_for_targets: HashMap<String, Option<usize>>,
    /// Diminishing returns threshold (where adding data helps <1% F1)
    pub diminishing_returns_threshold: Option<usize>,
    /// Current saturation level (0-1, how close to plateau)
    pub saturation_level: f64,
}

/// Power law curve fit parameters: y = a * x^b + c
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurveFitParams {
    /// Scaling coefficient
    pub a: f64,
    /// Power exponent (controls curve shape)
    pub b: f64,
    /// Asymptotic offset (theoretical maximum)
    pub c: f64,
    /// R² goodness of fit
    pub r_squared: f64,
}

// =============================================================================
// Learning Curve Analyzer
// =============================================================================

/// Analyzer for learning curves.
#[derive(Debug, Clone)]
pub struct LearningCurveAnalyzer {
    data_points: Vec<DataPoint>,
    per_entity_curves: HashMap<String, Vec<DataPoint>>,
}

impl LearningCurveAnalyzer {
    /// Create analyzer from aggregate data points.
    pub fn new(data_points: Vec<DataPoint>) -> Self {
        Self {
            data_points,
            per_entity_curves: HashMap::new(),
        }
    }

    /// Add per-entity-type curve data.
    pub fn with_entity_curves(mut self, curves: HashMap<String, Vec<DataPoint>>) -> Self {
        self.per_entity_curves = curves;
        self
    }

    /// Perform learning curve analysis.
    pub fn analyze(&self) -> LearningCurveAnalysis {
        let efficiency = self.compute_efficiency();
        let curve_fit = self.fit_power_law();
        let recommendations = self.generate_recommendations(&efficiency, &curve_fit);

        LearningCurveAnalysis {
            data_points: self.data_points.clone(),
            per_entity_curves: self.per_entity_curves.clone(),
            efficiency,
            curve_fit,
            recommendations,
        }
    }

    fn compute_efficiency(&self) -> SampleEfficiencyMetrics {
        let mut sorted_points = self.data_points.clone();
        sorted_points.sort_by_key(|p| p.train_size);

        // Compute average F1 improvement per 100 samples
        let f1_per_100 = if sorted_points.len() < 2 {
            0.0
        } else {
            let first = &sorted_points[0];
            let last = &sorted_points[sorted_points.len() - 1];
            let f1_improvement = last.f1 - first.f1;
            let sample_diff = last.train_size - first.train_size;
            if sample_diff == 0 {
                0.0
            } else {
                (f1_improvement / sample_diff as f64) * 100.0
            }
        };

        // Estimate samples for various F1 targets
        let targets = vec![0.80, 0.85, 0.90, 0.95];
        let mut samples_for_targets = HashMap::new();

        for target in targets {
            let key = format!("{:.0}%", target * 100.0);
            samples_for_targets.insert(key, self.estimate_samples_for_f1(target));
        }

        // Find diminishing returns threshold
        let diminishing_threshold = self.find_diminishing_returns(&sorted_points);

        // Compute saturation level
        let saturation = self.compute_saturation(&sorted_points);

        SampleEfficiencyMetrics {
            f1_per_100_samples: f1_per_100,
            samples_for_targets,
            diminishing_returns_threshold: diminishing_threshold,
            saturation_level: saturation,
        }
    }

    fn estimate_samples_for_f1(&self, target_f1: f64) -> Option<usize> {
        let mut sorted = self.data_points.clone();
        sorted.sort_by_key(|p| p.train_size);

        // Check if we've already achieved this F1
        for point in &sorted {
            if point.f1 >= target_f1 {
                return Some(point.train_size);
            }
        }

        // Extrapolate using power law if we have enough points
        if sorted.len() >= 3 {
            if let Some(fit) = self.fit_power_law() {
                // Solve for x: target_f1 = a * x^b + c
                // x = ((target_f1 - c) / a)^(1/b)
                let diff = target_f1 - fit.c;
                if diff > 0.0 && fit.a > 0.0 && fit.b != 0.0 {
                    let x = (diff / fit.a).powf(1.0 / fit.b);
                    if x.is_finite() && x > 0.0 {
                        return Some(x as usize);
                    }
                }
            }
        }

        None
    }

    fn find_diminishing_returns(&self, sorted: &[DataPoint]) -> Option<usize> {
        if sorted.len() < 3 {
            return None;
        }

        // Find where F1 improvement drops below 1% per doubling of data
        for i in 1..sorted.len() {
            let prev = &sorted[i - 1];
            let curr = &sorted[i];

            let sample_ratio = curr.train_size as f64 / prev.train_size as f64;
            let f1_improvement = curr.f1 - prev.f1;

            // If doubling data gives < 1% F1 improvement, we hit diminishing returns
            if sample_ratio >= 1.5 && f1_improvement < 0.01 {
                return Some(prev.train_size);
            }
        }

        None
    }

    fn compute_saturation(&self, sorted: &[DataPoint]) -> f64 {
        if sorted.len() < 3 {
            return 0.0;
        }

        // Compare recent improvement rate to initial improvement rate
        let first_third_end = sorted.len() / 3;
        let last_third_start = sorted.len() * 2 / 3;

        if first_third_end == 0 || last_third_start >= sorted.len() {
            return 0.0;
        }

        let initial_improvement =
            sorted[first_third_end].f1 - sorted[0].f1;
        let recent_improvement =
            sorted[sorted.len() - 1].f1 - sorted[last_third_start].f1;

        if initial_improvement <= 0.0 {
            return 1.0; // Already saturated from start
        }

        // Saturation = 1 - (recent_rate / initial_rate)
        let saturation = 1.0 - (recent_improvement / initial_improvement).min(1.0);
        saturation.max(0.0).min(1.0)
    }

    fn fit_power_law(&self) -> Option<CurveFitParams> {
        if self.data_points.len() < 3 {
            return None;
        }

        // Simple power law fit: y = a * x^b + c
        // Using least squares on log-transformed data for a and b,
        // then estimate c from residuals

        let mut sorted = self.data_points.clone();
        sorted.sort_by_key(|p| p.train_size);

        // For simplicity, use a basic heuristic fit
        // In production, would use proper nonlinear regression

        let x_log: Vec<f64> = sorted.iter().map(|p| (p.train_size as f64).ln()).collect();
        let y: Vec<f64> = sorted.iter().map(|p| p.f1).collect();

        let n = x_log.len() as f64;
        let sum_x = x_log.iter().sum::<f64>();
        let sum_y = y.iter().sum::<f64>();
        let sum_xy: f64 = x_log.iter().zip(y.iter()).map(|(x, y)| x * y).sum();
        let sum_x2: f64 = x_log.iter().map(|x| x * x).sum();

        let denom = n * sum_x2 - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            return None;
        }

        let b = (n * sum_xy - sum_x * sum_y) / denom;
        let a_log = (sum_y - b * sum_x) / n;
        let a = a_log.exp();

        // Estimate c as the asymptote (use last point's F1 + small buffer)
        let c = sorted.last().map(|p| p.f1 * 1.05).unwrap_or(1.0).min(1.0);

        // Compute R²
        let y_mean = sum_y / n;
        let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
        let ss_res: f64 = sorted
            .iter()
            .map(|p| {
                let predicted = a * (p.train_size as f64).powf(b);
                (p.f1 - predicted).powi(2)
            })
            .sum();

        let r_squared = if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };

        Some(CurveFitParams {
            a,
            b,
            c,
            r_squared: r_squared.max(0.0),
        })
    }

    fn generate_recommendations(
        &self,
        efficiency: &SampleEfficiencyMetrics,
        _curve_fit: &Option<CurveFitParams>,
    ) -> Vec<String> {
        let mut recs = Vec::new();

        // Saturation-based recommendations
        if efficiency.saturation_level > 0.8 {
            recs.push(
                "Model appears saturated - consider architectural changes rather than more data"
                    .to_string(),
            );
        } else if efficiency.saturation_level > 0.5 {
            recs.push(
                "Approaching saturation - additional data will have diminishing returns"
                    .to_string(),
            );
        } else {
            recs.push(
                "Model not saturated - more training data likely to improve performance"
                    .to_string(),
            );
        }

        // Efficiency-based recommendations
        if efficiency.f1_per_100_samples < 0.001 {
            recs.push(
                "Very low data efficiency - check for data quality issues or model capacity"
                    .to_string(),
            );
        } else if efficiency.f1_per_100_samples > 0.05 {
            recs.push(
                "High data efficiency - model is learning effectively from limited data"
                    .to_string(),
            );
        }

        // Target-based recommendations
        if let Some(Some(samples_90)) = efficiency.samples_for_targets.get("90%") {
            recs.push(format!(
                "Estimated ~{} samples needed to reach 90% F1",
                samples_90
            ));
        }

        recs
    }
}

impl LearningCurveAnalysis {
    /// Estimate samples needed for a specific F1 target.
    pub fn samples_for_target(&self, target_f1: f64) -> Option<usize> {
        let key = format!("{:.0}%", target_f1 * 100.0);
        self.efficiency
            .samples_for_targets
            .get(&key)
            .and_then(|v| *v)
    }

    /// Check if more data would likely help.
    pub fn more_data_would_help(&self) -> bool {
        self.efficiency.saturation_level < 0.7
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Generate learning curve data by training at different data sizes.
///
/// This is a helper for setting up learning curve experiments.
pub fn suggested_train_sizes(max_size: usize) -> Vec<usize> {
    let mut sizes = Vec::new();

    // Exponential spacing: 10, 25, 50, 100, 250, 500, 1000, ...
    let mut size = 10;
    while size <= max_size {
        sizes.push(size);
        // Roughly double each time (with some intermediate points)
        size = (size as f64 * 2.5) as usize;
    }

    // Always include the max
    if sizes.last() != Some(&max_size) {
        sizes.push(max_size);
    }

    sizes
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_analysis() {
        let points = vec![
            DataPoint { train_size: 100, f1: 0.60, precision: 0.65, recall: 0.55 },
            DataPoint { train_size: 500, f1: 0.75, precision: 0.78, recall: 0.72 },
            DataPoint { train_size: 1000, f1: 0.82, precision: 0.84, recall: 0.80 },
            DataPoint { train_size: 2000, f1: 0.85, precision: 0.86, recall: 0.84 },
        ];

        let analyzer = LearningCurveAnalyzer::new(points);
        let analysis = analyzer.analyze();

        assert!(analysis.efficiency.f1_per_100_samples > 0.0);
        assert!(!analysis.recommendations.is_empty());
    }

    #[test]
    fn test_saturation_detection() {
        // Simulated saturated model - big gains early, tiny gains late
        let points = vec![
            DataPoint { train_size: 100, f1: 0.50, precision: 0.50, recall: 0.50 },
            DataPoint { train_size: 200, f1: 0.70, precision: 0.70, recall: 0.70 },
            DataPoint { train_size: 400, f1: 0.80, precision: 0.80, recall: 0.80 },
            DataPoint { train_size: 800, f1: 0.82, precision: 0.82, recall: 0.82 },
            DataPoint { train_size: 1600, f1: 0.83, precision: 0.83, recall: 0.83 },
            DataPoint { train_size: 3200, f1: 0.835, precision: 0.835, recall: 0.835 },
        ];

        let analyzer = LearningCurveAnalyzer::new(points);
        let analysis = analyzer.analyze();

        // Should detect high saturation
        assert!(analysis.efficiency.saturation_level > 0.5);
    }

    #[test]
    fn test_suggested_train_sizes() {
        let sizes = suggested_train_sizes(10000);

        assert!(!sizes.is_empty());
        assert_eq!(*sizes.first().unwrap(), 10);
        assert_eq!(*sizes.last().unwrap(), 10000);

        // Should be roughly exponentially spaced
        for i in 1..sizes.len() {
            assert!(sizes[i] > sizes[i - 1]);
        }
    }

    #[test]
    fn test_more_data_would_help() {
        // Unsaturated model - consistent improvement rate throughout
        // (first third and last third have similar improvement rates)
        let unsaturated = vec![
            DataPoint { train_size: 100, f1: 0.40, precision: 0.40, recall: 0.40 },
            DataPoint { train_size: 200, f1: 0.48, precision: 0.48, recall: 0.48 },
            DataPoint { train_size: 400, f1: 0.56, precision: 0.56, recall: 0.56 },
            DataPoint { train_size: 800, f1: 0.64, precision: 0.64, recall: 0.64 },
            DataPoint { train_size: 1600, f1: 0.72, precision: 0.72, recall: 0.72 },
            DataPoint { train_size: 3200, f1: 0.80, precision: 0.80, recall: 0.80 },
        ];

        let analyzer = LearningCurveAnalyzer::new(unsaturated);
        let analysis = analyzer.analyze();

        // Linear improvement = low saturation
        assert!(analysis.efficiency.saturation_level < 0.5,
            "Saturation level {:.2} should be < 0.5 for linearly improving model",
            analysis.efficiency.saturation_level);
        assert!(analysis.more_data_would_help());
    }

    #[test]
    fn test_empty_data() {
        let analyzer = LearningCurveAnalyzer::new(vec![]);
        let analysis = analyzer.analyze();

        assert_eq!(analysis.efficiency.f1_per_100_samples, 0.0);
        assert!(analysis.curve_fit.is_none());
    }
}

