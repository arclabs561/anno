//! Temporal drift detection for production NER monitoring.
//!
//! Detects when model performance degrades over time due to:
//! - Distribution shift in input text
//! - New entity patterns not in training
//! - Concept drift (entity meanings change)
//!
//! # Key Metrics
//!
//! - **Performance Drift**: F1 change over time windows
//! - **Vocabulary Drift**: New tokens appearing in production
//! - **Confidence Drift**: Model confidence distribution changes
//! - **Entity Distribution Drift**: Entity type frequencies changing
//!
//! # Example
//!
//! ```rust
//! use anno::eval::drift::{DriftDetector, DriftWindow, DriftConfig};
//!
//! let mut detector = DriftDetector::new(DriftConfig::default());
//!
//! // Log predictions over time
//! detector.log_prediction(1000, 0.85, "PER", "John Smith");
//! detector.log_prediction(1001, 0.90, "ORG", "Google");
//!
//! // Check for drift
//! let report = detector.analyze();
//! if report.drift_detected {
//!     println!("Warning: Drift detected! {}", report.summary);
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for drift detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftConfig {
    /// Window size for computing rolling metrics (in number of predictions)
    pub window_size: usize,
    /// Number of windows to compare (current vs baseline)
    pub num_windows: usize,
    /// Threshold for confidence drift (change in mean confidence)
    pub confidence_drift_threshold: f64,
    /// Threshold for distribution drift (KL divergence)
    pub distribution_drift_threshold: f64,
    /// Threshold for vocabulary drift (proportion of new tokens)
    pub vocab_drift_threshold: f64,
    /// Minimum samples before drift detection activates
    pub min_samples: usize,
}

impl Default for DriftConfig {
    fn default() -> Self {
        Self {
            window_size: 1000,
            num_windows: 5,
            confidence_drift_threshold: 0.1,
            distribution_drift_threshold: 0.5,
            vocab_drift_threshold: 0.2,
            min_samples: 500,
        }
    }
}

// =============================================================================
// Data Structures
// =============================================================================

/// A single logged prediction.
#[derive(Debug, Clone)]
struct PredictionLog {
    /// Timestamp for future time-based windowing
    #[allow(dead_code)]
    timestamp: u64,
    confidence: f64,
    entity_type: String,
    entity_text: String,
}

/// Metrics for a single time window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftWindow {
    /// Window identifier (timestamp or index)
    pub window_id: usize,
    /// Mean confidence in this window
    pub mean_confidence: f64,
    /// Standard deviation of confidence
    pub std_confidence: f64,
    /// Entity type distribution
    pub type_distribution: HashMap<String, f64>,
    /// Number of predictions in window
    pub count: usize,
    /// Unique tokens seen
    pub unique_tokens: usize,
}

/// Drift analysis report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftReport {
    /// Whether significant drift was detected
    pub drift_detected: bool,
    /// Summary message
    pub summary: String,
    /// Confidence drift details
    pub confidence_drift: ConfidenceDrift,
    /// Distribution drift details
    pub distribution_drift: DistributionDrift,
    /// Vocabulary drift details
    pub vocabulary_drift: VocabularyDrift,
    /// Window-by-window metrics
    pub windows: Vec<DriftWindow>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Confidence drift analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceDrift {
    /// Baseline mean confidence
    pub baseline_mean: f64,
    /// Current mean confidence
    pub current_mean: f64,
    /// Change in mean confidence
    pub drift_amount: f64,
    /// Is drift significant?
    pub is_significant: bool,
}

/// Distribution drift analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionDrift {
    /// KL divergence from baseline to current
    pub kl_divergence: f64,
    /// Types that increased in frequency
    pub increased_types: Vec<(String, f64)>,
    /// Types that decreased in frequency
    pub decreased_types: Vec<(String, f64)>,
    /// New types not in baseline
    pub new_types: Vec<String>,
    /// Is drift significant?
    pub is_significant: bool,
}

/// Vocabulary drift analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabularyDrift {
    /// Baseline vocabulary size
    pub baseline_vocab_size: usize,
    /// Current vocabulary size
    pub current_vocab_size: usize,
    /// Proportion of new tokens
    pub new_token_rate: f64,
    /// Is drift significant?
    pub is_significant: bool,
}

// =============================================================================
// Drift Detector
// =============================================================================

/// Detector for temporal drift in NER predictions.
#[derive(Debug, Clone)]
pub struct DriftDetector {
    /// Configuration
    config: DriftConfig,
    /// Logged predictions (ring buffer)
    predictions: VecDeque<PredictionLog>,
    /// Baseline vocabulary
    baseline_vocab: HashMap<String, usize>,
    /// Current vocabulary
    current_vocab: HashMap<String, usize>,
    /// Whether baseline has been established
    baseline_established: bool,
}

impl DriftDetector {
    /// Create a new drift detector.
    pub fn new(config: DriftConfig) -> Self {
        let max_size = config.window_size * config.num_windows;
        Self {
            config,
            predictions: VecDeque::with_capacity(max_size),
            baseline_vocab: HashMap::new(),
            current_vocab: HashMap::new(),
            baseline_established: false,
        }
    }

    /// Log a prediction for drift analysis.
    pub fn log_prediction(
        &mut self,
        timestamp: u64,
        confidence: f64,
        entity_type: &str,
        entity_text: &str,
    ) {
        let log = PredictionLog {
            timestamp,
            confidence,
            entity_type: entity_type.to_string(),
            entity_text: entity_text.to_string(),
        };

        // Track vocabulary
        for token in entity_text.split_whitespace() {
            let lower = token.to_lowercase();
            *self.current_vocab.entry(lower).or_insert(0) += 1;
        }

        // Add to ring buffer
        let max_size = self.config.window_size * self.config.num_windows;
        if self.predictions.len() >= max_size {
            self.predictions.pop_front();
        }
        self.predictions.push_back(log);

        // Establish baseline after minimum samples
        if !self.baseline_established && self.predictions.len() >= self.config.min_samples {
            self.establish_baseline();
        }
    }

    /// Establish baseline from current predictions.
    fn establish_baseline(&mut self) {
        self.baseline_vocab = self.current_vocab.clone();
        self.baseline_established = true;
    }

    /// Reset the detector.
    pub fn reset(&mut self) {
        self.predictions.clear();
        self.baseline_vocab.clear();
        self.current_vocab.clear();
        self.baseline_established = false;
    }

    /// Analyze for drift.
    pub fn analyze(&self) -> DriftReport {
        if self.predictions.len() < self.config.min_samples {
            return DriftReport {
                drift_detected: false,
                summary: format!(
                    "Insufficient data: {} predictions (need {})",
                    self.predictions.len(),
                    self.config.min_samples
                ),
                confidence_drift: ConfidenceDrift {
                    baseline_mean: 0.0,
                    current_mean: 0.0,
                    drift_amount: 0.0,
                    is_significant: false,
                },
                distribution_drift: DistributionDrift {
                    kl_divergence: 0.0,
                    increased_types: Vec::new(),
                    decreased_types: Vec::new(),
                    new_types: Vec::new(),
                    is_significant: false,
                },
                vocabulary_drift: VocabularyDrift {
                    baseline_vocab_size: 0,
                    current_vocab_size: 0,
                    new_token_rate: 0.0,
                    is_significant: false,
                },
                windows: Vec::new(),
                recommendations: vec!["Collect more data for drift analysis".into()],
            };
        }

        // Split predictions into windows
        let windows = self.compute_windows();

        // Analyze confidence drift
        let confidence_drift = self.analyze_confidence_drift(&windows);

        // Analyze distribution drift
        let distribution_drift = self.analyze_distribution_drift(&windows);

        // Analyze vocabulary drift
        let vocabulary_drift = self.analyze_vocabulary_drift();

        // Determine if drift detected
        let drift_detected = confidence_drift.is_significant
            || distribution_drift.is_significant
            || vocabulary_drift.is_significant;

        // Generate summary and recommendations
        let (summary, recommendations) = self.generate_summary_and_recommendations(
            drift_detected,
            &confidence_drift,
            &distribution_drift,
            &vocabulary_drift,
        );

        DriftReport {
            drift_detected,
            summary,
            confidence_drift,
            distribution_drift,
            vocabulary_drift,
            windows,
            recommendations,
        }
    }

    fn compute_windows(&self) -> Vec<DriftWindow> {
        let predictions: Vec<_> = self.predictions.iter().collect();
        let window_size = self.config.window_size.min(predictions.len());

        if window_size == 0 {
            return Vec::new();
        }

        let num_windows = (predictions.len() / window_size).min(self.config.num_windows);
        let mut windows = Vec::new();

        for i in 0..num_windows {
            let start = predictions.len() - (num_windows - i) * window_size;
            let end = start + window_size;
            let window_preds = &predictions[start..end];

            // Compute metrics
            let confidences: Vec<f64> = window_preds.iter().map(|p| p.confidence).collect();
            let mean_conf = confidences.iter().sum::<f64>() / confidences.len() as f64;
            let std_conf = (confidences
                .iter()
                .map(|c| (c - mean_conf).powi(2))
                .sum::<f64>()
                / confidences.len() as f64)
                .sqrt();

            // Type distribution
            let mut type_counts: HashMap<String, usize> = HashMap::new();
            let mut unique_tokens = std::collections::HashSet::new();

            for pred in window_preds {
                *type_counts.entry(pred.entity_type.clone()).or_insert(0) += 1;
                for token in pred.entity_text.split_whitespace() {
                    unique_tokens.insert(token.to_lowercase());
                }
            }

            let total = window_preds.len() as f64;
            let type_distribution: HashMap<String, f64> = type_counts
                .iter()
                .map(|(t, c)| (t.clone(), *c as f64 / total))
                .collect();

            windows.push(DriftWindow {
                window_id: i,
                mean_confidence: mean_conf,
                std_confidence: std_conf,
                type_distribution,
                count: window_preds.len(),
                unique_tokens: unique_tokens.len(),
            });
        }

        windows
    }

    fn analyze_confidence_drift(&self, windows: &[DriftWindow]) -> ConfidenceDrift {
        if windows.len() < 2 {
            return ConfidenceDrift {
                baseline_mean: 0.0,
                current_mean: 0.0,
                drift_amount: 0.0,
                is_significant: false,
            };
        }

        let baseline_mean = windows[0].mean_confidence;
        let current_mean = windows.last().map(|w| w.mean_confidence).unwrap_or(0.0);
        let drift_amount = current_mean - baseline_mean;
        let is_significant = drift_amount.abs() > self.config.confidence_drift_threshold;

        ConfidenceDrift {
            baseline_mean,
            current_mean,
            drift_amount,
            is_significant,
        }
    }

    fn analyze_distribution_drift(&self, windows: &[DriftWindow]) -> DistributionDrift {
        if windows.len() < 2 {
            return DistributionDrift {
                kl_divergence: 0.0,
                increased_types: Vec::new(),
                decreased_types: Vec::new(),
                new_types: Vec::new(),
                is_significant: false,
            };
        }

        let baseline = &windows[0].type_distribution;
        // Safety: we checked windows.len() >= 2 above
        let current = &windows[windows.len() - 1].type_distribution;

        // Compute KL divergence
        let epsilon = 1e-10;
        let mut kl_div = 0.0;

        for (typ, p) in current {
            let q = baseline.get(typ).copied().unwrap_or(epsilon);
            kl_div += p * (p / q).ln();
        }

        // Find changed types
        let mut increased = Vec::new();
        let mut decreased = Vec::new();
        let mut new_types = Vec::new();

        for (typ, curr_freq) in current {
            if let Some(base_freq) = baseline.get(typ) {
                let change = curr_freq - base_freq;
                if change > 0.05 {
                    increased.push((typ.clone(), change));
                } else if change < -0.05 {
                    decreased.push((typ.clone(), change));
                }
            } else {
                new_types.push(typ.clone());
            }
        }

        increased.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        decreased.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let is_significant = kl_div > self.config.distribution_drift_threshold || !new_types.is_empty();

        DistributionDrift {
            kl_divergence: kl_div,
            increased_types: increased,
            decreased_types: decreased,
            new_types,
            is_significant,
        }
    }

    fn analyze_vocabulary_drift(&self) -> VocabularyDrift {
        if !self.baseline_established {
            return VocabularyDrift {
                baseline_vocab_size: 0,
                current_vocab_size: self.current_vocab.len(),
                new_token_rate: 0.0,
                is_significant: false,
            };
        }

        let baseline_size = self.baseline_vocab.len();
        let current_size = self.current_vocab.len();

        let new_tokens: usize = self
            .current_vocab
            .keys()
            .filter(|t| !self.baseline_vocab.contains_key(*t))
            .count();

        let new_token_rate = if current_size == 0 {
            0.0
        } else {
            new_tokens as f64 / current_size as f64
        };

        let is_significant = new_token_rate > self.config.vocab_drift_threshold;

        VocabularyDrift {
            baseline_vocab_size: baseline_size,
            current_vocab_size: current_size,
            new_token_rate,
            is_significant,
        }
    }

    fn generate_summary_and_recommendations(
        &self,
        drift_detected: bool,
        confidence: &ConfidenceDrift,
        distribution: &DistributionDrift,
        vocabulary: &VocabularyDrift,
    ) -> (String, Vec<String>) {
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();

        if confidence.is_significant {
            if confidence.drift_amount < 0.0 {
                issues.push(format!(
                    "Confidence dropped by {:.1}%",
                    confidence.drift_amount.abs() * 100.0
                ));
                recommendations.push("Model may be encountering harder examples - consider retraining".into());
            } else {
                issues.push(format!(
                    "Confidence increased by {:.1}%",
                    confidence.drift_amount * 100.0
                ));
                recommendations.push("Verify model isn't becoming overconfident on new patterns".into());
            }
        }

        if distribution.is_significant {
            issues.push(format!(
                "Entity type distribution shifted (KL={:.2})",
                distribution.kl_divergence
            ));
            if !distribution.new_types.is_empty() {
                recommendations.push(format!(
                    "New entity types detected: {:?} - update training data",
                    distribution.new_types
                ));
            }
        }

        if vocabulary.is_significant {
            issues.push(format!(
                "{:.1}% new vocabulary",
                vocabulary.new_token_rate * 100.0
            ));
            recommendations.push("Significant vocabulary shift - consider domain adaptation".into());
        }

        let summary = if drift_detected {
            format!("Drift detected: {}", issues.join("; "))
        } else {
            "No significant drift detected".into()
        };

        if recommendations.is_empty() {
            recommendations.push("Continue monitoring".into());
        }

        (summary, recommendations)
    }
}

impl Default for DriftDetector {
    fn default() -> Self {
        Self::new(DriftConfig::default())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insufficient_data() {
        let detector = DriftDetector::default();
        let report = detector.analyze();

        assert!(!report.drift_detected);
        assert!(report.summary.contains("Insufficient"));
    }

    #[test]
    fn test_no_drift() {
        let mut detector = DriftDetector::new(DriftConfig {
            min_samples: 10,
            window_size: 5,
            num_windows: 2,
            ..Default::default()
        });

        // Log consistent predictions
        for i in 0..20 {
            detector.log_prediction(i as u64, 0.90, "PER", "John Smith");
        }

        let report = detector.analyze();
        // With consistent data, drift should be minimal
        assert!(!report.confidence_drift.is_significant);
    }

    #[test]
    fn test_confidence_drift_detection() {
        let mut detector = DriftDetector::new(DriftConfig {
            min_samples: 10,
            window_size: 10,
            num_windows: 2,
            confidence_drift_threshold: 0.1,
            ..Default::default()
        });

        // First window: high confidence
        for i in 0..10 {
            detector.log_prediction(i as u64, 0.95, "PER", "John");
        }

        // Second window: low confidence
        for i in 10..20 {
            detector.log_prediction(i as u64, 0.60, "PER", "John");
        }

        let report = detector.analyze();
        assert!(report.confidence_drift.is_significant);
        assert!(report.confidence_drift.drift_amount < 0.0);
    }

    #[test]
    fn test_vocabulary_drift() {
        let mut detector = DriftDetector::new(DriftConfig {
            min_samples: 5,
            window_size: 5,
            num_windows: 2,
            vocab_drift_threshold: 0.3,
            ..Default::default()
        });

        // Establish baseline with common words
        for i in 0..5 {
            detector.log_prediction(i as u64, 0.90, "PER", "John Smith");
        }

        // Add predictions with completely new vocabulary
        for i in 5..10 {
            detector.log_prediction(i as u64, 0.90, "PER", "Xiangjun Chen Zhang Wei");
        }

        let report = detector.analyze();
        assert!(report.vocabulary_drift.new_token_rate > 0.0);
    }

    #[test]
    fn test_reset() {
        let mut detector = DriftDetector::default();
        detector.log_prediction(0, 0.9, "PER", "Test");
        detector.reset();

        let report = detector.analyze();
        assert!(report.summary.contains("Insufficient"));
    }
}

