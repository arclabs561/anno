//! NER evaluation trait and implementations.
//!
//! Provides trait-based evaluation matching the RetrievalEvaluator pattern
//! for consistency and extensibility.

use super::datasets::GoldEntity;
use super::types::{GoalCheckResult, MetricValue};
use super::TypeMetrics;
use crate::{Error, Model, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Per-test-case NER evaluation metrics.
///
/// Type-safe metrics using `MetricValue` for compile-time guarantees.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NERQueryMetrics {
    /// Test case text
    pub text: String,
    /// Optional test case ID
    pub test_case_id: Option<String>,
    /// Precision (type-safe, bounded 0.0-1.0)
    pub precision: MetricValue,
    /// Recall (type-safe, bounded 0.0-1.0)
    pub recall: MetricValue,
    /// F1 score (type-safe, bounded 0.0-1.0)
    pub f1: MetricValue,
    /// Per-entity-type metrics
    pub per_type: HashMap<String, TypeMetrics>,
    /// Number of entities found
    pub found: usize,
    /// Number of entities expected
    pub expected: usize,
    /// Number of correct predictions
    pub correct: usize,
    /// Processing speed (tokens per second)
    pub tokens_per_second: f64,
}

/// Aggregated NER evaluation metrics with statistical measures.
///
/// Provides mean, standard deviation, and confidence intervals
/// for comprehensive analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NERAggregateMetrics {
    /// Mean precision (type-safe, bounded 0.0-1.0)
    pub precision: MetricValue,
    /// Mean recall (type-safe, bounded 0.0-1.0)
    pub recall: MetricValue,
    /// Mean F1 score (type-safe, bounded 0.0-1.0)
    pub f1: MetricValue,
    /// Precision standard deviation
    pub precision_std: f64,
    /// Recall standard deviation
    pub recall_std: f64,
    /// F1 standard deviation
    pub f1_std: f64,
    /// Precision 95% confidence interval (lower, upper)
    pub precision_ci_95: Option<(f64, f64)>,
    /// Recall 95% confidence interval (lower, upper)
    pub recall_ci_95: Option<(f64, f64)>,
    /// F1 95% confidence interval (lower, upper)
    pub f1_ci_95: Option<(f64, f64)>,
    /// Per-entity-type aggregated metrics
    pub per_type: HashMap<String, TypeMetrics>,
    /// Mean tokens per second
    pub tokens_per_second: f64,
    /// Number of test cases evaluated
    pub num_test_cases: usize,
    /// Total entities found
    pub total_found: usize,
    /// Total entities expected
    pub total_expected: usize,
    /// Total correct predictions
    pub total_correct: usize,
}

/// Type-safe NER evaluation goals.
///
/// Allows setting minimum thresholds for metrics with compile-time guarantees.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NERMetricGoals {
    /// Minimum precision threshold
    pub min_precision: Option<MetricValue>,
    /// Minimum recall threshold
    pub min_recall: Option<MetricValue>,
    /// Minimum F1 threshold
    pub min_f1: Option<MetricValue>,
    /// Per-entity-type goals
    pub per_type_goals: HashMap<String, TypeMetricGoals>,
}

impl NERMetricGoals {
    /// Create new empty goals.
    pub fn new() -> Self {
        Self {
            min_precision: None,
            min_recall: None,
            min_f1: None,
            per_type_goals: HashMap::new(),
        }
    }

    /// Set minimum precision goal.
    pub fn with_min_precision(mut self, value: f64) -> Result<Self> {
        self.min_precision = Some(MetricValue::try_new(value)?);
        Ok(self)
    }

    /// Set minimum recall goal.
    pub fn with_min_recall(mut self, value: f64) -> Result<Self> {
        self.min_recall = Some(MetricValue::try_new(value)?);
        Ok(self)
    }

    /// Set minimum F1 goal.
    pub fn with_min_f1(mut self, value: f64) -> Result<Self> {
        self.min_f1 = Some(MetricValue::try_new(value)?);
        Ok(self)
    }

    /// Add per-type goal.
    pub fn with_type_goal(mut self, entity_type: String, goal: TypeMetricGoals) -> Self {
        self.per_type_goals.insert(entity_type, goal);
        self
    }
}

impl Default for NERMetricGoals {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-entity-type metric goals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeMetricGoals {
    /// Minimum precision for this entity type
    pub min_precision: Option<MetricValue>,
    /// Minimum recall for this entity type
    pub min_recall: Option<MetricValue>,
    /// Minimum F1 for this entity type
    pub min_f1: Option<MetricValue>,
}

impl TypeMetricGoals {
    /// Create new type goals.
    pub fn new() -> Self {
        Self {
            min_precision: None,
            min_recall: None,
            min_f1: None,
        }
    }

    /// Set minimum precision.
    pub fn with_min_precision(mut self, value: f64) -> Result<Self> {
        self.min_precision = Some(MetricValue::try_new(value)?);
        Ok(self)
    }

    /// Set minimum recall.
    pub fn with_min_recall(mut self, value: f64) -> Result<Self> {
        self.min_recall = Some(MetricValue::try_new(value)?);
        Ok(self)
    }

    /// Set minimum F1.
    pub fn with_min_f1(mut self, value: f64) -> Result<Self> {
        self.min_f1 = Some(MetricValue::try_new(value)?);
        Ok(self)
    }
}

impl Default for TypeMetricGoals {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for NER evaluation strategies.
///
/// Allows plugging in different evaluation implementations:
/// - Standard evaluator (exact match)
/// - Partial match evaluator (overlap-based)
/// - Custom evaluators (for research, special metrics)
///
/// # Example
///
/// ```rust
/// use anno::eval::{GoldEntity, StandardNEREvaluator, NEREvaluator};
/// use anno::{PatternNER, Model, EntityType};
///
/// let evaluator = StandardNEREvaluator::new();
/// let model = PatternNER::new();
/// let ground_truth = vec![
///     GoldEntity::new("$100", EntityType::Money, 6),
/// ];
///
/// let metrics = evaluator.evaluate_test_case(
///     &model,
///     "Cost: $100",
///     &ground_truth,
///     Some("test-1"),
/// ).unwrap();
///
/// assert!(metrics.precision.get() > 0.0);
/// ```
pub trait NEREvaluator: Send + Sync {
    /// Evaluate a single test case.
    ///
    /// # Arguments
    /// * `model` - NER model to evaluate
    /// * `text` - Text to extract entities from (must not be empty)
    /// * `ground_truth` - Expected entities
    /// * `test_case_id` - Optional test case identifier
    ///
    /// # Returns
    /// Per-test-case metrics with precision, recall, F1, and per-type breakdowns
    ///
    /// # Errors
    /// Returns `Error::InvalidInput` if:
    /// - Text is empty
    /// - Ground truth entities are invalid (overlapping, out of bounds)
    /// - Metrics are invalid (NaN or Inf)
    fn evaluate_test_case(
        &self,
        model: &dyn Model,
        text: &str,
        ground_truth: &[GoldEntity],
        test_case_id: Option<&str>,
    ) -> Result<NERQueryMetrics>;

    /// Aggregate metrics across multiple test cases.
    ///
    /// # Arguments
    /// * `query_metrics` - Per-test-case metrics
    ///
    /// # Returns
    /// Aggregate metrics with statistical measures
    fn aggregate(&self, query_metrics: &[NERQueryMetrics]) -> Result<NERAggregateMetrics>;

    /// Check if metrics meet goals.
    ///
    /// # Arguments
    /// * `metrics` - Aggregate metrics to check
    /// * `goals` - Goals to check against
    ///
    /// # Returns
    /// Goal check result with pass/fail status
    fn check_goals(
        &self,
        metrics: &NERAggregateMetrics,
        goals: &NERMetricGoals,
    ) -> Result<GoalCheckResult>;
}

/// Standard NER evaluator implementation.
///
/// Computes standard NER metrics: Precision, Recall, F1 (exact match).
pub struct StandardNEREvaluator;

impl StandardNEREvaluator {
    /// Create a new standard evaluator.
    pub fn new() -> Self {
        Self
    }
}

impl Default for StandardNEREvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl NEREvaluator for StandardNEREvaluator {
    fn evaluate_test_case(
        &self,
        model: &dyn Model,
        text: &str,
        ground_truth: &[GoldEntity],
        test_case_id: Option<&str>,
    ) -> Result<NERQueryMetrics> {
        // Validate input
        if text.is_empty() {
            return Err(Error::InvalidInput(
                "Text cannot be empty for NER evaluation".to_string(),
            ));
        }

        // Validate ground truth entities
        let validation = crate::eval::validation::validate_ground_truth_entities(
            text,
            ground_truth,
            false, // Warnings for overlaps, not errors
        );
        if !validation.is_valid {
            return Err(Error::InvalidInput(format!(
                "Invalid ground truth entities: {}",
                validation.errors.join("; ")
            )));
        }
        // Log warnings if any (using eprintln! for now, can be upgraded to proper logging)
        if !validation.warnings.is_empty() {
            eprintln!(
                "WARNING: Ground truth validation warnings: {}",
                validation.warnings.join("; ")
            );
        }

        let start_time = std::time::Instant::now();

        // Extract entities using model
        let predicted = model.extract_entities(text, None)?;

        let elapsed = start_time.elapsed().as_secs_f64();
        let tokens = text.split_whitespace().count();
        let tokens_per_second = if elapsed > 0.0 {
            tokens as f64 / elapsed
        } else {
            0.0
        };

        // Count correct predictions (exact match: same span and type)
        let mut correct = 0;
        for pred in &predicted {
            for gt in ground_truth {
                if pred.start == gt.start
                    && pred.end == gt.end
                    && super::entity_type_matches(&pred.entity_type, &gt.entity_type)
                {
                    correct += 1;
                    break;
                }
            }
        }

        // Calculate per-type statistics
        let mut per_type_stats: HashMap<String, (usize, usize, usize)> = HashMap::new(); // (found, expected, correct)

        // Count expected per type
        for gt in ground_truth {
            let type_key = super::entity_type_to_string(&gt.entity_type);
            let stats = per_type_stats.entry(type_key.clone()).or_insert((0, 0, 0));
            stats.1 += 1; // expected

            // Check if this ground truth entity was found
            for pred in &predicted {
                if pred.start == gt.start
                    && pred.end == gt.end
                    && super::entity_type_matches(&pred.entity_type, &gt.entity_type)
                {
                    stats.2 += 1; // correct
                    break;
                }
            }
        }

        // Count found per type
        for pred in &predicted {
            let type_key = super::entity_type_to_string(&pred.entity_type);
            let stats = per_type_stats.entry(type_key).or_insert((0, 0, 0));
            stats.0 += 1; // found
        }

        // Calculate overall metrics
        let found = predicted.len();
        let expected = ground_truth.len();

        let precision = if found > 0 {
            correct as f64 / found as f64
        } else {
            0.0
        };
        let recall = if expected > 0 {
            correct as f64 / expected as f64
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        // Validate metrics are finite (not NaN or Inf)
        if !precision.is_finite() || !recall.is_finite() || !f1.is_finite() {
            return Err(Error::InvalidInput(format!(
                "Invalid metric values: precision={}, recall={}, f1={}",
                precision, recall, f1
            )));
        }

        // Calculate per-type metrics
        let mut per_type = HashMap::new();
        for (type_name, (found_count, expected_count, correct_count)) in per_type_stats {
            let type_precision = if found_count > 0 {
                correct_count as f64 / found_count as f64
            } else {
                0.0
            };
            let type_recall = if expected_count > 0 {
                correct_count as f64 / expected_count as f64
            } else {
                0.0
            };
            let type_f1 = if type_precision + type_recall > 0.0 {
                2.0 * type_precision * type_recall / (type_precision + type_recall)
            } else {
                0.0
            };

            per_type.insert(
                type_name,
                TypeMetrics {
                    precision: type_precision,
                    recall: type_recall,
                    f1: type_f1,
                    found: found_count,
                    expected: expected_count,
                    correct: correct_count,
                },
            );
        }

        Ok(NERQueryMetrics {
            text: text.to_string(),
            test_case_id: test_case_id.map(|s| s.to_string()),
            precision: MetricValue::new(precision),
            recall: MetricValue::new(recall),
            f1: MetricValue::new(f1),
            per_type,
            found,
            expected,
            correct,
            tokens_per_second,
        })
    }

    fn aggregate(&self, query_metrics: &[NERQueryMetrics]) -> Result<NERAggregateMetrics> {
        if query_metrics.is_empty() {
            return Err(Error::InvalidInput(
                "Cannot aggregate empty metrics".to_string(),
            ));
        }

        // Extract metric values
        let precisions: Vec<f64> = query_metrics.iter().map(|m| m.precision.get()).collect();
        let recalls: Vec<f64> = query_metrics.iter().map(|m| m.recall.get()).collect();
        let f1s: Vec<f64> = query_metrics.iter().map(|m| m.f1.get()).collect();
        let tokens_per_second: Vec<f64> =
            query_metrics.iter().map(|m| m.tokens_per_second).collect();

        // Calculate means
        let mean_precision = precisions.iter().sum::<f64>() / precisions.len() as f64;
        let mean_recall = recalls.iter().sum::<f64>() / recalls.len() as f64;
        let mean_f1 = f1s.iter().sum::<f64>() / f1s.len() as f64;
        let mean_tokens_per_second =
            tokens_per_second.iter().sum::<f64>() / tokens_per_second.len() as f64;

        // Validate means are finite
        if !mean_precision.is_finite()
            || !mean_recall.is_finite()
            || !mean_f1.is_finite()
            || !mean_tokens_per_second.is_finite()
        {
            return Err(Error::InvalidInput(format!(
                "Invalid aggregate metric values: precision={}, recall={}, f1={}, tps={}",
                mean_precision, mean_recall, mean_f1, mean_tokens_per_second
            )));
        }

        // Calculate standard deviations
        let precision_std = calculate_std_dev(&precisions, mean_precision);
        let recall_std = calculate_std_dev(&recalls, mean_recall);
        let f1_std = calculate_std_dev(&f1s, mean_f1);

        // Calculate 95% confidence intervals
        let precision_ci_95 = calculate_ci_95(&precisions, mean_precision, precision_std);
        let recall_ci_95 = calculate_ci_95(&recalls, mean_recall, recall_std);
        let f1_ci_95 = calculate_ci_95(&f1s, mean_f1, f1_std);

        // Aggregate per-type metrics
        let mut per_type_aggregated: HashMap<String, Vec<TypeMetrics>> = HashMap::new();
        for metric in query_metrics {
            for (type_name, type_metric) in &metric.per_type {
                per_type_aggregated
                    .entry(type_name.clone())
                    .or_default()
                    .push(type_metric.clone());
            }
        }

        let mut per_type = HashMap::new();
        for (type_name, type_metrics_list) in per_type_aggregated {
            let type_precisions: Vec<f64> = type_metrics_list.iter().map(|m| m.precision).collect();
            let type_recalls: Vec<f64> = type_metrics_list.iter().map(|m| m.recall).collect();
            let type_f1s: Vec<f64> = type_metrics_list.iter().map(|m| m.f1).collect();

            let mean_type_precision =
                type_precisions.iter().sum::<f64>() / type_precisions.len() as f64;
            let mean_type_recall = type_recalls.iter().sum::<f64>() / type_recalls.len() as f64;
            let mean_type_f1 = type_f1s.iter().sum::<f64>() / type_f1s.len() as f64;

            let total_found: usize = type_metrics_list.iter().map(|m| m.found).sum();
            let total_expected: usize = type_metrics_list.iter().map(|m| m.expected).sum();
            let total_correct: usize = type_metrics_list.iter().map(|m| m.correct).sum();

            per_type.insert(
                type_name,
                TypeMetrics {
                    precision: mean_type_precision,
                    recall: mean_type_recall,
                    f1: mean_type_f1,
                    found: total_found,
                    expected: total_expected,
                    correct: total_correct,
                },
            );
        }

        // Calculate totals
        let total_found: usize = query_metrics.iter().map(|m| m.found).sum();
        let total_expected: usize = query_metrics.iter().map(|m| m.expected).sum();
        let total_correct: usize = query_metrics.iter().map(|m| m.correct).sum();

        Ok(NERAggregateMetrics {
            precision: MetricValue::new(mean_precision),
            recall: MetricValue::new(mean_recall),
            f1: MetricValue::new(mean_f1),
            precision_std,
            recall_std,
            f1_std,
            precision_ci_95,
            recall_ci_95,
            f1_ci_95,
            per_type,
            tokens_per_second: mean_tokens_per_second,
            num_test_cases: query_metrics.len(),
            total_found,
            total_expected,
            total_correct,
        })
    }

    fn check_goals(
        &self,
        metrics: &NERAggregateMetrics,
        goals: &NERMetricGoals,
    ) -> Result<GoalCheckResult> {
        let mut result = GoalCheckResult::new();

        // Check overall goals
        if let Some(min_precision) = goals.min_precision {
            let actual = metrics.precision.get();
            let goal = min_precision.get();
            if actual < goal {
                result.add_failure("precision".to_string(), actual, goal);
            }
        }

        if let Some(min_recall) = goals.min_recall {
            let actual = metrics.recall.get();
            let goal = min_recall.get();
            if actual < goal {
                result.add_failure("recall".to_string(), actual, goal);
            }
        }

        if let Some(min_f1) = goals.min_f1 {
            let actual = metrics.f1.get();
            let goal = min_f1.get();
            if actual < goal {
                result.add_failure("f1".to_string(), actual, goal);
            }
        }

        // Check per-type goals
        for (type_name, type_goals) in &goals.per_type_goals {
            if let Some(type_metrics) = metrics.per_type.get(type_name) {
                if let Some(min_precision) = type_goals.min_precision {
                    let actual = type_metrics.precision;
                    let goal = min_precision.get();
                    if actual < goal {
                        result.add_failure(format!("{}.precision", type_name), actual, goal);
                    }
                }

                if let Some(min_recall) = type_goals.min_recall {
                    let actual = type_metrics.recall;
                    let goal = min_recall.get();
                    if actual < goal {
                        result.add_failure(format!("{}.recall", type_name), actual, goal);
                    }
                }

                if let Some(min_f1) = type_goals.min_f1 {
                    let actual = type_metrics.f1;
                    let goal = min_f1.get();
                    if actual < goal {
                        result.add_failure(format!("{}.f1", type_name), actual, goal);
                    }
                }
            }
        }

        Ok(result)
    }
}

/// Calculate standard deviation.
fn calculate_std_dev(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let variance =
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

    variance.sqrt()
}

/// Calculate 95% confidence interval.
///
/// Uses t-distribution approximation (z-score for large samples).
///
/// # Note
/// Confidence intervals may extend beyond [0.0, 1.0] for small samples or high variance.
/// This is statistically valid and indicates uncertainty in the estimate.
/// For display purposes, you may want to clamp bounds to [0.0, 1.0], but the raw
/// intervals provide more accurate statistical information.
fn calculate_ci_95(values: &[f64], mean: f64, std_dev: f64) -> Option<(f64, f64)> {
    if values.len() < 2 {
        return None;
    }

    // Use z-score for 95% CI (1.96 for large samples)
    // For small samples, should use t-distribution, but z-score is acceptable approximation
    let z_score = 1.96;
    let margin = z_score * std_dev / (values.len() as f64).sqrt();

    // Clamp CI bounds to [0.0, 1.0] for metrics (precision, recall, F1)
    // Note: For very small samples, CI may extend beyond [0, 1], but we clamp
    // to maintain valid metric bounds. This is a reasonable approximation.
    let lower = (mean - margin).clamp(0.0, 1.0);
    let upper = (mean + margin).clamp(0.0, 1.0);

    Some((lower, upper))
}

// Tests moved to tests/ directory
