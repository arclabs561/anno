//! Unified evaluation report.
//!
//! Provides a single cohesive report structure that aggregates results from
//! all evaluation modules. This is the primary output type users should work with.
//!
//! # Example
//!
//! ```rust
//! use anno_eval::eval::report::{EvalReport, ReportBuilder};
//! use anno::RegexNER;
//!
//! let model = RegexNER::new();
//! let report = ReportBuilder::new("RegexNER")
//!     .with_core_metrics(true)
//!     .with_bias_analysis(false)  // Skip if no PER/ORG support
//!     .with_error_analysis(true)
//!     .build(&model)?;
//!
//! println!("{}", report.summary());
//! # Ok::<(), anno::Error>(())
//! ```

use anno::{Entity, Model, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// =============================================================================
// Core Report Structure
// =============================================================================

/// Unified evaluation report aggregating all analysis results.
///
/// Instead of 16 different Results structs, this provides one cohesive view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    /// Model/system identifier
    pub model_name: String,

    /// Timestamp of evaluation
    pub timestamp: String,

    /// Core NER metrics (always present)
    pub core: CoreMetrics,

    /// Per-entity-type breakdown
    pub per_type: HashMap<String, TypeMetrics>,

    /// Error analysis (if enabled)
    pub errors: Option<ErrorSummary>,

    /// Demographic NER bias analysis (if enabled)
    pub bias: Option<BiasSummary>,

    /// Data quality findings (if enabled)
    pub data_quality: Option<DataQualitySummary>,

    /// Calibration analysis (if model provides confidence)
    pub calibration: Option<CalibrationSummary>,

    /// Recommendations based on findings
    pub recommendations: Vec<Recommendation>,

    /// Raw warnings/notes generated during evaluation
    pub warnings: Vec<String>,
}

/// Core precision/recall/F1 metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMetrics {
    /// Micro-averaged precision (total_correct / total_predicted)
    pub precision: f64,
    /// Micro-averaged recall (total_correct / total_gold)
    pub recall: f64,
    /// Micro-averaged F1
    pub f1: f64,
    /// Total entities in gold standard
    pub total_gold: usize,
    /// Total entities predicted
    pub total_predicted: usize,
    /// Total correct predictions
    pub total_correct: usize,
    /// Macro-averaged F1 (for comparison)
    pub macro_f1: Option<f64>,
}

/// Per-type metrics breakdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeMetrics {
    /// Precision for this type
    pub precision: f64,
    /// Recall for this type
    pub recall: f64,
    /// F1 for this type
    pub f1: f64,
    /// Number of gold entities of this type
    pub support: usize,
    /// Number of predicted entities of this type
    pub predicted: usize,
    /// Number of correctly predicted entities
    pub correct: usize,
}

/// Error analysis summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorSummary {
    /// Total errors
    pub total_errors: usize,
    /// Boundary errors (correct type, wrong span)
    pub boundary_errors: usize,
    /// Type errors (correct span, wrong type)
    pub type_errors: usize,
    /// False positives (spurious entities)
    pub false_positives: usize,
    /// False negatives (missed entities)
    pub false_negatives: usize,
    /// Most common error patterns
    pub top_patterns: Vec<String>,
}

/// Bias analysis summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiasSummary {
    /// Whether significant bias was detected
    pub bias_detected: bool,
    /// Gender bias metrics (if applicable)
    pub gender: Option<GenderBiasMetrics>,
    /// Demographic bias metrics (if applicable)
    pub demographic: Option<DemographicBiasMetrics>,
    /// Length bias (short vs long entities)
    pub length: Option<LengthBiasMetrics>,
}

/// Gender bias metrics from WinoBias-style evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenderBiasMetrics {
    /// Accuracy on pro-stereotypical examples
    pub pro_stereotype_accuracy: f64,
    /// Accuracy on anti-stereotypical examples
    pub anti_stereotype_accuracy: f64,
    /// Gap between pro and anti (lower is better)
    pub gap: f64,
    /// Human-readable verdict
    pub verdict: String,
}

/// Demographic bias metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemographicBiasMetrics {
    /// Performance gap between best and worst demographic groups
    pub max_gap: f64,
    /// Groups with notably lower performance
    pub underperforming_groups: Vec<String>,
}

/// Entity length bias metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LengthBiasMetrics {
    /// F1 on short entities (1-2 tokens)
    pub short_entity_f1: f64,
    /// F1 on long entities (4+ tokens)
    pub long_entity_f1: f64,
    /// Gap between short and long (lower is better)
    pub gap: f64,
}

/// Data quality findings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualitySummary {
    /// Potential train/test leakage detected
    pub leakage_detected: bool,
    /// Percentage of redundant examples
    pub redundancy_rate: f64,
    /// Ambiguous entity annotations found
    pub ambiguous_count: usize,
}

/// Calibration analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationSummary {
    /// Expected Calibration Error
    pub ece: f64,
    /// Maximum Calibration Error
    pub mce: f64,
    /// Optimal confidence threshold
    pub optimal_threshold: f64,
    /// Grade (A/B/C/D/F)
    pub grade: char,
}

/// Actionable recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Priority: high, medium, low
    pub priority: Priority,
    /// Category of recommendation
    pub category: RecommendationCategory,
    /// Human-readable recommendation
    pub message: String,
    /// Estimated impact if addressed
    pub estimated_impact: Option<String>,
}

/// Priority level for recommendations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Priority {
    /// Urgent - blocks deployment
    High,
    /// Important - should be addressed
    Medium,
    /// Nice to have
    Low,
}

/// Category of recommendation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationCategory {
    /// Core performance issues (F1/P/R)
    Performance,
    /// Bias-related issues
    Bias,
    /// Dataset quality issues
    DataQuality,
    /// Confidence calibration issues
    Calibration,
    /// Entity type coverage issues
    Coverage,
}

// =============================================================================
// Report Builder
// =============================================================================

/// Builder for constructing evaluation reports.
pub struct ReportBuilder {
    model_name: String,
    include_core: bool,
    include_errors: bool,
    include_bias: bool,
    include_data_quality: bool,
    include_calibration: bool,
    test_data: Option<Vec<TestCase>>,
}

/// A single test case for evaluation.
pub struct TestCase {
    /// Input text
    pub text: String,
    /// Gold standard entities
    pub gold_entities: Vec<SimpleGoldEntity>,
}

/// Simplified gold entity for report generation.
///
/// This is a simplified version with string-based entity types,
/// designed for report generation where type normalization is handled externally.
///
/// For evaluation code, use [`super::datasets::GoldEntity`] instead.
#[derive(Debug, Clone)]
pub struct SimpleGoldEntity {
    /// Entity text
    pub text: String,
    /// Entity type label (e.g., "PER", "ORG", "DATE")
    pub entity_type: String,
    /// Start character offset
    pub start: usize,
    /// End character offset (exclusive)
    pub end: usize,
}

impl SimpleGoldEntity {
    /// Safely extract text from source using character offsets.
    ///
    /// SimpleGoldEntity stores character offsets, not byte offsets. This method
    /// correctly extracts text by iterating over characters.
    ///
    /// # Arguments
    /// * `source_text` - The original text from which this entity was extracted
    ///
    /// # Returns
    /// The extracted text, or empty string if offsets are invalid
    #[must_use]
    pub fn extract_text(&self, source_text: &str) -> String {
        let char_count = source_text.chars().count();
        if self.start >= char_count || self.end > char_count || self.start >= self.end {
            return String::new();
        }
        source_text
            .chars()
            .skip(self.start)
            .take(self.end - self.start)
            .collect()
    }
}

impl ReportBuilder {
    /// Create a new report builder.
    pub fn new(model_name: &str) -> Self {
        Self {
            model_name: model_name.to_string(),
            include_core: true,
            include_errors: true,
            include_bias: false,
            include_data_quality: false,
            include_calibration: false,
            test_data: None,
        }
    }

    /// Include core metrics (default: true).
    pub fn with_core_metrics(mut self, include: bool) -> Self {
        self.include_core = include;
        self
    }

    /// Include error analysis (default: true).
    pub fn with_error_analysis(mut self, include: bool) -> Self {
        self.include_errors = include;
        self
    }

    /// Include demographic NER bias analysis (default: false).
    ///
    /// This builder evaluates the supplied [`Model`] on person-name examples.
    /// It cannot evaluate coreference gender bias because no coreference resolver is supplied.
    pub fn with_bias_analysis(mut self, include: bool) -> Self {
        self.include_bias = include;
        self
    }

    /// Include data quality checks (default: false).
    pub fn with_data_quality(mut self, include: bool) -> Self {
        self.include_data_quality = include;
        self
    }

    /// Include calibration analysis (default: false).
    /// Only meaningful for models that provide confidence scores.
    pub fn with_calibration(mut self, include: bool) -> Self {
        self.include_calibration = include;
        self
    }

    /// Set test data for evaluation.
    pub fn with_test_data(mut self, data: Vec<TestCase>) -> Self {
        self.test_data = Some(data);
        self
    }

    /// Run demographic NER bias analysis for the supplied model.
    #[cfg(feature = "eval-bias")]
    fn run_bias_analysis<M: Model>(model: &M, warnings: &mut Vec<String>) -> Result<BiasSummary> {
        use crate::eval::demographic_bias::{
            create_diverse_name_dataset, DemographicBiasEvaluator,
        };

        // This API receives a NER model, not a coreference resolver. Do not
        // substitute a default resolver and attribute its result to `model`.
        warnings.push(
            "Gender bias is unavailable because ReportBuilder has no supplied coreference resolver."
                .to_string(),
        );

        let names = create_diverse_name_dataset();
        let evaluator = DemographicBiasEvaluator::new(true);
        let demo_results = evaluator.try_evaluate_ner(model, &names)?;

        let bias_detected = demo_results.ethnicity_parity_gap > 0.1;

        // Find underperforming groups
        let mut underperforming_groups = Vec::new();
        for (ethnicity, rate) in &demo_results.by_ethnicity {
            if *rate < demo_results.overall_recognition_rate - 0.1 {
                underperforming_groups.push(ethnicity.clone());
            }
        }

        Ok(BiasSummary {
            bias_detected,
            gender: None,
            demographic: Some(DemographicBiasMetrics {
                max_gap: demo_results
                    .ethnicity_parity_gap
                    .max(demo_results.script_bias_gap),
                underperforming_groups,
            }),
            length: None, // Can be added if needed
        })
    }

    /// Run calibration analysis.
    #[cfg(feature = "eval")]
    fn run_calibration_analysis<M: Model>(
        model: &M,
        test_cases: &[TestCase],
    ) -> Result<CalibrationSummary> {
        use crate::eval::calibration::CalibrationEvaluator;

        // Collect predictions with confidence scores
        let mut predictions = Vec::new();
        let mut has_calibrated_entities = false;

        for (case_index, case) in test_cases.iter().enumerate() {
            let entities = model.extract_entities(&case.text, None).map_err(|e| {
                crate::Error::Inference(format!(
                    "calibration extraction failed for test_case={}: {}",
                    case_index, e
                ))
            })?;

            for entity in &entities {
                // Check if this entity's extraction method is calibrated
                let is_calibrated = entity
                    .provenance
                    .as_ref()
                    .map(|p| p.method.is_calibrated())
                    .unwrap_or(false);

                if !is_calibrated {
                    continue; // Skip uncalibrated entities
                }

                has_calibrated_entities = true;

                // Match to gold to determine correctness
                let is_correct = case.gold_entities.iter().any(|gold| {
                    entity.start() == gold.start
                        && entity.end() == gold.end
                        && entity.entity_type.as_label() == gold.entity_type
                });

                predictions.push((entity.confidence.into(), is_correct));
            }
        }

        // Calibration cannot be measured without calibrated predictions.
        if !has_calibrated_entities || predictions.is_empty() {
            return Err(crate::Error::InvalidInput(
                "calibration requested but no calibrated predictions were produced".to_string(),
            ));
        }

        // Compute calibration metrics
        let results = CalibrationEvaluator::compute(&predictions);

        // Find optimal threshold (highest accuracy with reasonable coverage)
        let optimal_threshold = results
            .threshold_accuracy
            .iter()
            .filter(|(_, metrics)| metrics.coverage >= 0.1) // At least 10% coverage
            .max_by(|(_, a), (_, b)| {
                a.accuracy
                    .partial_cmp(&b.accuracy)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .and_then(|(thresh_str, _)| thresh_str.parse::<f64>().ok())
            .unwrap_or(0.5);

        // Convert ECE to grade
        let grade = if results.ece < 0.05 {
            'A'
        } else if results.ece < 0.10 {
            'B'
        } else if results.ece < 0.15 {
            'C'
        } else if results.ece < 0.25 {
            'D'
        } else {
            'F'
        };

        Ok(CalibrationSummary {
            ece: results.ece,
            mce: results.mce,
            optimal_threshold,
            grade,
        })
    }

    /// Run data quality checks.
    #[cfg(feature = "eval")]
    fn run_data_quality_checks(test_cases: &[TestCase]) -> Result<DataQualitySummary> {
        use std::collections::{HashMap, HashSet};

        if test_cases.is_empty() {
            return Ok(DataQualitySummary {
                leakage_detected: false,
                redundancy_rate: 0.0,
                ambiguous_count: 0,
            });
        }

        // Convert test cases to format expected by DatasetQualityAnalyzer
        // Since we only have test data (no train), we check for redundancy within test set
        let test_data: Vec<(&str, Vec<(&str, &str)>)> = test_cases
            .iter()
            .map(|case| {
                let entities: Vec<(&str, &str)> = case
                    .gold_entities
                    .iter()
                    .map(|e| (e.text.as_str(), e.entity_type.as_str()))
                    .collect();
                (case.text.as_str(), entities)
            })
            .collect();

        // Check for redundancy (duplicates within test set)
        let mut seen_texts = HashSet::new();
        let mut duplicate_count = 0;
        for (text, _) in &test_data {
            let normalized = text.to_lowercase();
            if !seen_texts.insert(normalized) {
                duplicate_count += 1;
            }
        }
        let redundancy_rate = if test_data.is_empty() {
            0.0
        } else {
            duplicate_count as f64 / test_data.len() as f64
        };

        // Check for ambiguous entities (same text, different types)
        let mut text_to_types: HashMap<String, HashSet<String>> = HashMap::new();
        for (_, entities) in &test_data {
            for (text, entity_type) in entities {
                text_to_types
                    .entry(text.to_lowercase())
                    .or_default()
                    .insert(entity_type.to_string());
            }
        }
        let ambiguous_count = text_to_types
            .values()
            .filter(|types| types.len() > 1)
            .count();

        // Note: We can't check for train-test leakage since we only have test data
        // This would require access to training data, which is not available in this context

        Ok(DataQualitySummary {
            leakage_detected: false, // Cannot determine without train data
            redundancy_rate,
            ambiguous_count,
        })
    }

    /// Build the report by running the model on test data.
    pub fn build<M: Model>(self, model: &M) -> Result<EvalReport> {
        let timestamp = chrono_lite_timestamp();
        let mut warnings = Vec::new();
        let mut recommendations = Vec::new();

        // Get test data (use synthetic if none provided)
        let test_cases = self.test_data.unwrap_or_else(|| {
            warnings.push("Using synthetic test data (no custom data provided)".into());
            default_synthetic_cases()
        });

        // Run model predictions
        let mut total_gold = 0;
        let mut total_predicted = 0;
        let mut total_correct = 0;
        let mut per_type_stats: HashMap<String, (usize, usize, usize)> = HashMap::new();
        let mut all_errors = Vec::new();

        for (case_index, case) in test_cases.iter().enumerate() {
            let predictions = model.extract_entities(&case.text, None).map_err(|e| {
                crate::Error::Inference(format!(
                    "report extraction failed for model={} test_case={}: {}",
                    self.model_name, case_index, e
                ))
            })?;

            total_gold += case.gold_entities.len();
            total_predicted += predictions.len();

            // Match predictions to gold one-to-one. A model can emit duplicate entities, and a
            // test set can contain duplicate annotations, so a prediction must not credit more
            // than one gold entity.
            let matched_predictions = match_prediction_indices(&case.gold_entities, &predictions);
            for (gold_index, gold) in case.gold_entities.iter().enumerate() {
                let type_key = gold.entity_type.clone();
                let entry = per_type_stats.entry(type_key.clone()).or_insert((0, 0, 0));
                entry.0 += 1; // gold count

                let matched = matched_predictions[gold_index];

                if matched {
                    total_correct += 1;
                    entry.2 += 1; // correct count
                } else {
                    all_errors.push(format!("Missed: {} ({})", gold.text, gold.entity_type));
                }
            }

            for pred in &predictions {
                let type_key = pred.entity_type.as_label().to_string();
                let entry = per_type_stats.entry(type_key).or_insert((0, 0, 0));
                entry.1 += 1; // predicted count
            }
        }

        // Compute core metrics
        let precision = if total_predicted > 0 {
            total_correct as f64 / total_predicted as f64
        } else {
            0.0
        };
        let recall = if total_gold > 0 {
            total_correct as f64 / total_gold as f64
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        let core = CoreMetrics {
            precision,
            recall,
            f1,
            total_gold,
            total_predicted,
            total_correct,
            macro_f1: None, // Computed below if needed
        };

        // Per-type metrics
        let mut per_type = HashMap::new();
        let mut type_f1s = Vec::new();
        for (type_name, (gold, pred, correct)) in &per_type_stats {
            let p = if *pred > 0 {
                *correct as f64 / *pred as f64
            } else {
                0.0
            };
            let r = if *gold > 0 {
                *correct as f64 / *gold as f64
            } else {
                0.0
            };
            let f = if p + r > 0.0 {
                2.0 * p * r / (p + r)
            } else {
                0.0
            };
            type_f1s.push(f);
            per_type.insert(
                type_name.clone(),
                TypeMetrics {
                    precision: p,
                    recall: r,
                    f1: f,
                    support: *gold,
                    predicted: *pred,
                    correct: *correct,
                },
            );
        }

        // Generate recommendations
        if f1 < 0.5 {
            recommendations.push(Recommendation {
                priority: Priority::High,
                category: RecommendationCategory::Performance,
                message: format!(
                    "F1 score ({:.1}%) is below acceptable threshold",
                    f1 * 100.0
                ),
                estimated_impact: Some("Core functionality compromised".into()),
            });
        }

        if recall < precision * 0.7 {
            recommendations.push(Recommendation {
                priority: Priority::Medium,
                category: RecommendationCategory::Coverage,
                message: "Recall significantly lower than precision - model is too conservative"
                    .into(),
                estimated_impact: Some("Missing many valid entities".into()),
            });
        }

        // Error summary
        let errors = if self.include_errors {
            let false_negatives = total_gold - total_correct;
            let false_positives = total_predicted - total_correct;
            Some(ErrorSummary {
                total_errors: false_negatives + false_positives,
                boundary_errors: 0, // Would need span comparison
                type_errors: 0,     // Would need type comparison
                false_positives,
                false_negatives,
                top_patterns: all_errors.into_iter().take(5).collect(),
            })
        } else {
            None
        };

        // Demographic NER bias analysis
        let bias = if self.include_bias {
            #[cfg(feature = "eval-bias")]
            {
                Some(Self::run_bias_analysis(model, &mut warnings)?)
            }
            #[cfg(not(feature = "eval-bias"))]
            {
                return Err(crate::Error::FeatureNotAvailable(
                    "bias analysis requires the eval-bias feature".to_string(),
                ));
            }
        } else {
            None
        };

        // Calibration (if enabled)
        let calibration = if self.include_calibration {
            #[cfg(feature = "eval")]
            {
                Some(Self::run_calibration_analysis(model, &test_cases)?)
            }
            #[cfg(not(feature = "eval"))]
            {
                return Err(crate::Error::FeatureNotAvailable(
                    "calibration analysis requires the eval feature".to_string(),
                ));
            }
        } else {
            None
        };

        // Data quality (if enabled)
        let data_quality = if self.include_data_quality {
            #[cfg(feature = "eval")]
            {
                Some(Self::run_data_quality_checks(&test_cases)?)
            }
            #[cfg(not(feature = "eval"))]
            {
                return Err(crate::Error::FeatureNotAvailable(
                    "data quality checks require the eval feature".to_string(),
                ));
            }
        } else {
            None
        };

        Ok(EvalReport {
            model_name: self.model_name,
            timestamp,
            core,
            per_type,
            errors,
            bias,
            data_quality,
            calibration,
            recommendations,
            warnings,
        })
    }
}

// =============================================================================
// Display Implementation
// =============================================================================

impl EvalReport {
    /// Generate a human-readable summary.
    pub fn summary(&self) -> String {
        let mut out = String::new();

        out.push_str(&format!("=== Evaluation Report: {} ===\n", self.model_name));
        out.push_str(&format!("Generated: {}\n\n", self.timestamp));

        // Core metrics
        out.push_str("## Core Metrics\n");
        out.push_str(&format!(
            "  Precision: {:.1}%\n",
            self.core.precision * 100.0
        ));
        out.push_str(&format!("  Recall:    {:.1}%\n", self.core.recall * 100.0));
        out.push_str(&format!("  F1:        {:.1}%\n", self.core.f1 * 100.0));
        out.push_str(&format!(
            "  ({} correct / {} predicted / {} gold)\n\n",
            self.core.total_correct, self.core.total_predicted, self.core.total_gold
        ));

        // Per-type breakdown
        if !self.per_type.is_empty() {
            out.push_str("## Per-Type Breakdown\n");
            let mut types: Vec<_> = self.per_type.iter().collect();
            types.sort_by_key(|b| std::cmp::Reverse(b.1.support));
            for (type_name, metrics) in types {
                out.push_str(&format!(
                    "  {:12} P={:.0}% R={:.0}% F1={:.0}% (n={})\n",
                    type_name,
                    metrics.precision * 100.0,
                    metrics.recall * 100.0,
                    metrics.f1 * 100.0,
                    metrics.support
                ));
            }
            out.push('\n');
        }

        // Error summary
        if let Some(ref errors) = self.errors {
            out.push_str("## Error Analysis\n");
            out.push_str(&format!("  Total errors: {}\n", errors.total_errors));
            out.push_str(&format!("  False positives: {}\n", errors.false_positives));
            out.push_str(&format!("  False negatives: {}\n", errors.false_negatives));
            if !errors.top_patterns.is_empty() {
                out.push_str("  Sample errors:\n");
                for pattern in &errors.top_patterns {
                    out.push_str(&format!("    - {}\n", pattern));
                }
            }
            out.push('\n');
        }

        // Recommendations
        if !self.recommendations.is_empty() {
            out.push_str("## Recommendations\n");
            for rec in &self.recommendations {
                let priority = match rec.priority {
                    Priority::High => "[HIGH]",
                    Priority::Medium => "[MED]",
                    Priority::Low => "[LOW]",
                };
                out.push_str(&format!("  {} {}\n", priority, rec.message));
            }
            out.push('\n');
        }

        // Warnings
        if !self.warnings.is_empty() {
            out.push_str("## Warnings\n");
            for warning in &self.warnings {
                out.push_str(&format!("  - {}\n", warning));
            }
        }

        out
    }

    /// Export report as JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| crate::Error::InvalidInput(format!("JSON serialization failed: {}", e)))
    }
}

impl fmt::Display for EvalReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// =============================================================================
// Helpers
// =============================================================================

fn chrono_lite_timestamp() -> String {
    // Simple timestamp without chrono dependency
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}s since epoch", duration.as_secs())
}

/// Return the gold entities that can be credited by a distinct exact-match prediction.
///
/// Predictions are reserved in gold order, which makes the result deterministic for duplicate
/// annotations while preserving the report's exact-span, exact-label metric definition.
fn match_prediction_indices(
    gold_entities: &[SimpleGoldEntity],
    predictions: &[Entity],
) -> Vec<bool> {
    let mut used_predictions = vec![false; predictions.len()];

    gold_entities
        .iter()
        .map(|gold| {
            let Some((index, _)) = predictions.iter().enumerate().find(|(index, prediction)| {
                !used_predictions[*index]
                    && prediction.start() == gold.start
                    && prediction.end() == gold.end
                    && prediction.entity_type.as_label() == gold.entity_type
            }) else {
                return false;
            };
            used_predictions[index] = true;
            true
        })
        .collect()
}

fn default_synthetic_cases() -> Vec<TestCase> {
    // Minimal synthetic test set for quick evaluation
    vec![
        TestCase {
            text: "Meeting on January 15, 2024 at 3:00 PM".into(),
            gold_entities: vec![
                SimpleGoldEntity {
                    text: "January 15, 2024".into(),
                    entity_type: "DATE".into(),
                    start: 11,
                    end: 27,
                },
                SimpleGoldEntity {
                    text: "3:00 PM".into(),
                    entity_type: "TIME".into(),
                    start: 31,
                    end: 38,
                },
            ],
        },
        TestCase {
            text: "Contact: user@example.com or call 555-1234".into(),
            gold_entities: vec![
                SimpleGoldEntity {
                    text: "user@example.com".into(),
                    entity_type: "EMAIL".into(),
                    start: 9,
                    end: 25,
                },
                SimpleGoldEntity {
                    text: "555-1234".into(),
                    entity_type: "PHONE".into(),
                    start: 34,
                    end: 42,
                },
            ],
        },
        TestCase {
            text: "Invoice total: $1,234.56 USD".into(),
            gold_entities: vec![SimpleGoldEntity {
                text: "$1,234.56".into(),
                entity_type: "MONEY".into(),
                start: 15,
                end: 24,
            }],
        },
    ]
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_gold_cannot_reuse_one_prediction() {
        let gold = SimpleGoldEntity {
            text: "Alice".into(),
            entity_type: "PER".into(),
            start: 0,
            end: 5,
        };
        let model = anno::AnyModel::new(
            "one",
            "one prediction",
            vec![anno::EntityType::Person],
            |_, _| {
                Ok(vec![Entity::new(
                    "Alice",
                    anno::EntityType::Person,
                    0,
                    5,
                    1.0,
                )])
            },
        );
        let report = ReportBuilder::new("one")
            .with_test_data(vec![TestCase {
                text: "Alice".into(),
                gold_entities: vec![gold.clone(), gold],
            }])
            .build(&model)
            .unwrap();
        assert_eq!(report.core.total_correct, 1);
        assert_eq!(report.core.total_gold, 2);
        assert_eq!(report.core.precision, 1.0);
        assert_eq!(report.core.recall, 0.5);
    }

    #[test]
    fn test_report_builder_basic() {
        use crate::RegexNER;
        let model = RegexNER::new();
        let report = ReportBuilder::new("RegexNER")
            .with_error_analysis(true)
            .build(&model)
            .unwrap();

        assert_eq!(report.model_name, "RegexNER");
        assert!(report.core.total_gold > 0);
    }

    #[test]
    fn test_report_summary_format() {
        let report = EvalReport {
            model_name: "TestModel".into(),
            timestamp: "2024-01-01".into(),
            core: CoreMetrics {
                precision: 0.85,
                recall: 0.75,
                f1: 0.80,
                total_gold: 100,
                total_predicted: 90,
                total_correct: 75,
                macro_f1: None,
            },
            per_type: HashMap::new(),
            errors: None,
            bias: None,
            data_quality: None,
            calibration: None,
            recommendations: vec![],
            warnings: vec![],
        };

        let summary = report.summary();
        assert!(summary.contains("TestModel"));
        assert!(summary.contains("85.0%")); // precision
        assert!(summary.contains("75.0%")); // recall
    }

    #[test]
    fn test_report_json_export() {
        let report = EvalReport {
            model_name: "TestModel".into(),
            timestamp: "test".into(),
            core: CoreMetrics {
                precision: 0.9,
                recall: 0.8,
                f1: 0.85,
                total_gold: 10,
                total_predicted: 10,
                total_correct: 8,
                macro_f1: None,
            },
            per_type: HashMap::new(),
            errors: None,
            bias: None,
            data_quality: None,
            calibration: None,
            recommendations: vec![],
            warnings: vec![],
        };

        let json = report.to_json().unwrap();
        assert!(json.contains("\"model_name\": \"TestModel\""));
        assert!(json.contains("\"f1\": 0.85"));
    }

    #[test]
    fn test_recommendations_generated() {
        use crate::RegexNER;
        let model = RegexNER::new();

        // Create test data that will result in low F1
        let test_data = vec![TestCase {
            text: "John Smith works at Google".into(),
            gold_entities: vec![
                SimpleGoldEntity {
                    text: "John Smith".into(),
                    entity_type: "PER".into(),
                    start: 0,
                    end: 10,
                },
                SimpleGoldEntity {
                    text: "Google".into(),
                    entity_type: "ORG".into(),
                    start: 20,
                    end: 26,
                },
            ],
        }];

        let report = ReportBuilder::new("RegexNER")
            .with_test_data(test_data)
            .build(&model)
            .unwrap();

        // RegexNER can't detect PER/ORG, so F1 should be low
        // and recommendations should be generated
        assert!(
            report.core.f1 < 0.5
                || !report.recommendations.is_empty()
                || report.core.total_gold == 0
        );
    }

    #[test]
    fn report_builder_propagates_extraction_failure_with_context() {
        let model = anno::AnyModel::new("failing", "always fails", vec![], |_, _| {
            Err(anno::Error::Inference("model unavailable".to_string()))
        });
        let error = ReportBuilder::new("failing-model")
            .with_test_data(vec![TestCase {
                text: "Alice".to_string(),
                gold_entities: vec![],
            }])
            .build(&model)
            .expect_err("report construction must fail when extraction fails");
        let message = error.to_string();
        assert!(message.contains("failing-model"));
        assert!(message.contains("test_case=0"));
        assert!(message.contains("model unavailable"));
    }

    #[cfg(feature = "eval-bias")]
    #[test]
    fn report_builder_bias_is_demographic_ner_only() {
        let model = anno::AnyModel::new("empty", "returns no entities", vec![], |_, _| Ok(vec![]));

        let report = ReportBuilder::new("empty")
            .with_bias_analysis(true)
            .with_test_data(vec![])
            .build(&model)
            .expect("demographic bias report should succeed");

        let bias = report.bias.expect("requested bias analysis is present");
        assert!(bias.gender.is_none());
        assert!(report.warnings.iter().any(|warning| {
            warning.contains("Gender bias is unavailable because ReportBuilder")
        }));
    }

    #[cfg(feature = "eval-bias")]
    #[test]
    fn report_builder_propagates_requested_bias_inference_failure() {
        let model = anno::AnyModel::new("failing", "fails during bias", vec![], |_, _| {
            Err(anno::Error::Inference(
                "bias backend unavailable".to_string(),
            ))
        });

        let error = ReportBuilder::new("failing")
            .with_bias_analysis(true)
            .with_test_data(vec![])
            .build(&model)
            .expect_err("requested bias analysis must propagate inference failure");

        let message = error.to_string();
        assert!(message.contains("demographic bias extraction failed for name_index=0"));
        assert!(message.contains("bias backend unavailable"));
    }

    #[cfg(feature = "eval")]
    #[test]
    fn report_builder_propagates_calibration_extraction_failure() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        let calls = Arc::new(AtomicUsize::new(0));
        let calls_for_model = Arc::clone(&calls);
        let model = anno::AnyModel::new(
            "stateful",
            "fails during calibration",
            vec![],
            move |_, _| {
                if calls_for_model.fetch_add(1, Ordering::SeqCst) == 0 {
                    Ok(vec![])
                } else {
                    Err(anno::Error::Inference(
                        "calibration pass unavailable".to_string(),
                    ))
                }
            },
        );
        let error = ReportBuilder::new("stateful")
            .with_calibration(true)
            .with_test_data(vec![TestCase {
                text: "Alice".to_string(),
                gold_entities: vec![],
            }])
            .build(&model)
            .expect_err("calibration extraction failure must fail the report");
        assert!(error.to_string().contains("calibration pass unavailable"));
    }
}
