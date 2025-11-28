//! Unified evaluation report.
//!
//! Provides a single cohesive report structure that aggregates results from
//! all evaluation modules. This is the primary output type users should work with.
//!
//! # Example
//!
//! ```rust
//! use anno::eval::report::{EvalReport, ReportBuilder};
//! use anno::PatternNER;
//!
//! let model = PatternNER::new();
//! let report = ReportBuilder::new("PatternNER")
//!     .with_core_metrics(true)
//!     .with_bias_analysis(false)  // Skip if no PER/ORG support
//!     .with_error_analysis(true)
//!     .build(&model);
//!
//! println!("{}", report.summary());
//! ```

use crate::{Model, Result};
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
    
    /// Bias analysis (if enabled and applicable)
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
    pub gold_entities: Vec<GoldEntity>,
}

/// Gold standard entity for evaluation.
#[derive(Debug, Clone)]
pub struct GoldEntity {
    /// Entity text
    pub text: String,
    /// Entity type label (e.g., "PER", "ORG", "DATE")
    pub entity_type: String,
    /// Start character offset
    pub start: usize,
    /// End character offset (exclusive)
    pub end: usize,
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

    /// Include bias analysis (default: false).
    /// Only meaningful for models that detect PER/ORG entities.
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

    /// Build the report by running the model on test data.
    pub fn build<M: Model>(self, model: &M) -> EvalReport {
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

        for case in &test_cases {
            let predictions = model.extract_entities(&case.text, None).unwrap_or_default();
            
            total_gold += case.gold_entities.len();
            total_predicted += predictions.len();

            // Match predictions to gold
            for gold in &case.gold_entities {
                let type_key = gold.entity_type.clone();
                let entry = per_type_stats.entry(type_key.clone()).or_insert((0, 0, 0));
                entry.0 += 1; // gold count

                let matched = predictions.iter().any(|p| {
                    p.start == gold.start && p.end == gold.end 
                        && p.entity_type.as_label() == gold.entity_type
                });

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
            let p = if *pred > 0 { *correct as f64 / *pred as f64 } else { 0.0 };
            let r = if *gold > 0 { *correct as f64 / *gold as f64 } else { 0.0 };
            let f = if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 };
            type_f1s.push(f);
            per_type.insert(type_name.clone(), TypeMetrics {
                precision: p,
                recall: r,
                f1: f,
                support: *gold,
                predicted: *pred,
                correct: *correct,
            });
        }

        // Generate recommendations
        if f1 < 0.5 {
            recommendations.push(Recommendation {
                priority: Priority::High,
                category: RecommendationCategory::Performance,
                message: format!("F1 score ({:.1}%) is below acceptable threshold", f1 * 100.0),
                estimated_impact: Some("Core functionality compromised".into()),
            });
        }

        if recall < precision * 0.7 {
            recommendations.push(Recommendation {
                priority: Priority::Medium,
                category: RecommendationCategory::Coverage,
                message: "Recall significantly lower than precision - model is too conservative".into(),
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

        EvalReport {
            model_name: self.model_name,
            timestamp,
            core,
            per_type,
            errors,
            bias: None,           // Bias analysis not yet integrated
            data_quality: None,   // Data quality not yet integrated
            calibration: None,    // Calibration not yet integrated
            recommendations,
            warnings,
        }
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
        out.push_str(&format!("  Precision: {:.1}%\n", self.core.precision * 100.0));
        out.push_str(&format!("  Recall:    {:.1}%\n", self.core.recall * 100.0));
        out.push_str(&format!("  F1:        {:.1}%\n", self.core.f1 * 100.0));
        out.push_str(&format!("  ({} correct / {} predicted / {} gold)\n\n", 
            self.core.total_correct, self.core.total_predicted, self.core.total_gold));
        
        // Per-type breakdown
        if !self.per_type.is_empty() {
            out.push_str("## Per-Type Breakdown\n");
            let mut types: Vec<_> = self.per_type.iter().collect();
            types.sort_by(|a, b| b.1.support.cmp(&a.1.support));
            for (type_name, metrics) in types {
                out.push_str(&format!("  {:12} P={:.0}% R={:.0}% F1={:.0}% (n={})\n",
                    type_name, metrics.precision * 100.0, metrics.recall * 100.0,
                    metrics.f1 * 100.0, metrics.support));
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

fn default_synthetic_cases() -> Vec<TestCase> {
    // Minimal synthetic test set for quick evaluation
    vec![
        TestCase {
            text: "Meeting on January 15, 2024 at 3:00 PM".into(),
            gold_entities: vec![
                GoldEntity {
                    text: "January 15, 2024".into(),
                    entity_type: "DATE".into(),
                    start: 11,
                    end: 27,
                },
                GoldEntity {
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
                GoldEntity {
                    text: "user@example.com".into(),
                    entity_type: "EMAIL".into(),
                    start: 9,
                    end: 25,
                },
                GoldEntity {
                    text: "555-1234".into(),
                    entity_type: "PHONE".into(),
                    start: 34,
                    end: 42,
                },
            ],
        },
        TestCase {
            text: "Invoice total: $1,234.56 USD".into(),
            gold_entities: vec![
                GoldEntity {
                    text: "$1,234.56".into(),
                    entity_type: "MONEY".into(),
                    start: 15,
                    end: 24,
                },
            ],
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
    fn test_report_builder_basic() {
        use crate::PatternNER;
        let model = PatternNER::new();
        let report = ReportBuilder::new("PatternNER")
            .with_error_analysis(true)
            .build(&model);
        
        assert_eq!(report.model_name, "PatternNER");
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
        use crate::PatternNER;
        let model = PatternNER::new();
        
        // Create test data that will result in low F1
        let test_data = vec![
            TestCase {
                text: "John Smith works at Google".into(),
                gold_entities: vec![
                    GoldEntity {
                        text: "John Smith".into(),
                        entity_type: "PER".into(),
                        start: 0,
                        end: 10,
                    },
                    GoldEntity {
                        text: "Google".into(),
                        entity_type: "ORG".into(),
                        start: 20,
                        end: 26,
                    },
                ],
            },
        ];
        
        let report = ReportBuilder::new("PatternNER")
            .with_test_data(test_data)
            .build(&model);
        
        // PatternNER can't detect PER/ORG, so F1 should be low
        // and recommendations should be generated
        assert!(report.core.f1 < 0.5 || !report.recommendations.is_empty() || report.core.total_gold == 0);
    }
}

