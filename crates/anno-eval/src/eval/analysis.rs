//! Error analysis and diagnostics for NER evaluation.
//!
//! Provides tools to understand *why* a model makes mistakes:
//! - Confusion matrix (which types get confused)
//! - Error categorization (boundary, type, spurious, missed)
//! - Statistical significance testing between systems

use super::datasets::GoldEntity;
use anno_core::Entity;
use std::collections::HashMap;

// =============================================================================
// Confusion Matrix
// =============================================================================

/// Confusion matrix for entity type predictions.
///
/// Shows which entity types get confused with which others.
#[derive(Debug, Clone, Default)]
pub struct ConfusionMatrix {
    /// Matrix[predicted][actual] = count
    matrix: HashMap<String, HashMap<String, usize>>,
    /// Total predictions per type
    pub predicted_totals: HashMap<String, usize>,
    /// Total ground truth per type
    pub actual_totals: HashMap<String, usize>,
}

impl ConfusionMatrix {
    /// Create a new empty confusion matrix.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a prediction-actual pair.
    pub fn add(&mut self, predicted: &str, actual: &str) {
        *self
            .matrix
            .entry(predicted.to_string())
            .or_default()
            .entry(actual.to_string())
            .or_insert(0) += 1;
        *self
            .predicted_totals
            .entry(predicted.to_string())
            .or_insert(0) += 1;
        *self.actual_totals.entry(actual.to_string()).or_insert(0) += 1;
    }

    /// Get count for a prediction-actual pair.
    #[must_use]
    pub fn get(&self, predicted: &str, actual: &str) -> usize {
        self.matrix
            .get(predicted)
            .and_then(|row| row.get(actual))
            .copied()
            .unwrap_or(0)
    }

    /// Get all entity types in the matrix.
    #[must_use]
    pub fn types(&self) -> Vec<String> {
        let mut types: Vec<String> = self
            .predicted_totals
            .keys()
            .chain(self.actual_totals.keys())
            .cloned()
            .collect();
        types.sort();
        types.dedup();
        types
    }

    /// Get per-type precision.
    #[must_use]
    pub fn precision(&self, entity_type: &str) -> f64 {
        let correct = self.get(entity_type, entity_type);
        let predicted = self.predicted_totals.get(entity_type).copied().unwrap_or(0);
        if predicted == 0 {
            0.0
        } else {
            correct as f64 / predicted as f64
        }
    }

    /// Get per-type recall.
    #[must_use]
    pub fn recall(&self, entity_type: &str) -> f64 {
        let correct = self.get(entity_type, entity_type);
        let actual = self.actual_totals.get(entity_type).copied().unwrap_or(0);
        if actual == 0 {
            0.0
        } else {
            correct as f64 / actual as f64
        }
    }

    /// Get most confused pairs (predicted, actual, count).
    #[must_use]
    pub fn most_confused(&self, top_n: usize) -> Vec<(String, String, usize)> {
        let mut confusions: Vec<(String, String, usize)> = Vec::new();

        for (pred, actuals) in &self.matrix {
            for (actual, &count) in actuals {
                if pred != actual && count > 0 {
                    confusions.push((pred.clone(), actual.clone(), count));
                }
            }
        }

        confusions.sort_by(|a, b| b.2.cmp(&a.2));
        confusions.truncate(top_n);
        confusions
    }
}

impl std::fmt::Display for ConfusionMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let types = self.types();

        // Header
        write!(f, "{:12}", "Pred\\Actual")?;
        for t in &types {
            write!(f, " {:>8}", &t[..t.len().min(8)])?;
        }
        writeln!(f)?;

        // Rows
        for pred in &types {
            write!(f, "{:12}", &pred[..pred.len().min(12)])?;
            for actual in &types {
                let count = self.get(pred, actual);
                if pred == actual {
                    write!(f, " {:>8}", format!("[{}]", count))?;
                } else if count > 0 {
                    write!(f, " {:>8}", count)?;
                } else {
                    write!(f, " {:>8}", ".")?;
                }
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

// =============================================================================
// Error Categorization
// =============================================================================

/// Categories of NER errors.
///
/// Note: This type overlaps with `ErrorCategory` in the `error_analysis` module
/// (available with the `eval-advanced` feature). The mapping is:
/// - `TypeMismatch` ↔ `ErrorCategory::TypeError`
/// - `BoundaryError` ↔ `ErrorCategory::BoundaryError`
/// - `BoundaryAndType` ↔ `ErrorCategory::PartialMatch`
/// - `Spurious` ↔ `ErrorCategory::FalsePositive`
/// - `Missed` ↔ `ErrorCategory::FalseNegative`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorType {
    /// Correct span, wrong type
    TypeMismatch,
    /// Overlapping span, correct type (boundary error)
    BoundaryError,
    /// Overlapping span, wrong type
    BoundaryAndType,
    /// Predicted entity with no ground truth match
    Spurious,
    /// Ground truth entity with no prediction
    Missed,
}

#[cfg(feature = "eval-advanced")]
impl ErrorType {
    /// Convert to the equivalent [`super::error_analysis::ErrorCategory`].
    #[must_use]
    pub fn to_error_category(self) -> super::error_analysis::ErrorCategory {
        use super::error_analysis::ErrorCategory;
        match self {
            ErrorType::TypeMismatch => ErrorCategory::TypeError,
            ErrorType::BoundaryError => ErrorCategory::BoundaryError,
            ErrorType::BoundaryAndType => ErrorCategory::PartialMatch,
            ErrorType::Spurious => ErrorCategory::FalsePositive,
            ErrorType::Missed => ErrorCategory::FalseNegative,
        }
    }
}

/// A single NER error instance.
#[derive(Debug, Clone)]
pub struct NERError {
    /// Error category
    pub error_type: ErrorType,
    /// Predicted entity (if any)
    pub predicted: Option<Entity>,
    /// Ground truth entity (if any)
    pub gold: Option<GoldEntity>,
    /// Text context
    pub context: String,
}

/// Error analysis results.
#[derive(Debug, Clone, Default)]
pub struct ErrorAnalysis {
    /// Errors by category
    pub errors: Vec<NERError>,
    /// Count by error type
    pub counts: HashMap<ErrorType, usize>,
    /// Total predictions
    pub total_predictions: usize,
    /// Total ground truth
    pub total_gold: usize,
}

impl ErrorAnalysis {
    /// Create a new error analysis.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Analyze predictions against ground truth.
    pub fn analyze(text: &str, predicted: &[Entity], gold: &[GoldEntity]) -> Self {
        let mut analysis = Self::new();
        analysis.total_predictions = predicted.len();
        analysis.total_gold = gold.len();

        let mut gold_matched = vec![false; gold.len()];

        // Check each prediction
        for pred in predicted {
            let mut best_match: Option<(usize, ErrorType)> = None;
            let mut is_perfect_match = false;

            for (i, g) in gold.iter().enumerate() {
                if gold_matched[i] {
                    continue;
                }

                // Check for exact match
                if pred.start == g.start && pred.end == g.end {
                    if pred.entity_type == g.entity_type {
                        // Correct - not an error
                        gold_matched[i] = true;
                        is_perfect_match = true;
                        break;
                    } else {
                        best_match = Some((i, ErrorType::TypeMismatch));
                    }
                }
                // Check for overlap
                else if pred.start < g.end && pred.end > g.start {
                    let error = if pred.entity_type == g.entity_type {
                        ErrorType::BoundaryError
                    } else {
                        ErrorType::BoundaryAndType
                    };
                    if best_match.is_none() {
                        best_match = Some((i, error));
                    }
                }
            }

            // Skip perfect matches - they're not errors
            if is_perfect_match {
                continue;
            }

            if let Some((gold_idx, error_type)) = best_match {
                // Found a match but with an error (type/boundary)
                gold_matched[gold_idx] = true;
                let char_count = text.chars().count();
                let context_start = pred.start.saturating_sub(20);
                let context_end = (pred.end + 20).min(char_count);

                // Extract context using character offsets (not byte offsets)
                let context: String = text
                    .chars()
                    .skip(context_start)
                    .take(context_end.saturating_sub(context_start))
                    .collect();

                analysis.errors.push(NERError {
                    error_type,
                    predicted: Some(pred.clone()),
                    gold: Some(gold[gold_idx].clone()),
                    context,
                });
                *analysis.counts.entry(error_type).or_insert(0) += 1;
            } else {
                // This prediction didn't match any unmatched gold entity.
                // Check if it would have matched a gold that was already matched
                // by a previous prediction - if so, this is a duplicate/spurious prediction.
                //
                // Note: We check gold_matched[i] to see if the gold was already matched.
                // If a gold was already matched AND this prediction matches it exactly,
                // then this prediction is a duplicate (spurious), not a correct match.
                let _is_duplicate = gold.iter().enumerate().any(|(i, g)| {
                    gold_matched[i]
                        && pred.start == g.start
                        && pred.end == g.end
                        && pred.entity_type == g.entity_type
                });

                // This prediction is spurious (either duplicate or no match)
                // Both duplicates and non-matching predictions are spurious
                let char_count = text.chars().count();
                let context_start = pred.start.saturating_sub(20);
                let context_end = (pred.end + 20).min(char_count);

                // Extract context using character offsets (not byte offsets)
                let context: String = text
                    .chars()
                    .skip(context_start)
                    .take(context_end.saturating_sub(context_start))
                    .collect();

                analysis.errors.push(NERError {
                    error_type: ErrorType::Spurious,
                    predicted: Some(pred.clone()),
                    gold: None,
                    context,
                });
                *analysis.counts.entry(ErrorType::Spurious).or_insert(0) += 1;
            }
        }

        // Check for missed entities
        for (i, g) in gold.iter().enumerate() {
            if !gold_matched[i] {
                let char_count = text.chars().count();
                let context_start = g.start.saturating_sub(20);
                let context_end = (g.end + 20).min(char_count);

                // Extract context using character offsets (not byte offsets)
                let context: String = text
                    .chars()
                    .skip(context_start)
                    .take(context_end.saturating_sub(context_start))
                    .collect();

                analysis.errors.push(NERError {
                    error_type: ErrorType::Missed,
                    predicted: None,
                    gold: Some(g.clone()),
                    context,
                });
                *analysis.counts.entry(ErrorType::Missed).or_insert(0) += 1;
            }
        }

        analysis
    }

    /// Get error rate by type.
    #[must_use]
    pub fn error_rate(&self, error_type: ErrorType) -> f64 {
        let count = self.counts.get(&error_type).copied().unwrap_or(0);
        let total = self.total_predictions.max(self.total_gold);
        if total == 0 {
            0.0
        } else {
            count as f64 / total as f64
        }
    }

    /// Summary of error distribution.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "Error Analysis ({} predictions, {} gold):\n",
            self.total_predictions, self.total_gold
        ));

        for error_type in [
            ErrorType::TypeMismatch,
            ErrorType::BoundaryError,
            ErrorType::BoundaryAndType,
            ErrorType::Spurious,
            ErrorType::Missed,
        ] {
            let count = self.counts.get(&error_type).copied().unwrap_or(0);
            let rate = self.error_rate(error_type) * 100.0;
            s.push_str(&format!(
                "  {:15} {:4} ({:.1}%)\n",
                format!("{:?}", error_type),
                count,
                rate
            ));
        }

        s
    }
}

// =============================================================================
// Statistical Significance for NER
// =============================================================================

/// Result of paired significance test for NER systems.
#[derive(Debug, Clone)]
pub struct NERSignificanceTest {
    /// System A name
    pub system_a: String,
    /// System B name
    pub system_b: String,
    /// System A mean F1
    pub mean_a: f64,
    /// System B mean F1
    pub mean_b: f64,
    /// Difference (A - B)
    pub difference: f64,
    /// Standard error of difference
    pub std_error: f64,
    /// t-statistic
    pub t_statistic: f64,
    /// p-value (two-tailed)
    pub p_value: f64,
    /// Number of test cases
    pub n: usize,
    /// Significant at p < 0.05?
    pub significant_05: bool,
    /// Significant at p < 0.01?
    pub significant_01: bool,
}

impl NERSignificanceTest {
    /// Perform paired t-test on F1 scores.
    #[must_use]
    pub fn paired_t_test(
        system_a: &str,
        scores_a: &[f64],
        system_b: &str,
        scores_b: &[f64],
    ) -> Self {
        assert_eq!(
            scores_a.len(),
            scores_b.len(),
            "Scores must have same length"
        );
        let n = scores_a.len();

        if n < 2 {
            return Self {
                system_a: system_a.to_string(),
                system_b: system_b.to_string(),
                mean_a: scores_a.first().copied().unwrap_or(0.0),
                mean_b: scores_b.first().copied().unwrap_or(0.0),
                difference: 0.0,
                std_error: 0.0,
                t_statistic: 0.0,
                p_value: 1.0,
                n,
                significant_05: false,
                significant_01: false,
            };
        }

        let differences: Vec<f64> = scores_a
            .iter()
            .zip(scores_b.iter())
            .map(|(a, b)| a - b)
            .collect();

        let mean_diff = differences.iter().sum::<f64>() / n as f64;
        let mean_a = scores_a.iter().sum::<f64>() / n as f64;
        let mean_b = scores_b.iter().sum::<f64>() / n as f64;

        let variance: f64 = differences
            .iter()
            .map(|&d| (d - mean_diff).powi(2))
            .sum::<f64>()
            / (n - 1) as f64;
        let std_diff = variance.sqrt();
        let std_error = std_diff / (n as f64).sqrt();

        let t_stat = if std_error > 0.0 {
            mean_diff / std_error
        } else {
            0.0
        };

        // Approximate p-value
        let p_value = Self::approximate_p_value(t_stat.abs(), n - 1);

        Self {
            system_a: system_a.to_string(),
            system_b: system_b.to_string(),
            mean_a,
            mean_b,
            difference: mean_diff,
            std_error,
            t_statistic: t_stat,
            p_value,
            n,
            significant_05: p_value < 0.05,
            significant_01: p_value < 0.01,
        }
    }

    fn approximate_p_value(t: f64, df: usize) -> f64 {
        let critical_05 = if df >= 30 { 1.96 } else { 2.1 };
        let critical_01 = if df >= 30 { 2.576 } else { 2.9 };

        if t < critical_05 {
            0.10
        } else if t < critical_01 {
            0.03
        } else {
            0.005
        }
    }
}

impl std::fmt::Display for NERSignificanceTest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Paired t-test (n={}):", self.n)?;
        writeln!(f, "  {}: {:.1}%", self.system_a, self.mean_a * 100.0)?;
        writeln!(f, "  {}: {:.1}%", self.system_b, self.mean_b * 100.0)?;
        writeln!(f, "  Difference: {:+.1}%", self.difference * 100.0)?;
        writeln!(f, "  t={:.3}, p={:.4}", self.t_statistic, self.p_value)?;

        let sig = if self.significant_01 {
            "** (p < 0.01)"
        } else if self.significant_05 {
            "* (p < 0.05)"
        } else {
            "not significant"
        };
        writeln!(f, "  {}", sig)?;

        Ok(())
    }
}

/// Compare two NER systems with significance testing.
#[must_use]
pub fn compare_ner_systems(
    system_a: &str,
    f1_scores_a: &[f64],
    system_b: &str,
    f1_scores_b: &[f64],
) -> NERSignificanceTest {
    NERSignificanceTest::paired_t_test(system_a, f1_scores_a, system_b, f1_scores_b)
}

/// Build confusion matrix from predictions and ground truth.
#[must_use]
pub fn build_confusion_matrix(predictions: &[(Vec<Entity>, Vec<GoldEntity>)]) -> ConfusionMatrix {
    let mut matrix = ConfusionMatrix::new();

    for (preds, golds) in predictions {
        let mut gold_matched = vec![false; golds.len()];

        for pred in preds {
            let pred_type = pred.entity_type.as_label().to_string();

            // Find best matching gold entity
            for (i, gold) in golds.iter().enumerate() {
                if gold_matched[i] {
                    continue;
                }

                // Check for overlap
                if pred.start < gold.end && pred.end > gold.start {
                    let gold_type = gold.entity_type.as_label().to_string();
                    matrix.add(&pred_type, &gold_type);
                    gold_matched[i] = true;
                    break;
                }
            }
        }

        // Count missed entities
        for (i, gold) in golds.iter().enumerate() {
            if !gold_matched[i] {
                let gold_type = gold.entity_type.as_label().to_string();
                matrix.add("MISSED", &gold_type);
            }
        }
    }

    matrix
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use anno_core::EntityType;

    #[test]
    fn test_confusion_matrix() {
        let mut cm = ConfusionMatrix::new();
        cm.add("PER", "PER");
        cm.add("PER", "PER");
        cm.add("PER", "ORG"); // Confusion
        cm.add("ORG", "ORG");

        assert_eq!(cm.get("PER", "PER"), 2);
        assert_eq!(cm.get("PER", "ORG"), 1);
        assert_eq!(cm.get("ORG", "ORG"), 1);

        // Precision for PER: 2/3
        assert!((cm.precision("PER") - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_most_confused() {
        let mut cm = ConfusionMatrix::new();
        cm.add("PER", "ORG");
        cm.add("PER", "ORG");
        cm.add("LOC", "ORG");

        let confused = cm.most_confused(2);
        assert_eq!(confused.len(), 2);
        assert_eq!(confused[0], ("PER".to_string(), "ORG".to_string(), 2));
    }

    #[test]
    fn test_significance_test() {
        let scores_a = vec![0.85, 0.82, 0.88, 0.79, 0.84];
        let scores_b = vec![0.78, 0.76, 0.82, 0.74, 0.79];

        let test = compare_ner_systems("A", &scores_a, "B", &scores_b);

        assert!(test.mean_a > test.mean_b);
        assert!(test.difference > 0.0);
    }

    #[test]
    fn test_error_analysis() {
        let text = "John Smith works at Google in New York.";

        // Note: "Microsoft" (20-29) overlaps with "Google" (20-26)
        // Both are Organization, so this is a BoundaryError
        let predicted = vec![
            Entity::new("John Smith", EntityType::Person, 0, 10, 0.9),
            Entity::new("Microsoft", EntityType::Organization, 20, 29, 0.8),
        ];

        let gold = vec![
            GoldEntity::new("John Smith", EntityType::Person, 0),
            GoldEntity::new("Google", EntityType::Organization, 20),
        ];

        let analysis = ErrorAnalysis::analyze(text, &predicted, &gold);

        // "Microsoft" overlaps with "Google" (same type) -> BoundaryError (not Spurious)
        assert_eq!(
            analysis
                .counts
                .get(&ErrorType::BoundaryError)
                .copied()
                .unwrap_or(0),
            1,
            "Expected boundary error for Microsoft/Google overlap"
        );
        // John Smith matches exactly, so no errors from that
        assert_eq!(
            analysis
                .counts
                .get(&ErrorType::TypeMismatch)
                .copied()
                .unwrap_or(0),
            0,
            "Expected no type mismatches"
        );
    }

    #[test]
    fn test_significance_equal_systems() {
        // Systems with same performance
        let scores = vec![0.80, 0.81, 0.79, 0.80, 0.80];
        let test = compare_ner_systems("A", &scores, "B", &scores);

        assert!((test.difference).abs() < 0.001);
        assert!(!test.significant_05);
    }

    #[test]
    fn test_confusion_matrix_display() {
        let mut cm = ConfusionMatrix::new();
        cm.add("PER", "PER");
        cm.add("ORG", "ORG");
        cm.add("LOC", "LOC");

        let display = format!("{}", cm);
        assert!(display.contains("PER"));
        assert!(display.contains("ORG"));
        assert!(display.contains("LOC"));
    }

    #[test]
    fn test_error_type_mismatch() {
        let text = "Test Person here.";

        let predicted = vec![
            Entity::new("Person", EntityType::Organization, 5, 11, 0.9), // Wrong type
        ];

        let gold = vec![GoldEntity::new("Person", EntityType::Person, 5)];

        let analysis = ErrorAnalysis::analyze(text, &predicted, &gold);
        assert_eq!(
            analysis
                .counts
                .get(&ErrorType::TypeMismatch)
                .copied()
                .unwrap_or(0),
            1
        );
    }

    #[test]
    fn test_error_boundary() {
        let text = "Dr. John Smith is here.";

        let predicted = vec![
            // Predicted "John Smith" but gold is "Dr. John Smith"
            Entity::new("John Smith", EntityType::Person, 4, 14, 0.9),
        ];

        let gold = vec![GoldEntity::new("Dr. John Smith", EntityType::Person, 0)];

        let analysis = ErrorAnalysis::analyze(text, &predicted, &gold);
        assert_eq!(
            analysis
                .counts
                .get(&ErrorType::BoundaryError)
                .copied()
                .unwrap_or(0),
            1
        );
    }

    #[test]
    fn test_perfect_match_no_errors() {
        let text = "John Smith works here.";

        let predicted = vec![Entity::new("John Smith", EntityType::Person, 0, 10, 0.9)];

        let gold = vec![GoldEntity::new("John Smith", EntityType::Person, 0)];

        let analysis = ErrorAnalysis::analyze(text, &predicted, &gold);

        // All error counts should be 0
        assert_eq!(
            analysis
                .counts
                .get(&ErrorType::TypeMismatch)
                .copied()
                .unwrap_or(0),
            0
        );
        assert_eq!(
            analysis
                .counts
                .get(&ErrorType::BoundaryError)
                .copied()
                .unwrap_or(0),
            0
        );
        assert_eq!(
            analysis
                .counts
                .get(&ErrorType::Spurious)
                .copied()
                .unwrap_or(0),
            0
        );
        assert_eq!(
            analysis
                .counts
                .get(&ErrorType::Missed)
                .copied()
                .unwrap_or(0),
            0
        );
    }

    #[test]
    fn test_recall_precision_from_confusion() {
        let mut cm = ConfusionMatrix::new();

        // 10 correct PER predictions
        for _ in 0..10 {
            cm.add("PER", "PER");
        }
        // 2 PER predicted as ORG
        cm.add("ORG", "PER");
        cm.add("ORG", "PER");

        // Recall for PER: 10 / 12 = 0.833
        assert!((cm.recall("PER") - 10.0 / 12.0).abs() < 0.01);

        // Precision for PER: 10 / 10 = 1.0
        assert!((cm.precision("PER") - 1.0).abs() < 0.01);
    }
}
