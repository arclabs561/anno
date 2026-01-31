//! Dataset quality metrics for NER evaluation.
//!
//! Implements metrics from Statistical Dataset Evaluation (Cambridge NLP, 2022):
//! - Reliability: Redundancy, Accuracy, Leakage Ratio
//! - Difficulty: Unseen Entity Ratio, Entity Ambiguity, Model Differentiation
//! - Validity: Entity Imbalance, Entity-Null Rate
//!
//! # Research Background
//!
//! These metrics apply Classical Test Theory (CTT) to NLP datasets:
//! - **Reliability**: Is the dataset trustworthy and consistent?
//! - **Difficulty**: How hard is the task with this dataset?
//! - **Validity**: Does the dataset measure what we intend?
//!
//! # Example
//!
//! ```rust
//! use anno::eval::dataset_quality::{DatasetQualityAnalyzer, QualityReport};
//!
//! let analyzer = DatasetQualityAnalyzer::default();
//!
//! let train_data = vec![
//!     ("John works at Google.", vec![("John", "PER"), ("Google", "ORG")]),
//! ];
//! let test_data = vec![
//!     ("Jane joined Microsoft.", vec![("Jane", "PER"), ("Microsoft", "ORG")]),
//! ];
//!
//! let report = analyzer.analyze(&train_data, &test_data);
//! println!("Leakage ratio: {:.2}%", report.reliability.leakage_ratio * 100.0);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// =============================================================================
// Quality Report Structure
// =============================================================================

/// Comprehensive dataset quality report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReport {
    /// Reliability metrics
    pub reliability: ReliabilityMetrics,
    /// Difficulty metrics
    pub difficulty: DifficultyMetrics,
    /// Validity metrics
    pub validity: ValidityMetrics,
    /// Overall quality grade
    pub overall_grade: String,
    /// Specific recommendations
    pub recommendations: Vec<String>,
}

/// Reliability metrics - dataset trustworthiness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityMetrics {
    /// Proportion of duplicate samples in training data
    pub redundancy: f64,
    /// Number of exact duplicates found
    pub duplicate_count: usize,
    /// Proportion of test samples appearing in training
    pub leakage_ratio: f64,
    /// Number of leaked samples
    pub leaked_count: usize,
}

/// Difficulty metrics - task challenge level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifficultyMetrics {
    /// Proportion of test entities not seen in training
    pub unseen_entity_ratio: f64,
    /// Number of unseen entities
    pub unseen_entity_count: usize,
    /// How often same surface form has different labels
    pub entity_ambiguity: f64,
    /// Ambiguous entity examples
    pub ambiguous_examples: Vec<(String, Vec<String>)>,
    /// Average entity density (entities per 100 tokens)
    pub entity_density: f64,
}

/// Validity metrics - measurement appropriateness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidityMetrics {
    /// Ratio of most common to least common entity type
    pub entity_imbalance: f64,
    /// Entity type distribution
    pub type_distribution: HashMap<String, usize>,
    /// Proportion of tokens that are not entities
    pub entity_null_rate: f64,
    /// Average entities per sample
    pub avg_entities_per_sample: f64,
}

// =============================================================================
// Dataset Quality Analyzer
// =============================================================================

/// Analyzer for dataset quality metrics.
#[derive(Debug, Clone, Default)]
pub struct DatasetQualityAnalyzer {
    /// Minimum samples for statistical validity
    pub min_samples: usize,
}

impl DatasetQualityAnalyzer {
    /// Create analyzer with custom minimum samples.
    pub fn new(min_samples: usize) -> Self {
        Self { min_samples }
    }

    /// Analyze dataset quality.
    ///
    /// # Arguments
    /// - `train_data`: Training samples as (text, [(entity_text, entity_type)])
    /// - `test_data`: Test samples in same format
    pub fn analyze<S: AsRef<str>, T: AsRef<str>>(
        &self,
        train_data: &[(S, Vec<(T, T)>)],
        test_data: &[(S, Vec<(T, T)>)],
    ) -> QualityReport {
        let reliability = self.compute_reliability(train_data, test_data);
        let difficulty = self.compute_difficulty(train_data, test_data);
        let validity = self.compute_validity(train_data);

        let (grade, recommendations) =
            self.compute_grade_and_recommendations(&reliability, &difficulty, &validity);

        QualityReport {
            reliability,
            difficulty,
            validity,
            overall_grade: grade,
            recommendations,
        }
    }

    fn compute_reliability<S: AsRef<str>, T: AsRef<str>>(
        &self,
        train_data: &[(S, Vec<(T, T)>)],
        test_data: &[(S, Vec<(T, T)>)],
    ) -> ReliabilityMetrics {
        // Check for duplicates in training
        let mut seen_texts = HashSet::new();
        let mut duplicate_count = 0;

        for (text, _) in train_data {
            let normalized = text.as_ref().to_lowercase();
            if !seen_texts.insert(normalized) {
                duplicate_count += 1;
            }
        }

        let redundancy = if train_data.is_empty() {
            0.0
        } else {
            duplicate_count as f64 / train_data.len() as f64
        };

        // Check for train-test leakage
        let train_texts: HashSet<String> = train_data
            .iter()
            .map(|(t, _)| t.as_ref().to_lowercase())
            .collect();

        let mut leaked_count = 0;
        for (text, _) in test_data {
            if train_texts.contains(&text.as_ref().to_lowercase()) {
                leaked_count += 1;
            }
        }

        let leakage_ratio = if test_data.is_empty() {
            0.0
        } else {
            leaked_count as f64 / test_data.len() as f64
        };

        ReliabilityMetrics {
            redundancy,
            duplicate_count,
            leakage_ratio,
            leaked_count,
        }
    }

    fn compute_difficulty<S: AsRef<str>, T: AsRef<str>>(
        &self,
        train_data: &[(S, Vec<(T, T)>)],
        test_data: &[(S, Vec<(T, T)>)],
    ) -> DifficultyMetrics {
        // Collect training entities
        let train_entities: HashSet<String> = train_data
            .iter()
            .flat_map(|(_, entities)| entities.iter().map(|(e, _)| e.as_ref().to_lowercase()))
            .collect();

        // Count unseen test entities
        let mut unseen_count = 0;
        let mut total_test_entities = 0;

        for (_, entities) in test_data {
            for (entity, _) in entities {
                total_test_entities += 1;
                if !train_entities.contains(&entity.as_ref().to_lowercase()) {
                    unseen_count += 1;
                }
            }
        }

        let unseen_entity_ratio = if total_test_entities == 0 {
            0.0
        } else {
            unseen_count as f64 / total_test_entities as f64
        };

        // Compute entity ambiguity (same surface form, different labels)
        let mut entity_labels: HashMap<String, HashSet<String>> = HashMap::new();

        for (_, entities) in train_data.iter().chain(test_data.iter()) {
            for (entity, label) in entities {
                entity_labels
                    .entry(entity.as_ref().to_lowercase())
                    .or_default()
                    .insert(label.as_ref().to_string());
            }
        }

        let ambiguous: Vec<_> = entity_labels
            .iter()
            .filter(|(_, labels)| labels.len() > 1)
            .map(|(entity, labels)| (entity.clone(), labels.iter().cloned().collect()))
            .collect();

        let entity_ambiguity = if entity_labels.is_empty() {
            0.0
        } else {
            ambiguous.len() as f64 / entity_labels.len() as f64
        };

        // Compute entity density
        let total_tokens: usize = train_data
            .iter()
            .map(|(t, _)| t.as_ref().split_whitespace().count())
            .sum();

        let total_entities: usize = train_data.iter().map(|(_, e)| e.len()).sum();

        let entity_density = if total_tokens == 0 {
            0.0
        } else {
            (total_entities as f64 / total_tokens as f64) * 100.0
        };

        DifficultyMetrics {
            unseen_entity_ratio,
            unseen_entity_count: unseen_count,
            entity_ambiguity,
            ambiguous_examples: ambiguous.into_iter().take(5).collect(),
            entity_density,
        }
    }

    fn compute_validity<S: AsRef<str>, T: AsRef<str>>(
        &self,
        train_data: &[(S, Vec<(T, T)>)],
    ) -> ValidityMetrics {
        // Count entity types
        let mut type_counts: HashMap<String, usize> = HashMap::new();

        for (_, entities) in train_data {
            for (_, label) in entities {
                *type_counts.entry(label.as_ref().to_string()).or_insert(0) += 1;
            }
        }

        let (max_count, min_count) = if type_counts.is_empty() {
            (0, 0)
        } else {
            let counts: Vec<_> = type_counts.values().copied().collect();
            (
                *counts.iter().max().unwrap_or(&0),
                *counts.iter().min().unwrap_or(&0),
            )
        };

        let entity_imbalance = if min_count == 0 {
            f64::INFINITY
        } else {
            max_count as f64 / min_count as f64
        };

        // Compute null rate
        let total_tokens: usize = train_data
            .iter()
            .map(|(t, _)| t.as_ref().split_whitespace().count())
            .sum();

        // Approximate entity tokens (rough estimate)
        let entity_tokens: usize = train_data
            .iter()
            .flat_map(|(_, entities)| {
                entities
                    .iter()
                    .map(|(e, _)| e.as_ref().split_whitespace().count())
            })
            .sum();

        let entity_null_rate = if total_tokens == 0 {
            1.0
        } else {
            1.0 - (entity_tokens as f64 / total_tokens as f64)
        };

        let total_entities: usize = train_data.iter().map(|(_, e)| e.len()).sum();
        let avg_entities_per_sample = if train_data.is_empty() {
            0.0
        } else {
            total_entities as f64 / train_data.len() as f64
        };

        ValidityMetrics {
            entity_imbalance,
            type_distribution: type_counts,
            entity_null_rate,
            avg_entities_per_sample,
        }
    }

    fn compute_grade_and_recommendations(
        &self,
        reliability: &ReliabilityMetrics,
        difficulty: &DifficultyMetrics,
        validity: &ValidityMetrics,
    ) -> (String, Vec<String>) {
        let mut issues = Vec::new();
        let mut score = 100;

        // Check reliability issues
        if reliability.redundancy > 0.1 {
            issues.push(format!(
                "High redundancy ({:.1}%): Remove duplicates from training data",
                reliability.redundancy * 100.0
            ));
            score -= 15;
        }
        if reliability.leakage_ratio > 0.01 {
            issues.push(format!(
                "Data leakage detected ({:.1}%): {} test samples appear in training",
                reliability.leakage_ratio * 100.0,
                reliability.leaked_count
            ));
            score -= 25;
        }

        // Check difficulty issues
        if difficulty.unseen_entity_ratio > 0.5 {
            issues.push(format!(
                "High unseen entity ratio ({:.1}%): Test set may be too different from training",
                difficulty.unseen_entity_ratio * 100.0
            ));
            score -= 10;
        }
        if difficulty.entity_ambiguity > 0.1 {
            issues.push(format!(
                "Entity ambiguity ({:.1}%): Some entities have multiple labels - review guidelines",
                difficulty.entity_ambiguity * 100.0
            ));
            score -= 10;
        }

        // Check validity issues
        if validity.entity_imbalance > 10.0 {
            issues.push(format!(
                "Severe class imbalance ({:.1}x): Consider oversampling rare entity types",
                validity.entity_imbalance
            ));
            score -= 15;
        }
        if validity.entity_null_rate > 0.95 {
            issues.push(format!(
                "Very sparse entities ({:.1}% null): May need more annotated data",
                validity.entity_null_rate * 100.0
            ));
            score -= 10;
        }

        let grade = match score {
            90..=100 => "A (Excellent)",
            80..=89 => "B (Good)",
            70..=79 => "C (Acceptable)",
            60..=69 => "D (Needs Improvement)",
            _ => "F (Critical Issues)",
        };

        (grade.to_string(), issues)
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Quick check for data leakage between train and test sets.
pub fn check_leakage<S: AsRef<str>>(train_texts: &[S], test_texts: &[S]) -> (usize, f64) {
    let train_set: HashSet<String> = train_texts
        .iter()
        .map(|t| t.as_ref().to_lowercase())
        .collect();

    let leaked = test_texts
        .iter()
        .filter(|t| train_set.contains(&t.as_ref().to_lowercase()))
        .count();

    let ratio = if test_texts.is_empty() {
        0.0
    } else {
        leaked as f64 / test_texts.len() as f64
    };

    (leaked, ratio)
}

/// Compute entity type imbalance ratio.
pub fn entity_imbalance_ratio<S: AsRef<str>>(entity_types: &[S]) -> f64 {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for t in entity_types {
        *counts.entry(t.as_ref()).or_insert(0) += 1;
    }

    if counts.is_empty() {
        return 1.0;
    }

    let max = *counts.values().max().unwrap_or(&0);
    let min = *counts.values().min().unwrap_or(&0);

    if min == 0 {
        f64::INFINITY
    } else {
        max as f64 / min as f64
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_redundancy_detection() {
        let train: Vec<(&str, Vec<(&str, &str)>)> = vec![
            ("John works at Google.", vec![("John", "PER")]),
            ("John works at Google.", vec![("John", "PER")]), // Duplicate
            ("Jane joined Microsoft.", vec![("Jane", "PER")]),
        ];
        let test: Vec<(&str, Vec<(&str, &str)>)> = vec![];

        let analyzer = DatasetQualityAnalyzer::default();
        let report = analyzer.analyze(&train, &test);

        assert_eq!(report.reliability.duplicate_count, 1);
        assert!(report.reliability.redundancy > 0.0);
    }

    #[test]
    fn test_leakage_detection() {
        let train: Vec<(&str, Vec<(&str, &str)>)> =
            vec![("John works at Google.", vec![("John", "PER")])];
        let test: Vec<(&str, Vec<(&str, &str)>)> = vec![
            ("John works at Google.", vec![("John", "PER")]), // Leaked!
            ("Jane joined Microsoft.", vec![("Jane", "PER")]),
        ];

        let analyzer = DatasetQualityAnalyzer::default();
        let report = analyzer.analyze(&train, &test);

        assert_eq!(report.reliability.leaked_count, 1);
        assert!((report.reliability.leakage_ratio - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_unseen_entity_ratio() {
        let train: Vec<(&str, Vec<(&str, &str)>)> = vec![(
            "John works at Google.",
            vec![("John", "PER"), ("Google", "ORG")],
        )];
        let test: Vec<(&str, Vec<(&str, &str)>)> = vec![(
            "Jane joined Microsoft.",
            vec![("Jane", "PER"), ("Microsoft", "ORG")],
        )];

        let analyzer = DatasetQualityAnalyzer::default();
        let report = analyzer.analyze(&train, &test);

        // Both test entities are unseen
        assert_eq!(report.difficulty.unseen_entity_count, 2);
        assert!((report.difficulty.unseen_entity_ratio - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_entity_ambiguity() {
        let train: Vec<(&str, Vec<(&str, &str)>)> = vec![
            ("Washington is a state.", vec![("Washington", "LOC")]),
            ("Washington was president.", vec![("Washington", "PER")]), // Same entity, different label
        ];
        let test: Vec<(&str, Vec<(&str, &str)>)> = vec![];

        let analyzer = DatasetQualityAnalyzer::default();
        let report = analyzer.analyze(&train, &test);

        assert!(report.difficulty.entity_ambiguity > 0.0);
        assert!(!report.difficulty.ambiguous_examples.is_empty());
    }

    #[test]
    fn test_entity_imbalance() {
        let train: Vec<(&str, Vec<(&str, &str)>)> = vec![
            ("Text 1", vec![("e1", "PER"), ("e2", "PER"), ("e3", "PER")]),
            ("Text 2", vec![("e4", "ORG")]), // Only 1 ORG vs 3 PER
        ];
        let test: Vec<(&str, Vec<(&str, &str)>)> = vec![];

        let analyzer = DatasetQualityAnalyzer::default();
        let report = analyzer.analyze(&train, &test);

        assert!((report.validity.entity_imbalance - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_quick_leakage_check() {
        let train = vec!["text a", "text b", "text c"];
        let test = vec!["text a", "text d"]; // "text a" is leaked

        let (count, ratio) = check_leakage(&train, &test);
        assert_eq!(count, 1);
        assert!((ratio - 0.5).abs() < 0.01);
    }
}
