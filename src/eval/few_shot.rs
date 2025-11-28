//! Few-shot learning evaluation for NER.
//!
//! Measures how well models can recognize entities with minimal examples.
//! Critical for practitioners who need to quickly adapt to new domains.
//!
//! # Example
//!
//! ```rust
//! use anno::eval::few_shot::{FewShotEvaluator, FewShotTask, SupportExample};
//!
//! // Create support set (few examples per entity type)
//! let task = FewShotTask {
//!     entity_type: "DISEASE".into(),
//!     support: vec![
//!         SupportExample::new("Patient has diabetes", "diabetes", 12, 20),
//!         SupportExample::new("Diagnosed with cancer", "cancer", 15, 21),
//!     ],
//!     query_texts: vec![
//!         "The patient presented with pneumonia".into(),
//!         "History of hypertension noted".into(),
//!     ],
//! };
//!
//! let evaluator = FewShotEvaluator::default();
//! // In practice, you'd run a model and evaluate its predictions
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Data Structures
// =============================================================================

/// A single example in the support set.
#[derive(Debug, Clone)]
pub struct SupportExample {
    /// Full text containing the entity
    pub text: String,
    /// Entity text
    pub entity_text: String,
    /// Start offset
    pub start: usize,
    /// End offset
    pub end: usize,
}

impl SupportExample {
    /// Create a new support example.
    pub fn new(text: impl Into<String>, entity_text: impl Into<String>, start: usize, end: usize) -> Self {
        Self {
            text: text.into(),
            entity_text: entity_text.into(),
            start,
            end,
        }
    }
}

/// A few-shot learning task for a single entity type.
#[derive(Debug, Clone)]
pub struct FewShotTask {
    /// The entity type to recognize
    pub entity_type: String,
    /// Support set: K examples showing the entity type
    pub support: Vec<SupportExample>,
    /// Query texts to evaluate
    pub query_texts: Vec<String>,
}

/// Gold annotation for a query in few-shot task.
#[derive(Debug, Clone)]
pub struct FewShotGold {
    /// Text being annotated
    pub text: String,
    /// Entity spans in the text
    pub entities: Vec<(String, usize, usize)>,  // (entity_text, start, end)
}

/// Model prediction for few-shot evaluation.
#[derive(Debug, Clone)]
pub struct FewShotPrediction {
    /// Text being annotated
    pub text: String,
    /// Predicted entity spans
    pub predicted: Vec<(String, usize, usize, f64)>,  // (entity_text, start, end, confidence)
}

/// Results for a single few-shot task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FewShotTaskResults {
    /// Entity type being evaluated
    pub entity_type: String,
    /// Number of examples in support set (K)
    pub k: usize,
    /// Precision on query set
    pub precision: f64,
    /// Recall on query set
    pub recall: f64,
    /// F1 score
    pub f1: f64,
    /// Number of gold entities in query set
    pub num_gold: usize,
    /// Number of predicted entities
    pub num_predicted: usize,
    /// Number of correct predictions
    pub num_correct: usize,
}

/// Overall few-shot evaluation results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FewShotResults {
    /// Results per entity type
    pub per_type: HashMap<String, FewShotTaskResults>,
    /// Macro-averaged F1 across types
    pub macro_f1: f64,
    /// Micro-averaged F1 (total correct / total predicted)
    pub micro_f1: f64,
    /// K values tested
    pub k_values: Vec<usize>,
    /// Performance by K (average F1 for each K)
    pub performance_by_k: Vec<(usize, f64)>,
    /// Types that failed (F1 < 0.1)
    pub failed_types: Vec<String>,
    /// Insights and recommendations
    pub insights: Vec<String>,
}

// =============================================================================
// Few-Shot Evaluator
// =============================================================================

/// Evaluator for few-shot NER learning.
#[derive(Debug, Clone)]
pub struct FewShotEvaluator {
    /// Minimum K values to test
    pub k_values: Vec<usize>,
    /// Minimum F1 to consider "successful"
    pub success_threshold: f64,
}

impl Default for FewShotEvaluator {
    fn default() -> Self {
        Self {
            k_values: vec![1, 2, 5, 10],
            success_threshold: 0.5,
        }
    }
}

impl FewShotEvaluator {
    /// Create evaluator with custom K values.
    pub fn new(k_values: Vec<usize>) -> Self {
        Self {
            k_values,
            success_threshold: 0.5,
        }
    }

    /// Evaluate few-shot predictions against gold annotations.
    pub fn evaluate(
        &self,
        entity_type: &str,
        k: usize,
        predictions: &[FewShotPrediction],
        gold: &[FewShotGold],
    ) -> FewShotTaskResults {
        assert_eq!(predictions.len(), gold.len(), "Predictions and gold must have same length");

        let mut total_correct = 0;
        let mut total_predicted = 0;
        let mut total_gold = 0;

        for (pred, g) in predictions.iter().zip(gold.iter()) {
            total_gold += g.entities.len();
            total_predicted += pred.predicted.len();

            // Count matches (exact span match)
            for (g_text, g_start, g_end) in &g.entities {
                for (p_text, p_start, p_end, _conf) in &pred.predicted {
                    if g_start == p_start && g_end == p_end {
                        total_correct += 1;
                        break;
                    }
                    // Also allow text match if spans differ slightly
                    if g_text.to_lowercase() == p_text.to_lowercase() {
                        total_correct += 1;
                        break;
                    }
                }
            }
        }

        let precision = if total_predicted == 0 {
            1.0
        } else {
            total_correct as f64 / total_predicted as f64
        };

        let recall = if total_gold == 0 {
            1.0
        } else {
            total_correct as f64 / total_gold as f64
        };

        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };

        FewShotTaskResults {
            entity_type: entity_type.to_string(),
            k,
            precision,
            recall,
            f1,
            num_gold: total_gold,
            num_predicted: total_predicted,
            num_correct: total_correct,
        }
    }

    /// Aggregate results across multiple entity types.
    pub fn aggregate(&self, results: Vec<FewShotTaskResults>) -> FewShotResults {
        let mut per_type: HashMap<String, FewShotTaskResults> = HashMap::new();
        let mut by_k: HashMap<usize, Vec<f64>> = HashMap::new();

        for r in &results {
            per_type.insert(r.entity_type.clone(), r.clone());
            by_k.entry(r.k).or_default().push(r.f1);
        }

        // Compute macro F1
        let macro_f1 = if results.is_empty() {
            0.0
        } else {
            results.iter().map(|r| r.f1).sum::<f64>() / results.len() as f64
        };

        // Compute micro F1
        let total_correct: usize = results.iter().map(|r| r.num_correct).sum();
        let total_predicted: usize = results.iter().map(|r| r.num_predicted).sum();
        let total_gold: usize = results.iter().map(|r| r.num_gold).sum();

        let micro_precision = if total_predicted == 0 { 1.0 } else { total_correct as f64 / total_predicted as f64 };
        let micro_recall = if total_gold == 0 { 1.0 } else { total_correct as f64 / total_gold as f64 };
        let micro_f1 = if micro_precision + micro_recall == 0.0 {
            0.0
        } else {
            2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
        };

        // Performance by K
        let mut performance_by_k: Vec<_> = by_k
            .iter()
            .map(|(k, scores)| (*k, scores.iter().sum::<f64>() / scores.len() as f64))
            .collect();
        performance_by_k.sort_by_key(|(k, _)| *k);

        // Find failed types
        let failed_types: Vec<_> = results
            .iter()
            .filter(|r| r.f1 < self.success_threshold)
            .map(|r| r.entity_type.clone())
            .collect();

        // Generate insights
        let mut insights = Vec::new();

        if !performance_by_k.is_empty() {
            let min_k_f1 = performance_by_k.first().map(|(_, f1)| *f1).unwrap_or(0.0);
            let max_k_f1 = performance_by_k.last().map(|(_, f1)| *f1).unwrap_or(0.0);
            let improvement = max_k_f1 - min_k_f1;

            if improvement > 0.2 {
                insights.push(format!(
                    "Strong learning: +{:.0}% F1 from K=1 to K={}",
                    improvement * 100.0,
                    performance_by_k.last().map(|(k, _)| *k).unwrap_or(10)
                ));
            } else if improvement < 0.05 {
                insights.push("Minimal improvement with more examples - may need different approach".into());
            }
        }

        if !failed_types.is_empty() {
            insights.push(format!(
                "Struggling with {} entity types: {:?}",
                failed_types.len(),
                &failed_types[..failed_types.len().min(3)]
            ));
        }

        if macro_f1 < 0.3 {
            insights.push("Low overall few-shot performance - consider pre-training on related data".into());
        }

        FewShotResults {
            per_type,
            macro_f1,
            micro_f1,
            k_values: self.k_values.clone(),
            performance_by_k,
            failed_types,
            insights,
        }
    }
}

// =============================================================================
// Simulation Utilities
// =============================================================================

/// Create a simulated few-shot task from existing annotated data.
///
/// Takes a dataset and creates K support examples + M query examples.
pub fn simulate_few_shot_task(
    entity_type: &str,
    all_examples: &[(String, Vec<(String, usize, usize)>)],  // (text, [(entity_text, start, end)])
    k: usize,
    max_queries: usize,
) -> Option<(FewShotTask, Vec<FewShotGold>)> {
    // Filter examples containing this entity type
    let mut matching: Vec<_> = all_examples
        .iter()
        .filter(|(_, entities)| !entities.is_empty())
        .cloned()
        .collect();

    if matching.len() < k + 1 {
        return None; // Not enough examples
    }

    // Split into support (first K) and query (rest)
    let support: Vec<_> = matching
        .drain(..k)
        .map(|(text, entities)| {
            let (entity_text, start, end) = entities.first().unwrap();
            SupportExample::new(text, entity_text.clone(), *start, *end)
        })
        .collect();

    let query_count = matching.len().min(max_queries);
    let queries: Vec<_> = matching[..query_count].to_vec();

    let task = FewShotTask {
        entity_type: entity_type.to_string(),
        support,
        query_texts: queries.iter().map(|(t, _)| t.clone()).collect(),
    };

    let gold: Vec<_> = queries
        .iter()
        .map(|(text, entities)| FewShotGold {
            text: text.clone(),
            entities: entities.clone(),
        })
        .collect();

    Some((task, gold))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_predictions() {
        let evaluator = FewShotEvaluator::default();

        let predictions = vec![
            FewShotPrediction {
                text: "Has diabetes".into(),
                predicted: vec![("diabetes".into(), 4, 12, 0.95)],
            },
        ];

        let gold = vec![
            FewShotGold {
                text: "Has diabetes".into(),
                entities: vec![("diabetes".into(), 4, 12)],
            },
        ];

        let results = evaluator.evaluate("DISEASE", 2, &predictions, &gold);
        assert!((results.f1 - 1.0).abs() < 0.01);
        assert_eq!(results.num_correct, 1);
    }

    #[test]
    fn test_no_predictions() {
        let evaluator = FewShotEvaluator::default();

        let predictions = vec![
            FewShotPrediction {
                text: "Has diabetes".into(),
                predicted: vec![],
            },
        ];

        let gold = vec![
            FewShotGold {
                text: "Has diabetes".into(),
                entities: vec![("diabetes".into(), 4, 12)],
            },
        ];

        let results = evaluator.evaluate("DISEASE", 2, &predictions, &gold);
        assert!((results.recall).abs() < 0.01);
        assert_eq!(results.num_correct, 0);
    }

    #[test]
    fn test_aggregate_results() {
        let evaluator = FewShotEvaluator::default();

        let results = vec![
            FewShotTaskResults {
                entity_type: "PER".into(),
                k: 2,
                precision: 0.8,
                recall: 0.7,
                f1: 0.75,
                num_gold: 10,
                num_predicted: 8,
                num_correct: 7,
            },
            FewShotTaskResults {
                entity_type: "ORG".into(),
                k: 2,
                precision: 0.6,
                recall: 0.5,
                f1: 0.55,
                num_gold: 10,
                num_predicted: 9,
                num_correct: 5,
            },
        ];

        let aggregated = evaluator.aggregate(results);
        assert!((aggregated.macro_f1 - 0.65).abs() < 0.01);
        assert_eq!(aggregated.per_type.len(), 2);
    }

    #[test]
    fn test_failed_types_detection() {
        let evaluator = FewShotEvaluator::default();

        let results = vec![
            FewShotTaskResults {
                entity_type: "EASY".into(),
                k: 5,
                precision: 0.9,
                recall: 0.85,
                f1: 0.87,
                num_gold: 10,
                num_predicted: 10,
                num_correct: 9,
            },
            FewShotTaskResults {
                entity_type: "HARD".into(),
                k: 5,
                precision: 0.2,
                recall: 0.1,
                f1: 0.13,
                num_gold: 10,
                num_predicted: 5,
                num_correct: 1,
            },
        ];

        let aggregated = evaluator.aggregate(results);
        assert!(aggregated.failed_types.contains(&"HARD".to_string()));
        assert!(!aggregated.failed_types.contains(&"EASY".to_string()));
    }
}

