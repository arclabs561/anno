//! Ensemble disagreement metrics for multi-model NER systems.
//!
//! When running multiple NER models, disagreement patterns reveal:
//! - Ambiguous or difficult examples
//! - Model-specific biases
//! - Opportunities for ensemble voting
//!
//! # Key Metrics
//!
//! - **Agreement Rate**: How often do models agree?
//! - **Fleiss' Kappa**: Inter-rater reliability adjusted for chance
//! - **Disagreement Entropy**: Information content of disagreements
//! - **Per-Entity-Type Agreement**: Where do models disagree most?
//!
//! # Example
//!
//! ```rust
//! use anno::eval::ensemble::{EnsembleAnalyzer, ModelPrediction};
//!
//! let predictions = vec![
//!     ModelPrediction {
//!         model_name: "model_a".into(),
//!         entities: vec![("John", "PER"), ("Google", "ORG")],
//!     },
//!     ModelPrediction {
//!         model_name: "model_b".into(),
//!         entities: vec![("John", "PER"), ("Google", "LOC")],  // Disagrees on Google
//!     },
//! ];
//!
//! let analyzer = EnsembleAnalyzer::default();
//! let results = analyzer.analyze_single(&predictions);
//! println!("Agreement rate: {:.1}%", results.agreement_rate * 100.0);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// =============================================================================
// Data Structures
// =============================================================================

/// A single model's predictions for one text.
#[derive(Debug, Clone)]
pub struct ModelPrediction {
    /// Model identifier
    pub model_name: String,
    /// Predicted entities as (text, type) pairs
    pub entities: Vec<(String, String)>,
}

/// Results of ensemble analysis for a single example.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingleExampleAnalysis {
    /// Proportion of entity predictions where all models agree
    pub agreement_rate: f64,
    /// Entities where all models agree
    pub agreed_entities: Vec<(String, String)>,
    /// Entities with disagreement: (text, model -> type)
    pub disagreed_entities: Vec<DisagreementDetail>,
    /// Number of models that participated
    pub num_models: usize,
}

/// Details about a disagreement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisagreementDetail {
    /// Entity text
    pub text: String,
    /// Model name -> predicted type (None if model didn't predict this entity)
    pub predictions: HashMap<String, Option<String>>,
    /// Most common prediction (majority vote)
    pub majority_vote: Option<String>,
    /// Confidence in majority vote (proportion of models agreeing)
    pub majority_confidence: f64,
}

/// Results of ensemble analysis across multiple examples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleAnalysisResults {
    /// Overall agreement rate across all examples
    pub overall_agreement_rate: f64,
    /// Fleiss' Kappa (inter-rater reliability)
    pub fleiss_kappa: f64,
    /// Per-entity-type agreement rates
    pub agreement_by_type: HashMap<String, f64>,
    /// Most disagreed entity types (sorted by disagreement rate)
    pub most_disagreed_types: Vec<(String, f64)>,
    /// Sample disagreements for inspection
    pub sample_disagreements: Vec<DisagreementDetail>,
    /// Total examples analyzed
    pub total_examples: usize,
    /// Total entities analyzed
    pub total_entities: usize,
    /// Model-pair agreement matrix
    pub pairwise_agreement: HashMap<String, HashMap<String, f64>>,
}

// =============================================================================
// Ensemble Analyzer
// =============================================================================

/// Analyzer for ensemble model disagreements.
#[derive(Debug, Clone, Default)]
pub struct EnsembleAnalyzer {
    /// Maximum sample disagreements to keep
    pub max_samples: usize,
}

impl EnsembleAnalyzer {
    /// Create analyzer with custom sample limit.
    pub fn new(max_samples: usize) -> Self {
        Self { max_samples }
    }

    /// Analyze disagreements for a single example.
    pub fn analyze_single(&self, predictions: &[ModelPrediction]) -> SingleExampleAnalysis {
        if predictions.is_empty() {
            return SingleExampleAnalysis {
                agreement_rate: 1.0,
                agreed_entities: Vec::new(),
                disagreed_entities: Vec::new(),
                num_models: 0,
            };
        }

        // Collect all unique entity texts
        let all_texts: HashSet<String> = predictions
            .iter()
            .flat_map(|p| p.entities.iter().map(|(t, _)| t.to_lowercase()))
            .collect();

        let mut agreed = Vec::new();
        let mut disagreed = Vec::new();

        for text in all_texts {
            // Collect predictions for this entity
            let mut entity_predictions: HashMap<String, Option<String>> = HashMap::new();

            for pred in predictions {
                let model_pred = pred
                    .entities
                    .iter()
                    .find(|(t, _)| t.to_lowercase() == text)
                    .map(|(_, typ)| typ.clone());

                entity_predictions.insert(pred.model_name.clone(), model_pred);
            }

            // Check for agreement
            let non_none_types: Vec<&String> = entity_predictions
                .values()
                .filter_map(|v| v.as_ref())
                .collect();

            if non_none_types.is_empty() {
                continue;
            }

            let first_type = non_none_types[0];
            let all_agree = non_none_types.iter().all(|t| *t == first_type)
                && entity_predictions.values().all(|v| v.is_some());

            if all_agree {
                agreed.push((text.clone(), first_type.clone()));
            } else {
                // Count types for majority vote
                let mut type_counts: HashMap<String, usize> = HashMap::new();
                for typ in &non_none_types {
                    *type_counts.entry((*typ).clone()).or_insert(0) += 1;
                }

                let (majority_type, majority_count) = type_counts
                    .iter()
                    .max_by_key(|(_, count)| *count)
                    .map(|(t, c)| (Some(t.clone()), *c))
                    .unwrap_or((None, 0));

                let majority_confidence =
                    majority_count as f64 / predictions.len() as f64;

                disagreed.push(DisagreementDetail {
                    text: text.clone(),
                    predictions: entity_predictions,
                    majority_vote: majority_type,
                    majority_confidence,
                });
            }
        }

        let total = agreed.len() + disagreed.len();
        let agreement_rate = if total == 0 {
            1.0
        } else {
            agreed.len() as f64 / total as f64
        };

        SingleExampleAnalysis {
            agreement_rate,
            agreed_entities: agreed,
            disagreed_entities: disagreed,
            num_models: predictions.len(),
        }
    }

    /// Analyze disagreements across multiple examples.
    pub fn analyze_batch(
        &self,
        batch: &[Vec<ModelPrediction>],
    ) -> EnsembleAnalysisResults {
        if batch.is_empty() {
            return EnsembleAnalysisResults {
                overall_agreement_rate: 1.0,
                fleiss_kappa: 1.0,
                agreement_by_type: HashMap::new(),
                most_disagreed_types: Vec::new(),
                sample_disagreements: Vec::new(),
                total_examples: 0,
                total_entities: 0,
                pairwise_agreement: HashMap::new(),
            };
        }

        let mut total_agreed = 0;
        let mut total_entities = 0;
        let mut all_disagreements = Vec::new();
        let mut type_agreed: HashMap<String, usize> = HashMap::new();
        let mut type_total: HashMap<String, usize> = HashMap::new();

        // Model names for pairwise analysis
        let model_names: Vec<String> = batch
            .first()
            .map(|preds| preds.iter().map(|p| p.model_name.clone()).collect())
            .unwrap_or_default();

        let mut pairwise_agreed: HashMap<(String, String), usize> = HashMap::new();
        let mut pairwise_total: HashMap<(String, String), usize> = HashMap::new();

        for example_preds in batch {
            let analysis = self.analyze_single(example_preds);

            total_agreed += analysis.agreed_entities.len();
            total_entities += analysis.agreed_entities.len() + analysis.disagreed_entities.len();

            // Track per-type agreement
            for (_, typ) in &analysis.agreed_entities {
                *type_agreed.entry(typ.clone()).or_insert(0) += 1;
                *type_total.entry(typ.clone()).or_insert(0) += 1;
            }

            for disagreement in &analysis.disagreed_entities {
                if let Some(ref majority) = disagreement.majority_vote {
                    *type_total.entry(majority.clone()).or_insert(0) += 1;
                }
                if all_disagreements.len() < self.max_samples.max(20) {
                    all_disagreements.push(disagreement.clone());
                }
            }

            // Pairwise agreement tracking
            for i in 0..model_names.len() {
                for j in (i + 1)..model_names.len() {
                    let key = (model_names[i].clone(), model_names[j].clone());

                    let pred_i = example_preds.iter().find(|p| p.model_name == model_names[i]);
                    let pred_j = example_preds.iter().find(|p| p.model_name == model_names[j]);

                    if let (Some(pi), Some(pj)) = (pred_i, pred_j) {
                        // Count entities where both models agree
                        let entities_i: HashSet<_> = pi.entities.iter().collect();
                        let entities_j: HashSet<_> = pj.entities.iter().collect();

                        let intersection = entities_i.intersection(&entities_j).count();
                        let union = entities_i.union(&entities_j).count();

                        *pairwise_agreed.entry(key.clone()).or_insert(0) += intersection;
                        *pairwise_total.entry(key).or_insert(0) += union;
                    }
                }
            }
        }

        // Compute overall agreement
        let overall_agreement_rate = if total_entities == 0 {
            1.0
        } else {
            total_agreed as f64 / total_entities as f64
        };

        // Compute per-type agreement
        let agreement_by_type: HashMap<String, f64> = type_total
            .iter()
            .map(|(typ, total)| {
                let agreed = type_agreed.get(typ).copied().unwrap_or(0);
                (typ.clone(), agreed as f64 / *total as f64)
            })
            .collect();

        // Sort types by disagreement rate
        let mut most_disagreed: Vec<(String, f64)> = agreement_by_type
            .iter()
            .map(|(t, rate)| (t.clone(), 1.0 - rate))
            .collect();
        most_disagreed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Compute pairwise agreement matrix
        let mut pairwise_agreement: HashMap<String, HashMap<String, f64>> = HashMap::new();
        for ((m1, m2), total) in &pairwise_total {
            let agreed = pairwise_agreed.get(&(m1.clone(), m2.clone())).copied().unwrap_or(0);
            let rate = if *total == 0 { 1.0 } else { agreed as f64 / *total as f64 };

            pairwise_agreement
                .entry(m1.clone())
                .or_default()
                .insert(m2.clone(), rate);
            pairwise_agreement
                .entry(m2.clone())
                .or_default()
                .insert(m1.clone(), rate);
        }

        // Compute Fleiss' Kappa (simplified)
        let fleiss_kappa = self.compute_fleiss_kappa(batch);

        EnsembleAnalysisResults {
            overall_agreement_rate,
            fleiss_kappa,
            agreement_by_type,
            most_disagreed_types: most_disagreed.into_iter().take(10).collect(),
            sample_disagreements: all_disagreements,
            total_examples: batch.len(),
            total_entities,
            pairwise_agreement,
        }
    }

    /// Compute Fleiss' Kappa for inter-rater reliability.
    fn compute_fleiss_kappa(&self, batch: &[Vec<ModelPrediction>]) -> f64 {
        if batch.is_empty() {
            return 1.0;
        }

        // Simplified Fleiss' Kappa computation
        // Treats each entity prediction as a rating

        let mut n_subjects = 0; // Number of entities rated
        let mut p_bar = 0.0; // Average agreement per subject
        let mut category_proportions: HashMap<String, f64> = HashMap::new();
        let mut total_ratings = 0;

        for example_preds in batch {
            if example_preds.is_empty() {
                continue;
            }

            let n_raters = example_preds.len();

            // Collect all entities mentioned
            let all_texts: HashSet<String> = example_preds
                .iter()
                .flat_map(|p| p.entities.iter().map(|(t, _)| t.to_lowercase()))
                .collect();

            for text in all_texts {
                n_subjects += 1;

                // Count ratings per category for this entity
                let mut category_counts: HashMap<String, usize> = HashMap::new();

                for pred in example_preds {
                    if let Some((_, typ)) = pred.entities.iter().find(|(t, _)| t.to_lowercase() == text) {
                        *category_counts.entry(typ.clone()).or_insert(0) += 1;
                        total_ratings += 1;
                        *category_proportions.entry(typ.clone()).or_insert(0.0) += 1.0;
                    }
                }

                // Compute agreement for this entity
                let sum_squared: f64 = category_counts.values().map(|&n| (n * n) as f64).sum();
                let n = n_raters as f64;
                let p_i = (sum_squared - n) / (n * (n - 1.0));
                p_bar += p_i;
            }
        }

        if n_subjects == 0 || total_ratings == 0 {
            return 1.0;
        }

        p_bar /= n_subjects as f64;

        // Compute expected agreement
        let p_e: f64 = category_proportions
            .values()
            .map(|&p| {
                let prop = p / total_ratings as f64;
                prop * prop
            })
            .sum();

        // Fleiss' Kappa
        if (1.0 - p_e).abs() < 1e-10 {
            1.0
        } else {
            (p_bar - p_e) / (1.0 - p_e)
        }
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Grade agreement rate.
pub fn agreement_grade(rate: f64) -> &'static str {
    if rate >= 0.95 {
        "Excellent agreement"
    } else if rate >= 0.85 {
        "Good agreement"
    } else if rate >= 0.70 {
        "Moderate agreement"
    } else if rate >= 0.50 {
        "Fair agreement"
    } else {
        "Poor agreement"
    }
}

/// Interpret Fleiss' Kappa value.
pub fn kappa_interpretation(kappa: f64) -> &'static str {
    if kappa < 0.0 {
        "Less than chance agreement"
    } else if kappa < 0.20 {
        "Slight agreement"
    } else if kappa < 0.40 {
        "Fair agreement"
    } else if kappa < 0.60 {
        "Moderate agreement"
    } else if kappa < 0.80 {
        "Substantial agreement"
    } else {
        "Almost perfect agreement"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_agreement() {
        let predictions = vec![
            ModelPrediction {
                model_name: "model_a".into(),
                entities: vec![("John".into(), "PER".into()), ("Google".into(), "ORG".into())],
            },
            ModelPrediction {
                model_name: "model_b".into(),
                entities: vec![("John".into(), "PER".into()), ("Google".into(), "ORG".into())],
            },
        ];

        let analyzer = EnsembleAnalyzer::default();
        let results = analyzer.analyze_single(&predictions);

        assert!((results.agreement_rate - 1.0).abs() < 0.01);
        assert_eq!(results.agreed_entities.len(), 2);
        assert!(results.disagreed_entities.is_empty());
    }

    #[test]
    fn test_partial_disagreement() {
        let predictions = vec![
            ModelPrediction {
                model_name: "model_a".into(),
                entities: vec![("John".into(), "PER".into()), ("Google".into(), "ORG".into())],
            },
            ModelPrediction {
                model_name: "model_b".into(),
                entities: vec![("John".into(), "PER".into()), ("Google".into(), "LOC".into())],
            },
        ];

        let analyzer = EnsembleAnalyzer::default();
        let results = analyzer.analyze_single(&predictions);

        assert!((results.agreement_rate - 0.5).abs() < 0.01);
        assert_eq!(results.agreed_entities.len(), 1);
        assert_eq!(results.disagreed_entities.len(), 1);
    }

    #[test]
    fn test_missing_entity() {
        let predictions = vec![
            ModelPrediction {
                model_name: "model_a".into(),
                entities: vec![("John".into(), "PER".into()), ("Google".into(), "ORG".into())],
            },
            ModelPrediction {
                model_name: "model_b".into(),
                entities: vec![("John".into(), "PER".into())], // Missing Google
            },
        ];

        let analyzer = EnsembleAnalyzer::default();
        let results = analyzer.analyze_single(&predictions);

        // Google is disagreed because model_b didn't predict it
        assert_eq!(results.disagreed_entities.len(), 1);
    }

    #[test]
    fn test_batch_analysis() {
        let batch = vec![
            vec![
                ModelPrediction {
                    model_name: "a".into(),
                    entities: vec![("x".into(), "T1".into())],
                },
                ModelPrediction {
                    model_name: "b".into(),
                    entities: vec![("x".into(), "T1".into())],
                },
            ],
            vec![
                ModelPrediction {
                    model_name: "a".into(),
                    entities: vec![("y".into(), "T2".into())],
                },
                ModelPrediction {
                    model_name: "b".into(),
                    entities: vec![("y".into(), "T3".into())],
                },
            ],
        ];

        let analyzer = EnsembleAnalyzer::new(10);
        let results = analyzer.analyze_batch(&batch);

        assert_eq!(results.total_examples, 2);
        assert!(results.overall_agreement_rate > 0.0);
        assert!(results.overall_agreement_rate < 1.0);
    }

    #[test]
    fn test_agreement_grades() {
        assert_eq!(agreement_grade(0.98), "Excellent agreement");
        assert_eq!(agreement_grade(0.90), "Good agreement");
        assert_eq!(agreement_grade(0.75), "Moderate agreement");
        assert_eq!(agreement_grade(0.55), "Fair agreement");
        assert_eq!(agreement_grade(0.30), "Poor agreement");
    }

    #[test]
    fn test_kappa_interpretation() {
        assert_eq!(kappa_interpretation(-0.1), "Less than chance agreement");
        assert_eq!(kappa_interpretation(0.10), "Slight agreement");
        assert_eq!(kappa_interpretation(0.35), "Fair agreement");
        assert_eq!(kappa_interpretation(0.55), "Moderate agreement");
        assert_eq!(kappa_interpretation(0.75), "Substantial agreement");
        assert_eq!(kappa_interpretation(0.90), "Almost perfect agreement");
    }
}

