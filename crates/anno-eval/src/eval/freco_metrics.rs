//! Evaluation metrics for Framing-divergent Event Coreference (FRECO).
//!
//! Implements metrics from Zhao et al. (EMNLP 2025) for evaluating FRECO systems,
//! covering classification, mining/retrieval, attitude detection, and cross-topic
//! generalization.
//!
//! # Metrics Overview
//!
//! | Metric | Description | Use Case |
//! |--------|-------------|----------|
//! | [`FrecoMetrics`] | P/R/F1/MCC for binary classification | Pair classification |
//! | [`FrecoMiningMetrics`] | P@K, R@K, Average Precision | Candidate retrieval |
//! | [`AttitudeMetrics`] | Accuracy, Macro-F1, Cohen's κ | Attitude classification |
//! | [`CrossTopicEvaluation`] | Leave-one-topic-out F1 | Generalization testing |
//! | [`BootstrapProgress`] | Convergence tracking | Semi-supervised mining |
//!
//! # Evaluation Protocols
//!
//! ## 1. Classification Evaluation
//!
//! Binary classification: given two event mentions, predict whether they form
//! a FRECO pair (coreferent + framing divergence).
//!
//! ```rust
//! use anno::eval::freco_metrics::FrecoMetrics;
//!
//! // Predictions: (model_prediction, gold_label)
//! let predictions = vec![
//!     (true, true),   // True positive
//!     (true, false),  // False positive
//!     (false, true),  // False negative
//!     (false, false), // True negative
//! ];
//!
//! let metrics = FrecoMetrics::from_predictions(predictions.into_iter());
//! assert_eq!(metrics.true_positives, 1);
//! ```
//!
//! ## 2. Mining/Retrieval Evaluation
//!
//! Given a large candidate pool, rank pairs by similarity and retrieve top-K.
//! Evaluated with precision@K and recall@K at various thresholds.
//!
//! ```rust
//! use anno::eval::freco_metrics::FrecoMiningMetrics;
//!
//! // Ranked candidates: (confidence_score, is_gold_positive)
//! let ranked = vec![
//!     (0.95, true),   // rank 1
//!     (0.90, true),   // rank 2
//!     (0.85, false),  // rank 3
//!     (0.80, true),   // rank 4
//!     (0.75, false),  // rank 5
//! ];
//!
//! let metrics = FrecoMiningMetrics::from_ranked(&ranked, &[3, 5]);
//! // P@3 = 2/3, R@3 = 2/3 (of 3 gold positives)
//! ```
//!
//! ## 3. Cross-Topic Generalization
//!
//! Leave-one-topic-out evaluation: train on N-1 topics, test on the held-out topic.
//! Tests whether the model captures generalizable framing patterns vs topic-specific
//! lexical cues.
//!
//! ```text
//! Topics: [al-shifa, jan-6, roe-v-wade, immigration]
//!
//! Fold 1: Train on {jan-6, roe, immigration}, Test on {al-shifa}
//! Fold 2: Train on {al-shifa, roe, immigration}, Test on {jan-6}
//! Fold 3: Train on {al-shifa, jan-6, immigration}, Test on {roe}
//! Fold 4: Train on {al-shifa, jan-6, roe}, Test on {immigration}
//!
//! Report: Mean F1 ± std across folds
//! ```
//!
//! ## 4. Bootstrapped Mining
//!
//! Semi-supervised expansion: start with seed FRECO pairs, iteratively mine
//! high-confidence candidates, retrain, repeat until convergence.
//!
//! Convergence criteria:
//! - Validation loss increases for N consecutive rounds
//! - Jaccard similarity between rounds falls below threshold
//! - Fewer than M new positive pairs found
//!
//! # Key Insight: MCC for Imbalanced Data
//!
//! FRECO pairs are rare (typically 5-15% of candidate pairs). F1 can be misleading;
//! Matthews Correlation Coefficient (MCC) is more informative:
//!
//! ```rust
//! use anno::eval::freco_metrics::FrecoMetrics;
//!
//! let metrics = FrecoMetrics {
//!     true_positives: 10,
//!     false_positives: 5,
//!     true_negatives: 85,
//!     false_negatives: 0,
//!     ..Default::default()
//! };
//!
//! let mcc = metrics.mcc();
//! // MCC accounts for all four quadrants of the confusion matrix
//! assert!(mcc > 0.5); // Good classifier
//! ```
//!
//! # References
//!
//! - Zhao et al. (2025): FRECO task definition and evaluation protocol
//! - Matthews (1975): MCC for binary classification
//! - Cohen (1960): Kappa statistic for agreement

use anno_core::types::{FrecoCorpus, FrecoPair, FramingAttitude};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Classification Metrics
// =============================================================================

/// Core classification metrics for FRECO pair classification.
///
/// Captures the full confusion matrix and derived metrics (P/R/F1/Accuracy)
/// for binary FRECO classification: given two event mentions, predict whether
/// they form a FRECO pair (coreferent events with framing divergence).
///
/// # Confusion Matrix
///
/// ```text
///                      Predicted
///                 Positive    Negative
///             ┌───────────┬───────────┐
///     Positive│    TP     │    FN     │
/// Gold       ├───────────┼───────────┤
///     Negative│    FP     │    TN     │
///             └───────────┴───────────┘
/// ```
///
/// # Construction
///
/// Use [`from_predictions`](Self::from_predictions) for boolean predictions or
/// [`from_scores`](Self::from_scores) for confidence scores with a threshold.
///
/// # Example
///
/// ```rust
/// use anno::eval::freco_metrics::FrecoMetrics;
///
/// // From boolean predictions
/// let preds = vec![(true, true), (true, false), (false, true), (false, false)];
/// let m = FrecoMetrics::from_predictions(preds.into_iter());
///
/// assert_eq!(m.true_positives, 1);
/// assert_eq!(m.false_positives, 1);
/// assert_eq!(m.true_negatives, 1);
/// assert_eq!(m.false_negatives, 1);
/// assert_eq!(m.f1, 0.5);
///
/// // From confidence scores at threshold 0.5
/// let scores = vec![(0.9, true), (0.6, false), (0.4, true), (0.2, false)];
/// let m2 = FrecoMetrics::from_scores(scores.into_iter(), 0.5);
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FrecoMetrics {
    /// Precision: TP / (TP + FP). "Of predicted positives, how many are correct?"
    pub precision: f64,

    /// Recall: TP / (TP + FN). "Of actual positives, how many did we find?"
    pub recall: f64,

    /// F1 score: harmonic mean of precision and recall.
    ///
    /// F1 = 2 * (P * R) / (P + R). Balances precision and recall.
    pub f1: f64,

    /// Accuracy: (TP + TN) / total. "Overall fraction correct."
    ///
    /// Misleading for imbalanced data; prefer [`mcc`](Self::mcc) instead.
    pub accuracy: f64,

    /// True positives: predicted positive, actually positive (correct FRECO pair).
    pub true_positives: usize,

    /// False positives: predicted positive, actually negative (spurious pair).
    pub false_positives: usize,

    /// True negatives: predicted negative, actually negative (correctly rejected).
    pub true_negatives: usize,

    /// False negatives: predicted negative, actually positive (missed FRECO pair).
    pub false_negatives: usize,

    /// Total number of examples evaluated.
    pub total: usize,
}

impl FrecoMetrics {
    /// Compute metrics from predictions and gold labels.
    ///
    /// # Arguments
    ///
    /// * `predictions` - Iterator of (predicted_label, gold_label) pairs
    ///
    /// # Returns
    ///
    /// Computed metrics.
    #[must_use]
    pub fn from_predictions<I>(predictions: I) -> Self
    where
        I: IntoIterator<Item = (bool, bool)>,
    {
        let mut tp = 0;
        let mut fp = 0;
        let mut tn = 0;
        let mut fn_ = 0;

        for (pred, gold) in predictions {
            match (pred, gold) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                (false, false) => tn += 1,
            }
        }

        let total = tp + fp + tn + fn_;
        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        let recall = if tp + fn_ > 0 {
            tp as f64 / (tp + fn_) as f64
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        let accuracy = if total > 0 {
            (tp + tn) as f64 / total as f64
        } else {
            0.0
        };

        Self {
            precision,
            recall,
            f1,
            accuracy,
            true_positives: tp,
            false_positives: fp,
            true_negatives: tn,
            false_negatives: fn_,
            total,
        }
    }

    /// Compute metrics from confidence scores and gold labels at a threshold.
    ///
    /// # Arguments
    ///
    /// * `scored` - Iterator of (confidence_score, gold_label) pairs
    /// * `threshold` - Classification threshold
    #[must_use]
    pub fn from_scores<I>(scored: I, threshold: f64) -> Self
    where
        I: IntoIterator<Item = (f64, bool)>,
    {
        Self::from_predictions(scored.into_iter().map(|(score, gold)| (score >= threshold, gold)))
    }

    /// Compute Matthews Correlation Coefficient (MCC).
    ///
    /// MCC is more informative than F1 for imbalanced datasets.
    /// Range: [-1, 1], where 1 is perfect, 0 is random, -1 is inverse.
    #[must_use]
    pub fn mcc(&self) -> f64 {
        let tp = self.true_positives as f64;
        let fp = self.false_positives as f64;
        let tn = self.true_negatives as f64;
        let fn_ = self.false_negatives as f64;

        let numerator = tp * tn - fp * fn_;
        let denominator = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
}

// =============================================================================
// Mining/Retrieval Metrics
// =============================================================================

/// Metrics for FRECO pair mining and retrieval evaluation.
///
/// Evaluates systems that retrieve FRECO pairs from a large candidate pool
/// by ranking pairs by similarity/confidence and selecting the top-K.
///
/// # Metrics
///
/// - **Precision@K**: Of the top K candidates, what fraction are true FRECO pairs?
/// - **Recall@K**: Of all gold FRECO pairs, what fraction appear in the top K?
/// - **Average Precision (AP)**: Area under the precision-recall curve, summarizing
///   ranking quality across all thresholds.
///
/// # Typical K Values
///
/// K values are chosen based on annotation budget and expected positive rate:
/// - `K = 10, 20, 50`: High-precision regime for manual review
/// - `K = 100, 200`: Semi-supervised expansion (bootstrap mining)
/// - `K = 1000`: Recall-focused evaluation
///
/// # Example
///
/// ```rust
/// use anno::eval::freco_metrics::FrecoMiningMetrics;
///
/// // Candidates ranked by similarity score (descending)
/// let ranked = vec![
///     (0.95, true),   // Rank 1: FRECO pair
///     (0.90, true),   // Rank 2: FRECO pair
///     (0.85, false),  // Rank 3: not FRECO
///     (0.80, true),   // Rank 4: FRECO pair
///     (0.75, false),  // Rank 5: not FRECO
/// ];
///
/// let metrics = FrecoMiningMetrics::from_ranked(&ranked, &[3, 5]);
///
/// // P@3 = 2/3 (2 positives in top 3)
/// assert!((metrics.precision_at_k[&3] - 0.666).abs() < 0.01);
///
/// // R@3 = 2/3 (found 2 of 3 gold positives)
/// assert!((metrics.recall_at_k[&3] - 0.666).abs() < 0.01);
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FrecoMiningMetrics {
    /// Precision at K: fraction of top-K that are true FRECO pairs.
    ///
    /// `P@K = (true positives in top K) / K`
    pub precision_at_k: HashMap<usize, f64>,

    /// Recall at K: fraction of all gold FRECO pairs found in top-K.
    ///
    /// `R@K = (true positives in top K) / (total gold positives)`
    pub recall_at_k: HashMap<usize, f64>,

    /// Average Precision: area under the precision-recall curve.
    ///
    /// Computed as `AP = (1/|positives|) * sum(P@k * rel(k))` where `rel(k)` is
    /// 1 if item at rank k is positive, 0 otherwise.
    pub average_precision: f64,

    /// Total number of gold positive FRECO pairs in the candidate pool.
    pub num_gold_positive: usize,

    /// Total number of candidate pairs evaluated.
    pub num_candidates: usize,

    /// Jaccard similarity with mined pairs from previous round (for bootstrap tracking).
    ///
    /// High Jaccard (>0.95) indicates convergence in iterative mining.
    pub jaccard_with_previous: Option<f64>,
}

impl FrecoMiningMetrics {
    /// Compute mining metrics from scored candidates.
    ///
    /// # Arguments
    ///
    /// * `scored` - Vec of (confidence_score, gold_label) sorted by score descending
    /// * `k_values` - Values of K for Precision@K and Recall@K
    #[must_use]
    pub fn from_ranked(scored: &[(f64, bool)], k_values: &[usize]) -> Self {
        let num_gold_positive = scored.iter().filter(|(_, gold)| *gold).count();
        let num_candidates = scored.len();

        // Compute P@K and R@K
        let mut precision_at_k = HashMap::new();
        let mut recall_at_k = HashMap::new();

        for &k in k_values {
            let k_capped = k.min(scored.len());
            let top_k = &scored[..k_capped];

            let positives_in_k = top_k.iter().filter(|(_, gold)| *gold).count();

            let p_at_k = if k_capped > 0 {
                positives_in_k as f64 / k_capped as f64
            } else {
                0.0
            };

            let r_at_k = if num_gold_positive > 0 {
                positives_in_k as f64 / num_gold_positive as f64
            } else {
                0.0
            };

            precision_at_k.insert(k, p_at_k);
            recall_at_k.insert(k, r_at_k);
        }

        // Compute Average Precision
        let mut ap_sum = 0.0;
        let mut running_positives = 0;

        for (i, (_, gold)) in scored.iter().enumerate() {
            if *gold {
                running_positives += 1;
                let precision_at_i = running_positives as f64 / (i + 1) as f64;
                ap_sum += precision_at_i;
            }
        }

        let average_precision = if num_gold_positive > 0 {
            ap_sum / num_gold_positive as f64
        } else {
            0.0
        };

        Self {
            precision_at_k,
            recall_at_k,
            average_precision,
            num_gold_positive,
            num_candidates,
            jaccard_with_previous: None,
        }
    }

    /// Compute Jaccard similarity between two sets of mined pairs.
    #[must_use]
    pub fn jaccard<T: Eq + std::hash::Hash>(set_a: &[T], set_b: &[T]) -> f64 {
        use std::collections::HashSet;

        let a: HashSet<_> = set_a.iter().collect();
        let b: HashSet<_> = set_b.iter().collect();

        let intersection = a.intersection(&b).count();
        let union = a.union(&b).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }
}

// =============================================================================
// Attitude Classification Metrics
// =============================================================================

/// Metrics for event attitude classification (supportive/skeptical/neutral).
///
/// Evaluates the ability to classify the framing attitude of individual event
/// mentions—a sub-task of FRECO that determines whether each mention frames
/// an event supportively, skeptically, or neutrally.
///
/// # Metrics
///
/// - **Accuracy**: Overall fraction of correct predictions
/// - **Macro-F1**: Average F1 across all three classes (balances class imbalance)
/// - **Cohen's Kappa**: Agreement beyond chance (κ = 0: chance, κ = 1: perfect)
/// - **Per-class**: Precision/Recall/F1 for each attitude class
///
/// # Example
///
/// ```rust
/// use anno::eval::freco_metrics::AttitudeMetrics;
/// use anno_core::types::framing::FramingAttitude;
///
/// let predictions = vec![
///     (FramingAttitude::Supportive, FramingAttitude::Supportive),  // Correct
///     (FramingAttitude::Skeptical, FramingAttitude::Skeptical),    // Correct
///     (FramingAttitude::Neutral, FramingAttitude::Supportive),      // Wrong
/// ];
///
/// let metrics = AttitudeMetrics::from_predictions(&predictions);
/// assert!((metrics.accuracy - 0.666).abs() < 0.01);
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AttitudeMetrics {
    /// Overall accuracy (correct / total).
    pub accuracy: f64,

    /// Macro-averaged F1 across all three attitude classes.
    ///
    /// Macro-F1 treats each class equally, important when classes are imbalanced
    /// (e.g., Neutral is often most common).
    pub macro_f1: f64,

    /// Per-class metrics (P/R/F1 for each attitude).
    ///
    /// Keys: `"supportive"`, `"skeptical"`, `"neutral"`.
    pub per_class: HashMap<String, FrecoMetrics>,

    /// Cohen's kappa: agreement corrected for chance.
    ///
    /// - κ = 0: Agreement no better than random chance
    /// - κ = 0.21-0.40: Fair agreement
    /// - κ = 0.41-0.60: Moderate agreement
    /// - κ = 0.61-0.80: Substantial agreement
    /// - κ = 0.81-1.00: Near-perfect agreement
    pub cohens_kappa: f64,

    /// Total number of examples evaluated.
    pub total: usize,
}

impl AttitudeMetrics {
    /// Compute metrics from predicted and gold attitudes.
    #[must_use]
    pub fn from_predictions(predictions: &[(FramingAttitude, FramingAttitude)]) -> Self {
        if predictions.is_empty() {
            return Self::default();
        }

        let total = predictions.len();
        let correct = predictions.iter().filter(|(p, g)| p == g).count();
        let accuracy = correct as f64 / total as f64;

        // Per-class metrics
        let mut per_class = HashMap::new();
        for attitude in [
            FramingAttitude::Supportive,
            FramingAttitude::Skeptical,
            FramingAttitude::Neutral,
        ] {
            let class_preds: Vec<_> = predictions
                .iter()
                .map(|(pred, gold)| (*pred == attitude, *gold == attitude))
                .collect();

            let metrics = FrecoMetrics::from_predictions(class_preds.into_iter());
            per_class.insert(attitude.to_string(), metrics);
        }

        // Macro F1
        let macro_f1 = per_class.values().map(|m| m.f1).sum::<f64>() / per_class.len() as f64;

        // Cohen's kappa
        let cohens_kappa = Self::compute_kappa(predictions);

        Self {
            accuracy,
            macro_f1,
            per_class,
            cohens_kappa,
            total,
        }
    }

    /// Compute Cohen's kappa for multi-class agreement.
    fn compute_kappa(predictions: &[(FramingAttitude, FramingAttitude)]) -> f64 {
        if predictions.is_empty() {
            return 0.0;
        }

        let n = predictions.len() as f64;

        // Observed agreement
        let p_o = predictions.iter().filter(|(p, g)| p == g).count() as f64 / n;

        // Expected agreement (chance)
        let mut pred_counts: HashMap<FramingAttitude, usize> = HashMap::new();
        let mut gold_counts: HashMap<FramingAttitude, usize> = HashMap::new();

        for (pred, gold) in predictions {
            *pred_counts.entry(*pred).or_insert(0) += 1;
            *gold_counts.entry(*gold).or_insert(0) += 1;
        }

        let p_e: f64 = [
            FramingAttitude::Supportive,
            FramingAttitude::Skeptical,
            FramingAttitude::Neutral,
        ]
        .iter()
        .map(|attitude| {
            let pred_frac = *pred_counts.get(attitude).unwrap_or(&0) as f64 / n;
            let gold_frac = *gold_counts.get(attitude).unwrap_or(&0) as f64 / n;
            pred_frac * gold_frac
        })
        .sum();

        if (1.0 - p_e).abs() < 1e-10 {
            1.0 // Perfect agreement when expected agreement is 1
        } else {
            (p_o - p_e) / (1.0 - p_e)
        }
    }
}

// =============================================================================
// Cross-Topic Evaluation
// =============================================================================

/// Results from leave-one-topic-out cross-validation.
///
/// Tests whether a FRECO model learns generalizable framing patterns versus
/// topic-specific lexical cues. In each fold, the model trains on N-1 topics
/// and tests on the held-out topic.
///
/// # Interpretation
///
/// - **High F1 variance**: Model is overfitting to topic-specific vocabulary
/// - **Low F1 variance**: Model captures generalizable framing patterns
/// - **Topic-specific drops**: Some topics may have unique framing conventions
///
/// # Example
///
/// ```text
/// Topics: [al-shifa, jan-6, roe-v-wade, immigration]
///
/// Fold 1: Train {jan-6, roe, immigration}, Test {al-shifa} → F1 = 0.72
/// Fold 2: Train {al-shifa, roe, immigration}, Test {jan-6} → F1 = 0.68
/// Fold 3: Train {al-shifa, jan-6, immigration}, Test {roe} → F1 = 0.75
/// Fold 4: Train {al-shifa, jan-6, roe}, Test {immigration} → F1 = 0.65
///
/// Mean F1 = 0.70 ± 0.04 (std)
/// ```
///
/// A model with mean F1 = 0.70 and std = 0.04 shows good generalization.
/// High variance (std > 0.10) suggests overfitting to training topics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CrossTopicEvaluation {
    /// Per-topic test results (topic name → metrics when that topic is held out).
    pub per_topic: HashMap<String, FrecoMetrics>,

    /// Aggregated metrics: confusion matrix pooled across all folds.
    ///
    /// This is micro-averaged (each example counts equally regardless of topic).
    pub aggregate: FrecoMetrics,

    /// Standard deviation of F1 scores across topics.
    ///
    /// Lower is better—indicates consistent performance across topics.
    pub f1_std: f64,

    /// Mean F1 score across all topic folds.
    pub f1_mean: f64,
}

impl CrossTopicEvaluation {
    /// Compute cross-topic evaluation from per-topic results.
    #[must_use]
    pub fn from_per_topic(per_topic: HashMap<String, FrecoMetrics>) -> Self {
        if per_topic.is_empty() {
            return Self::default();
        }

        let f1_values: Vec<f64> = per_topic.values().map(|m| m.f1).collect();
        let f1_mean = f1_values.iter().sum::<f64>() / f1_values.len() as f64;
        let f1_variance =
            f1_values.iter().map(|f| (f - f1_mean).powi(2)).sum::<f64>() / f1_values.len() as f64;
        let f1_std = f1_variance.sqrt();

        // Aggregate: pool all predictions
        let total_tp: usize = per_topic.values().map(|m| m.true_positives).sum();
        let total_fp: usize = per_topic.values().map(|m| m.false_positives).sum();
        let total_tn: usize = per_topic.values().map(|m| m.true_negatives).sum();
        let total_fn: usize = per_topic.values().map(|m| m.false_negatives).sum();
        let total = total_tp + total_fp + total_tn + total_fn;

        let precision = if total_tp + total_fp > 0 {
            total_tp as f64 / (total_tp + total_fp) as f64
        } else {
            0.0
        };
        let recall = if total_tp + total_fn > 0 {
            total_tp as f64 / (total_tp + total_fn) as f64
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        let accuracy = if total > 0 {
            (total_tp + total_tn) as f64 / total as f64
        } else {
            0.0
        };

        let aggregate = FrecoMetrics {
            precision,
            recall,
            f1,
            accuracy,
            true_positives: total_tp,
            false_positives: total_fp,
            true_negatives: total_tn,
            false_negatives: total_fn,
            total,
        };

        Self {
            per_topic,
            aggregate,
            f1_std,
            f1_mean,
        }
    }
}

// =============================================================================
// Bootstrapping Metrics
// =============================================================================

/// Metrics for a single round of bootstrapped FRECO mining.
///
/// In bootstrapped mining, we iteratively:
/// 1. Train a classifier on current labeled data
/// 2. Score unlabeled candidates
/// 3. Add high-confidence positives to training set
/// 4. Repeat until convergence
///
/// This struct tracks the state after each round.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BootstrapRoundMetrics {
    /// Round number (0 = initial seed set, 1+ = bootstrap rounds).
    pub round: usize,

    /// Confidence threshold used for selecting new positives this round.
    ///
    /// Typically starts high (0.9) and may decrease slightly as training set grows.
    pub threshold: f64,

    /// Number of new pairs added to training set this round.
    pub new_pairs: usize,

    /// Number of new positive (FRECO) pairs added this round.
    pub new_positive_pairs: usize,

    /// Cumulative training set size after this round.
    pub cumulative_size: usize,

    /// Cumulative number of positive pairs after this round.
    pub cumulative_positive: usize,

    /// Jaccard similarity between mined pairs from this round and previous.
    ///
    /// High Jaccard (>0.95) indicates convergence—the model is finding the same pairs.
    pub jaccard_with_previous: f64,

    /// Validation loss (cross-entropy) on held-out validation set.
    ///
    /// Rising validation loss signals overfitting to bootstrap noise.
    pub validation_loss: f64,

    /// Estimated precision on newly added pairs (from human spot-check).
    ///
    /// If available, used to estimate quality of the bootstrap expansion.
    pub estimated_precision: Option<f64>,
}

/// Track progress across multiple bootstrap mining rounds.
///
/// Used for semi-supervised expansion of FRECO training data: start with a
/// small seed set of annotated pairs, iteratively mine high-confidence pairs
/// from unlabeled data, and grow the training set.
///
/// # Convergence Criteria
///
/// The [`should_stop`](Self::should_stop) method checks three conditions:
///
/// 1. **Validation loss increases**: Training is overfitting to bootstrap noise
/// 2. **High Jaccard similarity**: Model keeps finding the same pairs (saturation)
/// 3. **Few new positives**: Diminishing returns on mining
///
/// # Example Progression
///
/// ```text
/// Round 0 (seed):   100 pairs,   50 positive, threshold=0.95
/// Round 1:         +200 pairs,  +95 positive, jaccard=0.10, loss=0.45
/// Round 2:         +150 pairs,  +70 positive, jaccard=0.35, loss=0.43
/// Round 3:          +80 pairs,  +35 positive, jaccard=0.60, loss=0.44 ← loss increased
/// Round 4:          +40 pairs,  +15 positive, jaccard=0.85, loss=0.46 ← STOP
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BootstrapProgress {
    /// Metrics from each bootstrap round.
    pub rounds: Vec<BootstrapRoundMetrics>,

    /// Final estimated precision on the expanded training set.
    ///
    /// Estimated from human spot-checks or held-out validation.
    pub final_precision: Option<f64>,

    /// Final estimated recall (upper bound).
    ///
    /// Computed as (mined positives) / (estimated total positives in corpus).
    pub final_recall: Option<f64>,
}

impl BootstrapProgress {
    /// Check convergence criteria.
    ///
    /// Returns true if:
    /// - Validation loss has increased for 2+ rounds
    /// - Jaccard is below threshold (finding diminishing new examples)
    /// - New positive pairs are below minimum
    #[must_use]
    pub fn should_stop(
        &self,
        max_loss_increases: usize,
        min_jaccard: f64,
        min_new_positives: usize,
    ) -> bool {
        if self.rounds.len() < 2 {
            return false;
        }

        // Check loss trend
        let recent: Vec<_> = self.rounds.iter().rev().take(max_loss_increases + 1).collect();
        let loss_increasing = recent.windows(2).all(|w| w[0].validation_loss >= w[1].validation_loss);

        // Check Jaccard
        let latest = self.rounds.last().unwrap();
        let jaccard_low = latest.jaccard_with_previous < min_jaccard;

        // Check new positives
        let few_new = latest.new_positive_pairs < min_new_positives;

        (loss_increasing && self.rounds.len() > max_loss_increases) || jaccard_low || few_new
    }
}

// =============================================================================
// FRECO Evaluator
// =============================================================================

/// Unified evaluator for FRECO classification and mining.
#[derive(Debug, Clone)]
pub struct FrecoEvaluator {
    /// Default K values for P@K and R@K
    pub k_values: Vec<usize>,
    /// Default classification threshold
    pub threshold: f64,
}

impl Default for FrecoEvaluator {
    fn default() -> Self {
        Self {
            k_values: vec![10, 50, 100, 500, 1000],
            threshold: 0.5,
        }
    }
}

impl FrecoEvaluator {
    /// Create a new evaluator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set K values for mining metrics.
    #[must_use]
    pub fn with_k_values(mut self, k_values: Vec<usize>) -> Self {
        self.k_values = k_values;
        self
    }

    /// Set classification threshold.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Evaluate classification from (confidence, gold_label) pairs.
    #[must_use]
    pub fn evaluate_classification(&self, predictions: &[(f64, bool)]) -> FrecoMetrics {
        FrecoMetrics::from_scores(predictions.iter().copied(), self.threshold)
    }

    /// Evaluate mining/retrieval from ranked (confidence, gold_label) pairs.
    ///
    /// Assumes `predictions` is sorted by confidence descending.
    #[must_use]
    pub fn evaluate_mining(&self, predictions: &[(f64, bool)]) -> FrecoMiningMetrics {
        FrecoMiningMetrics::from_ranked(predictions, &self.k_values)
    }

    /// Evaluate attitude classification.
    #[must_use]
    pub fn evaluate_attitudes(
        &self,
        predictions: &[(FramingAttitude, FramingAttitude)],
    ) -> AttitudeMetrics {
        AttitudeMetrics::from_predictions(predictions)
    }

    /// Run leave-one-topic-out evaluation on a corpus.
    ///
    /// Returns metrics for each held-out topic.
    pub fn evaluate_cross_topic<F>(
        &self,
        corpus: &FrecoCorpus,
        predict_fn: F,
    ) -> CrossTopicEvaluation
    where
        F: Fn(&[FrecoPair], &[FrecoPair]) -> Vec<(f64, bool)>,
    {
        let topics: Vec<_> = corpus.topic_names().into_iter().map(String::from).collect();
        let mut per_topic = HashMap::new();

        for test_topic in &topics {
            // Split: test_topic held out, rest for training
            let train_pairs: Vec<_> = corpus
                .topics
                .iter()
                .filter(|(t, _)| t.as_str() != test_topic.as_str())
                .flat_map(|(_, pairs)| pairs.iter().cloned())
                .collect();

            let test_pairs: Vec<_> = corpus
                .pairs_for_topic(test_topic)
                .map(|p| p.to_vec())
                .unwrap_or_default();

            if test_pairs.is_empty() {
                continue;
            }

            // Get predictions
            let predictions = predict_fn(&train_pairs, &test_pairs);

            // Compute metrics
            let metrics = self.evaluate_classification(&predictions);
            per_topic.insert(test_topic.clone(), metrics);
        }

        CrossTopicEvaluation::from_per_topic(per_topic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_freco_metrics_basic() {
        // 3 TP, 1 FP, 1 TN, 1 FN
        let predictions = vec![
            (true, true),   // TP
            (true, true),   // TP
            (true, true),   // TP
            (true, false),  // FP
            (false, false), // TN
            (false, true),  // FN
        ];

        let metrics = FrecoMetrics::from_predictions(predictions.into_iter());

        assert_eq!(metrics.true_positives, 3);
        assert_eq!(metrics.false_positives, 1);
        assert_eq!(metrics.true_negatives, 1);
        assert_eq!(metrics.false_negatives, 1);
        assert!((metrics.precision - 0.75).abs() < 0.001); // 3/4
        assert!((metrics.recall - 0.75).abs() < 0.001); // 3/4
        assert!((metrics.f1 - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_freco_metrics_from_scores() {
        let scored = vec![
            (0.9, true),  // Above threshold, gold positive -> TP
            (0.7, true),  // Above threshold, gold positive -> TP
            (0.4, true),  // Below threshold, gold positive -> FN
            (0.8, false), // Above threshold, gold negative -> FP
            (0.3, false), // Below threshold, gold negative -> TN
        ];

        let metrics = FrecoMetrics::from_scores(scored.into_iter(), 0.5);

        assert_eq!(metrics.true_positives, 2);
        assert_eq!(metrics.false_positives, 1);
        assert_eq!(metrics.true_negatives, 1);
        assert_eq!(metrics.false_negatives, 1);
    }

    #[test]
    fn test_mcc() {
        // Perfect prediction
        let metrics = FrecoMetrics {
            true_positives: 10,
            false_positives: 0,
            true_negatives: 10,
            false_negatives: 0,
            ..Default::default()
        };
        assert!((metrics.mcc() - 1.0).abs() < 0.001);

        // Random prediction (balanced)
        let metrics = FrecoMetrics {
            true_positives: 5,
            false_positives: 5,
            true_negatives: 5,
            false_negatives: 5,
            ..Default::default()
        };
        assert!((metrics.mcc() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_mining_metrics() {
        // Ranked by confidence, descending
        let scored = vec![
            (0.95, true),  // rank 1
            (0.90, true),  // rank 2
            (0.85, false), // rank 3
            (0.80, true),  // rank 4
            (0.75, false), // rank 5
            (0.70, true),  // rank 6
            (0.65, false), // rank 7
            (0.60, false), // rank 8
            (0.55, true),  // rank 9
            (0.50, false), // rank 10
        ];

        let metrics = FrecoMiningMetrics::from_ranked(&scored, &[5, 10]);

        // P@5: 3 positives in top 5 -> 3/5 = 0.6
        assert!((metrics.precision_at_k[&5] - 0.6).abs() < 0.001);

        // R@5: 3 positives found, 5 total gold -> 3/5 = 0.6
        assert!((metrics.recall_at_k[&5] - 0.6).abs() < 0.001);

        // Total gold positives
        assert_eq!(metrics.num_gold_positive, 5);
    }

    #[test]
    fn test_jaccard() {
        let set_a = vec!["a", "b", "c"];
        let set_b = vec!["b", "c", "d"];

        // Intersection: {b, c} = 2
        // Union: {a, b, c, d} = 4
        // Jaccard: 2/4 = 0.5
        let jaccard = FrecoMiningMetrics::jaccard(&set_a, &set_b);
        assert!((jaccard - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_attitude_metrics() {
        let predictions = vec![
            (FramingAttitude::Supportive, FramingAttitude::Supportive),
            (FramingAttitude::Skeptical, FramingAttitude::Skeptical),
            (FramingAttitude::Neutral, FramingAttitude::Supportive), // Wrong
            (FramingAttitude::Supportive, FramingAttitude::Supportive),
        ];

        let metrics = AttitudeMetrics::from_predictions(&predictions);

        assert!((metrics.accuracy - 0.75).abs() < 0.001); // 3/4 correct
        assert!(metrics.cohens_kappa > 0.0); // Some agreement beyond chance
    }

    #[test]
    fn test_cross_topic_evaluation() {
        let mut per_topic = HashMap::new();

        per_topic.insert(
            "topic_a".to_string(),
            FrecoMetrics {
                f1: 0.80,
                precision: 0.75,
                recall: 0.85,
                accuracy: 0.78,
                true_positives: 85,
                false_positives: 28,
                true_negatives: 72,
                false_negatives: 15,
                total: 200,
            },
        );

        per_topic.insert(
            "topic_b".to_string(),
            FrecoMetrics {
                f1: 0.70,
                precision: 0.65,
                recall: 0.76,
                accuracy: 0.68,
                true_positives: 76,
                false_positives: 41,
                true_negatives: 60,
                false_negatives: 24,
                total: 201,
            },
        );

        let eval = CrossTopicEvaluation::from_per_topic(per_topic);

        assert!((eval.f1_mean - 0.75).abs() < 0.001); // (0.80 + 0.70) / 2
        assert!(eval.f1_std > 0.0); // Non-zero std
    }

    #[test]
    fn test_bootstrap_convergence() {
        let progress = BootstrapProgress {
            rounds: vec![
                BootstrapRoundMetrics {
                    round: 0,
                    validation_loss: 0.5,
                    jaccard_with_previous: 1.0,
                    new_positive_pairs: 100,
                    ..Default::default()
                },
                BootstrapRoundMetrics {
                    round: 1,
                    validation_loss: 0.4,
                    jaccard_with_previous: 0.5,
                    new_positive_pairs: 50,
                    ..Default::default()
                },
                BootstrapRoundMetrics {
                    round: 2,
                    validation_loss: 0.42,
                    jaccard_with_previous: 0.2,
                    new_positive_pairs: 10,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        // Should stop: jaccard is low (0.2 < 0.25)
        assert!(progress.should_stop(2, 0.25, 5));

        // Should not stop with looser criteria
        assert!(!progress.should_stop(5, 0.1, 5));
    }
}

