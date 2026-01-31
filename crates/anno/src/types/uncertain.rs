//! Uncertain predictions and abstention for selective NER.
//!
//! # The Problem with Forced Labels
//!
//! Traditional NER systems must output a label for every span they consider,
//! even when uncertain. This leads to:
//!
//! - **Overconfident false positives**: System says "PERSON" with 0.51 confidence
//! - **Hidden uncertainty**: Users can't distinguish confident vs. guessing
//! - **No recourse**: Can't say "I don't know, ask a human"
//!
//! # Selective Prediction
//!
//! This module provides types for **selective prediction**, where the model can:
//!
//! 1. **Abstain**: Explicitly decline to label a span
//! 2. **Distribute**: Output probabilities over multiple types
//! 3. **Bound**: Provide confidence intervals, not point estimates
//!
//! # Research Background
//!
//! - Geifman & El-Yaniv (2017): "Selective Prediction via Deep Neural Networks"
//! - El-Yaniv & Wiener (2010): "On the Foundations of Noise-Free Selective Classification"
//! - Kamath et al. (2020): "Selective Question Answering under Domain Shift"
//!
//! # Trade-off: Coverage vs. Accuracy
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────┐
//! │                                                                │
//! │  Accuracy                                                      │
//! │     │                                                          │
//! │ 100%├────────────────*.                                        │
//! │     │               *  `.                                      │
//! │     │              *    `.                                     │
//! │ 80% ├─────────────*      `..                                   │
//! │     │            *          `..                                │
//! │     │           *              `..                             │
//! │ 60% ├──────────*                 `...                          │
//! │     │         *                      `...                      │
//! │     │        *                           `...                  │
//! │ 40% ├───────*─────────────────────────────────`...             │
//! │     │      *                                      `...         │
//! │     └──────┴────────────┴────────────┴────────────┴───Coverage │
//! │           20%          50%          80%         100%           │
//! │                                                                │
//! │  • Low coverage (20%): Only answer when very confident → 95%+  │
//! │  • High coverage (100%): Answer everything → baseline accuracy │
//! │                                                                │
//! └────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust
//! use anno::types::uncertain::{UncertainPrediction, TypeDistribution, Abstention};
//! use anno::EntityType;
//!
//! // A confident prediction
//! let confident = UncertainPrediction::from_type(EntityType::Person, 0.95);
//! assert!(confident.is_confident(0.8));
//!
//! // An uncertain prediction with distribution
//! let uncertain = UncertainPrediction::distributed(TypeDistribution::new(vec![
//!     (EntityType::Person, 0.45),
//!     (EntityType::Organization, 0.40),
//!     (EntityType::Location, 0.15),
//! ]));
//! assert!(!uncertain.is_confident(0.8));
//!
//! // Explicit abstention
//! let abstain = UncertainPrediction::abstain(Abstention::LowConfidence { max_score: 0.35 });
//! assert!(abstain.is_abstention());
//! ```

use crate::EntityType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// =============================================================================
// Type Distribution
// =============================================================================

/// Distribution over entity types.
///
/// Unlike a single EntityType prediction, this captures uncertainty
/// by assigning probability mass to multiple types.
///
/// # Invariants
///
/// - Probabilities are in [0, 1]
/// - Probabilities may not sum to 1 (unnormalized is allowed)
/// - Empty distributions are valid (no prediction)
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct TypeDistribution {
    /// Type → probability mapping
    probs: Vec<(EntityType, f64)>,
}

impl TypeDistribution {
    /// Create a new type distribution.
    ///
    /// Probabilities are clamped to [0, 1] but NOT normalized.
    /// This allows representing "I'm 30% sure it's PERSON, 20% ORG, and 50% nothing".
    #[must_use]
    pub fn new(probs: Vec<(EntityType, f64)>) -> Self {
        let probs = probs
            .into_iter()
            .map(|(t, p)| (t, p.clamp(0.0, 1.0)))
            .filter(|(_, p)| *p > 0.0)
            .collect();
        Self { probs }
    }

    /// Create a uniform distribution over types.
    #[must_use]
    pub fn uniform(types: &[EntityType]) -> Self {
        if types.is_empty() {
            return Self { probs: vec![] };
        }
        let p = 1.0 / types.len() as f64;
        Self::new(types.iter().map(|t| (t.clone(), p)).collect())
    }

    /// Create a distribution with all mass on one type.
    #[must_use]
    pub fn point_mass(entity_type: EntityType, confidence: f64) -> Self {
        Self::new(vec![(entity_type, confidence)])
    }

    /// Get the most likely type and its probability.
    #[must_use]
    pub fn argmax(&self) -> Option<(&EntityType, f64)> {
        self.probs
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(t, p)| (t, *p))
    }

    /// Get probability for a specific type.
    #[must_use]
    pub fn prob(&self, entity_type: &EntityType) -> f64 {
        self.probs
            .iter()
            .find(|(t, _)| t == entity_type)
            .map(|(_, p)| *p)
            .unwrap_or(0.0)
    }

    /// Get entropy of the distribution (higher = more uncertain).
    ///
    /// Returns 0 for point mass, log(n) for uniform over n types.
    #[must_use]
    pub fn entropy(&self) -> f64 {
        let total: f64 = self.probs.iter().map(|(_, p)| p).sum();
        if total <= 0.0 {
            return 0.0;
        }

        let mut h = 0.0;
        for (_, p) in &self.probs {
            if *p > 0.0 {
                let normalized = p / total;
                h -= normalized * normalized.ln();
            }
        }
        h
    }

    /// Get margin between top two predictions (higher = more confident).
    ///
    /// Returns 1.0 if only one type, 0.0 if tied.
    #[must_use]
    pub fn margin(&self) -> f64 {
        if self.probs.len() < 2 {
            return 1.0;
        }

        let mut sorted: Vec<f64> = self.probs.iter().map(|(_, p)| *p).collect();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        sorted[0] - sorted.get(1).unwrap_or(&0.0)
    }

    /// Check if the top prediction exceeds a confidence threshold.
    #[must_use]
    pub fn is_confident(&self, threshold: f64) -> bool {
        self.argmax().is_some_and(|(_, p)| p >= threshold)
    }

    /// Convert to a HashMap for easier iteration.
    #[must_use]
    pub fn to_map(&self) -> HashMap<EntityType, f64> {
        self.probs.iter().cloned().collect()
    }

    /// Get the number of types with non-zero probability.
    #[must_use]
    pub fn num_types(&self) -> usize {
        self.probs.len()
    }

    /// Check if distribution is empty (no predictions).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.probs.is_empty()
    }

    /// Iterate over (type, probability) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&EntityType, f64)> {
        self.probs.iter().map(|(t, p)| (t, *p))
    }
}

impl fmt::Display for TypeDistribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.probs.is_empty() {
            return write!(f, "∅");
        }
        let parts: Vec<String> = self
            .probs
            .iter()
            .map(|(t, p)| format!("{}:{:.1}%", t.as_label(), p * 100.0))
            .collect();
        write!(f, "{{{}}}", parts.join(", "))
    }
}

// =============================================================================
// Abstention
// =============================================================================

/// Reason for abstaining from a prediction.
///
/// Different abstention reasons may warrant different downstream handling:
/// - LowConfidence: Might benefit from more context or a human review
/// - Ambiguous: Might benefit from entity linking to disambiguate
/// - OutOfDomain: Should not be used for this domain
/// - Conflict: Multiple signals disagree, needs resolution
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Abstention {
    /// Maximum prediction score was below threshold.
    LowConfidence {
        /// The highest confidence score observed
        max_score: f64,
    },

    /// Multiple types have similar probabilities (high entropy).
    Ambiguous {
        /// The top two competing types
        top_types: Vec<EntityType>,
        /// The margin between them (close to 0 = ambiguous)
        margin: f64,
    },

    /// The text appears outside the model's training domain.
    OutOfDomain {
        /// Optional domain name if detected
        detected_domain: Option<String>,
    },

    /// Multiple extraction methods produced conflicting results.
    Conflict {
        /// The conflicting predictions
        predictions: Vec<(String, EntityType)>, // (source, type)
    },

    /// Span length or structure is invalid for entity extraction.
    InvalidSpan {
        /// Reason the span is invalid
        reason: String,
    },

    /// Model explicitly declined (e.g., safety filter, policy).
    Declined {
        /// Reason for declining
        reason: String,
    },
}

impl Abstention {
    /// Get a human-readable description of the abstention reason.
    #[must_use]
    pub fn description(&self) -> String {
        match self {
            Self::LowConfidence { max_score } => {
                format!("Low confidence: max score {:.1}%", max_score * 100.0)
            }
            Self::Ambiguous { top_types, margin } => {
                let types: Vec<_> = top_types.iter().map(|t| t.as_label()).collect();
                format!(
                    "Ambiguous between {} (margin: {:.1}%)",
                    types.join(" vs "),
                    margin * 100.0
                )
            }
            Self::OutOfDomain { detected_domain } => match detected_domain {
                Some(d) => format!("Out of domain: detected '{}'", d),
                None => "Out of domain".to_string(),
            },
            Self::Conflict { predictions } => {
                let conflicts: Vec<_> = predictions
                    .iter()
                    .map(|(src, t)| format!("{}→{}", src, t.as_label()))
                    .collect();
                format!("Conflict: {}", conflicts.join(", "))
            }
            Self::InvalidSpan { reason } => format!("Invalid span: {}", reason),
            Self::Declined { reason } => format!("Declined: {}", reason),
        }
    }

    /// Check if this abstention might be resolvable with more context.
    #[must_use]
    pub fn is_resolvable(&self) -> bool {
        matches!(
            self,
            Self::LowConfidence { .. } | Self::Ambiguous { .. } | Self::Conflict { .. }
        )
    }
}

impl fmt::Display for Abstention {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description())
    }
}

// =============================================================================
// Uncertain Prediction
// =============================================================================

/// A prediction that may include uncertainty or abstention.
///
/// This is the core type for selective prediction, replacing simple
/// (EntityType, confidence) pairs with richer uncertainty information.
///
/// # Variants
///
/// - **Single**: A single type prediction with confidence (traditional)
/// - **Distributed**: Probabilities over multiple types (soft prediction)
/// - **Abstained**: Explicit refusal to predict with reason
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UncertainPrediction {
    /// Single type prediction (traditional NER output).
    Single {
        /// The predicted type
        entity_type: EntityType,
        /// Confidence score [0, 1]
        confidence: f64,
    },

    /// Distribution over types (soft prediction).
    Distributed(TypeDistribution),

    /// Explicit abstention with reason.
    Abstained(Abstention),
}

impl UncertainPrediction {
    /// Create a single-type prediction.
    #[must_use]
    pub fn from_type(entity_type: EntityType, confidence: f64) -> Self {
        Self::Single {
            entity_type,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Create a distributed prediction.
    #[must_use]
    pub fn distributed(dist: TypeDistribution) -> Self {
        Self::Distributed(dist)
    }

    /// Create an abstention.
    #[must_use]
    pub fn abstain(reason: Abstention) -> Self {
        Self::Abstained(reason)
    }

    /// Create abstention due to low confidence.
    #[must_use]
    pub fn abstain_low_confidence(max_score: f64) -> Self {
        Self::Abstained(Abstention::LowConfidence { max_score })
    }

    /// Create abstention due to ambiguity.
    #[must_use]
    pub fn abstain_ambiguous(top_types: Vec<EntityType>, margin: f64) -> Self {
        Self::Abstained(Abstention::Ambiguous { top_types, margin })
    }

    /// Check if this is an abstention.
    #[must_use]
    pub fn is_abstention(&self) -> bool {
        matches!(self, Self::Abstained(_))
    }

    /// Check if this prediction is confident (above threshold).
    #[must_use]
    pub fn is_confident(&self, threshold: f64) -> bool {
        match self {
            Self::Single { confidence, .. } => *confidence >= threshold,
            Self::Distributed(dist) => dist.is_confident(threshold),
            Self::Abstained(_) => false,
        }
    }

    /// Get the best prediction type and confidence.
    ///
    /// Returns `None` if abstained or distribution is empty.
    #[must_use]
    pub fn best(&self) -> Option<(&EntityType, f64)> {
        match self {
            Self::Single {
                entity_type,
                confidence,
            } => Some((entity_type, *confidence)),
            Self::Distributed(dist) => dist.argmax(),
            Self::Abstained(_) => None,
        }
    }

    /// Get the entity type if prediction is confident.
    ///
    /// Returns `None` if abstained, distributed, or below threshold.
    #[must_use]
    pub fn get_type(&self) -> Option<&EntityType> {
        match self {
            Self::Single { entity_type, .. } => Some(entity_type),
            Self::Distributed(dist) => dist.argmax().map(|(t, _)| t),
            Self::Abstained(_) => None,
        }
    }

    /// Get confidence score.
    ///
    /// Returns 0.0 for abstentions, max prob for distributions.
    #[must_use]
    pub fn confidence(&self) -> f64 {
        match self {
            Self::Single { confidence, .. } => *confidence,
            Self::Distributed(dist) => dist.argmax().map(|(_, p)| p).unwrap_or(0.0),
            Self::Abstained(_) => 0.0,
        }
    }

    /// Get the type distribution if available.
    #[must_use]
    pub fn distribution(&self) -> Option<&TypeDistribution> {
        match self {
            Self::Distributed(dist) => Some(dist),
            _ => None,
        }
    }

    /// Get the abstention reason if abstained.
    #[must_use]
    pub fn abstention_reason(&self) -> Option<&Abstention> {
        match self {
            Self::Abstained(reason) => Some(reason),
            _ => None,
        }
    }

    /// Convert to a single prediction, applying threshold for abstention.
    ///
    /// If the prediction is below threshold, converts to abstention.
    #[must_use]
    pub fn with_threshold(self, threshold: f64) -> Self {
        match &self {
            Self::Single { confidence, .. } if *confidence < threshold => {
                Self::abstain_low_confidence(*confidence)
            }
            Self::Distributed(dist) => {
                if let Some((_, p)) = dist.argmax() {
                    if p < threshold {
                        if dist.num_types() >= 2 {
                            let top: Vec<_> = dist.iter().take(2).map(|(t, _)| t.clone()).collect();
                            return Self::abstain_ambiguous(top, dist.margin());
                        }
                        return Self::abstain_low_confidence(p);
                    }
                }
                self
            }
            _ => self,
        }
    }
}

impl fmt::Display for UncertainPrediction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Single {
                entity_type,
                confidence,
            } => {
                write!(f, "{} ({:.1}%)", entity_type.as_label(), confidence * 100.0)
            }
            Self::Distributed(dist) => write!(f, "{}", dist),
            Self::Abstained(reason) => write!(f, "ABSTAIN: {}", reason),
        }
    }
}

// =============================================================================
// Uncertain Entity
// =============================================================================

/// An entity extraction with uncertainty information.
///
/// Extends the standard Entity with richer uncertainty modeling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertainEntity {
    /// The text of the extracted entity
    pub text: String,
    /// Start offset (characters)
    pub start: usize,
    /// End offset (characters)
    pub end: usize,
    /// The uncertain prediction
    pub prediction: UncertainPrediction,
    /// Source backend name
    pub source: Option<String>,
}

impl UncertainEntity {
    /// Create a new uncertain entity.
    #[must_use]
    pub fn new(text: String, start: usize, end: usize, prediction: UncertainPrediction) -> Self {
        Self {
            text,
            start,
            end,
            prediction,
            source: None,
        }
    }

    /// Set the source backend.
    #[must_use]
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Check if this entity should be included at a given threshold.
    #[must_use]
    pub fn should_include(&self, threshold: f64) -> bool {
        self.prediction.is_confident(threshold)
    }

    /// Convert to standard Entity if confident enough.
    ///
    /// Returns `None` if abstained or below threshold.
    #[must_use]
    pub fn to_entity(&self, threshold: f64) -> Option<crate::Entity> {
        if !self.prediction.is_confident(threshold) {
            return None;
        }

        let (entity_type, confidence) = self.prediction.best()?;
        Some(crate::Entity::new(
            &self.text,
            entity_type.clone(),
            self.start,
            self.end,
            confidence,
        ))
    }
}

// =============================================================================
// Selective Metrics
// =============================================================================

/// Metrics for evaluating selective prediction.
///
/// Captures the coverage-accuracy tradeoff.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SelectiveMetrics {
    /// Total predictions made (excluding abstentions)
    pub predictions: usize,
    /// Total abstentions
    pub abstentions: usize,
    /// Correct predictions
    pub correct: usize,
    /// Coverage = predictions / (predictions + abstentions)
    pub coverage: f64,
    /// Accuracy = correct / predictions (among non-abstained)
    pub accuracy: f64,
    /// Risk = incorrect / (predictions + abstentions) (including abstention penalty)
    pub risk: f64,
}

impl SelectiveMetrics {
    /// Compute selective metrics from predictions and gold labels.
    ///
    /// # Arguments
    ///
    /// * `predictions` - Vec of (predicted_type_or_none, gold_type)
    ///   - `Some(t)` = prediction made
    ///   - `None` = abstained
    #[must_use]
    pub fn compute(predictions: &[(Option<EntityType>, EntityType)]) -> Self {
        let mut metrics = Self::default();
        let total = predictions.len();
        if total == 0 {
            return metrics;
        }

        for (pred, gold) in predictions {
            match pred {
                Some(pred_type) => {
                    metrics.predictions += 1;
                    if pred_type == gold {
                        metrics.correct += 1;
                    }
                }
                None => {
                    metrics.abstentions += 1;
                }
            }
        }

        metrics.coverage = metrics.predictions as f64 / total as f64;
        metrics.accuracy = if metrics.predictions > 0 {
            metrics.correct as f64 / metrics.predictions as f64
        } else {
            0.0
        };

        // Risk: fraction of incorrect predictions over all items
        // (abstentions count as 0 error, predictions count their actual error)
        let incorrect = metrics.predictions - metrics.correct;
        metrics.risk = incorrect as f64 / total as f64;

        metrics
    }

    /// Compute AUC for coverage-accuracy curve.
    ///
    /// Higher is better. Measures area under the coverage-accuracy curve
    /// as threshold varies.
    #[must_use]
    pub fn coverage_accuracy_auc(
        uncertain_predictions: &[(UncertainPrediction, EntityType)],
    ) -> f64 {
        if uncertain_predictions.is_empty() {
            return 0.0;
        }

        // Sort by confidence descending
        let mut sorted: Vec<_> = uncertain_predictions
            .iter()
            .map(|(pred, gold)| (pred.confidence(), pred.get_type(), gold))
            .collect();
        sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Compute accuracy at each coverage level
        let total = sorted.len() as f64;
        let mut correct = 0.0;
        let mut auc = 0.0;

        for (i, (_, pred_type, gold)) in sorted.iter().enumerate() {
            if pred_type.is_some_and(|pt| pt == *gold) {
                correct += 1.0;
            }
            let coverage = (i + 1) as f64 / total;
            let accuracy = correct / (i + 1) as f64;

            // Trapezoidal rule for AUC
            if i > 0 {
                let prev_coverage = i as f64 / total;
                auc += (coverage - prev_coverage) * accuracy;
            }
        }

        auc
    }
}

impl fmt::Display for SelectiveMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Coverage: {:.1}%, Accuracy: {:.1}%, Risk: {:.1}% ({}/{} predicted, {} abstained)",
            self.coverage * 100.0,
            self.accuracy * 100.0,
            self.risk * 100.0,
            self.predictions,
            self.predictions + self.abstentions,
            self.abstentions
        )
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_distribution_argmax() {
        let dist = TypeDistribution::new(vec![
            (EntityType::Person, 0.7),
            (EntityType::Organization, 0.2),
            (EntityType::Location, 0.1),
        ]);

        let (best, prob) = dist.argmax().unwrap();
        assert_eq!(*best, EntityType::Person);
        assert!((prob - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_type_distribution_entropy() {
        // Point mass should have 0 entropy
        let point = TypeDistribution::point_mass(EntityType::Person, 1.0);
        assert!((point.entropy() - 0.0).abs() < 1e-10);

        // Uniform should have higher entropy
        let uniform = TypeDistribution::uniform(&[
            EntityType::Person,
            EntityType::Organization,
            EntityType::Location,
        ]);
        assert!(uniform.entropy() > 0.0);
    }

    #[test]
    fn test_type_distribution_margin() {
        // Clear winner
        let clear = TypeDistribution::new(vec![
            (EntityType::Person, 0.9),
            (EntityType::Organization, 0.1),
        ]);
        assert!((clear.margin() - 0.8).abs() < 1e-10);

        // Tied
        let tied = TypeDistribution::new(vec![
            (EntityType::Person, 0.5),
            (EntityType::Organization, 0.5),
        ]);
        assert!((tied.margin() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_uncertain_prediction_single() {
        let pred = UncertainPrediction::from_type(EntityType::Person, 0.85);
        assert!(pred.is_confident(0.8));
        assert!(!pred.is_confident(0.9));
        assert!(!pred.is_abstention());

        let (t, c) = pred.best().unwrap();
        assert_eq!(*t, EntityType::Person);
        assert!((c - 0.85).abs() < 1e-10);
    }

    #[test]
    fn test_uncertain_prediction_abstain() {
        let pred = UncertainPrediction::abstain_low_confidence(0.35);
        assert!(pred.is_abstention());
        assert!(!pred.is_confident(0.1));
        assert!(pred.best().is_none());

        let reason = pred.abstention_reason().unwrap();
        assert!(matches!(reason, Abstention::LowConfidence { .. }));
    }

    #[test]
    fn test_with_threshold() {
        let pred = UncertainPrediction::from_type(EntityType::Person, 0.6);

        // Below threshold → abstain
        let result = pred.clone().with_threshold(0.7);
        assert!(result.is_abstention());

        // Above threshold → keep
        let result2 = pred.with_threshold(0.5);
        assert!(!result2.is_abstention());
    }

    #[test]
    fn test_selective_metrics() {
        let predictions = vec![
            (Some(EntityType::Person), EntityType::Person), // correct
            (Some(EntityType::Organization), EntityType::Person), // incorrect
            (None, EntityType::Location),                   // abstained
            (Some(EntityType::Location), EntityType::Location), // correct
        ];

        let metrics = SelectiveMetrics::compute(&predictions);

        assert_eq!(metrics.predictions, 3);
        assert_eq!(metrics.abstentions, 1);
        assert_eq!(metrics.correct, 2);
        assert!((metrics.coverage - 0.75).abs() < 1e-10);
        assert!((metrics.accuracy - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_uncertain_entity_to_entity() {
        let ue = UncertainEntity::new(
            "John".to_string(),
            0,
            4,
            UncertainPrediction::from_type(EntityType::Person, 0.9),
        );

        // Should convert when above threshold
        let entity = ue.to_entity(0.8);
        assert!(entity.is_some());
        let e = entity.unwrap();
        assert_eq!(e.text, "John");
        assert_eq!(e.entity_type, EntityType::Person);

        // Should not convert when below threshold
        let entity_low = ue.to_entity(0.95);
        assert!(entity_low.is_none());
    }

    #[test]
    fn test_abstention_resolvable() {
        assert!(Abstention::LowConfidence { max_score: 0.3 }.is_resolvable());
        assert!(Abstention::Ambiguous {
            top_types: vec![],
            margin: 0.1
        }
        .is_resolvable());
        assert!(!Abstention::OutOfDomain {
            detected_domain: None
        }
        .is_resolvable());
    }
}
