//! Active learning utilities for NER annotation.
//!
//! Helps identify which examples to annotate next for maximum model improvement.
//!
//! # Sampling Strategies
//!
//! - **Uncertainty Sampling**: Low-confidence predictions
//! - **Diversity Sampling**: Examples different from existing data
//! - **Query-by-Committee**: High model disagreement
//! - **Hybrid**: Combine multiple signals
//!
//! # Example
//!
//! ```rust
//! use anno::eval::active_learning::{ActiveLearner, SamplingStrategy, Candidate};
//!
//! let learner = ActiveLearner::new(SamplingStrategy::Uncertainty);
//!
//! let candidates = vec![
//!     Candidate::new("John works at Google.", 0.95),
//!     Candidate::new("Xiangjun joined Alibaba.", 0.45),  // Low confidence
//! ];
//!
//! let to_annotate = learner.select(&candidates, 1);
//! assert_eq!(to_annotate[0].text, "Xiangjun joined Alibaba.");
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

// =============================================================================
// Data Structures
// =============================================================================

/// A candidate example for annotation.
#[derive(Debug, Clone)]
pub struct Candidate {
    /// Text to potentially annotate
    pub text: String,
    /// Model's confidence on this example (lower = more uncertain)
    pub confidence: f64,
    /// Optional: entity types predicted
    pub predicted_types: Vec<String>,
    /// Optional: multiple model predictions for committee sampling
    pub committee_predictions: Vec<Vec<String>>,
    /// Optional: embedding for diversity sampling
    pub embedding: Option<Vec<f64>>,
}

impl Candidate {
    /// Create a simple candidate with text and confidence.
    pub fn new(text: impl Into<String>, confidence: f64) -> Self {
        Self {
            text: text.into(),
            confidence,
            predicted_types: Vec::new(),
            committee_predictions: Vec::new(),
            embedding: None,
        }
    }

    /// Create candidate with predicted types.
    pub fn with_types(mut self, types: Vec<String>) -> Self {
        self.predicted_types = types;
        self
    }

    /// Create candidate with committee predictions.
    pub fn with_committee(mut self, predictions: Vec<Vec<String>>) -> Self {
        self.committee_predictions = predictions;
        self
    }

    /// Create candidate with embedding.
    pub fn with_embedding(mut self, embedding: Vec<f64>) -> Self {
        self.embedding = Some(embedding);
        self
    }
}

/// Sampling strategy for active learning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SamplingStrategy {
    /// Select examples with lowest model confidence
    Uncertainty,
    /// Select examples most different from existing data
    Diversity,
    /// Select examples where model committee disagrees most
    QueryByCommittee,
    /// Combine uncertainty and diversity
    Hybrid,
    /// Random baseline
    Random,
}

/// Result of active learning selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionResult {
    /// Selected candidates with their scores
    pub selected: Vec<(String, f64)>,
    /// Total candidates considered
    pub total_candidates: usize,
    /// Strategy used
    pub strategy: SamplingStrategy,
    /// Score statistics
    pub score_stats: ScoreStats,
}

/// Statistics about selection scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreStats {
    /// Mean score of selected candidates
    pub mean_selected: f64,
    /// Mean score of all candidates
    pub mean_all: f64,
    /// Score of best candidate
    pub max_score: f64,
    /// Score of worst candidate
    pub min_score: f64,
}

// =============================================================================
// Active Learner
// =============================================================================

/// Active learning selector.
#[derive(Debug, Clone)]
pub struct ActiveLearner {
    /// Sampling strategy
    strategy: SamplingStrategy,
    /// Seed for random sampling
    seed: u64,
    /// Weight for uncertainty in hybrid mode (0-1)
    uncertainty_weight: f64,
}

impl ActiveLearner {
    /// Create a new active learner with given strategy.
    pub fn new(strategy: SamplingStrategy) -> Self {
        Self {
            strategy,
            seed: 42,
            uncertainty_weight: 0.7,
        }
    }

    /// Set random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set uncertainty weight for hybrid mode.
    pub fn with_uncertainty_weight(mut self, weight: f64) -> Self {
        self.uncertainty_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Select top-k candidates for annotation.
    pub fn select<'a>(&self, candidates: &'a [Candidate], k: usize) -> Vec<&'a Candidate> {
        if candidates.is_empty() || k == 0 {
            return Vec::new();
        }

        let k = k.min(candidates.len());

        match self.strategy {
            SamplingStrategy::Uncertainty => self.select_by_uncertainty(candidates, k),
            SamplingStrategy::Diversity => self.select_by_diversity(candidates, k),
            SamplingStrategy::QueryByCommittee => self.select_by_committee(candidates, k),
            SamplingStrategy::Hybrid => self.select_hybrid(candidates, k),
            SamplingStrategy::Random => self.select_random(candidates, k),
        }
    }

    /// Select with detailed results.
    pub fn select_with_scores(&self, candidates: &[Candidate], k: usize) -> SelectionResult {
        let scores = self.compute_scores(candidates);

        let mut indexed: Vec<(usize, f64)> = scores.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let k = k.min(candidates.len());
        let selected: Vec<(String, f64)> = indexed
            .iter()
            .take(k)
            .map(|(i, s)| (candidates[*i].text.clone(), *s))
            .collect();

        let all_scores: Vec<f64> = indexed.iter().map(|(_, s)| *s).collect();
        let mean_all = all_scores.iter().sum::<f64>() / all_scores.len().max(1) as f64;
        let mean_selected = selected.iter().map(|(_, s)| s).sum::<f64>() / k.max(1) as f64;

        SelectionResult {
            selected,
            total_candidates: candidates.len(),
            strategy: self.strategy,
            score_stats: ScoreStats {
                mean_selected,
                mean_all,
                max_score: all_scores.first().copied().unwrap_or(0.0),
                min_score: all_scores.last().copied().unwrap_or(0.0),
            },
        }
    }

    fn compute_scores(&self, candidates: &[Candidate]) -> Vec<f64> {
        match self.strategy {
            SamplingStrategy::Uncertainty => {
                candidates.iter().map(|c| 1.0 - c.confidence).collect()
            }
            SamplingStrategy::QueryByCommittee => {
                candidates.iter().map(|c| self.committee_disagreement(c)).collect()
            }
            SamplingStrategy::Diversity => {
                // For pure diversity, use embedding distances
                // Without embeddings, fall back to uncertainty
                candidates.iter().map(|c| {
                    if c.embedding.is_some() {
                        0.5 // Placeholder - real diversity needs pairwise comparison
                    } else {
                        1.0 - c.confidence
                    }
                }).collect()
            }
            SamplingStrategy::Hybrid => {
                let uncertainty: Vec<f64> = candidates.iter().map(|c| 1.0 - c.confidence).collect();
                let committee: Vec<f64> = candidates.iter().map(|c| self.committee_disagreement(c)).collect();

                uncertainty.iter().zip(committee.iter())
                    .map(|(u, c)| self.uncertainty_weight * u + (1.0 - self.uncertainty_weight) * c)
                    .collect()
            }
            SamplingStrategy::Random => {
                // Pseudo-random scores based on text hash
                candidates.iter().enumerate()
                    .map(|(i, c)| {
                        let hash = c.text.bytes().fold(self.seed, |acc, b| {
                            acc.wrapping_mul(31).wrapping_add(b as u64)
                        });
                        (hash.wrapping_add(i as u64) % 1000) as f64 / 1000.0
                    })
                    .collect()
            }
        }
    }

    fn select_by_uncertainty<'a>(&self, candidates: &'a [Candidate], k: usize) -> Vec<&'a Candidate> {
        let mut indexed: Vec<(usize, f64)> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.confidence))
            .collect();

        // Sort by confidence ascending (lowest = most uncertain)
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        indexed.iter().take(k).map(|(i, _)| &candidates[*i]).collect()
    }

    fn select_by_diversity<'a>(&self, candidates: &'a [Candidate], k: usize) -> Vec<&'a Candidate> {
        // Greedy diversity selection using embeddings
        // If no embeddings, fall back to uncertainty

        let has_embeddings = candidates.iter().all(|c| c.embedding.is_some());
        if !has_embeddings {
            return self.select_by_uncertainty(candidates, k);
        }

        let mut selected_indices = Vec::new();
        let mut remaining: HashSet<usize> = (0..candidates.len()).collect();

        // Start with most uncertain
        let first_idx = candidates
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.confidence.partial_cmp(&b.1.confidence).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        selected_indices.push(first_idx);
        remaining.remove(&first_idx);

        // Greedily add most diverse
        while selected_indices.len() < k && !remaining.is_empty() {
            let mut best_idx = 0;
            let mut best_min_dist = f64::NEG_INFINITY;

            for &idx in &remaining {
                // Find minimum distance to any selected
                let min_dist = selected_indices
                    .iter()
                    .map(|&sel_idx| {
                        self.embedding_distance(
                            candidates[idx].embedding.as_ref().unwrap(),
                            candidates[sel_idx].embedding.as_ref().unwrap(),
                        )
                    })
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0);

                if min_dist > best_min_dist {
                    best_min_dist = min_dist;
                    best_idx = idx;
                }
            }

            selected_indices.push(best_idx);
            remaining.remove(&best_idx);
        }

        selected_indices.iter().map(|&i| &candidates[i]).collect()
    }

    fn select_by_committee<'a>(&self, candidates: &'a [Candidate], k: usize) -> Vec<&'a Candidate> {
        let mut indexed: Vec<(usize, f64)> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| (i, self.committee_disagreement(c)))
            .collect();

        // Sort by disagreement descending
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        indexed.iter().take(k).map(|(i, _)| &candidates[*i]).collect()
    }

    fn select_hybrid<'a>(&self, candidates: &'a [Candidate], k: usize) -> Vec<&'a Candidate> {
        let scores = self.compute_scores(candidates);
        let mut indexed: Vec<(usize, f64)> = scores.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.iter().take(k).map(|(i, _)| &candidates[*i]).collect()
    }

    fn select_random<'a>(&self, candidates: &'a [Candidate], k: usize) -> Vec<&'a Candidate> {
        let scores = self.compute_scores(candidates);
        let mut indexed: Vec<(usize, f64)> = scores.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.iter().take(k).map(|(i, _)| &candidates[*i]).collect()
    }

    fn committee_disagreement(&self, candidate: &Candidate) -> f64 {
        if candidate.committee_predictions.len() < 2 {
            return 1.0 - candidate.confidence;
        }

        // Count agreement on each entity type
        let all_types: HashSet<&String> = candidate
            .committee_predictions
            .iter()
            .flat_map(|p| p.iter())
            .collect();

        if all_types.is_empty() {
            return 0.0;
        }

        let num_models = candidate.committee_predictions.len();
        let mut total_disagreement = 0.0;

        for entity_type in all_types {
            let count = candidate
                .committee_predictions
                .iter()
                .filter(|p| p.contains(entity_type))
                .count();

            // Disagreement is highest when count is closest to num_models/2
            let agreement_ratio = count as f64 / num_models as f64;
            let disagreement = 4.0 * agreement_ratio * (1.0 - agreement_ratio);
            total_disagreement += disagreement;
        }

        total_disagreement / all_types.len() as f64
    }

    fn embedding_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        // Euclidean distance
        if a.len() != b.len() {
            return 0.0;
        }

        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

impl Default for ActiveLearner {
    fn default() -> Self {
        Self::new(SamplingStrategy::Uncertainty)
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Estimate annotation budget needed for target performance.
pub fn estimate_budget(
    current_f1: f64,
    target_f1: f64,
    current_samples: usize,
    f1_per_100_samples: f64,
) -> Option<usize> {
    if target_f1 <= current_f1 || f1_per_100_samples <= 0.0 {
        return Some(0);
    }

    let f1_needed = target_f1 - current_f1;
    let hundreds_needed = f1_needed / f1_per_100_samples;
    Some((hundreds_needed * 100.0).ceil() as usize)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uncertainty_sampling() {
        let candidates = vec![
            Candidate::new("High confidence", 0.95),
            Candidate::new("Low confidence", 0.30),
            Candidate::new("Medium confidence", 0.60),
        ];

        let learner = ActiveLearner::new(SamplingStrategy::Uncertainty);
        let selected = learner.select(&candidates, 2);

        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].text, "Low confidence");
        assert_eq!(selected[1].text, "Medium confidence");
    }

    #[test]
    fn test_committee_sampling() {
        let mut low_agreement = Candidate::new("Disagreement", 0.5);
        low_agreement.committee_predictions = vec![
            vec!["PER".into()],
            vec!["ORG".into()],
            vec!["LOC".into()],
        ];

        let mut high_agreement = Candidate::new("Agreement", 0.5);
        high_agreement.committee_predictions = vec![
            vec!["PER".into()],
            vec!["PER".into()],
            vec!["PER".into()],
        ];

        let candidates = vec![low_agreement, high_agreement];
        let learner = ActiveLearner::new(SamplingStrategy::QueryByCommittee);
        let selected = learner.select(&candidates, 1);

        assert_eq!(selected[0].text, "Disagreement");
    }

    #[test]
    fn test_select_with_scores() {
        let candidates = vec![
            Candidate::new("A", 0.90),
            Candidate::new("B", 0.40),
            Candidate::new("C", 0.70),
        ];

        let learner = ActiveLearner::new(SamplingStrategy::Uncertainty);
        let result = learner.select_with_scores(&candidates, 2);

        assert_eq!(result.selected.len(), 2);
        assert_eq!(result.total_candidates, 3);
        assert!(result.score_stats.mean_selected > result.score_stats.mean_all);
    }

    #[test]
    fn test_estimate_budget() {
        let budget = estimate_budget(0.70, 0.85, 1000, 0.01);
        assert!(budget.is_some());
        assert!(budget.unwrap() > 0);
    }

    #[test]
    fn test_empty_candidates() {
        let learner = ActiveLearner::default();
        let selected = learner.select(&[], 5);
        assert!(selected.is_empty());
    }
}

