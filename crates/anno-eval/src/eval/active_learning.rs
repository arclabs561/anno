//! Active learning utilities for NER annotation.
//!
//! Helps identify which examples to annotate next for maximum model improvement.
//!
//! # Sampling Strategies
//!
//! - **Uncertainty Sampling**: Low-confidence predictions (requires confidence scores)
//! - **Diversity Sampling**: Examples most different from each other (requires embeddings)
//! - **Query-by-Committee**: High model disagreement (requires multiple model predictions)
//! - **Hybrid**: Combine uncertainty and committee signals
//!
//! # Strategy Requirements and Fallbacks
//!
//! Each strategy has specific data requirements. When requirements aren't met,
//! the strategy falls back as follows:
//!
//! | Strategy | Requires | Falls back to |
//! |----------|----------|---------------|
//! | Uncertainty | `confidence` | Always works (uses 0.5 if missing) |
//! | Diversity | `embedding` | Uncertainty (with warning) |
//! | QueryByCommittee | `committee_predictions` (≥2) | Uncertainty (with warning) |
//! | Hybrid | Both confidence and committee | Uncertainty if committee missing |
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
    /// Optional: embedding for diversity sampling (required for Diversity strategy)
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

    /// Check if this candidate has valid committee predictions (≥2 models).
    pub fn has_committee(&self) -> bool {
        self.committee_predictions.len() >= 2
    }

    /// Check if this candidate has an embedding.
    pub fn has_embedding(&self) -> bool {
        self.embedding.is_some()
    }
}

/// Sampling strategy for active learning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SamplingStrategy {
    /// Select examples with lowest model confidence.
    /// Always works - uses confidence field directly.
    Uncertainty,
    /// Select examples most different from existing data.
    /// **Requires embeddings** - falls back to Uncertainty if missing.
    Diversity,
    /// Select examples where model committee disagrees most.
    /// **Requires committee_predictions with ≥2 models** - falls back to Uncertainty.
    QueryByCommittee,
    /// Combine uncertainty and committee disagreement (0.7 uncertainty + 0.3 committee).
    /// Falls back to pure Uncertainty if no committee data.
    Hybrid,
    /// Random baseline (deterministic given seed).
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
    /// Actual strategy used (may differ if fallback occurred)
    pub actual_strategy: SamplingStrategy,
    /// Score statistics
    pub score_stats: ScoreStats,
    /// Warnings about strategy fallbacks or data issues
    pub warnings: Vec<String>,
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
        let (actual_strategy, _warnings) = self.resolve_strategy(candidates);

        match actual_strategy {
            SamplingStrategy::Uncertainty => self.select_by_uncertainty(candidates, k),
            SamplingStrategy::Diversity => self.select_by_diversity(candidates, k),
            SamplingStrategy::QueryByCommittee => self.select_by_committee(candidates, k),
            SamplingStrategy::Hybrid => self.select_hybrid(candidates, k),
            SamplingStrategy::Random => self.select_random(candidates, k),
        }
    }

    /// Select with detailed results including warnings about fallbacks.
    pub fn select_with_scores(&self, candidates: &[Candidate], k: usize) -> SelectionResult {
        let (actual_strategy, warnings) = self.resolve_strategy(candidates);
        let scores = self.compute_scores_with_strategy(candidates, actual_strategy);

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
            actual_strategy,
            score_stats: ScoreStats {
                mean_selected,
                mean_all,
                max_score: all_scores.first().copied().unwrap_or(0.0),
                min_score: all_scores.last().copied().unwrap_or(0.0),
            },
            warnings,
        }
    }

    /// Resolve the actual strategy to use, considering data availability.
    fn resolve_strategy(&self, candidates: &[Candidate]) -> (SamplingStrategy, Vec<String>) {
        let mut warnings = Vec::new();

        match self.strategy {
            SamplingStrategy::Diversity => {
                let has_all_embeddings = candidates.iter().all(|c| c.has_embedding());
                if !has_all_embeddings {
                    let missing = candidates.iter().filter(|c| !c.has_embedding()).count();
                    warnings.push(format!(
                        "Diversity sampling requires embeddings: {}/{} candidates missing embeddings. Falling back to Uncertainty.",
                        missing, candidates.len()
                    ));
                    return (SamplingStrategy::Uncertainty, warnings);
                }
            }
            SamplingStrategy::QueryByCommittee => {
                let has_all_committees = candidates.iter().all(|c| c.has_committee());
                if !has_all_committees {
                    let missing = candidates.iter().filter(|c| !c.has_committee()).count();
                    warnings.push(format!(
                        "Query-by-Committee requires committee predictions (≥2 models): {}/{} candidates missing. Falling back to Uncertainty.",
                        missing, candidates.len()
                    ));
                    return (SamplingStrategy::Uncertainty, warnings);
                }
            }
            SamplingStrategy::Hybrid => {
                let has_any_committees = candidates.iter().any(|c| c.has_committee());
                if !has_any_committees {
                    warnings.push(
                        "Hybrid mode has no committee data. Using pure Uncertainty.".to_string(),
                    );
                    // Still use Hybrid, but it will effectively be Uncertainty
                }
            }
            _ => {}
        }

        (self.strategy, warnings)
    }

    fn compute_scores_with_strategy(
        &self,
        candidates: &[Candidate],
        strategy: SamplingStrategy,
    ) -> Vec<f64> {
        match strategy {
            SamplingStrategy::Uncertainty => {
                candidates.iter().map(|c| 1.0 - c.confidence).collect()
            }
            SamplingStrategy::QueryByCommittee => candidates
                .iter()
                .map(|c| self.committee_disagreement(c))
                .collect(),
            SamplingStrategy::Diversity => {
                // For compute_scores, we return uncertainty-weighted diversity
                // The actual diversity selection uses greedy farthest-point
                self.compute_diversity_scores(candidates)
            }
            SamplingStrategy::Hybrid => {
                let uncertainty: Vec<f64> = candidates.iter().map(|c| 1.0 - c.confidence).collect();
                let committee: Vec<f64> = candidates
                    .iter()
                    .map(|c| self.committee_disagreement(c))
                    .collect();

                uncertainty
                    .iter()
                    .zip(committee.iter())
                    .map(|(u, c)| self.uncertainty_weight * u + (1.0 - self.uncertainty_weight) * c)
                    .collect()
            }
            SamplingStrategy::Random => {
                // Pseudo-random scores based on text hash
                candidates
                    .iter()
                    .enumerate()
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

    /// Compute diversity scores based on embedding distances.
    ///
    /// Uses average distance to all other candidates as the diversity score.
    /// Higher scores indicate more diverse (distant) candidates.
    fn compute_diversity_scores(&self, candidates: &[Candidate]) -> Vec<f64> {
        let n = candidates.len();
        if n == 0 {
            return Vec::new();
        }

        // Compute pairwise distances and use mean distance as diversity score
        let mut scores = vec![0.0; n];

        for i in 0..n {
            let emb_i = match &candidates[i].embedding {
                Some(e) => e,
                None => {
                    // No embedding - use uncertainty as fallback
                    scores[i] = 1.0 - candidates[i].confidence;
                    continue;
                }
            };

            let mut total_dist = 0.0;
            let mut count = 0;

            for (j, candidate) in candidates.iter().enumerate() {
                if i == j {
                    continue;
                }
                if let Some(emb_j) = &candidate.embedding {
                    total_dist += self.embedding_distance(emb_i, emb_j);
                    count += 1;
                }
            }

            scores[i] = if count > 0 {
                total_dist / count as f64
            } else {
                0.0
            };
        }

        // Normalize scores to [0, 1]
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_score = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let range = max_score - min_score;

        if range > 0.0 {
            scores
                .iter_mut()
                .for_each(|s| *s = (*s - min_score) / range);
        }

        scores
    }

    fn select_by_uncertainty<'a>(
        &self,
        candidates: &'a [Candidate],
        k: usize,
    ) -> Vec<&'a Candidate> {
        let mut indexed: Vec<(usize, f64)> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.confidence))
            .collect();

        // Sort by confidence ascending (lowest = most uncertain)
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        indexed
            .iter()
            .take(k)
            .map(|(i, _)| &candidates[*i])
            .collect()
    }

    fn select_by_diversity<'a>(&self, candidates: &'a [Candidate], k: usize) -> Vec<&'a Candidate> {
        // Greedy farthest-point sampling using embeddings.
        // This maximizes the minimum distance between selected points.

        // Verify embeddings exist (should have been checked by resolve_strategy)
        let has_embeddings = candidates.iter().all(|c| c.embedding.is_some());
        if !has_embeddings {
            return self.select_by_uncertainty(candidates, k);
        }

        let mut selected_indices = Vec::with_capacity(k);
        let mut remaining: HashSet<usize> = (0..candidates.len()).collect();

        // Start with the most uncertain candidate (combines uncertainty with diversity)
        let first_idx = candidates
            .iter()
            .enumerate()
            .min_by(|a, b| {
                a.1.confidence
                    .partial_cmp(&b.1.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        selected_indices.push(first_idx);
        remaining.remove(&first_idx);

        // Greedily add candidate that maximizes minimum distance to selected set
        while selected_indices.len() < k && !remaining.is_empty() {
            let mut best_idx = 0;
            let mut best_min_dist = f64::NEG_INFINITY;

            for &idx in &remaining {
                // Skip candidates without embeddings
                let Some(emb_idx) = candidates[idx].embedding.as_ref() else {
                    continue;
                };

                // Find minimum distance to any already-selected candidate
                let min_dist = selected_indices
                    .iter()
                    .filter_map(|&sel_idx| {
                        let emb_sel = candidates[sel_idx].embedding.as_ref()?;
                        Some(self.embedding_distance(emb_idx, emb_sel))
                    })
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
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

        indexed
            .iter()
            .take(k)
            .map(|(i, _)| &candidates[*i])
            .collect()
    }

    fn select_hybrid<'a>(&self, candidates: &'a [Candidate], k: usize) -> Vec<&'a Candidate> {
        let scores = self.compute_scores_with_strategy(candidates, SamplingStrategy::Hybrid);
        let mut indexed: Vec<(usize, f64)> = scores.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed
            .iter()
            .take(k)
            .map(|(i, _)| &candidates[*i])
            .collect()
    }

    fn select_random<'a>(&self, candidates: &'a [Candidate], k: usize) -> Vec<&'a Candidate> {
        let scores = self.compute_scores_with_strategy(candidates, SamplingStrategy::Random);
        let mut indexed: Vec<(usize, f64)> = scores.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed
            .iter()
            .take(k)
            .map(|(i, _)| &candidates[*i])
            .collect()
    }

    fn committee_disagreement(&self, candidate: &Candidate) -> f64 {
        if candidate.committee_predictions.len() < 2 {
            // No committee - fall back to uncertainty
            return 1.0 - candidate.confidence;
        }

        // Count agreement on each entity type using vote entropy
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

        let num_types = all_types.len();
        for entity_type in &all_types {
            let count = candidate
                .committee_predictions
                .iter()
                .filter(|p| p.contains(*entity_type))
                .count();

            // Disagreement is highest when count is closest to num_models/2
            // Using variance of binary votes: p(1-p) where p = count/num_models
            let agreement_ratio = count as f64 / num_models as f64;
            let disagreement = 4.0 * agreement_ratio * (1.0 - agreement_ratio);
            total_disagreement += disagreement;
        }

        total_disagreement / num_types as f64
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
///
/// This is a simple linear extrapolation based on observed learning rate.
/// For more accurate estimates, use `LearningCurveAnalyzer`.
pub fn estimate_budget(
    current_f1: f64,
    target_f1: f64,
    _current_samples: usize,
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
        low_agreement.committee_predictions =
            vec![vec!["PER".into()], vec!["ORG".into()], vec!["LOC".into()]];

        let mut high_agreement = Candidate::new("Agreement", 0.5);
        high_agreement.committee_predictions =
            vec![vec!["PER".into()], vec!["PER".into()], vec!["PER".into()]];

        let candidates = vec![low_agreement, high_agreement];
        let learner = ActiveLearner::new(SamplingStrategy::QueryByCommittee);
        let selected = learner.select(&candidates, 1);

        assert_eq!(selected[0].text, "Disagreement");
    }

    #[test]
    fn test_diversity_sampling_with_embeddings() {
        // Create candidates with embeddings at different points
        let candidates = vec![
            Candidate::new("Near origin", 0.5).with_embedding(vec![0.0, 0.0]),
            Candidate::new("Far positive", 0.5).with_embedding(vec![10.0, 10.0]),
            Candidate::new("Far negative", 0.5).with_embedding(vec![-10.0, -10.0]),
            Candidate::new("Near origin 2", 0.5).with_embedding(vec![0.1, 0.1]),
        ];

        let learner = ActiveLearner::new(SamplingStrategy::Diversity);
        let selected = learner.select(&candidates, 3);

        // Should select diverse points, not clustered ones
        assert_eq!(selected.len(), 3);
        let texts: Vec<&str> = selected.iter().map(|c| c.text.as_str()).collect();
        assert!(texts.contains(&"Far positive"));
        assert!(texts.contains(&"Far negative"));
    }

    #[test]
    fn test_diversity_fallback_without_embeddings() {
        let candidates = vec![
            Candidate::new("No embedding 1", 0.9),
            Candidate::new("No embedding 2", 0.3), // Most uncertain
        ];

        let learner = ActiveLearner::new(SamplingStrategy::Diversity);
        let result = learner.select_with_scores(&candidates, 1);

        // Should fall back to uncertainty
        assert_eq!(result.actual_strategy, SamplingStrategy::Uncertainty);
        assert!(!result.warnings.is_empty());
        assert_eq!(result.selected[0].0, "No embedding 2");
    }

    #[test]
    fn test_committee_fallback_without_predictions() {
        let candidates = vec![
            Candidate::new("No committee 1", 0.9),
            Candidate::new("No committee 2", 0.3),
        ];

        let learner = ActiveLearner::new(SamplingStrategy::QueryByCommittee);
        let result = learner.select_with_scores(&candidates, 1);

        // Should fall back to uncertainty
        assert_eq!(result.actual_strategy, SamplingStrategy::Uncertainty);
        assert!(!result.warnings.is_empty());
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
        assert!(result.warnings.is_empty());
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
