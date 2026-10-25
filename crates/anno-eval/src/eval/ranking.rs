//! Ranking evaluation metrics for Named Entity Disambiguation (NED).
//!
//! Integrates with `rank-eval` crate for standardized IR metrics:
//! - NDCG@k (Normalized Discounted Cumulative Gain)
//! - MRR (Mean Reciprocal Rank)
//! - Precision@k, Recall@k
//! - MAP (Mean Average Precision)
//!
//! # Use Cases
//!
//! 1. **Entity Linking**: Evaluate ranked candidate KB entities for each mention
//! 2. **Cross-doc Coref**: Score ranked cluster candidates
//! 3. **Entity Retrieval**: Evaluate search results over entity indices
//!
//! # Example
//!
//! ```rust,ignore
//! use anno_eval::eval::ranking::{NedMetrics, CandidateRanking};
//!
//! // Gold entity for a mention
//! let gold_kb_id = "Q76";  // Barack Obama in Wikidata
//!
//! // Ranked candidates from NED system (best first)
//! let candidates = vec![
//!     ("Q76", 0.95),   // Correct: Barack Obama
//!     ("Q15180901", 0.8),  // Wrong: Barack Obama Sr.
//!     ("Q45780", 0.6),     // Wrong: Some other entity
//! ];
//!
//! let ranking = CandidateRanking::new(candidates, gold_kb_id);
//! let metrics = NedMetrics::compute(&ranking, 5);
//!
//! println!("MRR: {:.3}", metrics.mrr);          // 1.0 (correct at rank 1)
//! println!("P@1: {:.3}", metrics.precision_at_1); // 1.0
//! println!("Hits@5: {:.3}", metrics.hits_at_5);  // 1.0
//! ```

use std::collections::HashSet;

/// Metrics for Named Entity Disambiguation evaluation.
#[derive(Debug, Clone, Default)]
pub struct NedMetrics {
    /// Mean Reciprocal Rank: 1/rank of first correct candidate
    pub mrr: f64,
    /// Precision at k=1 (accuracy of top prediction)
    pub precision_at_1: f64,
    /// Precision at k=5
    pub precision_at_5: f64,
    /// Recall at k=5
    pub recall_at_5: f64,
    /// Hits@5: 1.0 if correct in top 5, else 0.0
    pub hits_at_5: f64,
    /// NDCG@5 (Normalized Discounted Cumulative Gain)
    pub ndcg_at_5: f64,
    /// NDCG@10
    pub ndcg_at_10: f64,
}

/// Represents a ranked list of KB candidates for entity disambiguation.
#[derive(Debug, Clone)]
pub struct CandidateRanking {
    /// Ranked candidates (kb_id, score) - highest score first
    pub candidates: Vec<(String, f64)>,
    /// Gold KB ID(s) for this mention
    pub gold_ids: HashSet<String>,
}

impl CandidateRanking {
    /// Create a new ranking with single gold entity.
    pub fn new<I, S>(candidates: I, gold_id: &str) -> Self
    where
        I: IntoIterator<Item = (S, f64)>,
        S: Into<String>,
    {
        let mut gold_ids = HashSet::new();
        gold_ids.insert(gold_id.to_string());
        Self {
            candidates: candidates
                .into_iter()
                .map(|(s, score)| (s.into(), score))
                .collect(),
            gold_ids,
        }
    }

    /// Create a new ranking with multiple gold entities (for aliases).
    pub fn with_multiple_gold<I, S, G>(candidates: I, gold_ids: G) -> Self
    where
        I: IntoIterator<Item = (S, f64)>,
        S: Into<String>,
        G: IntoIterator<Item = String>,
    {
        Self {
            candidates: candidates
                .into_iter()
                .map(|(s, score)| (s.into(), score))
                .collect(),
            gold_ids: gold_ids.into_iter().collect(),
        }
    }

    /// Check if a candidate is correct (matches any gold ID).
    pub fn is_correct(&self, candidate_id: &str) -> bool {
        self.gold_ids.contains(candidate_id)
    }
}

impl NedMetrics {
    /// Compute all NED metrics for a single ranking.
    pub fn compute(ranking: &CandidateRanking, _max_k: usize) -> Self {
        // Map the candidate ranking to the inputs expected by `rankops`:
        // 1-indexed ranks of the first occurrence of each gold candidate, and
        // the per-position relevance vector used for NDCG.
        let mut all_ranks: Vec<usize> = Vec::new();
        let mut relevance: Vec<f64> = Vec::with_capacity(ranking.candidates.len());
        let mut seen_gold: HashSet<&str> = HashSet::new();

        for (i, (id, _)) in ranking.candidates.iter().enumerate() {
            if ranking.is_correct(id) && seen_gold.insert(id.as_str()) {
                all_ranks.push(i + 1);
                relevance.push(1.0);
            } else {
                relevance.push(0.0);
            }
        }

        let n_relevant = ranking.gold_ids.len();
        let first_rank = all_ranks.iter().copied().min().unwrap_or(0);
        // NDCG gain policy: linear (`rel` rather than `2^rel - 1`). The NED
        // doc comment does not specify a gain curve and `rankops` defaults to
        // linear, so we keep the standard binary-relevance NDCG definition.
        let ideal: Vec<f64> = vec![1.0; n_relevant];

        Self {
            mrr: rankops::metrics::mrr(&[first_rank]),
            precision_at_1: rankops::metrics::precision_at_k(&all_ranks, 1),
            precision_at_5: rankops::metrics::precision_at_k(&all_ranks, 5),
            recall_at_5: rankops::metrics::recall_at_k(&all_ranks, n_relevant, 5),
            hits_at_5: rankops::metrics::hits_at_k(&[first_rank], 5),
            ndcg_at_5: rankops::metrics::ndcg_at_k(&relevance, &ideal, 5),
            ndcg_at_10: rankops::metrics::ndcg_at_k(&relevance, &ideal, 10),
        }
    }

    /// Aggregate metrics across multiple rankings.
    pub fn aggregate(metrics: &[Self]) -> Self {
        if metrics.is_empty() {
            return Self::default();
        }
        let n = metrics.len() as f64;
        Self {
            mrr: metrics.iter().map(|m| m.mrr).sum::<f64>() / n,
            precision_at_1: metrics.iter().map(|m| m.precision_at_1).sum::<f64>() / n,
            precision_at_5: metrics.iter().map(|m| m.precision_at_5).sum::<f64>() / n,
            recall_at_5: metrics.iter().map(|m| m.recall_at_5).sum::<f64>() / n,
            hits_at_5: metrics.iter().map(|m| m.hits_at_5).sum::<f64>() / n,
            ndcg_at_5: metrics.iter().map(|m| m.ndcg_at_5).sum::<f64>() / n,
            ndcg_at_10: metrics.iter().map(|m| m.ndcg_at_10).sum::<f64>() / n,
        }
    }
}

/// Evaluate a full NED system on a dataset.
///
/// Returns aggregated metrics across all mentions.
pub fn evaluate_ned(rankings: &[CandidateRanking]) -> NedMetrics {
    let metrics: Vec<NedMetrics> = rankings
        .iter()
        .map(|r| NedMetrics::compute(r, 10))
        .collect();
    NedMetrics::aggregate(&metrics)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candidate_ranking_correct_at_1() {
        let ranking = CandidateRanking::new(
            vec![("Q76", 0.95), ("Q15180901", 0.8), ("Q45780", 0.6)],
            "Q76",
        );
        let metrics = NedMetrics::compute(&ranking, 10);

        assert!((metrics.mrr - 1.0).abs() < 1e-12);
        assert!((metrics.precision_at_1 - 1.0).abs() < 1e-12);
        assert!((metrics.precision_at_5 - 0.2).abs() < 1e-12);
        assert!((metrics.recall_at_5 - 1.0).abs() < 1e-12);
        assert!((metrics.hits_at_5 - 1.0).abs() < 1e-12);
        assert!((metrics.ndcg_at_5 - 1.0).abs() < 1e-12);
        assert!((metrics.ndcg_at_10 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_candidate_ranking_correct_at_3() {
        let ranking = CandidateRanking::new(
            vec![
                ("Q15180901", 0.9),
                ("Q45780", 0.8),
                ("Q76", 0.7), // Correct at position 3
            ],
            "Q76",
        );
        let metrics = NedMetrics::compute(&ranking, 10);

        // MRR = 1/3 = 0.333...
        assert!((metrics.mrr - 1.0 / 3.0).abs() < 1e-12);
        assert!((metrics.precision_at_1 - 0.0).abs() < 1e-12);
        assert!((metrics.precision_at_5 - 0.2).abs() < 1e-12);
        assert!((metrics.recall_at_5 - 1.0).abs() < 1e-12);
        assert!((metrics.hits_at_5 - 1.0).abs() < 1e-12);
        // DCG@5 = 1/log2(4) = 0.5; IDCG@5 = 1/log2(2) = 1.0
        assert!((metrics.ndcg_at_5 - 0.5).abs() < 1e-12);
        assert!((metrics.ndcg_at_10 - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_candidate_ranking_not_in_top_k() {
        let ranking = CandidateRanking::new(
            vec![
                ("Q1", 0.9),
                ("Q2", 0.8),
                ("Q3", 0.7),
                ("Q4", 0.6),
                ("Q5", 0.5),
                ("Q76", 0.4), // Correct at position 6
            ],
            "Q76",
        );
        let metrics = NedMetrics::compute(&ranking, 10);

        // Not in top 5
        assert!((metrics.hits_at_5 - 0.0).abs() < 1e-12);
        // MRR = 1/6
        assert!((metrics.mrr - 1.0 / 6.0).abs() < 1e-12);
        assert!((metrics.precision_at_5 - 0.0).abs() < 1e-12);
        assert!((metrics.recall_at_5 - 0.0).abs() < 1e-12);
        assert!((metrics.ndcg_at_5 - 0.0).abs() < 1e-12);
        // Gold is at rank 6: DCG@10 = 1/log2(7), IDCG@10 = 1.0
        assert!((metrics.ndcg_at_10 - 1.0 / 7.0_f64.log2()).abs() < 1e-12);
    }

    #[test]
    fn test_aggregate_metrics() {
        let metrics = vec![
            NedMetrics {
                mrr: 1.0,
                precision_at_1: 1.0,
                ..Default::default()
            },
            NedMetrics {
                mrr: 0.5,
                precision_at_1: 0.0,
                ..Default::default()
            },
        ];
        let agg = NedMetrics::aggregate(&metrics);

        assert!((agg.mrr - 0.75).abs() < 0.01);
        assert!((agg.precision_at_1 - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_multiple_gold_ids() {
        // Entity with multiple valid KB IDs (aliases)
        let ranking = CandidateRanking::with_multiple_gold(
            vec![("Q2", 0.9), ("Q1", 0.8)], // Q1 is correct
            vec!["Q1".to_string(), "Q3".to_string()],
        );
        let metrics = NedMetrics::compute(&ranking, 10);

        // Q1 is at position 2
        assert!((metrics.mrr - 0.5).abs() < 1e-12);
        assert!((metrics.precision_at_1 - 0.0).abs() < 1e-12);
        assert!((metrics.precision_at_5 - 0.2).abs() < 1e-12);
        assert!((metrics.recall_at_5 - 0.5).abs() < 1e-12);
        assert!((metrics.hits_at_5 - 1.0).abs() < 1e-12);
        // DCG@5 = 1/log2(3), IDCG@5 = 1/log2(2) + 1/log2(3)
        let expected_ndcg = (1.0 / 3.0_f64.log2()) / (1.0 + 1.0 / 3.0_f64.log2());
        assert!((metrics.ndcg_at_5 - expected_ndcg).abs() < 1e-12);
    }

    #[test]
    fn test_candidate_ranking_rank_3_of_10_with_2_relevant() {
        // 10 candidates; two gold entities at ranks 3 and 7.
        let candidates = vec![
            ("Q1", 0.99),
            ("Q2", 0.98),
            ("Q3", 0.97), // gold #1 at rank 3
            ("Q4", 0.96),
            ("Q5", 0.95),
            ("Q6", 0.94),
            ("Q7", 0.93), // gold #2 at rank 7
            ("Q8", 0.92),
            ("Q9", 0.91),
            ("Q10", 0.90),
        ];
        let ranking = CandidateRanking::with_multiple_gold(
            candidates,
            vec!["Q3".to_string(), "Q7".to_string()],
        );
        let metrics = NedMetrics::compute(&ranking, 10);

        assert!((metrics.mrr - 1.0 / 3.0).abs() < 1e-12);
        assert!((metrics.precision_at_1 - 0.0).abs() < 1e-12);
        assert!((metrics.precision_at_5 - 0.2).abs() < 1e-12);
        assert!((metrics.recall_at_5 - 0.5).abs() < 1e-12);
        assert!((metrics.hits_at_5 - 1.0).abs() < 1e-12);
        // DCG@5 = 1/log2(4) = 0.5; IDCG@5 = 1/log2(2) + 1/log2(3)
        let expected_ndcg_5 = 0.5 / (1.0 + 1.0 / 3.0_f64.log2());
        assert!((metrics.ndcg_at_5 - expected_ndcg_5).abs() < 1e-12);
        // DCG@10 = 1/log2(4) + 1/log2(8) = 0.5 + 1/3
        let expected_ndcg_10 = (0.5 + 1.0 / 3.0) / (1.0 + 1.0 / 3.0_f64.log2());
        assert!((metrics.ndcg_at_10 - expected_ndcg_10).abs() < 1e-12);
    }

    #[test]
    fn test_gold_not_found() {
        let ranking = CandidateRanking::new(vec![("Q1", 0.9), ("Q2", 0.8), ("Q3", 0.7)], "Q76");
        let metrics = NedMetrics::compute(&ranking, 10);

        assert_eq!(metrics.mrr, 0.0);
        assert_eq!(metrics.precision_at_1, 0.0);
        assert_eq!(metrics.precision_at_5, 0.0);
        assert_eq!(metrics.recall_at_5, 0.0);
        assert_eq!(metrics.hits_at_5, 0.0);
        assert_eq!(metrics.ndcg_at_5, 0.0);
        assert_eq!(metrics.ndcg_at_10, 0.0);
    }

    #[test]
    fn test_empty_prediction() {
        let ranking = CandidateRanking::new(Vec::<(String, f64)>::new(), "Q76");
        let metrics = NedMetrics::compute(&ranking, 10);

        assert_eq!(metrics.mrr, 0.0);
        assert_eq!(metrics.precision_at_1, 0.0);
        assert_eq!(metrics.precision_at_5, 0.0);
        assert_eq!(metrics.recall_at_5, 0.0);
        assert_eq!(metrics.hits_at_5, 0.0);
        assert_eq!(metrics.ndcg_at_5, 0.0);
        assert_eq!(metrics.ndcg_at_10, 0.0);
    }
}
