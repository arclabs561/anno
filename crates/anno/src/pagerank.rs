//! Shared PageRank algorithm for graph-based ranking.
//!
//! This module provides a simple PageRank implementation that operates on
//! adjacency matrices. It's used by:
//! - `salience.rs` (TextRankSalience)
//! - `keywords.rs` (TextRankExtractor)
//! - `summarize.rs` (LexRankSummarizer)
//!
//! For PageRank on a typed entity/relation graph, use a dedicated graph library.
//!
//! # Algorithm
//!
//! Standard PageRank with damping factor d:
//!
//! ```text
//! PR(i) = (1-d)/N + d × Σ w_ji × PR(j) / out_degree(j)
//!                    j→i
//! ```

/// PageRank configuration.
#[derive(Debug, Clone)]
pub struct PageRankConfig {
    /// Damping factor (probability of following a link vs teleporting).
    /// Default: 0.85
    pub damping: f64,
    /// Maximum iterations before stopping.
    /// Default: 30
    pub max_iterations: usize,
    /// Convergence threshold (sum of absolute differences).
    /// Default: 1e-6
    pub epsilon: f64,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping: 0.85,
            max_iterations: 30,
            epsilon: 1e-6,
        }
    }
}

impl PageRankConfig {
    /// Create with custom damping factor.
    pub fn with_damping(mut self, damping: f64) -> Self {
        self.damping = damping.clamp(0.0, 1.0);
        self
    }

    /// Create with custom max iterations.
    pub fn with_max_iterations(mut self, iterations: usize) -> Self {
        self.max_iterations = iterations;
        self
    }
}

/// Compute PageRank scores on an adjacency matrix.
///
/// # Arguments
/// * `adjacency` - NxN weighted adjacency matrix (row i, col j = weight from i to j)
/// * `config` - PageRank parameters
///
/// # Returns
/// Vector of scores for each node, summing to approximately 1.0.
///
/// # Example
/// ```rust
/// use anno::pagerank::{pagerank, PageRankConfig};
///
/// // Simple 3-node graph: 0→1→2, 0→2
/// let adj = vec![
///     vec![0.0, 1.0, 1.0],  // node 0 connects to 1, 2
///     vec![0.0, 0.0, 1.0],  // node 1 connects to 2
///     vec![0.0, 0.0, 0.0],  // node 2 (sink)
/// ];
///
/// let scores = pagerank(&adj, &PageRankConfig::default());
/// assert_eq!(scores.len(), 3);
/// ```
pub fn pagerank(adjacency: &[Vec<f64>], config: &PageRankConfig) -> Vec<f64> {
    let n = adjacency.len();
    if n == 0 {
        return vec![];
    }

    // Initialize scores uniformly
    let mut scores = vec![1.0 / n as f64; n];

    // Compute out-degree (sum of outgoing edge weights) for each node
    let out_degree: Vec<f64> = adjacency
        .iter()
        .map(|row| row.iter().sum::<f64>().max(1.0)) // Avoid div by zero
        .collect();

    let teleport = (1.0 - config.damping) / n as f64;

    // Power iteration
    for _ in 0..config.max_iterations {
        let mut new_scores = vec![teleport; n];

        for i in 0..n {
            for j in 0..n {
                if adjacency[j][i] > 0.0 {
                    // Contribution from j to i
                    new_scores[i] += config.damping * scores[j] * adjacency[j][i] / out_degree[j];
                }
            }
        }

        // Check convergence
        let diff: f64 = scores
            .iter()
            .zip(new_scores.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        scores = new_scores;

        if diff < config.epsilon {
            break;
        }
    }

    scores
}

/// Compute PageRank and return indices sorted by score (descending).
pub fn pagerank_ranked(adjacency: &[Vec<f64>], config: &PageRankConfig) -> Vec<(usize, f64)> {
    let scores = pagerank(adjacency, config);
    let mut indexed: Vec<(usize, f64)> = scores.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph() {
        let adj: Vec<Vec<f64>> = vec![];
        let scores = pagerank(&adj, &PageRankConfig::default());
        assert!(scores.is_empty());
    }

    #[test]
    fn test_single_node() {
        let adj = vec![vec![0.0]];
        let scores = pagerank(&adj, &PageRankConfig::default());
        assert_eq!(scores.len(), 1);
        // Single node with no edges gets teleport probability (1-d)/N = 0.15/1 = 0.15
        // Plus some contribution from the random walk that goes nowhere
        assert!(scores[0] > 0.0 && scores[0] <= 1.0);
    }

    #[test]
    fn test_two_connected_nodes() {
        // Symmetric graph: 0↔1
        let adj = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let scores = pagerank(&adj, &PageRankConfig::default());

        assert_eq!(scores.len(), 2);
        // Both should have equal scores
        assert!((scores[0] - scores[1]).abs() < 0.01);
        // Sum should be ~1.0
        assert!((scores[0] + scores[1] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_hub_spoke() {
        // Hub (0) connected to spokes (1, 2, 3)
        let adj = vec![
            vec![0.0, 1.0, 1.0, 1.0], // hub → all
            vec![1.0, 0.0, 0.0, 0.0], // spoke1 → hub
            vec![1.0, 0.0, 0.0, 0.0], // spoke2 → hub
            vec![1.0, 0.0, 0.0, 0.0], // spoke3 → hub
        ];
        let scores = pagerank(&adj, &PageRankConfig::default());

        assert_eq!(scores.len(), 4);
        // Hub should have highest score (receives from all spokes)
        assert!(scores[0] > scores[1]);
        assert!(scores[0] > scores[2]);
        assert!(scores[0] > scores[3]);
    }

    #[test]
    fn test_ranked() {
        let adj = vec![
            vec![0.0, 1.0, 1.0, 1.0],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0],
        ];
        let ranked = pagerank_ranked(&adj, &PageRankConfig::default());

        // First result should be the hub (index 0)
        assert_eq!(ranked[0].0, 0);
    }
}
