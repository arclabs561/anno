//! Cluster encoder for cross-context coreference resolution.
//!
//! This module implements learned cluster representations for merging
//! coreference clusters across document windows or separate documents.
//!
//! Based on: Martinelli, Gatti & Navigli (2025). "xCoRe: Cross-context
//! Coreference Resolution" (EMNLP 2025).
//!
//! # Architecture
//!
//! 1. Each mention in a cluster is represented by (start_hidden, end_hidden)
//! 2. A single-layer Transformer attends over all mentions in the cluster
//! 3. Pooling produces a fixed-size cluster embedding
//! 4. MergeScorer computes pairwise merge probabilities
//!
//! # Example
//!
//! ```ignore
//! // Within-context: form local clusters using standard coref
//! let clusters_doc1 = resolver.resolve_intra_doc(doc1);
//! let clusters_doc2 = resolver.resolve_intra_doc(doc2);
//!
//! // Cross-context: encode clusters and score merges
//! let embeddings1 = encoder.encode_clusters(&clusters_doc1, &hidden_states1);
//! let embeddings2 = encoder.encode_clusters(&clusters_doc2, &hidden_states2);
//!
//! // Find mergeable cluster pairs
//! for (i, emb_a) in embeddings1.iter().enumerate() {
//!     for (j, emb_b) in embeddings2.iter().enumerate() {
//!         let score = scorer.score(emb_a, emb_b);
//!         if score > 0.5 {
//!             merge_clusters(i, j);
//!         }
//!     }
//! }
//! ```

use std::collections::HashMap;

/// A mention within a cluster, represented by token positions.
#[derive(Debug, Clone)]
pub struct ClusterMention {
    /// Start token index (character offset in original text)
    pub start: usize,
    /// End token index (exclusive)
    pub end: usize,
    /// Surface text of the mention
    pub text: String,
    /// Context ID (document or window index)
    pub context_id: usize,
}

/// A coreference cluster containing mentions from a single context.
#[derive(Debug, Clone)]
pub struct LocalCluster {
    /// Unique identifier within the context
    pub id: usize,
    /// Mentions in this cluster
    pub mentions: Vec<ClusterMention>,
    /// Context identifier (document ID or window index)
    pub context_id: usize,
    /// Canonical representative (e.g., first non-pronoun mention)
    pub canonical: Option<String>,
}

impl LocalCluster {
    /// Create a new local cluster.
    pub fn new(id: usize, context_id: usize) -> Self {
        Self {
            id,
            mentions: Vec::new(),
            context_id,
            canonical: None,
        }
    }

    /// Add a mention to the cluster.
    pub fn add_mention(&mut self, mention: ClusterMention) {
        self.mentions.push(mention);
    }

    /// Compute canonical form from mentions (heuristic: longest non-pronoun).
    pub fn compute_canonical(&mut self) {
        let pronouns = [
            "he", "she", "it", "they", "him", "her", "them", "his", "hers", "its",
        ];

        let canonical = self
            .mentions
            .iter()
            .filter(|m| !pronouns.contains(&m.text.to_lowercase().as_str()))
            .max_by_key(|m| m.text.len())
            .map(|m| m.text.clone());

        self.canonical = canonical.or_else(|| self.mentions.first().map(|m| m.text.clone()));
    }
}

/// Configuration for cluster encoding.
#[derive(Debug, Clone)]
pub struct ClusterEncoderConfig {
    /// Hidden dimension from base encoder (e.g., 1024 for DeBERTa-large)
    pub hidden_dim: usize,
    /// Number of attention heads in cluster Transformer
    pub num_heads: usize,
    /// Pooling strategy for cluster embedding
    pub pooling: PoolingStrategy,
    /// Dropout rate
    pub dropout: f32,
}

impl Default for ClusterEncoderConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 1024,
            num_heads: 8,
            pooling: PoolingStrategy::Mean,
            dropout: 0.1,
        }
    }
}

/// Pooling strategy for reducing mention embeddings to cluster embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingStrategy {
    /// Average all mention embeddings
    Mean,
    /// Use first mention's embedding (assumes canonical ordering)
    First,
    /// Attention-weighted pooling with learned query
    AttentionWeighted,
    /// Max pooling per dimension
    Max,
}

/// Cluster embedding: a fixed-size representation of a coreference cluster.
#[derive(Debug, Clone)]
pub struct ClusterEmbedding {
    /// The embedding vector (dimension = 2 * hidden_dim for start+end concat)
    pub embedding: Vec<f32>,
    /// Source cluster ID
    pub cluster_id: usize,
    /// Source context ID
    pub context_id: usize,
    /// Number of mentions in source cluster
    pub mention_count: usize,
}

/// Trait for encoding clusters into fixed-size embeddings.
///
/// Implementations may use different strategies:
/// - CPU heuristics (string similarity, TF-IDF)
/// - Neural encoders (Transformer over mentions)
pub trait ClusterEncoder: Send + Sync {
    /// Encode a single cluster into an embedding.
    ///
    /// # Arguments
    /// * `cluster` - The local cluster to encode
    /// * `hidden_states` - Token hidden states from base encoder (optional for heuristic methods)
    ///
    /// # Returns
    /// A fixed-size cluster embedding
    fn encode_cluster(
        &self,
        cluster: &LocalCluster,
        hidden_states: Option<&[Vec<f32>]>,
    ) -> ClusterEmbedding;

    /// Encode multiple clusters (batch operation).
    fn encode_clusters(
        &self,
        clusters: &[LocalCluster],
        hidden_states: Option<&[Vec<f32>]>,
    ) -> Vec<ClusterEmbedding> {
        clusters
            .iter()
            .map(|c| self.encode_cluster(c, hidden_states))
            .collect()
    }

    /// Expected embedding dimension.
    fn embedding_dim(&self) -> usize;
}

// Allow configs/structs that store trait objects to derive `Debug` without requiring
// concrete encoder types to implement it.
impl std::fmt::Debug for dyn ClusterEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("dyn ClusterEncoder")
            .field("embedding_dim", &self.embedding_dim())
            .finish()
    }
}

/// Simple heuristic cluster encoder using TF-IDF style features.
///
/// This is a CPU-only fallback when neural encoding is unavailable.
#[derive(Debug, Clone)]
pub struct HeuristicClusterEncoder {
    /// Embedding dimension
    dim: usize,
    /// Character n-gram size for hashing
    ngram_size: usize,
}

impl HeuristicClusterEncoder {
    /// Create a new heuristic encoder.
    pub fn new(dim: usize) -> Self {
        Self { dim, ngram_size: 3 }
    }

    /// Hash a string into a sparse embedding using character n-grams.
    fn hash_string(&self, s: &str) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut embedding = vec![0.0f32; self.dim];
        let chars: Vec<char> = s.to_lowercase().chars().collect();

        for window in chars.windows(self.ngram_size) {
            let ngram: String = window.iter().collect();
            let mut hasher = DefaultHasher::new();
            ngram.hash(&mut hasher);
            let idx = (hasher.finish() as usize) % self.dim;
            embedding[idx] += 1.0;
        }

        // L2 normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        embedding
    }
}

impl ClusterEncoder for HeuristicClusterEncoder {
    fn encode_cluster(
        &self,
        cluster: &LocalCluster,
        _hidden_states: Option<&[Vec<f32>]>,
    ) -> ClusterEmbedding {
        // Combine embeddings from all mention texts
        let mut combined = vec![0.0f32; self.dim];

        for mention in &cluster.mentions {
            let mention_emb = self.hash_string(&mention.text);
            for (i, v) in mention_emb.into_iter().enumerate() {
                combined[i] += v;
            }
        }

        // L2 normalize the combined embedding
        let norm: f32 = combined.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut combined {
                *x /= norm;
            }
        }

        ClusterEmbedding {
            embedding: combined,
            cluster_id: cluster.id,
            context_id: cluster.context_id,
            mention_count: cluster.mentions.len(),
        }
    }

    fn embedding_dim(&self) -> usize {
        self.dim
    }
}

/// Configuration for merge scoring.
#[derive(Debug, Clone)]
pub struct MergeScorerConfig {
    /// Input embedding dimension
    pub embedding_dim: usize,
    /// Hidden dimension in scorer MLP
    pub hidden_dim: usize,
    /// Threshold for merge decision
    pub threshold: f32,
}

impl Default for MergeScorerConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 256,
            hidden_dim: 128,
            threshold: 0.5,
        }
    }
}

/// Trait for scoring cluster merge probability.
pub trait MergeScorer: Send + Sync {
    /// Score the probability that two clusters should be merged.
    ///
    /// # Returns
    /// Probability in [0, 1] that the clusters are coreferent.
    fn score(&self, cluster_a: &ClusterEmbedding, cluster_b: &ClusterEmbedding) -> f32;

    /// Batch scoring for efficiency.
    fn score_batch(
        &self,
        clusters_a: &[ClusterEmbedding],
        clusters_b: &[ClusterEmbedding],
    ) -> Vec<Vec<f32>> {
        clusters_a
            .iter()
            .map(|a| clusters_b.iter().map(|b| self.score(a, b)).collect())
            .collect()
    }

    /// Get merge decisions above threshold.
    fn get_merges(
        &self,
        clusters_a: &[ClusterEmbedding],
        clusters_b: &[ClusterEmbedding],
        threshold: f32,
    ) -> Vec<(usize, usize, f32)> {
        let scores = self.score_batch(clusters_a, clusters_b);
        let mut merges = Vec::new();

        for (i, row) in scores.iter().enumerate() {
            for (j, &score) in row.iter().enumerate() {
                if score >= threshold {
                    merges.push((i, j, score));
                }
            }
        }

        // Sort by score descending
        merges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        merges
    }
}

/// Simple cosine similarity scorer (CPU fallback).
#[derive(Debug, Clone)]
pub struct CosineMergeScorer {
    /// Merge threshold (not currently used in score(), but useful for configuration)
    #[allow(dead_code)]
    threshold: f32,
}

impl CosineMergeScorer {
    /// Create a new cosine similarity scorer.
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }
}

impl MergeScorer for CosineMergeScorer {
    fn score(&self, cluster_a: &ClusterEmbedding, cluster_b: &ClusterEmbedding) -> f32 {
        let a = &cluster_a.embedding;
        let b = &cluster_b.embedding;

        if a.len() != b.len() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            (dot / (norm_a * norm_b)).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

/// Cross-context coreference resolution combining encoder and scorer.
pub struct CrossContextResolver<E: ClusterEncoder, S: MergeScorer> {
    encoder: E,
    scorer: S,
    config: CrossContextConfig,
}

/// Configuration for cross-context resolution.
#[derive(Debug, Clone)]
pub struct CrossContextConfig {
    /// Merge threshold
    pub threshold: f32,
    /// Whether to compare clusters from the same context
    pub compare_same_context: bool,
    /// Maximum clusters to consider (for efficiency)
    pub max_clusters: Option<usize>,
}

impl Default for CrossContextConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            compare_same_context: false,
            max_clusters: None,
        }
    }
}

/// A merged cross-context cluster.
#[derive(Debug, Clone)]
pub struct MergedCluster {
    /// Unique ID for the merged cluster
    pub id: usize,
    /// Source local clusters (context_id, cluster_id)
    pub source_clusters: Vec<(usize, usize)>,
    /// All mentions from merged clusters
    pub mentions: Vec<ClusterMention>,
    /// Canonical representative
    pub canonical: Option<String>,
}

impl<E: ClusterEncoder, S: MergeScorer> CrossContextResolver<E, S> {
    /// Create a new cross-context resolver.
    pub fn new(encoder: E, scorer: S, config: CrossContextConfig) -> Self {
        Self {
            encoder,
            scorer,
            config,
        }
    }

    /// Resolve coreference across multiple contexts.
    ///
    /// # Arguments
    /// * `local_clusters` - Clusters from each context, indexed by context_id
    /// * `hidden_states` - Optional token hidden states per context
    ///
    /// # Returns
    /// Merged cross-context clusters
    pub fn resolve(
        &self,
        local_clusters: &HashMap<usize, Vec<LocalCluster>>,
        hidden_states: Option<&HashMap<usize, Vec<Vec<f32>>>>,
    ) -> Vec<MergedCluster> {
        // 1. Encode all clusters
        let mut all_embeddings: Vec<ClusterEmbedding> = Vec::new();
        for (context_id, clusters) in local_clusters {
            let hs = hidden_states.and_then(|h| h.get(context_id).map(|v| v.as_slice()));
            let embeddings = self.encoder.encode_clusters(clusters, hs);
            all_embeddings.extend(embeddings);
        }

        // 2. Score pairwise merges
        let mut merge_decisions: Vec<(usize, usize, f32)> = Vec::new();
        for (i, emb_a) in all_embeddings.iter().enumerate() {
            for (j, emb_b) in all_embeddings.iter().enumerate().skip(i + 1) {
                // Skip same-context comparisons unless configured
                if !self.config.compare_same_context && emb_a.context_id == emb_b.context_id {
                    continue;
                }

                let score = self.scorer.score(emb_a, emb_b);
                if score >= self.config.threshold {
                    merge_decisions.push((i, j, score));
                }
            }
        }

        // 3. Sort by score and apply greedy merging via union-find
        merge_decisions.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        let mut uf = UnionFind::new(all_embeddings.len());
        for (i, j, _score) in merge_decisions {
            uf.union(i, j);
        }

        // 4. Build merged clusters
        let mut merged_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..all_embeddings.len() {
            let root = uf.find(i);
            merged_map.entry(root).or_default().push(i);
        }

        // 5. Convert to MergedCluster output
        let mut result: Vec<MergedCluster> = Vec::new();
        for (merged_id, (_, indices)) in merged_map.into_iter().enumerate() {
            let mut merged = MergedCluster {
                id: merged_id,
                source_clusters: Vec::new(),
                mentions: Vec::new(),
                canonical: None,
            };

            for idx in indices {
                let emb = &all_embeddings[idx];
                merged
                    .source_clusters
                    .push((emb.context_id, emb.cluster_id));

                // Find original cluster and copy mentions
                if let Some(clusters) = local_clusters.get(&emb.context_id) {
                    if let Some(cluster) = clusters.iter().find(|c| c.id == emb.cluster_id) {
                        merged.mentions.extend(cluster.mentions.clone());
                        if merged.canonical.is_none() {
                            merged.canonical = cluster.canonical.clone();
                        }
                    }
                }
            }

            result.push(merged);
        }

        result
    }
}

/// Simple union-find data structure for cluster merging.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]]; // Path compression
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, x: usize, y: usize) {
        let px = self.find(x);
        let py = self.find(y);
        if px == py {
            return;
        }
        // Union by rank
        match self.rank[px].cmp(&self.rank[py]) {
            std::cmp::Ordering::Less => self.parent[px] = py,
            std::cmp::Ordering::Greater => self.parent[py] = px,
            std::cmp::Ordering::Equal => {
                self.parent[py] = px;
                self.rank[px] += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heuristic_encoder() {
        let encoder = HeuristicClusterEncoder::new(64);

        let mut cluster = LocalCluster::new(0, 0);
        cluster.add_mention(ClusterMention {
            start: 0,
            end: 12,
            text: "Barack Obama".to_string(),
            context_id: 0,
        });
        cluster.add_mention(ClusterMention {
            start: 50,
            end: 52,
            text: "he".to_string(),
            context_id: 0,
        });

        let embedding = encoder.encode_cluster(&cluster, None);

        assert_eq!(embedding.embedding.len(), 64);
        assert_eq!(embedding.cluster_id, 0);
        assert_eq!(embedding.mention_count, 2);

        // Verify normalization
        let norm: f32 = embedding
            .embedding
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cosine_scorer() {
        let scorer = CosineMergeScorer::new(0.5);

        let emb_a = ClusterEmbedding {
            embedding: vec![1.0, 0.0, 0.0, 0.0],
            cluster_id: 0,
            context_id: 0,
            mention_count: 1,
        };
        let emb_b = ClusterEmbedding {
            embedding: vec![1.0, 0.0, 0.0, 0.0],
            cluster_id: 1,
            context_id: 1,
            mention_count: 1,
        };
        let emb_c = ClusterEmbedding {
            embedding: vec![0.0, 1.0, 0.0, 0.0],
            cluster_id: 2,
            context_id: 1,
            mention_count: 1,
        };

        // Identical embeddings -> score 1.0
        let score_ab = scorer.score(&emb_a, &emb_b);
        assert!((score_ab - 1.0).abs() < 0.01);

        // Orthogonal embeddings -> score 0.0
        let score_ac = scorer.score(&emb_a, &emb_c);
        assert!(score_ac.abs() < 0.01);
    }

    #[test]
    fn test_cross_context_resolver() {
        let encoder = HeuristicClusterEncoder::new(64);
        let scorer = CosineMergeScorer::new(0.3);
        let config = CrossContextConfig::default();
        let resolver = CrossContextResolver::new(encoder, scorer, config);

        // Context 0: "Barack Obama" cluster
        let mut cluster0 = LocalCluster::new(0, 0);
        cluster0.add_mention(ClusterMention {
            start: 0,
            end: 12,
            text: "Barack Obama".to_string(),
            context_id: 0,
        });

        // Context 1: "Obama" cluster (should merge with cluster0)
        let mut cluster1 = LocalCluster::new(0, 1);
        cluster1.add_mention(ClusterMention {
            start: 10,
            end: 15,
            text: "Obama".to_string(),
            context_id: 1,
        });

        // Context 1: "Angela Merkel" cluster (should NOT merge)
        let mut cluster2 = LocalCluster::new(1, 1);
        cluster2.add_mention(ClusterMention {
            start: 50,
            end: 63,
            text: "Angela Merkel".to_string(),
            context_id: 1,
        });

        let mut local_clusters = HashMap::new();
        local_clusters.insert(0, vec![cluster0]);
        local_clusters.insert(1, vec![cluster1, cluster2]);

        let merged = resolver.resolve(&local_clusters, None);

        // We should have 2 merged clusters:
        // 1. Barack Obama + Obama
        // 2. Angela Merkel
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn test_canonical_computation() {
        let mut cluster = LocalCluster::new(0, 0);
        cluster.add_mention(ClusterMention {
            start: 0,
            end: 2,
            text: "he".to_string(),
            context_id: 0,
        });
        cluster.add_mention(ClusterMention {
            start: 10,
            end: 22,
            text: "Barack Obama".to_string(),
            context_id: 0,
        });
        cluster.add_mention(ClusterMention {
            start: 30,
            end: 35,
            text: "Obama".to_string(),
            context_id: 0,
        });

        cluster.compute_canonical();

        // Should pick "Barack Obama" (longest non-pronoun)
        assert_eq!(cluster.canonical, Some("Barack Obama".to_string()));
    }
}
