//! Cluster encoder for cross-context coreference resolution.
//!
//! This module provides shared primitives for representing within-context clusters and scoring
//! merges across contexts (document windows or separate documents). It is intentionally dependency
//! light so it can be used by both `anno` and `anno-eval` without cycles.

use std::collections::HashMap;

/// A mention within a cluster, represented by character offsets.
#[derive(Debug, Clone)]
pub struct ClusterMention {
    /// Start character offset in the original text.
    pub start: usize,
    /// End character offset (exclusive).
    pub end: usize,
    /// Surface text of the mention.
    pub text: String,
    /// Context ID (document or window index).
    pub context_id: usize,
}

/// A coreference cluster containing mentions from a single context.
#[derive(Debug, Clone)]
pub struct LocalCluster {
    /// Unique identifier within the context.
    pub id: usize,
    /// Mentions in this cluster.
    pub mentions: Vec<ClusterMention>,
    /// Context identifier (document ID or window index).
    pub context_id: usize,
    /// Canonical representative (e.g., first non-pronoun mention).
    pub canonical: Option<String>,
}

impl LocalCluster {
    /// Create a new local cluster.
    #[must_use]
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
    /// Hidden dimension from base encoder (e.g., 1024 for DeBERTa-large).
    pub hidden_dim: usize,
    /// Number of attention heads in a cluster Transformer.
    pub num_heads: usize,
    /// Pooling strategy for cluster embedding.
    pub pooling: PoolingStrategy,
    /// Dropout rate.
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

/// Pooling strategy for reducing mention embeddings to a cluster embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingStrategy {
    /// Average all mention embeddings.
    Mean,
    /// Use first mention's embedding (assumes canonical ordering).
    First,
    /// Attention-weighted pooling with a learned query.
    AttentionWeighted,
    /// Max pooling per dimension.
    Max,
}

/// Cluster embedding: a fixed-size representation of a coreference cluster.
#[derive(Debug, Clone)]
pub struct ClusterEmbedding {
    /// The embedding vector.
    pub embedding: Vec<f32>,
    /// Source cluster ID.
    pub cluster_id: usize,
    /// Source context ID.
    pub context_id: usize,
    /// Number of mentions in the source cluster.
    pub mention_count: usize,
}

/// Trait for encoding clusters into fixed-size embeddings.
pub trait ClusterEncoder: Send + Sync {
    /// Encode a single cluster into an embedding.
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

impl std::fmt::Debug for dyn ClusterEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("dyn ClusterEncoder")
            .field("embedding_dim", &self.embedding_dim())
            .finish()
    }
}

/// Simple heuristic cluster encoder using hashed character n-grams.
#[derive(Debug, Clone)]
pub struct HeuristicClusterEncoder {
    dim: usize,
    ngram_size: usize,
}

impl HeuristicClusterEncoder {
    /// Create a new heuristic encoder.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self { dim, ngram_size: 3 }
    }

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
        let mut combined = vec![0.0f32; self.dim];

        for mention in &cluster.mentions {
            let mention_emb = self.hash_string(&mention.text);
            for (i, v) in mention_emb.into_iter().enumerate() {
                combined[i] += v;
            }
        }

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
    /// Input embedding dimension.
    pub embedding_dim: usize,
    /// Hidden dimension in scorer MLP.
    pub hidden_dim: usize,
    /// Threshold for merge decision.
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
    /// Returns a value clamped to \([0, 1]\).
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

        merges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        merges
    }
}

/// Simple cosine similarity scorer (CPU fallback).
#[derive(Debug, Clone)]
pub struct CosineMergeScorer {
    #[allow(dead_code)]
    threshold: f32,
}

impl CosineMergeScorer {
    /// Create a new cosine similarity scorer.
    #[must_use]
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

/// Configuration for cross-context resolution.
#[derive(Debug, Clone)]
pub struct CrossContextConfig {
    /// Merge threshold.
    pub threshold: f32,
    /// Whether to compare clusters from the same context.
    pub compare_same_context: bool,
    /// Maximum clusters to consider (for efficiency).
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
    /// Unique ID for the merged cluster.
    pub id: usize,
    /// Source local clusters (context_id, cluster_id).
    pub source_clusters: Vec<(usize, usize)>,
    /// All mentions from merged clusters.
    pub mentions: Vec<ClusterMention>,
    /// Canonical representative.
    pub canonical: Option<String>,
}

/// Cross-context coreference resolution combining encoder and scorer.
pub struct CrossContextResolver<E: ClusterEncoder, S: MergeScorer> {
    encoder: E,
    scorer: S,
    config: CrossContextConfig,
}

impl<E: ClusterEncoder, S: MergeScorer> CrossContextResolver<E, S> {
    /// Create a new cross-context resolver.
    #[must_use]
    pub fn new(encoder: E, scorer: S, config: CrossContextConfig) -> Self {
        Self {
            encoder,
            scorer,
            config,
        }
    }

    /// Resolve coreference across multiple contexts.
    #[must_use]
    pub fn resolve(
        &self,
        local_clusters: &HashMap<usize, Vec<LocalCluster>>,
        hidden_states: Option<&HashMap<usize, Vec<Vec<f32>>>>,
    ) -> Vec<MergedCluster> {
        // 1) Encode all clusters.
        let mut all_embeddings: Vec<ClusterEmbedding> = Vec::new();
        for (context_id, clusters) in local_clusters {
            let hs = hidden_states.and_then(|h| h.get(context_id).map(|v| v.as_slice()));
            let embeddings = self.encoder.encode_clusters(clusters, hs);
            all_embeddings.extend(embeddings);
        }

        // Optional: cap number of clusters for very large inputs.
        if let Some(max) = self.config.max_clusters {
            if all_embeddings.len() > max {
                all_embeddings.truncate(max);
            }
        }

        // 2) Score pairwise merges.
        let mut merge_decisions: Vec<(usize, usize, f32)> = Vec::new();
        for (i, emb_a) in all_embeddings.iter().enumerate() {
            for (j, emb_b) in all_embeddings.iter().enumerate().skip(i + 1) {
                if !self.config.compare_same_context && emb_a.context_id == emb_b.context_id {
                    continue;
                }
                let score = self.scorer.score(emb_a, emb_b);
                if score >= self.config.threshold {
                    merge_decisions.push((i, j, score));
                }
            }
        }
        merge_decisions.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // 3) Union-find.
        let mut uf = UnionFind::new(all_embeddings.len());
        for (i, j, _score) in merge_decisions {
            uf.union(i, j);
        }

        // 4) Group by root.
        let mut merged_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..all_embeddings.len() {
            let root = uf.find(i);
            merged_map.entry(root).or_default().push(i);
        }

        // 5) Materialize output.
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

/// Union-find structure for greedy merge closure.
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
            self.parent[x] = self.parent[self.parent[x]];
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

